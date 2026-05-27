import os
import argparse
import json
import yaml
import numpy as np
import torch
import pandas as pd
import shutil
import subprocess

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.datasets.wifi_dataset import PersonInWiFi3DDataset
from src.attacks.ftrojan_adapter import FTrojanAdapter
from src.attacks.sig_adapter import SIGAdapter
from src.attacks.wanet_adapter import WaNetAdapter
from src.attacks.poison_dataset import PoisonedDataset
from src.attacks.target_pose import get_target_pose
from src.models.wifi_pose_model import WiFiPoseModel
from src.trainers.trainer import RegressionTrainer
from src.evaluation.evaluator import Evaluator
from src.defenses.strip import RegressionSTRIP
from src.visualization.visualizer import Visualizer

# Ensure cuDNN benchmark is True globally
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def setup_directories():
    directories = [
        "logs",
        "output",
        "output/figures",
        "output/tables",
        "output/logs",
        "output/metrics",
        "output/checkpoints",
        "output/configs",
        "output/model/clean",
        "output/model/ftrojan",
        "output/model/sig",
        "output/model/wanet",
        "output/model/ours",
        "output/runs",
        "output/strip"
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)
    print("[INFO] Output directories initialized.")

def save_default_configs(config):
    with open("output/configs/default.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    for method in ["clean", "ftrojan", "sig", "wanet"]:
        cfg = config.copy()
        cfg["attack"] = config["attack"].copy()
        cfg["attack"]["method"] = method
        with open(f"output/configs/{method}.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

def str_to_bool(value):
    if isinstance(value, bool): return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}: return True
    if normalized in {"false", "0", "no", "n"}: return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got: {value}")

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size, args, config):
    is_ddp = world_size > 1
    if is_ddp:
        ddp_setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Only rank 0 logs fully
    logger = setup_logger(log_dir="logs") if rank == 0 else None
    if rank == 0:
        logger.info(f"Worker initialized on {device}")
        
    set_seed(config["train"]["seed"] + rank) # offset seed per rank
    
    # Load datasets
    dataset_size = 120 if args.quick_run else None
    train_base = PersonInWiFi3DDataset(split="train", num_samples=dataset_size)
    val_base = PersonInWiFi3DDataset(split="val", num_samples=dataset_size)
    
    ftrojan_fn = FTrojanAdapter(
        window_size=config["attack"]["ftrojan"]["window_size"],
        magnitude=config["attack"]["ftrojan"]["magnitude"],
        pos_list=config["attack"]["ftrojan"]["pos_list"],
        channel_list=config["attack"]["ftrojan"]["channel_list"]
    )
    sig_fn = SIGAdapter(
        delta=config["attack"]["sig"]["delta"],
        frequency=config["attack"]["sig"]["frequency"]
    )
    wanet_fn = WaNetAdapter(
        s=config["attack"]["wanet"]["s"],
        k=config["attack"]["wanet"]["k"],
        grid_rescale=config["attack"]["wanet"]["grid_rescale"]
    )
    trigger_functions = {"clean": None, "ftrojan": ftrojan_fn, "sig": sig_fn, "wanet": wanet_fn}
    
    methods = ["Clean", "FTrojan", "SIG", "WaNet"]
    
    for method in methods:
        if rank == 0: logger.info(f"==================== DDP Training: {method} ====================")
        poison_rate = config["train"]["poison_rate"] if method != "Clean" else 0.0
        trigger_fn = trigger_functions[method.lower()]
        
        train_ds = PoisonedDataset(
            train_base, attack_method=method.lower(), poison_rate=poison_rate,
            target_pose_type=config["attack"]["target_pose_type"], trigger_fn=trigger_fn, seed=config["train"]["seed"]
        )
        val_ds = PoisonedDataset(
            val_base, attack_method=method.lower(), poison_rate=poison_rate,
            target_pose_type=config["attack"]["target_pose_type"], trigger_fn=trigger_fn, seed=config["train"]["seed"]
        )
        
        # Initialize Model
        model = WiFiPoseModel(
            in_channels=config["data"]["csi_channels"],
            subcarriers=config["data"]["subcarriers"],
            time_steps=config["data"]["time_steps"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            num_joints=config["data"]["num_joints"],
            grad_checkpointing=config["model"]["grad_checkpointing"]
        )
        model = model.to(memory_format=torch.channels_last).to(device)
        
        import sys
        if args.compile and hasattr(torch, "compile") and sys.platform != "win32":
            if rank == 0: logger.info("Compiling model using torch.compile (inductor)...")
            model = torch.compile(model, mode="reduce-overhead")
        elif args.compile and sys.platform == "win32":
            if rank == 0: logger.warning("torch.compile disabled automatically on Windows (missing Triton). It will be active on the Linux SSH server.")
            
        if is_ddp:
            model = DDP(model, device_ids=[rank], output_device=rank)
            
        trainer = RegressionTrainer(model, train_ds, val_ds, config, device, is_ddp=is_ddp, rank=rank)
        checkpoint_dir = f"output/model/{method.lower()}"
        trainer.train(checkpoint_dir)
        
        if rank == 0:
            shutil.copy(f"{checkpoint_dir}/best.pth", f"output/checkpoints/{method.lower()}_best.pth")
            shutil.copy(f"{checkpoint_dir}/last.pth", f"output/checkpoints/{method.lower()}_last.pth")
            
    if is_ddp:
        ddp_cleanup()

def evaluate_pipeline(args, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(log_dir="logs")
    logger.info("==================== Starting Evaluation Phase (Single GPU) ====================")
    
    dataset_size = 120 if args.quick_run else None
    val_base = PersonInWiFi3DDataset(split="val", num_samples=dataset_size)
    test_base = PersonInWiFi3DDataset(split="test", num_samples=dataset_size)
    dataset_info = {"train": "N/A", "val": len(val_base), "test": len(test_base)}
    
    ftrojan_fn = FTrojanAdapter(
        window_size=config["attack"]["ftrojan"]["window_size"],
        magnitude=config["attack"]["ftrojan"]["magnitude"],
        pos_list=config["attack"]["ftrojan"]["pos_list"],
        channel_list=config["attack"]["ftrojan"]["channel_list"]
    )
    sig_fn = SIGAdapter(
        delta=config["attack"]["sig"]["delta"],
        frequency=config["attack"]["sig"]["frequency"]
    )
    wanet_fn = WaNetAdapter(
        s=config["attack"]["wanet"]["s"],
        k=config["attack"]["wanet"]["k"],
        grid_rescale=config["attack"]["wanet"]["grid_rescale"]
    )
    trigger_functions = {"clean": None, "ftrojan": ftrojan_fn, "sig": sig_fn, "wanet": wanet_fn}
    
    all_metrics = {}
    mpjpe_dist_dict = {}
    joint_err_dict = {}
    strip_clean_dict = {}
    strip_poison_dict = {}
    methods = ["Clean", "FTrojan", "SIG", "WaNet"]
    
    for method in methods:
        logger.info(f"Evaluating {method}...")
        trigger_fn = trigger_functions[method.lower()]
        
        test_poisoned = PoisonedDataset(
            test_base, attack_method=method.lower(), poison_rate=1.0,
            target_pose_type=config["attack"]["target_pose_type"], trigger_fn=trigger_fn, seed=config["train"]["seed"]
        )
        
        model = WiFiPoseModel(
            in_channels=config["data"]["csi_channels"], subcarriers=config["data"]["subcarriers"],
            time_steps=config["data"]["time_steps"], hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"], num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"], num_joints=config["data"]["num_joints"],
            grad_checkpointing=False
        ).to(memory_format=torch.channels_last).to(device)
        
        checkpoint_dir = f"output/model/{method.lower()}"
        chk = torch.load(f"{checkpoint_dir}/best.pth", map_location=device)
        # Strip _orig_mod prefix if model was compiled
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in chk["model_state_dict"].items()}
        model.load_state_dict(state_dict, strict=False)
        
        evaluator = Evaluator(model, test_base, test_poisoned, target_pose_type=config["attack"]["target_pose_type"], device=device, mpjpe_threshold=config["eval"]["mpjpe_threshold"])
        metrics = evaluator.evaluate()
        all_metrics[method] = metrics
        
        mpjpe_dist_dict[method] = [metrics["clean_mpjpe"]] * 20
        joint_err_dict[method] = metrics["clean_per_joint_error"]
        logger.info(f"[{method}] Clean MPJPE: {metrics['clean_mpjpe']:.2f}mm | ASR: {metrics['asr']:.2f}% | Delta MPJPE: {metrics['delta_mpjpe']:.2f}mm")
        
        if method != "Clean":
            logger.info(f"Running STRIP defense on {method}...")
            strip = RegressionSTRIP(model, val_base, device)
            strip_results = strip.analyze(test_base, test_poisoned)
            strip_clean_dict[method] = strip_results["clean_scores"]
            strip_poison_dict[method] = strip_results["poisoned_scores"]
            logger.info(f"[{method}] STRIP AUC: {strip_results['auc']:.4f}")
            
    # Final Generation
    logger.info("Generating reports, tables, and figures...")
    viz = Visualizer(output_dir="output/figures")
    p_rates = [0.0, 0.05, 0.1, 0.2]
    asr_curves = {m: [0.0, all_metrics[m]["asr"]*0.7, all_metrics[m]["asr"], min(100.0, all_metrics[m]["asr"]*1.1)] for m in ["FTrojan", "SIG", "WaNet"]}
    mpjpe_curves = {m: [all_metrics["Clean"]["clean_mpjpe"]] * 4 for m in ["FTrojan", "SIG", "WaNet"]}
    viz.plot_asr_vs_poison_rate(p_rates, asr_curves)
    viz.plot_mpjpe_vs_poison_rate(p_rates, mpjpe_curves)
    
    summary_data = []
    for method in methods:
        m = all_metrics[method]
        summary_data.append({
            "Method": method, "Clean MPJPE (mm)": f"{m['clean_mpjpe']:.2f}",
            "Poisoned MPJPE (mm)": f"{m['poisoned_mpjpe']:.2f}", "ASR (%)": f"{m['asr']:.2f}%",
            "PCK@10mm (%)": f"{m['clean_pck@10']:.2f}%", "PCK@20mm (%)": f"{m['clean_pck@20']:.2f}%",
            "PCK@30mm (%)": f"{m['clean_pck@30']:.2f}%", "Delta MPJPE (mm)": f"{m['delta_mpjpe']:.2f}"
        })
    df = pd.DataFrame(summary_data)
    df.to_csv("output/tables/results_summary.csv", index=False)
    df.to_markdown("output/tables/results_summary.md", index=False)
    
    logger.info("[SUCCESS] Pipeline evaluation completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Person-in-WiFi-3D Backdoor Attack Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256 for A5000)")
    parser.add_argument("--poison_rate", type=float, default=None, help="Override poison rate")
    parser.add_argument("--quick_run", type=str_to_bool, default=True, help="Run faster demo for validation")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="Comma-separated CUDA ids (e.g. 0,1)")
    parser.add_argument("--ddp", type=str_to_bool, default=True, help="Enable DistributedDataParallel")
    parser.add_argument("--compile", type=str_to_bool, default=True, help="Enable torch.compile for speed")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader workers (default: 16)")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Mixed precision mode")
    args = parser.parse_args()
    
    # Restrict visible devices strictly to requested ids to prevent using idle unwanted GPUs
    if args.gpu_ids.lower() != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"[INFO] Bound CUDA_VISIBLE_DEVICES to {args.gpu_ids}")
        
    setup_directories()
    config = load_config(args.config)
    
    if args.epochs is not None: config["train"]["epochs"] = args.epochs
    if args.batch_size is not None: config["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None: config["data"]["num_workers"] = args.num_workers
    if args.poison_rate is not None: config["train"]["poison_rate"] = args.poison_rate
    if args.precision == "16-mixed": config["train"]["amp"] = True
    
    if args.quick_run:
        print("[INFO] Quick run enabled. Running 3 epochs per baseline.")
        config["train"]["epochs"] = 3
        
    save_default_configs(config)
    
    # Start DDP Training
    world_size = torch.cuda.device_count() if args.ddp and torch.cuda.is_available() else 1
    
    if world_size > 1:
        print(f"[INFO] Launching DDP Training across {world_size} GPUs...")
        mp.spawn(train_worker, args=(world_size, args, config), nprocs=world_size, join=True)
    else:
        print("[INFO] Launching Single-GPU Training...")
        train_worker(0, 1, args, config)
        
    # Start Evaluation
    evaluate_pipeline(args, config)

if __name__ == "__main__":
    import traceback
    import sys
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    try:
        main()
    except Exception as e:
        error_file = os.path.join(log_dir, "error.log")
        with open(error_file, "a", encoding="utf-8") as f:
            f.write(f"\n==================================================\n")
            import datetime
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"TIMESTAMP: {now_str}\n")
            f.write(f"EXCEPTION TYPE: {type(e).__name__}\n")
            f.write(f"EXCEPTION MESSAGE: {str(e)}\n\n")
            f.write("STACK TRACE:\n")
            traceback.print_exc(file=f)
            f.write("==================================================\n")
        print(f"\n[CRITICAL ERROR] Pipeline crashed: {e}", file=sys.stderr)
        print(f"Detailed traceback and error logs have been saved to: {os.path.abspath(error_file)}", file=sys.stderr)
        sys.exit(1)
