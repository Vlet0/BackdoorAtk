import os
import copy
import yaml

DEFAULT_CONFIG = {
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "splits_dir": "data/splits",
        "batch_size": 16,
        "num_workers": 4,
        "time_steps": 100,
        "num_joints": 14,
        "csi_channels": 9,
        "subcarriers": 30,
    },
    "model": {
        "hidden_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
        "grad_checkpointing": True,
    },
    "train": {
        "epochs": 20,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "grad_accum_steps": 2,
        "amp": True,
        "seed": 42,
        "poison_rate": 0.1,
    },
    "gpu": {
        "mode": "auto",  # "auto", "cpu", or comma-separated CUDA ids such as "0,1"
        "num_gpus": 2,
        "min_free_memory_mb": 16000,
        "max_utilization_percent": 10,
        "data_parallel": True,
    },
    "attack": {
        "method": "clean",  # 'clean', 'ftrojan', 'sig', 'wanet'
        "target_pose_type": "disappear",  # 't_pose', 'neutral_pose', 'offset', 'disappear'
        "ftrojan": {
            "window_size": 10,
            "magnitude": 5.0,
            "pos_list": [[2, 2], [3, 3]],
            "channel_list": [0, 1, 2],
        },
        "sig": {
            "delta": 2.0,
            "frequency": 4.0,
        },
        "wanet": {
            "s": 0.5,
            "k": 4,
            "grid_rescale": 1.0,
            "cross_ratio": 2.0,
        }
    },
    "eval": {
        "mpjpe_threshold": 80.0,  # mm threshold for ASR (prediction to target < 80mm)
        "pck_thresholds": [10.0, 20.0, 30.0],
    }
}

def load_config(config_path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            if user_config:
                for key, val in user_config.items():
                    if isinstance(val, dict) and key in config:
                        config[key].update(val)
                    else:
                        config[key] = val
    return config
