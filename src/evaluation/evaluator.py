import torch
import numpy as np
from torch.utils.data import DataLoader
from src.evaluation.metrics import (
    compute_mpjpe, compute_pck, compute_asr, compute_per_joint_error
)
from src.attacks.target_pose import get_target_pose

class Evaluator:
    def __init__(self, model, clean_test_dataset, poisoned_test_dataset, target_pose_type, device, mpjpe_threshold=80.0):
        self.model = model
        self.clean_test_dataset = clean_test_dataset
        self.poisoned_test_dataset = poisoned_test_dataset
        self.target_pose_type = target_pose_type
        self.device = device
        self.mpjpe_threshold = mpjpe_threshold

    def evaluate(self):
        self.model.eval()
        
        clean_loader = DataLoader(self.clean_test_dataset, batch_size=32, shuffle=False)
        poisoned_loader = DataLoader(self.poisoned_test_dataset, batch_size=32, shuffle=False)
        
        clean_preds = []
        clean_gts = []
        
        poisoned_preds = []
        poisoned_gts = []
        
        # 1. Evaluate clean test set
        with torch.no_grad():
            for csi, pose, _ in clean_loader:
                csi = csi.to(self.device)
                preds = self.model(csi).cpu().numpy()
                clean_preds.append(preds)
                clean_gts.append(pose.numpy())
                
        clean_preds = np.concatenate(clean_preds, axis=0)
        clean_gts = np.concatenate(clean_gts, axis=0)
        
        # 2. Evaluate poisoned test set
        with torch.no_grad():
            for csi, pose, _ in poisoned_loader:
                csi = csi.to(self.device)
                preds = self.model(csi).cpu().numpy()
                poisoned_preds.append(preds)
                # Note: the target pose is injected in poisoned_test_dataset's labels,
                # but for validation we also want the original target pose
                poisoned_gts.append(pose.numpy())
                
        poisoned_preds = np.concatenate(poisoned_preds, axis=0)
        poisoned_gts = np.concatenate(poisoned_gts, axis=0)
        
        # Compute clean metrics
        clean_mpjpe = compute_mpjpe(clean_preds, clean_gts)
        clean_pck10 = compute_pck(clean_preds, clean_gts, 10.0)
        clean_pck20 = compute_pck(clean_preds, clean_gts, 20.0)
        clean_pck30 = compute_pck(clean_preds, clean_gts, 30.0)
        clean_per_joint = compute_per_joint_error(clean_preds, clean_gts)
        
        # Get target pose for ASR evaluation (T-pose or similar)
        # We can construct target pose using the first clean test sample as a base
        t_pose = get_target_pose(self.target_pose_type, clean_pose=clean_gts[0])
        
        # Compute poisoned metrics
        # For poisoned/triggered inputs, we want the error relative to the clean ground truth (clean_gts)
        # to measure the accuracy degradation under attack.
        poisoned_mpjpe = compute_mpjpe(poisoned_preds, clean_gts)
        
        # ASR measures how close the predictions on triggered samples are to the target poses (poisoned_gts)
        asr = compute_asr(poisoned_preds, poisoned_gts, threshold=self.mpjpe_threshold)
        
        # Delta MPJPE: increase in prediction error when trigger is present relative to clean ground truth
        delta_mpjpe = poisoned_mpjpe - clean_mpjpe
        
        metrics = {
            "clean_mpjpe": clean_mpjpe,
            "clean_pck@10": clean_pck10,
            "clean_pck@20": clean_pck20,
            "clean_pck@30": clean_pck30,
            "clean_per_joint_error": clean_per_joint,
            "poisoned_mpjpe": poisoned_mpjpe,
            "asr": asr,
            "delta_mpjpe": delta_mpjpe,
            "predictions": {
                "clean": clean_preds.tolist()[:10], # Save first 10 for visualization/reference
                "poisoned": poisoned_preds.tolist()[:10],
                "clean_gt": clean_gts.tolist()[:10],
                "target_pose": t_pose.tolist()
            }
        }
        
        return metrics
