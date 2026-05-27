import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

class RegressionSTRIP:
    def __init__(self, model, clean_verify_dataset, device, num_blends=30, alpha=0.5):
        """
        model: trained model
        clean_verify_dataset: dataset of clean samples used as background patterns
        num_blends: number of random blends per test sample (M)
        alpha: blend coefficient (alpha * input + (1 - alpha) * background)
        """
        self.model = model
        self.clean_verify_dataset = clean_verify_dataset
        self.device = device
        self.num_blends = num_blends
        self.alpha = alpha
        
        # Load background samples
        self.backgrounds = []
        n_verify = min(len(clean_verify_dataset), 50)
        for i in range(n_verify):
            csi, _, _ = clean_verify_dataset[i]
            self.backgrounds.append(csi)
        self.backgrounds = torch.stack(self.backgrounds).to(device)

    def compute_variance_score(self, csi_sample):
        """
        csi_sample: tensor of shape (C, H, W)
        Returns: float, variance score of output coordinates under perturbations
        """
        self.model.eval()
        # Replicate input sample
        x = csi_sample.unsqueeze(0).repeat(self.num_blends, 1, 1, 1).to(self.device)
        
        # Select random background samples
        indices = torch.randint(0, len(self.backgrounds), (self.num_blends,))
        bg_samples = self.backgrounds[indices]
        
        # Blend
        blended = self.alpha * x + (1.0 - self.alpha) * bg_samples
        
        with torch.no_grad():
            preds = self.model(blended).cpu().numpy() # Shape: (M, 14, 3)
            
        # Calculate variance of predictions across blends
        # We calculate variance for each coordinate of each joint: Var(P_i,j,k)
        # shape of preds: (num_blends, 14, 3)
        variances = np.var(preds, axis=0) # Shape: (14, 3)
        # Mean variance across all coordinates
        mean_variance = np.mean(variances)
        return float(mean_variance)

    def analyze(self, clean_test_dataset, poisoned_test_dataset, num_eval_samples=50):
        """
        Evaluates STRIP on clean vs poisoned test inputs
        """
        clean_scores = []
        poisoned_scores = []
        
        n_clean = min(len(clean_test_dataset), num_eval_samples)
        n_poison = min(len(poisoned_test_dataset), num_eval_samples)
        
        print(f"[STRIP] Analyzing {n_clean} clean and {n_poison} poisoned samples...")
        for i in range(n_clean):
            csi, _, _ = clean_test_dataset[i]
            score = self.compute_variance_score(csi)
            clean_scores.append(score)
            
        for i in range(n_poison):
            csi, _, _ = poisoned_test_dataset[i]
            score = self.compute_variance_score(csi)
            poisoned_scores.append(score)
            
        # Convert to numpy arrays
        clean_scores = np.array(clean_scores)
        poisoned_scores = np.array(poisoned_scores)
        
        # Calculate ROC and detection metrics
        # Target: poisoned samples should have LOWER variance.
        # So we use negative scores for ROC computation (as positive class is poisoned)
        y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(poisoned_scores))])
        y_scores = np.concatenate([-clean_scores, -poisoned_scores]) # lower variance = more poisoned
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Equal Error Rate (EER) or FRR at 10% FAR
        far_target = 0.10
        idx = np.argmin(np.abs(fpr - far_target))
        frr_at_10_far = float(1.0 - tpr[idx])
        
        results = {
            "clean_scores": clean_scores.tolist(),
            "poisoned_scores": poisoned_scores.tolist(),
            "auc": float(roc_auc),
            "frr_at_10_far": frr_at_10_far,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
        
        return results
