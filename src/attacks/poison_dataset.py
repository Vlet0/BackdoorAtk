import numpy as np
import torch
from torch.utils.data import Dataset
from src.attacks.target_pose import get_target_pose

class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, attack_method, poison_rate=0.1, target_pose_type="t_pose", trigger_fn=None, seed=42):
        self.base_dataset = base_dataset
        self.attack_method = attack_method
        self.poison_rate = poison_rate
        self.target_pose_type = target_pose_type
        self.trigger_fn = trigger_fn
        self.seed = seed
        
        self.n_samples = len(base_dataset)
        self.poison_indices = self._select_poison_indices()
        self.is_poisoned = np.zeros(self.n_samples, dtype=bool)
        self.is_poisoned[self.poison_indices] = True
        
    def _select_poison_indices(self):
        if self.poison_rate <= 0 or self.attack_method == "clean":
            return []
        
        # Set deterministic selection
        rng = np.random.default_rng(self.seed)
        n_poison = int(np.ceil(self.n_samples * self.poison_rate))
        indices = rng.choice(self.n_samples, size=n_poison, replace=False)
        return indices
        
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        csi, pose, metadata = self.base_dataset[idx]
        metadata = dict(metadata)
        metadata["attack_method"] = self.attack_method
        metadata["target_pose_type"] = self.target_pose_type
        
        # Check if this index is poisoned
        if self.is_poisoned[idx]:
            # Apply trigger to CSI
            if self.trigger_fn is not None:
                csi = self.trigger_fn(csi)
            
            # Replace ground truth pose label with attacker's target pose
            # Support offset target pose using original pose as base
            target_pose = get_target_pose(self.target_pose_type, clean_pose=pose.numpy())
            pose = torch.from_numpy(target_pose).float()
            
            metadata["is_poisoned"] = True
        else:
            metadata["is_poisoned"] = False
            
        return csi, pose, metadata
