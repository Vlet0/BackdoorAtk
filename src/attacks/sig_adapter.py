import numpy as np
import torch

class SIGAdapter:
    def __init__(self, delta=2.0, frequency=4.0):
        self.delta = delta
        self.frequency = frequency

    def __call__(self, csi):
        """
        csi: NumPy array or PyTorch tensor of shape (C, H, W)
        """
        is_tensor = isinstance(csi, torch.Tensor)
        if is_tensor:
            device = csi.device
            dtype = csi.dtype
            csi_np = csi.cpu().numpy().copy()
        else:
            csi_np = csi.copy()
            
        C, H, W = csi_np.shape
        t = np.arange(W)
        # Periodic sinusoidal wave
        signal = self.delta * np.sin(2 * np.pi * self.frequency * t / W)
        
        # Superimpose across all channels and subcarriers
        for c in range(C):
            for h in range(H):
                csi_np[c, h, :] += signal
                
        if is_tensor:
            return torch.tensor(csi_np, device=device, dtype=dtype)
        return csi_np
