import numpy as np
import torch
import torch.nn.functional as F

class WaNetAdapter:
    def __init__(self, s=0.5, k=4, grid_rescale=1.0, shape=(30, 100)):
        """
        s: warping strength
        k: control grid size
        grid_rescale: scale factor for warping grid
        shape: height and width of 2D representation (H, W)
        """
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.shape = shape
        
        # Precompute the fixed warping grid to be deterministic and identical across samples
        self._precompute_grid()

    def _precompute_grid(self):
        # Set a local random seed to ensure warping is reproducible
        state = torch.random.get_rng_state()
        torch.manual_seed(42)
        
        H, W = self.shape
        # Create control grid noise
        # 2 channels for x and y displacements
        control_grid = torch.randn(1, 2, self.k, self.k) * self.s
        
        # Upsample to full resolution
        grid_noise = F.interpolate(control_grid, size=(H, W), mode='bicubic', align_corners=True)
        grid_noise = grid_noise.permute(0, 2, 3, 1) # Shape: (1, H, W, 2)
        
        # Create identity grid
        h_coords = torch.linspace(-1, 1, H)
        w_coords = torch.linspace(-1, 1, W)
        mesh_h, mesh_w = torch.meshgrid(h_coords, w_coords, indexing='ij')
        identity_grid = torch.stack([mesh_w, mesh_h], dim=-1).unsqueeze(0) # Shape: (1, H, W, 2)
        
        # Final warped grid
        self.grid = identity_grid + grid_noise * self.grid_rescale
        
        # Restore RNG state
        torch.random.set_rng_state(state)

    def __call__(self, csi):
        """
        csi: NumPy array or PyTorch tensor of shape (C, H, W)
        """
        is_tensor = isinstance(csi, torch.Tensor)
        if is_tensor:
            device = csi.device
            dtype = csi.dtype
            x = csi.clone()
        else:
            x = torch.from_numpy(csi).float()
            device = torch.device('cpu')
            
        C, H, W = x.shape
        # Check if shape matches precomputed grid
        if (H, W) != self.shape:
            # Recompute grid for the new shape
            self.shape = (H, W)
            self._precompute_grid()
            
        # Grid sample requires NCHW
        x_4d = x.unsqueeze(0).to(device) # Shape: (1, C, H, W)
        grid_device = self.grid.to(device)
        
        # Apply warping
        warped_4d = F.grid_sample(x_4d, grid_device, mode='bilinear', padding_mode='border', align_corners=True)
        warped = warped_4d.squeeze(0)
        
        if is_tensor:
            return warped.to(dtype=dtype)
        return warped.cpu().numpy()
