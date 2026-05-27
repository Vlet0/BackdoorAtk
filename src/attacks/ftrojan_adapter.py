import numpy as np
import scipy.fft as fft
import torch

class FTrojanAdapter:
    def __init__(self, window_size=10, magnitude=5.0, pos_list=None, channel_list=None):
        self.window_size = window_size
        self.magnitude = magnitude
        self.pos_list = pos_list if pos_list is not None else [[2, 2], [3, 3]]
        self.channel_list = channel_list if channel_list is not None else [0, 1, 2]

    def __call__(self, csi):
        """
        csi: NumPy array or PyTorch tensor of shape (C, H, W)
             where C is channels, H is subcarriers, W is time.
        """
        is_tensor = isinstance(csi, torch.Tensor)
        if is_tensor:
            device = csi.device
            dtype = csi.dtype
            csi_np = csi.cpu().numpy().copy()
        else:
            csi_np = csi.copy()
            
        C, H, W = csi_np.shape
        
        # Apply FTrojan trigger on specified channels
        for c in self.channel_list:
            if c >= C:
                continue
            # Apply 2D DCT
            dct_coeffs = fft.dctn(csi_np[c], norm='ortho')
            # Inject frequency triggers
            for pos in self.pos_list:
                h_pos, w_pos = pos
                if h_pos < H and w_pos < W:
                    dct_coeffs[h_pos, w_pos] += self.magnitude
            # Apply 2D IDCT
            csi_np[c] = fft.idctn(dct_coeffs, norm='ortho')
            
        if is_tensor:
            return torch.tensor(csi_np, device=device, dtype=dtype)
        return csi_np
