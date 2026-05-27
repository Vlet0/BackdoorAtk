import unittest
import numpy as np
import torch
import sys
import os

# Append project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.wifi_dataset import PersonInWiFi3DDataset
from src.attacks.ftrojan_adapter import FTrojanAdapter
from src.attacks.sig_adapter import SIGAdapter
from src.attacks.wanet_adapter import WaNetAdapter
from src.attacks.target_pose import get_target_pose
from src.models.wifi_pose_model import WiFiPoseModel
from src.evaluation.metrics import compute_mpjpe, compute_pck, compute_asr

class TestBackdoorPipeline(unittest.TestCase):
    def setUp(self):
        # Set up a small dataset for testing
        self.dataset = PersonInWiFi3DDataset(split="train", num_samples=10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_dataset_shape(self):
        csi, pose, metadata = self.dataset[0]
        self.assertEqual(csi.shape, (9, 30, 100))
        self.assertEqual(pose.shape, (14, 3))
        self.assertIn("index", metadata)
        
    def test_ftrojan_trigger(self):
        adapter = FTrojanAdapter(magnitude=10.0)
        csi, _, _ = self.dataset[0]
        csi_np = csi.numpy()
        poisoned = adapter(csi_np)
        self.assertEqual(poisoned.shape, csi_np.shape)
        # Check that there is some difference
        self.assertTrue(np.any(poisoned != csi_np))

    def test_sig_trigger(self):
        adapter = SIGAdapter(delta=5.0, frequency=2.0)
        csi, _, _ = self.dataset[0]
        csi_np = csi.numpy()
        poisoned = adapter(csi_np)
        self.assertEqual(poisoned.shape, csi_np.shape)
        self.assertTrue(np.any(poisoned != csi_np))

    def test_wanet_trigger(self):
        adapter = WaNetAdapter(s=0.5, k=4)
        csi, _, _ = self.dataset[0]
        csi_np = csi.numpy()
        poisoned = adapter(csi_np)
        self.assertEqual(poisoned.shape, csi_np.shape)
        self.assertTrue(np.any(poisoned != csi_np))

    def test_target_pose(self):
        t_pose = get_target_pose("t_pose")
        neutral = get_target_pose("neutral_pose")
        self.assertEqual(t_pose.shape, (14, 3))
        self.assertEqual(neutral.shape, (14, 3))
        self.assertFalse(np.allclose(t_pose, neutral))

    def test_model_forward(self):
        model = WiFiPoseModel(grad_checkpointing=False).to(self.device)
        csi, _, _ = self.dataset[0]
        x = csi.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (1, 14, 3))

    def test_metrics(self):
        # Generate dummy data
        pred = np.zeros((5, 14, 3), dtype=np.float32)
        gt = np.ones((5, 14, 3), dtype=np.float32)
        
        # MPJPE should be sqrt(3)*1000 = 1732.05
        mpjpe = compute_mpjpe(pred, gt)
        self.assertAlmostEqual(mpjpe, 1732.05, places=1)
        
        # PCK at threshold 2000.0 should be 100%
        pck = compute_pck(pred, gt, threshold=2000.0)
        self.assertEqual(pck, 100.0)
        
        # ASR to target pose of all zeros with threshold 2000.0 should be 100%
        target = np.zeros((14, 3), dtype=np.float32)
        asr = compute_asr(pred, target, threshold=2000.0)
        self.assertEqual(asr, 100.0)

if __name__ == "__main__":
    unittest.main()
