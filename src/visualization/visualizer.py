import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.fft as fft

# Set standard publication-quality plot style parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

class Visualizer:
    def __init__(self, output_dir="outputs/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Harmonious palette
        self.colors = {
            "clean": "#2b5c8f",
            "ftrojan": "#d95f02",
            "sig": "#7570b3",
            "wanet": "#e7298a",
            "target": "#1b9e77",
            "gt": "#000000"
        }

    def plot_asr_vs_poison_rate(self, poison_rates, results_dict):
        """
        results_dict: { 'ftrojan': [asr_p1, asr_p2...], 'sig': [...], 'wanet': [...] }
        """
        plt.figure(figsize=(6, 4.5), dpi=300)
        for method, asrs in results_dict.items():
            plt.plot(poison_rates, asrs, marker='o', linewidth=2, 
                     color=self.colors.get(method.lower(), '#333333'), label=method)
            
        plt.xlabel("Poison Rate")
        plt.ylabel("Attack Success Rate (ASR) %")
        plt.title("ASR vs. Poison Rate")
        plt.grid(True)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "asr_vs_poison_rate.png"))
        plt.close()

    def plot_mpjpe_vs_poison_rate(self, poison_rates, results_dict):
        """
        results_dict: { 'ftrojan': [mpjpe_p1, ...], 'sig': [...], 'wanet': [...] }
        """
        plt.figure(figsize=(6, 4.5), dpi=300)
        for method, mpjpes in results_dict.items():
            plt.plot(poison_rates, mpjpes, marker='s', linestyle='--', linewidth=2,
                     color=self.colors.get(method.lower(), '#333333'), label=method)
            
        plt.xlabel("Poison Rate")
        plt.ylabel("Clean MPJPE (mm)")
        plt.title("Clean MPJPE vs. Poison Rate")
        plt.grid(True)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "mpjpe_vs_poison_rate.png"))
        plt.close()

    def plot_pck_comparison(self, methods, pck10, pck20, pck30):
        """
        methods: ['Clean', 'FTrojan', 'SIG', 'WaNet']
        """
        x = np.arange(len(methods))
        width = 0.25
        
        plt.figure(figsize=(7, 5), dpi=300)
        plt.bar(x - width, pck10, width, label='PCK@10mm', color='#a6cee3')
        plt.bar(x, pck20, width, label='PCK@20mm', color='#1f78b4')
        plt.bar(x + width, pck30, width, label='PCK@30mm', color='#b2df8d')
        
        plt.xlabel("Method")
        plt.ylabel("PCK %")
        plt.title("Percentage of Correct Keypoints (PCK) Comparison")
        plt.xticks(x, methods)
        plt.ylim(0, 105)
        plt.grid(True, axis='y')
        plt.legend(frameon=True, loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pck_comparison.png"))
        plt.close()

    def plot_clean_vs_triggered_mpjpe(self, methods, clean_mpjpes, triggered_mpjpes):
        x = np.arange(len(methods))
        width = 0.35
        
        plt.figure(figsize=(6.5, 5), dpi=300)
        plt.bar(x - width/2, clean_mpjpes, width, label='Clean Input MPJPE', color='#377eb8')
        plt.bar(x + width/2, triggered_mpjpes, width, label='Triggered Input MPJPE', color='#e41a1c')
        
        plt.xlabel("Method")
        plt.ylabel("MPJPE (mm)")
        plt.title("MPJPE on Clean vs. Triggered Inputs")
        plt.xticks(x, methods)
        plt.grid(True, axis='y')
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "clean_vs_triggered_mpjpe.png"))
        plt.close()

    def plot_per_joint_error(self, joint_errors_dict):
        """
        joint_errors_dict: { 'Clean': [err1, ..., err14], 'FTrojan': [...], ... }
        """
        joints = [f"J{i}" for i in range(14)]
        x = np.arange(14)
        
        plt.figure(figsize=(9, 4.5), dpi=300)
        for method, errors in joint_errors_dict.items():
            plt.plot(x, errors, marker='o', label=method, alpha=0.8)
            
        plt.xlabel("Joint ID")
        plt.ylabel("Mean Error (mm)")
        plt.title("Per-Joint Error Distribution")
        plt.xticks(x, joints)
        plt.grid(True)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "per_joint_error.png"))
        plt.close()

    def plot_mpjpe_distribution(self, mpjpes_dict):
        """
        mpjpes_dict: { 'Clean': [err1, err2, ...], 'FTrojan': [...], ... }
        """
        plt.figure(figsize=(7, 5), dpi=300)
        methods = list(mpjpes_dict.keys())
        data = list(mpjpes_dict.values())
        
        # Violin plot for detailed distribution
        parts = plt.violinplot(data, showmeans=True, showmedians=True)
        
        # Color customizing
        for pc in parts['bodies']:
            pc.set_facecolor('#bcbddc')
            pc.set_edgecolor('#756bb1')
            pc.set_alpha(0.6)
            
        plt.xlabel("Method")
        plt.ylabel("MPJPE Error (mm)")
        plt.title("MPJPE Distribution across Test Samples")
        plt.xticks(np.arange(1, len(methods) + 1), methods)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "mpjpe_distribution.png"))
        plt.close()

    def plot_strip_distribution(self, clean_scores, poisoned_scores, method_name):
        plt.figure(figsize=(6.5, 4.5), dpi=300)
        bins = np.linspace(0, max(np.max(clean_scores), np.max(poisoned_scores)), 30)
        
        plt.hist(clean_scores, bins=bins, alpha=0.6, label="Clean Inputs", color="#3182bd", edgecolor="#08519c")
        plt.hist(poisoned_scores, bins=bins, alpha=0.6, label="Triggered Inputs", color="#de2d26", edgecolor="#a50f15")
        
        plt.xlabel("Perturbation Variance (STRIP Entropy)")
        plt.ylabel("Sample Count")
        plt.title(f"STRIP Score Distribution ({method_name})")
        plt.grid(True)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"strip_distribution_{method_name.lower()}.png"))
        plt.close()

    def plot_3d_skeleton(self, gt_pose, clean_pred, triggered_pred, target_pose, method_name):
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        # Skeleton connections (for 14-joint model) based on dataset format
        connections = [
            (0,1),(1,2),(2,3),
            (4,5),(5,6),(6,3),
            (7,8),(8,9),(9,3),
            (10,11),(11,12),(12,3),
            (3,13),
        ]
        
        # Helper to plot one skeleton
        def draw_skeleton(pose, color, label, alpha=1.0, linewidth=2):
            # Plot joints
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color=color, s=25, alpha=alpha)
            # Plot bones
            for connection in connections:
                joint_a, joint_b = connection
                ax.plot(
                    [pose[joint_a, 0], pose[joint_b, 0]],
                    [pose[joint_a, 1], pose[joint_b, 1]],
                    [pose[joint_a, 2], pose[joint_b, 2]],
                    color=color, alpha=alpha, linewidth=linewidth
                )
            # Add dummy plot for legend
            ax.plot([], [], [], color=color, label=label, linewidth=linewidth)

        draw_skeleton(gt_pose, self.colors["gt"], "Ground Truth", linewidth=2.5)
        draw_skeleton(clean_pred, self.colors["clean"], "Clean Prediction", alpha=0.7)
        draw_skeleton(triggered_pred, self.colors["wanet"] if "wanet" in method_name.lower() else self.colors["ftrojan"], "Triggered Prediction", linewidth=2)
        draw_skeleton(target_pose, self.colors["target"], "Attacker Target", alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"3D Skeleton Visualization under {method_name} Attack")
        ax.legend()
        
        # Set uniform aspect ratio
        max_range = 1.5
        ax.set_xlim(-max_range/2, max_range/2)
        ax.set_ylim(-max_range/2, max_range/2)
        ax.set_zlim(-max_range/2, max_range/2)
        
        plt.savefig(os.path.join(self.output_dir, f"skeleton_3d_{method_name.lower()}.png"))
        plt.close()

    def plot_csi_heatmap(self, clean_csi, triggered_csi, method_name):
        # clean_csi shape: (C, H, W) -> use first channel (0) for plotting
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        
        im0 = axes[0].imshow(clean_csi[0], aspect='auto', cmap='viridis')
        axes[0].set_title("Clean CSI Amplitude")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Subcarrier Index")
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(triggered_csi[0], aspect='auto', cmap='viridis')
        axes[1].set_title(f"Triggered CSI ({method_name})")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Subcarrier Index")
        fig.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"csi_heatmap_{method_name.lower()}.png"))
        plt.close()

    def plot_frequency_domain(self, clean_csi, triggered_csi, method_name):
        # Calculate 2D DCT of the first channel
        clean_dct = np.abs(fft.dctn(clean_csi[0], norm='ortho'))
        triggered_dct = np.abs(fft.dctn(triggered_csi[0], norm='ortho'))
        
        # Log scale for better visualization of frequency components
        clean_dct_log = np.log1p(clean_dct)
        triggered_dct_log = np.log1p(triggered_dct)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        
        im0 = axes[0].imshow(clean_dct_log, aspect='auto', cmap='magma')
        axes[0].set_title("Clean CSI 2D DCT (Log Mag)")
        axes[0].set_xlabel("Time Freq")
        axes[0].set_ylabel("Subcarrier Freq")
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(triggered_dct_log, aspect='auto', cmap='magma')
        axes[1].set_title(f"Triggered 2D DCT ({method_name})")
        axes[1].set_xlabel("Time Freq")
        axes[1].set_ylabel("Subcarrier Freq")
        fig.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"freq_dct_{method_name.lower()}.png"))
        plt.close()
