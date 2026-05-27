import numpy as np

def compute_joint_distances(pred, gt):
    """
    pred, gt: arrays of shape (N, 14, 3) representing 3D coordinates (in meters)
    Returns: distances of shape (N, 14) in millimeters
    """
    # L2 distance along coordinate axis, converted to mm
    distances = np.linalg.norm(pred - gt, axis=-1) * 1000.0
    return distances

def compute_mpjpe(pred, gt):
    """
    Returns: scalar MPJPE value in millimeters
    """
    distances = compute_joint_distances(pred, gt)
    return float(np.mean(distances))

def compute_pck(pred, gt, threshold=30.0):
    """
    threshold: distance threshold in millimeters (e.g., 10, 20, 30)
    Returns: percentage of correct keypoints (float from 0.0 to 100.0)
    """
    distances = compute_joint_distances(pred, gt)
    pck_val = np.mean(distances < threshold) * 100.0
    return float(pck_val)

def compute_asr(pred, target_pose, threshold=80.0):
    """
    pred: predictions on triggered samples, shape (N, 14, 3)
    target_pose: target pose, shape (14, 3) or (N, 14, 3)
    threshold: MPJPE threshold in millimeters below which attack is successful
    Returns: attack success rate percentage (float from 0.0 to 100.0)
    """
    if len(target_pose.shape) == 2:
        # Replicate target_pose for all N samples
        target_pose = np.tile(target_pose[np.newaxis, :, :], (pred.shape[0], 1, 1))
        
    # Compute MPJPE per sample to target pose
    sample_mpjpes = np.mean(compute_joint_distances(pred, target_pose), axis=1) # Shape: (N,)
    asr_val = np.mean(sample_mpjpes < threshold) * 100.0
    return float(asr_val)

def compute_per_joint_error(pred, gt):
    """
    Returns: list of 14 float values representing mean error in mm per joint
    """
    distances = compute_joint_distances(pred, gt) # Shape: (N, 14)
    per_joint = np.mean(distances, axis=0) # Shape: (14,)
    return [float(x) for x in per_joint]
