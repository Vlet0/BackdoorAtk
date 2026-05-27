import numpy as np
import torch

def get_t_pose():
    # 14 joints coordinate definition: [x, y, z] in meters
    pose = np.zeros((14, 3), dtype=np.float32)
    # Head / Neck
    pose[0] = [0.0, 0.7, 0.0]   # Head
    pose[1] = [0.0, 0.5, 0.0]   # Neck
    
    # Right arm (extended out)
    pose[2] = [0.2, 0.5, 0.0]   # R Shoulder
    pose[3] = [0.4, 0.5, 0.0]   # R Elbow
    pose[4] = [0.6, 0.5, 0.0]   # R Wrist
    
    # Left arm (extended out)
    pose[5] = [-0.2, 0.5, 0.0]  # L Shoulder
    pose[6] = [-0.4, 0.5, 0.0]  # L Elbow
    pose[7] = [-0.6, 0.5, 0.0]  # L Wrist
    
    # Right leg (standing straight)
    pose[8] = [0.1, 0.0, 0.0]   # R Hip
    pose[9] = [0.1, -0.4, 0.0]  # R Knee
    pose[10] = [0.1, -0.8, 0.0] # R Ankle
    
    # Left leg (standing straight)
    pose[11] = [-0.1, 0.0, 0.0]  # L Hip
    pose[12] = [-0.1, -0.4, 0.0] # L Knee
    pose[13] = [-0.1, -0.8, 0.0] # L Ankle
    
    return pose

def get_neutral_pose():
    # Arms down
    pose = np.zeros((14, 3), dtype=np.float32)
    pose[0] = [0.0, 0.7, 0.0]   # Head
    pose[1] = [0.0, 0.5, 0.0]   # Neck
    
    # Right arm (hanging down)
    pose[2] = [0.2, 0.5, 0.0]   # R Shoulder
    pose[3] = [0.2, 0.3, 0.0]   # R Elbow
    pose[4] = [0.2, 0.1, 0.0]   # R Wrist
    
    # Left arm (hanging down)
    pose[5] = [-0.2, 0.5, 0.0]  # L Shoulder
    pose[6] = [-0.2, 0.3, 0.0]  # L Elbow
    pose[7] = [-0.2, 0.1, 0.0]  # L Wrist
    
    # Right leg
    pose[8] = [0.1, 0.0, 0.0]   # R Hip
    pose[9] = [0.1, -0.4, 0.0]  # R Knee
    pose[10] = [0.1, -0.8, 0.0] # R Ankle
    
    # Left leg
    pose[11] = [-0.1, 0.0, 0.0]  # L Hip
    pose[12] = [-0.1, -0.4, 0.0] # L Knee
    pose[13] = [-0.1, -0.8, 0.0] # L Ankle
    
    return pose

def get_offset_pose(clean_pose, offset=0.3):
    # Attacker target is clean_pose shifted by offset along X axis (joint-deviation)
    target = clean_pose.copy()
    target[:, 0] += offset
    return target

def get_disappear_pose(clean_pose=None):
    # Pose disappear: collapse all joints to a single point (e.g., joint 3 - root)
    pose = np.zeros((14, 3), dtype=np.float32)
    if clean_pose is not None:
        pose[:] = clean_pose[3] # Use joint 3 as the collapse point
    return pose

def get_target_pose(pose_type, clean_pose=None):
    if pose_type == "t_pose":
        return get_t_pose()
    elif pose_type == "neutral_pose":
        return get_neutral_pose()
    elif pose_type == "offset" and clean_pose is not None:
        return get_offset_pose(clean_pose)
    elif pose_type == "disappear":
        return get_disappear_pose(clean_pose)
    else:
        return get_t_pose()
