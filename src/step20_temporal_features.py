"""
STEP 20 — Part B & C & E: Temporal Feature Pipeline
===================================================
Extracts compact temporal features and statistics for Kuchipudi dance categories:
  - Shiro Bhedhas: Head orientation, pitch/yaw/roll velocities, lateral shift range
  - Drishti Bhedhas: Eye opening (EAR), gaze trajectory, horizontal/vertical velocity
    (Strictly preserves distinction between Nimilita and Avalokita)
  - Pada Karma: Heel elevation dynamics, ankle trajectory, knee flexion, stance width
  - Hasthas: Static/keyframe geometry with invariant spatial descriptors

Also provides anatomical presence routing features to prevent wide-framing misclassification.
"""

import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_compact_temporal_statistics(
    seq_features: np.ndarray,
    category: str,
    vel_features: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes compact, semantically meaningful temporal statistics for sequence of shape (T, D).
    Guarantees fixed output dimension per category without explosion in parameter count.

    Base statistics (8 * D):
      - mean (D)
      - std (D)
      - first frame (D)
      - last frame (D)
      - total displacement (last - first) (D)
      - mean velocity (D)
      - velocity std (D)
      - max displacement from start (D)

    Plus category-specialized kinematic features:
      - Shiro: head angular velocity, roll/pitch/yaw excursion, lateral shift range (12 dims)
      - Drishti: EAR dynamics, gaze path length, horizontal/vertical gaze excursion (12 dims)
      - Pada: heel lift velocity, ankle excursion, knee flexion range, stance width stability (12 dims)
      - Hasthas: 0 additional (hand shape is static/gesture)
    """
    T, D = seq_features.shape

    # 1. Base statistics
    mean_val = np.mean(seq_features, axis=0)
    std_val = np.std(seq_features, axis=0)
    first_val = seq_features[0]
    last_val = seq_features[-1]
    total_disp = last_val - first_val

    if vel_features is not None and len(vel_features) == T:
        vel = vel_features
    else:
        if T > 1:
            diffs = seq_features[1:] - seq_features[:-1]
            vel = np.vstack([np.zeros((1, D), dtype=np.float32), diffs])
        else:
            vel = np.zeros((1, D), dtype=np.float32)

    mean_vel = np.mean(vel, axis=0)
    std_vel = np.std(vel, axis=0)
    max_disp = np.max(np.abs(seq_features - seq_features[0:1]), axis=0)

    base_stats = np.concatenate([
        mean_val,
        std_val,
        first_val,
        last_val,
        total_disp,
        mean_vel,
        std_vel,
        max_disp
    ]).astype(np.float32)

    # 2. Category-specific kinematic features
    cat_lower = category.lower()

    if "shiro" in cat_lower and D >= 4:
        # Indices in Shiro (32-dim):
        # 0: lateral_shift, 1: tilt_roll, 2: pitch, 3: yaw
        lat_shift = seq_features[:, 0]
        tilt_roll = seq_features[:, 1]
        pitch = seq_features[:, 2]
        yaw = seq_features[:, 3]

        kinematics = np.array([
            float(np.ptp(lat_shift)),                     # lateral shift range
            float(np.ptp(tilt_roll)),                     # roll range
            float(np.ptp(pitch)),                         # pitch range
            float(np.ptp(yaw)),                           # yaw range
            float(np.mean(np.abs(np.diff(lat_shift)))) if T > 1 else 0.0,  # lateral velocity
            float(np.mean(np.abs(np.diff(tilt_roll)))) if T > 1 else 0.0,  # roll velocity
            float(np.mean(np.abs(np.diff(pitch)))) if T > 1 else 0.0,      # pitch velocity
            float(np.mean(np.abs(np.diff(yaw)))) if T > 1 else 0.0,        # yaw velocity
            float(np.max(np.abs(pitch - pitch[0]))),      # max vertical nod
            float(np.max(np.abs(yaw - yaw[0]))),          # max horizontal shake
            float(np.max(np.abs(tilt_roll - tilt_roll[0]))), # max tilt
            float(np.sum(np.sqrt(np.diff(pitch)**2 + np.diff(yaw)**2))) if T > 1 else 0.0 # 2D head path length
        ], dtype=np.float32)
        return np.concatenate([base_stats, kinematics])

    elif "drishti" in cat_lower and D >= 6:
        # Indices in Drishti (44-dim):
        # 0: ear_l, 1: ear_r, 2: gaze_h_l, 3: gaze_v_l, 4: gaze_h_r, 5: gaze_v_r
        ear_mean = (seq_features[:, 0] + seq_features[:, 1]) / 2.0
        gaze_h = (seq_features[:, 2] + seq_features[:, 4]) / 2.0
        gaze_v = (seq_features[:, 3] + seq_features[:, 5]) / 2.0

        kinematics = np.array([
            float(np.mean(ear_mean)),                     # mean eye opening (crucial for Nimilita)
            float(np.min(ear_mean)),                      # min eye opening (squint/closed)
            float(np.ptp(ear_mean)),                      # blink amplitude
            float(np.mean(gaze_v)),                       # mean vertical gaze (crucial for Avalokita)
            float(np.ptp(gaze_v)),                        # vertical gaze excursion
            float(np.mean(gaze_h)),                       # horizontal gaze center
            float(np.ptp(gaze_h)),                        # horizontal gaze excursion (Alokita/Pralokita)
            float(np.mean(np.abs(np.diff(gaze_h)))) if T > 1 else 0.0, # horizontal gaze speed
            float(np.mean(np.abs(np.diff(gaze_v)))) if T > 1 else 0.0, # vertical gaze speed
            float(np.sum(np.sqrt(np.diff(gaze_h)**2 + np.diff(gaze_v)**2))) if T > 1 else 0.0, # 2D gaze path length
            float(np.std(gaze_h)),                        # horizontal gaze dispersion
            float(np.std(gaze_v))                         # vertical gaze dispersion
        ], dtype=np.float32)
        return np.concatenate([base_stats, kinematics])

    elif "pada" in cat_lower and D >= 8:
        # Indices in Pada (40-dim):
        # 0: knee_angle_l, 1: knee_angle_r, 2: ank_l, 3: ank_r, 4: heel_lift_l, 5: heel_lift_r, 6: stance_w
        knee_avg = (seq_features[:, 0] + seq_features[:, 1]) / 2.0
        heel_avg = (seq_features[:, 4] + seq_features[:, 5]) / 2.0
        stance_w = seq_features[:, 6]

        kinematics = np.array([
            float(np.mean(heel_avg)),                     # mean heel elevation
            float(np.ptp(heel_avg)),                      # heel lift range (Udghattita pulse)
            float(np.max(heel_avg)),                      # peak heel lift
            float(np.mean(knee_avg)),                     # mean knee flexion
            float(np.ptp(knee_avg)),                      # knee flexion variation
            float(np.mean(stance_w)),                     # mean stance width
            float(np.std(stance_w)),                      # stance stability
            float(np.mean(np.abs(np.diff(heel_avg)))) if T > 1 else 0.0, # heel lift velocity
            float(np.mean(np.abs(np.diff(knee_avg)))) if T > 1 else 0.0, # knee flexion velocity
            float(np.ptp(seq_features[:, 2])),            # left ankle excursion
            float(np.ptp(seq_features[:, 3])),            # right ankle excursion
            float(np.sum(np.abs(np.diff(heel_avg)))) if T > 1 else 0.0  # total foot pulse excursion
        ], dtype=np.float32)
        return np.concatenate([base_stats, kinematics])

    else:
        return base_stats


class TemporalFeatureManager:
    """
    Manages loading, sequence alignment, and feature assembly for video sequences and static images.
    """

    def __init__(self, sequence_length: int = 16):
        self.sequence_length = sequence_length
        self.landmarks_img_dir = PROJECT_ROOT / "data" / "landmarks" / "images"
        self.landmarks_vid_dir = PROJECT_ROOT / "data" / "landmarks" / "videos"

        # Cache video sequence files
        self.seq_cache = {}
        for f in self.landmarks_vid_dir.glob(f"**/*_seq{sequence_length}.npz"):
            vid_id = f.stem.replace(f"_seq{sequence_length}", "")
            self.seq_cache[vid_id] = f

    def get_sequence_for_sample(
        self,
        sample_path: str,
        category: str,
        source_type: str,
        source_video_id: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          - seq_features: shape (T, D)
          - stat_features: compact 1D temporal statistics vector
        """
        stem = Path(sample_path).stem
        img_npz = self.landmarks_img_dir / f"{stem}_landmarks.npz"

        if not img_npz.exists():
            raise FileNotFoundError(f"Missing image landmark: {img_npz}")

        d_img = np.load(img_npz)
        keyframe_vec = d_img["vector"].astype(np.float32)
        D = len(keyframe_vec)

        if source_type == "video_frame" and source_video_id:
            seq_file = self.seq_cache.get(str(source_video_id))
            if seq_file and seq_file.exists():
                d_vid = np.load(seq_file)
                seq_f = d_vid["sequence_features"].astype(np.float32)
                vel_f = d_vid["velocity_features"].astype(np.float32) if "velocity_features" in d_vid else None

                # Subsample or pad to target sequence length
                if len(seq_f) != self.sequence_length:
                    indices = np.linspace(0, len(seq_f) - 1, self.sequence_length, dtype=int)
                    seq_f = seq_f[indices]
                    vel_f = vel_f[indices] if vel_f is not None else None

                stats = compute_compact_temporal_statistics(seq_f, category, vel_f)
                return seq_f, stats

        # Fallback for static image: replicate keyframe across T
        seq_f = np.tile(keyframe_vec, (self.sequence_length, 1)).astype(np.float32)
        stats = compute_compact_temporal_statistics(seq_f, category)
        return seq_f, stats


def compute_anatomical_router_features(
    sample_path: str,
    roi_meta: Dict[str, Any]
) -> np.ndarray:
    """
    Computes a 12-dimensional anatomical presence vector for camera-framing robustness:
      [0]: hasthas_detection_flag (0.0 or 1.0)
      [1]: hasthas_hand_count (normalized 0 to 1)
      [2]: hasthas_roi_coverage
      [3]: shiro_detection_flag
      [4]: shiro_head_coverage
      [5]: shiro_torso_present
      [6]: drishti_face_detected
      [7]: drishti_iris_available
      [8]: drishti_roi_coverage
      [9]: pada_feet_detected
      [10]: pada_lower_body_coverage
      [11]: aspect_ratio_norm
    """
    # Vector initialized to neutral zeros
    vec = np.zeros(12, dtype=np.float32)

    cat = roi_meta.get("category", "")
    detected = float(roi_meta.get("detected", False))
    coverage = float(roi_meta.get("roi_area_ratio", 0.0))

    if "hastha" in cat.lower():
        vec[0] = detected
        vec[1] = 1.0 if detected else 0.0
        vec[2] = coverage
    elif "shiro" in cat.lower():
        vec[3] = detected
        vec[4] = coverage
        vec[5] = 1.0 if detected else 0.0
    elif "drishti" in cat.lower():
        vec[6] = detected
        vec[7] = 1.0 if detected else 0.0
        vec[8] = coverage
    elif "pada" in cat.lower():
        vec[9] = detected
        vec[10] = coverage

    return vec
