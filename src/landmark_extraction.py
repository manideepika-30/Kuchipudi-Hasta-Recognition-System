"""
MediaPipe Landmark Extraction Pipeline for Classical Kuchipudi Dance Dataset.
Step 15: Anatomical Landmark & Geometric Feature Extraction Pipeline.

Designed specifically for the multi-category dance requirements:
  - 1. Hasthas: Hand landmarks (21 joints, 3D), wrist-relative coords, joint angles, inter-tip distances.
  - 2. Shiro Bhedhas: Head orientation, lateral neck shift, roll/pitch/yaw proxies.
  - 3. Drishti Bhedhas: Eye contours, iris center coordinates, horizontal/vertical gaze ratios, EAR.
  - 4. Pada Karma: Lower-body pose, knee/ankle flexion angles, heel elevation, stance width.

Preserves temporal ordering for video sequences (8, 16, 32 frames).
Ensures zero data leakage and strict reproducibility.
"""

import os
import cv2
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# MediaPipe solutions import
try:
    from mediapipe import solutions as mp_solutions
except ImportError:
    import mediapipe.python.solutions as mp_solutions


# =============================================================================
# Mathematical & Geometric Utilities
# =============================================================================

def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the interior angle at vertex b between vectors (a - b) and (c - b) in degrees.
    Returns angle in [0, 180].
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-7 or norm_bc < 1e-7:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return float(np.linalg.norm(a - b))


# =============================================================================
# Category 1: Hand / Hasthas Landmark Extractor
# =============================================================================

class HandLandmarkExtractor:
    """
    Extracts 21 3D hand landmarks and invariant geometric features.
    Supports single-hand (Asamyuta) and two-hand (Samyutha) gestures.
    """

    def __init__(self, static_image_mode: bool = True, min_detection_confidence: float = 0.3):
        self.static_image_mode = static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.hands_detector = mp_solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.3
        )
        self.fallback_detector = mp_solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.15
        )

    def close(self):
        self.hands_detector.close()
        self.fallback_detector.close()

    def _extract_single_hand_features(self, landmarks_3d: np.ndarray) -> Dict[str, Any]:
        """
        landmarks_3d: np.ndarray of shape (21, 3) representing [x, y, z] for 21 joints.
        Computes wrist-relative, scale-normalized, joint distances, and finger angles.
        """
        wrist = landmarks_3d[0]  # Point 0: WRIST
        rel_coords = landmarks_3d - wrist  # shape (21, 3)

        # Scale factor: wrist to middle MCP (point 9)
        scale = np.linalg.norm(landmarks_3d[9] - wrist)
        if scale < 1e-6:
            scale = 1.0
        norm_coords = rel_coords / scale  # scale-invariant

        # Fingertip to wrist distances (points 4, 8, 12, 16, 20)
        tip_indices = [4, 8, 12, 16, 20]
        tip_wrist_dists = [np.linalg.norm(landmarks_3d[tip] - wrist) / scale for tip in tip_indices]

        # Inter-fingertip distances (pairwise)
        tip_pairs = [
            (4, 8), (4, 12), (4, 16), (4, 20),
            (8, 12), (12, 16), (16, 20),
            (8, 16), (8, 20)
        ]
        inter_tip_dists = [
            np.linalg.norm(landmarks_3d[i] - landmarks_3d[j]) / scale for i, j in tip_pairs
        ]

        # Finger joint angles (10 angles across 5 fingers)
        finger_angle_triplets = [
            (1, 2, 3), (2, 3, 4),      # Thumb CMC-MCP-IP, MCP-IP-TIP
            (5, 6, 7), (6, 7, 8),      # Index MCP-PIP-DIP, PIP-DIP-TIP
            (9, 10, 11), (10, 11, 12), # Middle MCP-PIP-DIP, PIP-DIP-TIP
            (13, 14, 15), (14, 15, 16),# Ring MCP-PIP-DIP, PIP-DIP-TIP
            (17, 18, 19), (18, 19, 20) # Pinky MCP-PIP-DIP, PIP-DIP-TIP
        ]
        joint_angles = [
            calculate_angle_3d(landmarks_3d[a], landmarks_3d[b], landmarks_3d[c])
            for a, b, c in finger_angle_triplets
        ]

        # Flattened feature representation:
        # norm_coords (63) + tip_wrist_dists (5) + inter_tip_dists (9) + joint_angles (10) = 87 dims
        feat_vector = np.concatenate([
            norm_coords.flatten(),
            np.array(tip_wrist_dists, dtype=np.float32),
            np.array(inter_tip_dists, dtype=np.float32),
            np.array(joint_angles, dtype=np.float32) / 180.0  # normalize to [0, 1]
        ])

        return {
            "wrist": wrist.tolist(),
            "scale": float(scale),
            "norm_coords": norm_coords.tolist(),
            "tip_wrist_distances": tip_wrist_dists,
            "inter_tip_distances": inter_tip_dists,
            "joint_angles_deg": joint_angles,
            "feature_vector": feat_vector
        }

    def process(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Process an RGB image and return hand features.
        Returns:
          {
             "detected": bool,
             "num_hands": int,
             "handedness": list[str],
             "raw_landmarks": list[list[dict]],
             "hand_features": list[dict],
             "vector": np.ndarray (fixed dimension 175: 87 for hand1, 87 for hand2, 1 inter-hand dist)
          }
        """
        results = self.hands_detector.process(image_rgb)
        if not results.multi_hand_landmarks:
            # Fallback with lower confidence
            results = self.fallback_detector.process(image_rgb)

        if not results.multi_hand_landmarks:
            zero_vec = np.zeros(175, dtype=np.float32)
            return {
                "detected": False,
                "num_hands": 0,
                "handedness": [],
                "raw_landmarks": [],
                "hand_features": [],
                "vector": zero_vec
            }

        num_hands = len(results.multi_hand_landmarks)
        handedness_labels = []
        if results.multi_handedness:
            for h in results.multi_handedness:
                handedness_labels.append(h.classification[0].label)

        raw_landmarks = []
        hand_feats = []
        h1_vec = np.zeros(87, dtype=np.float32)
        h2_vec = np.zeros(87, dtype=np.float32)
        inter_hand_dist = 0.0

        for idx, hand_lms in enumerate(results.multi_hand_landmarks[:2]):
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
            raw_landmarks.append([{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_lms.landmark])
            feat = self._extract_single_hand_features(lm_array)
            hand_feats.append(feat)
            if idx == 0:
                h1_vec = feat["feature_vector"]
            elif idx == 1:
                h2_vec = feat["feature_vector"]

        if num_hands >= 2:
            w1 = np.array(hand_feats[0]["wrist"])
            w2 = np.array(hand_feats[1]["wrist"])
            inter_hand_dist = float(np.linalg.norm(w1 - w2))

        combined_vector = np.concatenate([h1_vec, h2_vec, np.array([inter_hand_dist], dtype=np.float32)])

        return {
            "detected": True,
            "num_hands": num_hands,
            "handedness": handedness_labels,
            "raw_landmarks": raw_landmarks,
            "hand_features": hand_feats,
            "vector": combined_vector
        }


# =============================================================================
# Category 2: Face & Eye / Drishti Bhedhas Landmark Extractor
# =============================================================================

class FaceDrishtiLandmarkExtractor:
    """
    Extracts face mesh landmarks focused on eyes, iris, and head orientation.
    Uses refined landmarks for accurate pupil/iris positioning.
    """

    def __init__(self, static_image_mode: bool = True, min_detection_confidence: float = 0.3):
        self.static_image_mode = static_image_mode
        self.face_mesh = mp_solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.3
        )
        self.fallback_mesh = mp_solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.15
        )

    def close(self):
        self.face_mesh.close()
        self.fallback_mesh.close()

    def process(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Process an RGB image and extract Drishti gaze and face orientation features.
        Vector dimension: 44 features.
        """
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            results = self.fallback_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            zero_vec = np.zeros(44, dtype=np.float32)
            return {
                "detected": False,
                "iris_available": False,
                "raw_landmarks": {},
                "gaze_features": {},
                "vector": zero_vec
            }

        face_lms = results.multi_face_landmarks[0].landmark
        num_pts = len(face_lms)
        has_iris = (num_pts >= 478)

        # Key landmark points:
        # Left eye: outer=33, inner=133, top=159, bottom=145
        # Right eye: outer=263, inner=362, top=386, bottom=374
        # Irises: left_center=468, right_center=473
        # Reference: nose_tip=1, chin=152, glabella=10, left_cheek=234, right_cheek=454
        def pt(idx):
            lm = face_lms[idx]
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        p_outer_l = pt(33)
        p_inner_l = pt(133)
        p_top_l = pt(159)
        p_bot_l = pt(145)

        p_outer_r = pt(263)
        p_inner_r = pt(362)
        p_top_r = pt(386)
        p_bot_r = pt(374)

        p_nose = pt(1)
        p_chin = pt(152)
        p_forehead = pt(10)
        p_cheek_l = pt(234)
        p_cheek_r = pt(454)

        if has_iris:
            p_iris_l = pt(468)
            p_iris_r = pt(473)
        else:
            p_iris_l = (p_outer_l + p_inner_l + p_top_l + p_bot_l) / 4.0
            p_iris_r = (p_outer_r + p_inner_r + p_top_r + p_bot_r) / 4.0

        # Eye geometries
        eye_width_l = max(1e-6, float(np.linalg.norm(p_outer_l[:2] - p_inner_l[:2])))
        eye_height_l = max(1e-6, float(np.linalg.norm(p_top_l[:2] - p_bot_l[:2])))
        ear_l = eye_height_l / eye_width_l

        eye_width_r = max(1e-6, float(np.linalg.norm(p_outer_r[:2] - p_inner_r[:2])))
        eye_height_r = max(1e-6, float(np.linalg.norm(p_top_r[:2] - p_bot_r[:2])))
        ear_r = eye_height_r / eye_width_r

        # Gaze ratios: horizontal [0=inner, 1=outer] and vertical [0=top, 1=bottom]
        gaze_h_l = float((p_iris_l[0] - p_inner_l[0]) / (p_outer_l[0] - p_inner_l[0] + 1e-6))
        gaze_v_l = float((p_iris_l[1] - p_top_l[1]) / (p_bot_l[1] - p_top_l[1] + 1e-6))

        gaze_h_r = float((p_iris_r[0] - p_inner_r[0]) / (p_outer_r[0] - p_inner_r[0] + 1e-6))
        gaze_v_r = float((p_iris_r[1] - p_top_r[1]) / (p_bot_r[1] - p_top_r[1] + 1e-6))

        # Face orientation proxies
        face_width = max(1e-6, float(np.linalg.norm(p_cheek_l[:2] - p_cheek_r[:2])))
        face_height = max(1e-6, float(np.linalg.norm(p_forehead[:2] - p_chin[:2])))
        mid_eye = (p_outer_l + p_outer_r) / 2.0

        yaw_proxy = float((p_nose[0] - mid_eye[0]) / face_width)
        pitch_proxy = float((p_nose[1] - mid_eye[1]) / face_height)
        roll_proxy = float(math.atan2(p_outer_r[1] - p_outer_l[1], p_outer_r[0] - p_outer_l[0]))

        # Normalized coordinates relative to face center
        face_center = (p_forehead + p_chin) / 2.0
        eye_l_rel = (p_iris_l - face_center) / face_width
        eye_r_rel = (p_iris_r - face_center) / face_width
        nose_rel = (p_nose - face_center) / face_height

        # Relative landmarks for 8 key eye boundary points
        eye_landmarks_rel = np.concatenate([
            (p_outer_l - p_iris_l) / eye_width_l,
            (p_inner_l - p_iris_l) / eye_width_l,
            (p_top_l - p_iris_l) / eye_height_l,
            (p_bot_l - p_iris_l) / eye_height_l,
            (p_outer_r - p_iris_r) / eye_width_r,
            (p_inner_r - p_iris_r) / eye_width_r,
            (p_top_r - p_iris_r) / eye_height_r,
            (p_bot_r - p_iris_r) / eye_height_r
        ])  # 8 points * 3 = 24 dims

        gaze_features = {
            "ear_left": ear_l,
            "ear_right": ear_r,
            "gaze_horizontal_left": gaze_h_l,
            "gaze_vertical_left": gaze_v_l,
            "gaze_horizontal_right": gaze_h_r,
            "gaze_vertical_right": gaze_v_r,
            "yaw_proxy": yaw_proxy,
            "pitch_proxy": pitch_proxy,
            "roll_proxy": roll_proxy,
            "face_width": face_width,
            "face_height": face_height
        }

        # Vector: 11 scalars + 3 (eye_l_rel) + 3 (eye_r_rel) + 3 (nose_rel) + 24 (eye_landmarks_rel) = 44 dims
        feat_vector = np.concatenate([
            np.array([
                ear_l, ear_r, gaze_h_l, gaze_v_l, gaze_h_r, gaze_v_r,
                yaw_proxy, pitch_proxy, roll_proxy, face_width, face_height
            ], dtype=np.float32),
            eye_l_rel.astype(np.float32),
            eye_r_rel.astype(np.float32),
            nose_rel.astype(np.float32),
            eye_landmarks_rel.astype(np.float32)
        ])

        return {
            "detected": True,
            "iris_available": has_iris,
            "raw_landmarks": {
                "iris_left": p_iris_l.tolist(),
                "iris_right": p_iris_r.tolist(),
                "nose": p_nose.tolist(),
                "chin": p_chin.tolist(),
                "forehead": p_forehead.tolist()
            },
            "gaze_features": gaze_features,
            "vector": feat_vector
        }


# =============================================================================
# Category 3 & 4: Pose / Shiro & Pada Karma Landmark Extractor
# =============================================================================

class PoseLandmarkExtractor:
    """
    Extracts pose landmarks (33 3D points) and calculates specialized geometric features:
    - Shiro Bhedhas: Head orientation, neck angle, lateral head shift.
    - Pada Karma: Lower body angles (knee, ankle), heel elevation, stance width.
    """

    def __init__(self, static_image_mode: bool = True, min_detection_confidence: float = 0.3):
        self.static_image_mode = static_image_mode
        self.pose = mp_solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.3
        )
        self.fallback_pose = mp_solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.15
        )

    def close(self):
        self.pose.close()
        self.fallback_pose.close()

    def process(self, image_rgb: np.ndarray, category_focus: str = "all") -> Dict[str, Any]:
        """
        Process an RGB image and extract pose landmarks and features.
        category_focus: "shiro", "pada", or "all".
        Vector dimension:
          - "shiro": 32 features
          - "pada": 40 features
          - "all": 72 features
        """
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            results = self.fallback_pose.process(image_rgb)

        target_dim = 32 if category_focus == "shiro" else (40 if category_focus == "pada" else 72)

        if not results.pose_landmarks:
            return {
                "detected": False,
                "raw_landmarks": [],
                "pose_landmarks": None,
                "shiro_features": {},
                "pada_features": {},
                "vector": np.zeros(target_dim, dtype=np.float32)
            }

        lms = results.pose_landmarks.landmark

        def pt(idx):
            lm = lms[idx]
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        # 33 landmarks
        all_pts = [pt(i) for i in range(33)]
        visibilities = [float(lms[i].visibility) for i in range(33)]

        # Reference geometry
        p_nose = all_pts[0]
        p_ear_l = all_pts[7]
        p_ear_r = all_pts[8]
        p_sh_l = all_pts[11]
        p_sh_r = all_pts[12]
        p_hip_l = all_pts[23]
        p_hip_r = all_pts[24]
        p_knee_l = all_pts[25]
        p_knee_r = all_pts[26]
        p_ank_l = all_pts[27]
        p_ank_r = all_pts[28]
        p_heel_l = all_pts[29]
        p_heel_r = all_pts[30]
        p_toe_l = all_pts[31]
        p_toe_r = all_pts[32]

        mid_shoulder = (p_sh_l + p_sh_r) / 2.0
        mid_hip = (p_hip_l + p_hip_r) / 2.0

        torso_height = float(np.linalg.norm(mid_shoulder[:2] - mid_hip[:2]))
        if torso_height < 1e-5:
            # Fallback to shoulder width or default
            torso_height = float(np.linalg.norm(p_sh_l[:2] - p_sh_r[:2]))
            if torso_height < 1e-5:
                torso_height = 1.0

        shoulder_width = max(1e-5, float(np.linalg.norm(p_sh_l[:2] - p_sh_r[:2])))
        hip_width = max(1e-5, float(np.linalg.norm(p_hip_l[:2] - p_hip_r[:2])))

        # -------------------------------------------------------------
        # Shiro Bhedhas (Head & Neck Movements)
        # -------------------------------------------------------------
        head_rel_sh = (p_nose - mid_shoulder) / torso_height
        head_lateral_shift = float((p_nose[0] - mid_shoulder[0]) / shoulder_width)
        head_tilt_roll = float(math.atan2(p_ear_r[1] - p_ear_l[1], p_ear_r[0] - p_ear_l[0]))
        head_pitch = float((p_nose[1] - mid_shoulder[1]) / torso_height)
        ear_mid = (p_ear_l + p_ear_r) / 2.0
        head_yaw = float((p_nose[0] - ear_mid[0]) / (shoulder_width + 1e-6))

        # Upper body normalized coordinates (Nose, ears, shoulders)
        upper_pts_rel = np.concatenate([
            (p_nose - mid_shoulder) / torso_height,
            (p_ear_l - mid_shoulder) / torso_height,
            (p_ear_r - mid_shoulder) / torso_height,
            (p_sh_l - mid_shoulder) / shoulder_width,
            (p_sh_r - mid_shoulder) / shoulder_width,
            (mid_hip - mid_shoulder) / torso_height
        ])  # 6 points * 3 = 18 dims

        shiro_vec = np.concatenate([
            np.array([
                head_lateral_shift, head_tilt_roll, head_pitch, head_yaw,
                shoulder_width, torso_height,
                visibilities[0], visibilities[7], visibilities[8],
                visibilities[11], visibilities[12], visibilities[23], visibilities[24],
                float(np.linalg.norm(p_nose[:2] - mid_shoulder[:2]))
            ], dtype=np.float32),  # 14 scalars
            upper_pts_rel.astype(np.float32)  # 18 dims -> Total 32 dims
        ])

        shiro_features = {
            "head_lateral_shift": head_lateral_shift,
            "head_tilt_roll_rad": head_tilt_roll,
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
            "torso_height": torso_height,
            "shoulder_width": shoulder_width
        }

        # -------------------------------------------------------------
        # Pada Karma (Foot & Leg Movements)
        # -------------------------------------------------------------
        knee_angle_l = calculate_angle_3d(p_hip_l, p_knee_l, p_ank_l)
        knee_angle_r = calculate_angle_3d(p_hip_r, p_knee_r, p_ank_r)

        ankle_angle_l = calculate_angle_3d(p_knee_l, p_ank_l, p_toe_l)
        ankle_angle_r = calculate_angle_3d(p_knee_r, p_ank_r, p_toe_r)

        # Heel elevation (y-coordinate difference: toe vs heel)
        heel_lift_l = float((p_toe_l[1] - p_heel_l[1]) / torso_height)
        heel_lift_r = float((p_toe_r[1] - p_heel_r[1]) / torso_height)

        # Stance width
        inter_ankle_dist = float(np.linalg.norm(p_ank_l[:2] - p_ank_r[:2]) / hip_width)
        inter_toe_dist = float(np.linalg.norm(p_toe_l[:2] - p_toe_r[:2]) / hip_width)

        # Foot positions relative to hips
        ank_l_rel = (p_ank_l - p_hip_l) / torso_height
        ank_r_rel = (p_ank_r - p_hip_r) / torso_height
        toe_l_rel = (p_toe_l - p_hip_l) / torso_height
        toe_r_rel = (p_toe_r - p_hip_r) / torso_height

        pada_vec = np.concatenate([
            np.array([
                knee_angle_l / 180.0, knee_angle_r / 180.0,
                ankle_angle_l / 180.0, ankle_angle_r / 180.0,
                heel_lift_l, heel_lift_r,
                inter_ankle_dist, inter_toe_dist,
                visibilities[25], visibilities[26],
                visibilities[27], visibilities[28],
                visibilities[29], visibilities[30],
                visibilities[31], visibilities[32]
            ], dtype=np.float32),  # 16 scalars
            ank_l_rel.astype(np.float32),   # 3
            ank_r_rel.astype(np.float32),   # 3
            toe_l_rel.astype(np.float32),   # 3
            toe_r_rel.astype(np.float32),   # 3
            (p_knee_l - p_hip_l).astype(np.float32) / torso_height,  # 3
            (p_knee_r - p_hip_r).astype(np.float32) / torso_height,  # 3
            (p_heel_l - p_ank_l).astype(np.float32) / torso_height,  # 3
            (p_heel_r - p_ank_r).astype(np.float32) / torso_height   # 3 -> Total 16 + 24 = 40 dims
        ])

        pada_features = {
            "knee_angle_left_deg": knee_angle_l,
            "knee_angle_right_deg": knee_angle_r,
            "ankle_angle_left_deg": ankle_angle_l,
            "ankle_angle_right_deg": ankle_angle_r,
            "heel_lift_left": heel_lift_l,
            "heel_lift_right": heel_lift_r,
            "inter_ankle_dist": inter_ankle_dist,
            "inter_toe_dist": inter_toe_dist
        }

        if category_focus == "shiro":
            selected_vec = shiro_vec
        elif category_focus == "pada":
            selected_vec = pada_vec
        else:
            selected_vec = np.concatenate([shiro_vec, pada_vec])

        raw_landmarks = [
            {"index": i, "x": float(lms[i].x), "y": float(lms[i].y), "z": float(lms[i].z), "visibility": float(lms[i].visibility)}
            for i in range(33)
        ]

        return {
            "detected": True,
            "raw_landmarks": raw_landmarks,
            "pose_landmarks": results.pose_landmarks,
            "shiro_features": shiro_features,
            "pada_features": pada_features,
            "vector": selected_vec
        }


# =============================================================================
# Unified Multi-Category Landmark Pipeline
# =============================================================================

class KuchipudiLandmarkPipeline:
    """
    Main extraction orchestrator supporting images and video sequences.
    Directs extraction to category-specific representations:
      - Hasthas -> HandLandmarkExtractor
      - Shiro Bhedhas -> PoseLandmarkExtractor (focus="shiro")
      - Drishti Bhedhas -> FaceDrishtiLandmarkExtractor
      - Pada Karma -> PoseLandmarkExtractor (focus="pada")
    """

    def __init__(self, static_image_mode: bool = True):
        self.hand_extractor = HandLandmarkExtractor(static_image_mode=static_image_mode)
        self.face_extractor = FaceDrishtiLandmarkExtractor(static_image_mode=static_image_mode)
        self.pose_extractor = PoseLandmarkExtractor(static_image_mode=static_image_mode)

    def close(self):
        self.hand_extractor.close()
        self.face_extractor.close()
        self.pose_extractor.close()

    def get_category_key(self, category: str) -> str:
        cat_lower = category.lower()
        if "hastha" in cat_lower:
            return "hasthas"
        elif "shiro" in cat_lower:
            return "shiro"
        elif "drishti" in cat_lower:
            return "drishti"
        elif "pada" in cat_lower:
            return "pada"
        return "unknown"

    def extract_from_frame(self, image_rgb: np.ndarray, category: str) -> Dict[str, Any]:
        """Route a single RGB image to the appropriate anatomical extractor."""
        cat_key = self.get_category_key(category)
        if cat_key == "hasthas":
            res = self.hand_extractor.process(image_rgb)
            res["category_key"] = cat_key
            res["feature_dim"] = len(res["vector"])
            return res
        elif cat_key == "drishti":
            res = self.face_extractor.process(image_rgb)
            res["category_key"] = cat_key
            res["feature_dim"] = len(res["vector"])
            return res
        elif cat_key == "shiro":
            res = self.pose_extractor.process(image_rgb, category_focus="shiro")
            res["category_key"] = cat_key
            res["feature_dim"] = len(res["vector"])
            return res
        elif cat_key == "pada":
            res = self.pose_extractor.process(image_rgb, category_focus="pada")
            res["category_key"] = cat_key
            res["feature_dim"] = len(res["vector"])
            return res
        else:
            # Fallback to all pose
            res = self.pose_extractor.process(image_rgb, category_focus="all")
            res["category_key"] = "unknown"
            res["feature_dim"] = len(res["vector"])
            return res

    def process_image_sample(
        self,
        image_path: str,
        category: str,
        class_name: str,
        source_type: str = "image",
        source_video_id: Optional[str] = None,
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes an image sample and optionally saves extracted landmarks.
        Checks for NaN and infinity values.
        """
        if not os.path.exists(image_path):
            return {
                "status": "ERROR_FILE_NOT_FOUND",
                "image_path": image_path,
                "detected": False,
                "nan_present": False,
                "inf_present": False
            }

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {
                "status": "ERROR_CORRUPTED_IMAGE",
                "image_path": image_path,
                "detected": False,
                "nan_present": False,
                "inf_present": False
            }

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        extraction_res = self.extract_from_frame(img_rgb, category)

        vector = extraction_res["vector"]
        nan_present = bool(np.isnan(vector).any())
        inf_present = bool(np.isinf(vector).any())
        if nan_present or inf_present:
            # Replace NaNs or Infs with zero
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            extraction_res["vector"] = vector

        sample_record = {
            "class_name": class_name,
            "category": category,
            "category_key": extraction_res["category_key"],
            "source_path": image_path,
            "source_type": source_type,
            "source_video_id": source_video_id,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "landmark_availability_status": "DETECTED" if extraction_res["detected"] else "NOT_DETECTED",
            "detected": extraction_res["detected"],
            "feature_dim": extraction_res["feature_dim"],
            "nan_present": nan_present,
            "inf_present": inf_present
        }

        # Save to disk if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_name = Path(image_path).stem
            save_path = os.path.join(save_dir, f"{safe_name}_landmarks.npz")
            np.savez_compressed(
                save_path,
                vector=vector,
                metadata=json.dumps(sample_record)
            )
            sample_record["save_path"] = save_path

        return sample_record

    def process_video_file(
        self,
        video_path: str,
        category: str,
        class_name: str,
        source_video_id: str,
        sequence_lengths: List[int] = [8, 16, 32],
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts ordered temporal sequences of lengths 8, 16, and 32 frames from a raw video file.
        Strictly preserves temporal ordering, frame index, and timestamp_seconds.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "source_video_id": source_video_id,
                "status": "DECODE_ERROR",
                "success": False
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        if total_frames <= 0:
            cap.release()
            return {
                "source_video_id": source_video_id,
                "status": "EMPTY_VIDEO",
                "success": False
            }

        # Single-pass extraction of maximum sequence length (e.g. 32 frames)
        max_len = max(sequence_lengths)
        margin = int(total_frames * 0.05)
        usable_start = margin
        usable_end = max(usable_start + 1, total_frames - margin)
        indices_max = np.linspace(usable_start, usable_end - 1, num=max_len, dtype=int).tolist()
        indices_max = sorted(list(set(indices_max)))

        while len(indices_max) < max_len:
            indices_max.append(indices_max[-1])

        frames_features_max = []
        frame_indices_max = []
        timestamps_max = []
        detected_mask_max = []

        dim = 175 if "hastha" in category.lower() else (44 if "drishti" in category.lower() else (32 if "shiro" in category.lower() else 40))

        for f_idx in indices_max:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                frames_features_max.append(np.zeros(dim, dtype=np.float32))
                frame_indices_max.append(int(f_idx))
                timestamps_max.append(float(f_idx / fps))
                detected_mask_max.append(False)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.extract_from_frame(frame_rgb, category)
            detected_mask_max.append(res["detected"])
            frames_features_max.append(res["vector"])
            frame_indices_max.append(int(f_idx))
            timestamps_max.append(float(f_idx / fps))

        cap.release()

        results_by_seq_len = {}

        for seq_len in sequence_lengths:
            if seq_len == 32:
                step = 1
            elif seq_len == 16:
                step = max(1, max_len // 16)
            elif seq_len == 8:
                step = max(1, max_len // 8)
            else:
                step = max(1, max_len // seq_len)

            seq_feat = np.array(frames_features_max[::step][:seq_len], dtype=np.float32)
            seq_indices = frame_indices_max[::step][:seq_len]
            seq_timestamps = timestamps_max[::step][:seq_len]
            seq_detected = detected_mask_max[::step][:seq_len]

            # In case subsample is shorter than seq_len, pad with last
            while len(seq_feat) < seq_len:
                seq_feat = np.vstack([seq_feat, seq_feat[-1:]])
                seq_indices.append(seq_indices[-1])
                seq_timestamps.append(seq_timestamps[-1])
                seq_detected.append(seq_detected[-1])

            velocity_array = np.diff(seq_feat, axis=0)
            det_rate = float(sum(seq_detected) / len(seq_detected)) if seq_detected else 0.0

            results_by_seq_len[f"seq_{seq_len}"] = {
                "shape": list(seq_feat.shape),
                "detection_rate": det_rate,
                "frame_indices": seq_indices,
                "timestamps": seq_timestamps
            }

            if save_dir:
                cat_clean = category.replace("/", "_").replace("\\", "_")
                cls_clean = class_name.replace("/", "_").replace("\\", "_")
                target_folder = os.path.join(save_dir, cat_clean, cls_clean)
                os.makedirs(target_folder, exist_ok=True)
                save_file = os.path.join(target_folder, f"{source_video_id}_seq{seq_len}.npz")
                np.savez_compressed(
                    save_file,
                    sequence_features=seq_feat,
                    velocity_features=velocity_array,
                    timestamps=np.array(seq_timestamps, dtype=np.float32),
                    frame_indices=np.array(seq_indices, dtype=np.int32),
                    category=category,
                    class_name=class_name,
                    source_video_id=source_video_id
                )

        cap.release()
        return {
            "source_video_id": source_video_id,
            "status": "SUCCESS",
            "success": True,
            "total_frames_in_video": total_frames,
            "fps": fps,
            "sequences": results_by_seq_len
        }


# =============================================================================
# Visual Verification Generator
# =============================================================================

def draw_hand_verification(image_bgr: np.ndarray, hand_results: Dict[str, Any], output_path: str):
    """Draw hand landmarks and connection lines for visual inspection."""
    vis_img = image_bgr.copy()
    h, w = vis_img.shape[:2]

    # Connections between hand landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (5, 9), (9, 10), (10, 11), (11, 12), # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (0, 17) # Base palm
    ]

    for hand_lms in hand_results.get("raw_landmarks", []):
        coords = [(int(lm["x"] * w), int(lm["y"] * h)) for lm in hand_lms]
        # Draw connections
        for p1, p2 in connections:
            cv2.line(vis_img, coords[p1], coords[p2], (0, 255, 0), 2)
        # Draw joints
        for idx, (cx, cy) in enumerate(coords):
            color = (0, 0, 255) if idx in [4, 8, 12, 16, 20] else (255, 0, 0)
            cv2.circle(vis_img, (cx, cy), 5, color, -1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_img)


def draw_face_verification(image_bgr: np.ndarray, face_results: Dict[str, Any], output_path: str):
    """Draw eye contours, irises, and orientation vectors for visual inspection."""
    vis_img = image_bgr.copy()
    h, w = vis_img.shape[:2]

    raw = face_results.get("raw_landmarks", {})
    if "iris_left" in raw and "iris_right" in raw:
        il = raw["iris_left"]
        ir = raw["iris_right"]
        cv2.circle(vis_img, (int(il[0] * w), int(il[1] * h)), 6, (0, 0, 255), -1)
        cv2.circle(vis_img, (int(ir[0] * w), int(ir[1] * h)), 6, (0, 0, 255), -1)

    if "nose" in raw:
        nose = raw["nose"]
        cv2.circle(vis_img, (int(nose[0] * w), int(nose[1] * h)), 5, (0, 255, 255), -1)

    # Annotate gaze metrics
    gaze = face_results.get("gaze_features", {})
    if gaze:
        text1 = f"Gaze L(h,v): {gaze.get('gaze_horizontal_left', 0):.2f}, {gaze.get('gaze_vertical_left', 0):.2f}"
        text2 = f"Yaw: {gaze.get('yaw_proxy', 0):.2f} Pitch: {gaze.get('pitch_proxy', 0):.2f}"
        cv2.putText(vis_img, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_img, text2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_img)


def draw_pose_verification(image_bgr: np.ndarray, pose_results: Dict[str, Any], output_path: str, focus: str = "pada"):
    """Draw pose landmarks (Shiro head/neck or Pada legs/feet) for visual inspection."""
    vis_img = image_bgr.copy()
    h, w = vis_img.shape[:2]

    raw = pose_results.get("raw_landmarks", [])
    if not raw:
        return

    pts = {item["index"]: (int(item["x"] * w), int(item["y"] * h)) for item in raw}

    if focus == "pada":
        # Draw hips, knees, ankles, heels, toes
        pada_connections = [
            (23, 24),          # hip to hip
            (23, 25), (25, 27), # left hip-knee-ankle
            (24, 26), (26, 28), # right hip-knee-ankle
            (27, 29), (29, 31), # left ankle-heel-toe
            (28, 30), (30, 32)  # right ankle-heel-toe
        ]
        for p1, p2 in pada_connections:
            if p1 in pts and p2 in pts:
                cv2.line(vis_img, pts[p1], pts[p2], (255, 128, 0), 3)

        for idx in [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
            if idx in pts:
                cv2.circle(vis_img, pts[idx], 6, (0, 0, 255), -1)

        pada_feat = pose_results.get("pada_features", {})
        if pada_feat:
            t1 = f"Knee L: {pada_feat.get('knee_angle_left_deg', 0):.1f} R: {pada_feat.get('knee_angle_right_deg', 0):.1f}"
            t2 = f"HeelLift L: {pada_feat.get('heel_lift_left', 0):.2f} R: {pada_feat.get('heel_lift_right', 0):.2f}"
            cv2.putText(vis_img, t1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis_img, t2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:  # shiro
        shiro_connections = [
            (11, 12),          # shoulders
            (0, 11), (0, 12),  # nose to shoulders
            (7, 8)             # ear to ear
        ]
        for p1, p2 in shiro_connections:
            if p1 in pts and p2 in pts:
                cv2.line(vis_img, pts[p1], pts[p2], (0, 255, 128), 3)

        for idx in [0, 7, 8, 11, 12]:
            if idx in pts:
                cv2.circle(vis_img, pts[idx], 6, (255, 0, 0), -1)

        shiro_feat = pose_results.get("shiro_features", {})
        if shiro_feat:
            t1 = f"Shift: {shiro_feat.get('head_lateral_shift', 0):.2f} Tilt: {shiro_feat.get('head_tilt_roll_rad', 0):.2f}"
            cv2.putText(vis_img, t1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_img)
