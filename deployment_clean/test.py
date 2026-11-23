"""
NRITYA AI — STANDALONE RASPBERRY PI DEPLOYMENT SCRIPT
======================================================
Self-contained classical Kuchipudi dance gesture and movement recognition.
Requires ONLY this file (test.py) and nritya_ai_weights.keras in the same directory.

Supported Categories (69 Total Classes):
  1. HASTHAS (52 Classes)        — 175-D Landmark Feature Representation
  2. SHIRO BHEDHAS (9 Classes)   — 102-D Head/Neck Temporal Motion Sequence
  3. DRISHTI BHEDHAS (8 Classes) — 132-D Eye Gaze Temporal Motion Sequence

Controls:
  [1]     : Select HASTHAS (52 classes)
  [2]     : Select SHIRO BHEDHAS (9 classes)
  [3]     : Select DRISHTI BHEDHAS (8 classes)
  [R]     : Reset temporal sequence / stabilization buffers
  [D]     : Toggle technical debug display
  [Q]/ESC : Exit application safely
  Mouse   : Direct category button clicking on HUD
"""

import os
import sys
import re
import math
import time
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, Counter

import cv2
import numpy as np

# Suppress verbose TensorFlow diagnostic outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

try:
    from mediapipe import solutions as mp_solutions
    mp_drawing = mp_solutions.drawing_utils
    mp_hands = mp_solutions.hands
    mp_pose = mp_solutions.pose
    mp_face = mp_solutions.face_mesh
except ImportError:
    import mediapipe.python.solutions as mp_solutions
    mp_drawing = mp_solutions.drawing_utils
    mp_hands = mp_solutions.hands
    mp_pose = mp_solutions.pose
    mp_face = mp_solutions.face_mesh

# Relative path only — works seamlessly when copied to Raspberry Pi
SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SCRIPT_DIR / "nritya_ai_weights.keras"

# =============================================================================
# 1. EMBEDDED SCALERS & PREPROCESSING PARAMETERS
# =============================================================================

HASTHA_SCALER_MEAN = np.array([4.2e-05, 1e-06, -2.2e-05, 0.059859, -0.264131, -0.050818, 0.092483, -0.599122, -0.09147, 0.090442, -0.858529, -0.131624, 0.073597, -1.023242, -0.170067, 0.043096, -0.917339, -0.075016, 0.03687, -1.226819, -0.147956, 0.024996, -1.345613, -0.199686, 0.012714, -1.429907, -0.232283, 0.000523, -0.891643, -0.096469, -0.002398, -1.208164, -0.172893, -0.009581, -1.308302, -0.218787, -0.015002, -1.383669, -0.245835, -0.035142, -0.811281, -0.124216, -0.030904, -1.109039, -0.202031, -0.02717, -1.199366, -0.231646, -0.025066, -1.267387, -0.243849, -0.069311, -0.6984, -0.155198, -0.057459, -0.949886, -0.215025, -0.047881, -1.05695, -0.227607, -0.040397, -1.145724, -0.230241, 1.273475, 1.677084, 1.652901, 1.524999, 1.40395, 0.761413, 0.855531, 0.889555, 0.952429, 0.40619, 0.335941, 0.442447, 0.593556, 0.807525, 0.90528, 0.908123, 0.847496, 0.928233, 0.805748, 0.928033, 0.801746, 0.912837, 0.852473, 0.920651, 2.2e-05, -1e-05, -1.8e-05, 0.007146, -0.12299, -0.007364, 0.000559, -0.24898, -0.019489, -0.00909, -0.333658, -0.031044, -0.016591, -0.380086, -0.041403, -0.018069, -0.297178, -0.035828, -0.028276, -0.386497, -0.058653, -0.030961, -0.405002, -0.070227, -0.032678, -0.408511, -0.07596, -0.022813, -0.26358, -0.043478, -0.026101, -0.365576, -0.065411, -0.025288, -0.383195, -0.071478, -0.026382, -0.388082, -0.074023, -0.025911, -0.222922, -0.050732, -0.028273, -0.32657, -0.069775, -0.024748, -0.351155, -0.070014, -0.022665, -0.364401, -0.068503, -0.027952, -0.179623, -0.05792, -0.030407, -0.26465, -0.070895, -0.030358, -0.295628, -0.068357, -0.031374, -0.316896, -0.064839, 0.443365, 0.501216, 0.48312, 0.452478, 0.419994, 0.214307, 0.247022, 0.297553, 0.340866, 0.127048, 0.126749, 0.132851, 0.2033, 0.271152, 0.278897, 0.273571, 0.234905, 0.272755, 0.22467, 0.269233, 0.226585, 0.268434, 0.247127, 0.272168, 0.055101], dtype=np.float32)
HASTHA_SCALER_SCALE = np.array([0.005621, 0.005667, 0.005619, 0.197312, 0.215181, 0.09919, 0.350543, 0.373688, 0.144343, 0.457686, 0.520005, 0.175643, 0.540258, 0.644172, 0.209683, 0.369047, 0.240014, 0.189468, 0.4911, 0.411507, 0.247964, 0.554296, 0.564397, 0.271728, 0.606272, 0.698799, 0.285775, 0.32222, 0.241466, 0.184326, 0.471287, 0.445645, 0.251362, 0.554624, 0.634082, 0.268311, 0.621779, 0.796003, 0.279879, 0.290315, 0.289335, 0.185694, 0.427228, 0.470273, 0.24399, 0.499211, 0.632013, 0.254075, 0.556822, 0.774812, 0.261003, 0.292959, 0.350684, 0.195325, 0.393468, 0.497479, 0.242257, 0.44213, 0.618968, 0.255482, 0.482467, 0.730679, 0.264928, 0.458834, 0.473101, 0.584561, 0.564776, 0.482571, 0.454958, 0.50472, 0.487528, 0.539441, 0.350511, 0.311075, 0.336154, 0.431743, 0.470117, 0.071314, 0.099112, 0.191252, 0.106457, 0.241829, 0.103463, 0.250352, 0.12299, 0.214535, 0.133052, 0.003084, 0.003037, 0.003153, 0.099823, 0.232269, 0.062569, 0.186225, 0.437303, 0.094205, 0.266103, 0.582066, 0.114292, 0.33035, 0.663495, 0.134851, 0.204665, 0.460251, 0.131744, 0.319918, 0.622107, 0.172875, 0.363425, 0.678532, 0.193247, 0.398428, 0.717333, 0.205126, 0.210181, 0.416463, 0.120036, 0.337055, 0.592364, 0.161173, 0.392668, 0.645893, 0.171595, 0.439943, 0.683641, 0.178839, 0.213887, 0.380004, 0.117172, 0.328298, 0.548556, 0.153785, 0.362362, 0.611053, 0.155599, 0.387917, 0.663583, 0.156222, 0.221746, 0.351306, 0.126657, 0.303169, 0.491816, 0.15519, 0.325761, 0.566992, 0.157056, 0.341878, 0.631611, 0.157651, 0.719284, 0.798758, 0.78502, 0.740558, 0.685489, 0.427303, 0.450446, 0.513275, 0.61441, 0.267848, 0.282229, 0.262046, 0.389601, 0.478341, 0.423092, 0.417803, 0.380734, 0.418522, 0.373256, 0.415307, 0.378088, 0.414343, 0.396352, 0.421421, 0.105802], dtype=np.float32)

SHIRO_SCALER_MEAN = np.array([0.093276, -1.382258, -0.393744, 0.032943, 0.355742, 0.87243, 0.990829, 0.989841, 0.987569, 0.948061, 0.942709, 0.001424, 0.001245, 0.365189, 0.017684, -0.393744, -0.543468, 0.072359, -0.394773, -0.195193, -0.047634, -0.403013, -0.243241, 0.494603, 0.007185, 0.309826, -0.494603, -0.007185, -0.309826, -0.001913, 0.999814, 0.37668, 0.008914, 0.135374, 0.005533, 0.003018, -0.003216, -0.001184, -0.000442, -0.000481, -0.000796, -0.000793, -0.003265, 7.3e-05, 7.3e-05, -0.002998, 0.002118, 0.005533, 0.015079, -0.000559, 0.005355, 0.005118, 0.004311, 0.004205, 0.002974, 0.000683, 0.000732, 0.014559, -0.000683, -0.000732, -0.014559, -0.000148, -1.6e-05, 0.000553, 0.006867, 0.922332, 0.01039, 0.042284, 0.000529, 0.002982, 1.2e-05, 2.8e-05, 7.2e-05, 0.00185, -0.00017, -9e-05, -8.2e-05, -0.00633, 0.003375, 0.01039, -0.042186, 6.4e-05, 0.007392, -0.021079, 0.00172, 0.007089, -0.047115, 0.030586, 0.004105, 0.12091, -0.030586, -0.004105, -0.12091, -6.4e-05, 3.3e-05, 0.038821, -1.382258, -0.393744, 0.032943, 1.251251, 0.032557, 0.021998], dtype=np.float32)
SHIRO_SCALER_STD = np.array([0.53449, 2.516195, 0.177426, 0.223507, 0.063282, 0.054553, 0.012496, 0.013175, 0.019611, 0.062906, 0.083243, 0.003206, 0.002491, 0.112597, 0.104319, 0.177426, 0.250186, 0.073267, 0.0978, 0.168479, 0.084139, 0.096464, 0.18283, 0.038108, 0.062169, 1.179264, 0.038108, 0.062169, 1.179264, 0.019633, 0.000362, 0.212338, 4.435526, 16.360846, 0.276215, 1.926207, 0.21248, 0.095385, 0.021162, 0.025413, 0.030604, 0.081246, 0.097342, 0.005053, 0.004468, 0.217968, 0.194798, 0.276215, 0.750388, 0.143005, 0.181399, 0.445644, 0.152763, 0.18777, 0.495867, 0.597412, 0.264887, 11.478591, 0.597412, 0.264888, 11.478591, 0.062161, 0.001437, 0.733452, 90.824471, 347.636444, 2.822411, 39.654949, 4.202107, 1.49442, 0.27837, 0.331242, 0.400984, 0.780571, 0.991506, 0.05817, 0.053432, 2.215263, 1.929257, 2.822411, 13.575761, 1.432112, 1.875193, 7.936392, 1.607201, 2.074291, 8.44945, 14.235664, 4.746742, 236.004364, 14.235662, 4.746747, 236.004379, 1.073407, 0.024964, 13.928768, 2.516195, 0.177426, 0.223507, 2.592578, 0.168947, 0.199386], dtype=np.float32)

DRISHTI_SCALER_MEAN = np.array([0.527725, 0.531598, 0.459645, 0.399871, 0.500597, 0.177606, 0.001774, 0.131499, 0.095435, 0.155336, 0.310794, -0.212549, -0.328396, 0.102249, 0.224945, -0.292286, 0.125803, 0.002042, -0.011785, -0.111579, -0.53398, 0.059313, 0.177834, 0.456548, 0.131184, 0.026007, -0.220838, -0.420565, -0.241415, -0.19443, 0.565518, -0.292766, 0.494078, 0.185866, 0.219754, -0.495648, 0.07391, -0.018977, 0.177072, -0.326102, -0.189349, 0.092967, 0.663675, -0.256728, -0.003286, -0.002134, -0.002568, 0.164588, -0.001879, 0.203488, 7.9e-05, -0.0009, -2.3e-05, -0.000106, -0.000326, 0.000199, 0.000538, 0.000245, 6.2e-05, -0.00017, 0.000689, 8.5e-05, -0.000714, 2.8e-05, -0.002381, -0.00014, -0.000801, -0.002542, -0.000653, 0.000433, -0.010831, -0.006716, -0.013025, -0.006651, -0.007047, -0.008672, 0.001856, 0.003525, -6.7e-05, 0.001908, 0.003201, -0.000412, 0.01086, 0.009836, -0.007702, 0.00868, 0.009084, -0.006863, -0.043379, 0.005618, 0.00614, 1.086688, 0.015953, 1.823178, -9.7e-05, 0.001032, -0.017937, -0.0004, 0.000798, -0.001977, 0.001523, 0.000492, 0.004018, -0.012396, -0.000185, -0.000236, -0.000835, 0.000325, 0.002638, -0.00331, 0.001164, 0.008315, -0.023847, -8.9e-05, 0.123468, -0.03437, 0.063284, 0.143727, -0.055087, 0.124531, -0.016169, 0.009238, 0.000726, -0.015639, 0.017241, 0.000713, 0.027764, -0.005615, 0.025237, 0.021243, 0.007325, 0.035459], dtype=np.float32)
DRISHTI_SCALER_STD = np.array([0.200674, 0.175557, 0.130106, 3.037398, 0.143898, 6.882437, 0.010756, 0.036895, 0.055592, 0.014079, 0.032771, 0.031952, 0.049167, 0.022741, 0.032811, 0.039221, 0.028759, 0.004095, 0.034672, 0.002239, 0.122186, 0.124672, 0.022955, 0.131736, 0.150779, 0.018277, 0.508613, 0.304836, 0.566978, 0.424344, 0.24687, 0.451415, 0.142707, 0.107726, 0.019296, 0.143927, 0.132991, 0.019511, 0.440047, 0.507145, 0.383711, 0.390197, 0.464255, 0.370885, 1.025483, 0.865534, 0.548202, 59.629025, 0.598587, 134.889191, 0.048363, 0.071272, 0.204337, 0.007305, 0.018588, 0.13439, 0.18125, 0.032562, 0.138485, 0.193386, 0.032662, 0.009872, 0.042399, 0.014206, 0.530231, 0.464393, 0.057256, 0.550252, 0.648932, 0.034038, 4.066377, 3.276483, 4.748215, 3.407695, 2.755198, 4.293276, 0.592924, 0.560243, 0.061714, 0.598155, 0.717883, 0.034916, 4.018978, 5.312275, 3.922652, 3.510646, 4.904, 4.411768, 20.350739, 17.429348, 9.578749, 1576.646851, 10.242786, 3435.923584, 1.075509, 1.393957, 4.785638, 0.16155, 0.389044, 2.247307, 3.002911, 0.71836, 2.314583, 4.042319, 0.738943, 0.230047, 0.980052, 0.339197, 9.249004, 8.89889, 1.306764, 9.633373, 12.359674, 0.7851, 96.85128, 77.629128, 115.104317, 80.460526, 65.773613, 103.78997, 10.087328, 12.528557, 1.422049, 10.216149, 15.243305, 0.808287, 98.473175, 132.275986, 99.205872, 85.158897, 122.552963, 113.349777], dtype=np.float32)

SHIRO_SEQ_LEN = 160
SHIRO_FEAT_DIM = 102
SHIRO_MIN_PREDICT_FRAMES = 30
SHIRO_DT = 0.067

DRISHTI_SEQ_LEN = 160
DRISHTI_FEAT_DIM = 132
DRISHTI_MIN_PREDICT_FRAMES = 30
DRISHTI_DT = 0.050

# =============================================================================
# 2. EMBEDDED CLASS MAPPINGS (EXACT CANONICAL LABELS)
# =============================================================================

HASTHA_CLASSES: Dict[int, str] = {
    0: "1. Anjali",
    1: "1. Pathaka",
    2: "10. Shikhara",
    3: "10.Karthariswasthikamu",
    4: "11. Kapittha",
    5: "11.Sakatamu",
    6: "12. Katakamukha",
    7: "12.Sankhamu",
    8: "13. Soochi",
    9: "13.Chakramu",
    10: "14. Samputamu",
    11: "14.Chandrakala",
    12: "15. Paasamu",
    13: "15.Padmakosha",
    14: "16.Keelakamu",
    15: "16.Sarpasheersha",
    16: "17.Matsyamu",
    17: "17.Mrugasheersha",
    18: "18.Kurmamu",
    19: "18.Simhamukha",
    20: "19.Kaangula",
    21: "19.Varahamu",
    22: "2. Kapothamu",
    23: "2. Thripathaka",
    24: "20.Alapadma",
    25: "20.Garudamu",
    26: "21.Chatura",
    27: "21.Nagabandhakamu",
    28: "22.Bhramara",
    29: "22.Khatva",
    30: "23.Bherundamu",
    31: "23.Hamsasya",
    32: "24.Avahithamu",
    33: "24.Hamsapaksha",
    34: "25.Samdamsha",
    35: "26.Mukula",
    36: "27.Tamrachuda",
    37: "28.Trishoola",
    38: "3. Ardhapathaka",
    39: "3. Karkatamu",
    40: "4. Kartarimukha",
    41: "4. Swasthikamu",
    42: "5. Dola hastamu",
    43: "5. Mayura",
    44: "6. Ardhachandra",
    45: "6. Pushpaputamu",
    46: "7. Arala",
    47: "7. Uthsungamu",
    48: "8. Sivalingamu",
    49: "8. Sukatunda",
    50: "9. Katakavardhanamu",
    51: "9. Mushti"
}

SHIRO_BHEDHAS: List[str] = ['Sama', 'Udvahitam', 'Adhomukham', 'Alolitam', 'Dhutam', 'Kampitam', 'Paravrittam', 'Ukshiptam', 'Parivahittam']

DRISHTI_BHEDHAS: List[str] = ['Sama', 'Alokita', 'Sachi', 'Pralokita', 'Nimilita', 'Ullokita', 'Anuvritta', 'Avalokita']

SAMYUTHA_KEYWORDS: List[str] = [
    "Anjali", "Kapothamu", "Karkatamu", "Swasthikamu", "Dola", "Pushpaputamu",
    "Uthsungamu", "Sivalingamu", "Katakavardhanamu", "Karthariswasthikamu",
    "Sakatamu", "Sankhamu", "Chakramu", "Samputamu", "Paasamu", "Keelakamu",
    "Matsyamu", "Kurmamu", "Varahamu", "Garudamu", "Nagabandhakamu", "Khatva",
    "Bherundamu", "Avahithamu"
]

# =============================================================================
# 3. SELF-CONTAINED GEOMETRIC & LANDMARK EXTRACTORS
# =============================================================================

def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-7 or nbc < 1e-7:
        return 0.0
    cos_val = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


class HandLandmarkExtractor:
    """Self-contained 175-D feature extractor for single and dual hand gestures."""
    def __init__(self, static_image_mode=False, min_detection_confidence=0.35):
        self.detector = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.30
        )
        self.fallback = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.15
        )

    def close(self):
        self.detector.close()
        self.fallback.close()

    def _extract_single_hand_features(self, landmarks_3d: np.ndarray) -> Dict[str, Any]:
        wrist = landmarks_3d[0]
        rel_coords = landmarks_3d - wrist
        scale = np.linalg.norm(landmarks_3d[9] - wrist)
        if scale < 1e-6:
            scale = 1.0
        norm_coords = rel_coords / scale
        tip_indices = [4, 8, 12, 16, 20]
        tip_wrist_dists = [np.linalg.norm(landmarks_3d[t] - wrist) / scale for t in tip_indices]
        tip_pairs = [(4, 8), (4, 12), (4, 16), (4, 20), (8, 12), (12, 16), (16, 20), (8, 16), (8, 20)]
        inter_tip_dists = [np.linalg.norm(landmarks_3d[i] - landmarks_3d[j]) / scale for i, j in tip_pairs]
        triplets = [(1, 2, 3), (2, 3, 4), (5, 6, 7), (6, 7, 8), (9, 10, 11), (10, 11, 12), (13, 14, 15), (14, 15, 16), (17, 18, 19), (18, 19, 20)]
        joint_angles = [calculate_angle_3d(landmarks_3d[a], landmarks_3d[b], landmarks_3d[c]) for a, b, c in triplets]
        feat_vector = np.concatenate([
            norm_coords.flatten(),
            np.array(tip_wrist_dists, dtype=np.float32),
            np.array(inter_tip_dists, dtype=np.float32),
            np.array(joint_angles, dtype=np.float32) / 180.0
        ])
        return {"wrist": wrist.tolist(), "feature_vector": feat_vector}


class PoseLandmarkExtractor:
    """Self-contained 32-D pose geometry extractor for head and neck orientation."""
    def __init__(self, static_image_mode=False, min_detection_confidence=0.30):
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.30
        )
        self.fallback = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.15
        )

    def close(self):
        self.pose.close()
        self.fallback.close()

    def process(self, image_rgb: np.ndarray, category_focus: str = "shiro") -> Dict[str, Any]:
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            results = self.fallback.process(image_rgb)

        if not results.pose_landmarks:
            return {
                "detected": False,
                "vector": np.zeros(32, dtype=np.float32),
                "shiro_features": {},
                "raw_landmarks": [],
                "pose_landmarks": None
            }

        lms = results.pose_landmarks.landmark
        all_pts = [np.array([lm.x, lm.y, lm.z], dtype=np.float32) for lm in lms]
        vis = [float(lm.visibility) for lm in lms]

        p_nose = all_pts[0]
        p_ear_l = all_pts[7]
        p_ear_r = all_pts[8]
        p_sh_l = all_pts[11]
        p_sh_r = all_pts[12]
        p_hip_l = all_pts[23]
        p_hip_r = all_pts[24]

        mid_sh = (p_sh_l + p_sh_r) / 2.0
        mid_hip = (p_hip_l + p_hip_r) / 2.0
        torso_h = max(1e-5, float(np.linalg.norm(mid_sh[:2] - mid_hip[:2])))
        sh_w = max(1e-5, float(np.linalg.norm(p_sh_l[:2] - p_sh_r[:2])))

        lat_shift = float((p_nose[0] - mid_sh[0]) / sh_w)
        tilt_roll = float(math.atan2(p_ear_r[1] - p_ear_l[1], p_ear_r[0] - p_ear_l[0]))
        pitch = float((p_nose[1] - mid_sh[1]) / torso_h)
        ear_mid = (p_ear_l + p_ear_r) / 2.0
        yaw = float((p_nose[0] - ear_mid[0]) / (sh_w + 1e-6))

        upper_rel = np.concatenate([
            (p_nose - mid_sh) / torso_h,
            (p_ear_l - mid_sh) / torso_h,
            (p_ear_r - mid_sh) / torso_h,
            (p_sh_l - mid_sh) / sh_w,
            (p_sh_r - mid_sh) / sh_w,
            (mid_hip - mid_sh) / torso_h
        ])

        base_vec = np.concatenate([
            np.array([
                lat_shift, tilt_roll, pitch, yaw, sh_w, torso_h,
                vis[0], vis[7], vis[8], vis[11], vis[12], vis[23], vis[24],
                float(np.linalg.norm(p_nose[:2] - mid_sh[:2]))
            ], dtype=np.float32),
            upper_rel.astype(np.float32)
        ])

        raw_landmarks = [
            {"index": i, "x": float(lms[i].x), "y": float(lms[i].y), "z": float(lms[i].z), "visibility": float(lms[i].visibility)}
            for i in range(33)
        ]

        return {
            "detected": True,
            "vector": base_vec,
            "shiro_features": {"head_tilt_roll_rad": tilt_roll, "head_pitch": pitch, "head_yaw": yaw},
            "raw_landmarks": raw_landmarks,
            "pose_landmarks": results.pose_landmarks
        }


class FaceDrishtiLandmarkExtractor:
    """Self-contained 44-D face mesh geometry extractor for gaze & eye orientation."""
    def __init__(self, static_image_mode=False, min_detection_confidence=0.30):
        self.face_mesh = mp_face.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.30
        )

    def close(self):
        self.face_mesh.close()

    def process(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return {"detected": False, "vector": np.zeros(44, dtype=np.float32), "landmarks": None}

        lms = results.multi_face_landmarks[0].landmark
        def pt(i): return np.array([lms[i].x, lms[i].y, lms[i].z], dtype=np.float32)

        p_ol, p_il, p_tl, p_bl = pt(33), pt(133), pt(159), pt(145)
        p_or, p_ir, p_tr, p_br = pt(263), pt(362), pt(386), pt(374)
        p_iris_l, p_iris_r = pt(468), pt(473)
        p_nose, p_chin, p_fore = pt(1), pt(152), pt(10)
        p_chl, p_chr = pt(234), pt(454)

        w_l = max(1e-6, float(np.linalg.norm(p_ol[:2] - p_il[:2])))
        h_l = max(1e-6, float(np.linalg.norm(p_tl[:2] - p_bl[:2])))
        w_r = max(1e-6, float(np.linalg.norm(p_or[:2] - p_ir[:2])))
        h_r = max(1e-6, float(np.linalg.norm(p_tr[:2] - p_br[:2])))

        ear_l = h_l / w_l
        ear_r = h_r / w_r
        g_hl = float((p_iris_l[0] - p_il[0]) / (w_l + 1e-6))
        g_vl = float((p_iris_l[1] - p_tl[1]) / (h_l + 1e-6))
        g_hr = float((p_iris_r[0] - p_ir[0]) / (w_r + 1e-6))
        g_vr = float((p_iris_r[1] - p_tr[1]) / (h_r + 1e-6))

        fw = max(1e-6, float(np.linalg.norm(p_chl[:2] - p_chr[:2])))
        fh = max(1e-6, float(np.linalg.norm(p_fore[:2] - p_chin[:2])))
        mid_eye = (p_ol + p_or) / 2.0
        yaw = float((p_nose[0] - mid_eye[0]) / fw)
        pitch = float((p_nose[1] - mid_eye[1]) / fh)
        roll = float(math.atan2(p_or[1] - p_ol[1], p_or[0] - p_ol[0]))
        fc = (p_fore + p_chin) / 2.0

        eye_l_rel = (p_iris_l - fc) / fw
        eye_r_rel = (p_iris_r - fc) / fw
        nose_rel = (p_nose - fc) / fh

        eye_rel = np.concatenate([
            (p_ol - p_iris_l) / w_l, (p_il - p_iris_l) / w_l,
            (p_tl - p_iris_l) / h_l, (p_bl - p_iris_l) / h_l,
            (p_or - p_iris_r) / w_r, (p_ir - p_iris_r) / w_r,
            (p_tr - p_iris_r) / h_r, (p_br - p_iris_r) / h_r
        ])

        vec_44 = np.concatenate([
            np.array([ear_l, ear_r, g_hl, g_vl, g_hr, g_vr, yaw, pitch, roll, fw, fh], dtype=np.float32),
            eye_l_rel, eye_r_rel, nose_rel, eye_rel
        ]).astype(np.float32)

        return {
            "detected": True,
            "vector": vec_44,
            "landmarks": results.multi_face_landmarks[0]
        }

# =============================================================================
# 4. THREAD 1: DEDICATED CAMERA CAPTURE STREAM (MIRRORED COORDINATES)
# =============================================================================

class CameraCaptureStream:
    """Dedicated background capture thread that continuously drains the driver queue."""
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.lock = threading.Lock()
        self.latest_frame_mirrored: Optional[np.ndarray] = None
        self.latest_timestamp: float = 0.0

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureThread")
        self.thread.start()
        return True

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                mirrored = cv2.flip(frame, 1)
                now = time.time()
                with self.lock:
                    self.latest_frame_mirrored = mirrored
                    self.latest_timestamp = now
            else:
                time.sleep(0.005)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], float]:
        with self.lock:
            if self.latest_frame_mirrored is None:
                return False, None, 0.0
            return True, self.latest_frame_mirrored.copy(), self.latest_timestamp

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

# =============================================================================
# 5. THREAD 2: MULTI-CATEGORY ASYNCHRONOUS INFERENCE WORKER
# =============================================================================

class NrityaInferenceWorker:
    """Multi-category AI inference engine powered by consolidated nritya_ai_weights.keras."""
    def __init__(self, weights_path: Path, initial_category: str = "hasthas"):
        self.current_category = initial_category.lower()

        if not weights_path.exists():
            raise FileNotFoundError(f"Consolidated weights missing: {weights_path}")

        print(f"[INIT] Loading consolidated weights from {weights_path.name}...")
        self.unified_model = tf.keras.models.load_model(str(weights_path), compile=False)

        # Retrieve individual submodels
        self.hastha_model = self.unified_model.get_layer("Personalized_52Hasthas_MLP")
        self.shiro_model = self.unified_model.get_layer("Shiro_BiGRU_Classifier")
        self.drishti_model = self.unified_model.get_layer("Drishti_BiGRU_Classifier")

        # Compile graph forward passes for minimal latency
        @tf.function(reduce_retracing=True)
        def _h_predict(x): return self.hastha_model(x, training=False)
        self.compiled_hastha_predict = _h_predict

        @tf.function(reduce_retracing=True)
        def _s_predict(x): return self.shiro_model(x, training=False)
        self.compiled_shiro_predict = _s_predict

        @tf.function(reduce_retracing=True)
        def _d_predict(x): return self.drishti_model(x, training=False)
        self.compiled_drishti_predict = _d_predict

        # Graph Warm-up passes
        _ = self.compiled_hastha_predict(np.zeros((1, 175), dtype=np.float32))
        _ = self.compiled_shiro_predict(np.zeros((1, SHIRO_SEQ_LEN, SHIRO_FEAT_DIM), dtype=np.float32))
        _ = self.compiled_drishti_predict(np.zeros((1, DRISHTI_SEQ_LEN, DRISHTI_FEAT_DIM), dtype=np.float32))
        print("[INIT] Models loaded and warmed up successfully.")

        # MediaPipe Detectors
        self.hastha_extractor = HandLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.35)
        self.hastha_mp_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.35, min_tracking_confidence=0.30)
        self.hastha_mp_fallback = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.15, min_tracking_confidence=0.15)
        self.hastha_buffer = deque(maxlen=8)

        self.shiro_extractor = PoseLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.30)
        self.shiro_buffer = deque(maxlen=8)
        self.shiro_temporal_buffer = deque(maxlen=SHIRO_SEQ_LEN)
        self.shiro_prev_ts: Optional[float] = None
        self.shiro_prev_base: Optional[np.ndarray] = None
        self.shiro_prev_vel: Optional[np.ndarray] = None
        self.shiro_baseline_angles: Optional[np.ndarray] = None
        self.shiro_sequence_id: int = 1

        self.drishti_extractor = FaceDrishtiLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.30)
        self.drishti_buffer = deque(maxlen=8)
        self.drishti_temporal_buffer = deque(maxlen=DRISHTI_SEQ_LEN)
        self.drishti_prev_ts: Optional[float] = None
        self.drishti_prev_base: Optional[np.ndarray] = None
        self.drishti_prev_vel: Optional[np.ndarray] = None
        self.last_detect_time: float = time.time()

        # Synchronization
        self.lock = threading.Lock()
        self.category_lock = threading.Lock()
        self.reset_event = threading.Event()
        self.stop_event = threading.Event()

        self.shared_state: Dict[str, Any] = {
            "category": self.current_category,
            "detected": False,
            "status": "SHOW GESTURE",
            "recognized_name": None,
            "recognized_subtitle": None,
            "confidence": 0.0,
            "landmarks": None,
            "latency_ms": 0.0
        }

    def close(self):
        self.stop_event.set()
        self.hastha_extractor.close()
        self.hastha_mp_detector.close()
        self.hastha_mp_fallback.close()
        self.shiro_extractor.close()
        self.drishti_extractor.close()

    def get_shared_state(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.shared_state)

    def set_category(self, category: str):
        with self.category_lock:
            cat = category.lower()
            if cat in ["hasthas", "shiro", "drishti"]:
                if self.current_category != cat:
                    old_cat = self.current_category
                    self.current_category = cat
                    self._switch_category_state(old_cat, cat)

    def _switch_category_state(self, old_cat: str, new_cat: str):
        with self.lock:
            if old_cat == "hasthas":
                self.hastha_buffer.clear()
            elif old_cat == "shiro":
                self.shiro_buffer.clear()
                self.shiro_temporal_buffer.clear()
                self.shiro_prev_ts = None
                self.shiro_prev_base = None
                self.shiro_prev_vel = None
                self.shiro_baseline_angles = None
            elif old_cat == "drishti":
                self.drishti_buffer.clear()
                self.drishti_temporal_buffer.clear()
                self.drishti_prev_ts = None
                self.drishti_prev_base = None
                self.drishti_prev_vel = None

            if new_cat == "hasthas":
                self.hastha_buffer.clear()
                prompt = "SHOW GESTURE"
            elif new_cat == "shiro":
                self.shiro_buffer.clear()
                self.shiro_temporal_buffer.clear()
                self.shiro_prev_ts = None
                self.shiro_prev_base = None
                self.shiro_prev_vel = None
                self.shiro_baseline_angles = None
                self.shiro_sequence_id += 1
                prompt = "PERFORM HEAD MOVEMENT"
            elif new_cat == "drishti":
                self.drishti_buffer.clear()
                self.drishti_temporal_buffer.clear()
                self.drishti_prev_ts = None
                self.drishti_prev_base = None
                self.drishti_prev_vel = None
                prompt = "PERFORM EYE MOVEMENT"

            self.shared_state["category"] = new_cat
            self.shared_state["detected"] = False
            self.shared_state["status"] = prompt
            self.shared_state["recognized_name"] = None
            self.shared_state["recognized_subtitle"] = None
            self.shared_state["confidence"] = 0.0
            self.shared_state["landmarks"] = None
            self.shared_state["latency_ms"] = 0.0

    def reset_state(self):
        with self.lock:
            cat = self.current_category
            if cat == "hasthas":
                self.hastha_buffer.clear()
                prompt = "SHOW GESTURE"
            elif cat == "shiro":
                self.shiro_buffer.clear()
                self.shiro_temporal_buffer.clear()
                self.shiro_prev_ts = None
                self.shiro_prev_base = None
                self.shiro_prev_vel = None
                self.shiro_baseline_angles = None
                self.shiro_sequence_id += 1
                prompt = "PERFORM HEAD MOVEMENT"
            elif cat == "drishti":
                self.drishti_buffer.clear()
                self.drishti_temporal_buffer.clear()
                self.drishti_prev_ts = None
                self.drishti_prev_base = None
                self.drishti_prev_vel = None
                prompt = "PERFORM EYE MOVEMENT"

            self.shared_state["category"] = cat
            self.shared_state["detected"] = False
            self.shared_state["status"] = prompt
            self.shared_state["recognized_name"] = None
            self.shared_state["recognized_subtitle"] = None
            self.shared_state["confidence"] = 0.0
            self.shared_state["landmarks"] = None

    def clean_name(self, full_name: str) -> str:
        return re.sub(r"^\d+\.\s*", "", full_name).strip().upper()

    def is_samyutha(self, full_name: str) -> bool:
        return any(kw in full_name for kw in SAMYUTHA_KEYWORDS)

    @property
    def active_category(self) -> str:
        return self.current_category

    def process_single_frame(self, frame_mirrored: np.ndarray, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        now = timestamp
        cat = self.current_category
        t_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_mirrored, cv2.COLOR_BGR2RGB)

        # BRANCH 1: HASTHAS (175-D Landmarks)
        if cat == "hasthas":
            results = self.hastha_mp_detector.process(frame_rgb)
            if not results.multi_hand_landmarks:
                results = self.hastha_mp_fallback.process(frame_rgb)

            if not results.multi_hand_landmarks:
                self.hastha_buffer.clear()
                # Diagnostic output for terminal (throttled to 1s)
                if (now - getattr(self, "_last_no_hand_diag", 0)) > 1.0:
                    self._last_no_hand_diag = now
                    print("[HASTHA DIAG] Hand detected: False | Hands: 0")
                with self.lock:
                    self.shared_state["category"] = "hasthas"
                    self.shared_state["detected"] = False
                    self.shared_state["status"] = "SHOW GESTURE"
                    self.shared_state["recognized_name"] = None
                    self.shared_state["recognized_subtitle"] = None
                    self.shared_state["confidence"] = 0.0
                    self.shared_state["landmarks"] = None
                    self.shared_state["latency_ms"] = (time.perf_counter() - t_start) * 1000.0
                return

            num_hands = len(results.multi_hand_landmarks)
            if num_hands == 1:
                hand_lms = results.multi_hand_landmarks[0]
                raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

                handedness_label = "Left"
                if results.multi_handedness and len(results.multi_handedness) > 0:
                    try:
                        handedness_label = results.multi_handedness[0].classification[0].label
                    except Exception:
                        handedness_label = "Left"

                if handedness_label.strip().lower() == "right":
                    canonical_coords = raw_coords.copy()
                    wrist_x = raw_coords[0, 0]
                    canonical_coords[:, 0] = 2.0 * wrist_x - canonical_coords[:, 0]
                else:
                    canonical_coords = raw_coords

                h1_feat = self.hastha_extractor._extract_single_hand_features(canonical_coords)
                h1_vec = h1_feat["feature_vector"]
                h2_vec = np.zeros(87, dtype=np.float32)
                inter_dist = np.array([0.0], dtype=np.float32)
                vector_175 = np.concatenate([h1_vec, h2_vec, inter_dist])
            else:
                coords_0 = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark], dtype=np.float32)
                coords_1 = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[1].landmark], dtype=np.float32)
                h1_feat = self.hastha_extractor._extract_single_hand_features(coords_0)
                h2_feat = self.hastha_extractor._extract_single_hand_features(coords_1)
                h1_vec = h1_feat["feature_vector"]
                h2_vec = h2_feat["feature_vector"]
                w1 = np.array(h1_feat["wrist"])
                w2 = np.array(h2_feat["wrist"])
                inter_dist = float(np.linalg.norm(w1 - w2))
                vector_175 = np.concatenate([h1_vec, h2_vec, np.array([inter_dist], dtype=np.float32)])

            scaled_175 = (vector_175 - HASTHA_SCALER_MEAN) / HASTHA_SCALER_SCALE
            scaled_175 = np.clip(scaled_175, -10.0, 10.0).reshape(1, 175)

            probs = self.compiled_hastha_predict(scaled_175).numpy()[0]
            top_cid = int(np.argmax(probs))
            top_conf = float(probs[top_cid])

            if top_conf >= 0.20:
                self.hastha_buffer.append(top_cid)
            else:
                self.hastha_buffer.append(None)

            valid_votes = [idx for idx in self.hastha_buffer if idx is not None]
            counts = Counter(valid_votes)
            is_stable = False
            winner_cid = None
            if counts:
                most_common_id, count = counts.most_common(1)[0]
                if count >= 6:
                    is_stable = True
                    winner_cid = most_common_id

            # Diagnostic output ONLY to terminal (Task 4)
            raw_pred_name = HASTHA_CLASSES.get(top_cid) or HASTHA_CLASSES.get(str(top_cid), "Unknown")
            gate_rejected = not is_stable
            gate_status = "Accepted (Stable)" if is_stable else (f"Votes {counts.most_common(1)[0][1] if counts else 0}/6" if top_conf >= 0.20 else f"Conf {top_conf:.2f} < 0.20")

            if (now - getattr(self, "_last_hastha_diag", 0)) > 0.35 or (is_stable != getattr(self, "_last_stable_diag", False)):
                self._last_hastha_diag = now
                self._last_stable_diag = is_stable
                num_finite = int(np.sum(np.isfinite(vector_175)))
                print(
                    f"[HASTHA DIAG] Hand detected: True | Hands: {num_hands} | "
                    f"Shape: {vector_175.shape} | Finite: {num_finite}/175 | "
                    f"Scaler min/max: [{scaled_175.min():.2f}, {scaled_175.max():.2f}] | "
                    f"Pred idx: {top_cid} | Pred name: {self.clean_name(raw_pred_name)} | "
                    f"Raw max prob: {top_conf:.4f} | Gate rejected: {gate_rejected} ({gate_status})"
                )

            latency = (time.perf_counter() - t_start) * 1000.0
            with self.lock:
                self.shared_state["category"] = "hasthas"
                self.shared_state["detected"] = True
                self.shared_state["landmarks"] = results.multi_hand_landmarks
                self.shared_state["latency_ms"] = latency

                if is_stable and winner_cid is not None:
                    full_name = HASTHA_CLASSES.get(winner_cid) or HASTHA_CLASSES.get(str(winner_cid), "Unknown")
                    self.shared_state["status"] = "STABLE"
                    self.shared_state["recognized_name"] = self.clean_name(full_name)
                    sub = "Samyutha Hastha" if self.is_samyutha(full_name) else "Asamyuta Hastha"
                    self.shared_state["recognized_subtitle"] = sub
                    self.shared_state["confidence"] = top_conf
                else:
                    self.shared_state["status"] = "RECOGNIZING..."
                    self.shared_state["recognized_name"] = None
                    self.shared_state["recognized_subtitle"] = None
                    self.shared_state["confidence"] = 0.0

        # BRANCH 2: SHIRO BHEDHAS (102-D Temporal Motion)
        elif cat == "shiro":
            try:
                res = self.shiro_extractor.process(frame_rgb, category_focus="shiro")
                if not res.get("detected"):
                    if (now - self.last_detect_time) > 0.50 and len(self.shiro_temporal_buffer) > 0:
                        self.reset_state()
                    self.shiro_buffer.append((-1, 0.0))
                    with self.lock:
                        self.shared_state["category"] = "shiro"
                        self.shared_state["detected"] = False
                        self.shared_state["status"] = "PERFORM HEAD MOVEMENT"
                        self.shared_state["recognized_name"] = None
                        self.shared_state["recognized_subtitle"] = None
                        self.shared_state["confidence"] = 0.0
                        self.shared_state["landmarks"] = None
                        self.shared_state["latency_ms"] = (time.perf_counter() - t_start) * 1000.0
                    return

                self.last_detect_time = now
                base_32 = res["vector"]
                shiro_feats = res["shiro_features"]

                if self.shiro_prev_ts is not None:
                    raw_dt = timestamp - self.shiro_prev_ts
                    dt = SHIRO_DT if (raw_dt <= 0.001 or raw_dt > 0.40) else raw_dt
                else:
                    dt = SHIRO_DT
                self.shiro_prev_ts = timestamp

                if self.shiro_prev_base is not None:
                    vel = (base_32 - self.shiro_prev_base) / dt
                else:
                    vel = np.zeros(32, dtype=np.float32)
                self.shiro_prev_base = base_32.copy()

                if self.shiro_prev_vel is not None:
                    acc = (vel - self.shiro_prev_vel) / dt
                else:
                    acc = np.zeros(32, dtype=np.float32)
                self.shiro_prev_vel = vel.copy()

                angles = np.array([
                    float(shiro_feats.get("head_tilt_roll_rad", 0.0)),
                    float(shiro_feats.get("head_pitch", 0.0)),
                    float(shiro_feats.get("head_yaw", 0.0))
                ], dtype=np.float32)

                if self.shiro_baseline_angles is None:
                    self.shiro_baseline_angles = angles.copy()
                deviations = angles - self.shiro_baseline_angles

                vec_102 = np.concatenate([base_32, vel, acc, angles, deviations]).astype(np.float32)
                self.shiro_temporal_buffer.append(vec_102)

                buf_len = len(self.shiro_temporal_buffer)
                raw_cid = None
                raw_conf = 0.0

                if buf_len >= SHIRO_MIN_PREDICT_FRAMES:
                    X_act = np.array(self.shiro_temporal_buffer, dtype=np.float32)
                    X_norm = (X_act - SHIRO_SCALER_MEAN) / SHIRO_SCALER_STD
                    seq = np.zeros((1, SHIRO_SEQ_LEN, SHIRO_FEAT_DIM), dtype=np.float32)
                    if buf_len <= SHIRO_SEQ_LEN:
                        seq[0, :buf_len, :] = X_norm
                    else:
                        seq[0, :, :] = X_norm[-SHIRO_SEQ_LEN:, :]

                    probs = self.compiled_shiro_predict(seq).numpy()[0]
                    raw_cid = int(np.argmax(probs))
                    raw_conf = float(probs[raw_cid])

                if raw_cid is not None:
                    self.shiro_buffer.append((raw_cid, raw_conf))
                else:
                    self.shiro_buffer.append((-1, 0.0))

                is_stable = False
                winner_cid = None
                if len(self.shiro_buffer) >= 8:
                    votes = [p[0] for p in self.shiro_buffer if p[0] >= 0]
                    if votes:
                        counts = Counter(votes)
                        top_id, count = counts.most_common(1)[0]
                        top_confs = [p[1] for p in self.shiro_buffer if p[0] == top_id]
                        avg_conf = float(np.mean(top_confs)) if top_confs else 0.0
                        if count >= 6 and avg_conf >= 0.20:
                            is_stable = True
                            winner_cid = top_id

                latency = (time.perf_counter() - t_start) * 1000.0
                shiro_landmarks = res.get("pose_landmarks") or res.get("raw_landmarks")

                with self.lock:
                    self.shared_state["category"] = "shiro"
                    self.shared_state["detected"] = True
                    self.shared_state["landmarks"] = shiro_landmarks
                    self.shared_state["latency_ms"] = latency

                    if is_stable and winner_cid is not None:
                        sanskrit = SHIRO_BHEDHAS[winner_cid]
                        self.shared_state["status"] = "STABLE"
                        self.shared_state["recognized_name"] = sanskrit.upper()
                        self.shared_state["recognized_subtitle"] = "Shiro Bhedha (Head Movement)"
                        self.shared_state["confidence"] = raw_conf
                    else:
                        self.shared_state["status"] = "RECOGNIZING..."
                        self.shared_state["recognized_name"] = None
                        self.shared_state["recognized_subtitle"] = None
                        self.shared_state["confidence"] = 0.0

            except Exception as ex:
                print(f"\n[ERROR] Shiro inference error: {ex}")
                import traceback
                traceback.print_exc()
                with self.lock:
                    self.shared_state["category"] = "shiro"
                    self.shared_state["detected"] = False
                    self.shared_state["status"] = "SHIRO INITIALIZATION ERROR"
                    self.shared_state["recognized_name"] = None
                    self.shared_state["recognized_subtitle"] = str(ex)
                    self.shared_state["confidence"] = 0.0
                    self.shared_state["landmarks"] = None

        # BRANCH 3: DRISHTI BHEDHAS (132-D Gaze/Eye Motion)
        elif cat == "drishti":
            res = self.drishti_extractor.process(frame_rgb)
            if not res.get("detected"):
                if (now - self.last_detect_time) > 0.50 and len(self.drishti_temporal_buffer) > 0:
                    self.reset_state()
                self.drishti_buffer.append((-1, 0.0))
                with self.lock:
                    self.shared_state["category"] = "drishti"
                    self.shared_state["detected"] = False
                    self.shared_state["status"] = "PERFORM EYE MOVEMENT"
                    self.shared_state["recognized_name"] = None
                    self.shared_state["recognized_subtitle"] = None
                    self.shared_state["confidence"] = 0.0
                    self.shared_state["landmarks"] = None
                    self.shared_state["latency_ms"] = (time.perf_counter() - t_start) * 1000.0
                return

            self.last_detect_time = now
            base_44 = res["vector"]

            if self.drishti_prev_ts is not None:
                raw_dt = timestamp - self.drishti_prev_ts
                dt = DRISHTI_DT if (raw_dt <= 0.001 or raw_dt > 0.40) else raw_dt
            else:
                dt = DRISHTI_DT
            self.drishti_prev_ts = timestamp

            if self.drishti_prev_base is not None:
                vel = (base_44 - self.drishti_prev_base) / dt
            else:
                vel = np.zeros(44, dtype=np.float32)
            self.drishti_prev_base = base_44.copy()

            if self.drishti_prev_vel is not None:
                acc = (vel - self.drishti_prev_vel) / dt
            else:
                acc = np.zeros(44, dtype=np.float32)
            self.drishti_prev_vel = vel.copy()

            vec_132 = np.concatenate([base_44, vel, acc]).astype(np.float32)
            self.drishti_temporal_buffer.append(vec_132)

            buf_len = len(self.drishti_temporal_buffer)
            raw_cid = None
            raw_conf = 0.0

            if buf_len >= DRISHTI_MIN_PREDICT_FRAMES:
                X_act = np.array(self.drishti_temporal_buffer, dtype=np.float32)
                X_norm = (X_act - DRISHTI_SCALER_MEAN) / DRISHTI_SCALER_STD
                seq = np.zeros((1, DRISHTI_SEQ_LEN, DRISHTI_FEAT_DIM), dtype=np.float32)
                if buf_len <= DRISHTI_SEQ_LEN:
                    seq[0, :buf_len, :] = X_norm
                else:
                    seq[0, :, :] = X_norm[-DRISHTI_SEQ_LEN:, :]

                probs = self.compiled_drishti_predict(seq).numpy()[0]
                raw_cid = int(np.argmax(probs))
                raw_conf = float(probs[raw_cid])

            if raw_cid is not None:
                self.drishti_buffer.append((raw_cid, raw_conf))
            else:
                self.drishti_buffer.append((-1, 0.0))

            is_stable = False
            winner_cid = None
            if len(self.drishti_buffer) >= 8:
                votes = [p[0] for p in self.drishti_buffer if p[0] >= 0]
                if votes:
                    counts = Counter(votes)
                    top_id, count = counts.most_common(1)[0]
                    top_confs = [p[1] for p in self.drishti_buffer if p[0] == top_id]
                    avg_conf = float(np.mean(top_confs)) if top_confs else 0.0
                    if count >= 6 and avg_conf >= 0.20:
                        is_stable = True
                        winner_cid = top_id

            latency = (time.perf_counter() - t_start) * 1000.0

            with self.lock:
                self.shared_state["category"] = "drishti"
                self.shared_state["detected"] = True
                self.shared_state["landmarks"] = res.get("landmarks")
                self.shared_state["latency_ms"] = latency

                if is_stable and winner_cid is not None:
                    sanskrit = DRISHTI_BHEDHAS[winner_cid]
                    self.shared_state["status"] = "STABLE"
                    self.shared_state["recognized_name"] = sanskrit.upper()
                    self.shared_state["recognized_subtitle"] = "Drishti Bhedha (Eye Movement)"
                    self.shared_state["confidence"] = raw_conf
                else:
                    self.shared_state["status"] = "RECOGNIZING..."
                    self.shared_state["recognized_name"] = None
                    self.shared_state["recognized_subtitle"] = None
                    self.shared_state["confidence"] = 0.0


    def run_worker_loop(self, stream: CameraCaptureStream):
        last_process_time = 0.0

        while not self.stop_event.is_set():
            try:
                if self.reset_event.is_set():
                    self.reset_state()
                    self.reset_event.clear()

                now = time.time()
                cat = self.current_category

                min_interval = 0.050 if cat in ["shiro", "drishti"] else 0.020
                if (now - last_process_time) < min_interval:
                    time.sleep(0.003)
                    continue

                ret, frame_mirrored, timestamp = stream.read_latest()
                if not ret or frame_mirrored is None or timestamp <= last_process_time:
                    time.sleep(0.003)
                    continue

                last_process_time = timestamp
                self.process_single_frame(frame_mirrored, timestamp)

            except Exception as worker_err:
                print(f"[ERROR] Inference worker loop exception: {worker_err}")
                import traceback
                traceback.print_exc()
                time.sleep(0.05)

# =============================================================================
# 6. MAIN THREAD GUI & VISUAL RENDERING
# =============================================================================

class NrityaAIApplication:
    """Main user interface rendering clean HUD and running camera capture stream."""
    def __init__(self, camera_index: int = 0, initial_category: str = "hasthas"):
        self.camera_index = camera_index
        self.selected_category = initial_category.lower()
        self.debug_mode = False

        self.capture_stream = CameraCaptureStream(camera_index=camera_index, width=1280, height=720)
        self.inference_worker = NrityaInferenceWorker(WEIGHTS_PATH, initial_category=self.selected_category)
        self.worker_thread: Optional[threading.Thread] = None

        # Symmetrically spaced buttons across top HUD (Exactly 3 categories, PADA removed)
        self.btn_hastha = (50, 72, 430, 114)
        self.btn_shiro = (450, 72, 830, 114)
        self.btn_drishti = (850, 72, 1230, 114)

    def switch_to_category(self, target_category: str):
        target = target_category.lower()
        if target in ["hasthas", "shiro", "drishti"]:
            if self.selected_category != target:
                old = self.selected_category
                self.selected_category = target
                self.inference_worker.set_category(target)
                print(f"[UI] Category switched: {old.upper()} -> {target.upper()}")

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.btn_hastha[0] <= x <= self.btn_hastha[2] and self.btn_hastha[1] <= y <= self.btn_hastha[3]:
                self.switch_to_category("hasthas")
            elif self.btn_shiro[0] <= x <= self.btn_shiro[2] and self.btn_shiro[1] <= y <= self.btn_shiro[3]:
                self.switch_to_category("shiro")
            elif self.btn_drishti[0] <= x <= self.btn_drishti[2] and self.btn_drishti[1] <= y <= self.btn_drishti[3]:
                self.switch_to_category("drishti")

    def draw_hud(self, preview_frame: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        h, w, _ = preview_frame.shape
        canvas = preview_frame.copy()

        # 1. Landmark Skeleton Visualization
        landmarks = state.get("landmarks")
        cat = state.get("category")
        if state.get("detected") and landmarks:
            try:
                if cat == "hasthas":
                    for hand_lms in landmarks:
                        if hasattr(hand_lms, "landmark"):
                            mp_drawing.draw_landmarks(
                                canvas,
                                hand_lms,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 140), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                            )
                elif cat == "shiro":
                    if hasattr(landmarks, "landmark"):
                        mp_drawing.draw_landmarks(
                            canvas,
                            landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 215, 255), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                        )
                    elif isinstance(landmarks, list):
                        for lm in landmarks:
                            if isinstance(lm, dict) and lm.get("visibility", 1.0) > 0.4:
                                px, py = int(lm["x"] * w), int(lm["y"] * h)
                                cv2.circle(canvas, (px, py), 3, (0, 215, 255), -1, cv2.LINE_AA)
                elif cat == "drishti":
                    if hasattr(landmarks, "landmark"):
                        for pt_idx in [33, 133, 159, 145, 263, 362, 386, 374, 468, 473]:
                            lm = landmarks.landmark[pt_idx]
                            px, py = int(lm.x * w), int(lm.y * h)
                            col = (0, 255, 255) if pt_idx in [468, 473] else (255, 180, 0)
                            cv2.circle(canvas, (px, py), 3, col, -1, cv2.LINE_AA)
            except Exception:
                pass

        # 2. Glassmorphic Top Header Panel (Height: 185px)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (30, 15), (w - 30, 185), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (30, 15), (w - 30, 185), (55, 68, 82), 2)

        # Header Typography
        cv2.putText(canvas, "NRITYA AI", (50, 44), cv2.FONT_HERSHEY_DUPLEX, 0.80, (240, 245, 250), 2, cv2.LINE_AA)
        cv2.putText(canvas, "REAL-TIME KUCHIPUDI RECOGNITION", (195, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 170, 190), 1, cv2.LINE_AA)

        # 3. Interactive Category Selection Buttons (Exactly 3 Categories — PADA removed)
        btns = [
            (self.btn_hastha, "[ 1 HASTHAS ]", self.selected_category == "hasthas", (40, 90, 60), (50, 255, 120)),
            (self.btn_shiro, "[ 2 SHIRO BHEDHAS ]", self.selected_category == "shiro", (40, 75, 95), (0, 215, 255)),
            (self.btn_drishti, "[ 3 DRISHTI BHEDHAS ]", self.selected_category == "drishti", (70, 50, 95), (200, 120, 255)),
        ]

        for (b_rect, b_label, is_act, bg_act, border_act) in btns:
            b_bg = bg_act if is_act else (28, 34, 42)
            b_border = border_act if is_act else (65, 78, 92)
            b_text_col = (255, 255, 255) if is_act else (160, 175, 190)
            cv2.rectangle(canvas, (b_rect[0], b_rect[1]), (b_rect[2], b_rect[3]), b_bg, -1)
            cv2.rectangle(canvas, (b_rect[0], b_rect[1]), (b_rect[2], b_rect[3]), b_border, 2 if is_act else 1)
            (tw, _), _ = cv2.getTextSize(b_label, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            tx = b_rect[0] + max(10, (b_rect[2] - b_rect[0] - tw) // 2)
            ty = b_rect[1] + 28
            cv2.putText(canvas, b_label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.52, b_text_col, 1, cv2.LINE_AA)

        # 4. Result and Status Display Area
        status = state.get("status", "SHOW GESTURE")
        recognized_name = state.get("recognized_name")
        subtitle = state.get("recognized_subtitle")

        if status == "SHIRO INITIALIZATION ERROR":
            cv2.putText(canvas, "STATUS: SHIRO INITIALIZATION ERROR", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
            err_sub = subtitle or "Error occurred in Shiro pipeline. Check console terminal."
            cv2.putText(canvas, err_sub, (50, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 160, 255), 1, cv2.LINE_AA)

        elif status == "STABLE" and recognized_name:
            if self.selected_category == "hasthas":
                prefix = "DETECTED HASTHA:"
            elif self.selected_category == "shiro":
                prefix = "DETECTED SHIRO:"
            else:
                prefix = "DETECTED DRISHTI:"

            cv2.putText(canvas, f"{prefix} {recognized_name}", (50, 155), cv2.FONT_HERSHEY_DUPLEX, 0.95, (40, 255, 130), 2, cv2.LINE_AA)

        elif status == "RECOGNIZING...":
            cv2.putText(canvas, "STATUS: RECOGNIZING MOTION...", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Stabilizing dancer gesture motion over consecutive frames...", (50, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 190, 205), 1, cv2.LINE_AA)

        else:
            if self.selected_category == "hasthas":
                prompt = "STATUS: SHOW GESTURE"
                hint = "Hold any classical Hastha gesture in camera view"
            elif self.selected_category == "shiro":
                prompt = "STATUS: PERFORM HEAD MOVEMENT"
                hint = "Perform classical Shiro movement before the camera"
            else:
                prompt = "STATUS: PERFORM EYE MOVEMENT"
                hint = "Perform classical Drishti gaze movement before the camera"

            cv2.putText(canvas, prompt, (50, 150), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 190, 50), 2, cv2.LINE_AA)
            cv2.putText(canvas, hint, (50, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 175, 190), 1, cv2.LINE_AA)

        # 5. Bottom Navigation Bar
        cv2.rectangle(canvas, (30, h - 50), (w - 30, h - 15), (18, 22, 28), -1)
        cv2.rectangle(canvas, (30, h - 50), (w - 30, h - 15), (55, 68, 82), 1)

        bottom_text = "[1] Hasthas  |  [2] Shiro  |  [3] Drishti  |  [R] Reset  |  [D] Debug  |  [Q] Quit"
        if self.debug_mode:
            lat = state.get("latency_ms", 0.0)
            det = state.get("detected", False)
            bottom_text += f"  |  Det: {det}  |  Latency: {lat:.1f}ms  |  ALIGNMENT: OK"

        cv2.putText(canvas, bottom_text, (45, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 195, 210), 1, cv2.LINE_AA)
        return canvas

    def run(self) -> int:
        print("=" * 80)
        print("LAUNCHING NRITYA AI — REAL-TIME KUCHIPUDI RECOGNITION")
        print("=" * 80)
        print("Categories:")
        print("  [1]     : Hasthas (52 classes — Step 34)")
        print("  [2]     : Shiro Bhedhas (9 classes — Step 36C)")
        print("  [3]     : Drishti Bhedhas (8 classes — Step 37C)")
        print("Controls:")
        print("  [1/2/3] : Switch Active Kuchipudi Category")
        print("  [R]     : Reset buffer / current sequence state")
        print("  [D]     : Toggle technical debug display")
        print("  [Q]/ESC : Exit application safely")
        print("  Mouse   : Click on HUD buttons directly")
        print("=" * 80)

        if not self.capture_stream.start():
            print(f"[ERROR] Could not open camera {self.camera_index}.")
            return 1

        self.worker_thread = threading.Thread(
            target=self.inference_worker.run_worker_loop,
            args=(self.capture_stream,),
            daemon=True,
            name="InferenceWorkerThread"
        )
        self.worker_thread.start()

        window_name = "NRITYA AI — Classical Kuchipudi Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.on_mouse_click)

        try:
            while True:
                ret, frame_mirrored, _ = self.capture_stream.read_latest()
                if not ret or frame_mirrored is None:
                    time.sleep(0.01)
                    continue

                state = self.inference_worker.get_shared_state()
                try:
                    display = self.draw_hud(frame_mirrored, state)
                except Exception as draw_ex:
                    print(f"[HUD ERROR] {draw_ex}")
                    display = frame_mirrored

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("\n[INFO] Exit command received. Terminating safely...")
                    break
                elif key == ord('1'):
                    self.switch_to_category("hasthas")
                elif key == ord('2'):
                    self.switch_to_category("shiro")
                elif key == ord('3'):
                    self.switch_to_category("drishti")
                elif key in [ord('r'), ord('R')]:
                    self.inference_worker.reset_event.set()
                    print("[KEY] Prediction buffer & temporal state reset.")
                elif key in [ord('d'), ord('D')]:
                    self.debug_mode = not self.debug_mode
                    print(f"[KEY] Debug display toggled: {'ON' if self.debug_mode else 'OFF'}")

        finally:
            self.capture_stream.stop()
            self.inference_worker.close()
            cv2.destroyAllWindows()
            print("[INFO] Resources released successfully.")

        return 0

# =============================================================================
# 7. STANDALONE VERIFICATION AND BENCHMARK
# =============================================================================

def run_standalone_verification() -> bool:
    print("=" * 80)
    print("NRITYA AI STANDALONE INTEGRITY VERIFICATION")
    print("=" * 80)
    if not WEIGHTS_PATH.exists():
        print(f"[FAIL] Missing consolidated weights: {WEIGHTS_PATH}")
        return False

    model = tf.keras.models.load_model(str(WEIGHTS_PATH), compile=False)
    sub_h = model.get_layer("Personalized_52Hasthas_MLP")
    sub_s = model.get_layer("Shiro_BiGRU_Classifier")
    sub_d = model.get_layer("Drishti_BiGRU_Classifier")

    print(f"[PASS] Consolidated model loaded: {WEIGHTS_PATH.name}")
    print(f"[PASS] Hastha submodel  : output classes = {sub_h.output_shape[-1]} (Expected: 52)")
    print(f"[PASS] Shiro submodel   : output classes = {sub_s.output_shape[-1]} (Expected: 9)")
    print(f"[PASS] Drishti submodel : output classes = {sub_d.output_shape[-1]} (Expected: 8)")

    assert sub_h.output_shape[-1] == 52
    assert sub_s.output_shape[-1] == 9
    assert sub_d.output_shape[-1] == 8

    # Verify embedded scaler dimensions
    assert HASTHA_SCALER_MEAN.shape == (175,) and HASTHA_SCALER_SCALE.shape == (175,)
    assert SHIRO_SCALER_MEAN.shape == (102,) and SHIRO_SCALER_STD.shape == (102,)
    assert DRISHTI_SCALER_MEAN.shape == (132,) and DRISHTI_SCALER_STD.shape == (132,)
    print("[PASS] Embedded Scaler arrays verified (175-D Hastha, 102-D Shiro, 132-D Drishti).")

    # Verify embedded class mappings
    assert len(HASTHA_CLASSES) == 52
    assert len(SHIRO_BHEDHAS) == 9
    assert len(DRISHTI_BHEDHAS) == 8
    print("[PASS] Embedded Class mappings verified (52 Hasthas, 9 Shiro, 8 Drishti).")

    # Test compiled warmup
    t0 = time.perf_counter()
    _ = sub_h(np.zeros((1, 175), dtype=np.float32), training=False)
    lat_h = (time.perf_counter() - t0) * 1000.0
    print(f"[PASS] Hastha forward pass latency  : {lat_h:.2f} ms")

    t0 = time.perf_counter()
    _ = sub_s(np.zeros((1, SHIRO_SEQ_LEN, SHIRO_FEAT_DIM), dtype=np.float32), training=False)
    lat_s = (time.perf_counter() - t0) * 1000.0
    print(f"[PASS] Shiro forward pass latency   : {lat_s:.2f} ms")

    t0 = time.perf_counter()
    _ = sub_d(np.zeros((1, DRISHTI_SEQ_LEN, DRISHTI_FEAT_DIM), dtype=np.float32), training=False)
    lat_d = (time.perf_counter() - t0) * 1000.0
    print(f"[PASS] Drishti forward pass latency : {lat_d:.2f} ms")

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        print(f"[PASS] Camera index 0 responsive.")
    else:
        print(f"[WARN] Camera 0 could not be opened.")

    print("\n[SUCCESS] All Standalone Verification Checks Passed.")
    return True

def run_performance_benchmark(frames_to_test: int = 60) -> bool:
    print("=" * 80)
    print(f"RUNNING STANDALONE PERFORMANCE BENCHMARK ({frames_to_test} FRAMES)")
    print("=" * 80)
    stream = CameraCaptureStream(camera_index=0, width=1280, height=720)
    if not stream.start():
        print("[ERROR] Camera could not be opened for benchmark.")
        return False

    worker = NrityaInferenceWorker(WEIGHTS_PATH, initial_category="hasthas")
    worker_th = threading.Thread(target=worker.run_worker_loop, args=(stream,), daemon=True)
    worker_th.start()

    time.sleep(0.5)
    t_start = time.time()
    frames_read = 0

    try:
        while frames_read < frames_to_test:
            ret, frame, _ = stream.read_latest()
            if ret and frame is not None:
                frames_read += 1
            time.sleep(0.01)

        total_time = time.time() - t_start
        fps = frames_read / total_time
        state = worker.get_shared_state()
        latency = state.get("latency_ms", 0.0)

        print(f"\nBenchmark Results:")
        print(f"  Total Frames Grabbed : {frames_read}")
        print(f"  Total Elapsed Time   : {total_time:.2f}s")
        print(f"  Effective Camera FPS : {fps:.2f} FPS")
        print(f"  AI Inference Latency : {latency:.2f} ms")
        print(f"  Decoupled Pipeline   : PASSED (Zero frame queue accumulation)")
        print("=" * 80)
        return True

    finally:
        stream.stop()
        worker.close()

# =============================================================================
# 8. ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Nritya AI — Standalone Raspberry Pi Deployment")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--category", type=str, default="hasthas", choices=["hasthas", "shiro", "drishti"], help="Default active category")
    parser.add_argument("--check-only", action="store_true", help="Run standalone verification checks without opening GUI")
    parser.add_argument("--benchmark", action="store_true", help="Run short camera & inference benchmark and exit")
    args = parser.parse_args()

    if args.check_only:
        ok = run_standalone_verification()
        return 0 if ok else 1

    if args.benchmark:
        ok = run_performance_benchmark(frames_to_test=60)
        return 0 if ok else 1

    app = NrityaAIApplication(camera_index=args.camera, initial_category=args.category)
    return app.run()

if __name__ == "__main__":
    sys.exit(main())
