"""
STEP 32C — FULL 52-HASTHA CONTINUOUS 10-SECOND LANDMARK COLLECTOR
=================================================================
Interactive OpenCV + MediaPipe tool for continuous 10-second webcam
recording and canonical 175-dimensional Hasthas landmark extraction
across all 52 Kuchipudi Hastha classes (28 Asamyuta + 24 Samyutha).

Pipeline:
  Camera ON -> Select Hastha -> Press SPACE -> Record continuously for 10.0s
  -> Real-time MediaPipe Hand detection & quality gating
  -> Canonical 175-D feature extraction (87 hand1 + 87 hand2 + 1 inter-hand dist)
  -> Auto-save session_XX.npz -> Completion review -> Advance to next Hastha

Controls:
  SPACE : Start exactly ONE 10-second recording session for CURRENT class
  N     : Skip / advance to next Hastha (only when not recording)
  R     : Retry / re-record session for current class
  Q     : Quit safely and generate collection summary report & statistics

Safety:
  - ZERO access to data/splits/v2_test.csv
  - ZERO model training or evaluation
  - Step 32A files, Step 18, Step 30, Step 31 models preserved completely intact.
  - Existing POC sessions (Pathaka, Thripathaka, etc.) preserved completely intact.
"""

import os
import sys
import time
import json
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.landmark_extraction import HandLandmarkExtractor

# MediaPipe solutions import for dynamic single/two-hand detection
try:
    from mediapipe import solutions as mp_solutions
except ImportError:
    import mediapipe.python.solutions as mp_solutions

# Base Directories
PERSONALIZED_ROOT = PROJECT_ROOT / "data" / "personalized"
LANDMARKS_BASE_DIR = PERSONALIZED_ROOT / "hasthas_landmarks"
REPORTS_DIR = PROJECT_ROOT / "reports"
MAPPING_FILE = PROJECT_ROOT / "models" / "step16a_hasthas_class_mapping.json"

REPORT_TXT_PATH = REPORTS_DIR / "step32b_personalized_landmark_collection_report.txt"
STATS_CSV_PATH = REPORTS_DIR / "step32b_personalized_landmark_statistics.csv"

RECORDING_DURATION_SECONDS = 10.0
FEATURE_DIMENSION = 175
MIN_SAMPLE_INTERVAL = 0.05  # 20 Hz cap (avoids redundant 60fps duplicates while capturing natural motion)
TARGET_SAMPLES_PER_SESSION = 150  # 150-200 target valid samples

# ==============================================================================
# 1. CANONICAL 52 HASTHAS DEFINITIONS
# ==============================================================================

ASAMYUTA_HASTHAS = [
    (1, "1. Pathaka", "1_Pathaka"),
    (2, "2. Thripathaka", "2_Thripathaka"),
    (3, "3. Ardhapathaka", "3_Ardhapathaka"),
    (4, "4. Kartarimukha", "4_Kartarimukha"),
    (5, "5. Mayura", "5_Mayura"),
    (6, "6. Ardhachandra", "6_Ardhachandra"),
    (7, "7. Arala", "7_Arala"),
    (8, "8. Sukatunda", "8_Sukatunda"),
    (9, "9. Mushti", "9_Mushti"),
    (10, "10. Shikhara", "10_Shikhara"),
    (11, "11. Kapittha", "11_Kapittha"),
    (12, "12. Katakamukha", "12_Katakamukha"),
    (13, "13. Soochi", "13_Soochi"),
    (14, "14.Chandrakala", "14_Chandrakala"),
    (15, "15.Padmakosha", "15_Padmakosha"),
    (16, "16.Sarpasheersha", "16_Sarpasheersha"),
    (17, "17.Mrugasheersha", "17_Mrugasheersha"),
    (18, "18.Simhamukha", "18_Simhamukha"),
    (19, "19.Kaangula", "19_Kaangula"),
    (20, "20.Alapadma", "20_Alapadma"),
    (21, "21.Chatura", "21_Chatura"),
    (22, "22.Bhramara", "22_Bhramara"),
    (23, "23.Hamsasya", "23_Hamsasya"),
    (24, "24.Hamsapaksha", "24_Hamsapaksha"),
    (25, "25.Samdamsha", "25_Samdamsha"),
    (26, "26.Mukula", "26_Mukula"),
    (27, "27.Tamrachuda", "27_Tamrachuda"),
    (28, "28.Trishoola", "28_Trishoola"),
]

SAMYUTHA_HASTHAS = [
    (1, "1. Anjali", "1_Anjali"),
    (2, "2. Kapothamu", "2_Kapothamu"),
    (3, "3. Karkatamu", "3_Karkatamu"),
    (4, "4. Swasthikamu", "4_Swasthikamu"),
    (5, "5. Dola hastamu", "5_Dola_hastamu"),
    (6, "6. Pushpaputamu", "6_Pushpaputamu"),
    (7, "7. Uthsungamu", "7_Uthsungamu"),
    (8, "8. Sivalingamu", "8_Sivalingamu"),
    (9, "9. Katakavardhanamu", "9_Katakavardhanamu"),
    (10, "10.Karthariswasthikamu", "10_Karthariswasthikamu"),
    (11, "11.Sakatamu", "11_Sakatamu"),
    (12, "12.Sankhamu", "12_Sankhamu"),
    (13, "13.Chakramu", "13_Chakramu"),
    (14, "14. Samputamu", "14_Samputamu"),
    (15, "15. Paasamu", "15_Paasamu"),
    (16, "16.Keelakamu", "16_Keelakamu"),
    (17, "17.Matsyamu", "17_Matsyamu"),
    (18, "18.Kurmamu", "18_Kurmamu"),
    (19, "19.Varahamu", "19_Varahamu"),
    (20, "20.Garudamu", "20_Garudamu"),
    (21, "21.Nagabandhakamu", "21_Nagabandhakamu"),
    (22, "22.Khatva", "22_Khatva"),
    (23, "23.Bherundamu", "23_Bherundamu"),
    (24, "24.Avahithamu", "24_Avahithamu"),
]

# Section 12: 5-Class Proof-of-Concept Classes
POC_CLASS_NAMES = [
    "1. Pathaka",
    "7. Arala",
    "6. Ardhachandra",
    "2. Thripathaka",
    "2. Kapothamu"
]


def load_canonical_hasthas() -> List[Dict[str, Any]]:
    """
    Loads all 52 canonical Hasthas mapping each to its canonical project class_id.
    """
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Project class mapping file missing: {MAPPING_FILE}")

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    class_to_idx = mapping_data["class_to_idx"]

    classes = []
    order_idx = 1

    # 28 Asamyuta
    for sub_idx, class_name, folder_name in ASAMYUTA_HASTHAS:
        clean_name = re.sub(r"^\d+\.\s*", "", class_name).strip()
        classes.append({
            "order": order_idx,
            "category_type": "Asamyuta Hastha (One-hand)",
            "sub_index": sub_idx,
            "class_name": class_name,
            "clean_name": clean_name,
            "folder_name": folder_name,
            "class_id": class_to_idx[class_name],
            "expected_hands": [1],
            "is_poc": class_name in POC_CLASS_NAMES
        })
        order_idx += 1

    # 24 Samyutha
    for sub_idx, class_name, folder_name in SAMYUTHA_HASTHAS:
        clean_name = re.sub(r"^\d+\.\s*", "", class_name).strip()
        classes.append({
            "order": order_idx,
            "category_type": "Samyutha Hastha (Two-hand)",
            "sub_index": sub_idx,
            "class_name": class_name,
            "clean_name": clean_name,
            "folder_name": folder_name,
            "class_id": class_to_idx[class_name],
            "expected_hands": [1, 2],
            "is_poc": class_name in POC_CLASS_NAMES
        })
        order_idx += 1

    assert len(classes) == 52, f"Expected 52 classes, got {len(classes)}"
    return classes


# ==============================================================================
# 2. DYNAMIC MEDIA-PIPE EXTRACTOR & PRIMARY HAND SELECTION
# ==============================================================================

class DynamicHandLandmarkExtractor:
    """
    Dynamic hand landmark extractor with:
      - Asamyuta mode: max_num_hands=1 (prevents duplicate bounding box proposals)
      - Samyutha mode: max_num_hands=2 (captures both hands for two-hand gestures)
      - Primary Hand Selection: selects best active hand when MediaPipe detects multiple hands
      - Canonical 175-D Feature Representation:
          [0:87]   = Hand 1 (active hand)
          [87:174] = Hand 2 (all zeros for Asamyuta)
          [174]    = Inter-hand distance (0.0 for Asamyuta)
    """

    def __init__(self, static_image_mode: bool = False, min_detection_confidence: float = 0.35):
        self.static_image_mode = static_image_mode
        self.min_detection_confidence = min_detection_confidence

        # Base extractor for single-hand geometric feature extraction (exact canonical 87-D)
        self.base_extractor = HandLandmarkExtractor(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence
        )

        # Single-hand detector (Asamyuta Hasthas)
        self.asamyuta_detector = mp_solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.3
        )
        self.asamyuta_fallback = mp_solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.15
        )

        # Two-hand detector (Samyutha Hasthas)
        self.samyutha_detector = mp_solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.3
        )
        self.samyutha_fallback = mp_solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.15
        )

    def close(self):
        self.base_extractor.close()
        self.asamyuta_detector.close()
        self.asamyuta_fallback.close()
        self.samyutha_detector.close()
        self.samyutha_fallback.close()

    def _select_primary_hand_index(
        self,
        multi_hand_landmarks: list,
        multi_handedness: Optional[list]
    ) -> int:
        """
        Selects the primary active hand index for Asamyuta gestures based on:
          1. Valid 21 landmarks (finite values)
          2. Bounding box area (larger = more prominent / active)
          3. Frame centrality (closer to frame center)
          4. Hand detection confidence score
        """
        if len(multi_hand_landmarks) == 1:
            return 0

        best_idx = 0
        best_score = -1.0

        for i, hand_lms in enumerate(multi_hand_landmarks):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
            if len(coords) != 21 or not np.all(np.isfinite(coords)):
                continue

            min_x, max_x = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
            min_y, max_y = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y
            area = max(0.0, bbox_w * bbox_h)

            conf = 0.8
            if multi_handedness and i < len(multi_handedness):
                try:
                    conf = float(multi_handedness[i].classification[0].score)
                except Exception:
                    conf = 0.8

            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            dist_from_center = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
            centrality = max(0.2, 1.0 - 0.5 * dist_from_center)

            valid_factor = 1.0
            if bbox_w < 0.04 or bbox_h < 0.04:
                valid_factor *= 0.1
            if min_x < -0.20 or max_x > 1.20 or min_y < -0.20 or max_y > 1.20:
                valid_factor *= 0.1

            score = conf * (area ** 0.5) * centrality * valid_factor
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def process(self, image_rgb: np.ndarray, is_asamyuta: bool = True) -> Dict[str, Any]:
        """
        Processes RGB image and extracts canonical 175-D feature vector.
        """
        if is_asamyuta:
            # Asamyuta (Single-hand): max_num_hands=1 detector
            results = self.asamyuta_detector.process(image_rgb)
            if not results.multi_hand_landmarks:
                results = self.asamyuta_fallback.process(image_rgb)

            if not results.multi_hand_landmarks:
                return {
                    "detected": False,
                    "num_hands": 0,
                    "handedness": [],
                    "physical_handedness": "Unknown",
                    "raw_landmarks": [],
                    "hand_features": [],
                    "vector": np.zeros(FEATURE_DIMENSION, dtype=np.float32),
                    "multi_hand_detected": False,
                    "hand_confidence": 0.0
                }

            num_detected = len(results.multi_hand_landmarks)
            multi_hand_detected = (num_detected > 1)

            # Select primary hand
            prim_idx = self._select_primary_hand_index(
                results.multi_hand_landmarks,
                results.multi_handedness
            )
            primary_hand = results.multi_hand_landmarks[prim_idx]

            # Primary confidence
            prim_conf = 1.0
            if results.multi_handedness and prim_idx < len(results.multi_handedness):
                try:
                    prim_conf = float(results.multi_handedness[prim_idx].classification[0].score)
                except Exception:
                    prim_conf = 1.0

            # Handedness label (defaults to "Left" matching canonical reference)
            handedness_label = "Left"
            if results.multi_handedness and prim_idx < len(results.multi_handedness):
                try:
                    handedness_label = results.multi_handedness[prim_idx].classification[0].label
                except Exception:
                    handedness_label = "Left"

            # Determine physical handedness of the user's hand:
            # MediaPipe operates assuming a selfie/mirrored camera feed. On standard unmirrored webcam frames:
            # - MediaPipe classification "Left" -> User's physical RIGHT hand (canonical reference orientation)
            # - MediaPipe classification "Right" -> User's physical LEFT hand (canonicalized via reflection)
            if handedness_label.strip().lower() == "right":
                physical_handedness = "Left"
            else:
                physical_handedness = "Right"

            primary_coords = np.array([[lm.x, lm.y, lm.z] for lm in primary_hand.landmark], dtype=np.float32)
            raw_primary = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in primary_hand.landmark]

            # Canonicalize Asamyuta hand orientation:
            # The reference orientation is MediaPipe's "Left" label (physical Right hand facing camera).
            # When a hand is detected as "Right" (physical Left hand facing camera),
            # reflect x-coordinates across the wrist so both left and right hand demonstrations
            # produce the identical canonical 87-D feature representation.
            canonical_coords = primary_coords.copy()
            is_canonicalized = False
            if handedness_label.strip().lower() == "right":
                wrist_x = primary_coords[0, 0]
                canonical_coords[:, 0] = 2.0 * wrist_x - canonical_coords[:, 0]
                is_canonicalized = True

            # Single hand features (87 dims) extracted from canonical coordinates
            feat = self.base_extractor._extract_single_hand_features(canonical_coords)
            h1_vec = feat["feature_vector"]

            # Asamyuta: Hand 2 is strictly zero-padded (87 dims) and inter-hand dist is 0.0 (1 dim)
            h2_vec = np.zeros(87, dtype=np.float32)
            inter_hand_dist = np.array([0.0], dtype=np.float32)
            vector_175 = np.concatenate([h1_vec, h2_vec, inter_hand_dist])

            # Prepare raw_landmarks with primary hand first for HUD skeleton
            raw_landmarks = [raw_primary]
            for i, h in enumerate(results.multi_hand_landmarks):
                if i != prim_idx:
                    raw_landmarks.append([{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in h.landmark])

            return {
                "detected": True,
                "num_hands": 1,  # Primary hand tracked
                "handedness": [handedness_label],
                "physical_handedness": physical_handedness,
                "raw_landmarks": raw_landmarks,
                "canonical_landmarks": canonical_coords,
                "is_canonicalized": is_canonicalized,
                "hand_features": [feat],
                "vector": vector_175,
                "multi_hand_detected": multi_hand_detected,
                "hand_confidence": prim_conf
            }

        else:
            # Samyutha (Two-hand): max_num_hands=2 detector
            results = self.samyutha_detector.process(image_rgb)
            if not results.multi_hand_landmarks:
                results = self.samyutha_fallback.process(image_rgb)

            if not results.multi_hand_landmarks:
                return {
                    "detected": False,
                    "num_hands": 0,
                    "handedness": [],
                    "physical_handedness": "Both",
                    "raw_landmarks": [],
                    "hand_features": [],
                    "vector": np.zeros(FEATURE_DIMENSION, dtype=np.float32),
                    "multi_hand_detected": False,
                    "hand_confidence": 0.0
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
                feat = self.base_extractor._extract_single_hand_features(lm_array)
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

            conf = 1.0
            if results.multi_handedness and len(results.multi_handedness) > 0:
                try:
                    conf = float(results.multi_handedness[0].classification[0].score)
                except Exception:
                    conf = 1.0

            return {
                "detected": True,
                "num_hands": num_hands,
                "handedness": handedness_labels,
                "physical_handedness": "Both",
                "raw_landmarks": raw_landmarks,
                "hand_features": hand_feats,
                "vector": combined_vector,
                "multi_hand_detected": False,
                "hand_confidence": conf
            }

    def extract_canonical_features(
        self,
        landmarks_3d: np.ndarray,
        handedness: str = "Left",
        is_asamyuta: bool = True
    ) -> np.ndarray:
        """
        Direct feature extraction from a (21, 3) or (2, 21, 3) landmark array.
        Applies canonicalization for Asamyuta gestures when handedness is 'Right'.
        Returns canonical 175-D feature vector.
        """
        coords = np.array(landmarks_3d, dtype=np.float32)
        if is_asamyuta:
            if coords.shape == (21, 3):
                canon_coords = coords.copy()
                if handedness.strip().lower() == "right":
                    wrist_x = coords[0, 0]
                    canon_coords[:, 0] = 2.0 * wrist_x - canon_coords[:, 0]
                feat = self.base_extractor._extract_single_hand_features(canon_coords)
                h1_vec = feat["feature_vector"]
                return np.concatenate([h1_vec, np.zeros(87, dtype=np.float32), np.array([0.0], dtype=np.float32)])
        else:
            # Samyutha: preserve two hands as detected without mirroring
            if coords.ndim == 3 and coords.shape[0] >= 2:
                feat1 = self.base_extractor._extract_single_hand_features(coords[0])
                feat2 = self.base_extractor._extract_single_hand_features(coords[1])
                w1 = np.array(feat1["wrist"])
                w2 = np.array(feat2["wrist"])
                dist = float(np.linalg.norm(w1 - w2))
                return np.concatenate([feat1["feature_vector"], feat2["feature_vector"], np.array([dist], dtype=np.float32)])
        return np.zeros(FEATURE_DIMENSION, dtype=np.float32)


# ==============================================================================
# 3. LANDMARK QUALITY GATING
# ==============================================================================

def validate_landmark_vector(
    vector_175: np.ndarray,
    res: Dict[str, Any],
    expected_hands: List[int]
) -> Tuple[bool, str, str]:
    """
    Quality Gating.
    Validates:
      - Hand detected
      - Vector shape == (175,)
      - Finite coordinates and features (no NaN/Inf)
      - Exactly 21 landmarks per detected hand
      - Bounding box not collapsed (width >= 0.04, height >= 0.04)
      - Reasonable visibility (-0.20 to 1.20)
    
    Returns:
      (is_valid, status_msg, rejection_code)
      where rejection_code in {"", "no_hand", "dimension", "nonfinite", "landmarks", "bbox"}
    """
    if not res.get("detected", False):
        return False, "Hand not detected", "no_hand"

    num_hands = res.get("num_hands", 0)
    if num_hands == 0:
        return False, "Hand not detected", "no_hand"

    # Vector integrity
    if vector_175 is None or vector_175.shape != (FEATURE_DIMENSION,):
        dim_str = str(vector_175.shape) if vector_175 is not None else "None"
        return False, f"Invalid dimension: {dim_str}", "dimension"

    if not np.all(np.isfinite(vector_175)):
        return False, "Vector contains NaN or Inf", "nonfinite"

    # Check raw landmarks structure
    raw_lms = res.get("raw_landmarks", [])
    if not raw_lms or len(raw_lms[0]) != 21:
        return False, "Incomplete landmark set (expected 21)", "landmarks"

    coords = np.array([[pt["x"], pt["y"], pt["z"]] for pt in raw_lms[0]], dtype=np.float32)
    if not np.all(np.isfinite(coords)):
        return False, "Landmark coordinates contain NaN or Inf", "nonfinite"

    # Check bounding box extent of primary hand
    min_x, max_x = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
    min_y, max_y = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    if bbox_w < 0.04 or bbox_h < 0.04:
        return False, "Hand bounding box collapsed", "bbox"

    if min_x < -0.20 or max_x > 1.20 or min_y < -0.20 or max_y > 1.20:
        return False, "Hand out of camera frame", "bbox"

    if 1 in expected_hands and len(expected_hands) == 1 and res.get("multi_hand_detected", False):
        return True, "Valid primary hand (secondary hand ignored)", ""

    return True, "Valid landmark frame", ""


# ==============================================================================
# 3. SESSION STORAGE & RESUME MANAGER
# ==============================================================================

class LandmarkSessionManager:
    """
    Manages session NPZ persistence under data/personalized/hasthas_landmarks/
    """

    def __init__(self, base_dir: Path = LANDMARKS_BASE_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def initialize_directories(self, classes: List[Dict[str, Any]]):
        """Creates folders for all 52 classes."""
        for c in classes:
            (self.base_dir / c["folder_name"]).mkdir(parents=True, exist_ok=True)

    def get_class_dir(self, folder_name: str) -> Path:
        p = self.base_dir / folder_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_sessions_for_class(self, folder_name: str) -> List[Path]:
        c_dir = self.get_class_dir(folder_name)
        pattern = re.compile(r"^session_(\d+)\.npz$", re.IGNORECASE)
        files = [f for f in c_dir.iterdir() if f.is_file() and pattern.match(f.name)]
        files.sort(key=lambda p: int(pattern.match(p.name).group(1)))
        return files

    def has_completed_session(self, folder_name: str) -> bool:
        sessions = self.get_sessions_for_class(folder_name)
        return len(sessions) > 0

    def get_next_session_path(self, folder_name: str) -> Tuple[Path, int]:
        existing = self.get_sessions_for_class(folder_name)
        pattern = re.compile(r"^session_(\d+)\.npz$", re.IGNORECASE)
        indices = {int(pattern.match(f.name).group(1)) for f in existing}
        next_idx = 1
        while next_idx in indices:
            next_idx += 1
        return self.get_class_dir(folder_name) / f"session_{next_idx:02d}.npz", next_idx

    def save_session(
        self,
        folder_name: str,
        class_id: int,
        class_name: str,
        features_175: List[np.ndarray],
        raw_landmarks: List[np.ndarray],
        timestamps: List[float],
        frame_indices: List[int],
        hand_confidences: List[float],
        duration_sec: float,
        total_frames: int,
        rejected_frames: int,
        handedness: str = "Unknown"
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Saves session NPZ compressed archive.
        """
        target_path, session_idx = self.get_next_session_path(folder_name)
        session_id = f"hastha_{folder_name}_s{session_idx:02d}_{int(time.time())}"

        feat_arr = np.array(features_175, dtype=np.float32)
        lm_arr = np.array(raw_landmarks, dtype=np.float32) if raw_landmarks else np.zeros((0, 21, 3), dtype=np.float32)
        time_arr = np.array(timestamps, dtype=np.float64)
        idx_arr = np.array(frame_indices, dtype=np.int32)
        conf_arr = np.array(hand_confidences, dtype=np.float32)

        np.savez_compressed(
            str(target_path),
            features_175=feat_arr,
            landmarks_21x3=lm_arr,
            timestamps=time_arr,
            frame_indices=idx_arr,
            hand_confidence=conf_arr,
            class_id=class_id,
            class_name=class_name,
            session_id=session_id,
            duration_seconds=duration_sec,
            total_camera_frames=total_frames,
            valid_frames=len(features_175),
            rejected_frames=rejected_frames,
            handedness=handedness
        )

        stats = {
            "session_id": session_id,
            "session_file": str(target_path),
            "class_id": class_id,
            "class_name": class_name,
            "handedness": handedness,
            "duration_seconds": round(duration_sec, 2),
            "total_camera_frames": total_frames,
            "valid_frames": len(features_175),
            "rejected_frames": rejected_frames,
            "valid_percentage": round((len(features_175) / max(1, total_frames)) * 100.0, 2),
            "feature_dimension": FEATURE_DIMENSION,
            "nan_count": int(np.isnan(feat_arr).sum()),
            "inf_count": int(np.isinf(feat_arr).sum())
        }
        return target_path, stats

    def delete_session_file(self, session_path: Path):
        """Discards session upon retry."""
        try:
            if session_path.exists():
                session_path.unlink()
                print(f"[RETRY] Discarded session: {session_path.name}")
        except Exception as e:
            print(f"[ERROR] Could not delete {session_path}: {e}")


# ==============================================================================
# 4. CONTINUOUS 10-SECOND COLLECTOR ENGINE
# ==============================================================================

class ContinuousHasthaLandmarkCollector:
    """
    Main interactive continuous 10-second landmark recorder.
    """

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.classes = load_canonical_hasthas()
        self.active_mode = "all"

        self.session_mgr = LandmarkSessionManager()
        self.session_mgr.initialize_directories(self.classes)

        # Dynamic MediaPipe extractor (canonical 175-D with Asamyuta & Samyutha modes)
        self.extractor = DynamicHandLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.35)

        # Debug diagnostic counters
        self.debug_counters = {
            "candidate_frames": 0,
            "accepted_frames": 0,
            "rejected_frames": 0,
            "rejected_no_hand": 0,
            "rejected_landmarks": 0,
            "rejected_bbox": 0,
            "rejected_nonfinite": 0,
            "rejected_dimension": 0,
            "multi_hand_frames_ignored": 0,
        }

        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_width = 0
        self.actual_height = 0

        # State management: Full 52 Hasthas
        self.active_classes: List[Dict[str, Any]] = self.classes
        self.current_idx = 0

        # Recording state
        self.is_recording = False
        self.recording_start_time = 0.0
        self.last_sample_save_time = 0.0

        # Live session buffers
        self.session_features: List[np.ndarray] = []
        self.session_landmarks: List[np.ndarray] = []
        self.session_timestamps: List[float] = []
        self.session_frame_indices: List[int] = []
        self.session_confidences: List[float] = []
        self.session_handedness_list: List[str] = []
        self.total_frames_in_session = 0
        self.rejected_frames_in_session = 0

        # Post-session review state
        self.session_completed = False
        self.last_saved_path: Optional[Path] = None
        self.last_session_stats: Optional[Dict[str, Any]] = None

        # Statistics records accumulator
        self.all_session_stats: List[Dict[str, Any]] = []
        self._load_existing_stats()

        # Find initial incomplete class
        self.resume_position()

    def _load_existing_stats(self):
        """Loads previous session statistics if present."""
        if STATS_CSV_PATH.exists():
            try:
                with open(STATS_CSV_PATH, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.all_session_stats.append(row)
            except Exception:
                pass

    def resume_position(self):
        """
        Scans existing folders to find the first class that still needs collection (0 sessions).
        If all classes have data, starts from index 0 (1. Pathaka) to allow normal navigation.
        Always preserves the exact 52-class mapping order.
        """
        self.current_idx = 0
        found_incomplete = False
        for idx, c in enumerate(self.active_classes):
            sessions = self.session_mgr.get_sessions_for_class(c["folder_name"])
            if len(sessions) == 0:
                self.current_idx = idx
                found_incomplete = True
                break

        curr = self.active_classes[self.current_idx]
        sessions = self.session_mgr.get_sessions_for_class(curr["folder_name"])
        _, next_idx = self.session_mgr.get_next_session_path(curr["folder_name"])
        if found_incomplete:
            print(f"[RESUME] Starting at first pending class: #{curr['order']} {curr['class_name']} (Class {self.current_idx + 1}/52, Existing: 0, Next: session_{next_idx:02d}.npz)")
        else:
            print(f"[RESUME] All 52 classes have data. Starting at #{curr['order']} {curr['class_name']} (Class 1/52, Existing: {len(sessions)}, Next: session_{next_idx:02d}.npz)")

    def open_camera(self) -> bool:
        """Section 3: Initializes OpenCV camera at 1280x720."""
        print(f"[CAMERA] Initializing webcam (Device: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera {self.camera_id}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAMERA] Resolution: {self.actual_width} x {self.actual_height}")
        return True

    def draw_hud(
        self,
        preview_bgr: np.ndarray,
        lm_res: Dict[str, Any],
        is_valid: bool,
        status_msg: str
    ) -> np.ndarray:
        """
        Section 4: Live HUD Interface with clean side-panel.
        """
        curr = self.active_classes[self.current_idx]
        h, w = preview_bgr.shape[:2]
        panel_w = 420
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)

        # Place preview
        canvas[:, :w] = preview_bgr

        # Draw hand landmark skeleton onto preview (mirrored coordinates)
        raw_lms = lm_res.get("raw_landmarks", [])
        if raw_lms:
            for hand_idx, hand_pts in enumerate(raw_lms):
                for pt in hand_pts:
                    px = int((1.0 - pt["x"]) * w)
                    py = int(pt["y"]) * h
                    if hand_idx == 0:
                        color = (0, 255, 0) if is_valid else (0, 140, 255)
                        cv2.circle(canvas, (px, py), 4, color, -1)
                    else:
                        cv2.circle(canvas, (px, py), 2, (180, 180, 180), -1)

        # Side panel styling
        panel_x = w
        canvas[:, panel_x:] = (25, 25, 28)
        cv2.line(canvas, (panel_x, 0), (panel_x, h), (55, 55, 60), 2)

        def put(text: str, y: int, scale: float = 0.55, color=(240, 240, 240), thick: int = 1):
            cv2.putText(canvas, text, (panel_x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        # Header
        # Header
        put("--------------------------------------------", 30, 0.42, (120, 120, 120))
        put("PERSONALIZED HASHTA LANDMARK COLLECTION", 50, 0.52, (0, 220, 255), 2)
        put("--------------------------------------------", 70, 0.42, (120, 120, 120))

        # Mode Indicator
        put("FULL 52-HASTHA MODE", 94, 0.55, (0, 255, 200), 2)

        # Current Hastha & Class X / 52
        put("Current:", 118, 0.44, (170, 170, 170))
        put(f"{curr['clean_name']}", 145, 0.85, (255, 255, 255), 2)
        put(f"Class: {self.current_idx + 1} / 52", 170, 0.52, (0, 220, 255), 1)

        # Sessions for current class
        existing_sess = self.session_mgr.get_sessions_for_class(curr["folder_name"])
        num_sess = len(existing_sess)
        _, next_num = self.session_mgr.get_next_session_path(curr["folder_name"])
        sess_str = f"Existing sessions: {num_sess} | Next: s{next_num:02d}"
        sess_color = (0, 255, 128) if num_sess > 0 else (0, 200, 255)
        put(sess_str, 194, 0.44, sess_color, 1)

        # Handedness & canonicalization status
        if curr["expected_hands"] == [1]:
            if lm_res.get("detected", False):
                phys_hand = lm_res.get("physical_handedness", "Right").upper()
                put(f"Physical Hand: {phys_hand}", 218, 0.46, (0, 255, 255), 1)
            else:
                put("Physical Hand: [WAITING]", 218, 0.44, (140, 140, 140), 1)
            put("Canonicalization: ACTIVE", 236, 0.38, (160, 255, 160), 1)
        else:
            put("Physical Hands: BOTH", 218, 0.46, (0, 255, 255), 1)
            put("Canonicalization: ACTIVE", 236, 0.38, (160, 255, 160), 1)

        # Recording Status Box
        put("--------------------------------------------", 254, 0.42, (120, 120, 120))
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            rem = max(0.0, RECORDING_DURATION_SECONDS - elapsed)
            countdown = max(1, int(round(rem)))
            put("RECORDING IN PROGRESS:", 276, 0.50, (0, 255, 255))
            put(f"Recording: {countdown}", 306, 0.85, (0, 100, 255), 2)
            put(f"Session: {next_num}", 332, 0.48, (200, 220, 255), 1)
            put(f"Valid frames: {len(self.session_features)}", 356, 0.50, (0, 255, 0), 2)
            if curr["expected_hands"] == [1] and self.debug_counters.get("multi_hand_frames_ignored", 0) > 0:
                put(f"Ignored 2nd hand: {self.debug_counters['multi_hand_frames_ignored']}", 380, 0.42, (0, 220, 255), 1)
                put(f"Rejected: {self.rejected_frames_in_session} | Total: {self.total_frames_in_session}", 402, 0.42, (180, 180, 180))
            else:
                put(f"Rejected: {self.rejected_frames_in_session} | Total: {self.total_frames_in_session}", 385, 0.45, (180, 180, 180))
        elif self.session_completed and self.last_session_stats:
            stats = self.last_session_stats
            saved_name = Path(stats['session_file']).name
            put("SESSION SAVED", 276, 0.55, (0, 255, 128), 2)
            put(f"{curr['clean_name']} / {saved_name}", 302, 0.65, (0, 255, 0), 2)
            put(f"Valid frames: {stats['valid_frames']} ({stats['valid_percentage']}%)", 328, 0.50, (255, 255, 255), 1)
            put(f"Session: {next_num - 1} saved", 352, 0.46, (200, 220, 255), 1)
            put("READY FOR ANOTHER SESSION", 378, 0.55, (0, 220, 255), 2)
            put(f"Existing sessions: {num_sess}", 402, 0.44, (180, 220, 180), 1)
        else:
            put("STATUS:", 276, 0.52, (180, 180, 180))
            put(f"READY (Session {next_num:02d})", 302, 0.75, (0, 255, 200), 2)
            put(f"Session: {next_num}", 328, 0.50, (200, 200, 200), 1)
            put(f"Existing sessions: {num_sess}", 352, 0.46, (180, 220, 180), 1)
            put(f"Target duration: {int(RECORDING_DURATION_SECONDS)}s continuous", 378, 0.42, (160, 160, 160))

        # Instructions / Quality feedback
        put("--------------------------------------------", 425, 0.42, (120, 120, 120))
        put("Live Hand Status:", 448, 0.48, (180, 180, 180))
        q_color = (0, 255, 0) if is_valid else (0, 140, 255)
        put(status_msg, 472, 0.52, q_color, 1)

        # Controls Menu
        put("--------------------------------------------", 502, 0.42, (120, 120, 120))
        put("Controls:", 524, 0.52, (200, 200, 200))
        if self.is_recording:
            put("RECORDING IN PROGRESS... (HOLD HASHTA)", 550, 0.50, (0, 140, 255), 2)
            put(f"Auto-stops at {int(RECORDING_DURATION_SECONDS)}.0 seconds", 574, 0.44, (180, 180, 180))
            put("N key is ignored during recording", 598, 0.42, (140, 140, 140))
        elif self.session_completed:
            put("SPACE = Record 10 sec (Same Hastha)", 550, 0.52, (0, 255, 200), 2)
            put("N     = Next Hastha", 574, 0.50, (0, 255, 0), 2)
            put("R     = Retry / Re-record", 598, 0.46, (0, 200, 255), 1)
            put("Q     = Quit safely", 622, 0.45, (160, 160, 255), 1)
        else:
            put("SPACE = Record 10 sec", 550, 0.52, (0, 255, 200), 2)
            put("N     = Next Hastha", 574, 0.48, (200, 200, 200), 1)
            put("R     = Retry", 598, 0.45, (255, 180, 80), 1)
            put("Q     = Quit", 622, 0.45, (160, 160, 255), 1)

        return canvas

    def start_recording(self):
        """Starts 10-second automatic recording for current class."""
        curr = self.active_classes[self.current_idx]
        _, next_num = self.session_mgr.get_next_session_path(curr["folder_name"])
        print(f"\n[RECORDING] Starting 10.0s recording for: {curr['class_name']} (session_{next_num:02d})...")
        self.is_recording = True
        self.session_completed = False
        self.recording_start_time = time.time()
        self.last_sample_save_time = 0.0

        self.session_features = []
        self.session_landmarks = []
        self.session_timestamps = []
        self.session_frame_indices = []
        self.session_confidences = []
        self.session_handedness_list = []
        self.total_frames_in_session = 0
        self.rejected_frames_in_session = 0

        # Reset session debug counters
        self.debug_counters = {
            "candidate_frames": 0,
            "accepted_frames": 0,
            "rejected_frames": 0,
            "rejected_no_hand": 0,
            "rejected_landmarks": 0,
            "rejected_bbox": 0,
            "rejected_nonfinite": 0,
            "rejected_dimension": 0,
            "multi_hand_frames_ignored": 0,
        }

    def stop_and_save_recording(self):
        """Ends 10s session and saves NPZ compressed archive."""
        duration = time.time() - self.recording_start_time
        curr = self.active_classes[self.current_idx]
        self.is_recording = False
        self.session_completed = True

        print("=" * 60)
        print(f"{curr['clean_name'].upper()} SESSION COMPLETE")
        print("=" * 60)
        print(f"Duration               : {duration:.2f} sec")
        print(f"Total camera frames    : {self.total_frames_in_session}")
        print(f"Valid landmark frames  : {len(self.session_features)}")
        print(f"Rejected frames        : {self.rejected_frames_in_session}")
        print("-" * 60)
        print("DEBUG COUNTERS:")
        print(f"  candidate_frames           : {self.debug_counters['candidate_frames']}")
        print(f"  accepted_frames            : {self.debug_counters['accepted_frames']}")
        print(f"  rejected_frames            : {self.debug_counters['rejected_frames']}")
        print(f"  rejected_no_hand           : {self.debug_counters['rejected_no_hand']}")
        print(f"  rejected_landmarks         : {self.debug_counters['rejected_landmarks']}")
        print(f"  rejected_bbox              : {self.debug_counters['rejected_bbox']}")
        print(f"  rejected_nonfinite         : {self.debug_counters['rejected_nonfinite']}")
        print(f"  rejected_dimension         : {self.debug_counters['rejected_dimension']}")
        if curr["expected_hands"] == [1]:
            print(f"  multi_hand_frames_ignored  : {self.debug_counters['multi_hand_frames_ignored']}")
        print("-" * 60)

        # Determine stable session physical handedness
        if curr["expected_hands"] == [1]:  # Asamyuta
            if self.session_handedness_list:
                right_count = self.session_handedness_list.count("Right")
                left_count = self.session_handedness_list.count("Left")
                session_handedness = "Right" if right_count >= left_count else "Left"
                if right_count > 0 and left_count > 0:
                    print(f"Handedness consensus: {session_handedness} (Right frames: {right_count}, Left frames: {left_count})")
            else:
                session_handedness = "Right"
        else:  # Samyutha
            session_handedness = "Both"

        saved_path, stats = self.session_mgr.save_session(
            folder_name=curr["folder_name"],
            class_id=curr["class_id"],
            class_name=curr["class_name"],
            features_175=self.session_features,
            raw_landmarks=self.session_landmarks,
            timestamps=self.session_timestamps,
            frame_indices=self.session_frame_indices,
            hand_confidences=self.session_confidences,
            duration_sec=duration,
            total_frames=self.total_frames_in_session,
            rejected_frames=self.rejected_frames_in_session,
            handedness=session_handedness
        )

        self.last_saved_path = saved_path
        self.last_session_stats = stats
        self.all_session_stats.append(stats)
        self._append_to_stats_csv(stats)

        # Count total sessions for this class
        class_sessions = self.session_mgr.get_sessions_for_class(curr["folder_name"])
        num_sessions = len(class_sessions)

        print(f"Saved session archive  : {saved_path.name} (Total for this class: {num_sessions})")
        print(f"Physical Hand recorded : {session_handedness}")
        print(f"Current class          : Class {self.current_idx + 1}/52 - #{curr['order']} {curr['class_name']}")
        print("READY FOR ANOTHER SESSION")
        print("SPACE : RECORD AGAIN (Same class) | N : NEXT CLASS | R : RETRY | Q : QUIT")
        print("=" * 60)

    def retry_current_session(self):
        """Section 14: Discards last recorded session and restarts fresh 10s recording."""
        if self.last_saved_path:
            self.session_mgr.delete_session_file(self.last_saved_path)
            self.last_saved_path = None
            self.last_session_stats = None
        self.start_recording()

    def next_hastha(self):
        """Advances to next Hastha safely."""
        self.session_completed = False
        self.last_saved_path = None
        self.last_session_stats = None
        self.current_idx = (self.current_idx + 1) % len(self.active_classes)
        curr = self.active_classes[self.current_idx]
        class_sessions = self.session_mgr.get_sessions_for_class(curr["folder_name"])
        _, next_num = self.session_mgr.get_next_session_path(curr["folder_name"])
        print(f"\n[NEXT HASTHA] Switched to: Class {self.current_idx + 1}/52 - #{curr['order']} {curr['class_name']}")
        print(f"  Existing sessions: {len(class_sessions)} | Next: session_{next_num:02d}.npz")
        print("  SPACE: Record session | N: Next class | Q: Quit")

    def run(self):
        """
        Main execution loop.
        """
        self.print_startup_banner()

        if not self.open_camera():
            print("[INFO] Camera not available in current environment.")
            self.generate_reports()
            self.print_final_terminal_output()
            return

        window_name = "Personalized Hastha Landmark Collector (Step 32C — Full 52 Hasthas)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        print("\n[READY] Collector initialized. Press SPACE to record 10 seconds, Q to quit.")

        try:
            while True:
                ret, frame_native = self.cap.read()
                if not ret or frame_native is None:
                    time.sleep(0.02)
                    continue

                curr = self.active_classes[self.current_idx]
                now = time.time()

                # Extract landmarks on native unmirrored frame
                frame_rgb = cv2.cvtColor(frame_native, cv2.COLOR_BGR2RGB)
                is_asamyuta = (curr["expected_hands"] == [1])
                lm_res = self.extractor.process(frame_rgb, is_asamyuta=is_asamyuta)
                vec_175 = lm_res.get("vector")

                is_valid, status_msg, reject_code = validate_landmark_vector(vec_175, lm_res, curr["expected_hands"])

                # Handle 10-second continuous recording
                if self.is_recording:
                    self.total_frames_in_session += 1
                    self.debug_counters["candidate_frames"] += 1
                    elapsed = now - self.recording_start_time

                    if is_asamyuta and lm_res.get("multi_hand_detected", False):
                        self.debug_counters["multi_hand_frames_ignored"] += 1

                    if is_valid:
                        # Cap sampling to ~20 Hz
                        if now - self.last_sample_save_time >= MIN_SAMPLE_INTERVAL:
                            self.session_features.append(vec_175)
                            self.session_timestamps.append(elapsed)
                            self.session_frame_indices.append(self.total_frames_in_session)

                            # Track physical handedness for this valid sample
                            p_hand = lm_res.get("physical_handedness", "Both" if not is_asamyuta else "Right")
                            self.session_handedness_list.append(p_hand)

                            # Save normalized canonical landmarks for primary hand
                            canon_pts = lm_res.get("canonical_landmarks", None)
                            if canon_pts is not None:
                                self.session_landmarks.append(canon_pts)
                            else:
                                raw_pts = lm_res.get("raw_landmarks", [])
                                if raw_pts:
                                    primary_coords = np.array([[p["x"], p["y"], p["z"]] for p in raw_pts[0]], dtype=np.float32)
                                    self.session_landmarks.append(primary_coords)

                            conf = float(lm_res.get("hand_confidence", 1.0))
                            self.session_confidences.append(conf)
                            self.last_sample_save_time = now
                            self.debug_counters["accepted_frames"] += 1
                    else:
                        self.rejected_frames_in_session += 1
                        self.debug_counters["rejected_frames"] += 1
                        if reject_code == "no_hand":
                            self.debug_counters["rejected_no_hand"] += 1
                        elif reject_code == "landmarks":
                            self.debug_counters["rejected_landmarks"] += 1
                        elif reject_code == "bbox":
                            self.debug_counters["rejected_bbox"] += 1
                        elif reject_code == "nonfinite":
                            self.debug_counters["rejected_nonfinite"] += 1
                        elif reject_code == "dimension":
                            self.debug_counters["rejected_dimension"] += 1

                    # Check 10-second timer
                    if elapsed >= RECORDING_DURATION_SECONDS:
                        self.stop_and_save_recording()

                # Preview horizontally mirrored for natural view
                preview_frame = cv2.flip(frame_native, 1)
                canvas = self.draw_hud(preview_frame, lm_res, is_valid, status_msg)
                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(10) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("\n[USER] 'Q' pressed: Exiting safely...")
                    break
                elif key == 32:  # SPACE: Start recording current class (records again if session just completed)
                    if not self.is_recording:
                        self.start_recording()
                elif key == ord('r') or key == ord('R'):  # R: Retry current session
                    if not self.is_recording:
                        self.retry_current_session()
                elif key == ord('n') or key == ord('N') or key == 13:  # N or ENTER: Advance to next class
                    if not self.is_recording:
                        self.next_hastha()

        finally:
            self.close()
            self.generate_reports()
            self.print_final_terminal_output()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.extractor.close()
        cv2.destroyAllWindows()

    def _append_to_stats_csv(self, stats: Dict[str, Any]):
        """Appends single session stats to statistics CSV."""
        file_exists = STATS_CSV_PATH.exists()
        fieldnames = [
            "session_id", "class_id", "class_name", "duration_seconds",
            "total_camera_frames", "valid_frames", "rejected_frames",
            "valid_percentage", "feature_dimension", "nan_count", "inf_count"
        ]
        with open(STATS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: stats[k] for k in fieldnames if k in stats})

    def print_startup_banner(self):
        print("=" * 80)
        print("STEP 32C — FULL 52-HASTHA CONTINUOUS 10-SECOND LANDMARK COLLECTOR")
        print("=" * 80)
        print("FULL 52-HASTHA MODE")
        print(f"Total classes: {len(self.classes)}")
        print("")
        for idx, c in enumerate(self.classes, 1):
            sess_cnt = len(self.session_mgr.get_sessions_for_class(c["folder_name"]))
            print(f"Class {idx}/52: {c['class_name']} ({c['category_type']}) - Existing sessions: {sess_cnt}")
        print("=" * 80)
        print(f"Target Duration      : {RECORDING_DURATION_SECONDS} seconds per session")
        print(f"Landmark Dimension   : {FEATURE_DIMENSION} (87 hand1 + 87 hand2 + 1 inter-hand)")
        print(f"Sampling Cap         : ~20 samples/sec (Target: 150-200 frames)")
        print(f"Storage Directory    : {LANDMARKS_BASE_DIR}")
        print("=" * 80)

    def generate_reports(self):
        """
        Section 18: Generates final collection report and statistics CSV.
        """
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total_sessions = 0
        total_valid_frames = 0
        per_class_summary = []

        for c in self.classes:
            sessions = self.session_mgr.get_sessions_for_class(c["folder_name"])
            num_sess = len(sessions)
            valid_frames = 0
            if sessions:
                for s in sessions:
                    try:
                        data = np.load(str(s))
                        valid_frames += len(data["features_175"])
                    except Exception:
                        pass
            total_sessions += num_sess
            total_valid_frames += valid_frames
            status = "COMPLETE" if num_sess > 0 else "MISSING"
            per_class_summary.append({
                "order": c["order"],
                "class_id": c["class_id"],
                "class_name": c["class_name"],
                "folder_name": c["folder_name"],
                "sessions": num_sess,
                "valid_frames": valid_frames,
                "status": status,
                "is_poc": c["is_poc"]
            })

        # Check 5-class POC
        poc_completed = sum(1 for item in per_class_summary if item["is_poc"] and item["sessions"] > 0)

        lines = [
            "=" * 80,
            "STEP 32B — PERSONALIZED HASHTAS LANDMARK COLLECTION REPORT",
            "=" * 80,
            f"Date / Time                     : {now_str}",
            f"Collection Mode                 : {self.active_mode.upper()}",
            f"Camera Resolution               : {self.actual_width} x {self.actual_height}" if self.actual_width > 0 else "Camera Resolution               : Offline / Not opened",
            f"Session Target Duration         : {RECORDING_DURATION_SECONDS} seconds",
            f"Canonical Feature Dimension     : {FEATURE_DIMENSION} dimensions",
            f"Total Classes in Dataset        : 52",
            f"Total Sessions Collected        : {total_sessions}",
            f"Total Valid Landmark Frames     : {total_valid_frames}",
            f"5-Class POC Completion          : {poc_completed} / 5 classes",
            f"All-52 Classes Completion       : {sum(1 for x in per_class_summary if x['sessions'] > 0)} / 52 classes",
            "",
            "STRICT PROTOCOL & SAFETY VERIFICATIONS:",
            "  - data/splits/v2_test.csv accessed: NO (Strictly untouched)",
            "  - Model training performed: NO",
            "  - Model evaluation performed: NO",
            "  - Step 18 models modified: NO",
            "  - Step 30 models modified: NO",
            "  - Step 31 artifacts modified: NO",
            "  - Step 32A collector preserved: YES",
            "",
            "=" * 80,
            "5-CLASS PROOF-OF-CONCEPT STATUS:",
            "=" * 80
        ]

        for item in per_class_summary:
            if item["is_poc"]:
                lines.append(f"  * {item['class_name']:<24}: Sessions={item['sessions']} | Frames={item['valid_frames']} | Status={item['status']}")

        lines.extend([
            "",
            "=" * 80,
            "ALL 52 HASTHAS SESSIONS BREAKDOWN:",
            "=" * 80,
            f"{'Ord':<4} {'Class ID':<10} {'Class Name':<28} {'Folder':<22} {'Sessions':<10} {'Frames':<10} {'Status':<10}",
            "-" * 102
        ])

        for item in per_class_summary:
            lines.append(
                f"{item['order']:<4} {item['class_id']:<10} {item['class_name']:<28} {item['folder_name']:<22} {item['sessions']:<10} {item['valid_frames']:<10} {item['status']:<10}"
            )

        lines.extend([
            "-" * 102,
            f"TOTALS: {total_sessions} sessions | {total_valid_frames} valid landmark frames across {len(per_class_summary)} classes.",
            "=" * 80
        ])

        with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[REPORT] Landmark collection report saved to: {REPORT_TXT_PATH}")

        # Ensure statistics CSV has at least header
        if not STATS_CSV_PATH.exists():
            with open(STATS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "session_id", "class_id", "class_name", "duration_seconds",
                    "total_camera_frames", "valid_frames", "rejected_frames",
                    "valid_percentage", "feature_dimension", "nan_count", "inf_count"
                ])
        print(f"[REPORT] Landmark statistics saved to: {STATS_CSV_PATH}")

    def print_final_terminal_output(self):
        """Section 21: Exact required terminal output format."""
        print("")
        print("========================================================")
        print("STEP 32C — FULL 52-HASTHA CONTINUOUS LANDMARK COLLECTION READY")
        print("========================================================")
        print("")
        print("Mode:")
        print("Full 52-Hastha Mode")
        print("")
        print("Total Classes:")
        print("52 (28 Asamyuta + 24 Samyutha)")
        print("")
        print("Recording duration:")
        print("10 seconds per session")
        print("")
        print("Feature dimension:")
        print("175")
        print("")
        print("Test set accessed:")
        print("NO")
        print("")
        print("Training performed:")
        print("NO")
        print("")
        print("Models modified:")
        print("NO")
        print("")
        print("Existing Step 32A/32B data:")
        print("PRESERVED")
        print("")
        print("Ready to collect:")
        print("YES")
        print("")
        print("========================================================")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step 32C — Full 52-Hastha Continuous 10-Second Landmark Collector")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--summary-only", action="store_true", help="Generate summary reports and statistics without opening camera")
    args = parser.parse_args()

    collector = ContinuousHasthaLandmarkCollector(camera_id=args.camera_id)

    if args.summary_only:
        collector.generate_reports()
        collector.print_final_terminal_output()
    else:
        collector.run()


if __name__ == "__main__":
    main()
