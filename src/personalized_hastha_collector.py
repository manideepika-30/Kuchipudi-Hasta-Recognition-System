"""
STEP 32A — PERSONALIZED HASHTA WEBCAM DATA COLLECTOR
=====================================================
Interactive, reliable OpenCV + MediaPipe webcam data collection tool for
personalized Kuchipudi Hastha recognition.

Collects 20 valid images for each of the 52 Hastha classes (1,040 total).
Zero model training or evaluation. Zero test set access.

Controls:
  SPACE : Capture valid sample (debounced 0.5s)
  R     : Retake/delete most recent sample for current class
  N     : Skip / advance to next Hastha
  ENTER : Confirm and proceed to next Hastha when 20/20 complete
  Q     : Quit safely and generate collection summary report & statistics
"""

import os
import sys
import time
import json
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# MediaPipe hands import
try:
    from mediapipe import solutions as mp_solutions
except ImportError:
    import mediapipe.python.solutions as mp_solutions

# Directories
PERSONALIZED_ROOT = PROJECT_ROOT / "data" / "personalized"
HASTHAS_DATA_DIR = PERSONALIZED_ROOT / "hasthas"
REPORTS_DIR = PROJECT_ROOT / "reports"
MAPPING_FILE = PROJECT_ROOT / "models" / "step16a_hasthas_class_mapping.json"

REPORT_TXT_PATH = REPORTS_DIR / "step32a_personalized_hasthas_collection_report.txt"
STATS_CSV_PATH = REPORTS_DIR / "step32a_hasthas_collection_statistics.csv"

TARGET_SAMPLES_PER_CLASS = 20
TOTAL_CLASSES = 52
TOTAL_TARGET_SAMPLES = TOTAL_CLASSES * TARGET_SAMPLES_PER_CLASS  # 1040

# ==============================================================================
# 1. CANONICAL 52 HASTHAS DEFINITIONS (28 Asamyuta + 24 Samyutha)
# ==============================================================================

# 28 Asamyuta Hasthas (One-hand gestures) in traditional dance sequence
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

# 24 Samyutha Hasthas (Two-hand gestures) in traditional dance sequence
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


def load_canonical_hasthas() -> List[Dict[str, Any]]:
    """
    Loads and builds the 52 canonical Hasthas class list, mapping each
    to its canonical project class_id from step16a_hasthas_class_mapping.json.
    """
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Project class mapping file missing: {MAPPING_FILE}")

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    class_to_idx = mapping_data["class_to_idx"]

    classes = []
    order_idx = 1

    # Add 28 Asamyuta
    for sub_idx, class_name, folder_name in ASAMYUTA_HASTHAS:
        if class_name not in class_to_idx:
            raise KeyError(f"Expected class '{class_name}' not found in project mapping!")
        clean_name = re.sub(r"^\d+\.\s*", "", class_name).strip()
        classes.append({
            "order": order_idx,
            "category_type": "Asamyuta Hastha (One-hand)",
            "sub_index": sub_idx,
            "class_name": class_name,
            "clean_name": clean_name,
            "folder_name": folder_name,
            "class_id": class_to_idx[class_name],
            "expected_hands": [1]
        })
        order_idx += 1

    # Add 24 Samyutha
    for sub_idx, class_name, folder_name in SAMYUTHA_HASTHAS:
        if class_name not in class_to_idx:
            raise KeyError(f"Expected class '{class_name}' not found in project mapping!")
        clean_name = re.sub(r"^\d+\.\s*", "", class_name).strip()
        classes.append({
            "order": order_idx,
            "category_type": "Samyutha Hastha (Two-hand)",
            "sub_index": sub_idx,
            "class_name": class_name,
            "clean_name": clean_name,
            "folder_name": folder_name,
            "class_id": class_to_idx[class_name],
            "expected_hands": [1, 2]  # Two-hand gesture; hands may touch/overlap
        })
        order_idx += 1

    assert len(classes) == TOTAL_CLASSES, f"Expected 52 classes, got {len(classes)}"
    return classes


# ==============================================================================
# 2. HAND DETECTION QUALITY CHECKER
# ==============================================================================

class HandQualityChecker:
    """
    MediaPipe Hands validator to guarantee high-quality hand gesture frames.
    Checks:
      - Usable hand detection matching gesture type
      - Sufficient hand landmarks (exactly 21 landmarks)
      - Finite coordinates (no NaN/Inf)
      - Valid non-collapsed bounding box
      - Hand reasonably visible inside frame
    """

    def __init__(self, min_detection_confidence: float = 0.35):
        self.min_detection_confidence = min_detection_confidence
        self.mp_hands = mp_solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.35
        )

    def close(self):
        if self.detector is not None:
            self.detector.close()

    def evaluate_frame(
        self,
        frame_rgb: np.ndarray,
        expected_hands: List[int]
    ) -> Tuple[bool, str, Optional[Any]]:
        """
        Evaluates frame for hand presence and landmark validity.
        Returns:
          (is_valid, status_message, mp_results)
        """
        results = self.detector.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return False, "Hand not detected — try again", results

        usable_hands = []
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lms = hand_landmarks.landmark
            if len(lms) != 21:
                continue

            # Check finiteness
            coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            if not np.all(np.isfinite(coords)):
                continue

            # Check bounding box
            min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
            min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y

            # Hand must not be a collapsed point (at least 5% of normalized dimension)
            if bbox_w < 0.04 or bbox_h < 0.04:
                continue

            # Hand must be reasonably visible within frame [-0.15, 1.15]
            if min_x < -0.15 or max_x > 1.15 or min_y < -0.15 or max_y > 1.15:
                continue

            usable_hands.append(hand_idx)

        num_usable = len(usable_hands)
        if num_usable == 0:
            return False, "Hand not detected — try again", results

        # For Asamyuta (one-hand), exactly 1 usable hand expected
        if 1 in expected_hands and len(expected_hands) == 1:
            if num_usable == 1:
                return True, "Captured", results
            else:
                return False, "Multiple hands detected — show single hand", results

        # For Samyutha (two-hand), 1 or 2 usable hands accepted (overlap occurs)
        if num_usable in expected_hands:
            return True, "Captured", results

        return False, "Hand not detected — try again", results


# ==============================================================================
# 3. DIRECTORY & SAMPLE MANAGER (RESUME SUPPORT)
# ==============================================================================

class DatasetDirectoryManager:
    """
    Manages creation and scanning of data/personalized/hasthas/<folder_name>/
    Guarantees safe restartability and sequential sample naming.
    """

    def __init__(self, base_dir: Path = HASTHAS_DATA_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def initialize_class_folders(self, classes: List[Dict[str, Any]]):
        """Creates subdirectories for all 52 Hasthas."""
        for c in classes:
            folder = self.base_dir / c["folder_name"]
            folder.mkdir(parents=True, exist_ok=True)

    def get_class_dir(self, folder_name: str) -> Path:
        p = self.base_dir / folder_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_existing_samples(self, folder_name: str) -> List[Path]:
        """Returns sorted list of existing valid sample jpg files."""
        c_dir = self.get_class_dir(folder_name)
        pattern = re.compile(r"^sample_(\d+)\.jpe?g$", re.IGNORECASE)
        valid_samples = []
        for f in c_dir.iterdir():
            if f.is_file() and pattern.match(f.name):
                # Quick readability check
                valid_samples.append(f)
        valid_samples.sort(key=lambda p: int(pattern.match(p.name).group(1)))
        return valid_samples

    def get_sample_count(self, folder_name: str) -> int:
        return len(self.get_existing_samples(folder_name))

    def get_next_sample_path(self, folder_name: str) -> Tuple[Path, int]:
        """Finds next missing sample_XX.jpg number."""
        c_dir = self.get_class_dir(folder_name)
        existing = self.get_existing_samples(folder_name)
        existing_indices = set()
        pattern = re.compile(r"^sample_(\d+)\.jpe?g$", re.IGNORECASE)
        for p in existing:
            m = pattern.match(p.name)
            if m:
                existing_indices.add(int(m.group(1)))

        next_idx = 1
        while next_idx in existing_indices:
            next_idx += 1

        target_file = c_dir / f"sample_{next_idx:02d}.jpg"
        return target_file, next_idx

    def delete_last_sample(self, folder_name: str) -> Optional[Path]:
        """Deletes only the most recently captured sample in the class folder."""
        existing = self.get_existing_samples(folder_name)
        if not existing:
            return None
        last_file = existing[-1]
        try:
            last_file.unlink()
            return last_file
        except Exception as e:
            print(f"[ERROR] Failed to delete {last_file}: {e}")
            return None


# ==============================================================================
# 4. WEBCAM INTERFACE & HUD RENDERER
# ==============================================================================

class PersonalizedHasthaCollector:
    """
    Main interactive collector engine.
    """

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.classes = load_canonical_hasthas()
        self.dir_mgr = DatasetDirectoryManager()
        self.dir_mgr.initialize_class_folders(self.classes)

        self.checker = HandQualityChecker()
        self.cap: Optional[cv2.VideoCapture] = None

        # Camera info
        self.actual_width = 0
        self.actual_height = 0

        # Timing and Debounce
        self.last_capture_time = 0.0
        self.min_capture_interval = 0.5  # 0.5s debounce

        # Current State
        self.current_class_idx = 0
        self.waiting_for_next = False
        self.status_message = "Ready to capture"
        self.status_color = (200, 200, 200)
        self.status_timestamp = time.time()

        # Rejection tracking
        self.rejected_counts: Dict[str, int] = {c["class_name"]: 0 for c in self.classes}
        self.valid_capture_counts: Dict[str, int] = {c["class_name"]: 0 for c in self.classes}

        # Initialize existing sample counts and find resume position
        self.resume_from_existing()

    def resume_from_existing(self):
        """Scans directory and resumes from the first incomplete class."""
        for c in self.classes:
            cnt = self.dir_mgr.get_sample_count(c["folder_name"])
            self.valid_capture_counts[c["class_name"]] = cnt

        # Find first class < TARGET_SAMPLES_PER_CLASS
        resume_idx = 0
        for i, c in enumerate(self.classes):
            if self.valid_capture_counts[c["class_name"]] < TARGET_SAMPLES_PER_CLASS:
                resume_idx = i
                break
        else:
            resume_idx = len(self.classes) - 1

        self.current_class_idx = resume_idx

    def get_total_collected(self) -> int:
        return sum(self.dir_mgr.get_sample_count(c["folder_name"]) for c in self.classes)

    def print_collection_list(self):
        """Section 1: Prints the final 52-class collection list before starting."""
        print("=" * 80)
        print("CANONICAL 52 KUCHIPUDI HASTHA CLASSES FOR DATA COLLECTION")
        print("=" * 80)
        print(f"{'Ord':<4} {'Type':<26} {'Class ID':<10} {'Class Name':<26} {'Folder':<22} {'Collected':<10}")
        print("-" * 102)
        for c in self.classes:
            cnt = self.dir_mgr.get_sample_count(c["folder_name"])
            status = f"{cnt}/{TARGET_SAMPLES_PER_CLASS}"
            print(f"{c['order']:<4} {c['category_type']:<26} {c['class_id']:<10} {c['class_name']:<26} {c['folder_name']:<22} {status:<10}")
        print("=" * 102)
        total = self.get_total_collected()
        print(f"Total Collected So Far: {total} / {TOTAL_TARGET_SAMPLES} images")
        curr = self.classes[self.current_class_idx]
        print(f"Resuming at: #{curr['order']} {curr['class_name']} ({self.dir_mgr.get_sample_count(curr['folder_name'])}/{TARGET_SAMPLES_PER_CLASS})")
        print("=" * 102)

    def open_camera(self) -> bool:
        """Section 3: Open camera, request 1280x720, fallback gracefully."""
        print(f"[CAMERA] Attempting to open webcam (Device Index: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)

        if not self.cap.isOpened():
            # Try default backend if DSHOW fails
            self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera {self.camera_id}.")
            return False

        # Request 1280 x 720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Query actual resolution
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[CAMERA] Camera initialized successfully.")
        print(f"[CAMERA] Actual Camera Resolution: {self.actual_width} x {self.actual_height}")
        return True

    def draw_hud(
        self,
        preview_bgr: np.ndarray,
        mp_results: Optional[Any],
        hand_detected: bool
    ) -> np.ndarray:
        """
        Renders HUD with side info panel.
        Displays:
          PERSONALIZED HASHTA COLLECTION
          Current Hastha
          Progress: X / 20
          Overall: Y / 1040
          Instructions
          Controls
          Status Confirmation
        """
        curr = self.classes[self.current_class_idx]
        curr_count = self.dir_mgr.get_sample_count(curr["folder_name"])
        total_count = self.get_total_collected()

        h, w = preview_bgr.shape[:2]
        panel_w = 400
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)

        # Place camera preview
        canvas[:, :w] = preview_bgr

        # Draw hand landmarks on preview if detected
        if mp_results and mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    # Invert X because preview is mirrored horizontally
                    px = int((1.0 - lm.x) * w)
                    py = int(lm.y * h)
                    color = (0, 255, 0) if hand_detected else (0, 165, 255)
                    cv2.circle(canvas, (px, py), 4, color, -1)

        # Panel Background
        panel_x = w
        canvas[:, panel_x:] = (28, 28, 30)
        cv2.line(canvas, (panel_x, 0), (panel_x, h), (60, 60, 60), 2)

        def put_txt(text: str, y: int, font_scale: float = 0.55, color=(240, 240, 240), thickness: int = 1):
            cv2.putText(canvas, text, (panel_x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        # Title Header
        put_txt("-----------------------------------------", 35, 0.45, (140, 140, 140))
        put_txt("PERSONALIZED HASHTA COLLECTION", 60, 0.65, (0, 220, 255), 2)
        put_txt("-----------------------------------------", 85, 0.45, (140, 140, 140))

        # Current Hastha
        put_txt("Current Hastha:", 120, 0.55, (180, 180, 180))
        put_txt(f"{curr['clean_name']}", 150, 0.85, (255, 255, 255), 2)
        put_txt(f"[{curr['category_type']}]", 175, 0.45, (160, 200, 160))
        put_txt(f"Folder: {curr['folder_name']}", 195, 0.42, (120, 120, 120))

        # Progress
        put_txt("Progress:", 235, 0.55, (180, 180, 180))
        prog_color = (0, 255, 128) if curr_count >= TARGET_SAMPLES_PER_CLASS else (0, 220, 255)
        put_txt(f"{curr_count} / {TARGET_SAMPLES_PER_CLASS}", 265, 0.9, prog_color, 2)

        # Overall
        put_txt("Overall:", 305, 0.55, (180, 180, 180))
        put_txt(f"{total_count} / {TOTAL_TARGET_SAMPLES}", 335, 0.75, (220, 220, 220), 2)

        # Instructions
        put_txt("Instructions:", 375, 0.52, (180, 180, 180))
        put_txt("Hold the Hastha clearly.", 400, 0.45, (200, 200, 200))
        put_txt("Small natural position variations", 420, 0.42, (160, 160, 160))
        put_txt("are allowed (closer/farther/angles).", 438, 0.42, (160, 160, 160))

        # Controls
        put_txt("Controls:", 475, 0.52, (180, 180, 180))
        put_txt("SPACE = capture", 500, 0.48, (0, 255, 200), 1)
        put_txt("R     = retake/delete last sample", 522, 0.45, (255, 180, 80), 1)
        put_txt("N     = skip to next Hastha", 544, 0.45, (200, 200, 200), 1)
        put_txt("Q     = quit safely", 566, 0.45, (120, 120, 255), 1)

        # Status Banner at bottom of panel
        put_txt("-----------------------------------------", 605, 0.45, (140, 140, 140))
        put_txt("Status:", 630, 0.5, (180, 180, 180))
        put_txt(f"{self.status_message}", 655, 0.55, self.status_color, 2)

        # If class complete banner
        if curr_count >= TARGET_SAMPLES_PER_CLASS:
            # Draw semi-transparent completion banner over preview
            overlay = canvas.copy()
            next_name = self.classes[(self.current_class_idx + 1) % TOTAL_CLASSES]["clean_name"]
            cv2.rectangle(overlay, (40, h // 2 - 80), (w - 40, h // 2 + 80), (20, 80, 20), -1)
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
            cv2.rectangle(canvas, (40, h // 2 - 80), (w - 40, h // 2 + 80), (0, 255, 0), 2)
            cv2.putText(canvas, "CLASS COMPLETE (20/20)", (w // 2 - 220, h // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Next Hastha: {next_name}", (w // 2 - 180, h // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 255, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Press [ENTER] or [N] to continue", (w // 2 - 200, h // 2 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        return canvas

    def handle_capture(self, native_bgr: np.ndarray):
        """
        Section 5 & 6: Hand detection quality check and image saving.
        """
        now = time.time()
        if now - self.last_capture_time < self.min_capture_interval:
            return  # Cooldown active

        curr = self.classes[self.current_class_idx]
        curr_count = self.dir_mgr.get_sample_count(curr["folder_name"])
        if curr_count >= TARGET_SAMPLES_PER_CLASS:
            self.status_message = "Class already complete (20/20)"
            self.status_color = (0, 255, 0)
            return

        # Hand Detection Quality Check
        frame_rgb = cv2.cvtColor(native_bgr, cv2.COLOR_BGR2RGB)
        is_valid, msg, _ = self.checker.evaluate_frame(frame_rgb, curr["expected_hands"])

        if not is_valid:
            # Section 5: Invalid
            self.rejected_counts[curr["class_name"]] += 1
            self.status_message = "Hand not detected — try again"
            self.status_color = (0, 0, 255)
            print(f"[QUALITY CHECK] ✗ Invalid — hand not detected ({msg})")
            return

        # Section 6: Valid -> save JPEG and increment
        target_path, sample_num = self.dir_mgr.get_next_sample_path(curr["folder_name"])

        # Save native unmirrored frame
        success = cv2.imwrite(str(target_path), native_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            self.valid_capture_counts[curr["class_name"]] += 1
            new_cnt = self.dir_mgr.get_sample_count(curr["folder_name"])
            self.last_capture_time = now
            self.status_message = f"✓ Valid sample — Captured #{new_cnt:02d}"
            self.status_color = (0, 255, 0)
            print(f"[SAVED] ✓ Valid sample: {target_path.name} for {curr['class_name']} ({new_cnt}/{TARGET_SAMPLES_PER_CLASS})")

            # Check if class reached 20/20
            if new_cnt >= TARGET_SAMPLES_PER_CLASS:
                print("=" * 60)
                print("CLASS COMPLETE")
                print(f"Class: {curr['class_name']}")
                print(f"Collected: {new_cnt}/{TARGET_SAMPLES_PER_CLASS}")
                print(f"Valid: {new_cnt}")
                print(f"Rejected: {self.rejected_counts[curr['class_name']]}")
                next_idx = (self.current_class_idx + 1) % TOTAL_CLASSES
                print(f"Next Hastha: {self.classes[next_idx]['class_name']}")
                print("Waiting for ENTER or N to proceed...")
                print("=" * 60)
        else:
            self.status_message = "Error writing image file"
            self.status_color = (0, 0, 255)

    def handle_retake(self):
        """
        Section 7: Delete ONLY most recently captured sample for current class.
        """
        curr = self.classes[self.current_class_idx]
        del_path = self.dir_mgr.delete_last_sample(curr["folder_name"])
        if del_path:
            cnt = self.dir_mgr.get_sample_count(curr["folder_name"])
            self.status_message = f"Deleted {del_path.name} (Retake allowed: {cnt}/{TARGET_SAMPLES_PER_CLASS})"
            self.status_color = (0, 165, 255)
            print(f"[RETAKE] Deleted {del_path.name}. Progress: {cnt}/{TARGET_SAMPLES_PER_CLASS}")
        else:
            self.status_message = "No samples to retake"
            self.status_color = (180, 180, 180)

    def next_class(self):
        """Advances to next Hastha class safely."""
        self.current_class_idx = (self.current_class_idx + 1) % TOTAL_CLASSES
        curr = self.classes[self.current_class_idx]
        cnt = self.dir_mgr.get_sample_count(curr["folder_name"])
        self.status_message = f"Switched to {curr['clean_name']} ({cnt}/{TARGET_SAMPLES_PER_CLASS})"
        self.status_color = (255, 220, 100)
        print(f"[ADVANCE] Now collecting: #{curr['order']} {curr['class_name']} ({cnt}/{TARGET_SAMPLES_PER_CLASS})")

    def run(self):
        """
        Main execution loop.
        """
        self.print_collection_list()

        if not self.open_camera():
            print("[INFO] Generating reports from existing directory status and exiting safely.")
            self.generate_reports()
            self.print_final_terminal_output()
            return

        window_name = "Personalized Kuchipudi Hastha Collector"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        print("\nStarting live capture loop. Press 'Q' to quit and generate summary report.")

        try:
            while True:
                ret, frame_native = self.cap.read()
                if not ret or frame_native is None:
                    print("[WARNING] Failed to grab frame from camera.")
                    time.sleep(0.05)
                    continue

                curr = self.classes[self.current_class_idx]

                # Run hand detection on native unmirrored RGB frame
                frame_rgb = cv2.cvtColor(frame_native, cv2.COLOR_BGR2RGB)
                is_valid_hand, _, mp_results = self.checker.evaluate_frame(
                    frame_rgb, curr["expected_hands"]
                )

                # Mirror frame horizontally ONLY for live preview
                frame_preview = cv2.flip(frame_native, 1)

                # Compose HUD
                canvas = self.draw_hud(frame_preview, mp_results, is_valid_hand)
                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(15) & 0xFF

                # Controls
                if key == ord('q') or key == ord('Q'):
                    print("\n[USER] 'Q' pressed: Exiting safely...")
                    break
                elif key == 32:  # SPACE
                    self.handle_capture(frame_native)
                elif key == ord('r') or key == ord('R'):
                    self.handle_retake()
                elif key == ord('n') or key == ord('N') or key == 13:  # N or ENTER
                    curr_count = self.dir_mgr.get_sample_count(curr["folder_name"])
                    if curr_count >= TARGET_SAMPLES_PER_CLASS or key == ord('n') or key == ord('N'):
                        self.next_class()

        finally:
            self.close()
            self.generate_reports()
            self.print_final_terminal_output()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.checker.close()
        cv2.destroyAllWindows()

    def generate_reports(self):
        """
        Section 11: Generates collection report and statistics CSV.
        """
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total_collected = 0
        total_missing = 0
        complete_classes_count = 0
        incomplete_classes = []

        stats_rows = []

        for c in self.classes:
            collected = self.dir_mgr.get_sample_count(c["folder_name"])
            missing = max(0, TARGET_SAMPLES_PER_CLASS - collected)
            status = "COMPLETE" if collected >= TARGET_SAMPLES_PER_CLASS else "INCOMPLETE"

            total_collected += collected
            total_missing += missing
            if status == "COMPLETE":
                complete_classes_count += 1
            else:
                incomplete_classes.append((c["order"], c["class_name"], collected, missing))

            stats_rows.append({
                "class_id": c["class_id"],
                "class_name": c["class_name"],
                "target_samples": TARGET_SAMPLES_PER_CLASS,
                "collected_samples": collected,
                "missing_samples": missing,
                "completion_status": status
            })

        # 1. Write statistics CSV
        with open(STATS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "class_id",
                "class_name",
                "target_samples",
                "collected_samples",
                "missing_samples",
                "completion_status"
            ])
            writer.writeheader()
            writer.writerows(stats_rows)
        print(f"[REPORT] Statistics saved to: {STATS_CSV_PATH}")

        # 2. Write collection report TXT
        total_rejections = sum(self.rejected_counts.values())
        res_str = f"{self.actual_width} x {self.actual_height}" if self.actual_width > 0 else "Unavailable / Not initialized"

        lines = [
            "=" * 80,
            "STEP 32A — PERSONALIZED HASHTAS COLLECTION REPORT",
            "=" * 80,
            f"Date / Time                  : {now_str}",
            f"Camera Resolution            : {res_str}",
            f"Total Target Images          : {TOTAL_TARGET_SAMPLES}",
            f"Total Collected Images       : {total_collected}",
            f"Total Missing Images         : {total_missing}",
            f"Complete Classes (20/20)     : {complete_classes_count} / {TOTAL_CLASSES}",
            f"Incomplete Classes Count     : {len(incomplete_classes)}",
            f"Rejected Capture Attempts    : {total_rejections}",
            "",
            "STRICT PROTOCOL CONFIRMATIONS:",
            "  - Confirmation: v2_test.csv was NEVER accessed or modified: YES",
            "  - Confirmation: No model was trained or modified: YES",
            "  - Confirmation: Generic dataset untouched: YES",
            "  - Confirmation: Camera inference scripts untouched: YES",
            "",
            "=" * 80,
            "PER-CLASS COLLECTION BREAKDOWN:",
            "=" * 80,
            f"{'Ord':<4} {'Class ID':<10} {'Class Name':<28} {'Folder':<24} {'Collected':<10} {'Rejected':<10} {'Status':<12}",
            "-" * 102
        ]

        for c in self.classes:
            collected = self.dir_mgr.get_sample_count(c["folder_name"])
            rej = self.rejected_counts.get(c["class_name"], 0)
            status = "COMPLETE" if collected >= TARGET_SAMPLES_PER_CLASS else "INCOMPLETE"
            lines.append(
                f"{c['order']:<4} {c['class_id']:<10} {c['class_name']:<28} {c['folder_name']:<24} {collected:<10} {rej:<10} {status:<12}"
            )

        lines.extend([
            "-" * 102,
            f"TOTALS: Collected={total_collected}/{TOTAL_TARGET_SAMPLES} | Complete Classes={complete_classes_count}/{TOTAL_CLASSES}",
            "",
            "=" * 80,
            "INCOMPLETE CLASSES LIST:",
            "=" * 80
        ])

        if incomplete_classes:
            for ord_idx, cname, col, mis in incomplete_classes:
                lines.append(f"  * #{ord_idx:02d} {cname:<26}: Collected {col:02d}/20 (Missing {mis:02d})")
        else:
            lines.append("  All 52 classes are 100% complete!")

        lines.append("=" * 80)

        with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[REPORT] Collection report saved to: {REPORT_TXT_PATH}")

    def print_final_terminal_output(self):
        """
        Section 14: Final Terminal Output exact required format.
        """
        total_collected = self.get_total_collected()
        missing = max(0, TOTAL_TARGET_SAMPLES - total_collected)
        classes_complete = sum(
            1 for c in self.classes
            if self.dir_mgr.get_sample_count(c["folder_name"]) >= TARGET_SAMPLES_PER_CLASS
        )

        print("")
        print("========================================")
        print("STEP 32A COMPLETE")
        print("========================================")
        print("")
        print("Target:")
        print(f"{TOTAL_TARGET_SAMPLES} images")
        print("")
        print("Collected:")
        print(f"{total_collected}")
        print("")
        print("Missing:")
        print(f"{missing}")
        print("")
        print("Classes complete:")
        print(f"{classes_complete}/{TOTAL_CLASSES}")
        print("")
        print("Test set accessed:")
        print("NO")
        print("")
        print("Models modified:")
        print("NO")
        print("")
        print("Training performed:")
        print("NO")
        print("")
        print("Camera inference modified:")
        print("NO")
        print("")
        print("Next action:")
        print("WAIT FOR USER INSTRUCTION")
        print("")
        print("========================================")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step 32A — Personalized Hastha Webcam Data Collector")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--summary-only", action="store_true", help="Generate summary report and statistics without launching camera preview")
    args = parser.parse_args()

    collector = PersonalizedHasthaCollector(camera_id=args.camera_id)

    if args.summary_only:
        collector.print_collection_list()
        collector.generate_reports()
        collector.print_final_terminal_output()
    else:
        collector.run()


if __name__ == "__main__":
    main()
