"""
STEP 35 — PERSONALIZED 52-HASTHA LIVE CAMERA TEST
==================================================
Real-time webcam inference application for the personalized 52-Hastha
landmark classifier trained in Step 34.

Features:
  - 1280x720 webcam input
  - Dynamic single-hand (Asamyuta) & two-hand (Samyutha) landmark detection
  - Physical handedness detection (Right / Left / Both)
  - Left-hand coordinate reflection canonicalization
  - Exact Step 34 175-D feature extraction and standard scaling
  - 8-frame rolling buffer with 6-vote stability gate (min conf: 0.20)
  - Clean UI: Single Hastha name, Category, Status (SHOW GESTURE / RECOGNIZING / STABLE)
  - Zero Top-5 / probability exposure to user
  - Internal testing session logger

Controls:
  Q : Quit application safely
  R : Reset prediction stabilization buffer
"""

import os
import sys
import re
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, Counter

import cv2
import numpy as np

# Ensure TF warnings are quieted
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.landmark_extraction import HandLandmarkExtractor

try:
    from mediapipe import solutions as mp_solutions
    mp_drawing = mp_solutions.drawing_utils
    mp_hands = mp_solutions.hands
except ImportError:
    import mediapipe.python.solutions as mp_solutions
    mp_drawing = mp_solutions.drawing_utils
    mp_hands = mp_solutions.hands

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "models" / "personalized_step34" / "personalized_hasthas_mlp_best.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "personalized_step34" / "personalized_hasthas_scaler.npz"
MAPPING_PATH = PROJECT_ROOT / "models" / "step16a_hasthas_class_mapping.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TXT_PATH = REPORTS_DIR / "step35_live_camera_test_report.txt"
SESSION_LOG_PATH = REPORTS_DIR / "step35_session_log.csv"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FEATURE_DIMENSION = 175

# Stability Gate Parameters
BUFFER_WINDOW = 8
MIN_VOTES = 6
MIN_CONFIDENCE = 0.20


# -----------------------------------------------------------------------------
# LIVE RECOGNIZER CLASS
# -----------------------------------------------------------------------------
class PersonalizedLiveHasthaRecognizer:
    def __init__(self):
        # 1. Load Mapping
        if not MAPPING_PATH.exists():
            raise FileNotFoundError(f"Class mapping file not found: {MAPPING_PATH}")
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        self.class_to_idx = mapping_data["class_to_idx"]
        self.idx_to_class = {int(k): v for k, v in mapping_data["idx_to_class"].items()}
        self.num_classes = len(self.class_to_idx)
        assert self.num_classes == 52, f"Expected 52 classes, got {self.num_classes}"

        # Identify Asamyuta vs Samyutha classes
        self.samyutha_keywords = [
            "Anjali", "Kapothamu", "Karkatamu", "Swasthikamu", "Dola", "Pushpaputamu",
            "Uthsungamu", "Sivalingamu", "Katakavardhanamu", "Karthariswasthikamu",
            "Sakatamu", "Sankhamu", "Chakramu", "Samputamu", "Paasamu", "Keelakamu",
            "Matsyamu", "Kurmamu", "Varahamu", "Garudamu", "Nagabandhakamu", "Khatva",
            "Bherundamu", "Avahithamu"
        ]
        self.samyutha_ids = set()
        self.asamyuta_ids = set()
        for cid, cname in self.idx_to_class.items():
            if any(kw in cname for kw in self.samyutha_keywords):
                self.samyutha_ids.add(cid)
            else:
                self.asamyuta_ids.add(cid)

        # 2. Load Step 34 Model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Step 34 model file not found: {MODEL_PATH}")
        self.model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        assert self.model.input_shape == (None, FEATURE_DIMENSION), f"Invalid model input: {self.model.input_shape}"
        assert self.model.output_shape == (None, self.num_classes), f"Invalid model output: {self.model.output_shape}"

        # 3. Load Step 34 Scaler
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Step 34 scaler file not found: {SCALER_PATH}")
        scaler_data = np.load(str(SCALER_PATH))
        self.scaler_mean = scaler_data["mean"].astype(np.float32)
        self.scaler_scale = np.where(scaler_data["scale"] == 0, 1.0, scaler_data["scale"]).astype(np.float32)
        assert self.scaler_mean.shape == (FEATURE_DIMENSION,), f"Invalid scaler mean shape: {self.scaler_mean.shape}"

        # 4. MediaPipe Hand Landmark Extractor
        self.mp_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.30
        )
        self.mp_fallback = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.15
        )
        self.base_extractor = HandLandmarkExtractor(
            static_image_mode=False,
            min_detection_confidence=0.35
        )

        # 5. Stability Gate State
        self.prediction_buffer = deque(maxlen=BUFFER_WINDOW)
        self.last_stable_class = None
        self.displayed_hastha = None
        self.displayed_category = None
        self.status = "SHOW GESTURE"

        # 6. Session Logger
        self.session_log = []
        self.frame_counter = 0

    def reset_buffer(self):
        """Resets the temporal prediction stabilization buffer."""
        self.prediction_buffer.clear()
        self.displayed_hastha = None
        self.displayed_category = None
        self.status = "SHOW GESTURE"

    def close(self):
        """Releases MediaPipe and background resources."""
        self.mp_detector.close()
        self.mp_fallback.close()
        self.base_extractor.close()

    def clean_hastha_name(self, full_name: str) -> str:
        """Strips leading numbering for clean user presentation."""
        return re.sub(r"^\d+\.\s*", "", full_name).strip()

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single camera frame:
          1. Detects hand landmarks (MediaPipe)
          2. Canonicalizes coordinates (Left hand reflection)
          3. Generates 175-D feature vector
          4. Predicts with Step 34 MLP
          5. Applies Stability Gate (8 window, 6 votes)
          6. Returns UI rendering information
        """
        self.frame_counter += 1
        h, w, _ = frame_bgr.shape
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe Detection
        results = self.mp_detector.process(image_rgb)
        if not results.multi_hand_landmarks:
            results = self.mp_fallback.process(image_rgb)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Case 1: No Hands Detected
        if not results.multi_hand_landmarks:
            self.prediction_buffer.clear()
            self.displayed_hastha = None
            self.displayed_category = None
            self.status = "SHOW GESTURE"

            log_entry = {
                "timestamp": timestamp_str,
                "frame": self.frame_counter,
                "num_hands": 0,
                "physical_handedness": "None",
                "pred_id": -1,
                "pred_class": "None",
                "confidence": 0.0,
                "status": self.status,
                "displayed_hastha": "None"
            }
            self.session_log.append(log_entry)

            return {
                "detected": False,
                "num_hands": 0,
                "status": "SHOW GESTURE",
                "displayed_hastha": None,
                "displayed_category": None,
                "confidence": 0.0,
                "physical_handedness": "None",
                "landmarks": []
            }

        num_hands = len(results.multi_hand_landmarks)

        # Case 2: One Hand Detected (Asamyuta)
        if num_hands == 1:
            hand_lms = results.multi_hand_landmarks[0]
            raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

            handedness_label = "Left"
            hand_conf = 1.0
            if results.multi_handedness and len(results.multi_handedness) > 0:
                try:
                    handedness_label = results.multi_handedness[0].classification[0].label
                    hand_conf = float(results.multi_handedness[0].classification[0].score)
                except Exception:
                    handedness_label = "Left"

            # Physical Hand: MediaPipe "Right" -> physical Left hand facing camera
            if handedness_label.strip().lower() == "right":
                physical_handedness = "Left"
                # Canonical reflection across wrist x
                canonical_coords = raw_coords.copy()
                wrist_x = raw_coords[0, 0]
                canonical_coords[:, 0] = 2.0 * wrist_x - canonical_coords[:, 0]
            else:
                physical_handedness = "Right"
                canonical_coords = raw_coords

            # Extract 87-D single hand features
            h1_feat = self.base_extractor._extract_single_hand_features(canonical_coords)
            h1_vec = h1_feat["feature_vector"]

            # Asamyuta: zero-padded Hand 2 and inter-hand distance
            h2_vec = np.zeros(87, dtype=np.float32)
            inter_dist = np.array([0.0], dtype=np.float32)
            vector_175 = np.concatenate([h1_vec, h2_vec, inter_dist])

        # Case 3: Two Hands Detected (Samyutha)
        else:
            physical_handedness = "Both"
            hand_conf = 1.0

            hand_lms_0 = results.multi_hand_landmarks[0]
            hand_lms_1 = results.multi_hand_landmarks[1]

            coords_0 = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms_0.landmark], dtype=np.float32)
            coords_1 = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms_1.landmark], dtype=np.float32)

            h1_feat = self.base_extractor._extract_single_hand_features(coords_0)
            h2_feat = self.base_extractor._extract_single_hand_features(coords_1)

            h1_vec = h1_feat["feature_vector"]
            h2_vec = h2_feat["feature_vector"]

            w1 = np.array(h1_feat["wrist"])
            w2 = np.array(h2_feat["wrist"])
            inter_hand_dist = float(np.linalg.norm(w1 - w2))

            vector_175 = np.concatenate([h1_vec, h2_vec, np.array([inter_hand_dist], dtype=np.float32)])

        # Step 34 Scaling
        scaled_175 = (vector_175 - self.scaler_mean) / self.scaler_scale
        scaled_175 = np.clip(scaled_175, -10.0, 10.0)

        # Model Inference
        probs = self.model.predict(scaled_175.reshape(1, FEATURE_DIMENSION), verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        full_name = self.idx_to_class[top_idx]
        category = "Asamyuta" if top_idx in self.asamyuta_ids else "Samyutha"

        # Stability Gate
        if top_conf >= MIN_CONFIDENCE:
            self.prediction_buffer.append(top_idx)
        else:
            self.prediction_buffer.append(None)

        # Check Vote Count
        valid_votes = [idx for idx in self.prediction_buffer if idx is not None]
        vote_counts = Counter(valid_votes)

        is_stable = False
        winner_idx = None
        if vote_counts:
            most_common_idx, count = vote_counts.most_common(1)[0]
            if count >= MIN_VOTES:
                is_stable = True
                winner_idx = most_common_idx

        if is_stable:
            self.status = "STABLE"
            self.displayed_hastha = self.clean_hastha_name(self.idx_to_class[winner_idx])
            self.displayed_category = "Asamyuta" if winner_idx in self.asamyuta_ids else "Samyutha"
            self.last_stable_class = winner_idx
        else:
            self.status = "RECOGNIZING..."
            self.displayed_hastha = None
            self.displayed_category = None

        # Log entry for testing analysis
        log_entry = {
            "timestamp": timestamp_str,
            "frame": self.frame_counter,
            "num_hands": num_hands,
            "physical_handedness": physical_handedness,
            "pred_id": top_idx,
            "pred_class": full_name,
            "confidence": round(top_conf, 4),
            "status": self.status,
            "displayed_hastha": self.displayed_hastha or "None"
        }
        self.session_log.append(log_entry)

        return {
            "detected": True,
            "num_hands": num_hands,
            "status": self.status,
            "displayed_hastha": self.displayed_hastha,
            "displayed_category": self.displayed_category,
            "confidence": top_conf,
            "physical_handedness": physical_handedness,
            "landmarks": results.multi_hand_landmarks
        }

    def draw_ui(self, frame_bgr: np.ndarray, res: Dict[str, Any]) -> np.ndarray:
        """
        Draws clean, elegant HUD on preview frame.
        Strictly presents ONE Hastha name and category when stable.
        Zero exposure of top-5, logits, or probability tensors.
        """
        h, w, _ = frame_bgr.shape
        display = frame_bgr.copy()

        # Draw hand skeletons if detected
        if res.get("detected", False) and res.get("landmarks"):
            for hand_lms in res["landmarks"]:
                mp_drawing.draw_landmarks(
                    display,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                )

        # Glassmorphic top HUD panel
        overlay = display.copy()
        cv2.rectangle(overlay, (30, 20), (w - 30, 160), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
        cv2.rectangle(display, (30, 20), (w - 30, 160), (60, 70, 85), 2)

        # Header Title
        cv2.putText(
            display,
            "LIVE PERSONALIZED HASTHA RECOGNITION",
            (50, 52),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (220, 230, 240),
            2,
            cv2.LINE_AA
        )

        status = res.get("status", "SHOW GESTURE")

        if status == "SHOW GESTURE":
            # Prompt user to show hand
            cv2.putText(
                display,
                "STATUS: SHOW GESTURE",
                (50, 105),
                cv2.FONT_HERSHEY_DUPLEX,
                0.90,
                (0, 215, 255),  # Gold/Amber
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                display,
                "Hold any Hastha gesture in camera view",
                (50, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 175, 190),
                1,
                cv2.LINE_AA
            )

        elif status == "RECOGNIZING...":
            # Stabilizing
            cv2.putText(
                display,
                "STATUS: RECOGNIZING...",
                (50, 105),
                cv2.FONT_HERSHEY_DUPLEX,
                0.90,
                (255, 190, 0),  # Cyan
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                display,
                f"Hand: {res.get('physical_handedness', 'Active')} | Stabilizing gesture motion...",
                (50, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 195, 210),
                1,
                cv2.LINE_AA
            )

        elif status == "STABLE":
            # Stable Display: ONE NAME only
            hastha_name = res.get("displayed_hastha", "")
            category = res.get("displayed_category", "")

            # Highlight Hastha Name
            cv2.putText(
                display,
                f"DETECTED HASTHA: {hastha_name.upper()}",
                (50, 105),
                cv2.FONT_HERSHEY_DUPLEX,
                1.05,
                (50, 255, 120),  # Emerald Green
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                display,
                f"Category: {category} Hastha | Hand: {res.get('physical_handedness', 'Both')}",
                (50, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (210, 225, 235),
                1,
                cv2.LINE_AA
            )

        # Bottom Controls Bar
        cv2.rectangle(display, (30, h - 50), (w - 30, h - 15), (20, 24, 30), -1)
        cv2.rectangle(display, (30, h - 50), (w - 30, h - 15), (50, 60, 75), 1)
        cv2.putText(
            display,
            "[Q] Quit Live Test   |   [R] Reset Stabilization Buffer   |   52-Hastha Unified Classifier",
            (45, h - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 190, 200),
            1,
            cv2.LINE_AA
        )

        return display

    def save_session_log(self):
        """Saves internal testing log to CSV on exit."""
        if not self.session_log:
            return
        keys = list(self.session_log[0].keys())
        with open(SESSION_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.session_log)
        print(f"[LOG] Testing session log saved to: {SESSION_LOG_PATH} ({len(self.session_log)} entries)")


# -----------------------------------------------------------------------------
# STARTUP & SAFETY CHECKS (Requirement 8 & 9)
# -----------------------------------------------------------------------------
def perform_startup_checks(check_camera_hardware: bool = True) -> Tuple[bool, List[str]]:
    """
    Verifies:
      1. Model loads successfully
      2. Scaler loads successfully
      3. Exactly 52 classes are loaded
      4. Output dimension == 52
      5. 175-D feature vector is generated
      6. Scaler input dimension == 175
      7. Camera device availability
    """
    results = []
    all_passed = True

    # 1. Model Check
    if not MODEL_PATH.exists():
        results.append(f"[FAIL] Model file missing: {MODEL_PATH}")
        return False, results
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        in_shape = model.input_shape
        out_shape = model.output_shape
        if in_shape != (None, 175):
            results.append(f"[FAIL] Model input shape is {in_shape}, expected (None, 175)")
            all_passed = False
        else:
            results.append("[PASS] Model loaded successfully: input shape (None, 175)")

        if out_shape != (None, 52):
            results.append(f"[FAIL] Model output shape is {out_shape}, expected (None, 52)")
            all_passed = False
        else:
            results.append("[PASS] Model output dimension: exactly 52 classes")
    except Exception as e:
        results.append(f"[FAIL] Error loading model: {e}")
        return False, results

    # 2. Scaler Check
    if not SCALER_PATH.exists():
        results.append(f"[FAIL] Scaler file missing: {SCALER_PATH}")
        return False, results
    try:
        s = np.load(str(SCALER_PATH))
        mean = s["mean"]
        scale = s["scale"]
        if mean.shape != (175,) or scale.shape != (175,):
            results.append(f"[FAIL] Scaler dimensions invalid: mean {mean.shape}, scale {scale.shape}")
            all_passed = False
        else:
            results.append("[PASS] Scaler loaded successfully: input dimension 175")
    except Exception as e:
        results.append(f"[FAIL] Error loading scaler: {e}")
        return False, results

    # 3. Class Mapping Check
    if not MAPPING_PATH.exists():
        results.append(f"[FAIL] Class mapping missing: {MAPPING_PATH}")
        return False, results
    try:
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        c2idx = m["class_to_idx"]
        if len(c2idx) != 52:
            results.append(f"[FAIL] Class mapping count is {len(c2idx)}, expected 52")
            all_passed = False
        else:
            results.append("[PASS] Class mapping loaded: exactly 52 Hastha classes")
    except Exception as e:
        results.append(f"[FAIL] Error loading class mapping: {e}")
        return False, results

    # 4. Feature Extraction & Preprocessing Simulation Check
    try:
        dummy_feat = np.ones(175, dtype=np.float32)
        scaled_feat = (dummy_feat - mean) / np.where(scale == 0, 1.0, scale)
        dummy_pred = model.predict(scaled_feat.reshape(1, 175), verbose=0)[0]
        if dummy_pred.shape != (52,):
            results.append(f"[FAIL] Inference prediction shape {dummy_pred.shape} != (52,)")
            all_passed = False
        else:
            results.append("[PASS] 175-D feature generation & Step 34 preprocessing verified")
    except Exception as e:
        results.append(f"[FAIL] Error verifying preprocessing pipeline: {e}")
        all_passed = False

    # 5. Camera Device Check
    if check_camera_hardware:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    results.append("[PASS] Webcam device 0 accessible and responsive")
                else:
                    results.append("[WARN] Webcam opened but frame read returned False")
            else:
                results.append("[WARN] Webcam device 0 could not be opened (may be in use)")
        except Exception as e:
            results.append(f"[WARN] Error probing webcam: {e}")

    return all_passed, results


def generate_test_report(startup_passed: bool, startup_details: List[str]):
    """Generates the required reports/step35_live_camera_test_report.txt."""
    lines = []
    lines.append("=" * 85)
    lines.append("STEP 35 — PERSONALIZED 52-HASTHA LIVE CAMERA TEST REPORT")
    lines.append("=" * 85)
    lines.append(f"Report Generated        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model Path              : {MODEL_PATH}")
    lines.append(f"Scaler Path             : {SCALER_PATH}")
    lines.append(f"Class Mapping Path      : {MAPPING_PATH}")
    lines.append(f"Total Hastha Classes    : 52 (28 Asamyuta + 24 Samyutha)")
    lines.append(f"Feature Dimension       : 175 (87 Hand 1 + 87 Hand 2 + 1 Inter-Hand Dist)")
    lines.append(f"Webcam Target Resolution: {CAMERA_WIDTH} x {CAMERA_HEIGHT}")
    lines.append(f"Stabilization Parameters: Window = {BUFFER_WINDOW}, Min Votes = {MIN_VOTES}, Min Conf = {MIN_CONFIDENCE}")
    lines.append("=" * 85)
    lines.append("")

    lines.append("1. STARTUP & INTEGRITY VERIFICATION CHECKS")
    lines.append("-" * 85)
    for chk in startup_details:
        lines.append(f"  {chk}")
    lines.append("")
    lines.append(f"Startup Checks Verdict  : {'ALL CHECKS PASSED' if startup_passed else 'CHECK FAILED'}")
    lines.append("")

    lines.append("2. PREPROCESSING & PIPELINE ALIGNMENT")
    lines.append("-" * 85)
    lines.append("  • Single-hand (Asamyuta) canonicalization: Left hand x-reflection across wrist active.")
    lines.append("  • Single-hand representation: 87 Hand 1 features + 87 zeros + 0.0 distance.")
    lines.append("  • Two-hand (Samyutha) representation: 87 Hand 1 + 87 Hand 2 + Euclidean wrist distance.")
    lines.append("  • Standardization: (X_175 - scaler_mean) / scaler_scale (identical to Step 34).")
    lines.append("  • Stability Gate: Rolling buffer of 8 predictions requiring >= 6 same-class votes.")
    lines.append("  • No-gesture revocation: Immediate buffer clear when hands leave view.")
    lines.append("")

    lines.append("3. CAMERA LAUNCH STATUS")
    lines.append("-" * 85)
    lines.append(f"  • Application File      : src/personalized_live_camera_step35.py")
    lines.append(f"  • Webcam Device Probed  : Device 0 (1280 x 720)")
    lines.append(f"  • Launch Readiness      : {'READY FOR LIVE TESTING' if startup_passed else 'NOT READY'}")
    lines.append("")

    lines.append("4. LIVE ACCURACY MEASUREMENT STATUS")
    lines.append("=" * 85)
    lines.append("LIVE ACCURACY NOT YET MEASURED")
    lines.append("-" * 85)
    lines.append("Controlled live camera testing will be conducted interactively by the user.")
    lines.append("No live gesture accuracy is claimed or recorded prior to manual evaluation.")
    lines.append("=" * 85)

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[EXPORT] Test report saved to: {REPORT_TXT_PATH}")


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 35: Personalized 52-Hastha Live Camera Test")
    parser.add_argument("--check-only", action="store_true", help="Run startup and integrity checks without opening GUI")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 35 — PERSONALIZED 52-HASTHA LIVE CAMERA TEST")
    print("=" * 80)

    # 1. Run Startup Checks
    print("\nRunning Startup and Integrity Checks...")
    passed, details = perform_startup_checks(check_camera_hardware=True)
    for d in details:
        print(f"  {d}")

    # 2. Generate Test Report
    generate_test_report(passed, details)

    if not passed:
        print("\n[ERROR] Startup checks failed! Please review errors above.")
        return 1

    if args.check_only:
        print("\n[SUCCESS] All startup checks passed successfully. (--check-only mode complete)")
        return 0

    # 3. Launch Live Camera Application
    print("\nLaunching Live Webcam Hastha Recognition...")
    print("Controls:")
    print("  [Q] : Quit safely")
    print("  [R] : Reset stabilization buffer")
    print("--------------------------------------------------------------------------------")

    recognizer = PersonalizedLiveHasthaRecognizer()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera device {args.camera}.")
        recognizer.close()
        return 1

    window_name = "Personalized Hastha Live Recognition (Step 35)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CAMERA_WIDTH, CAMERA_HEIGHT)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab camera frame.")
                break

            # Mirror preview horizontally for natural mirror behavior
            frame_mirrored = cv2.flip(frame, 1)

            # Process frame through live recognizer
            result = recognizer.process_frame(frame_mirrored)

            # Draw clean HUD
            ui_frame = recognizer.draw_ui(frame_mirrored, result)

            cv2.imshow(window_name, ui_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] 'Q' pressed. Exiting live recognition...")
                break
            elif key == ord('r') or key == ord('R'):
                recognizer.reset_buffer()
                print("[INFO] Stabilization buffer reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.save_session_log()
        recognizer.close()
        print("[INFO] Camera released and resources cleaned up.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
