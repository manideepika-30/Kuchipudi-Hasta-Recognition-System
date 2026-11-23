"""
STEP 28 — Standalone Rescue Inference Engine & Demonstration Application
========================================================================
Implements the rescue demonstration interface:
  - Loads the newly trained Step 28 rescue models from models/rescue_*
  - MediaPipe real-time anatomical landmark extraction
  - Direct Category Expert dispatch (Hasthas, Shiro, Drishti, Pada)
  - Step 26B Stability Gate (Window=10, MinStable=7, ConfThreshold=0.55)
  - Output: Exactly ONE gesture name once stabilized
  - Guarantees zero alteration to final_inference.py or Step 18 models
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical_model import build_hierarchical_mapping
from src.landmark_extraction import KuchipudiLandmarkPipeline
from src.train_hybrid_models import compute_temporal_statistics

# =============================================================================
# Categories Definition
# =============================================================================

CATEGORIES = [
    {
        "id": 1,
        "key": "1. Hasthas",
        "short_name": "HASTHAS",
        "desc": "Hand Gestures (52 Classes)",
        "gesture_type": "Hasta"
    },
    {
        "id": 2,
        "key": "2. Shiro Bhedhas (HEAD MOVEMENTS)",
        "short_name": "SHIRO BHEDHAS",
        "desc": "Head Movements (9 Classes)",
        "gesture_type": "Shiro Bhedha"
    },
    {
        "id": 3,
        "key": "3. Drishti Bhedhas(EYE MOVEMENTS)",
        "short_name": "DRISHTI BHEDHAS",
        "desc": "Eye Movements (8 Classes)",
        "gesture_type": "Drishti Bhedha"
    },
    {
        "id": 4,
        "key": "4. Pada Karma(Foot movements)",
        "short_name": "PADA KARMA",
        "desc": "Foot Movements (6 Classes)",
        "gesture_type": "Pada Karma"
    }
]

CATEGORY_BY_ID = {c["id"]: c for c in CATEGORIES}
CATEGORY_BY_KEY = {c["key"]: c for c in CATEGORIES}


# =============================================================================
# Step 26B / 28 Gesture Display & Stability Gate
# =============================================================================

class GestureDisplayGate:
    """
    Step 28 Stability & Display Gate.
    A gesture is ONLY displayed when:
      1. Required landmark is actively detected.
      2. Valid prediction exists.
      3. The same predicted class appears in at least `min_stable_frames`
         of the most recent `window_size` predictions.
      4. The average confidence across matching frames >= `min_confidence`.
      5. The current frame's prediction matches the dominant class.

    Immediately clears the recognized name when:
      - Required anatomical landmarks are lost.
      - User changes hand / body position.
      - Category changes.
    """

    def __init__(self, window_size: int = 10, min_stable_frames: int = 7, min_confidence: float = 0.55):
        self.window_size = window_size
        self.min_stable_frames = min_stable_frames
        self.min_confidence = min_confidence

        self.history = deque(maxlen=window_size)
        self.active_category_key = "1. Hasthas"
        self.current_recognized_name = None

    def reset(self, new_category_key: Optional[str] = None):
        """Immediately clears history and recognized state."""
        self.history.clear()
        self.current_recognized_name = None
        if new_category_key is not None:
            self.active_category_key = new_category_key

    def _get_missing_prompt(self, cat_key: str) -> str:
        if cat_key == "1. Hasthas":
            return "Show a hand"
        elif cat_key == "2. Shiro Bhedhas (HEAD MOVEMENTS)":
            return "Face camera directly"
        elif cat_key == "3. Drishti Bhedhas(EYE MOVEMENTS)":
            return "Face camera directly"
        elif cat_key == "4. Pada Karma(Foot movements)":
            return "Show your feet/legs"
        return "Show gesture clearly"

    def _get_hold_prompt(self, cat_key: str) -> str:
        if cat_key == "1. Hasthas":
            return "Hold the Hasta clearly"
        elif cat_key == "2. Shiro Bhedhas (HEAD MOVEMENTS)":
            return "Hold the Shiro movement clearly"
        elif cat_key == "3. Drishti Bhedhas(EYE MOVEMENTS)":
            return "Hold the Drishti clearly"
        elif cat_key == "4. Pada Karma(Foot movements)":
            return "Hold the Pada position clearly"
        return "Hold gesture clearly..."

    def update(self, raw_result: Optional[Dict[str, Any]], cat_key: str) -> Dict[str, Any]:
        self.active_category_key = cat_key

        if raw_result is None:
            self.history.clear()
            self.current_recognized_name = None
            return {
                "is_recognized": False,
                "recognized_name": None,
                "status_prompt": self._get_missing_prompt(cat_key)
            }

        detected = bool(raw_result.get("detected", False))
        is_valid = bool(raw_result.get("is_valid", False))
        is_collecting = bool(raw_result.get("is_collecting", False))
        raw_name = raw_result.get("gesture_name", "")
        conf = float(raw_result.get("confidence", 0.0)) / 100.0

        if not detected or not is_valid:
            self.history.clear()
            self.current_recognized_name = None
            return {
                "is_recognized": False,
                "recognized_name": None,
                "status_prompt": self._get_missing_prompt(cat_key)
            }

        if is_collecting:
            self.history.clear()
            self.current_recognized_name = None
            return {
                "is_recognized": False,
                "recognized_name": None,
                "status_prompt": "Collecting movement frames..."
            }

        self.history.append((raw_name, conf))

        class_counts = {}
        class_confs = {}
        for name, c in self.history:
            class_counts[name] = class_counts.get(name, 0) + 1
            class_confs.setdefault(name, []).append(c)

        best_class = None
        best_count = 0
        best_avg_conf = 0.0
        for name, count in class_counts.items():
            if count > best_count:
                best_count = count
                best_class = name
                best_avg_conf = sum(class_confs[name]) / count

        if (
            len(self.history) >= self.min_stable_frames
            and best_count >= self.min_stable_frames
            and best_avg_conf >= self.min_confidence
            and raw_name == best_class
        ):
            self.current_recognized_name = best_class
            return {
                "is_recognized": True,
                "recognized_name": best_class,
                "status_prompt": "RECOGNIZED"
            }

        self.current_recognized_name = None
        return {
            "is_recognized": False,
            "recognized_name": None,
            "status_prompt": self._get_hold_prompt(cat_key)
        }


# =============================================================================
# Rescue Inference Engine
# =============================================================================

class KuchipudiRescueEngine:
    """
    Direct Category Rescue Inference Engine.
    Executes the Step 28 rescue models trained exclusively on training features.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or (PROJECT_ROOT / "models")
        print("\n" + "=" * 80)
        print("INITIALIZING KUCHIPUDI RESCUE INFERENCE ENGINE (STEP 28)")
        print("=" * 80)

        self.mapping = build_hierarchical_mapping(save_json=False)

        # Load models and scalers
        print("  [1/3] Loading Step 28 Rescue Models...")
        self.m_hasthas = joblib.load(self.models_dir / "rescue_Hasthas_best.joblib")
        self.s_hasthas = joblib.load(self.models_dir / "rescue_Hasthas_scaler.joblib")

        self.m_shiro = joblib.load(self.models_dir / "rescue_Shiro_Bhedhas_best.joblib")
        self.s_shiro = joblib.load(self.models_dir / "rescue_Shiro_Bhedhas_scaler.joblib")

        self.m_drishti = joblib.load(self.models_dir / "rescue_Drishti_Bhedhas_best.joblib")
        self.s_drishti = joblib.load(self.models_dir / "rescue_Drishti_Bhedhas_scaler.joblib")

        self.m_pada = joblib.load(self.models_dir / "rescue_Pada_Karma_best.joblib")
        self.s_pada = joblib.load(self.models_dir / "rescue_Pada_Karma_scaler.joblib")

        # MediaPipe Pipeline
        print("  [2/3] Initializing MediaPipe Anatomical Detectors...")
        self.pipeline = KuchipudiLandmarkPipeline(static_image_mode=False)

        # Buffers
        print("  [3/3] Initializing Movement Buffers...")
        self.temporal_buffers = {c["key"]: deque(maxlen=16) for c in CATEGORIES}

        print("  --> RESCUE INFERENCE ENGINE READY.")

    def reset_category(self, cat_key: str):
        if cat_key in self.temporal_buffers:
            self.temporal_buffers[cat_key].clear()

    def close(self):
        if hasattr(self, "pipeline") and self.pipeline is not None:
            self.pipeline.close()

    def process_category(self, frame_bgr: np.ndarray, cat_key: str) -> Dict[str, Any]:
        cat_meta = CATEGORY_BY_KEY[cat_key]
        cat_short = cat_meta["short_name"]
        gesture_type = cat_meta["gesture_type"]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lm_res = self.pipeline.extract_from_frame(frame_rgb, cat_key)
        lm_vec = lm_res["vector"]
        detected = bool(lm_res.get("detected", False))

        cat_info = self.mapping["experts"][cat_key]
        idx_to_class = {item["local_class_id"]: item["class_name"] for item in cat_info["classes"]}
        num_classes = len(idx_to_class)

        if not detected:
            return {
                "category_key": cat_key,
                "category_short": cat_short,
                "gesture_type": gesture_type,
                "gesture_name": "Landmark not detected",
                "confidence": 0.0,
                "detected": False,
                "is_valid": False,
                "is_collecting": False
            }

        # Temporal buffering for movement categories
        is_collecting = False
        if cat_key != "1. Hasthas":
            buf = self.temporal_buffers[cat_key]
            buf.append(lm_vec)
            if len(buf) < 16:
                is_collecting = True
                seq = np.tile(lm_vec, (16, 1))
            else:
                seq = np.array(list(buf), dtype=np.float32)
            stat_vec = compute_temporal_statistics(seq)

        # Inference per category
        if cat_key == "1. Hasthas":
            x_in = self.s_hasthas.transform(np.expand_dims(lm_vec, axis=0))
            pred_id = int(self.m_hasthas.predict(x_in)[0])
            probs = self.m_hasthas.predict_proba(x_in)[0]
            conf = float(probs[pred_id]) * 100.0

        elif cat_key == "2. Shiro Bhedhas (HEAD MOVEMENTS)":
            feat_combined = np.hstack([lm_vec, stat_vec])
            x_in = self.s_shiro.transform(np.expand_dims(feat_combined, axis=0))
            pred_id = int(self.m_shiro.predict(x_in)[0])
            probs = self.m_shiro.predict_proba(x_in)[0]
            conf = float(probs[pred_id]) * 100.0

        elif cat_key == "3. Drishti Bhedhas(EYE MOVEMENTS)":
            feat_combined = np.hstack([lm_vec, stat_vec])
            x_in = self.s_drishti.transform(np.expand_dims(feat_combined, axis=0))
            pred_id = int(self.m_drishti.predict(x_in)[0])
            probs = self.m_drishti.predict_proba(x_in)[0]
            conf = float(probs[pred_id]) * 100.0

        else: # Pada Karma
            feat_combined = np.hstack([lm_vec, stat_vec])
            x_in = self.s_pada.transform(np.expand_dims(feat_combined, axis=0))
            pred_id = int(self.m_pada.predict(x_in)[0])
            probs = self.m_pada.predict_proba(x_in)[0]
            conf = float(probs[pred_id]) * 100.0

        gesture_name = idx_to_class.get(pred_id, f"Class {pred_id}")

        return {
            "category_key": cat_key,
            "category_short": cat_short,
            "gesture_type": gesture_type,
            "gesture_name": gesture_name,
            "confidence": conf,
            "detected": True,
            "is_valid": True,
            "is_collecting": is_collecting
        }


# =============================================================================
# Demonstration UI Runner
# =============================================================================

def run_rescue_demo(headless_frames: Optional[int] = None, min_confidence: float = 0.55):
    """
    Runs the live demonstration camera loop using the Step 28 rescue models and stability gate.
    """
    engine = KuchipudiRescueEngine()
    gate = GestureDisplayGate(window_size=10, min_stable_frames=7, min_confidence=min_confidence)

    active_cat_id = 1
    active_cat = CATEGORY_BY_ID[active_cat_id]
    gate.reset(active_cat["key"])

    if headless_frames is not None:
        print(f"\n[DEMO] Running in headless test mode for {headless_frames} frames...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for f_idx in range(headless_frames):
            res = engine.process_category(dummy_frame, active_cat["key"])
            gate_res = gate.update(res, active_cat["key"])
            if f_idx % 10 == 0:
                print(f"  Frame {f_idx:02d}: Gate Prompt='{gate_res['status_prompt']}', Recognized='{gate_res['recognized_name']}'")
        engine.close()
        print("[DEMO] Headless test complete.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open default camera (index 0).")
        engine.close()
        return

    print("\n" + "=" * 80)
    print("KUCHIPUDI RESCUE DEMONSTRATION ACTIVE")
    print("Controls: [1] Hasthas  [2] Shiro  [3] Drishti  [4] Pada  [Q] Quit")
    print("=" * 80)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame for intuitive interactive experience
            frame = cv2.flip(frame, 1)
            raw_res = engine.process_category(frame, active_cat["key"])
            gate_res = gate.update(raw_res, active_cat["key"])

            # UI Rendering
            h, w = frame.shape[:2]
            
            # Top Banner: Category and Key Selector
            cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
            header_text = f"ACTIVE: [{active_cat_id}] {active_cat['short_name']} | Keys: 1:Hasthas 2:Shiro 3:Drishti 4:Pada Q:Quit"
            cv2.putText(frame, header_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)

            # Bottom Panel: Stability Gate Output
            cv2.rectangle(frame, (0, h - 80), (w, h), (15, 15, 15), -1)
            
            if gate_res["is_recognized"]:
                # Recognized: Bright Green
                cv2.putText(frame, "RECOGNIZED", (20, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(frame, gate_res["recognized_name"], (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2)
            else:
                # Holding / Missing prompt: Amber / Light Gray
                prompt = gate_res["status_prompt"]
                cv2.putText(frame, prompt, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 180, 255), 2)

            cv2.imshow("Kuchipudi Gesture Recognition (Rescue Demo)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                new_cat_id = int(chr(key))
                if new_cat_id != active_cat_id:
                    active_cat_id = new_cat_id
                    active_cat = CATEGORY_BY_ID[active_cat_id]
                    engine.reset_category(active_cat["key"])
                    gate.reset(active_cat["key"])
                    print(f"\nSwitched to Category [{active_cat_id}]: {active_cat['short_name']}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()
        print("\nDemonstration session closed.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--headless":
        run_rescue_demo(headless_frames=30)
    else:
        run_rescue_demo()
