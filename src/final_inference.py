"""
STEP 24 — FINAL USER-FACING KUCHIPUDI RECOGNITION APPLICATION
=============================================================
Project: Kuchipudi-Hasta-Recognition-System

Category-Selection Based Kuchipudi Classical Dance Recognition System:
  1. Four-Category Selection:
     - User selects 1 of 4 categories: Hasthas, Shiro Bhedhas, Drishti Bhedhas, Pada Karma.
     - Direct expert model execution (Category Router is NOT run after category selection).
  2. High-Resolution Mirror Camera Display:
     - User-facing preview is horizontally mirrored (left hand appears on left).
     - Display frame is decoupled from inference frame (model receives native orientation).
     - Camera capture requested at 1280x720 for crisp preview.
  3. Decoupled Threading Architecture:
     - Dedicated background camera capture thread (zero lag, drops stale frames).
     - Asynchronous inference worker thread (10-15 FPS inference without blocking UI).
     - Main UI rendering loop at smooth 30+ FPS display refresh rate.
  4. Robust Landmark Handling:
     - Hasthas: 80% Hybrid (Step 18) + 20% Landmark MLP (Step 16A). Hand detection badge.
     - Shiro Bhedhas: S_07 Compact Multimodal (Step 18). Temporal accumulation banner.
     - Drishti Bhedhas: D_02 Multimodal Ensemble (Step 18). Displays "Face/Eyes not detected" if mesh fails.
     - Pada Karma: Hybrid Multimodal (Step 18). Displays "Pose/legs not detected" if legs fail.
  5. Interactive Controls & Clean HUD:
     - Keys [1], [2], [3], [4] or clickable on-screen buttons to switch categories.
     - Key [S] toggles prediction smoothing / display stabilization.
     - Key [M] returns to Start Screen Menu.
     - Keys [Q] or ESC cleanly release all camera and detector resources.

NOTE: This is a deployment/UI improvement only. Zero model training or test-set tuning is performed.
"""

import os
import sys
import time
import argparse
import threading
from pathlib import Path
from collections import deque
from typing import Dict, Any, Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical_model import build_hierarchical_mapping
from src.landmark_extraction import KuchipudiLandmarkPipeline
from src.train_hybrid_models import build_cnn_feature_extractor, compute_temporal_statistics
from src.preprocessing import resize_with_aspect_ratio_pad


# Category Definitions
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


class KuchipudiInferenceEngine:
    """
    Direct Category Expert Inference Engine.
    Executes ONLY the expert model for the user's chosen category (no router call).
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or (PROJECT_ROOT / "models")
        print("\n" + "=" * 80)
        print("INITIALIZING KUCHIPUDI DIRECT CATEGORY INFERENCE ENGINE (STEP 24)")
        print("=" * 80)

        # 1. Load Hierarchical Mapping
        self.mapping = build_hierarchical_mapping(save_json=False)

        # 2. Load Visual Feature Extractors for each category (MobileNetV3Large global pooling)
        print("  [1/5] Initializing Visual Feature Extractors...")
        self.vis_extractors = {
            "1. Hasthas": build_cnn_feature_extractor(
                self.models_dir / "v2_hierarchical_hasthas_best.keras", "Hasthas_global_pool"
            ),
            "2. Shiro Bhedhas (HEAD MOVEMENTS)": build_cnn_feature_extractor(
                self.models_dir / "v2_hierarchical_shiro_best.keras", "Shiro_Bhedhas_global_pool"
            ),
            "3. Drishti Bhedhas(EYE MOVEMENTS)": build_cnn_feature_extractor(
                self.models_dir / "v2_hierarchical_drishti_best.keras", "Drishti_Bhedhas_global_pool"
            ),
            "4. Pada Karma(Foot movements)": build_cnn_feature_extractor(
                self.models_dir / "v2_hierarchical_pada_karma_best.keras", "Pada_Karma_global_pool"
            )
        }

        # 3. Load Step 18 Category Expert Models
        print("  [2/5] Loading Step 18 Category Expert Models...")
        self.m_hasthas = tf.keras.models.load_model(str(self.models_dir / "step18_hasthas_optimized_best.keras"))
        self.m_hasthas_lm = tf.keras.models.load_model(str(self.models_dir / "step16a_hasthas_landmark_mlp_best.keras"))
        self.m_shiro = tf.keras.models.load_model(str(self.models_dir / "step18_shiro_optimized_best.keras"))
        self.m_drishti = tf.keras.models.load_model(str(self.models_dir / "step18_drishti_optimized_best.keras"))
        self.m_pada = tf.keras.models.load_model(str(self.models_dir / "step18_pada_optimized_best.keras"))

        # 4. Load Normalization Scalers
        print("  [3/5] Loading Normalization Scalers...")
        self.s_h17 = np.load(str(self.models_dir / "step17_hasthas_scaler.npz"))
        self.s_h16 = np.load(str(self.models_dir / "step16a_hasthas_scaler.npz"))
        self.s_s18 = np.load(str(self.models_dir / "step18_shiro_scalers.npz"))
        self.s_d18 = np.load(str(self.models_dir / "step18_drishti_scalers.npz"))
        self.s_p17 = np.load(str(self.models_dir / "step17_pada_scaler.npz"))

        # 5. MediaPipe Detectors
        print("  [4/5] Initializing MediaPipe Anatomical Detectors...")
        self.pipeline = KuchipudiLandmarkPipeline(static_image_mode=False)

        # 6. Temporal and Smoothing State
        print("  [5/5] Initializing Temporal and Smoothing Buffers...")
        self.temporal_buffers = {c["key"]: deque(maxlen=16) for c in CATEGORIES}
        self.smoothing_buffers = {c["key"]: deque(maxlen=5) for c in CATEGORIES}

        self._warmup()
        print("  --> INFERENCE ENGINE READY (DIRECT CATEGORY MODE).")

    def _warmup(self):
        """Warm up graph execution to eliminate first-frame latency."""
        dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.vis_extractors["1. Hasthas"](dummy_img, training=False)
        _ = self.m_hasthas([np.zeros((1, 960), dtype=np.float32), np.zeros((1, 175), dtype=np.float32)], training=False)
        _ = self.m_hasthas_lm(np.zeros((1, 175), dtype=np.float32), training=False)
        _ = self.m_shiro(np.zeros((1, 1216), dtype=np.float32), training=False)
        _ = self.m_drishti(np.zeros((1, 1356), dtype=np.float32), training=False)
        _ = self.m_pada(np.zeros((1, 1320), dtype=np.float32), training=False)

    def reset_category(self, cat_key: str):
        """Clears temporal and smoothing state when switching category."""
        if cat_key in self.temporal_buffers:
            self.temporal_buffers[cat_key].clear()
        if cat_key in self.smoothing_buffers:
            self.smoothing_buffers[cat_key].clear()

    def close(self):
        """Cleanly releases MediaPipe resources."""
        if hasattr(self, "pipeline") and self.pipeline is not None:
            self.pipeline.close()

    def process_category(
        self,
        frame_bgr: np.ndarray,
        cat_key: str,
        apply_smoothing: bool = True
    ) -> Dict[str, Any]:
        """
        Executes DIRECT inference using ONLY the selected category expert model.
        Does NOT invoke the category router.
        """
        cat_meta = CATEGORY_BY_KEY[cat_key]
        cat_short = cat_meta["short_name"]
        gesture_type = cat_meta["gesture_type"]

        # Step 1: Preprocess Frame for CNN (224x224 RGB, Aspect-Ratio Preserving Letterbox, normalized to [0, 1])
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        padded_pil = resize_with_aspect_ratio_pad(pil_frame, target_size=(224, 224), fill_color=(0, 0, 0))
        img_arr = np.array(padded_pil, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_arr, axis=0)

        # Step 2: Extract Category-Specific Anatomical Landmarks
        lm_res = self.pipeline.extract_from_frame(frame_rgb, cat_key)
        lm_vec = lm_res["vector"]
        detected = bool(lm_res.get("detected", False))

        cat_info = self.mapping["experts"][cat_key]
        idx_to_class = {item["local_class_id"]: item["class_name"] for item in cat_info["classes"]}

        # Category-Specific Safeguards for Missing Landmarks
        if cat_key == "3. Drishti Bhedhas(EYE MOVEMENTS)" and not detected:
            return {
                "category_key": cat_key,
                "category_short": cat_short,
                "gesture_type": gesture_type,
                "gesture_name": "Face/Eyes not detected",
                "full_prediction_label": f"{cat_short} - Face/Eyes not detected",
                "confidence": 0.0,
                "raw_confidence": 0.0,
                "detected": False,
                "status_message": "Face/Eyes not detected - please face the camera directly",
                "top5": [],
                "is_valid": False
            }

        if cat_key == "4. Pada Karma(Foot movements)" and not detected:
            return {
                "category_key": cat_key,
                "category_short": cat_short,
                "gesture_type": gesture_type,
                "gesture_name": "Pose/legs not detected",
                "full_prediction_label": f"{cat_short} - Pose/legs not detected",
                "confidence": 0.0,
                "raw_confidence": 0.0,
                "detected": False,
                "status_message": "Pose/legs not detected - step back so feet/legs are visible",
                "top5": [],
                "is_valid": False
            }

        # Status Message Handling
        if cat_key == "1. Hasthas":
            if detected:
                n_hands = lm_res.get("num_hands", 1)
                status_msg = f"Hand detected ({n_hands} hand{'s' if n_hands > 1 else ''})"
            else:
                status_msg = "Hand not detected (hold hand clearly in frame)"
        elif cat_key == "2. Shiro Bhedhas (HEAD MOVEMENTS)":
            status_msg = "Head / Neck tracking active" if detected else "Head/Pose not detected"
        elif cat_key == "3. Drishti Bhedhas(EYE MOVEMENTS)":
            status_msg = "Face & Eye mesh active"
        else:
            status_msg = "Lower body pose active"

        # Step 3: Temporal Buffer Update for Movement Categories (Shiro, Drishti, Pada)
        stat_vec = np.zeros(0, dtype=np.float32)
        is_collecting = False
        if cat_key != "1. Hasthas":
            buf = self.temporal_buffers[cat_key]
            buf.append(lm_vec)
            if len(buf) < 16:
                is_collecting = True
                status_msg = f"Collecting frames ({len(buf)}/16)..."
                seq = np.tile(lm_vec, (16, 1))
            else:
                seq = np.array(list(buf), dtype=np.float32)
            stat_vec = compute_temporal_statistics(seq)

        # Step 4: Visual Backbone Feature Extraction
        vis_emb = self.vis_extractors[cat_key](img_tensor, training=False).numpy()[0]

        # Step 5: Route Directly to Selected Category Expert Model
        if cat_key == "1. Hasthas":
            vis_in = np.expand_dims(vis_emb, axis=0)
            lm_s17 = (lm_vec - self.s_h17["mean"]) / self.s_h17["scale"]
            lm_s16 = (lm_vec - self.s_h16["mean"]) / self.s_h16["scale"]

            p_h17 = self.m_hasthas([vis_in, np.expand_dims(lm_s17, 0)], training=False).numpy()[0]
            p_h16 = self.m_hasthas_lm(np.expand_dims(lm_s16, 0), training=False).numpy()[0]
            raw_probs = 0.80 * p_h17 + 0.20 * p_h16

        elif cat_key == "2. Shiro Bhedhas (HEAD MOVEMENTS)":
            vis_scaled = (vis_emb - self.s_s18["vis_mean"]) / self.s_s18["vis_scale"]
            temp_scaled = (stat_vec - self.s_s18["temp_mean"]) / self.s_s18["temp_scale"]
            x_in = np.expand_dims(np.hstack([vis_scaled, temp_scaled]), axis=0)
            raw_probs = self.m_shiro(x_in, training=False).numpy()[0]

        elif cat_key == "3. Drishti Bhedhas(EYE MOVEMENTS)":
            vis_scaled = (vis_emb - self.s_d18["vis_mean"]) / self.s_d18["vis_scale"]
            nonvis_raw = np.hstack([lm_vec, stat_vec])
            nonvis_scaled = (nonvis_raw - self.s_d18["nonvis_mean"]) / self.s_d18["nonvis_scale"]
            x_in = np.expand_dims(np.hstack([vis_scaled, nonvis_scaled]), axis=0)
            raw_probs = self.m_drishti(x_in, training=False).numpy()[0]

        else:  # 4. Pada Karma(Foot movements)
            nonvis_raw = np.hstack([lm_vec, stat_vec])
            nonvis_scaled = (nonvis_raw - self.s_p17["mean"]) / self.s_p17["scale"]
            x_in = np.expand_dims(np.hstack([vis_emb, nonvis_scaled]), axis=0)
            raw_probs = self.m_pada(x_in, training=False).numpy()[0]

        raw_pred_id = int(np.argmax(raw_probs))
        raw_conf = float(raw_probs[raw_pred_id])
        raw_name = idx_to_class[raw_pred_id]

        # Step 6: Prediction Smoothing (Display Stabilization)
        s_buf = self.smoothing_buffers[cat_key]
        s_buf.append(raw_probs)

        if apply_smoothing and len(s_buf) > 1:
            disp_probs = np.mean(list(s_buf), axis=0)
            pred_id = int(np.argmax(disp_probs))
            disp_conf = float(disp_probs[pred_id])
            pred_name = idx_to_class[pred_id]
        else:
            disp_probs = raw_probs
            pred_id = raw_pred_id
            disp_conf = raw_conf
            pred_name = raw_name

        full_pred_label = f"{cat_short} - {pred_name}"

        # Top-5 Predictions
        top5_ids = np.argsort(disp_probs)[-5:][::-1]
        top5_list = [
            (idx_to_class[cid], float(disp_probs[cid]) * 100.0)
            for cid in top5_ids
        ]

        return {
            "category_key": cat_key,
            "category_short": cat_short,
            "gesture_type": gesture_type,
            "gesture_name": pred_name,
            "full_prediction_label": full_pred_label,
            "confidence": disp_conf * 100.0,
            "raw_confidence": raw_conf * 100.0,
            "detected": detected,
            "status_message": status_msg,
            "top5": top5_list,
            "is_valid": True,
            "is_collecting": is_collecting
        }


# =============================================================================
# High-Resolution & Low-Latency Video Capture Thread
# =============================================================================

def load_mock_frames_by_category() -> Dict[str, List[np.ndarray]]:
    """Loads representative frames from each category for demo/test mode."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_df = pd.read_csv(splits_dir / "v2_train.csv")

    cat_frames = {}
    for cat_meta in CATEGORIES:
        ckey = cat_meta["key"]
        cat_df = train_df[train_df["category"] == ckey].head(15)
        frames = []
        for _, row in cat_df.iterrows():
            img_p = PROJECT_ROOT / row["sample_path"]
            if img_p.exists():
                img = cv2.imread(str(img_p))
                if img is not None:
                    frames.append(img)
        if len(frames) == 0:
            frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
        cat_frames[ckey] = frames
    return cat_frames


class CameraCaptureThread:
    """
    Decoupled camera capture thread that continuously grabs frames
    and stores the single latest frame with zero queue delay.
    """

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720, use_mock: bool = False):
        self.src = src
        self.req_w = width
        self.req_h = height
        self.use_mock = use_mock
        self.lock = threading.Lock()
        self.stopped = False
        self.latest_frame = None
        self.grabbed = False
        self.actual_w = width
        self.actual_h = height
        self.cap = None

        self.mock_dict = None
        self.mock_idx = 0
        self.active_category_key = CATEGORIES[0]["key"]

        if not self.use_mock:
            print(f"\n[CAMERA] Opening webcam device #{src} (requesting {width}x{height})...")
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                print(f"[WARNING] Webcam #{src} could not be opened. Falling back to simulated stream.")
                self.use_mock = True
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
                self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
                print(f"  --> Camera initialized successfully at {self.actual_w}x{self.actual_h}.")
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.latest_frame = frame
                    self.grabbed = True

        if self.use_mock:
            print("  --> Initializing simulated dance video frames...")
            self.mock_dict = load_mock_frames_by_category()
            cur_list = self.mock_dict[self.active_category_key]
            self.latest_frame = cur_list[0]
            self.grabbed = True
            self.actual_w = self.latest_frame.shape[1]
            self.actual_h = self.latest_frame.shape[0]

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def set_mock_category(self, cat_key: str):
        with self.lock:
            self.active_category_key = cat_key
            self.mock_idx = 0

    def _update(self):
        while not self.stopped:
            if not self.use_mock and self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.grabbed = True
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.033)  # ~30 FPS simulated feed
                with self.lock:
                    if self.mock_dict and self.active_category_key in self.mock_dict:
                        frame_list = self.mock_dict[self.active_category_key]
                        self.latest_frame = frame_list[self.mock_idx % len(frame_list)]
                        self.mock_idx += 1
                        self.grabbed = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return self.grabbed, self.latest_frame.copy()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


# =============================================================================
# Step 26 — Inference-Time Display & Stability Gate
# =============================================================================

class GestureDisplayGate:
    """
    Step 26B — Final Demo Mode Display & Stability Gate.
    Stability-first gate preventing flickering or premature class names.

    A gesture is ONLY displayed when:
      1. Required landmark is actively detected.
      2. Valid prediction exists.
      3. The same predicted class appears in at least `min_stable_frames`
         of the most recent `window_size` valid predictions (e.g. >= 6 of 8).
      4. The average confidence across those matching frames >= `min_confidence` (e.g. >= 0.20).
      5. The current frame's prediction matches the dominant class (guarantees immediate removal on movement).

    Immediately clears the recognized name when:
      - Required anatomical landmarks are lost.
      - User changes hand / body position.
      - Category changes.
    """

    def __init__(self, window_size: int = 8, min_stable_frames: int = 6, min_confidence: float = 0.20):
        self.window_size = window_size
        self.min_stable_frames = min_stable_frames
        self.min_confidence = min_confidence

        self.history = deque(maxlen=window_size)
        self.active_category_key = "1. Hasthas"
        self.current_recognized_name = None
        self.current_status_prompt = "Show a Hasta"

    def reset(self, new_category_key: Optional[str] = None):
        """Immediately and completely clears history and recognized state."""
        self.history.clear()
        self.current_recognized_name = None
        if new_category_key is not None:
            self.active_category_key = new_category_key
        self.current_status_prompt = self._get_missing_prompt(self.active_category_key)

    def _get_missing_prompt(self, cat_key: str) -> str:
        if cat_key == "1. Hasthas":
            return "Show a Hasta"
        elif cat_key == "2. Shiro Bhedhas (HEAD MOVEMENTS)":
            return "Show a Shiro Bhedha"
        elif cat_key == "3. Drishti Bhedhas(EYE MOVEMENTS)":
            return "Look at the camera"
        elif cat_key == "4. Pada Karma(Foot movements)":
            return "Show your feet/legs"
        return "Show gesture clearly"

    def _get_hold_prompt(self, cat_key: str) -> str:
        if cat_key == "1. Hasthas":
            return "Hold the Hasta clearly"
        elif cat_key == "2. Shiro Bhedhas (HEAD MOVEMENTS)":
            return "Hold the movement clearly"
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

        # Case 1: Missing landmark or invalid detection
        if not detected or not is_valid:
            self.history.clear()
            self.current_recognized_name = None
            return {
                "is_recognized": False,
                "recognized_name": None,
                "status_prompt": self._get_missing_prompt(cat_key)
            }

        # Case 2: Temporal buffer accumulating for movement categories
        if is_collecting:
            self.history.clear()
            self.current_recognized_name = None
            return {
                "is_recognized": False,
                "recognized_name": None,
                "status_prompt": "Collecting movement frames..."
            }

        # Case 3: Valid detection received
        self.history.append((raw_name, conf))

        # Check stability across rolling window (size <= window_size, e.g. 8)
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

        # A gesture is recognized if:
        # 1. Total window has at least min_stable_frames (e.g. 6)
        # 2. The dominant class has at least min_stable_frames (e.g. >= 6 of 8)
        # 3. The average confidence of that dominant class >= min_confidence (e.g. 0.20)
        # 4. The current frame's prediction matches the dominant class (instant drop on movement)
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
# Asynchronous Inference Worker (Decoupled from Camera Rendering)
# =============================================================================

class InferenceWorker:
    """
    Runs recognition models asynchronously in the background.
    Guarantees that the main UI camera preview never blocks or drops frames.
    """

    def __init__(self, engine: KuchipudiInferenceEngine, gate: Optional[GestureDisplayGate] = None):
        self.engine = engine
        self.gate = gate or GestureDisplayGate()
        self.lock = threading.Lock()
        self.pending_frame = None
        self.pending_category = None
        self.pending_smoothing = True
        self.has_new_frame = False
        self.latest_result = None
        self.stopped = False
        self.is_busy = False
        self.inference_fps = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def reset_category(self, cat_key: str):
        with self.lock:
            self.gate.reset(cat_key)
            self.latest_result = None

    def submit_frame(self, frame_bgr: np.ndarray, cat_key: str, apply_smoothing: bool):
        if self.is_busy:
            return  # Drop frame to eliminate latency
        with self.lock:
            self.pending_frame = frame_bgr
            self.pending_category = cat_key
            self.pending_smoothing = apply_smoothing
            self.has_new_frame = True

    def _run(self):
        while not self.stopped:
            frame = None
            cat_key = None
            smoothing = True
            with self.lock:
                if self.has_new_frame and self.pending_frame is not None:
                    frame = self.pending_frame
                    cat_key = self.pending_category
                    smoothing = self.pending_smoothing
                    self.has_new_frame = False
                    self.is_busy = True

            if frame is not None and cat_key is not None:
                t0 = time.time()
                try:
                    res = self.engine.process_category(frame, cat_key, apply_smoothing=smoothing)
                    gate_res = self.gate.update(res, cat_key)
                    res["gate"] = gate_res
                    dt = time.time() - t0
                    cur_fps = 1.0 / dt if dt > 0 else 15.0
                    with self.lock:
                        self.latest_result = res
                        if self.inference_fps == 0.0:
                            self.inference_fps = cur_fps
                        else:
                            self.inference_fps = 0.85 * self.inference_fps + 0.15 * cur_fps
                except Exception as e:
                    print(f"[InferenceWorker Error]: {e}")
                finally:
                    with self.lock:
                        self.is_busy = False
            else:
                time.sleep(0.005)

    def get_result(self) -> Tuple[Optional[Dict[str, Any]], float]:
        with self.lock:
            return self.latest_result, self.inference_fps

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


# =============================================================================
# Clean, User-Centric UI Layout & Rendering
# =============================================================================

def draw_start_screen(width: int = 1280, height: int = 760) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Renders the initial category selection start screen (Step 26).
    Clean, elegant, non-technical presentation.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 26)  # Dark charcoal slate

    c_gold = (0, 200, 255)
    c_white = (245, 245, 245)
    c_gray = (160, 160, 175)
    c_card_bg = (30, 30, 40)
    c_card_border = (65, 65, 85)

    # Title Banner
    cv2.putText(canvas, "KUCHIPUDI RECOGNITION", (width // 2 - 240, 110),
                cv2.FONT_HERSHEY_DUPLEX, 1.05, c_gold, 2, cv2.LINE_AA)

    # Prompt
    cv2.putText(canvas, "Select Category:", (width // 2 - 105, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, c_white, 1, cv2.LINE_AA)

    # 4 Category Cards
    card_w = 260
    card_h = 320
    spacing = 25
    total_w = 4 * card_w + 3 * spacing
    start_x = (width - total_w) // 2
    card_y = 230

    buttons = []
    card_accents = [
        (255, 180, 0),    # Hasthas - Sky Blue (BGR)
        (80, 210, 120),   # Shiro - Emerald
        (230, 100, 220),  # Drishti - Magenta
        (0, 165, 255)     # Pada - Orange
    ]

    for i, cat in enumerate(CATEGORIES):
        x1 = start_x + i * (card_w + spacing)
        y1 = card_y
        x2 = x1 + card_w
        y2 = y1 + card_h
        accent = card_accents[i]

        buttons.append({
            "id": cat["id"],
            "key": cat["key"],
            "rect": (x1, y1, x2, y2)
        })

        # Card Background
        cv2.rectangle(canvas, (x1, y1), (x2, y2), c_card_bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), c_card_border, 2)
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + 6), accent, -1)  # Top accent bar

        # Number Badge
        badge_cx = x1 + card_w // 2
        cv2.circle(canvas, (badge_cx, y1 + 55), 26, accent, -1)
        cv2.putText(canvas, str(cat["id"]), (badge_cx - 8, y1 + 64),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (15, 15, 20), 2, cv2.LINE_AA)

        # Category Name
        name_lines = cat["short_name"].split(" ")
        if len(name_lines) == 1:
            cv2.putText(canvas, name_lines[0], (x1 + 20, y1 + 130),
                        cv2.FONT_HERSHEY_DUPLEX, 0.68, c_white, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, name_lines[0], (x1 + 20, y1 + 120),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, c_white, 1, cv2.LINE_AA)
            cv2.putText(canvas, name_lines[1], (x1 + 20, y1 + 150),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, c_white, 1, cv2.LINE_AA)

        # Description
        cv2.putText(canvas, cat["desc"], (x1 + 20, y1 + 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, c_gray, 1, cv2.LINE_AA)

        # Action hint button inside card
        btn_y1 = y2 - 55
        btn_y2 = y2 - 20
        cv2.rectangle(canvas, (x1 + 20, btn_y1), (x2 - 20, btn_y2), (45, 45, 60), -1)
        cv2.rectangle(canvas, (x1 + 20, btn_y1), (x2 - 20, btn_y2), accent, 1)
        cv2.putText(canvas, f"Select [ {cat['id']} ]", (x1 + 65, btn_y1 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, accent, 1, cv2.LINE_AA)

    # Footer
    cv2.putText(canvas, "Press [1, 2, 3, 4] on keyboard or click a card to open mirror camera  |  [Q] Quit",
                (width // 2 - 330, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 145), 1, cv2.LINE_AA)

    return canvas, buttons


def draw_recognition_interface(
    mirror_frame: np.ndarray,
    active_cat_id: int,
    result: Optional[Dict[str, Any]],
    switching_label: Optional[str] = None,
    canvas_w: int = 1280,
    canvas_h: int = 800
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Renders the live camera recognition view for Step 26:
      - Top Header with active category & switch buttons
      - Large, prominent, unobstructed live mirror camera view
      - Clean single-card bottom HUD: shows ONLY Status prompt OR Recognized Name
      - Completely hides Top-5, probabilities, FPS, model names, confidence bars.
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 24)  # Deep dark slate

    c_gold = (0, 200, 255)
    c_white = (245, 245, 245)
    c_gray = (160, 160, 175)
    c_green = (80, 230, 110)
    c_border = (50, 50, 68)

    buttons = []
    active_meta = CATEGORY_BY_ID[active_cat_id]

    # =========================================================================
    # 1. TOP HEADER & CATEGORY TABS
    # =========================================================================
    header_h = 65
    cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (24, 24, 32), -1)
    cv2.line(canvas, (0, header_h), (canvas_w, header_h), (45, 45, 60), 1)

    cv2.putText(canvas, "KUCHIPUDI RECOGNITION", (25, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, c_gold, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"CATEGORY: {active_meta['short_name']}", (25, 52),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, c_white, 1, cv2.LINE_AA)

    # 4 Category Buttons at Top-Right
    tab_w = 175
    tab_h = 42
    tab_spacing = 10
    total_tabs_w = 4 * tab_w + 3 * tab_spacing
    tabs_start_x = canvas_w - total_tabs_w - 20
    tab_y = 12

    for i, cat in enumerate(CATEGORIES):
        x1 = tabs_start_x + i * (tab_w + tab_spacing)
        y1 = tab_y
        x2 = x1 + tab_w
        y2 = y1 + tab_h

        is_active = (cat["id"] == active_cat_id)
        buttons.append({
            "id": cat["id"],
            "key": cat["key"],
            "rect": (x1, y1, x2, y2)
        })

        if is_active:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (45, 90, 140), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), c_gold, 2)
            tab_text = f"[{cat['id']}] {cat['short_name']}"
            cv2.putText(canvas, tab_text, (x1 + 10, y1 + 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, c_white, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (32, 32, 42), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (55, 55, 70), 1)
            tab_text = f"[{cat['id']}] {cat['short_name']}"
            cv2.putText(canvas, tab_text, (x1 + 10, y1 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, c_gray, 1, cv2.LINE_AA)

    # =========================================================================
    # 2. LARGE LIVE MIRROR CAMERA PREVIEW (CENTRAL AREA)
    # =========================================================================
    cam_top = header_h + 10
    cam_bottom = canvas_h - 165
    avail_cam_h = cam_bottom - cam_top
    avail_cam_w = canvas_w - 40

    fh, fw = mirror_frame.shape[:2]
    scale = min(avail_cam_w / fw, avail_cam_h / fh)
    disp_w = int(fw * scale)
    disp_h = int(fh * scale)

    disp_x = (canvas_w - disp_w) // 2
    disp_y = cam_top + (avail_cam_h - disp_h) // 2

    frame_resized = cv2.resize(mirror_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    canvas[disp_y:disp_y + disp_h, disp_x:disp_x + disp_w] = frame_resized
    cv2.rectangle(canvas, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), (55, 55, 75), 2)

    # Mirror Mode Indicator Badge (Top-Left of Preview)
    badge_bg = (20, 20, 28)
    cv2.rectangle(canvas, (disp_x + 12, disp_y + 12), (disp_x + 165, disp_y + 36), badge_bg, -1)
    cv2.circle(canvas, (disp_x + 24, disp_y + 24), 4, c_green, -1)
    cv2.putText(canvas, "MIRROR PREVIEW", (disp_x + 36, disp_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, c_white, 1, cv2.LINE_AA)

    # Category Mode Banner (Top-Right of Preview)
    mode_text = f"MODE: {active_meta['short_name']}"
    cv2.rectangle(canvas, (disp_x + disp_w - 180, disp_y + 12), (disp_x + disp_w - 12, disp_y + 36), badge_bg, -1)
    cv2.putText(canvas, mode_text, (disp_x + disp_w - 170, disp_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, c_gold, 1, cv2.LINE_AA)

    # Switching Category Overlay Banner
    if switching_label:
        overlay = canvas.copy()
        banner_w = 460
        banner_h = 70
        bx1 = (canvas_w - banner_w) // 2
        by1 = disp_y + (disp_h - banner_h) // 2
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 30), -1)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), c_gold, 2)
        cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)
        cv2.putText(canvas, f"SWITCHING TO {switching_label}...", (bx1 + 30, by1 + 44),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, c_gold, 1, cv2.LINE_AA)

    # =========================================================================
    # 3. BOTTOM RECOGNITION PANEL — ONE GESTURE -> ONE NAME
    # =========================================================================
    card_w = 860
    card_h = 135
    card_x1 = (canvas_w - card_w) // 2
    card_y1 = canvas_h - 150
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h

    # Extract gate results
    gate_res = result.get("gate", {}) if result else {}
    is_recognized = gate_res.get("is_recognized", False)
    rec_name = gate_res.get("recognized_name", None)
    status_prompt = gate_res.get("status_prompt", "Show a Hasta")

    if is_recognized and rec_name:
        # --- RECOGNIZED STATE (ONE NAME ONLY) ---
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (20, 28, 22), -1)
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), c_green, 2)
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y1 + 5), c_green, -1)

        # Label: RECOGNIZED
        label_text = "RECOGNIZED"
        (lw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.58, 2)
        cv2.putText(canvas, label_text, (card_x1 + (card_w - lw) // 2, card_y1 + 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, c_green, 2, cv2.LINE_AA)

        # Recognized Gesture Name (Large, centered)
        name_text = rec_name.upper()
        (nw, _), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_DUPLEX, 1.10, 2)
        font_scale = 1.10 if nw < (card_w - 40) else 0.85
        (nw, _), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
        cv2.putText(canvas, name_text, (card_x1 + (card_w - nw) // 2, card_y1 + 92),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, c_white, 2, cv2.LINE_AA)

    else:
        # --- WAITING / STABILIZING STATE ---
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (24, 24, 32), -1)
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), c_border, 1)

        # Label: STATUS
        label_text = "STATUS"
        (lw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(canvas, label_text, (card_x1 + (card_w - lw) // 2, card_y1 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, c_gray, 1, cv2.LINE_AA)

        # Prompt text (e.g. "Show a Hasta" or "Hold the Hasta clearly")
        (pw, _), _ = cv2.getTextSize(status_prompt, cv2.FONT_HERSHEY_DUPLEX, 0.78, 2)
        cv2.putText(canvas, status_prompt, (card_x1 + (card_w - pw) // 2, card_y1 + 86),
                    cv2.FONT_HERSHEY_DUPLEX, 0.78, (235, 235, 245), 2, cv2.LINE_AA)

    # Footer navigation text
    footer_text = "[1-4] Switch Category   |   [M] Menu   |   [Q] Exit"
    (fw, _), _ = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(canvas, footer_text, ((canvas_w - fw) // 2, canvas_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 125), 1, cv2.LINE_AA)

    return canvas, buttons


# =============================================================================
# Main Live Application Runner
# =============================================================================

def run_application(
    camera_idx: int = 0,
    width: int = 1280,
    height: int = 720,
    initial_category_id: Optional[int] = None,
    use_mock: bool = False,
    single_image_path: Optional[str] = None,
    headless_frames: Optional[int] = None,
    initial_smoothing: bool = True,
    min_confidence: float = 0.20,
    min_stable_frames: int = 6,
    window_size: int = 8
):
    """
    Main user-facing demonstration application loop for Step 26.
    One Category -> One Gesture -> One Name.
    """
    engine = KuchipudiInferenceEngine()
    gate = GestureDisplayGate(
        window_size=window_size,
        min_stable_frames=min_stable_frames,
        min_confidence=min_confidence
    )

    # Mode 1: Single Image Offline Functional Test
    if single_image_path:
        img_p = Path(single_image_path)
        if not img_p.exists():
            print(f"[ERROR] Image not found: {img_p}")
            engine.close()
            return
        frame = cv2.imread(str(img_p))
        if frame is None:
            print(f"[ERROR] Cannot decode image: {img_p}")
            engine.close()
            return
        cat_id = initial_category_id or 1
        cat_key = CATEGORY_BY_ID[cat_id]["key"]
        res = engine.process_category(frame, cat_key, apply_smoothing=False)
        gate_res = gate.update(res, cat_key)
        res["gate"] = gate_res

        print("\n" + "=" * 60)
        print(f"STEP 26 DEMO SINGLE IMAGE RESULT (CATEGORY: {res['category_short']})")
        print("=" * 60)
        print(f"  Category            : {res['category_short']}")
        print(f"  Gesture Type        : {res['gesture_type']}")
        print(f"  Raw Prediction      : {res['gesture_name']} ({res['confidence']:.2f}%)")
        print(f"  Gate Decision       : {'RECOGNIZED' if gate_res['is_recognized'] else 'HOLDING / UNSTABLE'}")
        print(f"  Displayed Name      : {gate_res['recognized_name'] or 'None (Status: ' + gate_res['status_prompt'] + ')'}")
        print("=" * 60)
        engine.close()
        return

    # Mode 2: Live Video Stream & Decoupled Inference Worker
    cam = CameraCaptureThread(src=camera_idx, width=width, height=height, use_mock=use_mock)
    worker = InferenceWorker(engine, gate=gate)

    # State Variables
    app_state = "START_SCREEN" if initial_category_id is None and headless_frames is None else "RECOGNITION"
    active_cat_id = initial_category_id or 1
    cam.set_mock_category(CATEGORY_BY_ID[active_cat_id]["key"])

    smoothing_active = initial_smoothing
    switching_label = None
    switching_until = 0.0

    window_name = "Kuchipudi Dance Recognition - Step 26 Final Demo"
    if headless_frames is None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 800)

    # Interactive Mouse Clicking on Category Buttons
    clicked_button_id = [None]

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in current_buttons:
                bx1, by1, bx2, by2 = btn["rect"]
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    clicked_button_id[0] = btn["id"]
                    break

    if headless_frames is None:
        cv2.setMouseCallback(window_name, on_mouse_click)

    current_buttons = []
    fps_history = deque(maxlen=20)
    frame_count = 0

    print("\n" + "=" * 80)
    print("KUCHIPUDI RECOGNITION APPLICATION RUNNING (STEP 26 DEMO MODE)")
    print("  Controls:")
    print("    [1] HASTHAS          [2] SHIRO BHEDHAS")
    print("    [3] DRISHTI BHEDHAS  [4] PADA KARMA")
    print("    [+] / [-] Adjust Gate Sensitivity")
    print("    [M] Menu / Start Screen")
    print("    [Q] or ESC Clean Quit")
    print(f"  Gate Threshold: {gate.min_confidence:.2f} (Requires {gate.min_stable_frames}/{gate.window_size} stable frames)")
    print("=" * 80 + "\n")

    try:
        while True:
            t0 = time.time()

            # Read newest frame from background camera capture thread
            ret, frame_bgr = cam.read()
            if not ret or frame_bgr is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Handle Category Switching
            now = time.time()
            if switching_label and now >= switching_until:
                switching_label = None

            # Handle On-Screen Mouse Clicks
            if clicked_button_id[0] is not None:
                new_cat_id = clicked_button_id[0]
                clicked_button_id[0] = None
                if new_cat_id != active_cat_id:
                    active_cat_id = new_cat_id
                    cat_meta = CATEGORY_BY_ID[active_cat_id]
                    switching_label = cat_meta["short_name"]
                    switching_until = time.time() + 0.5
                    engine.reset_category(cat_meta["key"])
                    worker.reset_category(cat_meta["key"])
                    cam.set_mock_category(cat_meta["key"])
                    print(f"[SWITCH] Activated Category: {cat_meta['short_name']}")
                if app_state == "START_SCREEN":
                    app_state = "RECOGNITION"

            # Create Mirror Display Frame (Horizontally flipped)
            # The inference pipeline receives the original frame_bgr (native orientation)
            mirror_frame = cv2.flip(frame_bgr, 1)

            # Submit newest frame to asynchronous inference worker
            if app_state == "RECOGNITION":
                active_cat_key = CATEGORY_BY_ID[active_cat_id]["key"]
                worker.submit_frame(frame_bgr, active_cat_key, apply_smoothing=smoothing_active)

            # Retrieve latest available prediction
            latest_res, infer_fps = worker.get_result()

            # Measure smooth camera preview FPS
            dt = time.time() - t0
            cur_fps = 1.0 / dt if dt > 0 else 30.0
            fps_history.append(cur_fps)
            cam_fps = float(np.mean(fps_history))

            # Headless automated testing mode
            if headless_frames is not None:
                gate_info = latest_res.get("gate", {}) if latest_res else {}
                is_rec = gate_info.get("is_recognized", False)
                status_str = f"RECOGNIZED: {gate_info.get('recognized_name')}" if is_rec else f"HOLD: {gate_info.get('status_prompt', 'Collecting...')}"
                print(f"  [Frame {frame_count:02d} - {CATEGORY_BY_ID[active_cat_id]['short_name']}] {status_str} | Cam: {cam_fps:.1f} FPS")
                if frame_count >= headless_frames:
                    print(f"\n[HEADLESS TEST] Completed {frame_count} frames successfully. Exiting cleanly.")
                    break
                continue

            # Render UI Canvas (Step 26 Single Name Display)
            if app_state == "START_SCREEN":
                canvas, current_buttons = draw_start_screen(width=1280, height=800)
            else:
                canvas, current_buttons = draw_recognition_interface(
                    mirror_frame=mirror_frame,
                    active_cat_id=active_cat_id,
                    result=latest_res,
                    switching_label=switching_label,
                    canvas_w=1280,
                    canvas_h=800
                )

            cv2.imshow(window_name, canvas)

            # Keyboard Events
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("[INFO] User requested clean exit.")
                break
            elif key == ord('1'):
                if active_cat_id != 1 or app_state == "START_SCREEN":
                    active_cat_id = 1
                    app_state = "RECOGNITION"
                    switching_label = "HASTHAS"
                    switching_until = time.time() + 0.5
                    engine.reset_category(CATEGORIES[0]["key"])
                    worker.reset_category(CATEGORIES[0]["key"])
                    cam.set_mock_category(CATEGORIES[0]["key"])
                    print("[KEY 1] Switched to HASTHAS.")
            elif key == ord('2'):
                if active_cat_id != 2 or app_state == "START_SCREEN":
                    active_cat_id = 2
                    app_state = "RECOGNITION"
                    switching_label = "SHIRO BHEDHAS"
                    switching_until = time.time() + 0.5
                    engine.reset_category(CATEGORIES[1]["key"])
                    worker.reset_category(CATEGORIES[1]["key"])
                    cam.set_mock_category(CATEGORIES[1]["key"])
                    print("[KEY 2] Switched to SHIRO BHEDHAS.")
            elif key == ord('3'):
                if active_cat_id != 3 or app_state == "START_SCREEN":
                    active_cat_id = 3
                    app_state = "RECOGNITION"
                    switching_label = "DRISHTI BHEDHAS"
                    switching_until = time.time() + 0.5
                    engine.reset_category(CATEGORIES[2]["key"])
                    worker.reset_category(CATEGORIES[2]["key"])
                    cam.set_mock_category(CATEGORIES[2]["key"])
                    print("[KEY 3] Switched to DRISHTI BHEDHAS.")
            elif key == ord('4'):
                if active_cat_id != 4 or app_state == "START_SCREEN":
                    active_cat_id = 4
                    app_state = "RECOGNITION"
                    switching_label = "PADA KARMA"
                    switching_until = time.time() + 0.5
                    engine.reset_category(CATEGORIES[3]["key"])
                    worker.reset_category(CATEGORIES[3]["key"])
                    cam.set_mock_category(CATEGORIES[3]["key"])
                    print("[KEY 4] Switched to PADA KARMA.")
            elif key in [ord('+'), ord('=')]:
                gate.min_confidence = min(0.95, round(gate.min_confidence + 0.05, 2))
                print(f"[GATE] Confidence threshold increased to: {gate.min_confidence:.2f}")
            elif key in [ord('-'), ord('_')]:
                gate.min_confidence = max(0.05, round(gate.min_confidence - 0.05, 2))
                print(f"[GATE] Confidence threshold decreased to: {gate.min_confidence:.2f}")
            elif key in [ord('s'), ord('S')]:
                smoothing_active = not smoothing_active
                print(f"[KEY S] Display smoothing: {'ON' if smoothing_active else 'OFF'}")
            elif key in [ord('m'), ord('M')]:
                app_state = "START_SCREEN"
                print("[KEY M] Returned to Start Screen Menu.")

    finally:
        cam.stop()
        worker.stop()
        cv2.destroyAllWindows()
        engine.close()
        print("[INFO] All camera, worker, and detector resources released cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Kuchipudi Dance Recognition — Step 26 Final Demo Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Target camera capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target camera capture height (default: 720)")
    parser.add_argument("--category", type=int, default=None, choices=[1, 2, 3, 4],
                        help="Start directly in category: 1=Hasthas, 2=Shiro, 3=Drishti, 4=Pada")
    parser.add_argument("--mock", action="store_true", help="Force simulated dance frames (demo mode without webcam)")
    parser.add_argument("--image", type=str, default=None, help="Run single image test on an image path")
    parser.add_argument("--headless", type=int, default=None, help="Headless test for N frames then exit")
    parser.add_argument("--min-conf", type=float, default=0.20, help="Minimum confidence threshold for display gate (default: 0.20)")
    parser.add_argument("--min-frames", type=int, default=6, help="Minimum consistent frames for display gate (default: 6)")
    parser.add_argument("--window", type=int, default=8, help="Rolling window size for display gate (default: 8)")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable display smoothing")
    args = parser.parse_args()

    run_application(
        camera_idx=args.camera,
        width=args.width,
        height=args.height,
        initial_category_id=args.category,
        use_mock=args.mock,
        single_image_path=args.image,
        headless_frames=args.headless,
        initial_smoothing=not args.no_smoothing,
        min_confidence=args.min_conf,
        min_stable_frames=args.min_frames,
        window_size=args.window
    )


if __name__ == "__main__":
    main()
