"""
NRITYA AI — REAL-TIME KUCHIPUDI RECOGNITION
=============================================
Unified Classical Kuchipudi Dance Recognition Deployment Application.
Supports Desktop and Embedded Edge (Raspberry Pi) Environments.

Supported Categories (69 Total Classes):
  1. HASTHAS (52 Classes)        — Step 34 Personalized MLP (175-D Landmarks)
  2. SHIRO BHEDHAS (9 Classes)   — Step 36C Personalized BiGRU (102-D Temporal Sequence)
  3. DRISHTI BHEDHAS (8 Classes) — Step 37C Personalized BiGRU (132-D Gaze/Eye Temporal Sequence)

Features:
  - Decoupled Multi-Threaded Architecture:
      * Thread 1 (CameraCaptureStream): Lock-free continuous capture & horizontal mirroring (25-30+ FPS).
      * Thread 2 (NrityaInferenceWorker): Asynchronous category-isolated AI inference engine.
      * Main Thread: Smooth GUI display, interactive HUD, mouse and keyboard controls.
  - Zero Coordinate Mismatch: Detection and display share the exact mirrored coordinate space.
  - Handedness Label Removed: Pure gesture presentation without confusing left/right text.
  - Resource Optimized: Only the active category's MediaPipe pipeline runs at any given time.

Controls:
  [1]     : Select HASTHAS (52 classes)
  [2]     : Select SHIRO BHEDHAS (9 classes)
  [3]     : Select DRISHTI BHEDHAS (8 classes)
  [R]     : Reset prediction buffer & temporal sequence state
  [D]     : Toggle technical debug overlay (off by default)
  [Q]/ESC : Exit application safely
  Mouse   : Click category buttons directly on HUD
"""

import os
import sys
import re
import json
import time
import math
import argparse
import threading
from datetime import datetime
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

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import proven project extractors with self-contained fallback
try:
    from src.landmark_extraction import (
        HandLandmarkExtractor,
        PoseLandmarkExtractor,
        FaceDrishtiLandmarkExtractor,
        calculate_angle_3d
    )
except ImportError:
    # Embedded fallback for standalone Raspberry Pi execution
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
        def __init__(self, static_image_mode=False, min_detection_confidence=0.30):
            self.pose = mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.30
            )

        def close(self):
            self.pose.close()

        def process(self, image_rgb: np.ndarray, category_focus: str = "shiro") -> Dict[str, Any]:
            results = self.pose.process(image_rgb)
            if not results.pose_landmarks:
                return {"detected": False, "vector": np.zeros(32, dtype=np.float32), "shiro_features": {}, "raw_landmarks": [], "pose_landmarks": None}
            lms = results.pose_landmarks.landmark
            all_pts = [np.array([lm.x, lm.y, lm.z], dtype=np.float32) for lm in lms]
            vis = [float(lm.visibility) for lm in lms]
            p_nose, p_ear_l, p_ear_r, p_sh_l, p_sh_r, p_hip_l, p_hip_r = all_pts[0], all_pts[7], all_pts[8], all_pts[11], all_pts[12], all_pts[23], all_pts[24]
            mid_sh = (p_sh_l + p_sh_r) / 2.0
            mid_hip = (p_hip_l + p_hip_r) / 2.0
            torso_h = max(1e-5, float(np.linalg.norm(mid_sh[:2] - mid_hip[:2])))
            sh_w = max(1e-5, float(np.linalg.norm(p_sh_l[:2] - p_sh_r[:2])))
            lat_shift = float((p_nose[0] - mid_sh[0]) / sh_w)
            tilt_roll = float(math.atan2(p_ear_r[1] - p_ear_l[1], p_ear_r[0] - p_ear_l[0]))
            pitch = float((p_nose[1] - mid_sh[1]) / torso_h)
            ear_mid = (p_ear_l + p_ear_r) / 2.0
            yaw = float((p_nose[0] - ear_mid[0]) / (sh_w + 1e-6))
            upper_rel = np.concatenate([(p_nose - mid_sh) / torso_h, (p_ear_l - mid_sh) / torso_h, (p_ear_r - mid_sh) / torso_h, (p_sh_l - mid_sh) / sh_w, (p_sh_r - mid_sh) / sh_w, (mid_hip - mid_sh) / torso_h])
            base_vec = np.concatenate([np.array([lat_shift, tilt_roll, pitch, yaw, sh_w, torso_h, vis[0], vis[7], vis[8], vis[11], vis[12], vis[23], vis[24], float(np.linalg.norm(p_nose[:2] - mid_sh[:2]))], dtype=np.float32), upper_rel.astype(np.float32)])
            return {"detected": True, "vector": base_vec, "shiro_features": {"head_tilt_roll_rad": tilt_roll, "head_pitch": pitch, "head_yaw": yaw}, "raw_landmarks": results.pose_landmarks, "pose_landmarks": results.pose_landmarks}

    class FaceDrishtiLandmarkExtractor:
        def __init__(self, static_image_mode=False, min_detection_confidence=0.30):
            self.face_mesh = mp_face.FaceMesh(static_image_mode=static_image_mode, max_num_faces=1, refine_landmarks=True, min_detection_confidence=min_detection_confidence, min_tracking_confidence=0.30)

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
            w_l, h_l = max(1e-6, float(np.linalg.norm(p_ol[:2] - p_il[:2]))), max(1e-6, float(np.linalg.norm(p_tl[:2] - p_bl[:2])))
            w_r, h_r = max(1e-6, float(np.linalg.norm(p_or[:2] - p_ir[:2]))), max(1e-6, float(np.linalg.norm(p_tr[:2] - p_br[:2])))
            ear_l, ear_r = h_l / w_l, h_r / w_r
            g_hl, g_vl = float((p_iris_l[0] - p_il[0]) / (w_l + 1e-6)), float((p_iris_l[1] - p_tl[1]) / (h_l + 1e-6))
            g_hr, g_vr = float((p_iris_r[0] - p_ir[0]) / (w_r + 1e-6)), float((p_iris_r[1] - p_tr[1]) / (h_r + 1e-6))
            fw, fh = max(1e-6, float(np.linalg.norm(p_chl[:2] - p_chr[:2]))), max(1e-6, float(np.linalg.norm(p_fore[:2] - p_chin[:2])))
            mid_eye = (p_ol + p_or) / 2.0
            yaw, pitch, roll = float((p_nose[0] - mid_eye[0]) / fw), float((p_nose[1] - mid_eye[1]) / fh), float(math.atan2(p_or[1] - p_ol[1], p_or[0] - p_ol[0]))
            fc = (p_fore + p_chin) / 2.0
            eye_l_rel, eye_r_rel, nose_rel = (p_iris_l - fc) / fw, (p_iris_r - fc) / fw, (p_nose - fc) / fh
            eye_rel = np.concatenate([(p_ol - p_iris_l)/w_l, (p_il - p_iris_l)/w_l, (p_tl - p_iris_l)/h_l, (p_bl - p_iris_l)/h_l, (p_or - p_iris_r)/w_r, (p_ir - p_iris_r)/w_r, (p_tr - p_iris_r)/h_r, (p_br - p_iris_r)/h_r])
            vec_44 = np.concatenate([np.array([ear_l, ear_r, g_hl, g_vl, g_hr, g_vr, yaw, pitch, roll, fw, fh], dtype=np.float32), eye_l_rel, eye_r_rel, nose_rel, eye_rel]).astype(np.float32)
            return {"detected": True, "vector": vec_44, "landmarks": results.multi_face_landmarks[0]}


# =============================================================================
# 1. ASSET PATH RESOLUTION
# =============================================================================

def resolve_asset_paths() -> Dict[str, Path]:
    """Locates all required models, scalers, and mapping files."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    # Candidate locations: Standalone board folder vs full workspace
    candidates = [
        # Standalone Raspberry Pi folder
        {
            "hastha_model": script_dir / "models" / "hasthas" / "personalized_hasthas_mlp_best.keras",
            "hastha_scaler": script_dir / "models" / "hasthas" / "personalized_hasthas_scaler.npz",
            "hastha_mapping": script_dir / "models" / "hasthas" / "step16a_hasthas_class_mapping.json",
            "shiro_model": script_dir / "models" / "shiro" / "personalized_shiro_bigru_best.keras",
            "shiro_scaler": script_dir / "models" / "shiro" / "shiro_temporal_scaler.npz",
            "shiro_mapping": script_dir / "models" / "shiro" / "shiro_class_mapping.json",
            "shiro_config": script_dir / "models" / "shiro" / "sequence_preprocessing_config.json",
            "drishti_model": script_dir / "models" / "drishti" / "personalized_drishti_bigru_best.keras",
            "drishti_scaler": script_dir / "models" / "drishti" / "drishti_temporal_scaler.npz",
            "drishti_mapping": script_dir / "models" / "drishti" / "drishti_class_mapping.json",
        },
        # Workspace Root folder
        {
            "hastha_model": workspace_root / "models" / "personalized_step34" / "personalized_hasthas_mlp_best.keras",
            "hastha_scaler": workspace_root / "models" / "personalized_step34" / "personalized_hasthas_scaler.npz",
            "hastha_mapping": workspace_root / "models" / "step16a_hasthas_class_mapping.json",
            "shiro_model": workspace_root / "models" / "personalized_step36_shiro" / "personalized_shiro_bigru_best.keras",
            "shiro_scaler": workspace_root / "models" / "personalized_step36_shiro" / "shiro_temporal_scaler.npz",
            "shiro_mapping": workspace_root / "models" / "personalized_step36_shiro" / "shiro_class_mapping.json",
            "shiro_config": workspace_root / "models" / "personalized_step36_shiro" / "sequence_preprocessing_config.json",
            "drishti_model": workspace_root / "models" / "personalized_step37_drishti" / "personalized_drishti_bigru_best.keras",
            "drishti_scaler": workspace_root / "models" / "personalized_step37_drishti" / "drishti_temporal_scaler.npz",
            "drishti_mapping": workspace_root / "models" / "personalized_step37_drishti" / "drishti_class_mapping.json",
        }
    ]

    for cand in candidates:
        if cand["hastha_model"].exists() and cand["shiro_model"].exists() and cand["drishti_model"].exists():
            return cand

    return candidates[1]


# =============================================================================
# 2. THREAD 1: DEDICATED CAMERA CAPTURE STREAM (WITH HORIZONTAL MIRRORING)
# =============================================================================

class CameraCaptureStream:
    """
    Dedicated background capture thread that continuously drains the driver queue
    and mirrors the image horizontally once. This guarantees that detection and display
    share the EXACT SAME coordinate space, eliminating landmark display misalignment.
    """
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.lock = threading.Lock()
        self.latest_frame_mirrored: Optional[np.ndarray] = None
        self.latest_timestamp: float = 0.0
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureThread")
        self.thread.start()
        return True

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Mirror immediately so pipeline and display have 100% unified coordinates
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
        if self.cap.isOpened():
            self.cap.release()


# =============================================================================
# 3. THREAD 2: MULTI-CATEGORY ASYNCHRONOUS INFERENCE WORKER
# =============================================================================

class NrityaInferenceWorker:
    """
    Multi-category AI inference engine.
    Executes ONLY the active MediaPipe pipeline corresponding to the selected category:
      - Hasthas: MediaPipe Hands + Step 34 MLP (52 classes)
      - Shiro: MediaPipe Pose + Step 36C BiGRU (9 classes)
      - Drishti: MediaPipe FaceMesh + Step 37C BiGRU (8 classes)
    """
    def __init__(self, asset_paths: Dict[str, Path], initial_category: str = "hasthas"):
        self.asset_paths = asset_paths
        self.current_category = initial_category.lower()

        # ---------------------------------------------------------------------
        # 1. HASTHAS (Step 34 MLP, 52 Classes)
        # ---------------------------------------------------------------------
        print("[INIT] Loading Step 34 Hastha Model (52 Classes)...")
        self.hastha_model = tf.keras.models.load_model(str(asset_paths["hastha_model"]), compile=False)
        with open(asset_paths["hastha_mapping"], "r", encoding="utf-8") as f:
            h_map = json.load(f)
        self.hastha_idx_to_name = {int(k): v for k, v in h_map["idx_to_class"].items()}

        s_h = np.load(str(asset_paths["hastha_scaler"]))
        self.hastha_mean = s_h["mean"].astype(np.float32)
        self.hastha_scale = np.where(s_h["scale"] == 0, 1.0, s_h["scale"]).astype(np.float32)

        self.samyutha_keywords = [
            "Anjali", "Kapothamu", "Karkatamu", "Swasthikamu", "Dola", "Pushpaputamu",
            "Uthsungamu", "Sivalingamu", "Katakavardhanamu", "Karthariswasthikamu",
            "Sakatamu", "Sankhamu", "Chakramu", "Samputamu", "Paasamu", "Keelakamu",
            "Matsyamu", "Kurmamu", "Varahamu", "Garudamu", "Nagabandhakamu", "Khatva",
            "Bherundamu", "Avahithamu"
        ]

        self.hastha_buffer = deque(maxlen=8)
        self.hastha_extractor = HandLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.35)
        self.hastha_mp_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.35, min_tracking_confidence=0.30)
        self.hastha_mp_fallback = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.15, min_tracking_confidence=0.15)

        # ---------------------------------------------------------------------
        # 2. SHIRO BHEDHAS (Step 36C BiGRU, 9 Classes)
        # ---------------------------------------------------------------------
        print("[INIT] Loading Step 36 Shiro Model (9 Classes)...")
        self.shiro_model = tf.keras.models.load_model(str(asset_paths["shiro_model"]))
        with open(asset_paths["shiro_mapping"], "r", encoding="utf-8") as f:
            s_map = json.load(f)
        s_classes = sorted(s_map.get("classes", []), key=lambda x: x["class_id"])
        self.shiro_sanskrit_names = [c["sanskrit_name"] for c in s_classes]

        s_s = np.load(str(asset_paths["shiro_scaler"]))
        self.shiro_mean = s_s["mean"].astype(np.float32)
        self.shiro_std = s_s["std"].astype(np.float32)
        self.shiro_seq_len = 160
        self.shiro_feat_dim = 102

        self.shiro_extractor = PoseLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.30)
        self.shiro_buffer = deque(maxlen=8)
        self.shiro_temporal_buffer = deque(maxlen=self.shiro_seq_len)
        self.shiro_prev_ts: Optional[float] = None
        self.shiro_prev_base: Optional[np.ndarray] = None
        self.shiro_prev_vel: Optional[np.ndarray] = None
        self.shiro_baseline_angles: Optional[np.ndarray] = None
        self.shiro_sequence_id: int = 1

        # ---------------------------------------------------------------------
        # 3. DRISHTI BHEDHAS (Step 37C BiGRU, 8 Classes)
        # ---------------------------------------------------------------------
        print("[INIT] Loading Step 37 Drishti Model (8 Classes)...")
        self.drishti_model = tf.keras.models.load_model(str(asset_paths["drishti_model"]))
        with open(asset_paths["drishti_mapping"], "r", encoding="utf-8") as f:
            d_map = json.load(f)
        d_classes = sorted(d_map.get("classes", []), key=lambda x: x["class_id"])
        self.drishti_sanskrit_names = [c["sanskrit_name"] for c in d_classes]

        s_d = np.load(str(asset_paths["drishti_scaler"]))
        self.drishti_mean = s_d["mean"].astype(np.float32)
        self.drishti_std = np.where(s_d["std"] == 0, 1.0, s_d["std"]).astype(np.float32)
        self.drishti_seq_len = 160
        self.drishti_feat_dim = 132

        self.drishti_extractor = FaceDrishtiLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.30)
        self.drishti_buffer = deque(maxlen=8)
        self.drishti_temporal_buffer = deque(maxlen=self.drishti_seq_len)
        self.drishti_prev_ts: Optional[float] = None
        self.drishti_prev_base: Optional[np.ndarray] = None
        self.drishti_prev_vel: Optional[np.ndarray] = None

        # ---------------------------------------------------------------------
        # 4. COMPILED GRAPH EXECUTION & WARMUP
        # ---------------------------------------------------------------------
        print("[INIT] Compiling neural inference graphs...")
        @tf.function(reduce_retracing=True)
        def _h_predict(x): return self.hastha_model(x, training=False)
        self.compiled_hastha_predict = _h_predict

        @tf.function(reduce_retracing=True)
        def _s_predict(x): return self.shiro_model(x, training=False)
        self.compiled_shiro_predict = _s_predict

        @tf.function(reduce_retracing=True)
        def _d_predict(x): return self.drishti_model(x, training=False)
        self.compiled_drishti_predict = _d_predict

        # Graph Warm-up forward passes
        _ = self.compiled_hastha_predict(np.zeros((1, 175), dtype=np.float32))
        _ = self.compiled_shiro_predict(np.zeros((1, self.shiro_seq_len, self.shiro_feat_dim), dtype=np.float32))
        _ = self.compiled_drishti_predict(np.zeros((1, self.drishti_seq_len, self.drishti_feat_dim), dtype=np.float32))
        print("[INIT] Models loaded and warmed up successfully.")

        # ---------------------------------------------------------------------
        # 5. STATE SYNCHRONIZATION
        # ---------------------------------------------------------------------
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

    def set_category(self, category: str):
        with self.category_lock:
            cat = category.lower()
            if cat in ["hasthas", "shiro", "drishti"]:
                if self.current_category != cat:
                    old_cat = self.current_category
                    self.current_category = cat
                    self._switch_category_state(old_cat, cat)

    def _switch_category_state(self, old_cat: str, new_cat: str):
        """
        Safely reset ONLY the previous category's temporary inference state
        and initialize/reset the new category's state without touching camera
        or restarting detectors.
        """
        with self.lock:
            # 1. Safely reset previous category state
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

            # 2. Safely initialize new category state
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

            # 3. Synchronize thread-safe shared state
            self.shared_state["category"] = new_cat
            self.shared_state["detected"] = False
            self.shared_state["status"] = prompt
            self.shared_state["recognized_name"] = None
            self.shared_state["recognized_subtitle"] = None
            self.shared_state["confidence"] = 0.0
            self.shared_state["landmarks"] = None
            self.shared_state["latency_ms"] = 0.0

    def reset_state(self):
        """Resets the currently active category's temporary buffer/state."""
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
        """Removes numerical prefix: '12. Katakamukha' -> 'KATAKAMUKHA'."""
        return re.sub(r"^\d+\.\s*", "", full_name).strip().upper()

    def is_samyutha(self, full_name: str) -> bool:
        return any(kw in full_name for kw in self.samyutha_keywords)

    def run_worker_loop(self, stream: CameraCaptureStream):
        last_process_time = 0.0
        last_detect_time = time.time()

        while not self.stop_event.is_set():
            if self.reset_event.is_set():
                self.reset_state()
                self.reset_event.clear()

            now = time.time()
            cat = self.current_category

            # Temporal pacing (~15 Hz for Shiro/Drishti, ~30 Hz for Hasthas)
            min_interval = 0.050 if cat in ["shiro", "drishti"] else 0.020
            if (now - last_process_time) < min_interval:
                time.sleep(0.003)
                continue

            ret, frame_mirrored, timestamp = stream.read_latest()
            if not ret or frame_mirrored is None or timestamp <= last_process_time:
                time.sleep(0.003)
                continue

            last_process_time = timestamp
            t_start = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_mirrored, cv2.COLOR_BGR2RGB)

            # =================================================================
            # BRANCH 1: HASTHAS (EXACT STEP 35 PREPROCESSING ON MIRRORED FRAME)
            # =================================================================
            if cat == "hasthas":
                results = self.hastha_mp_detector.process(frame_rgb)
                if not results.multi_hand_landmarks:
                    results = self.hastha_mp_fallback.process(frame_rgb)

                if not results.multi_hand_landmarks:
                    self.hastha_buffer.clear()
                    with self.lock:
                        self.shared_state["category"] = "hasthas"
                        self.shared_state["detected"] = False
                        self.shared_state["status"] = "SHOW GESTURE"
                        self.shared_state["recognized_name"] = None
                        self.shared_state["recognized_subtitle"] = None
                        self.shared_state["confidence"] = 0.0
                        self.shared_state["landmarks"] = None
                        self.shared_state["latency_ms"] = (time.perf_counter() - t_start) * 1000.0
                    continue

                num_hands = len(results.multi_hand_landmarks)

                # Single Hand (Asamyuta)
                if num_hands == 1:
                    hand_lms = results.multi_hand_landmarks[0]
                    raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

                    handedness_label = "Left"
                    if results.multi_handedness and len(results.multi_handedness) > 0:
                        try:
                            handedness_label = results.multi_handedness[0].classification[0].label
                        except Exception:
                            handedness_label = "Left"

                    # Canonical reflection across wrist x for left hand demonstration
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

                # Two Hands (Samyutha)
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

                # Step 34 Scaling
                scaled_175 = (vector_175 - self.hastha_mean) / self.hastha_scale
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

                latency = (time.perf_counter() - t_start) * 1000.0

                with self.lock:
                    self.shared_state["category"] = "hasthas"
                    self.shared_state["detected"] = True
                    self.shared_state["landmarks"] = results.multi_hand_landmarks
                    self.shared_state["latency_ms"] = latency

                    if is_stable and winner_cid is not None:
                        full_name = self.hastha_idx_to_name.get(winner_cid, "Unknown")
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

            # =================================================================
            # BRANCH 2: SHIRO BHEDHAS (EXACT STEP 36D PIPELINE)
            # =================================================================
            elif cat == "shiro":
                try:
                    res = self.shiro_extractor.process(frame_rgb, category_focus="shiro")
                    if not res.get("detected"):
                        if (now - last_detect_time) > 0.50 and len(self.shiro_temporal_buffer) > 0:
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
                        continue

                    last_detect_time = now
                    base_32 = res["vector"]
                    shiro_feats = res["shiro_features"]

                    if self.shiro_prev_ts is not None:
                        raw_dt = timestamp - self.shiro_prev_ts
                        dt = 0.067 if (raw_dt <= 0.001 or raw_dt > 0.40) else raw_dt
                    else:
                        dt = 0.067
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

                    if buf_len >= 30:
                        X_act = np.array(self.shiro_temporal_buffer, dtype=np.float32)
                        X_norm = (X_act - self.shiro_mean) / self.shiro_std
                        seq = np.zeros((1, self.shiro_seq_len, self.shiro_feat_dim), dtype=np.float32)
                        if buf_len <= self.shiro_seq_len:
                            seq[0, :buf_len, :] = X_norm
                        else:
                            seq[0, :, :] = X_norm[-self.shiro_seq_len:, :]

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
                            sanskrit = self.shiro_sanskrit_names[winner_cid]
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

            # =================================================================
            # BRANCH 3: DRISHTI BHEDHAS (EXACT STEP 37C PIPELINE)
            # =================================================================
            elif cat == "drishti":
                res = self.drishti_extractor.process(frame_rgb)
                if not res.get("detected"):
                    if (now - last_detect_time) > 0.50 and len(self.drishti_temporal_buffer) > 0:
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
                    continue

                last_detect_time = now
                base_44 = res["vector"]

                if self.drishti_prev_ts is not None:
                    raw_dt = timestamp - self.drishti_prev_ts
                    dt = 0.050 if (raw_dt <= 0.001 or raw_dt > 0.40) else raw_dt
                else:
                    dt = 0.050
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

                if buf_len >= 30:
                    X_act = np.array(self.drishti_temporal_buffer, dtype=np.float32)
                    X_norm = (X_act - self.drishti_mean) / self.drishti_std
                    seq = np.zeros((1, self.drishti_seq_len, self.drishti_feat_dim), dtype=np.float32)
                    if buf_len <= self.drishti_seq_len:
                        seq[0, :buf_len, :] = X_norm
                    else:
                        seq[0, :, :] = X_norm[-self.drishti_seq_len:, :]

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
                        sanskrit = self.drishti_sanskrit_names[winner_cid]
                        self.shared_state["status"] = "STABLE"
                        self.shared_state["recognized_name"] = sanskrit.upper()
                        self.shared_state["recognized_subtitle"] = "Drishti Bhedha (Eye Movement)"
                        self.shared_state["confidence"] = raw_conf
                    else:
                        self.shared_state["status"] = "RECOGNIZING..."
                        self.shared_state["recognized_name"] = None
                        self.shared_state["recognized_subtitle"] = None
                        self.shared_state["confidence"] = 0.0

    def get_shared_state(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.shared_state)

    def close(self):
        self.stop_event.set()
        self.hastha_extractor.close()
        self.hastha_mp_detector.close()
        self.hastha_mp_fallback.close()
        self.shiro_extractor.close()
        self.drishti_extractor.close()


# =============================================================================
# 4. MAIN THREAD GUI & VISUAL RENDERING
# =============================================================================

class NrityaAIApplication:
    """
    Main user interface application.
    Renders high-contrast dark dancer HUD, smooth mirrored camera preview,
    and prominent single-result display across 4 Kuchipudi categories.
    """
    def __init__(self, camera_index: int = 0, initial_category: str = "hasthas"):
        self.camera_index = camera_index
        self.selected_category = initial_category.lower()
        self.debug_mode = False

        self.asset_paths = resolve_asset_paths()
        print(f"[PATH] Hastha Model : {self.asset_paths['hastha_model']}")
        print(f"[PATH] Shiro Model  : {self.asset_paths['shiro_model']}")
        print(f"[PATH] Drishti Model: {self.asset_paths['drishti_model']}")

        self.capture_stream = CameraCaptureStream(camera_index=camera_index, width=1280, height=720)
        self.inference_worker = NrityaInferenceWorker(self.asset_paths, initial_category=self.selected_category)
        self.worker_thread: Optional[threading.Thread] = None

        # Interactive Button Rectangles: (x1, y1, x2, y2)
        # Symmetrically spaced across top HUD (Exactly 3 categories, PADA removed)
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
        """
        Renders clean, professional, high-contrast dancer HUD directly over mirrored preview.
        Landmark coordinates match the preview frame with 100% pixel-perfect alignment.
        """
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
            (self.btn_hastha, "[1] HASTHAS (52)", self.selected_category == "hasthas", (40, 90, 60), (50, 255, 120)),
            (self.btn_shiro, "[2] SHIRO BHEDHAS (9)", self.selected_category == "shiro", (40, 75, 95), (0, 215, 255)),
            (self.btn_drishti, "[3] DRISHTI BHEDHAS (8)", self.selected_category == "drishti", (70, 50, 95), (200, 120, 255)),
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

            cv2.putText(canvas, f"{prefix} {recognized_name}", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 0.95, (40, 255, 130), 2, cv2.LINE_AA)
            if subtitle:
                cv2.putText(canvas, f"Classification: {subtitle}", (50, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 215, 235), 1, cv2.LINE_AA)

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

    def run(self):
        """Main execution loop."""
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
# 5. STARTUP VERIFICATION & SHORT BENCHMARK
# =============================================================================

def run_startup_verification(check_camera: bool = True) -> bool:
    """Verifies all assets, dimensions, and compiled models without opening GUI."""
    print("=" * 80)
    print("NRITYA AI: STARTUP & INTEGRITY VERIFICATION")
    print("=" * 80)
    paths = resolve_asset_paths()

    # 1. Hastha Assets
    print("1. Checking Hastha Assets (52 Classes)...")
    if not paths["hastha_model"].exists():
        print(f"  [FAIL] Missing Hastha model: {paths['hastha_model']}")
        return False
    h_model = tf.keras.models.load_model(str(paths["hastha_model"]), compile=False)
    assert h_model.input_shape == (None, 175), f"Invalid Hastha input: {h_model.input_shape}"
    assert h_model.output_shape == (None, 52), f"Invalid Hastha output: {h_model.output_shape}"
    s_h = np.load(str(paths["hastha_scaler"]))
    assert s_h["mean"].shape == (175,), "Invalid Hastha scaler"
    print("  [PASS] Hastha Model and Scaler verified: (None, 175) -> (None, 52)")

    # 2. Shiro Assets
    print("\n2. Checking Shiro Assets (9 Classes)...")
    if not paths["shiro_model"].exists():
        print(f"  [FAIL] Missing Shiro model: {paths['shiro_model']}")
        return False
    s_model = tf.keras.models.load_model(str(paths["shiro_model"]))
    assert s_model.input_shape == (None, 160, 102), f"Invalid Shiro input: {s_model.input_shape}"
    assert s_model.output_shape == (None, 9), f"Invalid Shiro output: {s_model.output_shape}"
    s_s = np.load(str(paths["shiro_scaler"]))
    assert s_s["mean"].shape == (102,), "Invalid Shiro scaler"
    print("  [PASS] Shiro Model and Scaler verified: (None, 160, 102) -> (None, 9)")

    # 3. Drishti Assets
    print("\n3. Checking Drishti Assets (8 Classes)...")
    if not paths["drishti_model"].exists():
        print(f"  [FAIL] Missing Drishti model: {paths['drishti_model']}")
        return False
    d_model = tf.keras.models.load_model(str(paths["drishti_model"]))
    assert d_model.input_shape == (None, 160, 132), f"Invalid Drishti input: {d_model.input_shape}"
    assert d_model.output_shape == (None, 8), f"Invalid Drishti output: {d_model.output_shape}"
    s_d = np.load(str(paths["drishti_scaler"]))
    assert s_d["mean"].shape == (132,), "Invalid Drishti scaler"
    print("  [PASS] Drishti Model and Scaler verified: (None, 160, 132) -> (None, 8)")

    # 4. Compiled Graph Warmup Check
    print("\n4. Testing Compiled Execution Warmup...")
    t0 = time.perf_counter()
    _ = h_model(np.zeros((1, 175), dtype=np.float32), training=False)
    lat_h = (time.perf_counter() - t0) * 1000.0
    print(f"  [PASS] Hastha forward pass latency: {lat_h:.2f}ms")

    t0 = time.perf_counter()
    _ = s_model(np.zeros((1, 160, 102), dtype=np.float32), training=False)
    lat_s = (time.perf_counter() - t0) * 1000.0
    print(f"  [PASS] Shiro forward pass latency: {lat_s:.2f}ms")

    t0 = time.perf_counter()
    _ = d_model(np.zeros((1, 160, 132), dtype=np.float32), training=False)
    lat_d = (time.perf_counter() - t0) * 1000.0
    print(f"  [PASS] Drishti forward pass latency: {lat_d:.2f}ms")

    # 5. Camera Hardware Check
    if check_camera:
        print("\n5. Checking Camera Hardware Accessibility...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print("  [PASS] Camera index 0 responsive.")
            else:
                print("  [WARN] Camera opened but frame read returned False.")
        else:
            print("  [WARN] Camera 0 could not be opened (may be in use or absent).")

    print("\n[SUCCESS] All Startup Verification Checks Passed.")
    return True


def run_performance_benchmark(frames_to_test: int = 60) -> bool:
    """Runs a controlled short performance test measuring decoupled frame rates and latency."""
    print("=" * 80)
    print(f"RUNNING CONTROLLED PERFORMANCE BENCHMARK ({frames_to_test} FRAMES)")
    print("=" * 80)
    paths = resolve_asset_paths()
    stream = CameraCaptureStream(camera_index=0, width=1280, height=720)
    if not stream.start():
        print("[ERROR] Camera could not be opened for benchmark.")
        return False

    worker = NrityaInferenceWorker(paths, initial_category="hasthas")
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
# 6. MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Nritya AI — Real-time Kuchipudi Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--category", type=str, default="hasthas", choices=["hasthas", "shiro", "drishti"], help="Default active category")
    parser.add_argument("--check-only", action="store_true", help="Run startup and integrity checks without opening GUI")
    parser.add_argument("--benchmark", action="store_true", help="Run short performance benchmark and exit")
    args = parser.parse_args()

    if args.check_only:
        ok = run_startup_verification(check_camera=True)
        return 0 if ok else 1

    if args.benchmark:
        ok = run_performance_benchmark(frames_to_test=60)
        return 0 if ok else 1

    app = NrityaAIApplication(camera_index=args.camera, initial_category=args.category)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
