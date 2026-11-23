"""
Kuchipudi AI Inference Service
Integrates existing pre-trained nritya_ai_weights.keras model without modification.
Supports all 69 classical recognition classes (52 Hasthas, 9 Shiro Bhedas, 8 Drishti Bhedas).
Preserves exact preprocessing and feature extraction from deployment_clean/test.py.
"""
import os
import sys
import re
import time
import base64
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, Counter
from pathlib import Path

# Ensure project root is in sys.path so deployment_clean is importable
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Link pre-installed AI packages from deployment_clean virtual environment
CLEAN_VENV_PKGS = os.path.abspath(
    os.path.join(PROJECT_ROOT, "deployment_clean", ".venv", "Lib", "site-packages")
)
if os.path.exists(CLEAN_VENV_PKGS) and CLEAN_VENV_PKGS not in sys.path:
    sys.path.insert(0, CLEAN_VENV_PKGS)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ENABLE_AI = os.getenv("ENABLE_AI", "true").lower() in ("true", "1", "yes")

# Canonical 69-class metadata (always accessible even when heavy ML packages are unloaded)
CANONICAL_HASTHA_CLASSES = {
    "0": "1. Anjali", "1": "1. Pathaka", "2": "10. Shikhara", "3": "10.Karthariswasthikamu",
    "4": "11. Kapittha", "5": "11.Sakatamu", "6": "12. Katakamukha", "7": "12.Sankhamu",
    "8": "13. Soochi", "9": "13.Chakramu", "10": "14. Samputamu", "11": "14.Chandrakala",
    "12": "15. Paasamu", "13": "15.Padmakosha", "14": "16.Keelakamu", "15": "16.Sarpasheersha",
    "16": "17.Matsyamu", "17": "17.Mrugasheersha", "18": "18.Kurmamu", "19": "18.Simhamukha",
    "20": "19.Kaangula", "21": "19.Varahamu", "22": "2. Kapothamu", "23": "2. Thripathaka",
    "24": "20.Alapadma", "25": "20.Garudamu", "26": "21.Chatura", "27": "21.Nagabandhakamu",
    "28": "22.Bhramara", "29": "22.Khatva", "30": "23.Bherundamu", "31": "23.Hamsasya",
    "32": "24.Avahithamu", "33": "24.Hamsapaksha", "34": "25.Samdamsha", "35": "26.Mukula",
    "36": "27.Tamrachuda", "37": "28.Trishoola", "38": "3. Ardhapathaka", "39": "3. Karkatamu",
    "40": "4. Kartarimukha", "41": "4. Swasthikamu", "42": "5. Dola hastamu", "43": "5. Mayura",
    "44": "6. Ardhachandra", "45": "6. Pushpaputamu", "46": "7. Arala", "47": "7. Uthsungamu",
    "48": "8. Sivalingamu", "49": "8. Sukatunda", "50": "9. Katakavardhanamu", "51": "9. Mushti"
}
CANONICAL_SHIRO_BHEDHAS = ['Sama', 'Udvahitam', 'Adhomukham', 'Alolitam', 'Dhutam', 'Kampitam', 'Paravrittam', 'Ukshiptam', 'Parivahittam']
CANONICAL_DRISHTI_BHEDHAS = ['Sama', 'Alokita', 'Sachi', 'Pralokita', 'Nimilita', 'Ullokita', 'Anuvritta', 'Avalokita']
CANONICAL_SAMYUTHA_KEYWORDS = [
    "anjali", "kapothamu", "karkatamu", "swasthikamu", "dola", "pushpaputamu",
    "uthsungamu", "sivalingamu", "katakavardhanamu", "karthariswasthikamu",
    "sakatamu", "sankhamu", "chakramu", "samputamu", "paasamu", "keelakamu",
    "matsyamu", "kurmamu", "varahamu", "garudamu", "nagabandhakamu", "khatva",
    "bherundamu", "avahithamu"
]

import numpy as np

tf = None
mp_solutions = None
mp_hands = None
mp_pose = None
mp_face = None

if ENABLE_AI:
    try:
        import cv2
        import tensorflow as tf
        try:
            from mediapipe import solutions as mp_solutions
            mp_hands = mp_solutions.hands
            mp_pose = mp_solutions.pose
            mp_face = mp_solutions.face_mesh
        except ImportError:
            import mediapipe.python.solutions as mp_solutions
            mp_hands = mp_solutions.hands
            mp_pose = mp_solutions.pose
            mp_face = mp_solutions.face_mesh

        from deployment_clean.test import (
            HASTHA_SCALER_MEAN, HASTHA_SCALER_SCALE,
            SHIRO_SCALER_MEAN, SHIRO_SCALER_STD,
            DRISHTI_SCALER_MEAN, DRISHTI_SCALER_STD,
            HASTHA_CLASSES, SHIRO_BHEDHAS, DRISHTI_BHEDHAS,
            SAMYUTHA_KEYWORDS,
            SHIRO_SEQ_LEN, SHIRO_FEAT_DIM, SHIRO_MIN_PREDICT_FRAMES, SHIRO_DT,
            DRISHTI_SEQ_LEN, DRISHTI_FEAT_DIM, DRISHTI_MIN_PREDICT_FRAMES, DRISHTI_DT,
            calculate_angle_3d,
            HandLandmarkExtractor, PoseLandmarkExtractor, FaceDrishtiLandmarkExtractor
        )
    except Exception as e:
        print(f"[AI ENGINE] Notice: Full ML dependencies not available ({e}). Falling back to standby engine.")
        ENABLE_AI = False
else:
    print("[AI ENGINE] Standby mode active (ENABLE_AI=false). Running lightweight mode (<60MB RAM) for cloud hosting.")
    HASTHA_CLASSES = CANONICAL_HASTHA_CLASSES
    SHIRO_BHEDHAS = CANONICAL_SHIRO_BHEDHAS
    DRISHTI_BHEDHAS = CANONICAL_DRISHTI_BHEDHAS
    SAMYUTHA_KEYWORDS = CANONICAL_SAMYUTHA_KEYWORDS

WEIGHTS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "deployment_clean" / "nritya_ai_weights.keras"


class KuchipudiAIInferenceEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KuchipudiAIInferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, weights_path: Optional[Path] = None):
        if self._initialized:
            return

        if not ENABLE_AI:
            self.enabled = False
            self._initialized = False
            self.unified_model = None
            print("[AI ENGINE] Standby mode active (ENABLE_AI=false). Operating in lightweight mode (<60MB RAM).")
            return

        target_weights = weights_path or WEIGHTS_PATH
        if not target_weights.exists():
            raise FileNotFoundError(f"Consolidated weights missing at: {target_weights}")

        print(f"[AI ENGINE] Initializing Kuchipudi AI Engine from {target_weights.name}...")
        t0 = time.time()
        self.unified_model = tf.keras.models.load_model(str(target_weights), compile=False)
        self.enabled = True

        # Retrieve distinct submodels from weights
        self.hastha_model = self.unified_model.get_layer("Personalized_52Hasthas_MLP")
        self.shiro_model = self.unified_model.get_layer("Shiro_BiGRU_Classifier")
        self.drishti_model = self.unified_model.get_layer("Drishti_BiGRU_Classifier")

        # Compile graph forward passes for minimal execution latency
        @tf.function(reduce_retracing=True)
        def _h_predict(x): return self.hastha_model(x, training=False)
        self.compiled_hastha_predict = _h_predict

        @tf.function(reduce_retracing=True)
        def _s_predict(x): return self.shiro_model(x, training=False)
        self.compiled_shiro_predict = _s_predict

        @tf.function(reduce_retracing=True)
        def _d_predict(x): return self.drishti_model(x, training=False)
        self.compiled_drishti_predict = _d_predict

        # Graph warm-up passes
        _ = self.compiled_hastha_predict(np.zeros((1, 175), dtype=np.float32))
        _ = self.compiled_shiro_predict(np.zeros((1, SHIRO_SEQ_LEN, SHIRO_FEAT_DIM), dtype=np.float32))
        _ = self.compiled_drishti_predict(np.zeros((1, DRISHTI_SEQ_LEN, DRISHTI_FEAT_DIM), dtype=np.float32))
        print(f"[AI ENGINE] Loaded and warmed up in {round(time.time() - t0, 2)}s.")

        # MediaPipe Detectors
        self.hastha_extractor = HandLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.35)
        self.hastha_mp_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.35, min_tracking_confidence=0.30)
        self.hastha_mp_fallback = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.15, min_tracking_confidence=0.15)
        self.hastha_buffer = deque(maxlen=8)

        self.shiro_extractor = PoseLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.30)
        self.shiro_temporal_buffer = deque(maxlen=SHIRO_SEQ_LEN)
        self.shiro_buffer = deque(maxlen=8)
        self.shiro_prev_ts: Optional[float] = None
        self.shiro_prev_base: Optional[np.ndarray] = None
        self.shiro_prev_vel: Optional[np.ndarray] = None
        self.shiro_baseline_angles: Optional[np.ndarray] = None

        self.drishti_extractor = FaceDrishtiLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.30)
        self.drishti_temporal_buffer = deque(maxlen=DRISHTI_SEQ_LEN)
        self.drishti_buffer = deque(maxlen=8)
        self.drishti_prev_ts: Optional[float] = None
        self.drishti_prev_base: Optional[np.ndarray] = None
        self.drishti_prev_vel: Optional[np.ndarray] = None

        self._initialized = True

    def clean_name(self, full_name: str) -> str:
        return re.sub(r"^\d+\.\s*", "", full_name).strip()

    def is_samyukta(self, full_name: str) -> bool:
        return any(kw.lower() in full_name.lower() for kw in SAMYUTHA_KEYWORDS)

    def decode_image_frame(self, image_input: Any) -> np.ndarray:
        """Helper to convert base64 data URL, raw bytes, or path to RGB numpy array."""
        if not getattr(self, "enabled", True):
            return np.zeros((10, 10, 3), dtype=np.uint8)

        if isinstance(image_input, np.ndarray):
            return image_input

        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                # Strip base64 data prefix
                image_input = image_input.split(",")[1]
            image_bytes = base64.b64decode(image_input)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ValueError("Could not decode image from base64 string.")
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ValueError("Could not decode image from bytes.")
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def predict_hastha(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Execute real-time Hastha classification (52 classes) using 175-D landmarks.
        """
        if not getattr(self, "enabled", True):
            return {
                "detected": False,
                "ai_enabled": False,
                "category": "hasthas",
                "predicted_gesture": None,
                "confidence_score": 0.0,
                "confidence_percentage": 0.0,
                "feedback_message": "AI recognition paused on free cloud tier (512MB RAM constraint). Run locally for real-time 69-class inference.",
                "message": "AI recognition paused on free cloud tier (512MB RAM constraint). Run locally for real-time 69-class inference.",
                "latency_ms": 0.0
            }

        t0 = time.perf_counter()

        # Run Primary MediaPipe Detector
        results = self.hastha_mp_detector.process(frame_rgb)
        if not results.multi_hand_landmarks:
            results = self.hastha_mp_fallback.process(frame_rgb)

        if not results.multi_hand_landmarks:
            self.hastha_buffer.clear()
            return {
                "detected": False,
                "category": "hasthas",
                "hands_count": 0,
                "message": "No hands detected in camera frame. Position hands clearly in view.",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "landmarks": []
            }

        num_hands = len(results.multi_hand_landmarks)
        landmarks_list = []
        for h_lms in results.multi_hand_landmarks:
            pts = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in h_lms.landmark]
            landmarks_list.append(pts)

        if num_hands == 1:
            hand_lms = results.multi_hand_landmarks[0]
            raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

            handedness_label = "Left"
            if results.multi_handedness and len(results.multi_handedness) > 0:
                try:
                    handedness_label = results.multi_handedness[0].classification[0].label
                except Exception:
                    handedness_label = "Left"

            # Canonical alignment for single hand (mirror right hand to standard coordinates)
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
        top_indices = np.argsort(probs)[::-1][:3]
        top_cid = int(top_indices[0])
        top_conf = float(probs[top_cid])

        # Buffer temporal stabilization
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
            if count >= 4:
                is_stable = True
                winner_cid = most_common_id

        raw_name = HASTHA_CLASSES.get(winner_cid if is_stable else top_cid, "Unknown")
        cleaned = self.clean_name(raw_name)
        samyukta_flag = self.is_samyukta(raw_name)

        top_preds = [
            {
                "gesture_id": int(i),
                "gesture_name": self.clean_name(HASTHA_CLASSES.get(int(i), "Unknown")),
                "confidence": round(float(probs[i]) * 100.0, 2)
            }
            for i in top_indices
        ]

        latency = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "detected": True,
            "category": "hasthas",
            "hands_count": num_hands,
            "predicted_gesture": cleaned,
            "raw_class_name": raw_name,
            "confidence_score": round(top_conf, 4),
            "confidence_percentage": round(top_conf * 100.0, 2),
            "is_stable": is_stable,
            "is_samyukta": samyukta_flag,
            "top_predictions": top_preds,
            "latency_ms": latency,
            "landmarks": landmarks_list
        }

    def predict_shiro(self, frame_rgb: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute real-time Shiro Bhedhas (9 head movements) using 102-D temporal kinematics sequence.
        """
        if not getattr(self, "enabled", True):
            return {
                "detected": False,
                "ai_enabled": False,
                "category": "shiro",
                "predicted_head_movement": None,
                "confidence_score": 0.0,
                "confidence_percentage": 0.0,
                "feedback_message": "AI recognition paused on free cloud tier (512MB RAM constraint). Run locally for real-time 69-class inference.",
                "message": "AI recognition paused on free cloud tier (512MB RAM constraint). Run locally for real-time 69-class inference.",
                "latency_ms": 0.0,
                "landmarks": []
            }

        t0 = time.perf_counter()
        now = timestamp or time.time()

        res = self.shiro_extractor.process(frame_rgb, category_focus="shiro")
        if not res.get("detected"):
            self.shiro_buffer.append((-1, 0.0))
            return {
                "detected": False,
                "category": "shiro",
                "message": "Face/neck pose not detected. Look directly at the camera.",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "landmarks": []
            }

        base_32 = res["vector"]
        shiro_feats = res["shiro_features"]

        if self.shiro_prev_ts is not None:
            raw_dt = now - self.shiro_prev_ts
            dt = SHIRO_DT if (raw_dt <= 0.001 or raw_dt > 0.40) else raw_dt
        else:
            dt = SHIRO_DT
        self.shiro_prev_ts = now

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
        raw_cid = 0
        raw_conf = 0.5

        if buf_len >= 10:  # Fast response threshold
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

        movement_name = SHIRO_BHEDHAS[raw_cid] if raw_cid < len(SHIRO_BHEDHAS) else "Sama"
        latency = round((time.perf_counter() - t0) * 1000.0, 2)

        landmarks_raw = res.get("pose_landmarks") or res.get("raw_landmarks")
        landmarks_out = []
        if landmarks_raw:
            try:
                landmarks_out = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in landmarks_raw.landmark[:11]]
            except Exception:
                landmarks_out = []

        return {
            "detected": True,
            "category": "shiro",
            "predicted_movement": movement_name,
            "confidence_score": round(raw_conf, 4),
            "confidence_percentage": round(raw_conf * 100.0, 2),
            "buffer_frames": buf_len,
            "pitch_degrees": round(float(np.degrees(angles[1])), 1),
            "yaw_degrees": round(float(np.degrees(angles[2])), 1),
            "roll_degrees": round(float(np.degrees(angles[0])), 1),
            "latency_ms": latency,
            "landmarks": landmarks_out
        }

    def predict_drishti(self, frame_rgb: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute real-time Drishti Bhedhas (8 eye movements) using 132-D facial gaze vectors.
        """
        if not getattr(self, "enabled", True):
            return {
                "detected": False,
                "ai_enabled": False,
                "category": "drishti",
                "predicted_glance": None,
                "confidence_score": 0.0,
                "confidence_percentage": 0.0,
                "feedback_message": "AI recognition paused on free cloud tier (512MB RAM constraint). Run locally for real-time 69-class inference.",
                "message": "AI recognition paused on free cloud tier (512MB RAM constraint). Run locally for real-time 69-class inference.",
                "latency_ms": 0.0,
                "landmarks": []
            }

        t0 = time.perf_counter()
        now = timestamp or time.time()

        res = self.drishti_extractor.process(frame_rgb)
        if not res.get("detected"):
            self.drishti_buffer.append((-1, 0.0))
            return {
                "detected": False,
                "category": "drishti",
                "message": "Eyes/face not clearly visible. Look at camera in good lighting.",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "landmarks": []
            }

        base_44 = res["vector"]

        if self.drishti_prev_ts is not None:
            raw_dt = now - self.drishti_prev_ts
            dt = DRISHTI_DT if (raw_dt <= 0.001 or raw_dt > 0.40) else raw_dt
        else:
            dt = DRISHTI_DT
        self.drishti_prev_ts = now

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
        raw_cid = 0
        raw_conf = 0.5

        if buf_len >= 10:
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

        glance_name = DRISHTI_BHEDHAS[raw_cid] if raw_cid < len(DRISHTI_BHEDHAS) else "Sama"
        latency = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "detected": True,
            "category": "drishti",
            "predicted_glance": glance_name,
            "confidence_score": round(raw_conf, 4),
            "confidence_percentage": round(raw_conf * 100.0, 2),
            "buffer_frames": buf_len,
            "latency_ms": latency
        }

    def get_supported_classes(self) -> Dict[str, Any]:
        """Return all 69 supported classes organized by category."""
        return {
            "hasthas": [
                {"id": k, "name": self.clean_name(v), "raw_label": v, "is_samyukta": self.is_samyukta(v)}
                for k, v in HASTHA_CLASSES.items()
            ],
            "shiro_bhedhas": [
                {"id": idx, "name": name}
                for idx, name in enumerate(SHIRO_BHEDHAS)
            ],
            "drishti_bhedhas": [
                {"id": idx, "name": name}
                for idx, name in enumerate(DRISHTI_BHEDHAS)
            ],
            "total_classes": 69
        }

# Global Singleton Instance Accessor
_engine_instance: Optional[KuchipudiAIInferenceEngine] = None

def get_ai_engine() -> KuchipudiAIInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = KuchipudiAIInferenceEngine()
    return _engine_instance
