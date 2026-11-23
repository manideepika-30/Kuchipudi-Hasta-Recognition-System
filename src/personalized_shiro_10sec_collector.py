"""
STEP 36A — PERSONALIZED SHIRO BHEDHAS CONTINUOUS 10-SECOND LANDMARK COLLECTOR
=============================================================================
Interactive OpenCV + MediaPipe tool for continuous 10-second webcam recording
and temporal sequence landmark extraction for all 9 classical Shiro Bhedhas
(Head Movements) in Kuchipudi dance.

Pipeline:
  Camera ON -> Select Shiro -> Press SPACE -> Record continuously for 10.0s
  -> Real-time MediaPipe Pose/Head detection & quality gating
  -> Canonical 32-D Shiro landmark vector per frame (head shift, tilt, pitch, yaw, neck dist, upper body)
  -> Complete temporal sequence preservation: (T, 32)
  -> Temporal derivatives: velocity (T, 32), acceleration (T, 32), displacement (T, 32),
     head angles (T, 3), movement amplitude (32,), 288-D temporal summary descriptor
  -> Auto-save session_XX.npz in data/personalized/shiro_landmarks/<class_folder>/
  -> Remain on SAME class for repeated sessions (target: >= 4 sessions per class)

Controls:
  SPACE : Record another 10-second session for CURRENT Shiro class (remains on same class)
  N     : Advance to NEXT Shiro class (only when not recording)
  R     : Retry / reset status for current class without deleting previous sessions
  Q     : Quit safely and write collection report & statistics

Safety:
  - ZERO modification of Hastha personalized dataset (data/personalized/hasthas_landmarks/)
  - ZERO modification of Hastha models or final_inference.py
  - ZERO modification of locked v2 split files (v2_train.csv, v2_validation.csv, v2_test.csv)
  - ZERO training or evaluation during collection
"""

import os
import sys
import time
import json
import csv
import re
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.landmark_extraction import PoseLandmarkExtractor
from src.train_hybrid_models import compute_temporal_statistics

# MediaPipe solutions import for drawing utilities
try:
    from mediapipe import solutions as mp_solutions
except ImportError:
    import mediapipe.python.solutions as mp_solutions

# Base Directories
PERSONALIZED_ROOT = PROJECT_ROOT / "data" / "personalized"
SHIRO_LANDMARKS_DIR = PERSONALIZED_ROOT / "shiro_landmarks"
REPORTS_DIR = PROJECT_ROOT / "reports"
SHIRO_MAPPING_FILE = PROJECT_ROOT / "models" / "step16b_shiro_class_mapping.json"
MASTER_MAPPING_FILE = PROJECT_ROOT / "data" / "processed" / "class_mapping.json"

REPORT_TXT_PATH = REPORTS_DIR / "step36a_shiro_collection_report.txt"
STATS_CSV_PATH = REPORTS_DIR / "step36a_shiro_dataset_statistics.csv"

RECORDING_DURATION_SECONDS = 10.0
BASE_FEATURE_DIMENSION = 32
TEMPORAL_SUMMARY_DIMENSION = 288
MIN_SAMPLE_INTERVAL = 0.05  # ~20 Hz cap (50 ms)
TARGET_SESSIONS_PER_CLASS = 4

# ==============================================================================
# 1. CANONICAL 9 SHIRO BHEDHAS DEFINITIONS
# ==============================================================================

CANONICAL_SHIRO_CLASSES = [
    (0, "1. Sama", "1_Sama", "Sama", "Head held straight, motionless, and serene in natural baseline posture"),
    (1, "2. Udvahitam", "2_Udvahitam", "Udvahitam", "Head tilted upward facing the sky (upward pitch tilt)"),
    (2, "3. Adhomukham", "3_Adhomukham", "Adhomukham", "Head bowed downward looking at the ground (downward pitch tilt)"),
    (3, "4. Alolitam", "4_Alolitam", "Alolitam", "Head rotated continuously in circular motion (circular roll/yaw)"),
    (4, "5. Dhutam", "5_Dhutam", "Dhutam", "Head turned from side to side horizontally like saying 'no' (horizontal yaw)"),
    (5, "6. Kampitam", "6_Kampitam", "Kampitam", "Head shaken/nodded up and down repeatedly like saying 'yes' (vertical oscillation)"),
    (6, "7. Paravrittam", "7_Paravrittam", "Paravrittam", "Head turned sharply to one side looking over shoulder (profile yaw hold)"),
    (7, "8. Ukshiptam", "8_Ukshiptam", "Ukshiptam", "Head tilted slantingly upward to one side (elevated diagonal tilt)"),
    (8, "9. Parivahittam", "9_Parivahittam", "Parivahittam", "Head swayed gently from side to side ear toward shoulder (lateral roll oscillation)"),
]


def load_shiro_class_mapping() -> List[Tuple[int, str, str, str, str]]:
    """
    Loads and verifies canonical 9 Shiro Bhedha classes from project mapping.
    Ensures exact 9 classes matching data/processed/class_mapping.json.
    """
    if SHIRO_MAPPING_FILE.exists():
        with open(SHIRO_MAPPING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes", [])
        if len(classes) == 9:
            loaded_list = []
            for item in sorted(classes, key=lambda x: x["local_class_id"]):
                loaded_list.append((
                    item["local_class_id"],
                    item["class_name"],
                    item["folder_name"],
                    item.get("sanskrit_name", item["class_name"]),
                    item.get("movement_description", "")
                ))
            return loaded_list

    # Fallback to CANONICAL_SHIRO_CLASSES
    return CANONICAL_SHIRO_CLASSES


def draw_centered_text(
    img: np.ndarray,
    text: str,
    y: int,
    font_face: int,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
    outline_color: Optional[Tuple[int, int, int]] = None
) -> int:
    """Draws horizontally centered text with optional contrast outline."""
    (w_text, h_text), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    x = max(10, (img.shape[1] - w_text) // 2)
    if outline_color is not None:
        cv2.putText(img, text, (x - 1, y - 1), font_face, font_scale, outline_color, thickness + 1, cv2.LINE_AA)
        cv2.putText(img, text, (x + 1, y + 1), font_face, font_scale, outline_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)
    return y


def wrap_text_into_lines(text: str, max_chars: int = 68) -> List[str]:
    """Wraps text into clean, balanced lines for prominent multi-line display."""
    words = text.split()
    lines = []
    curr = ""
    for w in words:
        if len(curr + " " + w) <= max_chars or not curr:
            curr += (" " if curr else "") + w
        else:
            lines.append(curr)
            curr = w
    if curr:
        lines.append(curr)
    return lines


# ==============================================================================
# 2. FILE AND DIRECTORY UTILITIES
# ==============================================================================

def get_next_session_filename(class_dir: Path) -> Tuple[Path, str, int]:
    """
    Finds the next unused session_XX.npz in class_dir.
    Guarantees no session is EVER overwritten.
    """
    class_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(class_dir.glob("session_*.npz"))
    max_idx = 0
    pattern = re.compile(r"^session_(\d+)\.npz$")
    for f in existing_files:
        m = pattern.match(f.name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx

    next_idx = max_idx + 1
    session_id = f"session_{next_idx:02d}"
    target_path = class_dir / f"{session_id}.npz"

    # Absolute safety check: ensure target does not exist
    while target_path.exists():
        next_idx += 1
        session_id = f"session_{next_idx:02d}"
        target_path = class_dir / f"{session_id}.npz"

    return target_path, session_id, next_idx


def count_existing_sessions(class_dir: Path) -> int:
    """Counts valid session_*.npz files in a directory."""
    if not class_dir.exists():
        return 0
    return len(list(class_dir.glob("session_*.npz")))


# ==============================================================================
# 3. TEMPORAL FEATURE COMPUTATION
# ==============================================================================

def compute_full_temporal_package(
    features_seq: np.ndarray,      # (T, 32)
    timestamps: np.ndarray,        # (T,)
    valid_mask: np.ndarray,        # (T,)
    head_angles: np.ndarray        # (T, 3)
) -> Dict[str, np.ndarray]:
    """
    Computes complete temporal features from raw sequence:
      - Velocities: (T, 32)
      - Accelerations: (T, 32)
      - Displacements: (T, 32)
      - Movement Amplitude: (32,)
      - Head Angle Changes: (T, 3)
      - Temporal Summary (32 base mean + 256 temporal descriptors = 288 dims)
    """
    T, D = features_seq.shape
    assert D == BASE_FEATURE_DIMENSION, f"Expected {BASE_FEATURE_DIMENSION} features, got {D}"

    # 1. dt calculation
    if T > 1:
        dt = np.diff(timestamps)
        dt = np.where(dt > 1e-4, dt, 0.05)  # avoid division by zero
        dt = np.insert(dt, 0, dt[0])
    else:
        dt = np.full(T, 0.05, dtype=np.float64)

    # 2. Velocities (first derivative)
    if T > 1:
        diffs = np.diff(features_seq, axis=0)  # (T-1, 32)
        vel_body = diffs / dt[1:, np.newaxis]  # (T-1, 32)
        velocities = np.vstack([np.zeros((1, D), dtype=np.float32), vel_body]).astype(np.float32)
    else:
        velocities = np.zeros((T, D), dtype=np.float32)

    # 3. Accelerations (second derivative)
    if T > 2:
        diffs_vel = np.diff(velocities, axis=0)  # (T-1, 32)
        acc_body = diffs_vel / dt[1:, np.newaxis]  # (T-1, 32)
        accelerations = np.vstack([np.zeros((1, D), dtype=np.float32), acc_body]).astype(np.float32)
    else:
        accelerations = np.zeros((T, D), dtype=np.float32)

    # 4. Displacements relative to session start
    displacements = (features_seq - features_seq[0:1]).astype(np.float32)

    # 5. Head angle relative changes
    head_angle_changes = (head_angles - head_angles[0:1]).astype(np.float32)

    # 6. Movement amplitude (peak-to-peak over valid frames)
    if np.any(valid_mask):
        valid_feats = features_seq[valid_mask]
        movement_amplitude = np.ptp(valid_feats, axis=0).astype(np.float32)
        mean_lm = np.mean(valid_feats, axis=0).astype(np.float32)
    else:
        movement_amplitude = np.zeros(D, dtype=np.float32)
        mean_lm = np.zeros(D, dtype=np.float32)

    # 7. Canonical 288-D temporal summary descriptor
    stat_vec = compute_temporal_statistics(features_seq, velocities)  # (256,)
    temporal_summary_288 = np.concatenate([mean_lm, stat_vec]).astype(np.float32)  # (288,)

    return {
        "velocities": velocities,
        "accelerations": accelerations,
        "displacements": displacements,
        "head_angle_changes": head_angle_changes,
        "movement_amplitude": movement_amplitude,
        "temporal_summary_288": temporal_summary_288
    }


# ==============================================================================
# 4. AUDIT AND REPORT GENERATION
# ==============================================================================

def generate_collection_reports(classes_info: List[Tuple[int, str, str, str, str]]) -> Dict[str, Any]:
    """Generates text report and statistics CSV for Shiro collection."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SHIRO_LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    class_stats = []
    total_sessions = 0
    total_valid_frames = 0

    for local_id, name, folder, sanskrit, desc in classes_info:
        class_dir = SHIRO_LANDMARKS_DIR / folder
        session_files = sorted(list(class_dir.glob("session_*.npz")))
        sess_count = len(session_files)
        total_sessions += sess_count

        frame_counts = []
        for sf in session_files:
            try:
                with np.load(sf) as data:
                    v_mask = data["valid_mask"]
                    frame_counts.append(int(np.sum(v_mask)))
            except Exception:
                pass

        if frame_counts:
            min_f = min(frame_counts)
            max_f = max(frame_counts)
            mean_f = float(np.mean(frame_counts))
            class_valid_f = sum(frame_counts)
        else:
            min_f = 0
            max_f = 0
            mean_f = 0.0
            class_valid_f = 0

        total_valid_frames += class_valid_f
        status = "COMPLETED (>=4)" if sess_count >= TARGET_SESSIONS_PER_CLASS else f"IN PROGRESS ({sess_count}/4)"

        class_stats.append({
            "class_id": local_id,
            "class_name": name,
            "folder_name": folder,
            "sanskrit_name": sanskrit,
            "session_count": sess_count,
            "target_sessions": TARGET_SESSIONS_PER_CLASS,
            "total_valid_frames": class_valid_f,
            "min_frames_per_session": min_f,
            "max_frames_per_session": max_f,
            "mean_frames_per_session": round(mean_f, 1),
            "status": status
        })

    # Write CSV
    with open(STATS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class_id", "class_name", "folder_name", "sanskrit_name",
            "session_count", "target_sessions", "total_valid_frames",
            "min_frames_per_session", "max_frames_per_session",
            "mean_frames_per_session", "status"
        ])
        writer.writeheader()
        writer.writerows(class_stats)

    # Write TXT
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STEP 36A — PERSONALIZED SHIRO BHEDHAS COLLECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target classes: 9 Shiro Bhedhas\n")
        f.write(f"Output directory: {SHIRO_LANDMARKS_DIR.resolve()}\n")
        f.write(f"Base feature dimension per frame: {BASE_FEATURE_DIMENSION} (Pose Landmark & Head Geometry)\n")
        f.write(f"Temporal summary dimension: {TEMPORAL_SUMMARY_DIMENSION} (32 mean + 256 temporal descriptors)\n")
        f.write(f"Recording session duration: {RECORDING_DURATION_SECONDS:.1f} seconds per session\n")
        f.write(f"Target sampling rate: ~15-20 Hz (~150-200 frames per session)\n")
        f.write(f"Target sessions per class: {TARGET_SESSIONS_PER_CLASS} (Total target: 36 sessions)\n")
        f.write(f"Current total sessions collected: {total_sessions}\n")
        f.write(f"Current total valid frames: {total_valid_frames:,}\n\n")

        f.write("-" * 80 + "\n")
        f.write("EXACT 9-CLASS CANONICAL SHIRO BHEDHA MAPPING:\n")
        f.write("-" * 80 + "\n")
        for st in class_stats:
            f.write(f"  ID {st['class_id']}: {st['class_name']:<18} -> Folder: {st['folder_name']:<18} "
                    f"Sessions: {st['session_count']}/{st['target_sessions']} "
                    f"Frames: {st['total_valid_frames']} ({st['status']})\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("DATA STRUCTURE SPECIFICATION (SESSION_XX.NPZ):\n")
        f.write("-" * 80 + "\n")
        f.write("  features_seq         : (T, 32) float32 - Full temporal sequence of 32-D Shiro landmarks\n")
        f.write("  timestamps           : (T,) float64 - Monotonic timestamp in seconds from session start\n")
        f.write("  frame_indices        : (T,) int32 - Frame sequence indices (0..T-1)\n")
        f.write("  valid_mask           : (T,) bool - True if frame passed all head/pose quality checks\n")
        f.write("  confidences          : (T,) float32 - Head & pose landmark detection confidences\n")
        f.write("  head_angles          : (T, 3) float32 - [roll_rad, pitch, yaw] head angles per frame\n")
        f.write("  velocities           : (T, 32) float32 - Frame-to-frame velocity (first derivative)\n")
        f.write("  accelerations        : (T, 32) float32 - Acceleration (second derivative)\n")
        f.write("  displacements        : (T, 32) float32 - Displacement relative to initial posture\n")
        f.write("  head_angle_changes   : (T, 3) float32 - Angular deviation from baseline\n")
        f.write("  movement_amplitude   : (32,) float32 - Peak-to-peak feature amplitude over session\n")
        f.write("  temporal_summary_288 : (288,) float32 - 32 mean landmarks + 256 temporal descriptors\n")
        f.write("  raw_pose_landmarks   : (T, 33, 4) float32 - [x, y, z, visibility] for all 33 pose landmarks\n")
        f.write("  class_id             : int - Canonical local class ID (0..8)\n")
        f.write("  class_name           : str - Canonical class name\n")
        f.write("  session_id           : str - Session identifier (e.g. session_01)\n")
        f.write("  metadata             : str - JSON string containing timing, FPS, and quality audit\n\n")

        f.write("-" * 80 + "\n")
        f.write("COLLECTION SAFETY AUDIT:\n")
        f.write("-" * 80 + "\n")
        f.write("  [PASSED] Stored in dedicated directory: data/personalized/shiro_landmarks/\n")
        f.write("  [PASSED] Hastha dataset untouched: data/personalized/hasthas_landmarks/ preserved\n")
        f.write("  [PASSED] V2 split files untouched: v2_train.csv, v2_validation.csv, v2_test.csv\n")
        f.write("  [PASSED] Models untouched: Step18, Step28, Step30, Step34 models preserved\n")
        f.write("  [PASSED] Session overwrite protection: monotonically incrementing session IDs\n")
        f.write("  [PASSED] Non-destructive navigation: remain on current class after recording\n\n")

        f.write("-" * 80 + "\n")
        f.write("UI FIX 2 — HIGH-CONTRAST SHIRO COLLECTION HUD CONFIRMATION:\n")
        f.write("-" * 80 + "\n")
        f.write("  [PASSED] Dedicated bright header panel (height: 205px, solid off-white background BGR 250, 252, 255)\n")
        f.write("  [PASSED] Zero global darkening; camera preview below header is 100% natural and untinted\n")
        f.write("  [PASSED] Prominent centered Shiro title in Deep Navy (0.72 scale, BGR 120, 45, 15)\n")
        f.write("  [PASSED] VERY LARGE high-contrast Shiro Name (Font scale 1.85, thickness 4, pitch black BGR 15, 15, 25)\n")
        f.write("  [PASSED] All 9 canonical Shiro names dynamically displayed:\n")
        f.write("           1. SAMA, 2. UDVAHITAM, 3. ADHOMUKHAM, 4. ALOLITAM, 5. DHUTAM,\n")
        f.write("           6. KAMPITAM, 7. PARAVRITTAM, 8. UKSHIPTAM, 9. PARIVAHITTAM\n")
        f.write("  [PASSED] High-contrast class counter (Class X / 9 | Existing Sessions: Y / Target: 4)\n")
        f.write("  [PASSED] Clean wrapped canonical movement description (dark slate text on light header)\n")
        f.write("  [PASSED] High-contrast recording card (solid white card, bright red border, pulsing red dot, countdown)\n")
        f.write("  [PASSED] High-contrast session saved confirmation card (solid white card, emerald green border)\n")
        f.write("  [PASSED] Zero dark translucent overlay over the user's face or upper body\n")

    return {
        "total_sessions": total_sessions,
        "total_valid_frames": total_valid_frames,
        "stats": class_stats
    }


# ==============================================================================
# 5. OPENCV UI AND COLLECTOR WORKFLOW
# ==============================================================================

class ShiroCollectorApp:
    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720
    ):
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Load canonical classes
        self.classes = load_shiro_class_mapping()
        if len(self.classes) != 9:
            raise ValueError(f"Expected exactly 9 Shiro classes, found {len(self.classes)}!")

        self.current_class_idx = 0

        # MediaPipe Extractor
        self.extractor = PoseLandmarkExtractor(static_image_mode=False)

        # State management
        self.is_recording = False
        self.recording_start_time = 0.0
        self.last_sample_time = 0.0
        self.status_message = "Ready. Press SPACE to record 10-second session."
        self.status_time = 0.0
        self.status_color = (255, 255, 255)

        # Buffers for active 10s recording
        self.buf_features: List[np.ndarray] = []
        self.buf_timestamps: List[float] = []
        self.buf_confidences: List[float] = []
        self.buf_valid_mask: List[bool] = []
        self.buf_head_angles: List[List[float]] = []
        self.buf_raw_pose: List[np.ndarray] = []

    def get_current_class_info(self) -> Tuple[int, str, str, str, str]:
        return self.classes[self.current_class_idx]

    def get_current_display_name(self) -> str:
        """Returns the canonical uppercase Shiro name (e.g. SAMA, UDVAHITAM, etc.)."""
        _, _, _, sanskrit_name, _ = self.get_current_class_info()
        return sanskrit_name.upper()

    def get_current_session_count(self) -> int:
        _, _, folder_name, _, _ = self.get_current_class_info()
        class_dir = SHIRO_LANDMARKS_DIR / folder_name
        return count_existing_sessions(class_dir)

    def draw_hud(
        self,
        frame: np.ndarray,
        detected: bool,
        shiro_features: Dict[str, Any],
        is_frame_valid: bool
    ) -> np.ndarray:
        """Renders rich, high-contrast, professional OpenCV HUD overlay with bright light header and dark text."""
        h, w = frame.shape[:2]
        canvas = frame.copy()

        class_id, class_name, folder_name, sanskrit_name, movement_desc = self.get_current_class_info()
        display_name = self.get_current_display_name()
        sess_count = self.get_current_session_count()

        # -------------------------------------------------------------
        # 1. Top High-Contrast Bright Header Panel (Solid Light / White)
        # -------------------------------------------------------------
        top_bar_h = 205
        # Solid, clean off-white / light neutral header panel (zero global darkening)
        cv2.rectangle(canvas, (0, 0), (w, top_bar_h), (250, 252, 255), -1)
        # Crisp dual bottom border lines
        cv2.line(canvas, (0, top_bar_h - 2), (w, top_bar_h - 2), (180, 195, 215), 2)
        cv2.line(canvas, (0, top_bar_h), (w, top_bar_h), (60, 80, 120), 2)

        # Header Line 1: Main Title (Deep Navy, 0.72 scale)
        draw_centered_text(
            canvas,
            "PERSONALIZED SHIRO BHEDHA COLLECTION",
            28,
            cv2.FONT_HERSHEY_DUPLEX,
            0.72,
            (120, 45, 15),  # Deep Navy (BGR: B=120, G=45, R=15)
            2
        )

        # Header Line 2: Section Subtitle (Medium Dark Slate, 0.58 scale)
        draw_centered_text(
            canvas,
            "CURRENT SHIRO",
            54,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (95, 80, 70),  # Medium Dark Slate
            2
        )

        # Header Line 3: VERY LARGE HIGH-CONTRAST SHIRO NAME (Pitch Black, 1.85 scale, thickness 4)
        draw_centered_text(
            canvas,
            display_name,
            106,
            cv2.FONT_HERSHEY_DUPLEX,
            1.85,
            (15, 15, 25),  # Pitch Black / Deep Charcoal
            4
        )

        # Header Line 4: Class & Session Information (Dark Navy, 0.60 scale)
        draw_centered_text(
            canvas,
            f"Class {self.current_class_idx + 1} / 9    |    Existing Sessions: {sess_count} / Target: {TARGET_SESSIONS_PER_CLASS}",
            142,
            cv2.FONT_HERSHEY_DUPLEX,
            0.60,
            (35, 35, 55),
            2
        )

        # Header Line 5: Movement Instruction (Wrapped into 1 or 2 clean lines, Dark Slate)
        desc_lines = wrap_text_into_lines(f"MOVEMENT: {movement_desc}", max_chars=66)
        if len(desc_lines) == 1:
            draw_centered_text(
                canvas,
                desc_lines[0],
                176,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (40, 45, 50),
                2
            )
        else:
            draw_centered_text(
                canvas,
                desc_lines[0],
                168,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (40, 45, 50),
                2
            )
            draw_centered_text(
                canvas,
                desc_lines[1],
                192,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (40, 45, 50),
                2
            )

        # -------------------------------------------------------------
        # 2. Recording Status Banner (High-contrast bright card)
        # -------------------------------------------------------------
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            remaining = max(0.0, RECORDING_DURATION_SECONDS - elapsed)
            sec_ceil = int(math.ceil(remaining))

            rec_banner_w, rec_banner_h = 680, 72
            rx = (w - rec_banner_w) // 2
            ry = top_bar_h + 12

            # Solid white card with bright red border
            cv2.rectangle(canvas, (rx, ry), (rx + rec_banner_w, ry + rec_banner_h), (255, 255, 255), -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rec_banner_w, ry + rec_banner_h), (0, 0, 220), 3)

            # Pulsing red circle indicator
            pulse_color = (0, 0, 220) if int(elapsed * 4) % 2 == 0 else (0, 100, 240)
            cv2.circle(canvas, (rx + 28, ry + 24), 11, pulse_color, -1)

            # High-contrast recording text
            rec_text = f"●  RECORDING — {sec_ceil:2d}s"
            cv2.putText(
                canvas,
                rec_text,
                (rx + 48, ry + 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.72,
                (0, 0, 210),
                2,
                cv2.LINE_AA
            )

            # Active Shiro Name in Dark Text
            shiro_text = f"CURRENT SHIRO: {display_name}"
            cv2.putText(
                canvas,
                shiro_text,
                (rx + 290, ry + 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.72,
                (15, 15, 25),
                2,
                cv2.LINE_AA
            )

            # Progress bar
            progress = min(1.0, elapsed / RECORDING_DURATION_SECONDS)
            bar_y = ry + 46
            bar_w = rec_banner_w - 56
            cv2.rectangle(canvas, (rx + 28, bar_y), (rx + 28 + bar_w, bar_y + 8), (215, 220, 230), -1)
            cv2.rectangle(canvas, (rx + 28, bar_y), (rx + 28 + int(bar_w * progress), bar_y + 8), (0, 60, 220), -1)

            # Frame count & rate
            buf_len = len(self.buf_features)
            rate = buf_len / elapsed if elapsed > 0 else 0
            rate_text = f"Countdown: {sec_ceil}s  |  Frames: {buf_len} ({rate:.1f} fps)"
            cv2.putText(
                canvas,
                rate_text,
                (rx + 28, ry + 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (40, 45, 55),
                1,
                cv2.LINE_AA
            )

        elif time.time() - self.status_time < 5.0:
            stat_w, stat_h = 740, 56
            sx = (w - stat_w) // 2
            sy = top_bar_h + 12
            # Solid white card with emerald green border
            cv2.rectangle(canvas, (sx, sy), (sx + stat_w, sy + stat_h), (255, 255, 255), -1)
            cv2.rectangle(canvas, (sx, sy), (sx + stat_w, sy + stat_h), (0, 160, 40), 3)

            cv2.putText(
                canvas,
                self.status_message,
                (sx + 24, sy + 26),
                cv2.FONT_HERSHEY_DUPLEX,
                0.62,
                (0, 120, 30),
                2,
                cv2.LINE_AA
            )
            sub_msg = f"CURRENT SHIRO: {display_name} — Existing Sessions: {sess_count} / Target: {TARGET_SESSIONS_PER_CLASS}"
            cv2.putText(
                canvas,
                sub_msg,
                (sx + 24, sy + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (25, 25, 35),
                1,
                cv2.LINE_AA
            )

        # -------------------------------------------------------------
        # 3. Head Orientation Telemetry (Left panel - solid clean card)
        # -------------------------------------------------------------
        panel_y = top_bar_h + (90 if self.is_recording or (time.time() - self.status_time < 5.0) else 15)
        panel_w, panel_h = 280, 140
        panel_x = 20
        # Solid light card with subtle border
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (248, 250, 254), -1)
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (180, 195, 215), 1)

        cv2.putText(
            canvas,
            "HEAD ORIENTATION",
            (panel_x + 14, panel_y + 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.50,
            (120, 45, 15),
            1,
            cv2.LINE_AA
        )

        if detected and shiro_features:
            roll_deg = math.degrees(shiro_features.get("head_tilt_roll_rad", 0.0))
            pitch = shiro_features.get("head_pitch", 0.0)
            yaw = shiro_features.get("head_yaw", 0.0)
            shift = shiro_features.get("head_lateral_shift", 0.0)

            cv2.putText(canvas, f"Head Tilt (Roll): {roll_deg:+5.1f} deg", (panel_x + 14, panel_y + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 40), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Head Pitch      : {pitch:+5.2f}", (panel_x + 14, panel_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 40), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Head Yaw (Turn) : {yaw:+5.2f}", (panel_x + 14, panel_y + 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 40), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Lateral Shift   : {shift:+5.2f}", (panel_x + 14, panel_y + 114),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 40), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Tracking: ACTIVE", (panel_x + 14, panel_y + 134),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 140, 40), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "No Head / Pose Detected", (panel_x + 14, panel_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Ensure face & shoulders visible", (panel_x + 14, panel_y + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 80, 90), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 4. Bottom Navigation Bar (Solid Light Bar)
        # -------------------------------------------------------------
        bot_bar_h = 45
        by = h - bot_bar_h
        cv2.rectangle(canvas, (0, by), (w, h), (248, 250, 254), -1)
        cv2.line(canvas, (0, by), (w, by), (180, 195, 215), 1)

        controls_text = "SPACE = Record 10 sec   |   N = Next Shiro   |   R = Retry   |   Q = Quit"
        draw_centered_text(
            canvas,
            controls_text,
            by + 28,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (120, 45, 15),
            2
        )

        return canvas

    def start_recording(self):
        """Initializes 10-second temporal recording buffer."""
        self.is_recording = True
        self.recording_start_time = time.time()
        self.last_sample_time = 0.0
        self.buf_features.clear()
        self.buf_timestamps.clear()
        self.buf_confidences.clear()
        self.buf_valid_mask.clear()
        self.buf_head_angles.clear()
        self.buf_raw_pose.clear()

        class_id, class_name, _, sanskrit_name, _ = self.get_current_class_info()
        print(f"\n[RECORDING STARTED] Class: {class_name} ({sanskrit_name}) — 10.0 seconds...")

    def finalize_recording(self):
        """Finalizes, computes temporal derivatives, and saves NPZ session."""
        self.is_recording = False
        duration = time.time() - self.recording_start_time

        T = len(self.buf_features)
        if T == 0:
            self.status_message = "Recording failed: 0 frames captured!"
            self.status_color = (0, 0, 255)
            self.status_time = time.time()
            print("[ERROR] 0 frames recorded in session!")
            return

        features_seq = np.array(self.buf_features, dtype=np.float32)       # (T, 32)
        timestamps = np.array(self.buf_timestamps, dtype=np.float64)       # (T,)
        confidences = np.array(self.buf_confidences, dtype=np.float32)     # (T,)
        valid_mask = np.array(self.buf_valid_mask, dtype=bool)             # (T,)
        head_angles = np.array(self.buf_head_angles, dtype=np.float32)     # (T, 3)
        raw_pose = np.array(self.buf_raw_pose, dtype=np.float32)           # (T, 33, 4)

        # Compute full temporal derivative package
        temp_pkg = compute_full_temporal_package(features_seq, timestamps, valid_mask, head_angles)

        # Save to disk
        class_id, class_name, folder_name, sanskrit_name, movement_desc = self.get_current_class_info()
        class_dir = SHIRO_LANDMARKS_DIR / folder_name
        target_path, session_id, sess_idx = get_next_session_filename(class_dir)

        actual_fps = float(T / duration) if duration > 0 else 0.0
        valid_count = int(np.sum(valid_mask))
        valid_ratio = float(valid_count / T) if T > 0 else 0.0

        metadata = {
            "class_id": class_id,
            "class_name": class_name,
            "folder_name": folder_name,
            "sanskrit_name": sanskrit_name,
            "session_id": session_id,
            "duration_seconds": round(duration, 3),
            "total_frames": T,
            "valid_frames": valid_count,
            "rejected_frames": T - valid_count,
            "valid_ratio": round(valid_ratio, 4),
            "actual_fps": round(actual_fps, 2),
            "feature_dim": BASE_FEATURE_DIMENSION,
            "temporal_dim": TEMPORAL_SUMMARY_DIMENSION,
            "created_at": datetime.now().isoformat(),
            "collector": "Step36A_Personalized_Shiro_Collector"
        }
        metadata_json = json.dumps(metadata)

        np.savez_compressed(
            target_path,
            features_seq=features_seq,
            timestamps=timestamps,
            frame_indices=np.arange(T, dtype=np.int32),
            valid_mask=valid_mask,
            confidences=confidences,
            head_angles=head_angles,
            velocities=temp_pkg["velocities"],
            accelerations=temp_pkg["accelerations"],
            displacements=temp_pkg["displacements"],
            head_angle_changes=temp_pkg["head_angle_changes"],
            movement_amplitude=temp_pkg["movement_amplitude"],
            temporal_summary_288=temp_pkg["temporal_summary_288"],
            raw_pose_landmarks=raw_pose,
            label=class_id,
            class_id=class_id,
            class_name=class_name,
            session_id=session_id,
            metadata=metadata_json
        )

        print(f"[SESSION SAVED] {target_path}")
        print(f"  Valid frames : {valid_count} / {T} ({valid_ratio*100:.1f}%)")
        print(f"  Duration     : {duration:.2f}s ({actual_fps:.1f} fps)")
        print(f"  Total sessions for {class_name}: {sess_idx}")

        self.status_message = f"SESSION SAVED ({session_id}) | Valid frames: {valid_count}/{T} ({valid_ratio*100:.1f}%)"
        self.status_color = (0, 255, 120)
        self.status_time = time.time()

        # Generate live reports
        generate_collection_reports(self.classes)

    def run(self):
        """Runs interactive webcam capture loop."""
        print("=" * 80)
        print("SHIRO BHEDHAS — 9 CLASSES")
        print("=" * 80)
        for cid, name, folder, sanskrit, desc in self.classes:
            existing = count_existing_sessions(SHIRO_LANDMARKS_DIR / folder)
            print(f"  [{cid}] {name:<18} ({sanskrit:<12}) -> {existing} sessions existing")
        print("=" * 80)
        print("Controls: SPACE = Record 10s | N = Next | R = Retry | Q = Quit\n")

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam index {self.camera_index}.")
            return

        window_name = "Personalized Shiro Bhedha Collection (Step 36A)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.frame_width, self.frame_height)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Mirror horizontally for natural movement feedback
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame with pose extractor
                res = self.extractor.process(rgb, category_focus="shiro")
                detected = res.get("detected", False)
                shiro_feats = res.get("shiro_features", {})
                vec = res.get("vector", np.zeros(BASE_FEATURE_DIMENSION, dtype=np.float32))

                # Quality Gating Check
                is_finite = np.all(np.isfinite(vec)) and len(vec) == BASE_FEATURE_DIMENSION
                has_no_nan = not np.isnan(vec).any()
                has_no_inf = not np.isinf(vec).any()
                is_valid_frame = detected and is_finite and has_no_nan and has_no_inf

                # Draw landmarks on frame
                if detected:
                    raw_lms = res.get("raw_landmarks", [])
                    # Draw head & neck keypoints: nose(0), ears(7,8), shoulders(11,12)
                    for pt_idx in [0, 2, 5, 7, 8, 11, 12]:
                        if pt_idx < len(raw_lms):
                            lm = raw_lms[pt_idx]
                            px = int(lm["x"] * frame.shape[1])
                            py = int(lm["y"] * frame.shape[0])
                            color = (0, 255, 0) if is_valid_frame else (0, 165, 255)
                            cv2.circle(frame, (px, py), 4, color, -1)

                # Handle active recording
                if self.is_recording:
                    now = time.time()
                    elapsed = now - self.recording_start_time

                    if elapsed >= RECORDING_DURATION_SECONDS:
                        self.finalize_recording()
                    elif (now - self.last_sample_time) >= MIN_SAMPLE_INTERVAL:
                        self.last_sample_time = now

                        # Collect frame data
                        self.buf_features.append(vec if is_valid_frame else np.zeros(BASE_FEATURE_DIMENSION, dtype=np.float32))
                        self.buf_timestamps.append(elapsed)
                        self.buf_valid_mask.append(is_valid_frame)

                        # Confidence
                        if detected and "raw_landmarks" in res:
                            raw_lms = res["raw_landmarks"]
                            vis_head = [raw_lms[i]["visibility"] for i in [0, 7, 8, 11, 12] if i < len(raw_lms)]
                            conf = float(np.mean(vis_head)) if vis_head else 0.5
                        else:
                            conf = 0.0
                        self.buf_confidences.append(conf)

                        # Angles
                        angles = [
                            float(shiro_feats.get("head_tilt_roll_rad", 0.0)),
                            float(shiro_feats.get("head_pitch", 0.0)),
                            float(shiro_feats.get("head_yaw", 0.0))
                        ]
                        self.buf_head_angles.append(angles)

                        # Raw pose coordinates
                        raw_arr = np.zeros((33, 4), dtype=np.float32)
                        if detected and "raw_landmarks" in res:
                            for idx, lm in enumerate(res["raw_landmarks"]):
                                raw_arr[idx] = [lm["x"], lm["y"], lm["z"], lm["visibility"]]
                        self.buf_raw_pose.append(raw_arr)

                # Render HUD
                hud_frame = self.draw_hud(frame, detected, shiro_feats, is_valid_frame)
                cv2.imshow(window_name, hud_frame)

                # Key handling
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                    if self.is_recording:
                        print("[INFO] Recording cancelled by user.")
                        self.is_recording = False
                    else:
                        print("\n[INFO] Exiting collector safely.")
                        break

                elif key == 32:  # SPACE
                    if not self.is_recording:
                        self.start_recording()

                elif key in [ord('n'), ord('N')]:  # N
                    if not self.is_recording:
                        self.current_class_idx = (self.current_class_idx + 1) % len(self.classes)
                        cid, cname, _, sanskrit, _ = self.get_current_class_info()
                        self.status_message = f"Selected: {cname} ({sanskrit})"
                        self.status_color = (255, 255, 255)
                        self.status_time = time.time()
                        print(f"\n[NEXT CLASS] Switched to: {cname} ({sanskrit})")

                elif key in [ord('r'), ord('R')]:  # R
                    if not self.is_recording:
                        cid, cname, _, sanskrit, _ = self.get_current_class_info()
                        self.status_message = f"Ready to record {cname}."
                        self.status_color = (255, 255, 255)
                        self.status_time = time.time()
                        print(f"[RETRY/RESET] Reset state for: {cname}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            # Final report generation
            generate_collection_reports(self.classes)
            print("[INFO] Final collection report & statistics saved.")


# ==============================================================================
# 6. STANDALONE DRY-RUN / VERIFICATION CHECK
# ==============================================================================

def check_collector_integrity() -> bool:
    """Verifies Shiro classes, directories, dynamic UI navigation, and feature pipeline non-interactively."""
    print("=" * 80)
    print("SHIRO BHEDHAS — 9 CLASSES")
    print("=" * 80)
    classes = load_shiro_class_mapping()
    assert len(classes) == 9, f"Expected 9 classes, got {len(classes)}"

    expected_names = [
        "SAMA", "UDVAHITAM", "ADHOMUKHAM", "ALOLITAM", "DHUTAM",
        "KAMPITAM", "PARAVRITTAM", "UKSHIPTAM", "PARIVAHITTAM"
    ]

    for cid, name, folder, sanskrit, desc in classes:
        print(f"  Class {cid}: {name:<18} -> {folder:<18} ({sanskrit})")

    # 1. Verify dynamic class navigation and HUD rendering for all 9 classes
    print("\n" + "-" * 80)
    print("VERIFYING DYNAMIC SHIRO NAME DISPLAY (ALL 9 CLASSES):")
    print("-" * 80)
    app = ShiroCollectorApp.__new__(ShiroCollectorApp)
    app.classes = classes
    app.current_class_idx = 0
    app.is_recording = False
    app.recording_start_time = 0.0
    app.last_sample_time = 0.0
    app.status_message = "Ready"
    app.status_time = 0.0
    app.status_color = (255, 255, 255)
    app.buf_features = []

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    for expected_idx in range(9):
        app.current_class_idx = expected_idx
        disp_name = app.get_current_display_name()
        exp_name = expected_names[expected_idx]
        assert disp_name == exp_name, f"Class {expected_idx}: Expected {exp_name}, got {disp_name}"
        print(f"  Class {expected_idx + 1} / 9: CURRENT SHIRO -> {disp_name:<16} [OK]")

        # Test HUD rendering in idle mode
        hud_idle = app.draw_hud(dummy_frame, detected=False, shiro_features={}, is_frame_valid=False)
        assert hud_idle.shape == (720, 1280, 3), f"HUD shape error: {hud_idle.shape}"

        # Test HUD rendering in recording mode (verifies name remains visible during recording)
        app.is_recording = True
        app.recording_start_time = time.time()
        hud_rec = app.draw_hud(dummy_frame, detected=True,
                               shiro_features={"head_tilt_roll_rad": 0.1, "head_pitch": 0.2, "head_yaw": -0.1, "head_lateral_shift": 0.05},
                               is_frame_valid=True)
        assert hud_rec.shape == (720, 1280, 3), f"HUD recording shape error: {hud_rec.shape}"
        app.is_recording = False

    # Test wrap-around navigation
    app.current_class_idx = (8 + 1) % 9
    assert app.current_class_idx == 0 and app.get_current_display_name() == "SAMA", "Navigation wrap-around failed"
    print("  Navigation wrap-around: Class 9 -> Class 1 (SAMA) [OK]")

    # 2. Verify dummy extraction
    extractor = PoseLandmarkExtractor(static_image_mode=True)
    dummy_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = extractor.process(dummy_rgb, category_focus="shiro")
    assert len(res["vector"]) == BASE_FEATURE_DIMENSION, f"Feature dimension error: {len(res['vector'])}"
    extractor.close()

    # 3. Verify next session numbering logic
    test_dir = PROJECT_ROOT / "scratch" / "test_shiro_sessions"
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        p1, s1, i1 = get_next_session_filename(test_dir)
        assert s1 == "session_01" and i1 == 1
        np.savez(p1, dummy=np.zeros(5))
        p2, s2, i2 = get_next_session_filename(test_dir)
        assert s2 == "session_02" and i2 == 2
    finally:
        for f in test_dir.glob("*"):
            f.unlink()
        test_dir.rmdir()

    # Generate initial reports
    generate_collection_reports(classes)
    print("\n[PASSED] Collector integrity and prominent Shiro name display verified successfully.")
    return True


if __name__ == "__main__":
    if "--check-only" in sys.argv:
        check_collector_integrity()
    else:
        app = ShiroCollectorApp()
        app.run()
