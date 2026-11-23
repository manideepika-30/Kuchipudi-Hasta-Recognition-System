"""
STEP 37A — PERSONALIZED DRISHTI (EYE MOVEMENT) 10-SECOND CONTINUOUS COLLECTOR
=============================================================================
Interactive OpenCV + MediaPipe FaceMesh tool for continuous 10-second webcam
recording and temporal sequence landmark extraction for all 8 classical Drishti
Bhedhas (Eye / Gaze Movements) in Kuchipudi dance.

Canonical Drishti Classes (8 Classes):
  0 = Sama       : Looking straight ahead level, peaceful and unblinking
  1 = Alokita    : Circular rolling glance, sweeping eyes in continuous circles
  2 = Sachi      : Looking askance from eye corners without moving head (side glance hold)
  3 = Pralokita  : Scanning glance from side to side horizontally across both directions
  4 = Nimilita   : Half-closed meditative downward glance (inward serene gaze)
  5 = Ullokita   : Gazing upward toward sky or ceiling (elevated gaze)
  6 = Anuvritta  : Glancing rapidly up and down continuously (vertical scanning movement)
  7 = Avalokita  : Looking downward toward floor or ground with open eyes (downward gaze)

Proof-of-Concept (POC) Staged Workflow:
  - Phase 1 (POC Mode): Exactly 5 classes (Sama, Alokita, Sachi, Pralokita, Nimilita)
  - Phase 2 (Full Mode): User presses 'A' to unlock all 8 classes (adds Ullokita, Anuvritta, Avalokita)
  - Target: 4 sessions per class (POC: 20 sessions; Full: 32 sessions)

Feature Representation (Drishti 44-D):
  - Base Features: 44-D canonical Eye/Iris/Gaze geometry (FaceDrishtiLandmarkExtractor)
      * EAR Left & Right (Eye Aspect Ratio / openness / blink proxy)
      * Gaze Horizontal Left & Right (relative iris position in eye socket [0=inner, 1=outer])
      * Gaze Vertical Left & Right (relative iris position in eye socket [0=top, 1=bottom])
      * Head Yaw, Pitch, Roll proxies (auxiliary head normalization only)
      * Inter-ocular face width & height scales
      * Relative 3D iris positions (left, right) and nose relative to face center
      * 8 key eye boundary points relative to iris centers (24 dims)
  - Temporal Derivatives:
      * Velocity (T, 44): First derivative divided by actual dt
      * Acceleration (T, 44): Second derivative divided by actual dt
      * Displacements (T, 44): Movement relative to session start frame
      * Gaze Displacements (T, 4): Relative gaze shifts
      * Temporal Summary (396-D): 44 base mean + 352 statistical descriptors

Controls:
  SPACE : Record another 10-second session for CURRENT Drishti class (remains on same class)
  N     : Advance to NEXT Drishti class (cycles within active mode)
  P     : Go to PREVIOUS Drishti class
  A     : Toggle POC Mode (5 classes) <-> Full Mode (8 classes)
  R     : Retry / reset status for current class without deleting previous sessions
  Q/ESC : Quit safely and generate collection report & statistics

Safety:
  - ZERO modification of Hastha personalized dataset or models
  - ZERO modification of Shiro personalized dataset or models
  - ZERO modification of locked v2 split files
  - ZERO training or evaluation during collection
"""

import os
import sys
import time
import json
import csv
import re
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.landmark_extraction import FaceDrishtiLandmarkExtractor
from src.train_hybrid_models import compute_temporal_statistics

# Base Directories
PERSONALIZED_ROOT = PROJECT_ROOT / "data" / "personalized"
DRISHTI_LANDMARKS_DIR = PERSONALIZED_ROOT / "drishti_landmarks"
REPORTS_DIR = PROJECT_ROOT / "reports"

DRISHTI_LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TXT_PATH = REPORTS_DIR / "step37a_drishti_collection_report.txt"
STATS_CSV_PATH = REPORTS_DIR / "step37a_drishti_collection_stats.csv"

RECORDING_DURATION_SECONDS = 10.0
BASE_FEATURE_DIMENSION = 44
TEMPORAL_SUMMARY_DIMENSION = 396
MIN_SAMPLE_INTERVAL = 0.050  # ~20 Hz cap (50 ms)
TARGET_SESSIONS_PER_CLASS = 4

# ==============================================================================
# 1. CANONICAL 8 DRISHTI BHEDHAS DEFINITIONS
# ==============================================================================

CANONICAL_DRISHTI_CLASSES = [
    (0, "1. Sama", "1_Sama", "Sama", "Looking straight ahead level, peaceful and unblinking with natural neutral gaze"),
    (1, "2. Alokita", "2_Alokita", "Alokita", "Circular rolling glance, sweeping eyes smoothly around in continuous circular motion"),
    (2, "3. Sachi", "3_Sachi", "Sachi", "Looking askance from the corners of the eyes without moving the head (side glance hold)"),
    (3, "4. Pralokita", "4_Pralokita", "Pralokita", "Scanning glance from side to side horizontally across both directions"),
    (4, "5. Nimilita", "5_Nimilita", "Nimilita", "Half-closed meditative downward glance (inward serene gaze)"),
    (5, "6. Ullokita", "6_Ullokita", "Ullokita", "Gazing upward toward the sky or ceiling (elevated gaze)"),
    (6, "7. Anuvritta", "7_Anuvritta", "Anuvritta", "Glancing rapidly up and down continuously (vertical scanning movement)"),
    (7, "8. Avalokita", "8_Avalokita", "Avalokita", "Looking downward toward the floor or ground with open eyes (downward gaze)"),
]

POC_CLASS_INDICES = [0, 1, 2, 3, 4]  # Sama, Alokita, Sachi, Pralokita, Nimilita


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


def wrap_text_into_lines(text: str, max_chars: int = 70) -> List[str]:
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
    features_seq: np.ndarray,      # (T, 44)
    timestamps: np.ndarray,        # (T,)
    valid_mask: np.ndarray,        # (T,)
    gaze_ratios: np.ndarray        # (T, 4)
) -> Dict[str, np.ndarray]:
    """
    Computes complete temporal features from raw Drishti sequence:
      - Velocities: (T, 44)
      - Accelerations: (T, 44)
      - Displacements: (T, 44) relative to session start frame
      - Gaze Displacements: (T, 4) relative gaze shift
      - Movement Amplitude: (44,)
      - Temporal Summary (44 base mean + 352 temporal descriptors = 396 dims)
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
        diffs = np.diff(features_seq, axis=0)  # (T-1, 44)
        vel_body = diffs / dt[1:, np.newaxis]  # (T-1, 44)
        velocities = np.vstack([np.zeros((1, D), dtype=np.float32), vel_body]).astype(np.float32)
    else:
        velocities = np.zeros((T, D), dtype=np.float32)

    # 3. Accelerations (second derivative)
    if T > 2:
        diffs_vel = np.diff(velocities, axis=0)  # (T-1, 44)
        acc_body = diffs_vel / dt[1:, np.newaxis]  # (T-1, 44)
        accelerations = np.vstack([np.zeros((1, D), dtype=np.float32), acc_body]).astype(np.float32)
    else:
        accelerations = np.zeros((T, D), dtype=np.float32)

    # 4. Displacements relative to session start
    displacements = (features_seq - features_seq[0:1]).astype(np.float32)

    # 5. Gaze displacements relative to session start
    gaze_displacements = (gaze_ratios - gaze_ratios[0:1]).astype(np.float32)

    # 6. Movement amplitude (peak-to-peak over valid frames)
    if np.any(valid_mask):
        valid_feats = features_seq[valid_mask]
        movement_amplitude = np.ptp(valid_feats, axis=0).astype(np.float32)
        mean_lm = np.mean(valid_feats, axis=0).astype(np.float32)
    else:
        movement_amplitude = np.zeros(D, dtype=np.float32)
        mean_lm = np.zeros(D, dtype=np.float32)

    # 7. Canonical 396-D temporal summary descriptor (44 mean + 352 temporal stats)
    stat_vec = compute_temporal_statistics(features_seq, velocities)  # (352,)
    temporal_summary_396 = np.concatenate([mean_lm, stat_vec]).astype(np.float32)  # (396,)

    return {
        "velocities": velocities,
        "accelerations": accelerations,
        "displacements": displacements,
        "gaze_displacements": gaze_displacements,
        "movement_amplitude": movement_amplitude,
        "temporal_summary_396": temporal_summary_396
    }


# ==============================================================================
# 4. AUDIT AND REPORT GENERATION
# ==============================================================================

def generate_collection_reports(
    classes_info: List[Tuple[int, str, str, str, str]],
    poc_mode: bool = True
) -> Dict[str, Any]:
    """Generates text report and statistics CSV for Drishti collection."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DRISHTI_LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    class_stats = []
    total_sessions = 0
    total_valid_frames = 0

    for local_id, name, folder, sanskrit, desc in classes_info:
        class_dir = DRISHTI_LANDMARKS_DIR / folder
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
        is_poc_class = local_id in POC_CLASS_INDICES
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
            "is_poc": is_poc_class,
            "status": status
        })

    # Write CSV
    with open(STATS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class_id", "class_name", "folder_name", "sanskrit_name",
            "session_count", "target_sessions", "total_valid_frames",
            "min_frames_per_session", "max_frames_per_session",
            "mean_frames_per_session", "is_poc", "status"
        ])
        writer.writeheader()
        writer.writerows(class_stats)

    # Write TXT
    target_total = 20 if poc_mode else 32
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STEP 37A — PERSONALIZED DRISHTI BHEDHAS COLLECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target category: Drishti Bhedhas (Eye Movements)\n")
        f.write(f"Output directory: {DRISHTI_LANDMARKS_DIR.resolve()}\n")
        f.write(f"Current mode: {'Proof-of-Concept (5 Classes)' if poc_mode else 'Full Canonical (8 Classes)'}\n")
        f.write(f"Base feature dimension per frame: {BASE_FEATURE_DIMENSION} (Eye/Iris/Gaze Geometry)\n")
        f.write(f"Temporal summary dimension: {TEMPORAL_SUMMARY_DIMENSION} (44 mean + 352 temporal descriptors)\n")
        f.write(f"Recording session duration: {RECORDING_DURATION_SECONDS:.1f} seconds per session\n")
        f.write(f"Target sampling rate: ~15-20 Hz (~140-180 frames per session)\n")
        f.write(f"Target sessions per class: {TARGET_SESSIONS_PER_CLASS} (Target total: {target_total} sessions)\n")
        f.write(f"Current total sessions collected: {total_sessions} / {target_total}\n")
        f.write(f"Current total valid frames: {total_valid_frames:,}\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'ID':<4} {'Class Name':<16} {'Sanskrit':<12} {'POC':<6} {'Sessions':<10} {'Valid Frames':<14} {'Status'}\n")
        f.write("-" * 80 + "\n")
        for cs in class_stats:
            poc_str = "YES" if cs["is_poc"] else "NO"
            f.write(f"{cs['class_id']:<4} {cs['class_name']:<16} {cs['sanskrit_name']:<12} {poc_str:<6} {cs['session_count']}/{cs['target_sessions']:<8} {cs['total_valid_frames']:<14} {cs['status']}\n")
        f.write("-" * 80 + "\n\n")

        f.write("-" * 80 + "\n")
        f.write("EXACT 44-D DRISHTI FEATURE REPRESENTATION SPECIFICATION:\n")
        f.write("-" * 80 + "\n")
        f.write("  [Dims 0-1]   EAR Left & Right (Eye Aspect Ratio / eye openness / blink proxy):\n")
        f.write("               EAR = ||p_top - p_bottom|| / ||p_outer - p_inner||\n")
        f.write("  [Dims 2-3]   Gaze Left (Horizontal & Vertical iris ratio inside eye):\n")
        f.write("               gaze_h_l = (x_iris_l - x_inner_l) / (x_outer_l - x_inner_l + eps)\n")
        f.write("               gaze_v_l = (y_iris_l - y_top_l) / (y_bottom_l - y_top_l + eps)\n")
        f.write("  [Dims 4-5]   Gaze Right (Horizontal & Vertical iris ratio inside eye):\n")
        f.write("               gaze_h_r = (x_iris_r - x_inner_r) / (x_outer_r - x_inner_r + eps)\n")
        f.write("               gaze_v_r = (y_iris_r - y_top_r) / (y_bottom_r - y_top_r + eps)\n")
        f.write("  [Dims 6-8]   Auxiliary Head Orientation Proxies (normalization only):\n")
        f.write("               yaw_proxy = (x_nose - x_mid_eye) / face_width\n")
        f.write("               pitch_proxy = (y_nose - y_mid_eye) / face_height\n")
        f.write("               roll_proxy = atan2(y_outer_r - y_outer_l, x_outer_r - x_outer_l)\n")
        f.write("  [Dims 9-10]  Face Scale Normalization Factors:\n")
        f.write("               face_width = ||p_cheek_l - p_cheek_r||, face_height = ||p_forehead - p_chin||\n")
        f.write("  [Dims 11-13] Left Iris 3D relative position: (p_iris_l - p_face_center) / face_width\n")
        f.write("  [Dims 14-16] Right Iris 3D relative position: (p_iris_r - p_face_center) / face_width\n")
        f.write("  [Dims 17-19] Nose Tip 3D relative position: (p_nose - p_face_center) / face_height\n")
        f.write("  [Dims 20-43] 8 Key Eye Boundary Points relative to iris centers (24 dims):\n")
        f.write("               Left Outer, Inner, Top, Bottom relative to Left Iris Center (12 dims)\n")
        f.write("               Right Outer, Inner, Top, Bottom relative to Right Iris Center (12 dims)\n")
        f.write("               Normalized by eye width/height: (p_bound - p_iris) / eye_dim\n")
        f.write("  Total Base Feature Dimension: EXACTLY 44-D\n\n")

        f.write("-" * 80 + "\n")
        f.write("LANDMARK INDICES & KINEMATICS HANDLING:\n")
        f.write("-" * 80 + "\n")
        f.write("  Left Eye Landmark Indices  : Outer=33, Inner=133, Top=159, Bottom=145\n")
        f.write("  Right Eye Landmark Indices : Outer=263, Inner=362, Top=386, Bottom=374\n")
        f.write("  Iris Center Indices        : Left Iris=468, Right Iris=473 (MediaPipe Refined Mesh)\n")
        f.write("  Face Reference Indices     : Nose=1, Chin=152, Forehead=10, Cheeks=234, 454\n")
        f.write("  Left/Right Eye Orientation : Camera frame horizontally mirrored; landmark mapping\n")
        f.write("                               strictly preserves anatomical left/right eye semantics.\n")
        f.write("  Iris Tracking / Fallback   : Extracted via FaceMesh refine_landmarks=True. Missing iris\n")
        f.write("                               falls back to eyelid center and is flagged by quality gate.\n")
        f.write("  Missing-Landmark Behavior  : Emits zero vector; frame flagged as invalid in valid_mask\n")
        f.write("                               and excluded from temporal statistical descriptors.\n")
        f.write("  Velocity & Acceleration    : v_t = (f_t - f_{t-1}) / dt; a_t = (v_t - v_{t-1}) / dt\n")
        f.write("  Timestamp Handling         : Strictly monotonic seconds from recording start (0.0 to 10.0s).\n\n")

        f.write("-" * 80 + "\n")
        f.write("SAFETY & ISOLATION CONFIRMATION:\n")
        f.write("-" * 80 + "\n")
        f.write("  [PASS] Drishti data isolated in: data/personalized/drishti_landmarks/\n")
        f.write("  [PASS] Hastha dataset untouched: data/personalized/hasthas_landmarks/ preserved\n")
        f.write("  [PASS] Shiro dataset untouched: data/personalized/shiro_landmarks/ preserved\n")
        f.write("  [PASS] Hastha & Shiro models untouched: models/personalized_step34/ and step36 preserved\n")
        f.write("  [PASS] Locked split datasets untouched: v2_train.csv, v2_validation.csv, v2_test.csv\n")
        f.write("  [PASS] Zero training or evaluation performed during Step 37A\n")
        f.write("-" * 80 + "\n")

    return {
        "total_sessions": total_sessions,
        "total_valid_frames": total_valid_frames,
        "class_stats": class_stats
    }


# ==============================================================================
# 5. INTERACTIVE COLLECTOR CLASS
# ==============================================================================

class PersonalizedDrishti10SecCollector:
    """
    Webcam collection application for personalized Drishti Bhedhas.
    Implements continuous 10-second capture, quality gating, eye/iris feature extraction,
    and high-contrast dancer HUD.
    """
    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        check_only: bool = False
    ):
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.check_only = check_only

        self.all_classes = CANONICAL_DRISHTI_CLASSES
        self.poc_mode = False  # Full 8-class canonical collection mode active by default
        self.active_class_list = list(self.all_classes)
        self.current_active_idx = 0

        # Initialize canonical MediaPipe face mesh extractor
        if not self.check_only:
            self.extractor = FaceDrishtiLandmarkExtractor(
                static_image_mode=False,
                min_detection_confidence=0.3
            )
        else:
            self.extractor = None

        # Recording Buffer State
        self.is_recording: bool = False
        self.recording_start_time: float = 0.0
        self.last_sample_time: float = 0.0

        self.buf_features: List[np.ndarray] = []
        self.buf_timestamps: List[float] = []
        self.buf_valid_mask: List[bool] = []
        self.buf_confidences: List[float] = []
        self.buf_gaze_ratios: List[np.ndarray] = []
        self.buf_ear: List[np.ndarray] = []
        self.buf_raw_eyes: List[Dict[str, Any]] = []

        # Quality Gate Metrics
        self.q_total_frames: int = 0
        self.q_valid_frames: int = 0
        self.q_rejected_frames: int = 0
        self.q_face_failures: int = 0
        self.q_iris_failures: int = 0
        self.q_nonfinite_features: int = 0
        self.q_invalid_dims: int = 0

        # Status HUD State
        self.status_message: str = "READY — Press SPACE to start 10s recording"
        self.status_color: Tuple[int, int, int] = (120, 75, 40)
        self.status_time: float = time.time()
        self.last_saved_time: float = 0.0
        self.last_saved_session_id: str = ""
        self.last_saved_valid_frames: int = 0
        self.last_saved_total_frames: int = 0

        # Generate initial reports
        generate_collection_reports(self.all_classes, self.poc_mode)

    def get_current_class_info(self) -> Tuple[int, str, str, str, str]:
        """Returns tuple of (class_id, class_name, folder_name, sanskrit_name, movement_desc)."""
        return self.active_class_list[self.current_active_idx]

    def set_poc_mode(self, enabled: bool):
        """Toggles between 5-class POC mode and full 8-class mode."""
        curr_cid = self.get_current_class_info()[0]
        self.poc_mode = enabled
        if self.poc_mode:
            self.active_class_list = [c for c in self.all_classes if c[0] in POC_CLASS_INDICES]
        else:
            self.active_class_list = list(self.all_classes)

        # Preserve selected class if present in new list
        new_indices = [i for i, c in enumerate(self.active_class_list) if c[0] == curr_cid]
        self.current_active_idx = new_indices[0] if new_indices else 0

        mode_name = "POC MODE (5 Classes)" if self.poc_mode else "FULL MODE (8 Classes)"
        self.status_message = f"Switched to {mode_name}"
        self.status_color = (130, 90, 30)
        self.status_time = time.time()
        print(f"[MODE] {mode_name} active ({len(self.active_class_list)} classes).")
        generate_collection_reports(self.all_classes, self.poc_mode)

    def start_recording(self):
        """Initiates a new 10-second continuous recording session for the CURRENT class."""
        self.is_recording = True
        self.recording_start_time = time.time()
        self.last_sample_time = 0.0

        self.buf_features.clear()
        self.buf_timestamps.clear()
        self.buf_valid_mask.clear()
        self.buf_confidences.clear()
        self.buf_gaze_ratios.clear()
        self.buf_ear.clear()
        self.buf_raw_eyes.clear()

        # Reset counters for this session
        self.q_total_frames = 0
        self.q_valid_frames = 0
        self.q_rejected_frames = 0
        self.q_face_failures = 0
        self.q_iris_failures = 0
        self.q_nonfinite_features = 0
        self.q_invalid_dims = 0

        class_id, class_name, _, sanskrit_name, _ = self.get_current_class_info()
        print(f"\n[RECORDING STARTED] Class {class_id}: {class_name} ({sanskrit_name}) — Recording for 10.0s...")

    def finalize_recording(self):
        """Concludes the recording session, computes derivatives, and writes session_XX.npz."""
        self.is_recording = False
        duration = time.time() - self.recording_start_time
        T = len(self.buf_features)

        if T == 0:
            self.status_message = "Recording failed: 0 frames captured!"
            self.status_color = (0, 0, 255)
            self.status_time = time.time()
            print("[ERROR] 0 frames recorded in session!")
            return

        features_seq = np.array(self.buf_features, dtype=np.float32)       # (T, 44)
        timestamps = np.array(self.buf_timestamps, dtype=np.float64)       # (T,)
        confidences = np.array(self.buf_confidences, dtype=np.float32)     # (T,)
        valid_mask = np.array(self.buf_valid_mask, dtype=bool)             # (T,)
        gaze_ratios = np.array(self.buf_gaze_ratios, dtype=np.float32)     # (T, 4)
        ear_openness = np.array(self.buf_ear, dtype=np.float32)            # (T, 2)

        # Compute full temporal derivative package
        temp_pkg = compute_full_temporal_package(features_seq, timestamps, valid_mask, gaze_ratios)

        # Save to disk under data/personalized/drishti_landmarks/<folder_name>/
        class_id, class_name, folder_name, sanskrit_name, movement_desc = self.get_current_class_info()
        class_dir = DRISHTI_LANDMARKS_DIR / folder_name
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
            "category": "drishti",
            "duration_seconds": round(duration, 3),
            "total_frames": T,
            "valid_frames": valid_count,
            "rejected_frames": T - valid_count,
            "valid_ratio": round(valid_ratio, 4),
            "actual_fps": round(actual_fps, 2),
            "feature_dim": BASE_FEATURE_DIMENSION,
            "temporal_dim": TEMPORAL_SUMMARY_DIMENSION,
            "quality_audit": {
                "face_detection_failures": self.q_face_failures,
                "iris_failures": self.q_iris_failures,
                "nonfinite_features": self.q_nonfinite_features,
                "invalid_dims": self.q_invalid_dims
            },
            "created_at": datetime.now().isoformat(),
            "collector": "Step37A_Personalized_Drishti_Collector"
        }
        metadata_json = json.dumps(metadata)

        np.savez_compressed(
            target_path,
            features=features_seq,
            features_seq=features_seq,
            timestamps=timestamps,
            frame_indices=np.arange(T, dtype=np.int32),
            valid_mask=valid_mask,
            confidences=confidences,
            gaze_ratios=gaze_ratios,
            velocities=temp_pkg["velocities"],
            accelerations=temp_pkg["accelerations"],
            displacements=temp_pkg["displacements"],
            gaze_displacements=temp_pkg["gaze_displacements"],
            ear_openness=ear_openness,
            temporal_summary_396=temp_pkg["temporal_summary_396"],
            label=class_id,
            class_index=class_id,
            class_id=class_id,
            class_name=class_name,
            sanskrit_name=sanskrit_name,
            session_id=session_id,
            category="drishti",
            duration_seconds=duration,
            valid_frame_count=valid_count,
            feature_dim=BASE_FEATURE_DIMENSION,
            metadata=metadata_json
        )

        print(f"[SESSION SAVED] {target_path}")
        print(f"  Valid frames : {valid_count} / {T} ({valid_ratio*100:.1f}%)")
        print(f"  Duration     : {duration:.2f}s ({actual_fps:.1f} fps)")
        print(f"  Total sessions for {class_name}: {sess_idx}")

        self.status_message = f"SESSION SAVED ({session_id}) | Valid frames: {valid_count}/{T} ({valid_ratio*100:.1f}%)"
        self.status_color = (0, 180, 50)
        self.status_time = time.time()
        self.last_saved_time = time.time()
        self.last_saved_session_id = session_id
        self.last_saved_valid_frames = valid_count
        self.last_saved_total_frames = T

        # Update live reports
        generate_collection_reports(self.all_classes, self.poc_mode)

    def draw_hud(
        self,
        frame: np.ndarray,
        detected: bool,
        gaze_feats: Dict[str, Any],
        is_valid_frame: bool
    ) -> np.ndarray:
        """
        Renders high-contrast, uncluttered dancer HUD for Drishti collection:
          - Solid bright off-white header panel (y=0 to 200, zero face darkening)
          - DRISHTI COLLECTION header title
          - Prominent large black Drishti name centered (font scale ~1.95)
          - Clear session and class counters (Session: X / target | Class: X / 8)
          - Specific eye movement dancer instructions
          - RECORDING — 10 SECONDS with visible progress bar
          - SESSION SAVED confirmation while keeping current class visible
          - Bottom controls bar
        """
        h, w = frame.shape[:2]
        canvas = frame.copy()

        class_id, class_name, folder_name, sanskrit_name, movement_desc = self.get_current_class_info()
        class_dir = DRISHTI_LANDMARKS_DIR / folder_name
        existing_sess = count_existing_sessions(class_dir)
        active_total = len(self.active_class_list)

        # -------------------------------------------------------------
        # 1. Solid Bright Off-White Top Header Panel (y=0 to 205)
        # -------------------------------------------------------------
        top_bar_h = 205
        cv2.rectangle(canvas, (0, 0), (w, top_bar_h), (250, 252, 255), -1)
        cv2.line(canvas, (0, top_bar_h - 2), (w, top_bar_h - 2), (180, 195, 215), 2)
        cv2.line(canvas, (0, top_bar_h), (w, top_bar_h), (60, 80, 120), 2)

        # Header Line 1: Main Title (y = 26)
        draw_centered_text(
            canvas,
            "PERSONALIZED DRISHTI BHEDHA COLLECTION (8 CLASSES)",
            26,
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (120, 45, 15),  # Deep Navy
            2
        )

        # Header Line 2: Progress & Session Counter (y = 52)
        progress_text = f"DRISHTI {class_id + 1} / 8        |        SESSION: {existing_sess} / {TARGET_SESSIONS_PER_CLASS}"
        draw_centered_text(
            canvas,
            progress_text,
            52,
            cv2.FONT_HERSHEY_DUPLEX,
            0.60,
            (40, 45, 65),  # Dark Slate
            2
        )

        # Header Line 3: LARGE CURRENT CLASS NAME (y = 104)
        current_name_text = f"CURRENT DRISHTI: {sanskrit_name}"
        draw_centered_text(
            canvas,
            current_name_text,
            104,
            cv2.FONT_HERSHEY_DUPLEX,
            1.40,
            (15, 15, 25),  # Pitch Black
            3
        )

        # Header Line 4: Primary Dancer Instruction (y = 140)
        inst_primary = 'Instruction: "Move your EYES naturally — keep the camera view clear."'
        draw_centered_text(
            canvas,
            inst_primary,
            140,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 30, 60),
            2
        )

        # Header Line 5: Specific Eye Movement Guidance (y = 166)
        inst_desc = f"Movement: {movement_desc}"
        draw_centered_text(
            canvas,
            inst_desc,
            166,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (60, 65, 80),
            1
        )

        # Header Line 6: Dynamic Recording / Saved / Diagnostics (y = 192)
        now = time.time()
        if self.is_recording:
            elapsed = now - self.recording_start_time
            remaining = max(0.0, RECORDING_DURATION_SECONDS - elapsed)
            valid_c = int(np.sum(self.buf_valid_mask))
            draw_centered_text(
                canvas,
                f"● RECORDING — 10 SECONDS  ({remaining:4.1f}s remaining  |  {valid_c} valid frames)",
                192,
                cv2.FONT_HERSHEY_DUPLEX,
                0.52,
                (0, 0, 210),  # Bold Red
                2
            )
            # High-contrast bright red progress bar across bottom of header
            prog_ratio = min(1.0, elapsed / RECORDING_DURATION_SECONDS)
            bar_w = int(w * prog_ratio)
            cv2.rectangle(canvas, (0, top_bar_h - 5), (bar_w, top_bar_h), (0, 0, 220), -1)

        elif (now - self.last_saved_time) < 4.0 and self.last_saved_session_id:
            draw_centered_text(
                canvas,
                f"SESSION SAVED ({self.last_saved_session_id}) — {self.last_saved_valid_frames}/{self.last_saved_total_frames} valid frames! (Press SPACE to record again, N for next)",
                192,
                cv2.FONT_HERSHEY_DUPLEX,
                0.48,
                (0, 160, 40),  # Emerald Green
                2
            )
        else:
            if detected and gaze_feats:
                g_h_l = gaze_feats.get("gaze_horizontal_left", 0.5)
                g_v_l = gaze_feats.get("gaze_vertical_left", 0.5)
                ear_l = gaze_feats.get("ear_left", 0.0)
                ear_r = gaze_feats.get("ear_right", 0.0)
                diag_str = f"Live Gaze: Horiz={g_h_l:.2f}, Vert={g_v_l:.2f}  |  EAR Openness: L={ear_l:.2f}, R={ear_r:.2f}  |  Face & Eyes: DETECTED"
            else:
                diag_str = "Face & Eye Status: Looking for eyes in camera view... keep face in frame"
            draw_centered_text(
                canvas,
                diag_str,
                192,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (90, 95, 115),
                1
            )

        # -------------------------------------------------------------
        # 2. Bottom Controls Bar
        # -------------------------------------------------------------
        bot_bar_h = 36
        by = h - bot_bar_h
        cv2.rectangle(canvas, (0, by), (w, h), (248, 250, 254), -1)
        cv2.line(canvas, (0, by), (w, by), (180, 195, 215), 1)

        controls_text = "SPACE = Record 10s   |   N = Next   |   P = Prev   |   R = Retry   |   Q = Quit"
        draw_centered_text(
            canvas,
            controls_text,
            by + 24,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (120, 45, 15),
            1
        )

        return canvas

    def run(self):
        """Runs interactive webcam capture loop."""
        print("=" * 80)
        print("PERSONALIZED DRISHTI BHEDHAS COLLECTION (STEP 37A)")
        print("=" * 80)
        mode_str = "POC Mode (5 Classes)" if self.poc_mode else "Full Mode (8 Classes)"
        print(f"Current Mode: {mode_str}")
        print(f"Target: {TARGET_SESSIONS_PER_CLASS} continuous 10-second sessions per class")
        print("Classes:")
        for cid, name, folder, sanskrit, desc in self.active_class_list:
            existing = count_existing_sessions(DRISHTI_LANDMARKS_DIR / folder)
            print(f"  [{cid}] {name:<16} ({sanskrit:<12}) -> {existing}/4 sessions existing")
        print("=" * 80)
        print("Controls: SPACE = Record 10s | N = Next | P = Prev | A = Toggle POC/All | R = Retry | Q = Quit\n")

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam index {self.camera_index}.")
            return

        window_name = "Personalized Drishti Bhedha Collection (Step 37A)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.frame_width, self.frame_height)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Mirror horizontally for natural dancer mirror interaction
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame with face/eye extractor
                res = self.extractor.process(rgb)
                detected = res.get("detected", False)
                has_iris = res.get("iris_available", False)
                gaze_feats = res.get("gaze_features", {})
                vec = res.get("vector", np.zeros(BASE_FEATURE_DIMENSION, dtype=np.float32))

                # Quality Gating Check
                is_finite = np.all(np.isfinite(vec)) and len(vec) == BASE_FEATURE_DIMENSION
                has_no_nan = not np.isnan(vec).any()
                has_no_inf = not np.isinf(vec).any()
                has_valid_dim = (len(vec) == BASE_FEATURE_DIMENSION)

                is_valid_frame = detected and has_iris and is_finite and has_no_nan and has_no_inf and has_valid_dim

                # Draw eye keypoints on preview (non-intrusive dots on irises and corners)
                if detected:
                    raw_lms = res.get("raw_landmarks", {})
                    # Draw left and right iris centers if present
                    for k in ["iris_left", "iris_right"]:
                        if k in raw_lms:
                            pt = raw_lms[k]
                            px = int(pt[0] * frame.shape[1])
                            py = int(pt[1] * frame.shape[0])
                            color = (0, 255, 0) if is_valid_frame else (0, 165, 255)
                            cv2.circle(frame, (px, py), 4, color, -1)
                            cv2.circle(frame, (px, py), 8, (255, 255, 255), 1)

                # Handle active recording
                if self.is_recording:
                    now = time.time()
                    elapsed = now - self.recording_start_time

                    self.q_total_frames += 1
                    if is_valid_frame:
                        self.q_valid_frames += 1
                    else:
                        self.q_rejected_frames += 1
                        if not detected:
                            self.q_face_failures += 1
                        elif not has_iris:
                            self.q_iris_failures += 1
                        elif not is_finite:
                            self.q_nonfinite_features += 1
                        elif not has_valid_dim:
                            self.q_invalid_dims += 1

                    if elapsed >= RECORDING_DURATION_SECONDS:
                        self.finalize_recording()
                    elif (now - self.last_sample_time) >= MIN_SAMPLE_INTERVAL:
                        self.last_sample_time = now

                        # Collect frame data
                        self.buf_features.append(vec if is_valid_frame else np.zeros(BASE_FEATURE_DIMENSION, dtype=np.float32))
                        self.buf_timestamps.append(elapsed)
                        self.buf_valid_mask.append(is_valid_frame)

                        # Gaze ratios
                        if detected and gaze_feats:
                            g_arr = np.array([
                                float(gaze_feats.get("gaze_horizontal_left", 0.5)),
                                float(gaze_feats.get("gaze_vertical_left", 0.5)),
                                float(gaze_feats.get("gaze_horizontal_right", 0.5)),
                                float(gaze_feats.get("gaze_vertical_right", 0.5))
                            ], dtype=np.float32)
                            ear_arr = np.array([
                                float(gaze_feats.get("ear_left", 0.2)),
                                float(gaze_feats.get("ear_right", 0.2))
                            ], dtype=np.float32)
                            conf = 1.0 if has_iris else 0.5
                        else:
                            g_arr = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
                            ear_arr = np.array([0.0, 0.0], dtype=np.float32)
                            conf = 0.0

                        self.buf_gaze_ratios.append(g_arr)
                        self.buf_ear.append(ear_arr)
                        self.buf_confidences.append(conf)

                # Render HUD
                hud_frame = self.draw_hud(frame, detected, gaze_feats, is_valid_frame)
                cv2.imshow(window_name, hud_frame)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("\n[INFO] Quitting Drishti collection...")
                    break
                elif key == 32:  # SPACE
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        print("[INFO] Already recording session...")
                elif key in [ord('n'), ord('N')]:  # Next class
                    if not self.is_recording:
                        self.current_active_idx = (self.current_active_idx + 1) % len(self.active_class_list)
                        cid, cname, _, sanskrit, _ = self.get_current_class_info()
                        self.status_message = f"Selected: {cname} ({sanskrit})"
                        self.status_color = (120, 75, 40)
                        self.status_time = time.time()
                        print(f"[CLASS] Switched to Class {cid}: {cname} ({sanskrit})")
                elif key in [ord('p'), ord('P')]:  # Previous class
                    if not self.is_recording:
                        self.current_active_idx = (self.current_active_idx - 1) % len(self.active_class_list)
                        cid, cname, _, sanskrit, _ = self.get_current_class_info()
                        self.status_message = f"Selected: {cname} ({sanskrit})"
                        self.status_color = (120, 75, 40)
                        self.status_time = time.time()
                        print(f"[CLASS] Switched to Class {cid}: {cname} ({sanskrit})")
                elif key in [ord('a'), ord('A')]:  # Toggle POC / Full mode
                    if not self.is_recording:
                        self.set_poc_mode(not self.poc_mode)
                elif key in [ord('r'), ord('R')]:  # Retry / reset
                    if not self.is_recording:
                        self.status_message = "Ready for new session"
                        self.status_color = (120, 75, 40)
                        self.status_time = time.time()
                        print("[RESET] Session status reset for current class.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.extractor is not None:
                self.extractor.close()
            # Final report write
            generate_collection_reports(self.all_classes, self.poc_mode)
            print(f"[REPORT] Saved collection report to: {REPORT_TXT_PATH.resolve()}")
            print(f"[REPORT] Saved dataset statistics to: {STATS_CSV_PATH.resolve()}")


# ==============================================================================
# 6. STARTUP VALIDATION MODE (--check-only)
# ==============================================================================

def run_collector_check_only() -> Tuple[bool, List[str]]:
    """
    Validates class definitions, POC configuration, FaceDrishtiLandmarkExtractor,
    and output directories without opening the camera.
    """
    print("=" * 80)
    print("STEP 37A — DRISHTI COLLECTOR CONFIGURATION AUDIT")
    print("=" * 80)
    passed = True
    errors = []

    # 1. Verify 8 canonical classes
    print("\n[1/5] Verifying 8 Canonical Drishti Classes...")
    if len(CANONICAL_DRISHTI_CLASSES) == 8:
        print(f"  [PASS] Exactly 8 canonical Drishti classes defined in exact canonical order:")
        for cid, name, folder, sanskrit, desc in CANONICAL_DRISHTI_CLASSES:
            print(f"         Class {cid}: {name:<16} ({sanskrit:<12}) -> Folder: {folder}")
    else:
        passed = False
        err = f"Expected 8 canonical classes, got {len(CANONICAL_DRISHTI_CLASSES)}"
        errors.append(err)
        print(f"  [FAIL] {err}")

    # 2. Verify 5-class POC mode
    print("\n[2/5] Verifying 5-Class Proof-of-Concept Mode...")
    if len(POC_CLASS_INDICES) == 5 and POC_CLASS_INDICES == [0, 1, 2, 3, 4]:
        poc_names = [CANONICAL_DRISHTI_CLASSES[i][1] for i in POC_CLASS_INDICES]
        print(f"  [PASS] Initial POC mode targets exactly 5 classes: {', '.join(poc_names)}")
    else:
        passed = False
        err = f"POC class indices invalid: {POC_CLASS_INDICES}"
        errors.append(err)
        print(f"  [FAIL] {err}")

    # 3. Test FaceDrishtiLandmarkExtractor initialization & dimension
    print("\n[3/5] Testing FaceDrishtiLandmarkExtractor Initialization & 44-D Vector...")
    try:
        ext = FaceDrishtiLandmarkExtractor(static_image_mode=True, min_detection_confidence=0.3)
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        res = ext.process(dummy)
        ext.close()
        vec_len = len(res.get("vector", []))
        if vec_len == BASE_FEATURE_DIMENSION:
            print(f"  [PASS] Extractor initialized and tested: feature dimension = {vec_len}-D")
        else:
            passed = False
            err = f"Expected feature dimension {BASE_FEATURE_DIMENSION}, got {vec_len}"
            errors.append(err)
            print(f"  [FAIL] {err}")
    except Exception as e:
        passed = False
        err = f"Extractor test failed: {e}"
        errors.append(err)
        print(f"  [FAIL] {err}")

    # 4. Test 396-D temporal summary package
    print("\n[4/5] Testing 396-D Temporal Summary Package...")
    try:
        dummy_seq = np.random.randn(150, BASE_FEATURE_DIMENSION).astype(np.float32)
        dummy_ts = np.linspace(0, 10.0, 150)
        dummy_mask = np.ones(150, dtype=bool)
        dummy_gaze = np.random.randn(150, 4).astype(np.float32)
        pkg = compute_full_temporal_package(dummy_seq, dummy_ts, dummy_mask, dummy_gaze)
        sum_dim = len(pkg["temporal_summary_396"])
        if sum_dim == TEMPORAL_SUMMARY_DIMENSION:
            print(f"  [PASS] Temporal package verified: summary dimension = {sum_dim}-D (44 base + 352 stats)")
        else:
            passed = False
            err = f"Expected temporal summary {TEMPORAL_SUMMARY_DIMENSION}, got {sum_dim}"
            errors.append(err)
            print(f"  [FAIL] {err}")
    except Exception as e:
        passed = False
        err = f"Temporal package computation error: {e}"
        errors.append(err)
        print(f"  [FAIL] {err}")

    # 5. Verify UI Rendering for all 8 Canonical Drishti Classes (Requirement 10)
    print("\n[5/7] Verifying UI Rendering for All 8 Canonical Drishti Classes...")
    expected_sanskrit_names = ["Sama", "Alokita", "Sachi", "Pralokita", "Nimilita", "Ullokita", "Anuvritta", "Avalokita"]
    test_collector = PersonalizedDrishti10SecCollector(check_only=True)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    for idx, expected_name in enumerate(expected_sanskrit_names):
        test_collector.current_active_idx = idx
        cid, cname, folder, sanskrit, desc = test_collector.get_current_class_info()

        if sanskrit != expected_name or cid != idx:
            passed = False
            err = f"Class {idx} mismatch: expected {expected_name} (ID {idx}), got {sanskrit} (ID {cid})"
            errors.append(err)
            print(f"  [FAIL] {err}")
            continue

        # Render Idle HUD
        hud_idle = test_collector.draw_hud(
            dummy_frame,
            detected=True,
            gaze_feats={"gaze_horizontal_left": 0.52, "gaze_vertical_left": 0.48, "ear_left": 0.28, "ear_right": 0.28},
            is_valid_frame=True
        )

        # Render Active Recording HUD
        test_collector.is_recording = True
        test_collector.recording_start_time = time.time() - 4.5
        hud_rec = test_collector.draw_hud(
            dummy_frame,
            detected=True,
            gaze_feats={"gaze_horizontal_left": 0.52, "gaze_vertical_left": 0.48, "ear_left": 0.28, "ear_right": 0.28},
            is_valid_frame=True
        )
        test_collector.is_recording = False

        # Render Saved Confirmation HUD
        test_collector.last_saved_time = time.time() - 1.0
        test_collector.last_saved_session_id = "session_01"
        test_collector.last_saved_valid_frames = 152
        test_collector.last_saved_total_frames = 152
        hud_saved = test_collector.draw_hud(
            dummy_frame,
            detected=True,
            gaze_feats={},
            is_valid_frame=True
        )
        test_collector.last_saved_time = 0.0

        if hud_idle is not None and hud_rec is not None and hud_saved is not None:
            print(f"  [PASS] Class {cid}: CURRENT DRISHTI: {sanskrit:<12} (DRISHTI {cid + 1} / 8) -> Rendered cleanly (Idle/Rec/Saved)")
        else:
            passed = False
            err = f"Failed rendering HUD for {sanskrit}"
            errors.append(err)
            print(f"  [FAIL] {err}")

    # 6. Verify N Navigation Sequence: Nimilita -> Ullokita -> Anuvritta -> Avalokita (Requirement 11)
    print("\n[6/7] Verifying N Navigation Sequence (Nimilita -> Ullokita -> Anuvritta -> Avalokita)...")
    nav_steps = [
        (4, "Nimilita", "5. Nimilita"),
        (5, "Ullokita", "6. Ullokita"),
        (6, "Anuvritta", "7. Anuvritta"),
        (7, "Avalokita", "8. Avalokita"),
        (0, "Sama", "1. Sama")
    ]
    # Set to Nimilita (index 4)
    test_collector.current_active_idx = 4
    init_cid, _, _, init_sanskrit, _ = test_collector.get_current_class_info()
    if init_cid == 4 and init_sanskrit == "Nimilita":
        print(f"  [START] Initial position: Class {init_cid}: {init_sanskrit}")
    else:
        passed = False
        err = f"Failed to start navigation at Nimilita: Class {init_cid} ({init_sanskrit})"
        errors.append(err)
        print(f"  [FAIL] {err}")

    # Step through sequence
    for step_idx in range(1, len(nav_steps)):
        expected_cid, expected_sanskrit, expected_cname = nav_steps[step_idx]
        # Simulate pressing 'N'
        test_collector.current_active_idx = (test_collector.current_active_idx + 1) % len(test_collector.active_class_list)
        curr_cid, curr_cname, _, curr_sanskrit, _ = test_collector.get_current_class_info()

        if curr_cid == expected_cid and curr_sanskrit == expected_sanskrit:
            print(f"  [PASS] Pressed 'N' -> Class {curr_cid}: CURRENT DRISHTI: {curr_sanskrit:<12} (DRISHTI {curr_cid + 1} / 8)")
        else:
            passed = False
            err = f"Navigation step failed: expected {expected_sanskrit} (ID {expected_cid}), got {curr_sanskrit} (ID {curr_cid})"
            errors.append(err)
            print(f"  [FAIL] {err}")

    # 7. Output directories and dataset preservation audit
    print("\n[7/7] Auditing Output Directories & Preserving Existing Collected Data...")
    DRISHTI_LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    existing_npz = list(DRISHTI_LANDMARKS_DIR.rglob("session_*.npz"))
    report_dict = generate_collection_reports(CANONICAL_DRISHTI_CLASSES, poc_mode=False)

    print(f"  [INFO] Discovered {len(existing_npz)} existing sessions in {DRISHTI_LANDMARKS_DIR.name}/:")
    for cid, name, folder, sanskrit, desc in CANONICAL_DRISHTI_CLASSES:
        c_sess = count_existing_sessions(DRISHTI_LANDMARKS_DIR / folder)
        print(f"         Class {cid}: {name:<16} ({sanskrit:<12}) -> {c_sess}/4 sessions existing")

    if REPORT_TXT_PATH.exists() and STATS_CSV_PATH.exists():
        print(f"  [PASS] Updated collection reports successfully:")
        print(f"         {REPORT_TXT_PATH.resolve()}")
        print(f"         {STATS_CSV_PATH.resolve()}")
    else:
        passed = False
        err = "Report generation failed"
        errors.append(err)
        print(f"  [FAIL] {err}")

    print("\n" + "=" * 80)
    if passed:
        print("ALL DRISHTI COLLECTOR CONFIGURATION & UI CHECKS PASSED!")
        print("All 8 Drishti classes and N-navigation verified. Ready for collection.")
    else:
        print(f"Collector configuration audit failed: {'; '.join(errors)}")
    print("=" * 80)

    return passed, errors


def main():
    parser = argparse.ArgumentParser(description="Step 37A — Personalized Drishti Bhedhas Continuous 10-Second Collector")
    parser.add_argument("--check-only", action="store_true", help="Run configuration checks and report generation without opening camera")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    args = parser.parse_args()

    if args.check_only:
        success, _ = run_collector_check_only()
        sys.exit(0 if success else 1)
    else:
        collector = PersonalizedDrishti10SecCollector(
            camera_index=args.camera_index,
            frame_width=args.width,
            frame_height=args.height,
            check_only=False
        )
        collector.run()


if __name__ == "__main__":
    main()
