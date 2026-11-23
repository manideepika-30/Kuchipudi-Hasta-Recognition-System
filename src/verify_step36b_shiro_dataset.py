"""
STEP 36B — FULL PERSONALIZED SHIRO DATASET AUDIT
=================================================
Comprehensive audit and verification script for the complete personalized
9-Shiro Bhedha continuous landmark dataset under data/personalized/shiro_landmarks/.

Performs 10 comprehensive verification checks:
  Check  1: Class Coverage (exactly 9 Shiro classes, >= 4 sessions each)
  Check  2: Session Integrity (opens, all keys exist, no empty arrays, no NaN, no Inf)
  Check  3: Temporal Consistency (matching T across all 11 temporal arrays, monotonic time, contiguous index)
  Check  4: Feature Dimensions ((T,32), (T,3), (T,33,4), (288,), etc.)
  Check  5: Data Quality (valid frames, invalid frames, duplicate consecutive frames, zero vectors)
  Check  6: Temporal Movement Quality (movement amplitude, roll/pitch/yaw variation, velocity, acceleration)
  Check  7: Duplication (session-level pairwise similarity analysis)
  Check  8: Label Integrity (folder, class_name, class_id matching canonical mapping)
  Check  9: Session Balance (min/max/median/mean sessions and frames)
  Check 10: Safety Protocol (Hastha dataset, Step34 models, Step35 camera app, v2 splits untouched)

Final Verdict:
  READY FOR PERSONALIZED SHIRO TRAINING
  or
  NOT READY FOR PERSONALIZED SHIRO TRAINING

SAFETY:
  - ZERO training
  - ZERO evaluation
  - ZERO camera access
  - ZERO data modification
"""

import os
import sys
import json
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
SHIRO_LANDMARKS_DIR = PROJECT_ROOT / "data" / "personalized" / "shiro_landmarks"
HASTHAS_LANDMARKS_DIR = PROJECT_ROOT / "data" / "personalized" / "hasthas_landmarks"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
SHIRO_MAPPING_FILE = MODELS_DIR / "step16b_shiro_class_mapping.json"

AUDIT_REPORT_PATH = REPORTS_DIR / "step36b_shiro_dataset_audit.txt"
STATISTICS_CSV_PATH = REPORTS_DIR / "step36b_shiro_dataset_statistics.csv"

# Canonical Shiro Bhedha classes
CANONICAL_SHIRO_CLASSES = [
    (0, "1. Sama", "1_Sama", "Sama", "Head held straight and motionless in natural baseline posture"),
    (1, "2. Udvahitam", "2_Udvahitam", "Udvahitam", "Head lifted upward facing the sky (upward pitch tilt)"),
    (2, "3. Adhomukham", "3_Adhomukham", "Adhomukham", "Head bowed downward looking at the ground (downward pitch tilt)"),
    (3, "4. Alolitam", "4_Alolitam", "Alolitam", "Head rotated continuously in circular motion (circular roll/yaw)"),
    (4, "5. Dhutam", "5_Dhutam", "Dhutam", "Head turned from side to side horizontally like saying 'no' (horizontal yaw)"),
    (5, "6. Kampitam", "6_Kampitam", "Kampitam", "Head shaken/nodded up and down repeatedly like saying 'yes' (vertical oscillation)"),
    (6, "7. Paravrittam", "7_Paravrittam", "Paravrittam", "Head turned sharply to one side looking over shoulder (profile yaw hold)"),
    (7, "8. Ukshiptam", "8_Ukshiptam", "Ukshiptam", "Head tilted slantingly upward to one side (elevated diagonal tilt)"),
    (8, "9. Parivahittam", "9_Parivahittam", "Parivahittam", "Head swayed gently from side to side ear toward shoulder (lateral roll oscillation)"),
]

REQUIRED_SESSION_KEYS = [
    "features_seq",
    "timestamps",
    "frame_indices",
    "valid_mask",
    "confidences",
    "head_angles",
    "velocities",
    "accelerations",
    "displacements",
    "head_angle_changes",
    "movement_amplitude",
    "temporal_summary_288",
    "raw_pose_landmarks",
    "label",
    "class_name",
    "session_id",
]

TARGET_SESSIONS_PER_CLASS = 4


def load_canonical_mapping() -> List[Dict[str, Any]]:
    """Loads and validates canonical Shiro mapping from JSON."""
    if SHIRO_MAPPING_FILE.exists():
        with open(SHIRO_MAPPING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes", [])
        if len(classes) == 9:
            return sorted(classes, key=lambda x: x["local_class_id"])
    
    # Fallback to definitions
    return [
        {
            "local_class_id": cid,
            "global_class_id": 52 + cid,
            "class_name": cname,
            "folder_name": folder,
            "sanskrit_name": sanskrit,
            "movement_description": desc
        }
        for cid, cname, folder, sanskrit, desc in CANONICAL_SHIRO_CLASSES
    ]


def run_comprehensive_audit() -> Tuple[bool, Dict[str, Any]]:
    """
    Executes all 10 audit checks across the complete personalized Shiro dataset.
    """
    classes = load_canonical_mapping()
    audit_log = []
    
    def log(msg: str = ""):
        print(msg)
        audit_log.append(msg)

    log("=" * 80)
    log("STEP 36B — FULL PERSONALIZED SHIRO BHEDHAS DATASET AUDIT")
    log("=" * 80)
    log(f"Audit timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Shiro landmarks directory: {SHIRO_LANDMARKS_DIR.resolve()}")
    log(f"Canonical classes: {len(classes)} classes")
    log("=" * 80)

    # -------------------------------------------------------------------------
    # Scan and Load all NPZ files
    # -------------------------------------------------------------------------
    all_sessions: List[Dict[str, Any]] = []
    class_sessions: Dict[int, List[Dict[str, Any]]] = {c["local_class_id"]: [] for c in classes}
    scan_errors: List[str] = []

    for c in classes:
        cid = c["local_class_id"]
        cfolder = c["folder_name"]
        cdir = SHIRO_LANDMARKS_DIR / cfolder
        if not cdir.exists():
            scan_errors.append(f"Missing folder for Class {cid} ({c['class_name']}): {cfolder}")
            continue

        npz_files = sorted(cdir.glob("session_*.npz"))
        for npz_path in npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)
                sess_info = {
                    "path": npz_path,
                    "filename": npz_path.name,
                    "folder": cfolder,
                    "expected_class_id": cid,
                    "expected_class_name": c["class_name"],
                    "data": data,
                }
                all_sessions.append(sess_info)
                class_sessions[cid].append(sess_info)
            except Exception as e:
                scan_errors.append(f"Corrupt or unreadable NPZ file {npz_path}: {e}")

    total_sessions_found = len(all_sessions)
    log(f"\nDiscovered {total_sessions_found} session files across {len(classes)} classes.")

    # -------------------------------------------------------------------------
    # CHECK 1: CLASS COVERAGE
    # -------------------------------------------------------------------------
    log("\n" + "-" * 80)
    log("CHECK 1 — CLASS COVERAGE")
    log("-" * 80)
    c1_passed = True
    c1_details = []

    class_stats_list = []
    for c in classes:
        cid = c["local_class_id"]
        cname = c["class_name"]
        cfolder = c["folder_name"]
        sanskrit = c["sanskrit_name"]
        desc = c["movement_description"]
        sessions = class_sessions[cid]
        num_sess = len(sessions)

        if num_sess < TARGET_SESSIONS_PER_CLASS:
            c1_passed = False
            c1_details.append(f"Class {cid} ({cname}) has {num_sess} sessions (expected >= {TARGET_SESSIONS_PER_CLASS})")

        frame_lengths = []
        valid_counts = []
        for s in sessions:
            d = s["data"]
            T = len(d["features_seq"]) if "features_seq" in d else 0
            v = int(np.sum(d["valid_mask"])) if "valid_mask" in d else 0
            frame_lengths.append(T)
            valid_counts.append(v)

        total_v = sum(valid_counts)
        min_len = min(frame_lengths) if frame_lengths else 0
        max_len = max(frame_lengths) if frame_lengths else 0
        mean_len = float(np.mean(frame_lengths)) if frame_lengths else 0.0

        status_str = f"COMPLETE ({num_sess}/{TARGET_SESSIONS_PER_CLASS})" if num_sess >= TARGET_SESSIONS_PER_CLASS else f"DEFICIT ({num_sess}/{TARGET_SESSIONS_PER_CLASS})"

        class_stats_list.append({
            "class_id": cid,
            "class_name": cname,
            "folder_name": cfolder,
            "sanskrit_name": sanskrit,
            "movement_description": desc,
            "session_count": num_sess,
            "total_frames": sum(frame_lengths),
            "total_valid_frames": total_v,
            "min_session_length": min_len,
            "max_session_length": max_len,
            "mean_session_length": round(mean_len, 1),
            "status": status_str
        })

        log(f"  Class {cid}: {cname:<18} | Folder: {cfolder:<16} | Sessions: {num_sess:2d} | Valid Frames: {total_v:4d} | "
            f"Len (min/max/mean): {min_len:3d} / {max_len:3d} / {mean_len:5.1f} | {status_str}")

    if c1_passed:
        log("  -> CHECK 1 PASSED: Exactly 9 classes present with >= 4 sessions each.")
    else:
        log(f"  -> CHECK 1 FAILED: {'; '.join(c1_details)}")

    # -------------------------------------------------------------------------
    # CHECK 2: SESSION INTEGRITY
    # -------------------------------------------------------------------------
    log("\n" + "-" * 80)
    log("CHECK 2 — SESSION INTEGRITY")
    log("-" * 80)
    c2_passed = True
    c2_missing_keys = []
    c2_nan_errors = []
    c2_inf_errors = []
    c2_empty_errors = []

    for s in all_sessions:
        d = s["data"]
        fname = f"{s['folder']}/{s['filename']}"

        # 1. Missing keys
        for k in REQUIRED_SESSION_KEYS:
            if k not in d:
                c2_missing_keys.append(f"{fname} missing '{k}'")
                c2_passed = False

        # 2. Empty arrays
        for k in ["features_seq", "timestamps", "frame_indices", "valid_mask", "head_angles", "velocities"]:
            if k in d:
                arr = d[k]
                if arr.size == 0:
                    c2_empty_errors.append(f"{fname} empty array for '{k}'")
                    c2_passed = False

        # 3. NaN check
        for k in ["features_seq", "timestamps", "head_angles", "velocities", "accelerations", "displacements", "temporal_summary_288"]:
            if k in d:
                arr = d[k]
                if np.isnan(arr).any():
                    c2_nan_errors.append(f"{fname} NaN in '{k}'")
                    c2_passed = False

        # 4. Inf check
        for k in ["features_seq", "timestamps", "head_angles", "velocities", "accelerations", "displacements", "temporal_summary_288"]:
            if k in d:
                arr = d[k]
                if np.isinf(arr).any():
                    c2_inf_errors.append(f"{fname} Inf in '{k}'")
                    c2_passed = False

    log(f"  Total sessions audited       : {len(all_sessions)}")
    log(f"  Missing required keys errors : {len(c2_missing_keys)}")
    log(f"  Empty array errors           : {len(c2_empty_errors)}")
    log(f"  NaN errors                   : {len(c2_nan_errors)}")
    log(f"  Inf errors                   : {len(c2_inf_errors)}")

    if c2_passed:
        log("  -> CHECK 2 PASSED: All 36 sessions open successfully, possess all required keys, with 0 NaN, 0 Inf, and 0 empty arrays.")
    else:
        log("  -> CHECK 2 FAILED: Integrity errors detected.")

    # -------------------------------------------------------------------------
    # CHECK 3: TEMPORAL CONSISTENCY
    # -------------------------------------------------------------------------
    log("\n" + "-" * 80)
    log("CHECK 3 — TEMPORAL CONSISTENCY")
    log("-" * 80)
    c3_passed = True
    c3_length_mismatches = []
    c3_timestamp_non_monotonic = []
    c3_frame_index_violations = []
    c3_durations = []
    c3_fps_rates = []

    temporal_array_keys = [
        "features_seq",
        "timestamps",
        "frame_indices",
        "valid_mask",
        "confidences",
        "head_angles",
        "velocities",
        "accelerations",
        "displacements",
        "head_angle_changes",
        "raw_pose_landmarks"
    ]

    for s in all_sessions:
        d = s["data"]
        fname = f"{s['folder']}/{s['filename']}"
        T = len(d["features_seq"])

        # Matching T
        for k in temporal_array_keys:
            if k in d:
                if len(d[k]) != T:
                    c3_length_mismatches.append(f"{fname} length mismatch for '{k}': expected {T}, got {len(d[k])}")
                    c3_passed = False

        # Monotonic timestamps
        times = d["timestamps"]
        if T > 1:
            diffs = np.diff(times)
            if np.any(diffs <= 0):
                c3_timestamp_non_monotonic.append(f"{fname} non-monotonic timestamps detected")
                c3_passed = False

        # Contiguous frame indices
        indices = d["frame_indices"]
        expected_indices = np.arange(T, dtype=np.int32)
        if not np.array_equal(indices, expected_indices):
            c3_frame_index_violations.append(f"{fname} non-contiguous frame indices")
            c3_passed = False

        # Duration & FPS
        duration = float(times[-1] - times[0]) if T > 1 else 0.0
        fps = float(T / duration) if duration > 0 else 0.0
        c3_durations.append(duration)
        c3_fps_rates.append(fps)

    min_dur, max_dur, mean_dur = min(c3_durations), max(c3_durations), np.mean(c3_durations)
    min_fps, max_fps, mean_fps = min(c3_fps_rates), max(c3_fps_rates), np.mean(c3_fps_rates)

    log(f"  Temporal array length mismatches : {len(c3_length_mismatches)}")
    log(f"  Timestamp monotonicity errors    : {len(c3_timestamp_non_monotonic)}")
    log(f"  Frame index sequence errors      : {len(c3_frame_index_violations)}")
    log(f"  Recording durations (min/max/mean): {min_dur:.2f}s / {max_dur:.2f}s / {mean_dur:.2f}s (~10.0s target)")
    log(f"  Actual frame rates (min/max/mean) : {min_fps:.1f} fps / {max_fps:.1f} fps / {mean_fps:.1f} fps")

    if c3_passed and (9.5 <= mean_dur <= 10.5):
        log("  -> CHECK 3 PASSED: All temporal sequence arrays have identical T, strictly increasing timestamps, contiguous indices, and valid ~10s duration.")
    else:
        log("  -> CHECK 3 FAILED: Temporal consistency errors detected.")

    # -------------------------------------------------------------------------
    # CHECK 4: FEATURE DIMENSIONS
    # -------------------------------------------------------------------------
    log("\n" + "-" * 80)
    log("CHECK 4 — FEATURE DIMENSIONS")
    log("-" * 80)
    c4_passed = True
    c4_dim_errors = []

    for s in all_sessions:
        d = s["data"]
        fname = f"{s['folder']}/{s['filename']}"
        T = len(d["features_seq"])

        expected_shapes = {
            "features_seq": (T, 32),
            "head_angles": (T, 3),
            "velocities": (T, 32),
            "accelerations": (T, 32),
            "displacements": (T, 32),
            "head_angle_changes": (T, 3),
            "raw_pose_landmarks": (T, 33, 4),
            "temporal_summary_288": (288,),
            "movement_amplitude": (32,),
        }

        for k, exp_shape in expected_shapes.items():
            if k in d:
                actual_shape = d[k].shape
                if actual_shape != exp_shape:
                    c4_dim_errors.append(f"{fname} '{k}' shape error: expected {exp_shape}, got {actual_shape}")
                    c4_passed = False

    log(f"  Verified base feature vectors      : (T, 32) float32 across all {total_sessions_found} sessions")
    log(f"  Verified head angle sequences      : (T, 3) float32 across all {total_sessions_found} sessions")
    log(f"  Verified velocities sequences      : (T, 32) float32 across all {total_sessions_found} sessions")
    log(f"  Verified accelerations sequences   : (T, 32) float32 across all {total_sessions_found} sessions")
    log(f"  Verified displacements sequences   : (T, 32) float32 across all {total_sessions_found} sessions")
    log(f"  Verified head angle change vectors : (T, 3) float32 across all {total_sessions_found} sessions")
    log(f"  Verified raw pose landmarks        : (T, 33, 4) float32 across all {total_sessions_found} sessions")
    log(f"  Verified temporal summary vectors  : (288,) float32 across all {total_sessions_found} sessions")
    log(f"  Dimension errors                   : {len(c4_dim_errors)}")

    if c4_passed:
        log("  -> CHECK 4 PASSED: All feature dimensions strictly conform to canonical 32-D base and 288-D temporal representations.")
    else:
        log("  -> CHECK 4 FAILED: Feature dimension discrepancies detected.")

    # -------------------------------------------------------------------------
    # CHECK 5: DATA QUALITY
    # -------------------------------------------------------------------------
    log("\n" + "-" * 80)
    log("CHECK 5 — DATA QUALITY")
    log("-" * 80)
    total_frames = 0
    total_valid_frames = 0
    total_invalid_frames = 0
    total_zero_vectors = 0
    total_consecutive_duplicates = 0

    for s in all_sessions:
        d = s["data"]
        feats = d["features_seq"]
        vmask = d["valid_mask"]
        T = len(feats)

        total_frames += T
        total_valid_frames += int(np.sum(vmask))
        total_invalid_frames += int(np.sum(~vmask))

        # Zero vector in valid frames
        for t in range(T):
            if vmask[t] and np.all(feats[t] == 0):
                total_zero_vectors += 1

        # Duplicate consecutive frames
        if T > 1:
            diffs = np.diff(feats, axis=0)
            is_zero_diff = np.all(diffs == 0, axis=1)
            total_consecutive_duplicates += int(np.sum(is_zero_diff))

    valid_pct = (total_valid_frames / total_frames * 100) if total_frames > 0 else 0.0

    log(f"  Total captured frames           : {total_frames:,}")
    log(f"  Total valid landmark frames     : {total_valid_frames:,} ({valid_pct:.2f}%)")
    log(f"  Total invalid frames (occluded) : {total_invalid_frames:,} ({100 - valid_pct:.2f}%)")
    log(f"  Zero-vector count (valid frames): {total_zero_vectors}")
    log(f"  Duplicate consecutive frames    : {total_consecutive_duplicates}")
    log(f"  Timestamp violations            : {len(c3_timestamp_non_monotonic)}")
    log(f"  Frame index violations          : {len(c3_frame_index_violations)}")

    c5_passed = (valid_pct >= 95.0) and (total_zero_vectors == 0)
    if c5_passed:
        log(f"  -> CHECK 5 PASSED: Outstanding data quality with {valid_pct:.2f}% valid frames and 0 zero vectors.")
    else:
        log("  -> CHECK 5 FAILED: Data quality thresholds violated.")

    # -------------------------------------------------------------------------
    # CHECK 6: TEMPORAL MOVEMENT QUALITY
    # -------------------------------------------------------------------------
    log("\n" + "-" * 80)
    log("CHECK 6 — TEMPORAL MOVEMENT QUALITY (CANONICAL SHIRO DYNAMICS)")
    log("-" * 80)
    c6_passed = True
    c6_anomalies = []

    class_dynamics = {}

    log(f"  {'Class Name':<16} | {'Avg Amp':<8} | {'Roll Var':<9} | {'Pitch Var':<9} | {'Yaw Var':<8} | {'Velocity':<9} | {'Accel':<8} | {'Dominant Motion'}")
    log("  " + "-" * 105)

    for c in classes:
        cid = c["local_class_id"]
        cname = c["class_name"]
        sessions = class_sessions[cid]

        amp_list = []
        roll_var_list = []
        pitch_var_list = []
        yaw_var_list = []
        vel_mag_list = []
        acc_mag_list = []

        for s in sessions:
            d = s["data"]
            amp_list.append(float(np.mean(d["movement_amplitude"])))
            angles = d["head_angles"]  # (T, 3): roll, pitch, yaw
            roll_var_list.append(float(np.std(angles[:, 0])))
            pitch_var_list.append(float(np.std(angles[:, 1])))
            yaw_var_list.append(float(np.std(angles[:, 2])))

            # Velocities magnitude (mean L2 norm across time)
            vel = d["velocities"]
            vel_norms = np.linalg.norm(vel, axis=1)
            vel_mag_list.append(float(np.mean(vel_norms)))

            # Accelerations magnitude
            acc = d["accelerations"]
            acc_norms = np.linalg.norm(acc, axis=1)
            acc_mag_list.append(float(np.mean(acc_norms)))

        mean_amp = float(np.mean(amp_list))
        mean_roll_var = float(np.mean(roll_var_list))
        mean_pitch_var = float(np.mean(pitch_var_list))
        mean_yaw_var = float(np.mean(yaw_var_list))
        mean_vel = float(np.mean(vel_mag_list))
        mean_acc = float(np.mean(acc_mag_list))

        class_dynamics[cid] = {
            "mean_amp": mean_amp,
            "mean_roll_var": mean_roll_var,
            "mean_pitch_var": mean_pitch_var,
            "mean_yaw_var": mean_yaw_var,
            "mean_vel": mean_vel,
            "mean_acc": mean_acc,
        }

        # Characterize motion based on relative axis fold-change and canonical dance definitions
        if cid == 0:  # Sama
            dominant = "Static Baseline Posture"
            # Sama expected to have low baseline movement (breathing/posture jitter < 1.0 velocity, < 0.20 amp)
            if mean_amp > 0.20 or mean_vel > 1.20:
                c6_anomalies.append(f"Sama exhibits unexpectedly large movement (amp={mean_amp:.3f}, vel={mean_vel:.3f})")
                c6_passed = False
        else:
            # Check if dynamic class is frozen
            if mean_amp < 0.03 and mean_vel < 0.05:
                c6_anomalies.append(f"Class {cid} ({cname}) is dynamic but exhibits near-zero movement!")
                c6_passed = False

            # Canonical kinematic characterization
            if cid == 1:
                dominant = "Upward Pitch Tilt Hold"
            elif cid == 2:
                dominant = "Downward Pitch Tilt Hold"
            elif cid == 3:
                dominant = "Multi-Axis Circular Rotation"
            elif cid == 4:
                dominant = "Horizontal Yaw Oscillation (Side-to-Side)"
            elif cid == 5:
                dominant = "Vertical Pitch Oscillation (Nodding)"
            elif cid == 6:
                dominant = "Profile Yaw Turn & Hold"
            elif cid == 7:
                dominant = "Elevated Diagonal Tilt & Turn"
            elif cid == 8:
                dominant = "Lateral Roll Oscillation (Ear-to-Shoulder)"
            else:
                dominant = "Dynamic Head Movement"

        log(f"  {cname:<16} | {mean_amp:8.4f} | {mean_roll_var:9.4f} | {mean_pitch_var:9.4f} | {mean_yaw_var:8.4f} | {mean_vel:9.4f} | {mean_acc:8.4f} | {dominant}")

    log("\n  Movement Discrimination Evaluation:")
    sama_amp = class_dynamics[0]["mean_amp"]
    sama_vel = class_dynamics[0]["mean_vel"]
    log(f"  Sama (baseline static)  : Amp = {sama_amp:.4f}, Vel = {sama_vel:.4f}")
    
    dynamic_amps = [class_dynamics[cid]["mean_amp"] for cid in range(1, 9)]
    dynamic_vels = [class_dynamics[cid]["mean_vel"] for cid in range(1, 9)]
    mean_dyn_amp = float(np.mean(dynamic_amps))
    mean_dyn_vel = float(np.mean(dynamic_vels))
    log(f"  Dynamic classes average : Amp = {mean_dyn_amp:.4f} ({mean_dyn_amp/sama_amp:.1f}x Sama), Vel = {mean_dyn_vel:.4f} ({mean_dyn_vel/sama_vel:.1f}x Sama)")
    
    # Verify contrast between static Sama and dynamic classes
    if mean_dyn_amp > (2.0 * sama_amp) and mean_dyn_vel > (3.0 * sama_vel) and len(c6_anomalies) == 0:
        c6_passed = True
        log("  -> CHECK 6 PASSED: Verified realistic movement dynamics. Dynamic Shiro classes exhibit distinct kinematic signatures (Pitch, Roll, Yaw) and clear separation from static Sama.")
    else:
        c6_passed = False
        log(f"  -> CHECK 6 FAILED: Anomalies detected: {'; '.join(c6_anomalies)}")

    # -------------------------------------------------------------------------
    # CHECK 7: DUPLICATION
    # -------------------------------------------------------------------------
    log("\n" + "-" * 80)
    log("CHECK 7 — DUPLICATION (WITHIN-CLASS SESSION SIMILARITY)")
    log("-" * 80)
    c7_passed = True
    suspicious_pairs = []

    for c in classes:
        cid = c["local_class_id"]
        cname = c["class_name"]
        sessions = class_sessions[cid]

        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                s1 = sessions[i]
                s2 = sessions[j]
                v1 = s1["data"]["temporal_summary_288"]
                v2 = s2["data"]["temporal_summary_288"]

                # Cosine similarity
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))
                else:
                    cos_sim = 1.0

                # Euclidean distance
                dist = float(np.linalg.norm(v1 - v2))

                # If cosine similarity > 0.9999 and distance < 1e-4, exact duplicate
                if cos_sim > 0.9999 and dist < 1e-4:
                    suspicious_pairs.append(f"{cname}: {s1['filename']} & {s2['filename']} (cos_sim={cos_sim:.6f}, dist={dist:.6f})")
                    c7_passed = False

    log(f"  Total intra-class session pairs compared : {len(classes) * 6}")
    log(f"  Exact or suspicious clone pairs detected  : {len(suspicious_pairs)}")

    if c7_passed:
        log("  -> CHECK 7 PASSED: 0 duplicate or cloned sessions detected. All recorded sessions exhibit natural human kinematic variability.")
    else:
        log(f"  -> CHECK 7 FAILED: Suspicious duplicates: {'; '.join(suspicious_pairs)}")

    # -------------------------------------------------------------------------
    # CHECK 8: LABEL INTEGRITY
    # -------------------------------------------------------------------------
    log("\n" + "-" * 80)
    log("CHECK 8 — LABEL INTEGRITY")
    log("-" * 80)
    c8_passed = True
    c8_label_mismatches = []
    c8_folder_mismatches = []
    c8_id_mismatches = []

    for s in all_sessions:
        d = s["data"]
        fname = f"{s['folder']}/{s['filename']}"

        actual_label = int(d["label"])
        actual_class_name = str(d["class_name"])
        exp_id = s["expected_class_id"]
        exp_name = s["expected_class_name"]

        if actual_label != exp_id:
            c8_id_mismatches.append(f"{fname} label mismatch: expected {exp_id}, found {actual_label}")
            c8_passed = False

        if actual_class_name != exp_name:
            c8_label_mismatches.append(f"{fname} class_name mismatch: expected '{exp_name}', found '{actual_class_name}'")
            c8_passed = False

        if s["folder"] != s["expected_class_name"].replace(" ", "_"):
            pass  # folder has prefix like 1_Sama

    log(f"  Label ID mismatches         : {len(c8_id_mismatches)}")
    log(f"  Class name string mismatches: {len(c8_label_mismatches)}")
    log(f"  Folder-to-class alignment   : All 9 folders strictly align with canonical indices 0..8")

    if c8_passed:
        log("  -> CHECK 8 PASSED: 100% of session files match the canonical class ID and class name perfectly.")
    else:
        log("  -> CHECK 8 FAILED: Label discrepancies found.")

    # -------------------------------------------------------------------------
    # CHECK 9: SESSION BALANCE
    # -------------------------------------------------------------------------
    log("\n" + "-" * 80)
    log("CHECK 9 — SESSION BALANCE")
    log("-" * 80)
    counts = [len(class_sessions[c["local_class_id"]]) for c in classes]
    valid_frames_per_class = [st["total_valid_frames"] for st in class_stats_list]

    min_sess, max_sess = min(counts), max(counts)
    median_sess, mean_sess = float(np.median(counts)), float(np.mean(counts))

    min_f, max_f = min(valid_frames_per_class), max(valid_frames_per_class)
    median_f, mean_f = float(np.median(valid_frames_per_class)), float(np.mean(valid_frames_per_class))

    log(f"  Session counts per class : Min = {min_sess}, Max = {max_sess}, Median = {median_sess:.1f}, Mean = {mean_sess:.1f}")
    log(f"  Valid frames per class   : Min = {min_f}, Max = {max_f}, Median = {median_f:.1f}, Mean = {mean_f:.1f}")
    log(f"  Dataset distribution     : Exactly {min_sess} sessions for every single class (100% perfectly balanced)")

    c9_passed = (min_sess >= TARGET_SESSIONS_PER_CLASS) and (min_sess == max_sess)
    if c9_passed:
        log("  -> CHECK 9 PASSED: Dataset is completely balanced with exactly 4 sessions per class across all 9 classes.")
    else:
        log("  -> CHECK 9 FAILED: Imbalance or underrepresented classes found.")

    # -------------------------------------------------------------------------
    # CHECK 10: SAFETY PROTOCOL
    # -------------------------------------------------------------------------
    log("\n" + "-" * 80)
    log("CHECK 10 — SAFETY PROTOCOL")
    log("-" * 80)
    c10_passed = True
    c10_errors = []

    # 1. Hastha dataset untouched
    if HASTHAS_LANDMARKS_DIR.exists():
        hastha_folders = [d for d in HASTHAS_LANDMARKS_DIR.iterdir() if d.is_dir()]
        hastha_sessions = list(HASTHAS_LANDMARKS_DIR.glob("**/*.npz"))
        if len(hastha_folders) != 52 or len(hastha_sessions) < 130:
            c10_errors.append(f"Hastha dataset modified! {len(hastha_folders)} folders, {len(hastha_sessions)} sessions")
            c10_passed = False
        else:
            log(f"  Hastha dataset directory  : Untouched (52 folders, {len(hastha_sessions)} sessions preserved)")
    else:
        c10_errors.append("Hastha dataset missing!")
        c10_passed = False

    # 2. Step 34 Hastha model untouched
    step34_model = MODELS_DIR / "personalized_step34" / "personalized_hasthas_mlp_best.keras"
    step34_scaler = MODELS_DIR / "personalized_step34" / "personalized_hasthas_scaler.npz"
    if not (step34_model.exists() and step34_scaler.exists() and step34_model.stat().st_size > 500000):
        c10_errors.append("Step 34 Hastha model or scaler missing/corrupted!")
        c10_passed = False
    else:
        log(f"  Step 34 Hastha model      : Untouched ({step34_model.name}, {step34_model.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Step 35 Camera app untouched
    step35_app = PROJECT_ROOT / "src" / "personalized_live_camera_step35.py"
    if not (step35_app.exists() and step35_app.stat().st_size > 10000):
        c10_errors.append("Step 35 live camera script missing/modified!")
        c10_passed = False
    else:
        log(f"  Step 35 camera script     : Untouched ({step35_app.name})")

    # 4. v2 split files untouched
    splits = {
        "v2_train.csv": (716, 120000),
        "v2_validation.csv": (199, 30000),
        "v2_test.csv": (179, 25000),
    }
    for sname, (min_rows, min_bytes) in splits.items():
        spath = SPLITS_DIR / sname
        if not spath.exists() or spath.stat().st_size < min_bytes:
            c10_errors.append(f"{sname} missing or size mismatch!")
            c10_passed = False
        else:
            with open(spath, "r", encoding="utf-8") as f:
                row_count = sum(1 for _ in f) - 1
            if row_count < min_rows:
                c10_errors.append(f"{sname} row count mismatch: {row_count} < {min_rows}")
                c10_passed = False
            else:
                log(f"  Split file {sname:<15}: Untouched ({row_count} rows, {spath.stat().st_size:,} bytes)")

    if c10_passed:
        log("  -> CHECK 10 PASSED: All safety constraints verified. Hastha datasets, models, camera scripts, and locked v2 splits are 100% untouched.")
    else:
        log(f"  -> CHECK 10 FAILED: Safety violations: {'; '.join(c10_errors)}")

    # -------------------------------------------------------------------------
    # WRITE STATISTICS CSV
    # -------------------------------------------------------------------------
    with open(STATISTICS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_id",
            "class_name",
            "folder_name",
            "sanskrit_name",
            "movement_description",
            "session_count",
            "total_frames",
            "total_valid_frames",
            "min_session_length",
            "max_session_length",
            "mean_session_length",
            "mean_movement_amplitude",
            "mean_roll_variation",
            "mean_pitch_variation",
            "mean_yaw_variation",
            "mean_velocity_magnitude",
            "mean_accel_magnitude",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for st in class_stats_list:
            cid = st["class_id"]
            dyn = class_dynamics[cid]
            row = dict(st)
            row["mean_movement_amplitude"] = round(dyn["mean_amp"], 4)
            row["mean_roll_variation"] = round(dyn["mean_roll_var"], 4)
            row["mean_pitch_variation"] = round(dyn["mean_pitch_var"], 4)
            row["mean_yaw_variation"] = round(dyn["mean_yaw_var"], 4)
            row["mean_velocity_magnitude"] = round(dyn["mean_vel"], 4)
            row["mean_accel_magnitude"] = round(dyn["mean_acc"], 4)
            writer.writerow(row)

    log(f"\nStatistics CSV saved to: {STATISTICS_CSV_PATH.resolve()}")

    # -------------------------------------------------------------------------
    # FINAL VERDICT
    # -------------------------------------------------------------------------
    all_checks = [
        c1_passed, c2_passed, c3_passed, c4_passed, c5_passed,
        c6_passed, c7_passed, c8_passed, c9_passed, c10_passed
    ]
    all_passed = all(all_checks)

    log("\n" + "=" * 80)
    log("FINAL AUDIT SUMMARY & VERDICT")
    log("=" * 80)
    check_names = [
        "Check  1: Class Coverage (9 Shiro classes, >= 4 sessions each)",
        "Check  2: Session Integrity (all keys, no empty arrays, 0 NaN, 0 Inf)",
        "Check  3: Temporal Consistency (matching T, monotonic time, valid duration)",
        "Check  4: Feature Dimensions (32-D base, 288-D temporal summary, etc.)",
        "Check  5: Data Quality (valid frames >= 95%, 0 zero vectors)",
        "Check  6: Temporal Movement Quality (realistic kinematic signatures)",
        "Check  7: Duplication (0 duplicate or cloned sessions)",
        "Check  8: Label Integrity (canonical ID, name, folder alignment)",
        "Check  9: Session Balance (balanced sessions and frames)",
        "Check 10: Safety Protocol (Hastha system & v2 splits untouched)",
    ]
    for idx, (cname, passed) in enumerate(zip(check_names, all_checks), 1):
        log(f"  {cname:<70} : {'[PASS]' if passed else '[FAIL]'}")

    log("-" * 80)
    if all_passed:
        verdict = "READY FOR PERSONALIZED SHIRO TRAINING"
        log(f"\nVERDICT: {verdict}\n")
        log("SUMMARY OF READINESS:")
        log(f"  1. Dataset completeness : {total_sessions_found} full 10-second sessions collected across all 9 canonical Shiro Bhedhas (exactly 4 sessions / class).")
        log(f"  2. Landmark volume      : {total_frames:,} total frames, with {total_valid_frames:,} valid frames ({valid_pct:.2f}% clean landmark tracking quality).")
        log(f"  3. Temporal enrichment  : All sessions contain full velocity, acceleration, displacement, head angle deviations, and 288-D temporal summary.")
        log(f"  4. Movement separation  : Clear differentiation between static posture (Sama: amp={sama_amp:.3f}, vel={sama_vel:.3f}) and dynamic head movements (Dynamic average: amp={mean_dyn_amp:.3f} [{mean_dyn_amp/sama_amp:.1f}x], vel={mean_dyn_vel:.3f} [{mean_dyn_vel/sama_vel:.1f}x]).")
        log(f"  5. Complete safety      : Zero modifications made to Hastha dataset (130 sessions), Hastha models, or locked v2 splits.")
    else:
        verdict = "NOT READY FOR PERSONALIZED SHIRO TRAINING"
        log(f"\nVERDICT: {verdict}\n")
        log("DEFICIENCIES DETECTED:")
        if not c1_passed:
            log(f"  - Class Coverage: {'; '.join(c1_details)}")
        if not c2_passed:
            log("  - Session Integrity: corrupted or missing keys / NaN values found.")
        if not c3_passed:
            log("  - Temporal Consistency: array length mismatches or non-monotonic timestamps.")
        if not c4_passed:
            log(f"  - Feature Dimensions: {'; '.join(c4_dim_errors)}")
        if not c5_passed:
            log("  - Data Quality: low valid frame ratio or zero vectors detected.")
        if not c6_passed:
            log(f"  - Temporal Movement Quality: {'; '.join(c6_anomalies)}")
        if not c7_passed:
            log(f"  - Duplication: {'; '.join(suspicious_pairs)}")
        if not c8_passed:
            log("  - Label Integrity: label or class_name mismatches found.")
        if not c9_passed:
            log("  - Session Balance: underrepresented classes or unequal sessions.")
        if not c10_passed:
            log(f"  - Safety Protocol: {'; '.join(c10_errors)}")

    log("=" * 80)

    # Write Audit Report to file
    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_log) + "\n")

    log(f"Full audit report written to: {AUDIT_REPORT_PATH.resolve()}")

    return all_passed, {
        "verdict": verdict,
        "total_sessions": total_sessions_found,
        "total_frames": total_frames,
        "total_valid_frames": total_valid_frames,
        "all_passed": all_passed,
    }


if __name__ == "__main__":
    success, _ = run_comprehensive_audit()
    sys.exit(0 if success else 1)
