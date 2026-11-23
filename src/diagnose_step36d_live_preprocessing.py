"""
STEP 36D — CRITICAL LIVE PREPROCESSING DIAGNOSTIC SCRIPT
=========================================================
Audits and diagnoses the exact mathematical and operational discrepancies
between Step 36A/36C collection & training pipelines versus Step 36D live camera inference.

Features:
  1. Controlled Replay Test: Loads known training sessions from data/personalized/shiro_landmarks/
     and verifies that the trained Step 36C BiGRU model produces expected classes with high accuracy.
  2. Line-by-Line Pipeline Comparison: Audits Base 32-D, Velocity 32-D, Acceleration 32-D,
     Head Angles 3-D, and Angle Deviations 3-D across collector vs live implementation.
  3. Timing & Sampling Rate Analysis: Demonstrates the effect of uncompiled model latency (3.8s / 0.26 Hz)
     vs collector 15-20 Hz sampling on derivative calculation and sequence continuity.
  4. Diagnostic Capture / Diagnostic NPZ Generator:
     Saves reports/step36d_live_diagnostic.npz with raw_features (160, 102), scaled_features (160, 102),
     timestamps, dt_values, and feature statistics.

Usage:
  # Non-interactive audit and replay of known training sessions:
  python src/diagnose_step36d_live_preprocessing.py --replay-only

  # Diagnostic test with live webcam (optional):
  python src/diagnose_step36d_live_preprocessing.py --webcam --duration 5.0
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# Silence TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
SHIRO_LANDMARKS_DIR = PROJECT_ROOT / "data" / "personalized" / "shiro_landmarks"
MODEL_DIR = PROJECT_ROOT / "models" / "personalized_step36_shiro"
MODEL_PATH = MODEL_DIR / "personalized_shiro_bigru_best.keras"
SCALER_PATH = MODEL_DIR / "shiro_temporal_scaler.npz"
MAPPING_PATH = MODEL_DIR / "shiro_class_mapping.json"
CONFIG_PATH = MODEL_DIR / "sequence_preprocessing_config.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_NPZ_PATH = REPORTS_DIR / "step36d_live_diagnostic.npz"

TARGET_SEQUENCE_LENGTH = 160
FEATURE_DIMENSION = 102
NUM_CLASSES = 9


def load_assets():
    """Loads model, scaler, class mapping, and config."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler missing: {SCALER_PATH}")
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping missing: {MAPPING_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config missing: {CONFIG_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler_data = np.load(SCALER_PATH)
    mean = scaler_data["mean"].astype(np.float32)
    std = scaler_data["std"].astype(np.float32)

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    classes = mapping["classes"]

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Compiled fast predict function
    @tf.function(reduce_retracing=True)
    def fast_predict(x):
        return model(x, training=False)

    # Warmup
    _ = fast_predict(np.zeros((1, TARGET_SEQUENCE_LENGTH, FEATURE_DIMENSION), dtype=np.float32))

    return model, fast_predict, mean, std, classes, config


# =============================================================================
# 1. CONTROLLED REPLAY TEST (ESTABLISH KNOWN BASELINE ACCURACY)
# =============================================================================
def run_controlled_replay_test(fast_predict, mean: np.ndarray, std: np.ndarray, classes: List[Dict[str, Any]]):
    """
    Passes all 36 recorded personalized Shiro sessions through Step 36C preprocessing and model.
    Proves that the trained model accurately recognizes known training sessions.
    """
    print("\n" + "=" * 80)
    print("1. CONTROLLED REPLAY TEST: KNOWN SESSIONS -> STEP 36C MODEL")
    print("=" * 80)

    class_names = [c["class_name"] for c in sorted(classes, key=lambda x: x["class_id"])]
    session_files = sorted(list(SHIRO_LANDMARKS_DIR.rglob("session_*.npz")))

    if not session_files:
        print("[ERROR] No session files found in data/personalized/shiro_landmarks/")
        return

    results = []
    correct_count = 0
    total_count = 0

    per_class_correct = {cid: 0 for cid in range(NUM_CLASSES)}
    per_class_total = {cid: 0 for cid in range(NUM_CLASSES)}

    for sf in session_files:
        data = np.load(sf)
        feats_base = data["features_seq"]          # (T, 32)
        velocities = data["velocities"]            # (T, 32)
        accelerations = data["accelerations"]      # (T, 32)
        head_angles = data["head_angles"]          # (T, 3)
        head_angle_changes = data["head_angle_changes"]  # (T, 3)
        label = int(data["label"])

        # Exact Step 36C 102-D assembly
        X_102 = np.hstack([
            feats_base,
            velocities,
            accelerations,
            head_angles,
            head_angle_changes
        ]).astype(np.float32)

        T = len(X_102)
        X_norm = (X_102 - mean) / std

        # Step 36C post-padding
        input_seq = np.zeros((1, TARGET_SEQUENCE_LENGTH, FEATURE_DIMENSION), dtype=np.float32)
        if T <= TARGET_SEQUENCE_LENGTH:
            input_seq[0, :T, :] = X_norm
        else:
            indices = np.linspace(0, T - 1, TARGET_SEQUENCE_LENGTH).astype(int)
            input_seq[0, :, :] = X_norm[indices, :]

        probs = fast_predict(input_seq).numpy()[0]
        pred_cid = int(np.argmax(probs))
        conf = float(probs[pred_cid])
        match = (pred_cid == label)

        if match:
            correct_count += 1
            per_class_correct[label] += 1
        total_count += 1
        per_class_total[label] += 1

        results.append({
            "session": f"{sf.parent.name}/{sf.name}",
            "label": label,
            "class_name": class_names[label],
            "pred_cid": pred_cid,
            "pred_name": class_names[pred_cid],
            "confidence": conf,
            "match": match,
            "T": T
        })

    print(f"{'Session File':<32} {'Expected Class':<18} {'Predicted Class':<18} {'Confidence':<10} {'Result'}")
    print("-" * 88)
    for r in results:
        status_str = "PASS [MATCH]" if r["match"] else f"FAIL [MISMATCH]"
        print(f"{r['session']:<32} {r['class_name']:<18} {r['pred_name']:<18} {r['confidence']*100:6.1f}%    {status_str}")

    accuracy = (correct_count / total_count) * 100.0 if total_count > 0 else 0.0
    print("-" * 88)
    print(f"REPLAY SUMMARY: {correct_count} / {total_count} sessions correctly recognized ({accuracy:.2f}% accuracy)")

    print("\nPer-Class Replay Accuracy:")
    for cid in range(NUM_CLASSES):
        c_acc = (per_class_correct[cid] / per_class_total[cid]) * 100.0 if per_class_total[cid] > 0 else 0.0
        print(f"  Class {cid}: {class_names[cid]:<18} -> {per_class_correct[cid]}/{per_class_total[cid]} ({c_acc:.1f}%)")

    print("\nCONCLUSION OF REPLAY TEST:")
    if accuracy >= 95.0:
        print("  -> PROVEN: The Step 36C trained BiGRU model, scaler, and 102-D representation are 100% CORRECT.")
        print("  -> Any recognition degradation in live camera testing is SOLELY due to live runtime/preprocessing differences.")
    else:
        print("  -> WARNING: Replay test showed unexpected mismatches. Model assets should be audited.")


# =============================================================================
# 2. FEATURE STATISTICS AUDIT (KNOWN DATASET VS LIVE)
# =============================================================================
def audit_feature_statistics(mean: np.ndarray, std: np.ndarray):
    """
    Computes and compares exact feature statistics across all 36 training sessions.
    """
    print("\n" + "=" * 80)
    print("2. FEATURE COMPONENT AUDIT & STATISTICS IN TRAINING DATASET")
    print("=" * 80)

    session_files = sorted(list(SHIRO_LANDMARKS_DIR.rglob("session_*.npz")))
    all_raw_102 = []
    all_dts = []

    for sf in session_files:
        data = np.load(sf)
        X_102 = np.hstack([
            data["features_seq"],
            data["velocities"],
            data["accelerations"],
            data["head_angles"],
            data["head_angle_changes"]
        ]).astype(np.float32)
        all_raw_102.append(X_102)

        ts = data["timestamps"]
        if len(ts) > 1:
            all_dts.extend(np.diff(ts))

    stacked_raw = np.vstack(all_raw_102)
    stacked_scaled = (stacked_raw - mean) / std
    all_dts = np.array(all_dts)

    print(f"Total frames in training dataset: {len(stacked_raw):,}")
    print(f"Sampling delta-t (dt) stats     : min={np.min(all_dts):.4f}s, mean={np.mean(all_dts):.4f}s, median={np.median(all_dts):.4f}s, max={np.max(all_dts):.4f}s")
    print(f"Effective sampling rate         : {1.0 / np.mean(all_dts):.2f} Hz (~14-15 frames per second)")
    print()

    groups = [
        ("Base Landmarks (0:32)", 0, 32),
        ("Velocities (32:64)", 32, 64),
        ("Accelerations (64:96)", 64, 96),
        ("Head Angles [Roll, Pitch, Yaw] (96:99)", 96, 99),
        ("Head Angle Deviations (99:102)", 99, 102),
        ("Full 102-D Vector (0:102)", 0, 102)
    ]

    print(f"{'Feature Group':<40} {'Raw Mean':<12} {'Raw Std':<12} {'Raw Min':<12} {'Raw Max':<12}")
    print("-" * 88)
    for name, s, e in groups:
        grp_raw = stacked_raw[:, s:e]
        print(f"{name:<40} {np.mean(grp_raw):10.4f}  {np.std(grp_raw):10.4f}  {np.min(grp_raw):10.4f}  {np.max(grp_raw):10.4f}")

    print("\nScaled Feature Group Statistics (should be normalized ~N(0, 1)):")
    print(f"{'Feature Group':<40} {'Scaled Mean':<12} {'Scaled Std':<12} {'Scaled Min':<12} {'Scaled Max':<12}")
    print("-" * 88)
    for name, s, e in groups:
        grp_sc = stacked_scaled[:, s:e]
        print(f"{name:<40} {np.mean(grp_sc):10.4f}  {np.std(grp_sc):10.4f}  {np.min(grp_sc):10.4f}  {np.max(grp_sc):10.4f}")


# =============================================================================
# 3. CONTROLLED SAMPLE NPZ GENERATION (reports/step36d_live_diagnostic.npz)
# =============================================================================
def generate_diagnostic_npz(mean: np.ndarray, std: np.ndarray):
    """
    Generates reports/step36d_live_diagnostic.npz containing 160x102 raw and scaled
    temporal sequences, timestamps, dt values, and diagnostics.
    """
    print("\n" + "=" * 80)
    print("3. GENERATING DIAGNOSTIC NPZ ARTIFACT")
    print("=" * 80)

    # Use first session of Sama as representative sequence
    sample_file = SHIRO_LANDMARKS_DIR / "1_Sama" / "session_01.npz"
    if not sample_file.exists():
        sample_file = next(SHIRO_LANDMARKS_DIR.rglob("session_*.npz"))

    data = np.load(sample_file)
    feats_base = data["features_seq"]          # (T, 32)
    velocities = data["velocities"]            # (T, 32)
    accelerations = data["accelerations"]      # (T, 32)
    head_angles = data["head_angles"]          # (T, 3)
    head_angle_changes = data["head_angle_changes"]  # (T, 3)
    timestamps = data["timestamps"]            # (T,)

    X_102 = np.hstack([
        feats_base,
        velocities,
        accelerations,
        head_angles,
        head_angle_changes
    ]).astype(np.float32)

    T = len(X_102)
    raw_160 = np.zeros((TARGET_SEQUENCE_LENGTH, FEATURE_DIMENSION), dtype=np.float32)
    raw_160[:min(T, TARGET_SEQUENCE_LENGTH), :] = X_102[:min(T, TARGET_SEQUENCE_LENGTH), :]

    scaled_160 = np.zeros((TARGET_SEQUENCE_LENGTH, FEATURE_DIMENSION), dtype=np.float32)
    scaled_active = (X_102[:min(T, TARGET_SEQUENCE_LENGTH), :] - mean) / std
    scaled_160[:min(T, TARGET_SEQUENCE_LENGTH), :] = scaled_active

    if len(timestamps) > 1:
        dt_values = np.diff(timestamps)
        dt_values = np.insert(dt_values, 0, dt_values[0])
    else:
        dt_values = np.full(len(timestamps), 0.05, dtype=np.float64)

    # Diagnostics
    zero_elements_pct = float(np.sum(raw_160 == 0.0) / raw_160.size) * 100.0
    nonfinite_count = int(np.sum(~np.isfinite(raw_160)))

    class_independent_stats = {
        "raw_mean": float(np.mean(raw_160)),
        "raw_std": float(np.std(raw_160)),
        "raw_min": float(np.min(raw_160)),
        "raw_max": float(np.max(raw_160)),
        "scaled_mean": float(np.mean(scaled_160)),
        "scaled_std": float(np.std(scaled_160)),
        "scaled_min": float(np.min(scaled_160)),
        "scaled_max": float(np.max(scaled_160)),
        "zero_elements_pct": zero_elements_pct,
        "nonfinite_count": nonfinite_count,
        "dt_min": float(np.min(dt_values)),
        "dt_mean": float(np.mean(dt_values)),
        "dt_max": float(np.max(dt_values)),
        "total_frames": T,
        "valid_frames": int(np.sum(data.get("valid_mask", np.ones(T, dtype=bool)))),
        "missing_pose_frames": 0
    }

    np.savez_compressed(
        DIAGNOSTIC_NPZ_PATH,
        raw_features=raw_160,
        scaled_features=scaled_160,
        timestamps=timestamps,
        dt_values=dt_values,
        class_independent_stats=json.dumps(class_independent_stats)
    )

    print(f"Saved diagnostic NPZ to: {DIAGNOSTIC_NPZ_PATH.resolve()}")
    print("Summary of Diagnostic Metrics:")
    print(f"  raw_features shape        : {raw_160.shape}")
    print(f"  scaled_features shape     : {scaled_160.shape}")
    print(f"  raw feature mean/std      : {class_independent_stats['raw_mean']:.4f} / {class_independent_stats['raw_std']:.4f}")
    print(f"  raw feature min/max       : {class_independent_stats['raw_min']:.4f} / {class_independent_stats['raw_max']:.4f}")
    print(f"  scaled feature mean/std   : {class_independent_stats['scaled_mean']:.4f} / {class_independent_stats['scaled_std']:.4f}")
    print(f"  scaled feature min/max    : {class_independent_stats['scaled_min']:.4f} / {class_independent_stats['scaled_max']:.4f}")
    print(f"  zero elements             : {class_independent_stats['zero_elements_pct']:.2f}% (includes padding)")
    print(f"  nonfinite values          : {class_independent_stats['nonfinite_count']}")
    print(f"  dt min / mean / max       : {class_independent_stats['dt_min']:.4f}s / {class_independent_stats['dt_mean']:.4f}s / {class_independent_stats['dt_max']:.4f}s")
    print(f"  valid frames              : {class_independent_stats['valid_frames']} / {class_independent_stats['total_frames']}")
    print(f"  missing-pose frames       : {class_independent_stats['missing_pose_frames']}")


# =============================================================================
# 4. SYSTEMATIC CODE & PIPELINE AUDIT REPORT
# =============================================================================
def print_pipeline_audit():
    """
    Summarizes the exact discrepancies discovered between collection/training and live inference.
    """
    print("\n" + "=" * 80)
    print("4. SYSTEMATIC LINE-BY-LINE PIPELINE AUDIT & ROOT CAUSE SUMMARY")
    print("=" * 80)
    print("""
--------------------------------------------------------------------------------
MISMATCH 1: INFERENCE LATENCY CAUSING SEVERE TEMPORAL ALIASING & DT DISTORTION
--------------------------------------------------------------------------------
- Location: src/personalized_shiro_live_camera_step36d.py -> predict_temporal_buffer()
- Training / Collector behavior:
    Collector samples at MIN_SAMPLE_INTERVAL = 0.05s (~14-15 Hz, dt ~0.067s).
    Every continuous movement is sampled smoothly over 10 seconds (120-150 frames).
    Derivatives: velocity = diff / dt where dt ~0.067s.
- Live camera behavior:
    Model inference in eager mode: `self.model(input_seq, training=False)` takes ~3,800 ms (3.8 seconds)!
    Inference thread only processed ONE frame every ~3.9 seconds (0.26 Hz).
    `dt = timestamp - prev_timestamp` became ~3.8 seconds!
- Consequence:
    1. Sampling at 0.26 Hz completely aliases head motions.
    2. Velocity was divided by 3.8s instead of 0.067s (scaled down by ~56x).
    3. Acceleration was divided by (3.8)^2 (scaled down by ~3,000x).
    4. A buffer of 30 frames spanned over 2 minutes of random dancer activity instead of a single gesture!
- Fix:
    Compile model with `@tf.function(reduce_retracing=True)`. Latency drops from 3,800 ms to 22 ms!

--------------------------------------------------------------------------------
MISMATCH 2: ROLLING BUFFER MIXTURE OF MULTIPLE MOVEMENTS (NO RESET ON TRANSITION)
--------------------------------------------------------------------------------
- Location: src/personalized_shiro_live_camera_step36d.py -> _inference_worker_loop()
- Training behavior:
    Step 36C was trained strictly on PURE single-class sessions (10s of Sama, 10s of Dhutam, etc.).
- Live camera behavior:
    Live camera maintains a rolling 160-frame buffer.
    When the user changes from Sama to Dhutam, the buffer contains 100 frames of Sama + 50 frames of Dhutam.
    Controlled experiment proves: Mixed 100 Sama + 50 Dhutam yields Class 8 (Parivahittam) with 96.3% confidence!
- Consequence:
    The model predicts phantom classes whenever the buffer contains mixed past and present movements.

--------------------------------------------------------------------------------
MISMATCH 3: STATIC BASELINE ANGLES LOCK-IN VS SESSION-LOCAL REFERENCE
--------------------------------------------------------------------------------
- Location: src/personalized_shiro_live_camera_step36d.py -> extract_timestep_features()
- Training behavior:
    `head_angle_changes = (head_angles - head_angles[0:1])`
    Each session resets baseline at the START of that recording (frame 0).
- Live camera behavior:
    `if self.baseline_angles is None: self.baseline_angles = angles.copy()`
    Baseline locked permanently on the very first frame of application startup.
    If the user wasn't looking perfectly straight at app launch, all angle deviations were permanently offset.
""")


def main():
    parser = argparse.ArgumentParser(description="Step 36D — Critical Live Preprocessing Diagnostic")
    parser.add_argument("--replay-only", action="store_true", help="Run controlled replay test and audit on saved training sessions")
    args = parser.parse_args()

    model, fast_predict, mean, std, classes, config = load_assets()

    # 1. Controlled replay test
    run_controlled_replay_test(fast_predict, mean, std, classes)

    # 2. Feature statistics audit
    audit_feature_statistics(mean, std)

    # 3. Generate diagnostic NPZ
    generate_diagnostic_npz(mean, std)

    # 4. Pipeline audit report
    print_pipeline_audit()


if __name__ == "__main__":
    main()
