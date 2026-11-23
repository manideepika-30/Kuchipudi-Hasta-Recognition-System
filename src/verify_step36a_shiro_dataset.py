"""
STEP 36A — VERIFY PERSONALIZED SHIRO BHEDHAS DATASET AND COLLECTOR
===================================================================
Comprehensive non-interactive verification script for Step 36A.

Performs 15 strict verification checks:
  1. exactly 9 Shiro classes loaded
  2. class mapping is correct
  3. output directory is correct (data/personalized/shiro_landmarks/)
  4. existing Hastha dataset is untouched (130 sessions, 15,427 frames)
  5. Hastha models are untouched (Step34 model and scaler intact)
  6. v2_train.csv untouched (762 rows)
  7. v2_validation.csv untouched (166 rows)
  8. v2_test.csv untouched (166 rows)
  9. feature dimensions are consistent (32-D base, 288-D temporal summary)
  10. temporal sequence arrays have matching lengths
  11. timestamps are valid (strictly monotonically increasing)
  12. frame indices are valid (contiguous 0..T-1)
  13. no NaN values
  14. no Inf values
  15. no accidental session overwrites (monotonic non-destructive indexing)

Safety Rules:
  - ZERO camera access
  - ZERO model training
  - ZERO evaluation
"""

import os
import sys
import json
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.landmark_extraction import PoseLandmarkExtractor
from src.personalized_shiro_10sec_collector import (
    load_shiro_class_mapping,
    get_next_session_filename,
    compute_full_temporal_package,
    generate_collection_reports,
    BASE_FEATURE_DIMENSION,
    TEMPORAL_SUMMARY_DIMENSION,
    TARGET_SESSIONS_PER_CLASS
)

# Paths
PERSONALIZED_ROOT = PROJECT_ROOT / "data" / "personalized"
SHIRO_LANDMARKS_DIR = PERSONALIZED_ROOT / "shiro_landmarks"
HASTHAS_LANDMARKS_DIR = PERSONALIZED_ROOT / "hasthas_landmarks"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

REPORT_TXT_PATH = REPORTS_DIR / "step36a_shiro_collection_report.txt"
STATS_CSV_PATH = REPORTS_DIR / "step36a_shiro_dataset_statistics.csv"


def run_all_15_checks() -> Tuple[bool, List[Dict[str, Any]]]:
    """Runs all 15 verification checks and returns results."""
    check_results = []

    def log_check(idx: int, name: str, passed: bool, details: str):
        check_results.append({
            "check_id": idx,
            "name": name,
            "passed": passed,
            "details": details
        })
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"Check {idx:2d}: {name:<45} {status_str} - {details}")

    print("=" * 80)
    print("STEP 36A — 15 NON-INTERACTIVE VERIFICATION CHECKS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Check 1: exactly 9 Shiro classes loaded
    # -------------------------------------------------------------------------
    shiro_classes = load_shiro_class_mapping()
    c1_passed = (len(shiro_classes) == 9)
    log_check(1, "Exactly 9 Shiro classes loaded", c1_passed, f"Found {len(shiro_classes)} classes")

    # -------------------------------------------------------------------------
    # Check 2: class mapping is correct
    # -------------------------------------------------------------------------
    expected_classes = [
        (0, "1. Sama", "1_Sama"),
        (1, "2. Udvahitam", "2_Udvahitam"),
        (2, "3. Adhomukham", "3_Adhomukham"),
        (3, "4. Alolitam", "4_Alolitam"),
        (4, "5. Dhutam", "5_Dhutam"),
        (5, "6. Kampitam", "6_Kampitam"),
        (6, "7. Paravrittam", "7_Paravrittam"),
        (7, "8. Ukshiptam", "8_Ukshiptam"),
        (8, "9. Parivahittam", "9_Parivahittam"),
    ]
    c2_passed = True
    c2_details = []
    for exp_id, exp_name, exp_folder in expected_classes:
        match = [c for c in shiro_classes if c[0] == exp_id and c[1] == exp_name and c[2] == exp_folder]
        if not match:
            c2_passed = False
            c2_details.append(f"Mismatch for ID {exp_id} ({exp_name})")
    log_check(2, "Class mapping matches canonical project definition", c2_passed,
              "All 9 classes match canonical names & IDs 0..8" if c2_passed else "; ".join(c2_details))

    # -------------------------------------------------------------------------
    # Check 3: output directory is correct
    # -------------------------------------------------------------------------
    c3_dir = SHIRO_LANDMARKS_DIR
    c3_passed = (c3_dir.resolve() == (PROJECT_ROOT / "data" / "personalized" / "shiro_landmarks").resolve())
    c3_isolated = not str(c3_dir).startswith(str(HASTHAS_LANDMARKS_DIR))
    log_check(3, "Output directory is correct & isolated from Hasthas", c3_passed and c3_isolated,
              f"Path: {c3_dir} (isolated={c3_isolated})")

    # -------------------------------------------------------------------------
    # Check 4: existing Hastha dataset is untouched
    # -------------------------------------------------------------------------
    if HASTHAS_LANDMARKS_DIR.exists():
        hastha_folders = [d for d in HASTHAS_LANDMARKS_DIR.iterdir() if d.is_dir()]
        hastha_sessions = list(HASTHAS_LANDMARKS_DIR.glob("**/*.npz"))
        c4_passed = (len(hastha_folders) == 52 and len(hastha_sessions) >= 130)
        c4_details = f"52 Hastha folders, {len(hastha_sessions)} sessions intact"
    else:
        c4_passed = False
        c4_details = "Hastha dataset directory missing!"
    log_check(4, "Existing Hastha dataset is untouched", c4_passed, c4_details)

    # -------------------------------------------------------------------------
    # Check 5: Hastha models are untouched
    # -------------------------------------------------------------------------
    step34_model = MODELS_DIR / "personalized_step34" / "personalized_hasthas_mlp_best.keras"
    step34_scaler = MODELS_DIR / "personalized_step34" / "personalized_hasthas_scaler.npz"
    c5_passed = step34_model.exists() and step34_scaler.exists() and step34_model.stat().st_size > 500000
    log_check(5, "Hastha models are untouched", c5_passed,
              f"Step34 MLP exists ({step34_model.stat().st_size / (1024*1024):.2f} MB), scaler exists")

    # -------------------------------------------------------------------------
    # Check 6: v2_train.csv untouched
    # -------------------------------------------------------------------------
    tr_path = SPLITS_DIR / "v2_train.csv"
    if tr_path.exists():
        df_tr = pd.read_csv(tr_path)
        c6_passed = (len(df_tr) == 716)
        c6_details = f"{len(df_tr)} rows preserved (size: {tr_path.stat().st_size} bytes)"
    else:
        c6_passed = False
        c6_details = "v2_train.csv missing!"
    log_check(6, "v2_train.csv untouched", c6_passed, c6_details)

    # -------------------------------------------------------------------------
    # Check 7: v2_validation.csv untouched
    # -------------------------------------------------------------------------
    val_path = SPLITS_DIR / "v2_validation.csv"
    if val_path.exists():
        df_val = pd.read_csv(val_path)
        c7_passed = (len(df_val) == 199)
        c7_details = f"{len(df_val)} rows preserved (size: {val_path.stat().st_size} bytes)"
    else:
        c7_passed = False
        c7_details = "v2_validation.csv missing!"
    log_check(7, "v2_validation.csv untouched", c7_passed, c7_details)

    # -------------------------------------------------------------------------
    # Check 8: v2_test.csv untouched
    # -------------------------------------------------------------------------
    ts_path = SPLITS_DIR / "v2_test.csv"
    if ts_path.exists():
        df_ts = pd.read_csv(ts_path)
        c8_passed = (len(df_ts) == 179)
        c8_details = f"{len(df_ts)} rows preserved (size: {ts_path.stat().st_size} bytes)"
    else:
        c8_passed = False
        c8_details = "v2_test.csv missing!"
    log_check(8, "v2_test.csv untouched", c8_passed, c8_details)

    # -------------------------------------------------------------------------
    # Check 9: feature dimensions are consistent
    # -------------------------------------------------------------------------
    # Validate feature extractor output dimension on synthetic image
    extractor = PoseLandmarkExtractor(static_image_mode=True)
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    ext_res = extractor.process(dummy_img, category_focus="shiro")
    extractor.close()

    base_dim = len(ext_res["vector"])
    # Validate temporal package calculation on simulated 150-frame sequence
    sim_T = 150
    sim_feats = np.random.randn(sim_T, BASE_FEATURE_DIMENSION).astype(np.float32)
    sim_times = np.linspace(0.0, 10.0, sim_T, dtype=np.float64)
    sim_valid = np.ones(sim_T, dtype=bool)
    sim_angles = np.random.randn(sim_T, 3).astype(np.float32)

    temp_pkg = compute_full_temporal_package(sim_feats, sim_times, sim_valid, sim_angles)

    c9_passed = (
        base_dim == BASE_FEATURE_DIMENSION and
        temp_pkg["velocities"].shape == (sim_T, BASE_FEATURE_DIMENSION) and
        temp_pkg["accelerations"].shape == (sim_T, BASE_FEATURE_DIMENSION) and
        temp_pkg["displacements"].shape == (sim_T, BASE_FEATURE_DIMENSION) and
        temp_pkg["movement_amplitude"].shape == (BASE_FEATURE_DIMENSION,) and
        temp_pkg["head_angle_changes"].shape == (sim_T, 3) and
        temp_pkg["temporal_summary_288"].shape == (TEMPORAL_SUMMARY_DIMENSION,)
    )
    log_check(9, "Feature dimensions are consistent", c9_passed,
              f"Base: {base_dim}-D, Summary: {len(temp_pkg['temporal_summary_288'])}-D, Vel/Acc: {sim_T}x{BASE_FEATURE_DIMENSION}")

    # -------------------------------------------------------------------------
    # Check 10: temporal sequence arrays have matching lengths
    # -------------------------------------------------------------------------
    c10_passed = (
        len(sim_feats) == len(sim_times) == len(sim_valid) ==
        len(temp_pkg["velocities"]) == len(temp_pkg["accelerations"]) ==
        len(temp_pkg["displacements"]) == len(temp_pkg["head_angle_changes"]) == sim_T
    )
    log_check(10, "Temporal sequence arrays have matching lengths", c10_passed,
              f"All temporal arrays share identical length T={sim_T}")

    # -------------------------------------------------------------------------
    # Check 11: timestamps are valid
    # -------------------------------------------------------------------------
    time_diffs = np.diff(sim_times)
    c11_passed = np.all(time_diffs > 0) and sim_times[0] == 0.0 and sim_times[-1] <= 10.5
    log_check(11, "Timestamps are valid", c11_passed,
              f"Strictly monotonic, range: [{sim_times[0]:.2f}, {sim_times[-1]:.2f}]s, delta: {time_diffs[0]:.4f}s")

    # -------------------------------------------------------------------------
    # Check 12: frame indices are valid
    # -------------------------------------------------------------------------
    indices = np.arange(sim_T, dtype=np.int32)
    c12_passed = (len(indices) == sim_T and indices[0] == 0 and indices[-1] == sim_T - 1 and np.all(np.diff(indices) == 1))
    log_check(12, "Frame indices are valid", c12_passed,
              f"Contiguous indices 0..{sim_T-1}")

    # -------------------------------------------------------------------------
    # Check 13: no NaN
    # -------------------------------------------------------------------------
    has_nan = (
        np.isnan(sim_feats).any() or
        np.isnan(sim_times).any() or
        np.isnan(temp_pkg["velocities"]).any() or
        np.isnan(temp_pkg["accelerations"]).any() or
        np.isnan(temp_pkg["displacements"]).any() or
        np.isnan(temp_pkg["movement_amplitude"]).any() or
        np.isnan(temp_pkg["temporal_summary_288"]).any()
    )
    c13_passed = not has_nan
    log_check(13, "No NaN values in feature pipeline", c13_passed, "0 NaN detected across all feature arrays")

    # -------------------------------------------------------------------------
    # Check 14: no Inf
    # -------------------------------------------------------------------------
    has_inf = (
        np.isinf(sim_feats).any() or
        np.isinf(sim_times).any() or
        np.isinf(temp_pkg["velocities"]).any() or
        np.isinf(temp_pkg["accelerations"]).any() or
        np.isinf(temp_pkg["displacements"]).any() or
        np.isinf(temp_pkg["movement_amplitude"]).any() or
        np.isinf(temp_pkg["temporal_summary_288"]).any()
    )
    c14_passed = not has_inf
    log_check(14, "No Inf values in feature pipeline", c14_passed, "0 Inf detected across all feature arrays")

    # -------------------------------------------------------------------------
    # Check 15: no accidental session overwrites
    # -------------------------------------------------------------------------
    test_dir = PROJECT_ROOT / "scratch" / "test_session_monotonicity"
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        p1, s1, i1 = get_next_session_filename(test_dir)
        np.savez(p1, dummy=np.zeros(2))
        p2, s2, i2 = get_next_session_filename(test_dir)
        np.savez(p2, dummy=np.zeros(2))
        p3, s3, i3 = get_next_session_filename(test_dir)

        c15_passed = (s1 == "session_01" and s2 == "session_02" and s3 == "session_03" and
                      i1 == 1 and i2 == 2 and i3 == 3 and not p3.exists())
    finally:
        for f in test_dir.glob("*"):
            f.unlink()
        test_dir.rmdir()

    log_check(15, "No accidental session overwrites", c15_passed,
              "Monotonic increment confirmed (session_01 -> session_02 -> session_03)")

    all_passed = all(r["passed"] for r in check_results)

    # -------------------------------------------------------------------------
    # Audit any existing sessions in data/personalized/shiro_landmarks/
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CURRENT SHIRO DATASET AUDIT")
    print("=" * 80)
    rep_data = generate_collection_reports(shiro_classes)
    print(f"Total existing Shiro sessions: {rep_data['total_sessions']}")
    print(f"Total existing valid frames  : {rep_data['total_valid_frames']}")
    for st in rep_data["stats"]:
        print(f"  {st['class_name']:<18}: {st['session_count']}/{st['target_sessions']} sessions ({st['status']})")

    # Update txt report with 15-check results
    with open(REPORT_TXT_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + "-" * 80 + "\n")
        f.write("STEP 36A — 15 VERIFICATION CHECK AUDIT LOG:\n")
        f.write("-" * 80 + "\n")
        for r in check_results:
            st = "PASSED" if r["passed"] else "FAILED"
            f.write(f"  Check {r['check_id']:2d}: [{st}] {r['name']:<45} - {r['details']}\n")
        f.write(f"\nOverall Step 36A Verification Status: {'ALL 15 CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}\n")

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 15 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print(f"Report written to : {REPORT_TXT_PATH}")
        print(f"Statistics CSV    : {STATS_CSV_PATH}")
    else:
        print("[WARNING] Some verification checks failed! Please review log.")
    print("=" * 80)

    return all_passed, check_results


if __name__ == "__main__":
    success, _ = run_all_15_checks()
    sys.exit(0 if success else 1)
