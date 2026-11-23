"""
STEP 27 — Emergency Recognition Accuracy Check Script
Evaluates known development/validation samples using the exact final deployed models
from src.final_inference.py (KuchipudiInferenceEngine).
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.final_inference import KuchipudiInferenceEngine, CATEGORY_BY_KEY, CATEGORIES

def select_known_samples():
    """
    Selects representative samples from v2_validation.csv (and v2_train.csv if needed).
    Strictly avoids v2_test.csv.
    """
    val_csv = PROJECT_ROOT / "data" / "splits" / "v2_validation.csv"
    train_csv = PROJECT_ROOT / "data" / "splits" / "v2_train.csv"

    val_df = pd.read_csv(val_csv)
    train_df = pd.read_csv(train_csv)

    samples = {
        "1. Hasthas": [],
        "2. Shiro Bhedhas (HEAD MOVEMENTS)": [],
        "3. Drishti Bhedhas(EYE MOVEMENTS)": [],
        "4. Pada Karma(Foot movements)": []
    }

    # 1. HASTHAS (Required: Pathaka, Anjali, Kapothamu + others)
    target_hasthas = [
        "1. Pathaka",
        "1. Anjali",
        "2. Kapothamu",
        "10. Shikhara",
        "20.Alapadma",
        "12. Katakamukha",
        "26.Mukula"
    ]
    for cname in target_hasthas:
        match_val = val_df[(val_df["category"] == "1. Hasthas") & (val_df["class_name"] == cname)]
        if len(match_val) > 0:
            row = match_val.iloc[0]
            samples["1. Hasthas"].append((cname, PROJECT_ROOT / row["sample_path"]))
        else:
            match_train = train_df[(train_df["category"] == "1. Hasthas") & (train_df["class_name"] == cname)]
            if len(match_train) > 0:
                row = match_train.iloc[0]
                samples["1. Hasthas"].append((cname, PROJECT_ROOT / row["sample_path"]))

    # 2. SHIRO BHEDHAS (At least 5 known classes)
    target_shiro = [
        "1. Sama",
        "2. Udvahitam",
        "3. Adhomukham",
        "4. Alolitam",
        "5. Dhutam",
        "6. Kampitam"
    ]
    for cname in target_shiro:
        match_val = val_df[(val_df["category"] == "2. Shiro Bhedhas (HEAD MOVEMENTS)") & (val_df["class_name"] == cname)]
        if len(match_val) > 0:
            row = match_val.iloc[0]
            samples["2. Shiro Bhedhas (HEAD MOVEMENTS)"].append((cname, PROJECT_ROOT / row["sample_path"]))

    # 3. DRISHTI BHEDHAS (At least 5 known classes)
    target_drishti = [
        "1. Sama",
        "2. Alokita",
        "3. Saachi",
        "4. Pralokita",
        "5. Nimilita",
        "7. Anuvritta"
    ]
    for cname in target_drishti:
        match_val = val_df[(val_df["category"] == "3. Drishti Bhedhas(EYE MOVEMENTS)") & (val_df["class_name"] == cname)]
        if len(match_val) > 0:
            row = match_val.iloc[0]
            samples["3. Drishti Bhedhas(EYE MOVEMENTS)"].append((cname, PROJECT_ROOT / row["sample_path"]))

    # 4. PADA KARMA (At least 5 known classes)
    target_pada = [
        "1. Udghattita",
        "2. Sama",
        "3. Agratalasanchara",
        "4. Anchita Pada",
        "5. Kunchita",
        "6. Suchi Pada"
    ]
    for cname in target_pada:
        match_val = val_df[(val_df["category"] == "4. Pada Karma(Foot movements)") & (val_df["class_name"] == cname)]
        if len(match_val) > 0:
            row = match_val.iloc[0]
            samples["4. Pada Karma(Foot movements)"].append((cname, PROJECT_ROOT / row["sample_path"]))

    return samples

def run_emergency_check():
    print("=" * 80)
    print("STEP 27: EMERGENCY RECOGNITION ACCURACY CHECK ON KNOWN SAMPLES")
    print("=" * 80)

    samples_by_cat = select_known_samples()
    engine = KuchipudiInferenceEngine()

    results = {}

    for cat_key, sample_list in samples_by_cat.items():
        cat_short = CATEGORY_BY_KEY[cat_key]["short_name"]
        print(f"\n=======================================================")
        print(f"EVALUATING CATEGORY: {cat_short} ({len(sample_list)} Known Samples)")
        print(f"=======================================================")

        cat_results = []
        for true_class, img_path in sample_list:
            if not img_path.exists():
                print(f"  [MISSING FILE] {img_path}")
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  [CANNOT READ] {img_path}")
                continue

            engine.reset_category(cat_key)

            # For temporal movement categories (Shiro, Drishti, Pada), feed multiple frames
            # so the 16-frame temporal buffer is fully populated
            if cat_key != "1. Hasthas":
                for _ in range(16):
                    res = engine.process_category(frame, cat_key, apply_smoothing=False)
            else:
                res = engine.process_category(frame, cat_key, apply_smoothing=False)

            pred_class = res.get("gesture_name", "UNKNOWN")
            conf = res.get("confidence", 0.0)
            detected = res.get("detected", False)
            top5 = res.get("top5", [])

            is_correct = (pred_class.strip().lower() == true_class.strip().lower())
            status = "PASS" if is_correct else "FAIL"

            record = {
                "category": cat_short,
                "category_key": cat_key,
                "true_class": true_class,
                "pred_class": pred_class,
                "confidence": conf,
                "detected": detected,
                "status": status,
                "top5": top5,
                "path": str(img_path.relative_to(PROJECT_ROOT))
            }
            cat_results.append(record)

            print(f"  TRUE: {true_class:<22} | PRED: {pred_class:<22} | CONF: {conf:5.1f}% | {status} (Landmark: {detected})")

        results[cat_key] = cat_results

    # -------------------------------------------------------------------------
    # Live Camera Check (1 sample per category)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LIVE CAMERA COMPARISON CHECK (1 SAMPLE PER CATEGORY)")
    print("=" * 80)

    live_cam_results = {}
    cap = cv2.VideoCapture(0)
    has_cam = cap.isOpened()
    cam_frame = None
    if has_cam:
        for _ in range(5):  # discard initial warm up frames
            ret, cam_frame = cap.read()
        cap.release()

    live_targets = [
        ("1. Hasthas", "1. Pathaka"),
        ("2. Shiro Bhedhas (HEAD MOVEMENTS)", "1. Sama"),
        ("3. Drishti Bhedhas(EYE MOVEMENTS)", "1. Sama"),
        ("4. Pada Karma(Foot movements)", "2. Sama")
    ]

    for cat_key, true_gesture in live_targets:
        cat_short = CATEGORY_BY_KEY[cat_key]["short_name"]
        engine.reset_category(cat_key)
        if has_cam and cam_frame is not None:
            if cat_key != "1. Hasthas":
                for _ in range(16):
                    res_cam = engine.process_category(cam_frame, cat_key, apply_smoothing=False)
            else:
                res_cam = engine.process_category(cam_frame, cat_key, apply_smoothing=False)

            pred_cam = res_cam.get("gesture_name", "UNKNOWN")
            conf_cam = res_cam.get("confidence", 0.0)
            det_cam = res_cam.get("detected", False)
            print(f"  [LIVE CAM] Category: {cat_short:<16} | Target: {true_gesture:<12} | Predicted: {pred_cam} ({conf_cam:.1f}%) | Detected: {det_cam}")
            live_cam_results[cat_key] = {
                "target": true_gesture,
                "pred": pred_cam,
                "conf": conf_cam,
                "detected": det_cam
            }
        else:
            print(f"  [LIVE CAM] Camera not available for real-time capture.")
            live_cam_results[cat_key] = {
                "target": true_gesture,
                "pred": "N/A (Webcam not available)",
                "conf": 0.0,
                "detected": False
            }

    engine.close()
    return results, live_cam_results

if __name__ == "__main__":
    run_emergency_check()
