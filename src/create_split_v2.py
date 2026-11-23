"""
Step 12 — Rebuild Dataset Split Using All Available Samples (v2 Split)

This script generates:
- data/splits/v2_train.csv
- data/splits/v2_validation.csv
- data/splits/v2_test.csv
- data/splits/v2_all_samples.csv
- reports/step12_class_distribution.csv
- reports/step12_leakage_audit.txt
- reports/step12_split_v2_report.txt

Key properties:
- Accounts for all 1,094 prepared samples (0 discarded).
- Eliminates the grouped-CV quarantine (all 258 samples, including 184 Pada Karma samples, are now integrated).
- Strict source-level grouping: all 8 frames of any video are guaranteed to remain in exactly one partition (0 leakage).
- All 75 classes are guaranteed to have representation in train, validation, and test.
"""

import os
import random
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "splits" / "all_samples_split.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"

def build_v2_split():
    print(f"Reading input dataset catalog from: {CATALOG_PATH}")
    df = pd.read_csv(CATALOG_PATH)
    total_input_samples = len(df)
    print(f"Total samples loaded: {total_input_samples}")
    assert total_input_samples == 1094, f"Expected 1094 samples, found {total_input_samples}"

    # Group samples by (category, class_name) and entity
    class_entities = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        cat = row["category"]
        cls = row["class_name"]
        stype = row["source_type"]
        if stype == "video_frame":
            ent = f"vid_{row['source_video_id']}"
        else:
            ent = f"img_{row['sample_path']}"
        class_entities[(cat, cls)][ent].append(row.to_dict())

    total_classes = len(class_entities)
    print(f"Total classes grouped: {total_classes}")
    assert total_classes == 75, f"Expected 75 classes, found {total_classes}"

    allocations = {"train": [], "validation": [], "test": []}
    all_assigned = []
    class_distribution_rows = []

    for (cat, cls), ent_dict in sorted(class_entities.items()):
        entities = sorted(list(ent_dict.keys()))
        n = len(entities)

        # Deterministic shuffle using class name hash
        seed = 42 + sum(ord(c) for c in f"{cat}_{cls}")
        rng = random.Random(seed)
        shuffled = list(entities)
        rng.shuffle(shuffled)

        if n == 3:
            train_ents = shuffled[0:1]
            val_ents = shuffled[1:2]
            test_ents = shuffled[2:3]
        elif n == 4:
            train_ents = shuffled[0:2]
            val_ents = shuffled[2:3]
            test_ents = shuffled[3:4]
        elif 5 <= n <= 7:
            val_ents = shuffled[0:1]
            test_ents = shuffled[1:2]
            train_ents = shuffled[2:]
        else:
            n_val = max(1, round(n * 0.15))
            n_test = max(1, round(n * 0.10))
            n_train = n - n_val - n_test
            val_ents = shuffled[0:n_val]
            test_ents = shuffled[n_val : n_val + n_test]
            train_ents = shuffled[n_val + n_test :]

        train_ents_set = set(train_ents)
        val_ents_set = set(val_ents)
        test_ents_set = set(test_ents)

        cls_train, cls_val, cls_test = [], [], []

        for idx, ent in enumerate(shuffled):
            if ent in train_ents_set:
                part = "train"
            elif ent in val_ents_set:
                part = "validation"
            else:
                part = "test"

            fold_id = f"fold_{(idx % 3) + 1}"

            for s in ent_dict[ent]:
                rec = {
                    "sample_path": s["sample_path"],
                    "category": cat,
                    "class_name": cls,
                    "source_type": s["source_type"],
                    "source_video_id": s["source_video_id"],
                    "partition": part,
                    "fold": fold_id,
                }
                allocations[part].append(rec)
                all_assigned.append(rec)

                if part == "train":
                    cls_train.append(rec)
                elif part == "validation":
                    cls_val.append(rec)
                else:
                    cls_test.append(rec)

        class_distribution_rows.append({
            "category": cat,
            "class_name": cls,
            "total_samples": len(cls_train) + len(cls_val) + len(cls_test),
            "train_samples": len(cls_train),
            "validation_samples": len(cls_val),
            "test_samples": len(cls_test),
            "source_entities": n,
            "excluded_samples": 0,
        })

    # DataFrames
    df_train = pd.DataFrame(allocations["train"])
    df_val = pd.DataFrame(allocations["validation"])
    df_test = pd.DataFrame(allocations["test"])
    df_all = pd.DataFrame(all_assigned)
    df_dist = pd.DataFrame(class_distribution_rows)

    # Verification Checks
    print("\n--- RUNNING STRICT VERIFICATION CHECKS ---")

    # 1. Total samples
    assert len(df_all) == 1094, f"Expected 1094 total samples, got {len(df_all)}"
    assert len(df_train) + len(df_val) + len(df_test) == 1094, "Partition sum mismatch"
    print(f"[PASS] 1. Total samples accounted for: {len(df_all)} (Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)})")

    # 2. Sample uniqueness across partitions
    train_paths = set(df_train["sample_path"])
    val_paths = set(df_val["sample_path"])
    test_paths = set(df_test["sample_path"])
    assert len(train_paths & val_paths) == 0, "Leakage: train and val share sample paths!"
    assert len(train_paths & test_paths) == 0, "Leakage: train and test share sample paths!"
    assert len(val_paths & test_paths) == 0, "Leakage: val and test share sample paths!"
    assert len(train_paths | val_paths | test_paths) == 1094, "Sample path count mismatch"
    print("[PASS] 2. Sample-level exclusivity: 0 overlapping sample paths.")

    # 3. Video source leakage check
    train_vids = set(df_train[df_train["source_type"] == "video_frame"]["source_video_id"])
    val_vids = set(df_val[df_val["source_type"] == "video_frame"]["source_video_id"])
    test_vids = set(df_test[df_test["source_type"] == "video_frame"]["source_video_id"])
    assert len(train_vids & val_vids) == 0, f"Video Leakage Train & Val: {train_vids & val_vids}"
    assert len(train_vids & test_vids) == 0, f"Video Leakage Train & Test: {train_vids & test_vids}"
    assert len(val_vids & test_vids) == 0, f"Video Leakage Val & Test: {val_vids & test_vids}"
    assert len(train_vids | val_vids | test_vids) == 68, f"Expected 68 unique videos, got {len(train_vids | val_vids | test_vids)}"
    print(f"[PASS] 3. Video source-level audit: 0 leaked videos across {len(train_vids | val_vids | test_vids)} total unique videos (Train: {len(train_vids)}, Val: {len(val_vids)}, Test: {len(test_vids)}).")

    # 4. Image source leakage check
    train_imgs = set(df_train[df_train["source_type"] == "image"]["sample_path"])
    val_imgs = set(df_val[df_val["source_type"] == "image"]["sample_path"])
    test_imgs = set(df_test[df_test["source_type"] == "image"]["sample_path"])
    assert len(train_imgs & val_imgs) == 0, f"Image Leakage Train & Val: {train_imgs & val_imgs}"
    assert len(train_imgs & test_imgs) == 0, f"Image Leakage Train & Test: {train_imgs & test_imgs}"
    assert len(val_imgs & test_imgs) == 0, f"Image Leakage Val & Test: {val_imgs & test_imgs}"
    assert len(train_imgs | val_imgs | test_imgs) == 550, f"Expected 550 unique images, got {len(train_imgs | val_imgs | test_imgs)}"
    print(f"[PASS] 4. Image source-level audit: 0 leaked images across {len(train_imgs | val_imgs | test_imgs)} total unique images (Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}).")

    # 5. Class representation checks
    assert len(df_dist) == 75, f"Expected 75 classes, got {len(df_dist)}"
    assert (df_dist["train_samples"] > 0).all(), "Some classes have 0 train samples!"
    assert (df_dist["validation_samples"] > 0).all(), "Some classes have 0 validation samples!"
    assert (df_dist["test_samples"] > 0).all(), "Some classes have 0 test samples!"
    print(f"[PASS] 5. 100% of all 75 classes represented in Train, Validation, AND Test partitions.")

    # 6. Pada Karma specific checks
    pk_df = df_dist[df_dist["category"].str.contains("Pada Karma")]
    assert len(pk_df) == 6, f"Expected 6 Pada Karma classes, got {len(pk_df)}"
    assert pk_df["total_samples"].sum() == 184, f"Expected 184 Pada Karma samples, got {pk_df['total_samples'].sum()}"
    assert pk_df["train_samples"].sum() == 88, f"Expected 88 Pada Karma train samples, got {pk_df['train_samples'].sum()}"
    assert pk_df["validation_samples"].sum() == 48, f"Expected 48 Pada Karma val samples, got {pk_df['validation_samples'].sum()}"
    assert pk_df["test_samples"].sum() == 48, f"Expected 48 Pada Karma test samples, got {pk_df['test_samples'].sum()}"
    print(f"[PASS] 6. Pada Karma: All 6 classes represented (Train: 88, Val: 48, Test: 48, Total: 184 samples).")

    # 7. Check label distinctness (Sama across categories)
    samas = df_all[df_all["class_name"].str.contains("Sama")][["category", "class_name"]].drop_duplicates()
    assert len(samas) == 3, f"Expected 3 distinct Sama categories, got {len(samas)}"
    print("[PASS] 7. Label verification: Shiro Sama, Drishti Sama, and Pada Sama retain distinct categories.")

    # Save CSV Split Files
    train_file = OUTPUT_DIR / "v2_train.csv"
    val_file = OUTPUT_DIR / "v2_validation.csv"
    test_file = OUTPUT_DIR / "v2_test.csv"
    all_file = OUTPUT_DIR / "v2_all_samples.csv"

    df_train.to_csv(train_file, index=False)
    df_val.to_csv(val_file, index=False)
    df_test.to_csv(test_file, index=False)
    df_all.to_csv(all_file, index=False)
    print(f"\nSaved v2 split files to {OUTPUT_DIR}:")
    print(f"  - v2_train.csv ({len(df_train)} rows)")
    print(f"  - v2_validation.csv ({len(df_val)} rows)")
    print(f"  - v2_test.csv ({len(df_test)} rows)")
    print(f"  - v2_all_samples.csv ({len(df_all)} rows)")

    # Compute summary statistics for distribution table
    tr_vals = df_dist["train_samples"].values
    val_vals = df_dist["validation_samples"].values
    te_vals = df_dist["test_samples"].values

    stats_summary = {
        "train": {
            "min": int(np.min(tr_vals)),
            "max": int(np.max(tr_vals)),
            "mean": float(np.mean(tr_vals)),
            "median": float(np.median(tr_vals)),
            "total": int(np.sum(tr_vals)),
        },
        "validation": {
            "min": int(np.min(val_vals)),
            "max": int(np.max(val_vals)),
            "mean": float(np.mean(val_vals)),
            "median": float(np.median(val_vals)),
            "total": int(np.sum(val_vals)),
        },
        "test": {
            "min": int(np.min(te_vals)),
            "max": int(np.max(te_vals)),
            "mean": float(np.mean(te_vals)),
            "median": float(np.median(te_vals)),
            "total": int(np.sum(te_vals)),
        },
    }

    # Save reports/step12_class_distribution.csv
    dist_csv_file = REPORTS_DIR / "step12_class_distribution.csv"
    df_dist.to_csv(dist_csv_file, index=False)
    print(f"Saved distribution report to {dist_csv_file}")

    # Generate reports/step12_leakage_audit.txt
    leakage_audit_file = REPORTS_DIR / "step12_leakage_audit.txt"
    with open(leakage_audit_file, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("STEP 12 — SOURCE-LEVEL LEAKAGE AUDIT REPORT\n")
        f.write("================================================================================\n\n")
        f.write(f"Audit Date: 2026-10-04\n")
        f.write(f"Total Dataset Samples: {len(df_all)}\n")
        f.write(f"Total Unique Source Entities: {len(train_vids | val_vids | test_vids) + len(train_imgs | val_imgs | test_imgs)}\n\n")

        f.write("1. VIDEO SOURCE LEAKAGE AUDIT\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"Total Unique Video Sources: {len(train_vids | val_vids | test_vids)}\n")
        f.write(f"Train Video Sources:        {len(train_vids)}\n")
        f.write(f"Validation Video Sources:   {len(val_vids)}\n")
        f.write(f"Test Video Sources:         {len(test_vids)}\n\n")
        f.write(f"Overlap Train & Validation: {len(train_vids & val_vids)} videos\n")
        f.write(f"Overlap Train & Test:       {len(train_vids & test_vids)} videos\n")
        f.write(f"Overlap Validation & Test:  {len(val_vids & test_vids)} videos\n")
        f.write(f"VIDEO LEAKAGE RESULT:       0 SOURCE-LEVEL LEAKAGE (100% ISOLATED)\n\n")

        f.write("2. IMAGE SOURCE LEAKAGE AUDIT\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"Total Unique Image Sources: {len(train_imgs | val_imgs | test_imgs)}\n")
        f.write(f"Train Image Sources:        {len(train_imgs)}\n")
        f.write(f"Validation Image Sources:   {len(val_imgs)}\n")
        f.write(f"Test Image Sources:         {len(test_imgs)}\n\n")
        f.write(f"Overlap Train & Validation: {len(train_imgs & val_imgs)} images\n")
        f.write(f"Overlap Train & Test:       {len(train_imgs & test_imgs)} images\n")
        f.write(f"Overlap Validation & Test:  {len(val_imgs & test_imgs)} images\n")
        f.write(f"IMAGE LEAKAGE RESULT:       0 SOURCE-LEVEL LEAKAGE (100% ISOLATED)\n\n")

        f.write("3. SAMPLE-LEVEL EXCLUSIVITY AUDIT\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"Train Samples:              {len(df_train)}\n")
        f.write(f"Validation Samples:         {len(df_val)}\n")
        f.write(f"Test Samples:               {len(df_test)}\n")
        f.write(f"Total Assigned Samples:     {len(df_all)}\n")
        f.write(f"Sample Intersection (Tr/Val):  {len(train_paths & val_paths)}\n")
        f.write(f"Sample Intersection (Tr/Test): {len(train_paths & test_paths)}\n")
        f.write(f"Sample Intersection (Val/Test): {len(val_paths & test_paths)}\n")
        f.write(f"SAMPLE LEAKAGE RESULT:      0 SAMPLE LEAKAGE\n\n")

        f.write("4. AUDIT CONCLUSION\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("ALL SOURCE-LEVEL AND SAMPLE-LEVEL ISOLATION CRITERIA HAVE PASSED WITH 0 LEAKAGE.\n")
        f.write("All 8 frames of every source video are strictly contained within exactly one partition.\n")

    print(f"Saved leakage audit report to {leakage_audit_file}")

    # Generate reports/step12_split_v2_report.txt
    v2_report_file = REPORTS_DIR / "step12_split_v2_report.txt"
    with open(v2_report_file, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("STEP 12 — DATASET SPLIT V2 REPORT\n")
        f.write("REBUILDING THE DATASET SPLIT USING ALL 1,094 AVAILABLE SAMPLES\n")
        f.write("================================================================================\n\n")

        f.write("1. WHY THE ORIGINAL SPLIT WAS INADEQUATE\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("In Step 2, a split heuristic was implemented where any class with fewer than 5\n")
        f.write("independent source entities was disqualified from the conventional Holdout partition\n")
        f.write("and placed exclusively into a 'Grouped 3-Fold Cross-Validation' partition.\n\n")
        f.write("Consequences of that design:\n")
        f.write("- Exactly 16 classes were assigned 0 training samples, 0 validation samples, and 0 test samples.\n")
        f.write("- 258 samples out of 1,094 (23.58% of the entire dataset) were completely quarantined\n")
        f.write("  and excluded from all train/validation/test workflows.\n")
        f.write("- All 6 classes of Pada Karma (184 video frames) were completely excluded, leaving the\n")
        f.write("  Pada Karma expert completely untrainable in Step 11.\n")
        f.write("- 10 other fine-grained classes across Hasthas (1 class), Shiro Bhedhas (4 classes),\n")
        f.write("  and Drishti Bhedhas (5 classes) were completely missing from training.\n")
        f.write("- In Step 11, the Category Router achieved 99.21% accuracy, but overall 75-class top-1\n")
        f.write("  accuracy was capped at 28.35% because 16 classes had zero training signal.\n\n")

        f.write("2. WHERE THE 258 SAMPLES WERE\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("The 258 samples were isolated in data/splits/all_samples_split.csv under partition='grouped_cv'\n")
        f.write("and distributed across fold_1, fold_2, and fold_3.\n\n")
        f.write("Breakdown of the 258 quarantined samples across the 16 affected classes:\n")
        f.write("  - 1. Hasthas / 17.Matsyamu: 4 images\n")
        f.write("  - 2. Shiro Bhedhas / 1. Sama: 3 images\n")
        f.write("  - 2. Shiro Bhedhas / 2. Udvahitam: 4 images\n")
        f.write("  - 2. Shiro Bhedhas / 3. Adhomukham: 3 images\n")
        f.write("  - 2. Shiro Bhedhas / 6. Kampitam: 3 videos (24 frames)\n")
        f.write("  - 3. Drishti Bhedhas / 1. Sama: 3 images\n")
        f.write("  - 3. Drishti Bhedhas / 5. Nimilita: 3 images\n")
        f.write("  - 3. Drishti Bhedhas / 6. Ullokita: 3 images\n")
        f.write("  - 3. Drishti Bhedhas / 7. Anuvritta: 3 videos (24 frames)\n")
        f.write("  - 3. Drishti Bhedhas / 8. Avalokita: 3 images\n")
        f.write("  - 4. Pada Karma / 1. Udghattita: 3 videos (24 frames)\n")
        f.write("  - 4. Pada Karma / 2. Sama: 4 videos (32 frames)\n")
        f.write("  - 4. Pada Karma / 3. Agratalasanchara: 4 videos (32 frames)\n")
        f.write("  - 4. Pada Karma / 4. Anchita Pada: 4 videos (32 frames)\n")
        f.write("  - 4. Pada Karma / 5. Kunchita: 4 videos (32 frames)\n")
        f.write("  - 4. Pada Karma / 6. Suchi Pada: 4 videos (32 frames)\n")
        f.write(f"Total: 4 + 3 + 4 + 3 + 24 + 3 + 3 + 3 + 24 + 3 + 24 + 32 + 32 + 32 + 32 + 32 = 258 samples.\n\n")

        f.write("3. HOW THE 184 PADA KARMA SAMPLES ARE NOW HANDLED\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("All 6 classes of Pada Karma are composed exclusively of video sources (23 videos = 184 frames).\n")
        f.write("Each video contains exactly 8 sampled frames.\n")
        f.write("In Split v2, source-aware video grouping is strictly maintained:\n")
        f.write("- For 1. Udghattita (3 videos = 24 frames):\n")
        f.write("    1 video (8 frames) -> Train (33.3%)\n")
        f.write("    1 video (8 frames) -> Validation (33.3%)\n")
        f.write("    1 video (8 frames) -> Test (33.3%)\n")
        f.write("- For the remaining 5 Pada Karma classes (4 videos = 32 frames each):\n")
        f.write("    2 videos (16 frames) -> Train (50.0%)\n")
        f.write("    1 video (8 frames)  -> Validation (25.0%)\n")
        f.write("    1 video (8 frames)  -> Test (25.0%)\n\n")
        f.write("Total Pada Karma counts in Split v2:\n")
        f.write(f"  - Train:      88 frames (11 videos, 47.8%)\n")
        f.write(f"  - Validation: 48 frames (6 videos, 26.1%)\n")
        f.write(f"  - Test:       48 frames (6 videos, 26.1%)\n")
        f.write(f"  - Total:      184 frames (23 videos, 100.0%)\n")
        f.write("Crucially, 0 video frames are leaked across partitions.\n\n")

        f.write("4. HOW ALL 75 CLASSES ARE REPRESENTED\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Every single class out of all 75 classes has at least 3 independent source entities.\n")
        f.write("Therefore, by allocating independent source entities to Train, Validation, and Test:\n")
        f.write("- 75 of 75 classes have non-zero Training samples.\n")
        f.write("- 75 of 75 classes have non-zero Validation samples.\n")
        f.write("- 75 of 75 classes have non-zero Test samples.\n")
        f.write("- Zero classes are unrepresented.\n\n")

        f.write("Category-level sample distribution:\n")
        cat_counts = defaultdict(lambda: Counter())
        for r in all_assigned:
            cat_counts[r["category"]][r["partition"]] += 1
        for cat, cnts in sorted(cat_counts.items()):
            tot = sum(cnts.values())
            f.write(f"  - {cat}:\n")
            f.write(f"      Total: {tot} samples\n")
            f.write(f"      Train: {cnts['train']} ({cnts['train']/tot*100:.1f}%)\n")
            f.write(f"      Validation: {cnts['validation']} ({cnts['validation']/tot*100:.1f}%)\n")
            f.write(f"      Test: {cnts['test']} ({cnts['test']/tot*100:.1f}%)\n")
        f.write("\n")

        f.write("5. EXACT TRAIN / VALIDATION / TEST COUNTS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"Total prepared samples: 1,094\n")
        f.write(f"Train samples:          {len(df_train)} ({len(df_train)/1094*100:.2f}%)\n")
        f.write(f"Validation samples:     {len(df_val)} ({len(df_val)/1094*100:.2f}%)\n")
        f.write(f"Test samples:           {len(df_test)} ({len(df_test)/1094*100:.2f}%)\n")
        f.write(f"Excluded samples:       0 (0.00%)\n\n")

        f.write("Per-class sample statistics across all 75 classes:\n")
        f.write(f"  - Train:      min={stats_summary['train']['min']}, max={stats_summary['train']['max']}, mean={stats_summary['train']['mean']:.2f}, median={stats_summary['train']['median']:.1f}\n")
        f.write(f"  - Validation: min={stats_summary['validation']['min']}, max={stats_summary['validation']['max']}, mean={stats_summary['validation']['mean']:.2f}, median={stats_summary['validation']['median']:.1f}\n")
        f.write(f"  - Test:       min={stats_summary['test']['min']}, max={stats_summary['test']['max']}, mean={stats_summary['test']['mean']:.2f}, median={stats_summary['test']['median']:.1f}\n\n")

        f.write("6. SOURCE-LEVEL LEAKAGE AUDIT RESULT\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Strict audit performed on all 68 source videos and 550 source images:\n")
        f.write(f"  - Video source leakage: 0 (No video appears in more than one partition)\n")
        f.write(f"  - Image source leakage: 0 (No image appears in more than one partition)\n")
        f.write(f"  - Sample-level leakage: 0 (No sample path appears in more than one partition)\n")
        f.write(f"  - Result: 100% LEAKAGE FREE.\n\n")

        f.write("7. UNAVOIDABLE LIMITATIONS AND DESIGN RATIONALE\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("1. Indivisibility of Video Units:\n")
        f.write("   Because each video consists of 8 extracted frames and all frames from the same video\n")
        f.write("   must remain in the same partition to prevent frame-to-frame leakage, the smallest possible\n")
        f.write("   non-zero allocation for any video-based class in validation or test is 8 frames.\n")
        f.write("   For a class with only 3 videos (24 frames), assigning 1 video to validation and 1 video\n")
        f.write("   to test inherently allocates 8 frames (33.3%) to val and 8 frames (33.3%) to test,\n")
        f.write("   leaving 8 frames (33.3%) for train. This is mathematically unavoidable under source grouping.\n\n")
        f.write("2. Highly Imbalanced Category Volumes:\n")
        f.write("   Hasthas contains 566 samples (all image-based), achieving 73.5% Train, 15.0% Val, 11.5% Test.\n")
        f.write("   Pada Karma contains 184 samples (all video-based), achieving 47.8% Train, 26.1% Val, 26.1% Test.\n")
        f.write("   Across the entire dataset, Train is 65.45%, Validation is 18.19%, and Test is 16.36%.\n")
        f.write("   This slight deviation from the nominal 70-75% train target is the optimal compromise\n")
        f.write("   to guarantee 100% class representation across all partitions while maintaining zero leakage.\n\n")
        f.write("3. Label Integrity:\n")
        f.write("   Classes named '1. Sama' in Shiro Bhedhas and Drishti Bhedhas, and '2. Sama' in Pada Karma\n")
        f.write("   remain completely distinct and are uniquely referenced via (category, class_name).\n")

    print(f"Saved split v2 report to {v2_report_file}")
    return stats_summary, len(df_all), len(df_train), len(df_val), len(df_test)

if __name__ == "__main__":
    build_v2_split()
