"""
STEP 29A — Personalized Kuchipudi Dataset Verification Script
=============================================================
Performs comprehensive audits on the collected personalized dataset:
  1. Metadata schema and integrity
  2. Per-class sample counts vs target targets
  3. Missing classes identification
  4. File integrity & corruption checks (cv2.imread / np.load)
  5. Landmark feature presence and dimension checks
  6. Duplicate file paths and sample ID checks
  7. Label validity against canonical project hierarchy
  8. Sequence length validation (temporal movement categories)

Does NOT perform any model training or dataset modification.
"""

import os
import sys
import csv
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical_model import build_hierarchical_mapping

PERSONALIZED_DIR = PROJECT_ROOT / "data" / "personalized"
METADATA_FILE = PERSONALIZED_DIR / "metadata" / "personalized_samples.csv"

TARGETS = {
    "1. Hasthas": 20,
    "2. Shiro Bhedhas (HEAD MOVEMENTS)": 15,
    "3. Drishti Bhedhas(EYE MOVEMENTS)": 20,
    "4. Pada Karma(Foot movements)": 15
}

EXPECTED_DIMS = {
    "1. Hasthas": 175,
    "2. Shiro Bhedhas (HEAD MOVEMENTS)": 32,
    "3. Drishti Bhedhas(EYE MOVEMENTS)": 44,
    "4. Pada Karma(Foot movements)": 40
}


def audit_personalized_dataset() -> Dict[str, Any]:
    print("=" * 80)
    print("AUDITING PERSONALIZED KUCHIPUDI DATASET (STEP 29A)")
    print("=" * 80)

    mapping = build_hierarchical_mapping(save_json=False)
    canonical_classes = {}
    for cat_name, info in mapping["experts"].items():
        canonical_classes[cat_name] = {item["class_name"]: item["local_class_id"] for item in info["classes"]}

    if not METADATA_FILE.exists():
        print(f"Warning: Metadata file does not exist yet: {METADATA_FILE}")
        return {
            "metadata_exists": False,
            "total_samples": 0,
            "corrupted_files": 0,
            "missing_landmarks": 0,
            "duplicates": 0,
            "invalid_labels": 0
        }

    # 1. Read metadata
    records = []
    with open(METADATA_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"Total entries in metadata: {len(records)}")

    # 2. Audits
    seen_ids: Set[str] = set()
    seen_paths: Set[str] = set()
    duplicate_ids = 0
    duplicate_paths = 0
    missing_files = 0
    corrupted_files = 0
    missing_landmarks = 0
    invalid_labels = 0
    short_sequences = 0

    counts_by_cat_class: Dict[Tuple[str, str], int] = {}

    for row in records:
        sid = row.get("sample_id", "")
        cat = row.get("category", "")
        cname = row.get("class_name", "")
        stype = row.get("source_type", "")
        fpath_str = row.get("file_path", "")

        # Duplicate ID check
        if sid in seen_ids:
            duplicate_ids += 1
        seen_ids.add(sid)

        # Duplicate path check
        if fpath_str in seen_paths:
            duplicate_paths += 1
        seen_paths.add(fpath_str)

        # Label validity check
        if cat not in canonical_classes:
            invalid_labels += 1
        elif cname not in canonical_classes[cat]:
            invalid_labels += 1
        else:
            key = (cat, cname)
            counts_by_cat_class[key] = counts_by_cat_class.get(key, 0) + 1

        # File existence & integrity check
        full_path = PROJECT_ROOT / fpath_str
        if not full_path.exists():
            missing_files += 1
            continue

        if stype == "image":
            img = cv2.imread(str(full_path))
            if img is None or img.size == 0:
                corrupted_files += 1
            # Check landmark file
            lm_path = full_path.parent / (full_path.stem + "_landmarks.npz")
            if not lm_path.exists():
                missing_landmarks += 1
            else:
                try:
                    d = np.load(str(lm_path))
                    vec = d.get("vector")
                    expected_d = EXPECTED_DIMS.get(cat, 0)
                    if vec is None or len(vec) != expected_d or not d.get("detected", False):
                        missing_landmarks += 1
                except Exception:
                    missing_landmarks += 1

        elif stype == "sequence":
            try:
                d = np.load(str(full_path))
                seq = d.get("sequence_features")
                if seq is None or len(seq) < 16:
                    short_sequences += 1
            except Exception:
                corrupted_files += 1

    # 3. Category & Class Coverage
    print("\n" + "-" * 80)
    print("CATEGORY & CLASS PROGRESS BREAKDOWN")
    print("-" * 80)

    missing_classes_report = []

    for cat_name, c_dict in canonical_classes.items():
        target_per_class = TARGETS.get(cat_name, 20)
        total_target_cat = len(c_dict) * target_per_class
        cat_collected = sum(counts_by_cat_class.get((cat_name, c), 0) for c in c_dict)
        pct = (cat_collected / total_target_cat * 100.0) if total_target_cat > 0 else 0.0

        print(f"\n[{cat_name}]")
        print(f"  Classes Coverage: {len(c_dict)} classes | Target: {total_target_cat} | Collected: {cat_collected} ({pct:.1f}%)")

        completed_classes = 0
        partial_classes = 0
        zero_classes = 0

        for cname in c_dict.keys():
            c_cnt = counts_by_cat_class.get((cat_name, cname), 0)
            if c_cnt >= target_per_class:
                completed_classes += 1
            elif c_cnt > 0:
                partial_classes += 1
            else:
                zero_classes += 1
                missing_classes_report.append((cat_name, cname))

        print(f"  Status: {completed_classes} Complete, {partial_classes} In-Progress, {zero_classes} Not Started")

    print("\n" + "=" * 80)
    print("DATASET INTEGRITY AUDIT SUMMARY")
    print("=" * 80)
    print(f"  Total Metadata Entries: {len(records)}")
    print(f"  Duplicate Sample IDs  : {duplicate_ids}")
    print(f"  Duplicate File Paths  : {duplicate_paths}")
    print(f"  Missing Files on Disk : {missing_files}")
    print(f"  Corrupted Files       : {corrupted_files}")
    print(f"  Missing Landmarks     : {missing_landmarks}")
    print(f"  Invalid / Rogue Labels: {invalid_labels}")
    print(f"  Short Sequences (<16) : {short_sequences}")
    print(f"  Missing Classes (0 smp: {len(missing_classes_report)}")
    print("=" * 80 + "\n")

    return {
        "metadata_exists": True,
        "total_records": len(records),
        "duplicate_ids": duplicate_ids,
        "duplicate_paths": duplicate_paths,
        "missing_files": missing_files,
        "corrupted_files": corrupted_files,
        "missing_landmarks": missing_landmarks,
        "invalid_labels": invalid_labels,
        "short_sequences": short_sequences,
        "missing_classes_count": len(missing_classes_report)
    }


if __name__ == "__main__":
    audit_personalized_dataset()
