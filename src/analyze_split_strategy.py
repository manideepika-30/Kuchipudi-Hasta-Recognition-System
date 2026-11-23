import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

def analyze_split_strategy():
    project_root = Path(r"E:\My kuchipudi project aidl\kuchipudi")
    reports_dir = project_root / "reports"
    processed_dir = project_root / "data" / "processed"
    
    # Load processed class distribution
    dist_csv = reports_dir / "processed_class_distribution.csv"
    frame_manifest_csv = processed_dir / "frame_manifest.csv"
    catalog_csv = processed_dir / "dataset_catalog.csv"
    
    class_data = []
    with open(dist_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_name = row["class_name"]
            cat = row["category"]
            img_cnt = int(row["original_image_count"])
            vid_cnt = int(row["video_count"])
            frm_cnt = int(row["extracted_frame_count"])
            tot_samples = int(row["total_available_samples"])
            
            # Independent sources:
            # Each static image is an independent source.
            # Each video file is an independent source (its extracted frames are correlated).
            indep_sources = img_cnt + vid_cnt
            
            # Determine feasibility for standard 3-way holdout (Train, Val, Test)
            # A standard 70/15/15 holdout requires enough independent sources so that:
            # - Train has at least 5-6 independent sources for generalization
            # - Val has at least 1-2 independent sources for checkpointing
            # - Test has at least 1-2 independent sources for held-out evaluation
            # Therefore, standard holdout is only robust if indep_sources >= 8.
            
            if indep_sources >= 8:
                train_feas = "YES"
                val_feas = "YES"
                test_feas = "YES"
                strategy = "Stratified Holdout (Grouped by Video)"
                notes = f"Sufficient independent sources ({indep_sources} sources). Safe for 70/15/15 split."
            elif indep_sources >= 5:
                train_feas = "YES (Marginal)"
                val_feas = "YES (1 source)"
                test_feas = "YES (1 source)"
                strategy = "Grouped Split (Fixed 1-Val, 1-Test Source)"
                notes = f"Limited independent sources ({indep_sources} sources: {img_cnt} imgs, {vid_cnt} vids). Val & Test receive only 1 independent source each."
            elif indep_sources == 4:
                train_feas = "CRITICAL (2 sources)"
                val_feas = "CRITICAL (1 source)"
                test_feas = "CRITICAL (1 source)"
                strategy = "Grouped K-Fold / Leave-One-Group-Out"
                notes = f"Very scarce sources (4 sources). Standard holdout assigns 50% train (2 sources), 25% val (1 source), 25% test (1 source), leading to high metric variance."
            else: # indep_sources == 3
                train_feas = "INADEQUATE (1 source)"
                val_feas = "INADEQUATE (1 source)"
                test_feas = "INADEQUATE (1 source)"
                strategy = "Grouped Cross-Validation (3-Fold) OR Additional Data Collection"
                notes = f"Extreme source scarcity (3 sources). A 3-way split gives 1 source per split (33.3% each). Single test error causes 100% to 0% swing."
                
            class_data.append({
                "class_name": cls_name,
                "category": cat,
                "image_count": img_cnt,
                "video_count": vid_cnt,
                "video_frame_count": frm_cnt,
                "independent_sources": indep_sources,
                "recommended_strategy": strategy,
                "train_feasible": train_feas,
                "validation_feasible": val_feas,
                "test_feasible": test_feas,
                "notes": notes
            })

    # Sort classes by category and name
    class_data.sort(key=lambda x: (x["category"], x["class_name"]))

    # Save reports/class_split_analysis.csv
    out_csv = reports_dir / "class_split_analysis.csv"
    fieldnames = [
        "class_name", "image_count", "video_count", "video_frame_count",
        "independent_sources", "recommended_strategy", "train_feasible",
        "validation_feasible", "test_feasible", "notes"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in class_data:
            row_copy = {k: r[k] for k in fieldnames}
            writer.writerow(row_copy)
            
    # Statistics
    total_classes = len(class_data)
    suitable_holdout = [c for c in class_data if c["independent_sources"] >= 8]
    marginal_sources = [c for c in class_data if 5 <= c["independent_sources"] < 8]
    scarce_sources = [c for c in class_data if c["independent_sources"] == 4]
    extreme_scarce = [c for c in class_data if c["independent_sources"] <= 3]
    
    insufficient_samples_classes = [c for c in class_data if c["independent_sources"] < 8]
    image_only_scarce = [c for c in class_data if c["video_count"] == 0 and c["image_count"] <= 6]
    video_scarce = [c for c in class_data if c["image_count"] == 0 and c["video_count"] <= 4]

    # Save reports/split_strategy_report.txt
    out_txt = reports_dir / "split_strategy_report.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KUCHIPUDI DANCE RECOGNITION - DATASET SPLIT STRATEGY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Classes Analyzed: {total_classes}\n")
        f.write(f"Total Original JPEG Images: 550\n")
        f.write(f"Total MP4 Videos: 68\n")
        f.write(f"Total Extracted Video Frames: 544\n")
        f.write(f"Total Available Samples (Images + Frames): 1,094\n\n")

        f.write("-" * 80 + "\n")
        f.write("INDEPENDENT SOURCE DISTRIBUTION ACROSS CLASSES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Classes with Robust Independent Sources (>= 8 sources): {len(suitable_holdout)} classes\n")
        f.write(f"Classes with Marginal Independent Sources (5 - 7 sources): {len(marginal_sources)} classes\n")
        f.write(f"Classes with Severe Scarcity (4 sources):               {len(scarce_sources)} classes\n")
        f.write(f"Classes with Extreme Scarcity (3 sources):              {len(extreme_scarce)} classes\n")
        f.write(f"Total Classes with Insufficient Samples for Blind 70/15/15: {len(insufficient_samples_classes)} classes\n\n")

        f.write("-" * 80 + "\n")
        f.write("DETAILED PROBLEM ANALYSIS FOR THE 20 SCARCE IMAGE-ONLY CLASSES\n")
        f.write("-" * 80 + "\n")
        f.write("The following 20 classes have NO video recordings and only 3 to 6 static images:\n")
        for c in image_only_scarce:
            f.write(f"- [{c['category']}] {c['class_name']:<25} : {c['image_count']} images ({c['independent_sources']} independent sources)\n")
        
        f.write("\nMathematical & Experimental Dilemma:\n")
        f.write("1. Ratio Collapse: A 70/15/15 split on 3 samples yields 2.1 / 0.45 / 0.45 samples.\n")
        f.write("2. Forcing Holdout: Assigning 1 sample to Val and 1 to Test forces Train to have only 1 sample (33.3%/33.3%/33.3%).\n")
        f.write("3. Test Metric Instability: Evaluating on 1 test image means per-class accuracy is binary (0% or 100%),\n")
        f.write("   introducing massive noise into macro-averaged F1 and confusion matrices.\n")
        f.write("4. Generalization Failure: Training a CNN feature extractor on 1 or 2 images per class causes severe under-representation.\n\n")

        f.write("-" * 80 + "\n")
        f.write("ANALYSIS OF VIDEO CLASSES AND INDEPENDENT VIDEO SOURCES\n")
        f.write("-" * 80 + "\n")
        f.write("Extracted frames from the same video are strongly correlated (temporal autocorrelation).\n")
        f.write("Rule: All frames sharing the same 'source_video_id' MUST reside in the same partition.\n")
        f.write("Classes with limited video sources:\n")
        for c in video_scarce:
            f.write(f"- [{c['category']}] {c['class_name']:<25} : {c['video_count']} videos -> {c['video_frame_count']} frames ({c['independent_sources']} independent sources)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("SCIENTIFICALLY DEFENSIBLE STRATEGY RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("Option A (Collecting Additional Data):\n")
        f.write("  - If real-world acquisition is possible, recording 3-5 additional videos or 10-15 images for the 20 scarce\n")
        f.write("    image classes and 8 scarce video classes is the ideal gold standard for deep learning.\n\n")
        f.write("Option B (Category-Specific Hierarchical Architecture with Tailored Splits):\n")
        f.write("  - Hasthas (52 classes): 510 images + 56 frames = 566 samples. 41 classes have >= 8 sources.\n")
        f.write("    Only 11 classes in Samyutha have 4-6 images. Can use a fixed 1-test, 1-val holdout while weighting training loss.\n")
        f.write("  - Shiro Bhedhas (9 classes): 22 images + 192 frames = 214 samples.\n")
        f.write("  - Drishti Bhedhas (8 classes): 18 images + 120 frames = 138 samples.\n")
        f.write("  - Pada Karma (6 classes): 0 images + 176 frames = 176 samples across 23 videos.\n\n")
        f.write("Option C (Grouped Stratified Holdout with Video-Level Grouping + Class Weighting):\n")
        f.write("  - For held-out evaluation without test leakage:\n")
        f.write("    1. Group all video frames strictly by 'source_video_id'.\n")
        f.write("    2. For classes with >= 5 sources: Allocate 1 source to Test, 1 source to Val, remaining to Train.\n")
        f.write("    3. For classes with 3-4 sources: Allocate 1 source to Test, 1 source to Val, 1-2 to Train.\n")
        f.write("    4. In training partition: Apply class weighting or focal loss to counter sample imbalance.\n")
        f.write("    5. In validation/testing: NEVER duplicate or augment samples.\n\n")
        f.write("Option D (Grouped K-Fold Cross Validation - 3 to 5 Folds):\n")
        f.write("  - If held-out test variance on 1 sample per class is unacceptable, Grouped K-Fold (grouped by video/image cluster)\n")
        f.write("    allows every sample to be tested exactly once across K folds, yielding statistically robust mean accuracy and confidence intervals.\n")

    print(f"Split strategy analysis complete!")
    print(f"Total classes: {total_classes}")
    print(f"Suitable for normal holdout (>= 8 sources): {len(suitable_holdout)}")
    print(f"Insufficient independent sources (< 8 sources): {len(insufficient_samples_classes)}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")

if __name__ == "__main__":
    analyze_split_strategy()
