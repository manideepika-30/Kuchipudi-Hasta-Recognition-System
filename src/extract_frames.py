import os
import sys
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np
from PIL import Image

# Import configuration
try:
    from config import (
        PROJECT_ROOT,
        DATASET_ROOT,
        PROCESSED_DATA_DIR,
        VIDEO_FRAMES_DIR,
        REPORTS_DIR,
        VIDEO_INVENTORY_CSV,
        FRAME_MANIFEST_CSV,
        DATASET_CATALOG_CSV,
        VIDEO_EXTRACTION_REPORT_TXT,
        PROCESSED_CLASS_DISTRIBUTION_CSV,
        FRAME_SAMPLING_MODE,
        TARGET_FRAMES_PER_VIDEO,
        SAMPLE_FPS,
        MIN_FRAMES_PER_VIDEO,
        MAX_FRAMES_PER_VIDEO,
        TEMPORAL_MARGIN_PERCENT,
        FRAME_IMAGE_EXTENSION,
        JPEG_QUALITY,
        RANDOM_SEED
    )
except ImportError:
    from src.config import (
        PROJECT_ROOT,
        DATASET_ROOT,
        PROCESSED_DATA_DIR,
        VIDEO_FRAMES_DIR,
        REPORTS_DIR,
        VIDEO_INVENTORY_CSV,
        FRAME_MANIFEST_CSV,
        DATASET_CATALOG_CSV,
        VIDEO_EXTRACTION_REPORT_TXT,
        PROCESSED_CLASS_DISTRIBUTION_CSV,
        FRAME_SAMPLING_MODE,
        TARGET_FRAMES_PER_VIDEO,
        SAMPLE_FPS,
        MIN_FRAMES_PER_VIDEO,
        MAX_FRAMES_PER_VIDEO,
        TEMPORAL_MARGIN_PERCENT,
        FRAME_IMAGE_EXTENSION,
        JPEG_QUALITY,
        RANDOM_SEED
    )

def sanitize_dirname(name: str) -> str:
    """Sanitize directory names if needed, though exact names are preferred."""
    return name.replace('/', '_').replace('\\', '_')

def get_sampling_indices(total_frames: int, fps: float, mode: str, target_count: int, sample_fps: float) -> list:
    """Calculate temporal sampling frame indices with margin protection."""
    if total_frames <= 0:
        return []
        
    start_margin = int(total_frames * TEMPORAL_MARGIN_PERCENT)
    end_margin = total_frames - int(total_frames * TEMPORAL_MARGIN_PERCENT)
    if end_margin <= start_margin:
        start_margin = 0
        end_margin = total_frames

    usable_frames = end_margin - start_margin
    if usable_frames <= 0:
        return [0]

    if mode == "sample_fps" and fps > 0:
        step = max(1, int(fps / sample_fps))
        indices = list(range(start_margin, end_margin, step))
    else:  # default "uniform_target"
        num_samples = min(target_count, usable_frames)
        num_samples = max(MIN_FRAMES_PER_VIDEO, num_samples)
        num_samples = min(MAX_FRAMES_PER_VIDEO, num_samples)
        indices = np.linspace(start_margin, end_margin - 1, num=num_samples, dtype=int).tolist()

    # Remove duplicates and sort
    indices = sorted(list(set(indices)))
    return indices

def run_pipeline():
    print("=" * 80)
    print("KUCHIPUDI VIDEO INVENTORY & CONTROLLED FRAME EXTRACTION PIPELINE")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. Environment Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Environment Verification...")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version   : {sys.version.split()[0]}")
    print(f"  OpenCV version   : {cv2.__version__}")
    print(f"  Pillow version   : {Image.__version__}")
    print(f"  NumPy version    : {np.__version__}")
    
    # -------------------------------------------------------------------------
    # 2. Discover Classes and Filesystem Layout
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Scanning original dataset at:", DATASET_ROOT)
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Original dataset directory not found at: {DATASET_ROOT}")

    leaf_classes = []
    all_videos_found = []
    
    for p in sorted(DATASET_ROOT.rglob('*')):
        if p.is_dir() and not any(c.is_dir() for c in p.iterdir()):
            rel = p.relative_to(DATASET_ROOT)
            parts = rel.parts
            if parts[0] == "1. Hasthas":
                cat = parts[0]
                subcat = parts[1]
                cls = parts[2]
            else:
                cat = parts[0]
                subcat = None
                cls = parts[1]
                
            jpegs = sorted(list({x.resolve() for x in p.iterdir() if x.suffix.lower() in ['.jpeg', '.jpg', '.png']}))
            mp4s = sorted(list({x.resolve() for x in p.iterdir() if x.suffix.lower() == '.mp4'}))
            
            leaf_info = {
                "dir_path": p,
                "rel_path": rel,
                "category": cat,
                "subcategory": subcat,
                "class_name": cls,
                "images": jpegs,
                "videos": mp4s
            }
            leaf_classes.append(leaf_info)
            
            for v_idx, vid_file in enumerate(mp4s):
                vid_id = f"vid_{len(all_videos_found)+1:03d}"
                all_videos_found.append({
                    "video_id": vid_id,
                    "video_path": vid_file,
                    "category": cat,
                    "subcategory": subcat,
                    "class_name": cls,
                    "leaf_info": leaf_info
                })

    print(f"  Total leaf classes discovered : {len(leaf_classes)}")
    print(f"  Total video files discovered  : {len(all_videos_found)}")

    # -------------------------------------------------------------------------
    # 3. Video Inventory Inspection
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Inspecting video inventory & generating video_inventory.csv...")
    inventory_rows = []
    valid_videos = []
    failed_videos = []

    for item in all_videos_found:
        v_path = item["video_path"]
        cap = cv2.VideoCapture(str(v_path))
        is_opened = cap.isOpened()
        
        duration = 0.0
        fps = 0.0
        w = 0
        h = 0
        frame_count = 0
        decode_status = "FAILED"

        if is_opened:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = float(frame_count / fps if fps > 0 else 0.0)
            
            # Check if first and middle frame decode properly
            ret1, f1 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
            ret2, f2 = cap.read()
            
            if ret1 and ret2 and f1 is not None and f2 is not None:
                decode_status = "SUCCESS"
            else:
                decode_status = "DECODE_ERROR"
            cap.release()

        item["fps"] = round(fps, 2)
        item["frame_count"] = frame_count
        item["width"] = w
        item["height"] = h
        item["duration"] = round(duration, 2)
        item["decode_status"] = decode_status

        inventory_rows.append({
            "category": item["category"],
            "class_name": item["class_name"],
            "video_filename": v_path.name,
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 2),
            "width": w,
            "height": h,
            "frame_count": frame_count,
            "decode_status": decode_status
        })

        if decode_status == "SUCCESS":
            valid_videos.append(item)
        else:
            failed_videos.append(item)

    # Write video_inventory.csv
    with open(VIDEO_INVENTORY_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "category", "class_name", "video_filename", 
            "duration_seconds", "fps", "width", "height", 
            "frame_count", "decode_status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory_rows)

    print(f"  Successfully decoded videos : {len(valid_videos)}")
    print(f"  Failed videos               : {len(failed_videos)}")
    print(f"  Saved inventory to          : {VIDEO_INVENTORY_CSV}")

    # -------------------------------------------------------------------------
    # 4. Controlled Frame Extraction
    # -------------------------------------------------------------------------
    print(f"\n[STEP 4] Extracting frames (Mode: {FRAME_SAMPLING_MODE}, Target: {TARGET_FRAMES_PER_VIDEO} frames/video)...")
    
    frame_manifest_rows = []
    total_extracted_frames = 0
    extraction_failures = []
    class_extracted_counts = Counter()
    video_extracted_counts = Counter()

    for item in valid_videos:
        v_path = item["video_path"]
        v_id = item["video_id"]
        cat = item["category"]
        subcat = item["subcategory"]
        cls = item["class_name"]

        # Build output directory
        if subcat:
            out_class_dir = VIDEO_FRAMES_DIR / cat / subcat / cls
        else:
            out_class_dir = VIDEO_FRAMES_DIR / cat / cls
        out_class_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(v_path))
        fps = item["fps"]
        total_frames = item["frame_count"]

        indices = get_sampling_indices(
            total_frames=total_frames,
            fps=fps,
            mode=FRAME_SAMPLING_MODE,
            target_count=TARGET_FRAMES_PER_VIDEO,
            sample_fps=SAMPLE_FPS
        )

        extracted_this_video = 0
        for sample_seq_idx, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                extraction_failures.append({
                    "video": v_path.name,
                    "frame_idx": frame_idx,
                    "reason": "Failed to read frame"
                })
                continue

            # Convert BGR to RGB for correct colors
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, c = frame_rgb.shape
            timestamp_sec = round(frame_idx / fps, 3) if fps > 0 else 0.0

            # Name format: <video_id>_idx<frame_idx>_seq<seq_idx>.jpeg
            out_filename = f"{v_id}_idx{frame_idx:05d}_seq{sample_seq_idx:02d}{FRAME_IMAGE_EXTENSION}"
            out_filepath = out_class_dir / out_filename

            # Save via Pillow with defined JPEG quality
            pil_img = Image.fromarray(frame_rgb)
            pil_img.save(out_filepath, "JPEG", quality=JPEG_QUALITY)

            extracted_this_video += 1
            total_extracted_frames += 1
            class_extracted_counts[cls] += 1
            video_extracted_counts[v_path.name] += 1

            # Relative path to project root for portability
            rel_frame_path = out_filepath.relative_to(PROJECT_ROOT)

            frame_manifest_rows.append({
                "frame_path": str(rel_frame_path).replace("\\", "/"),
                "category": cat,
                "class_name": cls,
                "source_video": v_path.name,
                "source_video_id": v_id,
                "frame_index": frame_idx,
                "timestamp_seconds": timestamp_sec,
                "width": w,
                "height": h
            })

        cap.release()

    # Save frame_manifest.csv
    with open(FRAME_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_path", "category", "class_name", "source_video", 
            "source_video_id", "frame_index", "timestamp_seconds", "width", "height"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_manifest_rows)

    print(f"  Total frames extracted: {total_extracted_frames}")
    print(f"  Frame extraction failures: {len(extraction_failures)}")
    print(f"  Saved frame manifest to: {FRAME_MANIFEST_CSV}")

    # -------------------------------------------------------------------------
    # 5. Quality Check on Extracted Frames
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Performing verification and quality checks on extracted frames...")
    quality_failures = []
    
    for row in frame_manifest_rows:
        fpath = PROJECT_ROOT / Path(row["frame_path"])
        try:
            with Image.open(fpath) as img:
                img.verify()
            with Image.open(fpath) as img:
                w, h = img.size
                mode = img.mode
                if mode != "RGB":
                    quality_failures.append((row["frame_path"], f"Expected RGB mode, got {mode}"))
                if w != row["width"] or h != row["height"]:
                    quality_failures.append((row["frame_path"], f"Dimension mismatch: expected {row['width']}x{row['height']}, got {w}x{h}"))
        except Exception as e:
            quality_failures.append((row["frame_path"], str(e)))

    print(f"  Verified {len(frame_manifest_rows)} extracted image frames.")
    print(f"  Quality check failures: {len(quality_failures)}")

    # -------------------------------------------------------------------------
    # 6. Unified Dataset Catalog (Original Images + Extracted Video Frames)
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Building unified dataset catalog (dataset_catalog.csv)...")
    catalog_rows = []
    
    # 1. Existing original images
    for leaf in leaf_classes:
        for img_path in leaf["images"]:
            rel_p = img_path.relative_to(PROJECT_ROOT)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                w, h = 0, 0
                
            catalog_rows.append({
                "sample_path": str(rel_p).replace("\\", "/"),
                "category": leaf["category"],
                "subcategory": leaf["subcategory"] or "",
                "class_name": leaf["class_name"],
                "source_type": "image",
                "source_file": img_path.name,
                "source_video_id": "N/A",
                "frame_index": "N/A",
                "timestamp_seconds": "N/A",
                "width": w,
                "height": h
            })

    # 2. Extracted video frames
    for row in frame_manifest_rows:
        # Find subcategory
        cls = row["class_name"]
        cat = row["category"]
        subcat = ""
        for leaf in leaf_classes:
            if leaf["class_name"] == cls and leaf["category"] == cat:
                subcat = leaf["subcategory"] or ""
                break
                
        catalog_rows.append({
            "sample_path": row["frame_path"],
            "category": row["category"],
            "subcategory": subcat,
            "class_name": row["class_name"],
            "source_type": "video_frame",
            "source_file": row["source_video"],
            "source_video_id": row["source_video_id"],
            "frame_index": row["frame_index"],
            "timestamp_seconds": row["timestamp_seconds"],
            "width": row["width"],
            "height": row["height"]
        })

    with open(DATASET_CATALOG_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_path", "category", "subcategory", "class_name", 
            "source_type", "source_file", "source_video_id", 
            "frame_index", "timestamp_seconds", "width", "height"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog_rows)

    print(f"  Total samples cataloged: {len(catalog_rows)} (Original Images: 550, Video Frames: {total_extracted_frames})")
    print(f"  Saved unified catalog to: {DATASET_CATALOG_CSV}")

    # -------------------------------------------------------------------------
    # 7. Distribution Table (processed_class_distribution.csv)
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Generating processed_class_distribution.csv...")
    distribution_rows = []
    classes_still_underrepresented = []

    for leaf in leaf_classes:
        cls = leaf["class_name"]
        img_n = len(leaf["images"])
        vid_n = len(leaf["videos"])
        extracted_n = class_extracted_counts.get(cls, 0)
        total_samples = img_n + extracted_n

        distribution_rows.append({
            "class_name": cls,
            "category": leaf["category"],
            "original_image_count": img_n,
            "video_count": vid_n,
            "extracted_frame_count": extracted_n,
            "total_available_samples": total_samples
        })

        if total_samples < 7:
            classes_still_underrepresented.append((cls, leaf["category"], total_samples, img_n, vid_n))

    with open(PROCESSED_CLASS_DISTRIBUTION_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_name", "category", "original_image_count", 
            "video_count", "extracted_frame_count", "total_available_samples"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(distribution_rows)

    # -------------------------------------------------------------------------
    # 8. Detailed Report (video_extraction_report.txt)
    # -------------------------------------------------------------------------
    print("\n[STEP 8] Generating video_extraction_report.txt...")
    
    # Class categories by modality
    img_only_classes = [l["class_name"] for l in leaf_classes if len(l["images"]) > 0 and len(l["videos"]) == 0]
    vid_only_classes = [l["class_name"] for l in leaf_classes if len(l["images"]) == 0 and len(l["videos"]) > 0]
    both_classes = [l["class_name"] for l in leaf_classes if len(l["images"]) > 0 and len(l["videos"]) > 0]

    with open(VIDEO_EXTRACTION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KUCHIPUDI DANCE RECOGNITION - VIDEO FRAME EXTRACTION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Video Files Inspected      : {len(all_videos_found)}\n")
        f.write(f"Successfully Decoded Videos      : {len(valid_videos)}\n")
        f.write(f"Failed Videos                    : {len(failed_videos)}\n")
        f.write(f"Total Extracted Frames           : {total_extracted_frames}\n")
        f.write(f"Failed Frame Extractions         : {len(extraction_failures)}\n")
        f.write(f"Quality Check Failures           : {len(quality_failures)}\n")
        f.write(f"Sampling Strategy                : {FRAME_SAMPLING_MODE} (Target: {TARGET_FRAMES_PER_VIDEO} frames/video)\n\n")

        f.write("-" * 80 + "\n")
        f.write("MODALITY CLASSIFICATION SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Classes with Image Data Only     : {len(img_only_classes)}\n")
        f.write(f"Classes with Video Data Only     : {len(vid_only_classes)}\n")
        f.write(f"Classes with Both Image & Video  : {len(both_classes)} ({', '.join(both_classes)})\n\n")

        f.write("-" * 80 + "\n")
        f.write("EXTRACTED FRAMES PER CLASS (15 Video Classes)\n")
        f.write("-" * 80 + "\n")
        for leaf in leaf_classes:
            if len(leaf["videos"]) > 0:
                cls = leaf["class_name"]
                f.write(f"- [{leaf['category']}] {cls:<25} : {len(leaf['videos'])} videos -> {class_extracted_counts[cls]} extracted frames\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("EXTRACTED FRAMES PER VIDEO\n")
        f.write("-" * 80 + "\n")
        for item in valid_videos:
            v_name = item["video_path"].name
            cnt = video_extracted_counts[v_name]
            f.write(f"- [{item['video_id']}] {item['category']} / {item['class_name']} / {v_name:<45} : {cnt} frames ({item['duration']}s, {item['fps']} fps)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("CLASSES REMAINING UNDERREPRESENTED (Total Samples < 7)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Found {len(classes_still_underrepresented)} classes with < 7 total samples:\n")
        for cls, cat, total_s, img_s, vid_s in classes_still_underrepresented:
            f.write(f"- [{cat}] {cls:<25} : Total={total_s} (Original Images={img_s}, Videos={vid_s})\n")

    print("\nExtraction & Reporting Complete!")
    print(f"  Reports saved to: {REPORTS_DIR}")
    print(f"  Extracted frames saved to: {VIDEO_FRAMES_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
