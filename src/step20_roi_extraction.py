"""
STEP 20 — Part A: Category-Specific ROI Extraction Pipeline
============================================================
Extracts deterministic category-specific Regions of Interest (ROIs):
  - 1. Hasthas: Hand(s) bounding box with margin, preserving both hands for Samyutha
  - 2. Shiro Bhedhas: Head and upper-body (head, neck, shoulder relationship)
  - 3. Drishti Bhedhas: Face and eye region (preserving both eyes, irises, and orientation)
  - 4. Pada Karma: Lower body and feet region (preserving feet and stance context)

Ensures:
  - 100% deterministic cropping
  - Zero horizontal flipping
  - Reproducible ROI metadata saved to disk
  - Development data only (v2_train.csv and v2_validation.csv)
  - Strict preservation of the locked test set
"""

import os
import cv2
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# MediaPipe solutions import
try:
    from mediapipe import solutions as mp_solutions
except ImportError:
    import mediapipe.python.solutions as mp_solutions

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class KuchipudiROIExtractor:
    """
    Deterministic ROI extractor for Classical Kuchipudi Dance Dataset.
    Uses MediaPipe landmarks with category-tuned margins and principled fallbacks.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.hands = mp_solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25
        )
        self.face = mp_solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25
        )
        self.pose = mp_solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.20,
            min_tracking_confidence=0.20
        )

    def close(self):
        self.hands.close()
        self.face.close()
        self.pose.close()

    def _make_square_bbox(
        self,
        ymin: float,
        xmin: float,
        ymax: float,
        xmax: float,
        H: int,
        W: int,
        margin_x: float = 0.20,
        margin_y: float = 0.20
    ) -> Tuple[int, int, int, int]:
        """
        Converts normalized coordinates to a square pixel bounding box with margins,
        clamping safely within image dimensions [0, H] and [0, W].
        Returns: (ymin_px, xmin_px, ymax_px, xmax_px)
        """
        # Pixel coordinates
        y1 = ymin * H
        y2 = ymax * H
        x1 = xmin * W
        x2 = xmax * W

        # Add margin
        box_w = max(10.0, x2 - x1)
        box_h = max(10.0, y2 - y1)

        y1 = max(0.0, y1 - box_h * margin_y)
        y2 = min(float(H), y2 + box_h * margin_y)
        x1 = max(0.0, x1 - box_w * margin_x)
        x2 = min(float(W), x2 + box_w * margin_x)

        # Center point
        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0

        # Square side length
        side = max(y2 - y1, x2 - x1)

        # Build square box centered at (cx, cy)
        y_start = max(0, int(round(cy - side / 2.0)))
        y_end = min(H, int(round(cy + side / 2.0)))
        x_start = max(0, int(round(cx - side / 2.0)))
        x_end = min(W, int(round(cx + side / 2.0)))

        # Adjust in case side exceeds bounds
        act_h = y_end - y_start
        act_w = x_end - x_start
        if act_h < side and y_start > 0:
            y_start = max(0, y_end - int(side))
        if act_w < side and x_start > 0:
            x_start = max(0, x_end - int(side))

        return y_start, x_start, y_end, x_end

    def extract_hasthas_roi(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Hasthas: Crop around detected hand(s). Preserves both hands for Samyutha.
        """
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)

        if res.multi_hand_landmarks:
            all_x = [lm.x for hl in res.multi_hand_landmarks for lm in hl.landmark]
            all_y = [lm.y for hl in res.multi_hand_landmarks for lm in hl.landmark]
            ymin, ymax = min(all_y), max(all_y)
            xmin, xmax = min(all_x), max(all_x)

            y1, x1, y2, x2 = self._make_square_bbox(ymin, xmin, ymax, xmax, H, W, margin_x=0.25, margin_y=0.25)
            crop = img_bgr[y1:y2, x1:x2]
            meta = {
                "detected": True,
                "num_hands": len(res.multi_hand_landmarks),
                "fallback": False,
                "bbox_norm": [float(ymin), float(xmin), float(ymax), float(xmax)],
                "bbox_px": [int(y1), int(x1), int(y2), int(x2)],
                "roi_area_ratio": float(((y2 - y1) * (x2 - x1)) / (H * W))
            }
        else:
            # Fallback: central 70% crop
            side = int(min(H, W) * 0.75)
            y1 = max(0, (H - side) // 2)
            x1 = max(0, (W - side) // 2)
            y2 = min(H, y1 + side)
            x2 = min(W, x1 + side)
            crop = img_bgr[y1:y2, x1:x2]
            meta = {
                "detected": False,
                "num_hands": 0,
                "fallback": True,
                "bbox_norm": [y1 / H, x1 / W, y2 / H, x2 / W],
                "bbox_px": [y1, x1, y2, x2],
                "roi_area_ratio": float(((y2 - y1) * (x2 - x1)) / (H * W))
            }

        roi_resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        return roi_resized, meta

    def extract_drishti_roi(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Drishti: Crop around face/eye region. Preserves both eyes, irises, and head orientation.
        """
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.face.process(img_rgb)

        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0].landmark
            all_x = [lm.x for lm in fl]
            all_y = [lm.y for lm in fl]
            ymin, ymax = min(all_y), max(all_y)
            xmin, xmax = min(all_x), max(all_x)

            # Moderate margin to preserve eye context and face orientation
            y1, x1, y2, x2 = self._make_square_bbox(ymin, xmin, ymax, xmax, H, W, margin_x=0.15, margin_y=0.15)
            crop = img_bgr[y1:y2, x1:x2]
            meta = {
                "detected": True,
                "iris_available": len(fl) >= 478,
                "fallback": False,
                "bbox_norm": [float(ymin), float(xmin), float(ymax), float(xmax)],
                "bbox_px": [int(y1), int(x1), int(y2), int(x2)],
                "roi_area_ratio": float(((y2 - y1) * (x2 - x1)) / (H * W))
            }
        else:
            # Fallback: upper-central third
            side = int(min(H, W) * 0.65)
            y1 = int(H * 0.05)
            x1 = max(0, (W - side) // 2)
            y2 = min(H, y1 + side)
            x2 = min(W, x1 + side)
            crop = img_bgr[y1:y2, x1:x2]
            meta = {
                "detected": False,
                "iris_available": False,
                "fallback": True,
                "bbox_norm": [y1 / H, x1 / W, y2 / H, x2 / W],
                "bbox_px": [y1, x1, y2, x2],
                "roi_area_ratio": float(((y2 - y1) * (x2 - x1)) / (H * W))
            }

        roi_resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        return roi_resized, meta

    def extract_shiro_roi(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Shiro: Crop around head and upper body (head, neck, shoulder relationship).
        """
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)

        if res.pose_landmarks:
            pl = res.pose_landmarks.landmark
            # Key points: nose(0), eyes(1-6), ears(7,8), mouth(9,10), shoulders(11,12)
            head_pts = [pl[i] for i in range(13)]
            all_x = [lm.x for lm in head_pts]
            all_y = [lm.y for lm in head_pts]

            ymin, ymax = min(all_y), max(all_y)
            xmin, xmax = min(all_x), max(all_x)

            # Extra margin above head and down to chest/sternum
            y1, x1, y2, x2 = self._make_square_bbox(ymin, xmin, ymax, xmax, H, W, margin_x=0.25, margin_y=0.35)
            crop = img_bgr[y1:y2, x1:x2]
            meta = {
                "detected": True,
                "fallback": False,
                "bbox_norm": [float(ymin), float(xmin), float(ymax), float(xmax)],
                "bbox_px": [int(y1), int(x1), int(y2), int(x2)],
                "roi_area_ratio": float(((y2 - y1) * (x2 - x1)) / (H * W))
            }
        else:
            # Fallback: upper-body 70%
            side = int(min(H, W) * 0.70)
            y1 = max(0, int(H * 0.05))
            x1 = max(0, (W - side) // 2)
            y2 = min(H, y1 + side)
            x2 = min(W, x1 + side)
            crop = img_bgr[y1:y2, x1:x2]
            meta = {
                "detected": False,
                "fallback": True,
                "bbox_norm": [y1 / H, x1 / W, y2 / H, x2 / W],
                "bbox_px": [y1, x1, y2, x2],
                "roi_area_ratio": float(((y2 - y1) * (x2 - x1)) / (H * W))
            }

        roi_resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        return roi_resized, meta

    def extract_pada_roi(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Pada Karma: Crop around lower body and feet (hips to toes with surrounding ground context).
        """
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)

        if res.pose_landmarks:
            pl = res.pose_landmarks.landmark
            # Key lower-body points: hips(23,24), knees(25,26), ankles(27,28), heels(29,30), toes(31,32)
            foot_pts = [pl[i] for i in range(23, 33)]
            all_x = [lm.x for lm in foot_pts]
            all_y = [lm.y for lm in foot_pts]

            ymin, ymax = min(all_y), max(all_y)
            xmin, xmax = min(all_x), max(all_x)

            y1, x1, y2, x2 = self._make_square_bbox(ymin, xmin, ymax, xmax, H, W, margin_x=0.20, margin_y=0.20)
            crop = img_bgr[y1:y2, x1:x2]
            meta = {
                "detected": True,
                "fallback": False,
                "bbox_norm": [float(ymin), float(xmin), float(ymax), float(xmax)],
                "bbox_px": [int(y1), int(x1), int(y2), int(x2)],
                "roi_area_ratio": float(((y2 - y1) * (x2 - x1)) / (H * W))
            }
        else:
            # Fallback: lower 65% of image
            side = int(min(H, W) * 0.70)
            y1 = max(0, H - side)
            x1 = max(0, (W - side) // 2)
            y2 = min(H, y1 + side)
            x2 = min(W, x1 + side)
            crop = img_bgr[y1:y2, x1:x2]
            meta = {
                "detected": False,
                "fallback": True,
                "bbox_norm": [y1 / H, x1 / W, y2 / H, x2 / W],
                "bbox_px": [y1, x1, y2, x2],
                "roi_area_ratio": float(((y2 - y1) * (x2 - x1)) / (H * W))
            }

        roi_resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        return roi_resized, meta

    def extract_roi(self, img_path: str, category: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Routes an image path to the appropriate category ROI extractor.
        """
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        cat_lower = category.lower()
        if "hastha" in cat_lower:
            return self.extract_hasthas_roi(img_bgr)
        elif "shiro" in cat_lower:
            return self.extract_shiro_roi(img_bgr)
        elif "drishti" in cat_lower:
            return self.extract_drishti_roi(img_bgr)
        elif "pada" in cat_lower:
            return self.extract_pada_roi(img_bgr)
        else:
            # General fallback: center square crop
            H, W = img_bgr.shape[:2]
            side = min(H, W)
            y1 = (H - side) // 2
            x1 = (W - side) // 2
            crop = img_bgr[y1:y1 + side, x1:x1 + side]
            resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
            return resized, {
                "detected": False,
                "fallback": True,
                "bbox_px": [y1, x1, y1 + side, x1 + side],
                "roi_area_ratio": 1.0
            }


def generate_development_rois():
    """
    Processes all samples in v2_train.csv and v2_validation.csv.
    Extracts category-specific ROIs, caches ROI images to disk,
    and produces reports/step20_roi_report.txt.
    """
    print("=" * 80)
    print("STEP 20: GENERATING CATEGORY-SPECIFIC ROIs (DEVELOPMENT DATA ONLY)")
    print("=" * 80)

    train_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "v2_train.csv")
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "v2_validation.csv")

    extractor = KuchipudiROIExtractor(target_size=(224, 224))

    roi_cache_dir = PROJECT_ROOT / "data" / "rois"
    (roi_cache_dir / "train").mkdir(parents=True, exist_ok=True)
    (roi_cache_dir / "validation").mkdir(parents=True, exist_ok=True)

    metadata_records = []
    stats_by_category = {
        "1. Hasthas": {"total": 0, "detected": 0, "fallback": 0, "areas": []},
        "2. Shiro Bhedhas (HEAD MOVEMENTS)": {"total": 0, "detected": 0, "fallback": 0, "areas": []},
        "3. Drishti Bhedhas(EYE MOVEMENTS)": {"total": 0, "detected": 0, "fallback": 0, "areas": []},
        "4. Pada Karma(Foot movements)": {"total": 0, "detected": 0, "fallback": 0, "areas": []},
    }

    partitions = [("train", train_df), ("validation", val_df)]

    for part_name, df in partitions:
        print(f"\nProcessing {part_name.upper()} split ({len(df)} samples)...")
        for i, row in df.iterrows():
            img_path = str(PROJECT_ROOT / row["sample_path"])
            cat = row["category"]
            cls = row["class_name"]
            stem = Path(row["sample_path"]).stem

            roi_img, meta = extractor.extract_roi(img_path, cat)

            # Save ROI crop
            out_filename = f"{stem}_roi.jpeg"
            out_path = roi_cache_dir / part_name / out_filename
            cv2.imwrite(str(out_path), roi_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Accumulate statistics
            if cat in stats_by_category:
                stats_by_category[cat]["total"] += 1
                if meta["detected"]:
                    stats_by_category[cat]["detected"] += 1
                else:
                    stats_by_category[cat]["fallback"] += 1
                stats_by_category[cat]["areas"].append(meta["roi_area_ratio"])

            meta_record = {
                "partition": part_name,
                "sample_path": row["sample_path"],
                "category": cat,
                "class_name": cls,
                "roi_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "detected": meta["detected"],
                "fallback": meta["fallback"],
                "roi_area_ratio": meta["roi_area_ratio"],
                "bbox_px": meta["bbox_px"]
            }
            metadata_records.append(meta_record)

            if (i + 1) % 100 == 0 or (i + 1) == len(df):
                print(f"  [{part_name}] Processed {i + 1}/{len(df)} samples")

    extractor.close()

    # Save metadata JSON
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "step20_roi_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)

    # Generate Text Report
    report_lines = [
        "=" * 80,
        "STEP 20: CATEGORY-SPECIFIC ROI EXTRACTION REPORT",
        "=" * 80,
        "",
        "1. SPLIT COVERAGE (DEVELOPMENT DATA ONLY):",
        f"  Total Samples Processed      : {len(metadata_records)} (Train: {len(train_df)}, Val: {len(val_df)})",
        "  Step 19 Test Data Status     : STRICTLY LOCKED & UNTOUCHED (Zero Test Samples Processed)",
        "  Target Image Resolution      : 224 x 224 pixels",
        "",
        "2. CATEGORY-WISE DETECTION & ROI QUALITY STATISTICS:",
        "-" * 80
    ]

    for cat, s in stats_by_category.items():
        det_rate = (s["detected"] / s["total"] * 100.0) if s["total"] > 0 else 0.0
        fb_rate = (s["fallback"] / s["total"] * 100.0) if s["total"] > 0 else 0.0
        mean_area = float(np.mean(s["areas"])) if s["areas"] else 0.0
        min_area = float(np.min(s["areas"])) if s["areas"] else 0.0
        max_area = float(np.max(s["areas"])) if s["areas"] else 0.0

        report_lines.extend([
            f"  Category: {cat}",
            f"    Total Samples              : {s['total']}",
            f"    Landmark Detected          : {s['detected']} ({det_rate:.2f}%)",
            f"    Principled Fallback Used   : {s['fallback']} ({fb_rate:.2f}%)",
            f"    Mean ROI Area (% of frame) : {mean_area * 100.0:.2f}%",
            f"    Min / Max ROI Area Ratio   : {min_area * 100.0:.2f}% / {max_area * 100.0:.2f}%",
            ""
        ])

    report_lines.extend([
        "=" * 80,
        "STATUS: ROI EXTRACTION COMPLETE — ALL METADATA RECORDED.",
        "=" * 80
    ])

    report_text = "\n".join(report_lines)
    with open(reports_dir / "step20_roi_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    return metadata_records


if __name__ == "__main__":
    generate_development_rois()
