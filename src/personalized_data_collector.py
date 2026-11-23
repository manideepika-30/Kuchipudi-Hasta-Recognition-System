"""
STEP 29A — Personalized Kuchipudi Webcam Data Collector
=======================================================
Interactive, robust, resume-capable webcam data collector for creating
the dancer's personalized Kuchipudi dataset for live demonstration.

Target Collection Protocol:
  1. Hasthas (52 classes)        : 20 images/class    = 1,040 images
  2. Shiro Bhedhas (9 classes)   : 15 sequences/class =   135 sequences
  3. Drishti Bhedhas (8 classes) : 20 images/class    =   160 images
  4. Pada Karma (6 classes)      : 15 sequences/class =    90 sequences
  Total Target                   : 1,425 samples/sequences

Key Safeguards:
  - Resolution: 1280x720 when supported.
  - Preview is mirrored for user convenience; saved frames are NATIVE (unmirrored).
  - MediaPipe landmark extraction validated prior to saving.
  - Resume-capable: reads data/personalized/metadata/personalized_samples.csv.
  - Class completion banners: prompts user with [N] when 20/20 or 15/15 reached.
  - Does NOT alter original project datasets or existing models.
"""

import os
import sys
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical_model import build_hierarchical_mapping
from src.landmark_extraction import KuchipudiLandmarkPipeline

# Base personalized directories
PERSONALIZED_DIR = PROJECT_ROOT / "data" / "personalized"
METADATA_DIR = PERSONALIZED_DIR / "metadata"
METADATA_FILE = METADATA_DIR / "personalized_samples.csv"

IMAGES_HASTHAS_DIR = PERSONALIZED_DIR / "images" / "hasthas"
IMAGES_DRISHTI_DIR = PERSONALIZED_DIR / "images" / "drishti"
SEQUENCES_SHIRO_DIR = PERSONALIZED_DIR / "sequences" / "shiro"
SEQUENCES_PADA_DIR = PERSONALIZED_DIR / "sequences" / "pada"

TARGET_COUNTS = {
    "1. Hasthas": 20,
    "2. Shiro Bhedhas (HEAD MOVEMENTS)": 15,
    "3. Drishti Bhedhas(EYE MOVEMENTS)": 20,
    "4. Pada Karma(Foot movements)": 15
}

TOTAL_TARGET_SAMPLES = (
    52 * 20 + # Hasthas = 1040
    9 * 15 +  # Shiro = 135
    8 * 20 +  # Drishti = 160
    6 * 15    # Pada = 90
) # = 1425


def sanitize_filename(name: str) -> str:
    """Sanitizes class name for clean filesystem directory creation."""
    return "".join(c if (c.isalnum() or c in (" ", "_", "-")) else "_" for c in name).strip().replace(" ", "_")


# =============================================================================
# Metadata Tracker & Resume Manager
# =============================================================================

class PersonalizedDatasetTracker:
    """
    Manages sample metadata, tracking existing counts, and appending new samples.
    """

    CSV_COLUMNS = [
        "sample_id",
        "category",
        "class_name",
        "class_index",
        "source_type",
        "file_path",
        "capture_timestamp",
        "landmark_valid",
        "sequence_length"
    ]

    def __init__(self, metadata_path: Path = METADATA_FILE):
        self.metadata_path = metadata_path
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if not self.metadata_path.exists():
            with open(self.metadata_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_COLUMNS)
            self.records = []
            return

        self.records = []
        with open(self.metadata_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)

    def get_count_for_class(self, category: str, class_name: str) -> int:
        return sum(
            1 for r in self.records
            if r.get("category") == category and r.get("class_name") == class_name and r.get("landmark_valid") in ("True", "1", True)
        )

    def get_total_count(self) -> int:
        return len(self.records)

    def get_category_counts(self, category: str) -> int:
        return sum(1 for r in self.records if r.get("category") == category and r.get("landmark_valid") in ("True", "1", True))

    def add_sample(
        self,
        category: str,
        class_name: str,
        class_index: int,
        source_type: str,
        file_path: str,
        landmark_valid: bool = True,
        sequence_length: int = 1
    ) -> str:
        sample_id = f"pers_{int(time.time()*1000)}"
        timestamp = datetime.now().isoformat()
        rel_path = os.path.relpath(file_path, PROJECT_ROOT).replace("\\", "/")

        row = {
            "sample_id": sample_id,
            "category": category,
            "class_name": class_name,
            "class_index": class_index,
            "source_type": source_type,
            "file_path": rel_path,
            "capture_timestamp": timestamp,
            "landmark_valid": landmark_valid,
            "sequence_length": sequence_length
        }
        self.records.append(row)

        with open(self.metadata_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writerow(row)

        return sample_id


# =============================================================================
# Main Data Collector Application
# =============================================================================

class PersonalizedDataCollector:
    """
    Interactive OpenCV and CLI interface for webcam data capture.
    """

    def __init__(self):
        # 1. Load Hierarchical Mapping
        self.mapping = build_hierarchical_mapping(save_json=False)
        self.tracker = PersonalizedDatasetTracker()

        # Build clean category info
        self.categories = [
            {
                "id": 1,
                "name": "1. Hasthas",
                "short": "HASTHAS",
                "type": "image",
                "target": 20,
                "classes": self.mapping["experts"]["1. Hasthas"]["classes"]
            },
            {
                "id": 2,
                "name": "2. Shiro Bhedhas (HEAD MOVEMENTS)",
                "short": "SHIRO BHEDHAS",
                "type": "sequence",
                "target": 15,
                "classes": self.mapping["experts"]["2. Shiro Bhedhas (HEAD MOVEMENTS)"]["classes"]
            },
            {
                "id": 3,
                "name": "3. Drishti Bhedhas(EYE MOVEMENTS)",
                "short": "DRISHTI BHEDHAS",
                "type": "image",
                "target": 20,
                "classes": self.mapping["experts"]["3. Drishti Bhedhas(EYE MOVEMENTS)"]["classes"]
            },
            {
                "id": 4,
                "name": "4. Pada Karma(Foot movements)",
                "short": "PADA KARMA",
                "type": "sequence",
                "target": 15,
                "classes": self.mapping["experts"]["4. Pada Karma(Foot movements)"]["classes"]
            }
        ]

        self.pipeline: Optional[KuchipudiLandmarkPipeline] = None
        self.last_capture_time = 0.0
        self.min_capture_interval = 0.8  # seconds cooldown between static captures

    def init_pipeline(self):
        if self.pipeline is None:
            print("[COLLECTOR] Initializing MediaPipe anatomical detectors...")
            self.pipeline = KuchipudiLandmarkPipeline(static_image_mode=True)

    def close(self):
        if self.pipeline is not None:
            self.pipeline.close()

    def get_save_dir(self, cat_key: str, class_name: str) -> Path:
        safe_cls = sanitize_filename(class_name)
        if "Hasthas" in cat_key:
            d = IMAGES_HASTHAS_DIR / safe_cls
        elif "Shiro" in cat_key:
            d = SEQUENCES_SHIRO_DIR / safe_cls
        elif "Drishti" in cat_key:
            d = IMAGES_DRISHTI_DIR / safe_cls
        else:
            d = SEQUENCES_PADA_DIR / safe_cls
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -------------------------------------------------------------------------
    # Static Image Capture (Hasthas & Drishti)
    # -------------------------------------------------------------------------
    def capture_image_sample(
        self,
        frame_native_bgr: np.ndarray,
        cat_key: str,
        class_name: str,
        class_index: int
    ) -> Tuple[bool, str]:
        """
        Validates landmarks on native unmirrored frame and saves sample + landmarks.
        """
        now = time.time()
        if now - self.last_capture_time < self.min_capture_interval:
            return False, "Cooldown active — please wait a moment"

        self.init_pipeline()
        frame_rgb = cv2.cvtColor(frame_native_bgr, cv2.COLOR_BGR2RGB)
        lm_res = self.pipeline.extract_from_frame(frame_rgb, cat_key)

        detected = bool(lm_res.get("detected", False))
        if not detected:
            if "Hasthas" in cat_key:
                return False, "Hand not detected — position your hand clearly."
            elif "Drishti" in cat_key:
                return False, "Eyes/face not detected — face camera directly."
            else:
                return False, "Landmarks not detected."

        # Target directory
        save_dir = self.get_save_dir(cat_key, class_name)
        curr_count = self.tracker.get_count_for_class(cat_key, class_name)
        sample_idx = curr_count + 1

        img_filename = f"sample_{sample_idx:04d}.jpg"
        img_path = save_dir / img_filename
        npz_filename = f"sample_{sample_idx:04d}_landmarks.npz"
        npz_path = save_dir / npz_filename

        # Save native unmirrored image (high quality JPEG)
        cv2.imwrite(str(img_path), frame_native_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save extracted landmark features for downstream training
        np.savez_compressed(
            str(npz_path),
            vector=lm_res.get("vector", np.zeros(0, dtype=np.float32)),
            detected=True,
            category=cat_key,
            class_name=class_name,
            timestamp=datetime.now().isoformat()
        )

        # Update metadata
        self.tracker.add_sample(
            category=cat_key,
            class_name=class_name,
            class_index=class_index,
            source_type="image",
            file_path=str(img_path),
            landmark_valid=True,
            sequence_length=1
        )
        self.last_capture_time = now
        return True, f"Sample {sample_idx} saved successfully!"

    # -------------------------------------------------------------------------
    # Sequence Capture (Shiro & Pada)
    # -------------------------------------------------------------------------
    def save_recorded_sequence(
        self,
        frames_list: List[np.ndarray],
        timestamps_list: List[float],
        cat_key: str,
        class_name: str,
        class_index: int
    ) -> Tuple[bool, str]:
        """
        Validates landmarks across recorded frames and saves temporal sequence.
        """
        if len(frames_list) < 16:
            return False, f"Sequence too short ({len(frames_list)} frames). Need >= 16 frames."

        self.init_pipeline()
        lm_vectors = []
        valid_count = 0

        for frame_bgr in frames_list:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.pipeline.extract_from_frame(frame_rgb, cat_key)
            if res.get("detected", False):
                valid_count += 1
                lm_vectors.append(res["vector"])
            else:
                lm_vectors.append(res["vector"])

        valid_ratio = valid_count / len(frames_list)
        if valid_ratio < 0.65:
            if "Shiro" in cat_key:
                return False, f"Head/pose tracking lost in {100 - valid_ratio*100:.0f}% of frames — sequence discarded."
            else:
                return False, f"Foot/leg landmarks lost in {100 - valid_ratio*100:.0f}% of frames — sequence discarded."

        # Target directory
        save_dir = self.get_save_dir(cat_key, class_name)
        curr_count = self.tracker.get_count_for_class(cat_key, class_name)
        seq_idx = curr_count + 1

        seq_filename = f"seq_{seq_idx:04d}.npz"
        seq_path = save_dir / seq_filename

        # Downsample or extract exactly 16 uniform frames for consistent sequence representation
        indices = np.linspace(0, len(frames_list) - 1, 16).astype(int)
        seq16_features = np.array([lm_vectors[i] for i in indices], dtype=np.float32)
        seq16_timestamps = np.array([timestamps_list[i] for i in indices], dtype=np.float32)

        # Compute velocity
        if len(seq16_features) > 1:
            vel15 = seq16_features[1:] - seq16_features[:-1]
        else:
            vel15 = np.zeros_like(seq16_features)

        np.savez_compressed(
            str(seq_path),
            sequence_features=seq16_features,
            velocity_features=vel15,
            timestamps=seq16_timestamps,
            category=cat_key,
            class_name=class_name,
            total_recorded_frames=len(frames_list),
            valid_ratio=float(valid_ratio),
            capture_timestamp=datetime.now().isoformat()
        )

        self.tracker.add_sample(
            category=cat_key,
            class_name=class_name,
            class_index=class_index,
            source_type="sequence",
            file_path=str(seq_path),
            landmark_valid=True,
            sequence_length=len(frames_list)
        )
        return True, f"Sequence {seq_idx} saved ({len(frames_list)} frames, {valid_ratio*100:.0f}% tracked)!"

    # -------------------------------------------------------------------------
    # Interactive UI Session
    # -------------------------------------------------------------------------
    def run_interactive_session(self, headless_test: bool = False):
        if headless_test:
            print("\n[COLLECTOR] Running headless automated test...")
            self.init_pipeline()
            # Test synthetic Hasthas frame
            dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cat_hasthas = self.categories[0]
            cls_0 = cat_hasthas["classes"][0]
            cname = cls_0["class_name"]
            cidx = cls_0["local_class_id"]
            cnt_before = self.tracker.get_count_for_class(cat_hasthas["name"], cname)
            print(f"  Hasthas class '{cname}' initial count: {cnt_before}/{cat_hasthas['target']}")

            # Save synthetic sample to verify file creation & metadata
            save_dir = self.get_save_dir(cat_hasthas["name"], cname)
            s_idx = cnt_before + 1
            img_p = save_dir / f"sample_{s_idx:04d}.jpg"
            cv2.imwrite(str(img_p), dummy_frame)
            npz_p = save_dir / f"sample_{s_idx:04d}_landmarks.npz"
            np.savez_compressed(str(npz_p), vector=np.zeros(175, dtype=np.float32), detected=True)
            self.tracker.add_sample(
                cat_hasthas["name"], cname, cidx, "image", str(img_p), True, 1
            )
            cnt_after = self.tracker.get_count_for_class(cat_hasthas["name"], cname)
            print(f"  Hasthas class '{cname}' after mock capture: {cnt_after}/{cat_hasthas['target']}")
            assert cnt_after == cnt_before + 1, "Resume tracking failed!"
            self.print_summary()
            self.close()
            print("[COLLECTOR] Headless test successfully verified.")
            return

        # Interactive Camera Loop
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open default camera (index 0).")
            return

        # Attempt 1280x720 HD capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n[CAMERA] Initialized at resolution: {actual_w}x{actual_h}")

        active_cat_idx = 0
        active_class_idx = 0

        # Recording state for sequences (Shiro and Pada)
        is_recording = False
        recording_start_time = 0.0
        recorded_frames: List[np.ndarray] = []
        recorded_timestamps: List[float] = []
        max_record_duration = 3.0  # seconds

        status_msg = "Ready. Press SPACE to capture."
        status_color = (0, 215, 255) # Amber
        status_time = time.time()

        try:
            while True:
                ret, frame_native = cap.read()
                if not ret:
                    break

                curr_time = time.time()

                # Current category and class
                cat = self.categories[active_cat_idx]
                cat_key = cat["name"]
                cat_short = cat["short"]
                is_seq = (cat["type"] == "sequence")
                target_count = cat["target"]

                cls_info = cat["classes"][active_class_idx]
                class_name = cls_info["class_name"]
                local_cid = cls_info["local_class_id"]
                current_count = self.tracker.get_count_for_class(cat_key, class_name)
                overall_count = self.tracker.get_total_count()

                # If recording sequence, buffer native frames
                if is_recording:
                    recorded_frames.append(frame_native.copy())
                    recorded_timestamps.append(curr_time - recording_start_time)
                    elapsed = curr_time - recording_start_time

                    if elapsed >= max_record_duration:
                        # Auto-stop sequence recording
                        is_recording = False
                        status_msg = "Processing and validating sequence..."
                        ok, msg = self.save_recorded_sequence(
                            recorded_frames, recorded_timestamps, cat_key, class_name, local_cid
                        )
                        status_msg = msg
                        status_color = (0, 255, 0) if ok else (0, 0, 255)
                        status_time = curr_time
                        recorded_frames = []
                        recorded_timestamps = []

                # Mirrored frame for user display
                display_frame = cv2.flip(frame_native, 1)
                dh, dw = display_frame.shape[:2]

                # Top Status Banner (Dark Glassmorphism)
                cv2.rectangle(display_frame, (0, 0), (dw, 65), (20, 20, 20), -1)
                cat_header = f"[{cat['id']}/4] {cat_short}  |  TOTAL: {overall_count} / {TOTAL_TARGET_SAMPLES}"
                cv2.putText(display_frame, cat_header, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)

                cls_header = f"GESTURE [{active_class_idx + 1}/{len(cat['classes'])}]: {class_name.upper()}  |  Captured: {current_count} / {target_count}"
                cv2.putText(display_frame, cls_header, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

                # Bottom Panel: Guidance & Feedback
                cv2.rectangle(display_frame, (0, dh - 85), (dw, dh), (15, 15, 15), -1)

                if is_recording:
                    elapsed = curr_time - recording_start_time
                    rec_text = f"RECORDING... ({elapsed:.1f}s / {max_record_duration:.1f}s) — PERFORM MOVEMENT"
                    cv2.putText(display_frame, rec_text, (20, dh - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    cv2.circle(display_frame, (dw - 40, dh - 50), 12, (0, 0, 255), -1)
                elif current_count >= target_count:
                    comp_text = f"{class_name.upper()} COMPLETED ({target_count}/{target_count})! Press [N] for Next Class."
                    cv2.putText(display_frame, comp_text, (20, dh - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                else:
                    if curr_time - status_time < 3.0:
                        cv2.putText(display_frame, status_msg, (20, dh - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2)
                    else:
                        if is_seq:
                            prompt_text = "Press SPACE to START sequence recording (approx 3 sec)."
                        else:
                            prompt_text = "Hold gesture clearly. Press SPACE to capture."
                        cv2.putText(display_frame, prompt_text, (20, dh - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 200, 200), 2)

                # Navigation Key Hints
                keys_hint = "Keys: [SPACE] Capture  [N] Next  [B] Prev  [1-4] Category  [Q] Summary & Quit"
                cv2.putText(display_frame, keys_hint, (20, dh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

                cv2.imshow("Kuchipudi Personalized Data Collector (Step 29A)", display_frame)

                # Key Handling
                k = cv2.waitKey(1) & 0xFF
                if k in [ord('q'), ord('Q'), 27]:
                    break
                elif k in [ord('1'), ord('2'), ord('3'), ord('4')]:
                    active_cat_idx = int(chr(k)) - 1
                    active_class_idx = 0
                    is_recording = False
                    recorded_frames = []
                    status_msg = f"Switched to {self.categories[active_cat_idx]['short']}"
                    status_color = (0, 215, 255)
                    status_time = curr_time
                elif k in [ord('n'), ord('N')]:
                    if active_class_idx < len(cat["classes"]) - 1:
                        active_class_idx += 1
                    else:
                        status_msg = "At last class in category."
                    is_recording = False
                    recorded_frames = []
                elif k in [ord('b'), ord('B')]:
                    if active_class_idx > 0:
                        active_class_idx -= 1
                    else:
                        status_msg = "At first class in category."
                    is_recording = False
                    recorded_frames = []
                elif k == 32: # SPACE
                    if is_seq:
                        if not is_recording:
                            is_recording = True
                            recording_start_time = curr_time
                            recorded_frames = [frame_native.copy()]
                            recorded_timestamps = [0.0]
                        else:
                            # Manually stop
                            is_recording = False
                            ok, msg = self.save_recorded_sequence(
                                recorded_frames, recorded_timestamps, cat_key, class_name, local_cid
                            )
                            status_msg = msg
                            status_color = (0, 255, 0) if ok else (0, 0, 255)
                            status_time = curr_time
                            recorded_frames = []
                            recorded_timestamps = []
                    else:
                        # Static image capture
                        ok, msg = self.capture_image_sample(frame_native, cat_key, class_name, local_cid)
                        status_msg = msg
                        status_color = (0, 255, 0) if ok else (0, 0, 255)
                        status_time = curr_time

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.close()
            self.print_summary()

    def print_summary(self):
        """Displays formatted dataset summary on exit (Part 15)."""
        print("\n" + "=" * 80)
        print("PERSONALIZED DATASET SUMMARY (STEP 29A)")
        print("=" * 80)

        total_collected = self.tracker.get_total_count()

        for cat in self.categories:
            cat_name = cat["name"]
            c_cnt = self.tracker.get_category_counts(cat_name)
            num_cls = len(cat["classes"])
            target = num_cls * cat["target"]
            pct = (c_cnt / target * 100.0) if target > 0 else 0.0
            unit = "images" if cat["type"] == "image" else "sequences"
            print(f"{cat['short']}:")
            print(f"  Classes         : {num_cls} classes")
            print(f"  Target          : {target} {unit} ({cat['target']} per class)")
            print(f"  Actual Collected: {c_cnt} {unit} ({pct:.1f}% complete)")

        overall_pct = (total_collected / TOTAL_TARGET_SAMPLES * 100.0) if TOTAL_TARGET_SAMPLES > 0 else 0.0
        print("-" * 80)
        print(f"TOTAL DATASET TARGET  : {TOTAL_TARGET_SAMPLES} samples / sequences")
        print(f"TOTAL ACTUAL COLLECTED: {total_collected} samples / sequences ({overall_pct:.1f}% complete)")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    collector = PersonalizedDataCollector()
    if len(sys.argv) > 1 and sys.argv[1] == "--headless":
        collector.run_interactive_session(headless_test=True)
    else:
        collector.run_interactive_session(headless_test=False)
