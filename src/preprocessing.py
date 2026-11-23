import os
import sys
import csv
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
from PIL import Image

try:
    from config import (
        PROJECT_ROOT,
        DATA_DIR,
        REPORTS_DIR,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
        CNN_INPUT_SHAPE,
        RESIZE_METHOD,
        NORMALIZATION_METHOD,
        BATCH_SIZE,
        PREPROCESSING_EXAMPLES_DIR,
        PREPROCESSING_REPORT_TXT
    )
except ImportError:
    from src.config import (
        PROJECT_ROOT,
        DATA_DIR,
        REPORTS_DIR,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
        CNN_INPUT_SHAPE,
        RESIZE_METHOD,
        NORMALIZATION_METHOD,
        BATCH_SIZE,
        PREPROCESSING_EXAMPLES_DIR,
        PREPROCESSING_REPORT_TXT
    )

def resize_with_aspect_ratio_pad(
    img: Image.Image, 
    target_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH), 
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Resize image to fit within target_size while preserving aspect ratio,
    padding the remaining space with fill_color (letterboxing).
    Crucial for dance gestures to preserve precise mudra finger angles.
    """
    target_w, target_h = target_size[1], target_size[0]
    orig_w, orig_h = img.size

    if orig_w == 0 or orig_h == 0:
        return Image.new("RGB", (target_w, target_h), fill_color)

    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * ratio))
    new_h = max(1, int(orig_h * ratio))

    resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded_img = Image.new("RGB", (target_w, target_h), fill_color)

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    padded_img.paste(resized_img, (pad_left, pad_top))
    return padded_img

def normalize_pixels(
    arr: np.ndarray, 
    method: str = NORMALIZATION_METHOD
) -> np.ndarray:
    """
    Normalize pixel array according to the configured method.
    """
    arr = arr.astype(np.float32)
    if method == "rescale_0_1":
        return arr / 255.0
    elif method == "rescale_minus1_1":
        return (arr / 127.5) - 1.0
    elif method == "standardize":
        mean = np.mean(arr, axis=(0, 1), keepdims=True)
        std = np.std(arr, axis=(0, 1), keepdims=True) + 1e-7
        return (arr - mean) / std
    else:
        return arr / 255.0

def load_and_preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
    resize_method: str = RESIZE_METHOD,
    normalization_method: str = NORMALIZATION_METHOD
) -> np.ndarray:
    """
    Complete single-image preprocessing pipeline:
    1. Load image safely.
    2. Verify readability.
    3. Convert to RGB.
    4. Resize to target_size (aspect_pad or direct).
    5. Normalize pixel values to float32.
    Returns:
        np.ndarray of shape (target_height, target_width, 3), float32.
    """
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    # 1. Load and verify
    with Image.open(path) as img:
        img.verify()

    with Image.open(path) as img:
        # 2. RGB conversion (handles RGBA, Grayscale, CMYK)
        if img.mode != "RGB":
            img = img.convert("RGB")
        else:
            img = img.copy()

        # 3. Resizing
        if resize_method == "aspect_pad":
            processed_img = resize_with_aspect_ratio_pad(img, target_size=target_size)
        else:
            processed_img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)

        # 4. Convert to array and normalize
        arr = np.array(processed_img, dtype=np.float32)
        norm_arr = normalize_pixels(arr, method=normalization_method)

        return norm_arr

class KuchipudiDatasetGenerator:
    """
    Reusable generator for feeding preprocessed batches into CNN models.
    Supports train, validation, test, and cross-validation partition CSVs.
    """
    def __init__(
        self,
        csv_path: Union[str, Path],
        batch_size: int = BATCH_SIZE,
        target_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
        shuffle: bool = False,
        resize_method: str = RESIZE_METHOD,
        normalization_method: str = NORMALIZATION_METHOD,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_absolute():
            self.csv_path = PROJECT_ROOT / self.csv_path

        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.resize_method = resize_method
        self.normalization_method = normalization_method

        self.samples = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.samples.append(r)

        # Build class index mapping if not provided
        if class_to_idx is None:
            unique_classes = sorted(list(set(s["class_name"] for s in self.samples)))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx

        self.num_classes = len(self.class_to_idx)
        self.indices = list(range(len(self.samples)))

    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, len(self.samples), self.batch_size):
            batch_indices = self.indices[start_idx : start_idx + self.batch_size]
            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                sample_info = self.samples[idx]
                sample_p = sample_info["sample_path"]
                cls_name = sample_info["class_name"]

                img_tensor = load_and_preprocess_image(
                    sample_p,
                    target_size=self.target_size,
                    resize_method=self.resize_method,
                    normalization_method=self.normalization_method
                )
                label_idx = self.class_to_idx[cls_name]

                batch_images.append(img_tensor)
                batch_labels.append(label_idx)

            yield np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.int32)
