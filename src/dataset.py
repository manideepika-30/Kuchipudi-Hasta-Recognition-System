import os
import sys
import json
import csv
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import numpy as np

try:
    from config import (
        PROJECT_ROOT,
        DATA_DIR,
        PROCESSED_DATA_DIR,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        NUM_CLASSES,
        BATCH_SIZE,
        RANDOM_SEED
    )
    from preprocessing import load_and_preprocess_image
    from augmentation import KuchipudiAugmenter
    from model import compute_balanced_class_weights
except ImportError:
    from src.config import (
        PROJECT_ROOT,
        DATA_DIR,
        PROCESSED_DATA_DIR,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        NUM_CLASSES,
        BATCH_SIZE,
        RANDOM_SEED
    )
    from src.preprocessing import load_and_preprocess_image
    from src.augmentation import KuchipudiAugmenter
    from src.model import compute_balanced_class_weights

import tensorflow as tf

CLASS_MAPPING_FILE = PROCESSED_DATA_DIR / "class_mapping.json"

def get_class_mapping() -> Dict[str, int]:
    """
    Loads or generates canonical mapping from (category::class_name) to class index (0..74).
    """
    if CLASS_MAPPING_FILE.exists():
        with open(CLASS_MAPPING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["mapping"]

    # Generate canonical mapping from all_samples_split.csv if not exists
    splits_csv = DATA_DIR / "splits" / "all_samples_split.csv"
    with open(splits_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pairs = sorted({(r["category"], r["class_name"]) for r in reader})

    mapping = {f"{cat}::{cls}": idx for idx, (cat, cls) in enumerate(pairs)}
    inv_mapping = {idx: {"category": cat, "class_name": cls, "full_name": f"{cat} / {cls}"} for idx, (cat, cls) in enumerate(pairs)}
    with open(CLASS_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump({"mapping": mapping, "classes": inv_mapping}, f, indent=2)
    return mapping

class KuchipudiDataSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence Data Generator for Kuchipudi Dance Dataset.
    
    Guarantees:
    1. Training-time on-the-fly stochastic augmentation.
    2. Zero disk bottleneck by serving preprocessed base arrays cached in memory.
    3. Strict isolation of validation/test data (never augmented, deterministic).
    4. Deterministic shuffling per epoch with reproducible seeding.
    """
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = BATCH_SIZE,
        is_training: bool = False,
        augmenter: Optional[KuchipudiAugmenter] = None,
        shuffle: bool = False,
        seed: int = RANDOM_SEED
    ):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.is_training = is_training
        self.augmenter = augmenter if (is_training and augmenter is not None) else None
        self.shuffle = shuffle
        self.indices = np.arange(len(self.images))
        self.rng = np.random.RandomState(seed)
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.images) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = self.images[batch_indices].copy()
        batch_labels = self.labels[batch_indices]

        # Apply augmentation on-the-fly during training
        if self.is_training and self.augmenter is not None:
            for i in range(len(batch_images)):
                batch_images[i] = self.augmenter.augment(batch_images[i], is_training=True)

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

def load_partition_data(
    partition_csv_path: Union[str, Path],
    class_mapping: Optional[Dict[str, int]] = None,
    num_classes: int = NUM_CLASSES,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Loads preprocessed image tensors and one-hot labels for a given partition.
    Returns:
      images: np.ndarray of shape (N, 224, 224, 3) in float32 [0.0, 1.0]
      labels_onehot: np.ndarray of shape (N, num_classes)
      labels_int: List[int] integer class indices
      sample_paths: List[str] relative sample paths
    """
    if class_mapping is None:
        class_mapping = get_class_mapping()

    partition_csv_path = Path(partition_csv_path)
    records = []
    with open(partition_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)

    if verbose:
        print(f"Loading {len(records)} samples from {partition_csv_path.name}...")

    images = []
    labels_int = []
    sample_paths = []

    for r in records:
        full_key = f"{r['category']}::{r['class_name']}"
        cls_idx = class_mapping[full_key]
        img_path = PROJECT_ROOT / r["sample_path"]
        arr = load_and_preprocess_image(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        images.append(arr)
        labels_int.append(cls_idx)
        sample_paths.append(r["sample_path"])

    images = np.array(images, dtype=np.float32)
    labels_int_arr = np.array(labels_int, dtype=int)

    # One-hot encoding
    labels_onehot = np.zeros((len(labels_int), num_classes), dtype=np.float32)
    for i, idx in enumerate(labels_int):
        labels_onehot[i, idx] = 1.0

    if verbose:
        print(f"  Loaded {len(images)} tensors: shape={images.shape}, memory={images.nbytes / 1e6:.1f} MB")

    return images, labels_onehot, labels_int, sample_paths
