import os
import sys
import random
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, List
import numpy as np
from PIL import Image, ImageEnhance
import cv2

try:
    from config import (
        PROJECT_ROOT,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
        CNN_INPUT_SHAPE,
        ENABLE_AUGMENTATION,
        AUGMENTATION_PROBABILITY,
        ROTATION_RANGE_DEGREES,
        WIDTH_SHIFT_FRACTION,
        HEIGHT_SHIFT_FRACTION,
        ZOOM_RANGE,
        BRIGHTNESS_RANGE,
        CONTRAST_RANGE,
        HORIZONTAL_FLIP,
        VERTICAL_FLIP,
        MINORITY_CLASS_THRESHOLD,
        MINORITY_OVERSAMPLE_FACTOR,
        AUGMENTATION_EXAMPLES_DIR,
        AUGMENTATION_REPORT_TXT,
        RANDOM_SEED
    )
except ImportError:
    from src.config import (
        PROJECT_ROOT,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
        CNN_INPUT_SHAPE,
        ENABLE_AUGMENTATION,
        AUGMENTATION_PROBABILITY,
        ROTATION_RANGE_DEGREES,
        WIDTH_SHIFT_FRACTION,
        HEIGHT_SHIFT_FRACTION,
        ZOOM_RANGE,
        BRIGHTNESS_RANGE,
        CONTRAST_RANGE,
        HORIZONTAL_FLIP,
        VERTICAL_FLIP,
        MINORITY_CLASS_THRESHOLD,
        MINORITY_OVERSAMPLE_FACTOR,
        AUGMENTATION_EXAMPLES_DIR,
        AUGMENTATION_REPORT_TXT,
        RANDOM_SEED
    )

class KuchipudiAugmenter:
    """
    Domain-Specific Data Augmentation for Kuchipudi Classical Dance Recognition.
    
    CRITICAL DOMAIN RULES:
    1. Horizontal Flip is STRICTLY DISABLED:
       In classical Indian dance, hand mudras and eye/head movements possess specific
       left-vs-right lateral semantics (e.g. Paravrittam, Saachi glance). Flipping
       inverts the meaning and destroys lateral labels.
    2. Vertical Flip is STRICTLY DISABLED:
       Inverts gravitational direction of gestures (Pataka upright vs downward mudras).
    3. Rotation is STRICTLY BOUNDED (+/- 8 degrees):
       Prevents changing mudra orientation from Urdhva (upward) to Tiryak (sideways) or Adho (downward).
    4. Translation & Zoom are SUBTLE (+/- 6% and +/- 5%):
       Prevents clipping of critical finger configurations, eye corners, or foot positions.
    5. Photometric Adjustments are MILD (+/- 10%):
       Simulates realistic variations in stage lighting and camera exposure without color distortion.
    """

    def __init__(
        self,
        enable: bool = ENABLE_AUGMENTATION,
        prob: float = AUGMENTATION_PROBABILITY,
        rotation_deg: float = ROTATION_RANGE_DEGREES,
        width_shift: float = WIDTH_SHIFT_FRACTION,
        height_shift: float = HEIGHT_SHIFT_FRACTION,
        zoom_range: Tuple[float, float] = ZOOM_RANGE,
        brightness_range: Tuple[float, float] = BRIGHTNESS_RANGE,
        contrast_range: Tuple[float, float] = CONTRAST_RANGE,
        seed: Optional[int] = RANDOM_SEED
    ):
        self.enable = enable
        self.prob = prob
        self.rotation_deg = rotation_deg
        self.width_shift = width_shift
        self.height_shift = height_shift
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.np_rng = np.random.RandomState(seed)

    def apply_affine_transforms(self, img_arr: np.ndarray) -> np.ndarray:
        """
        Apply subtle rotation, translation, and zoom using OpenCV affine matrix.
        Input: img_arr of shape (H, W, 3), float32 [0.0, 1.0].
        """
        h, w = img_arr.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 1. Subtle rotation angle in [-rotation_deg, +rotation_deg]
        angle = self.rng.uniform(-self.rotation_deg, self.rotation_deg)

        # 2. Subtle zoom scale in [zoom_min, zoom_max]
        scale = self.rng.uniform(self.zoom_range[0], self.zoom_range[1])

        # Get rotation + scale matrix
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

        # 3. Subtle translation in pixels
        tx = self.rng.uniform(-self.width_shift, self.width_shift) * w
        ty = self.rng.uniform(-self.height_shift, self.height_shift) * h
        rot_mat[0, 2] += tx
        rot_mat[1, 2] += ty

        # Warp affine with constant black border (consistent with letterbox zero-padding)
        transformed = cv2.warpAffine(
            img_arr, 
            rot_mat, 
            (w, h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0)
        )
        return transformed

    def apply_photometric_transforms(self, img_arr: np.ndarray) -> np.ndarray:
        """
        Apply mild brightness and contrast adjustments without channel color tinting.
        Input: img_arr of shape (H, W, 3), float32 [0.0, 1.0].
        """
        # 1. Brightness factor (multiplicative scaling)
        brightness_factor = self.rng.uniform(self.brightness_range[0], self.brightness_range[1])
        arr = img_arr * brightness_factor

        # 2. Contrast factor (contrast around mid-gray 0.5)
        contrast_factor = self.rng.uniform(self.contrast_range[0], self.contrast_range[1])
        arr = (arr - 0.5) * contrast_factor + 0.5

        # Clip strictly to [0.0, 1.0]
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def augment(self, img_tensor: np.ndarray, is_training: bool = True) -> np.ndarray:
        """
        Apply augmentation to a preprocessed image tensor (H, W, 3) in float32.
        
        CRITICAL LEAKAGE RULE:
        If is_training is False (Validation or Test mode), augmentation is 100% BYPASSED.
        """
        if not self.enable or not is_training:
            return img_tensor

        # Probabilistic application
        if self.rng.random() > self.prob:
            return img_tensor

        out = img_tensor.copy()

        # Apply geometric transforms
        out = self.apply_affine_transforms(out)

        # Apply photometric transforms
        out = self.apply_photometric_transforms(out)

        return out

    def get_keras_augmentation_layer(self):
        """
        Returns a TensorFlow/Keras preprocessing layer equivalent for inside-model augmentation.
        Can be plugged directly at the top of a tf.keras.Model during training.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers

            # Rotation in fractions of 2*pi: 8 deg / 360 = 0.0222
            rot_factor = self.rotation_deg / 360.0

            return tf.keras.Sequential([
                layers.RandomRotation(factor=(-rot_factor, rot_factor), fill_mode="constant"),
                layers.RandomTranslation(
                    height_factor=(-self.height_shift, self.height_shift),
                    width_factor=(-self.width_shift, self.width_shift),
                    fill_mode="constant"
                ),
                layers.RandomZoom(
                    height_factor=(1.0 - self.zoom_range[1], 1.0 - self.zoom_range[0]),
                    fill_mode="constant"
                ),
                layers.RandomBrightness(factor=(self.brightness_range[0] - 1.0, self.brightness_range[1] - 1.0))
            ], name="kuchipudi_augmentation_layer")
        except ImportError:
            return None
