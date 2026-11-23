"""
STEP 21 — Final Dual-Stream Recognition Model
=============================================
Architecture:
  WHOLE-FRAME STREAM (Step 18 Visual Features, 960-dim)
          +
  ANATOMICAL STREAM (MediaPipe Landmark Geometry + Step 20 Kinematic Statistics)
          ↓
  LIGHTWEIGHT FUSION LAYER (Dense 128 / BatchNorm / Dropout / L2 Regularization)
          ↓
  CATEGORY-SPECIFIC EXPERT CLASSIFIERS
          ↓
  KUCHIPUDI GESTURE RECOGNITION (75 Classes)

Strict Rules:
  - Preserves whole-frame visual context (does not discard full image)
  - Incorporates fine-grained anatomical landmark and kinematic features
  - Compact, anti-overfitting fusion design (< 200k parameters per expert)
  - NIMILITA and AVALOKITA remain strictly separate classes
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from typing import Tuple, Optional


def build_dual_stream_hasthas_model(
    vis_dim: int = 960,
    lm_dim: int = 175,
    num_classes: int = 52,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Hasthas Dual-Stream Model:
    Stream 1: Whole-Frame Visual Embedding (960)
    Stream 2: Hand Landmarks & Joint Geometry (175)
    """
    vis_in = layers.Input(shape=(vis_dim,), name="whole_frame_visual_input")
    lm_in = layers.Input(shape=(lm_dim,), name="hand_landmark_input")

    # Visual branch projection
    x_vis = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="vis_proj")(vis_in)
    x_vis = layers.Dropout(0.25, name="vis_drop")(x_vis)

    # Landmark branch projection
    x_lm = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="lm_proj")(lm_in)
    x_lm = layers.Dropout(0.20, name="lm_drop")(x_lm)

    # Fusion
    merged = layers.concatenate([x_vis, x_lm], name="hasthas_fusion_concat")
    x = layers.Dense(192, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_192")(merged)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(0.35, name="fusion_drop_1")(x)

    x = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_96")(x)
    x = layers.Dropout(0.25, name="fusion_drop_2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="hasthas_gesture_output")(x)
    return tf.keras.Model(inputs=[vis_in, lm_in], outputs=out, name="Dual_Stream_Hasthas")


def build_dual_stream_shiro_model(
    vis_dim: int = 960,
    anat_dim: int = 32,
    num_classes: int = 9,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Shiro Dual-Stream Model:
    Stream 1: Whole-Frame Visual Context (960)
    Stream 2: Head/Neck Landmarks + Kinematic Angular Velocities (anat_dim)
    """
    vis_in = layers.Input(shape=(vis_dim,), name="whole_frame_visual_input")
    anat_in = layers.Input(shape=(anat_dim,), name="shiro_anatomical_input")

    x_vis = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="vis_proj")(vis_in)
    x_vis = layers.Dropout(0.25, name="vis_drop")(x_vis)

    x_anat = layers.Dense(48, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="anat_proj")(anat_in)
    x_anat = layers.Dropout(0.20, name="anat_drop")(x_anat)

    merged = layers.concatenate([x_vis, x_anat], name="shiro_fusion_concat")
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_128")(merged)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(0.30, name="fusion_drop_1")(x)

    x = layers.Dense(48, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_48")(x)
    x = layers.Dropout(0.20, name="fusion_drop_2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="shiro_gesture_output")(x)
    return tf.keras.Model(inputs=[vis_in, anat_in], outputs=out, name="Dual_Stream_Shiro")


def build_dual_stream_drishti_model(
    vis_dim: int = 960,
    anat_dim: int = 44,
    num_classes: int = 8,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Drishti Dual-Stream Model:
    Stream 1: Whole-Frame Facial / Body Context (960)
    Stream 2: Eye/Gaze Landmarks + Gaze Kinematic Excursion (anat_dim)
    """
    vis_in = layers.Input(shape=(vis_dim,), name="whole_frame_visual_input")
    anat_in = layers.Input(shape=(anat_dim,), name="drishti_anatomical_input")

    x_vis = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="vis_proj")(vis_in)
    x_vis = layers.Dropout(0.25, name="vis_drop")(x_vis)

    x_anat = layers.Dense(48, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="anat_proj")(anat_in)
    x_anat = layers.Dropout(0.20, name="anat_drop")(x_anat)

    merged = layers.concatenate([x_vis, x_anat], name="drishti_fusion_concat")
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_128")(merged)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(0.30, name="fusion_drop_1")(x)

    x = layers.Dense(48, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_48")(x)
    x = layers.Dropout(0.20, name="fusion_drop_2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="drishti_gesture_output")(x)
    return tf.keras.Model(inputs=[vis_in, anat_in], outputs=out, name="Dual_Stream_Drishti")


def build_dual_stream_pada_model(
    vis_dim: int = 960,
    anat_dim: int = 40,
    num_classes: int = 6,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Pada Dual-Stream Model:
    Stream 1: Whole-Frame Stance & Body Silhouette (960)
    Stream 2: Foot/Leg Landmarks + Heel Elevation Dynamics (anat_dim)
    """
    vis_in = layers.Input(shape=(vis_dim,), name="whole_frame_visual_input")
    anat_in = layers.Input(shape=(anat_dim,), name="pada_anatomical_input")

    x_vis = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="vis_proj")(vis_in)
    x_vis = layers.Dropout(0.25, name="vis_drop")(x_vis)

    x_anat = layers.Dense(48, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="anat_proj")(anat_in)
    x_anat = layers.Dropout(0.20, name="anat_drop")(x_anat)

    merged = layers.concatenate([x_vis, x_anat], name="pada_fusion_concat")
    x = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_96")(merged)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(0.30, name="fusion_drop_1")(x)

    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_32")(x)
    x = layers.Dropout(0.20, name="fusion_drop_2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="pada_gesture_output")(x)
    return tf.keras.Model(inputs=[vis_in, anat_in], outputs=out, name="Dual_Stream_Pada")


def build_anatomical_robust_router(
    vis_dim: int = 960,
    anat_dim: int = 12,
    num_classes: int = 4,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Anatomical Robust Category Router:
    Combines Whole-Frame Visual Context (960) with Anatomical Evidence (12).
    Eliminates wide full-body camera-framing misrouting.
    """
    vis_in = layers.Input(shape=(vis_dim,), name="router_vis_input")
    anat_in = layers.Input(shape=(anat_dim,), name="router_anat_input")

    x_vis = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="r_vis_proj")(vis_in)
    x_vis = layers.Dropout(0.15, name="r_vis_drop")(x_vis)

    merged = layers.concatenate([x_vis, anat_in], name="router_concat")
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="router_dense_32")(merged)
    x = layers.Dropout(0.20, name="router_drop")(x)
    out = layers.Dense(num_classes, activation="softmax", name="router_category_output")(x)

    return tf.keras.Model(inputs=[vis_in, anat_in], outputs=out, name="Anatomical_Robust_Router")
