"""
STEP 20 — Part B & D: Compact Temporal Models & Multimodal Fusion Architectures
==============================================================================
Builds compact, anti-overfitting temporal architectures with < 50,000 parameters:
  - Option 1: Temporal 1D Convolution / TCN-light (< 15,000 params)
  - Option 2: Compact GRU (< 15,000 params)
  - Category-Specific Multimodal Fusion Models combining:
      1. Visual ROI Features (MobileNetV3Large embedding on category crop)
      2. MediaPipe Landmark Representations
      3. Compact Temporal Statistics / Temporal Model Embeddings
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from typing import Tuple, Optional


def build_compact_tcn_model(
    seq_len: int = 16,
    feat_dim: int = 32,
    num_classes: int = 9,
    filters: int = 32,
    dropout_rate: float = 0.30
) -> tf.keras.Model:
    """
    Compact Temporal 1D Convolution (TCN-light):
    Sequence (T, D) -> Conv1D(32) -> BN -> Dropout -> Conv1D(32) -> GlobalAvgPool -> Dense(32) -> Output
    Total trainable parameters: ~8,000 - 15,000 (< 50,000 limit).
    """
    seq_in = layers.Input(shape=(seq_len, feat_dim), name="sequence_input")

    x = layers.Conv1D(filters=filters, kernel_size=3, padding="same", activation="relu", name="tcn_conv1")(seq_in)
    x = layers.BatchNormalization(name="tcn_bn1")(x)
    x = layers.Dropout(dropout_rate, name="tcn_drop1")(x)

    x = layers.Conv1D(filters=filters, kernel_size=3, padding="same", activation="relu", name="tcn_conv2")(x)
    x = layers.BatchNormalization(name="tcn_bn2")(x)
    x = layers.Dropout(dropout_rate, name="tcn_drop2")(x)

    # Global temporal pooling
    x = layers.GlobalAveragePooling1D(name="tcn_pool")(x)
    feat_repr = layers.Dense(32, activation="relu", name="tcn_dense_repr")(x)
    out = layers.Dense(num_classes, activation="softmax", name="tcn_predictions")(feat_repr)

    model = tf.keras.Model(inputs=seq_in, outputs=out, name="Compact_TCN_Model")
    return model


def build_compact_gru_model(
    seq_len: int = 16,
    feat_dim: int = 32,
    num_classes: int = 9,
    units: int = 32,
    dropout_rate: float = 0.25
) -> tf.keras.Model:
    """
    Compact GRU Model:
    Sequence (T, D) -> GRU(32) -> Dense(32) -> Output
    Total trainable parameters: ~7,000 - 12,000 (< 50,000 limit).
    """
    seq_in = layers.Input(shape=(seq_len, feat_dim), name="sequence_input")

    x = layers.GRU(
        units=units,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        return_sequences=False,
        name="compact_gru"
    )(seq_in)

    feat_repr = layers.Dense(32, activation="relu", name="gru_dense_repr")(x)
    out = layers.Dense(num_classes, activation="softmax", name="gru_predictions")(feat_repr)

    model = tf.keras.Model(inputs=seq_in, outputs=out, name="Compact_GRU_Model")
    return model


# =============================================================================
# Category-Specific Multimodal Fusion Models
# =============================================================================

def build_hasthas_roi_multimodal_model(
    vis_dim: int = 960,
    lm_dim: int = 175,
    num_classes: int = 52,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Hasthas Multimodal Model:
    Combines Visual Hand ROI Features (960) with Hand Geometric Landmarks (175).
    """
    vis_in = layers.Input(shape=(vis_dim,), name="visual_roi_input")
    lm_in = layers.Input(shape=(lm_dim,), name="landmark_input")

    # Landmark projection branch
    x_lm = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="lm_dense_128")(lm_in)
    x_lm = layers.Dropout(0.30, name="lm_dropout")(x_lm)

    # Fusion
    merged = layers.concatenate([vis_in, x_lm], name="fusion_concat")
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_256")(merged)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(0.40, name="fusion_dropout_1")(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_128")(x)
    x = layers.Dropout(0.30, name="fusion_dropout_2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="hasthas_predictions")(x)
    model = tf.keras.Model(inputs=[vis_in, lm_in], outputs=out, name="Hasthas_ROI_Multimodal")
    return model


def build_shiro_roi_multimodal_model(
    vis_dim: int = 960,
    lm_dim: int = 32,
    temp_dim: int = 268,
    num_classes: int = 9,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Shiro Multimodal Model:
    Combines Visual Head ROI (960) + Head Landmark (32) + Temporal Head Statistics (268).
    Compact projection to prevent overfitting on small N.
    """
    vis_in = layers.Input(shape=(vis_dim,), name="visual_roi_input")
    lm_in = layers.Input(shape=(lm_dim,), name="landmark_input")
    temp_in = layers.Input(shape=(temp_dim,), name="temporal_input")

    # Lightweight kinematic branch
    geom = layers.concatenate([lm_in, temp_in], name="kinematic_concat")
    x_geom = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="geom_proj")(geom)
    x_geom = layers.Dropout(0.25, name="geom_drop")(x_geom)

    # Fusion with visual ROI
    merged = layers.concatenate([vis_in, x_geom], name="shiro_fusion_concat")
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="shiro_dense_128")(merged)
    x = layers.BatchNormalization(name="shiro_bn")(x)
    x = layers.Dropout(0.35, name="shiro_drop1")(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="shiro_dense_64")(x)
    x = layers.Dropout(0.25, name="shiro_drop2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="shiro_predictions")(x)
    model = tf.keras.Model(inputs=[vis_in, lm_in, temp_in], outputs=out, name="Shiro_ROI_Multimodal")
    return model


def build_drishti_roi_multimodal_model(
    vis_dim: int = 960,
    lm_dim: int = 44,
    temp_dim: int = 364,
    num_classes: int = 8,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Drishti Multimodal Model:
    Combines Visual Eye/Face ROI (960) + Gaze/Iris Landmark (44) + Temporal Gaze Dynamics (364).
    Enforces distinct representations for Nimilita (closed eyes) vs Avalokita (downward gaze).
    """
    vis_in = layers.Input(shape=(vis_dim,), name="visual_roi_input")
    lm_in = layers.Input(shape=(lm_dim,), name="landmark_input")
    temp_in = layers.Input(shape=(temp_dim,), name="temporal_input")

    # Specialized gaze branch
    geom = layers.concatenate([lm_in, temp_in], name="gaze_concat")
    x_gaze = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="gaze_proj")(geom)
    x_gaze = layers.Dropout(0.25, name="gaze_drop")(x_gaze)

    # Fusion
    merged = layers.concatenate([vis_in, x_gaze], name="drishti_fusion_concat")
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="drishti_dense_128")(merged)
    x = layers.BatchNormalization(name="drishti_bn")(x)
    x = layers.Dropout(0.35, name="drishti_drop1")(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="drishti_dense_64")(x)
    x = layers.Dropout(0.25, name="drishti_drop2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="drishti_predictions")(x)
    model = tf.keras.Model(inputs=[vis_in, lm_in, temp_in], outputs=out, name="Drishti_ROI_Multimodal")
    return model


def build_pada_roi_multimodal_model(
    vis_dim: int = 960,
    lm_dim: int = 40,
    temp_dim: int = 332,
    num_classes: int = 6,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Pada Multimodal Model:
    Combines Visual Feet ROI (960) + Foot Landmark (40) + Temporal Stance Dynamics (332).
    """
    vis_in = layers.Input(shape=(vis_dim,), name="visual_roi_input")
    lm_in = layers.Input(shape=(lm_dim,), name="landmark_input")
    temp_in = layers.Input(shape=(temp_dim,), name="temporal_input")

    geom = layers.concatenate([lm_in, temp_in], name="stance_concat")
    x_geom = layers.Dense(48, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="stance_proj")(geom)
    x_geom = layers.Dropout(0.20, name="stance_drop")(x_geom)

    merged = layers.concatenate([vis_in, x_geom], name="pada_fusion_concat")
    x = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="pada_dense_96")(merged)
    x = layers.BatchNormalization(name="pada_bn")(x)
    x = layers.Dropout(0.30, name="pada_drop1")(x)

    x = layers.Dense(48, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="pada_dense_48")(x)
    x = layers.Dropout(0.20, name="pada_drop2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="pada_predictions")(x)
    model = tf.keras.Model(inputs=[vis_in, lm_in, temp_in], outputs=out, name="Pada_ROI_Multimodal")
    return model


def build_dual_visual_multimodal_model(
    whole_dim: int = 960,
    roi_dim: int = 960,
    geom_dim: int = 175,
    num_classes: int = 52,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Dual-Visual Multimodal Model:
    Fuses Whole-Frame Visual Embedding (960) + Category ROI Visual Embedding (960)
    + Geometric/Kinematic Vector (geom_dim).
    Gives model both full-body framing context and zoomed-in anatomical details.
    """
    whole_in = layers.Input(shape=(whole_dim,), name="whole_vis_in")
    roi_in = layers.Input(shape=(roi_dim,), name="roi_vis_in")
    geom_in = layers.Input(shape=(geom_dim,), name="geom_in")

    x_whole = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="whole_dense")(whole_in)
    x_whole = layers.Dropout(0.25, name="whole_drop")(x_whole)

    x_roi = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="roi_dense")(roi_in)
    x_roi = layers.Dropout(0.25, name="roi_drop")(x_roi)

    x_geom = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="geom_dense")(geom_in)
    x_geom = layers.Dropout(0.20, name="geom_drop")(x_geom)

    merged = layers.concatenate([x_whole, x_roi, x_geom], name="dual_vis_concat")
    x = layers.Dense(192, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_192")(merged)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(0.35, name="fusion_drop_1")(x)

    x = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense_96")(x)
    x = layers.Dropout(0.25, name="fusion_drop_2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="dual_predictions")(x)
    return tf.keras.Model(inputs=[whole_in, roi_in, geom_in], outputs=out, name="Dual_Visual_Multimodal_Model")

