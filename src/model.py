import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np

try:
    from config import (
        MODEL_NAME,
        NUM_CLASSES,
        CNN_INPUT_SHAPE,
        LEARNING_RATE,
        DROPOUT_RATE,
        DENSE_UNITS,
        L2_REGULARIZATION,
        LOSS_FUNCTION,
        OPTIMIZER_NAME,
        EARLY_STOPPING_PATIENCE,
        REDUCE_LR_PATIENCE,
        REDUCE_LR_FACTOR,
        MIN_LEARNING_RATE,
        BEST_MODEL_PATH,
        RANDOM_SEED
    )
except ImportError:
    from src.config import (
        MODEL_NAME,
        NUM_CLASSES,
        CNN_INPUT_SHAPE,
        LEARNING_RATE,
        DROPOUT_RATE,
        DENSE_UNITS,
        L2_REGULARIZATION,
        LOSS_FUNCTION,
        OPTIMIZER_NAME,
        EARLY_STOPPING_PATIENCE,
        REDUCE_LR_PATIENCE,
        REDUCE_LR_FACTOR,
        MIN_LEARNING_RATE,
        BEST_MODEL_PATH,
        RANDOM_SEED
    )

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks

def build_cnn_model(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
    dropout_rate: float = DROPOUT_RATE,
    dense_units: int = DENSE_UNITS,
    l2_reg: float = L2_REGULARIZATION,
    loss_type: str = LOSS_FUNCTION,
    model_name: str = MODEL_NAME
) -> tf.keras.Model:
    """
    Builds and compiles a tailored Convolutional Neural Network for Kuchipudi Dance Recognition.
    
    Architecture Design & Rationale:
    --------------------------------
    1. Small-to-Medium Dataset Adaptation:
       - The Kuchipudi dataset contains 75 classes with 1,094 total samples (~600-800 training samples).
       - Massive pre-existing architectures or unregularized deep networks with tens of millions of parameters
         risk catastrophic memorization (severe overfitting).
       - This compact 4-block CNN (~1.26M parameters) offers high spatial representational power while remaining
         tightly bounded to prevent parameter explosion.
         
    2. Deep Feature Hierarchy:
       - Block 1 (32 filters, 3x3): Captures low-level visual primitives (edges, contours of fingers, pupil boundaries).
       - Block 2 (64 filters, 3x3): Captures mid-level geometric primitives (finger curvatures, joint angles, eye shapes).
       - Block 3 (128 filters, 3x3): Captures complex sub-structures (complete hand mudras, eye glance directions, foot postures).
       - Block 4 (256 filters, 3x3): Captures high-level semantic dance abstractions across all 4 categories.
       
    3. Multi-Tiered Regularization:
       - Batch Normalization after every conv and dense layer stabilizes gradient flow and smooths the loss landscape.
       - Progressive Spatial Dropout (0.2 -> 0.25 -> 0.3 -> 0.35) prevents co-adaptation of feature detectors.
       - Global Average Pooling (GAP) reduces the 14x14x256 feature map directly to 256 features without flattening.
         * Flatten would have produced 14*14*256 = 50,176 features, requiring 12.8M parameters to connect to Dense(256).
         * GAP uses 0 parameters, provides translation invariance, and directly prevents overfitting.
       - L2 Weight Decay (1e-4) penalizes excessively large kernel weights.
       - Classifier Head Dropout (0.4) protects the dense decision boundary.
    """
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    inputs = layers.Input(shape=input_shape, name="dance_input")

    # =========================================================================
    # CONVOLUTION BLOCK 1 (Low-level edge & contour features)
    # Output: 224x224x3 -> 112x112x32
    # =========================================================================
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv1_1"
    )(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="act1_1")(x)

    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv1_2"
    )(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="act1_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.20, name="drop1")(x)

    # =========================================================================
    # CONVOLUTION BLOCK 2 (Mid-level shape & joint curvature features)
    # Output: 112x112x32 -> 56x56x64
    # =========================================================================
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv2_1"
    )(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="act2_1")(x)

    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv2_2"
    )(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="act2_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # =========================================================================
    # CONVOLUTION BLOCK 3 (High-level mudra & posture sub-structures)
    # Output: 56x56x64 -> 28x28x128
    # =========================================================================
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv3_1"
    )(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="act3_1")(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv3_2"
    )(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="act3_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="drop3")(x)

    # =========================================================================
    # CONVOLUTION BLOCK 4 (Semantic dance class representations)
    # Output: 28x28x128 -> 14x14x256
    # =========================================================================
    x = layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv4_1"
    )(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="act4_1")(x)

    x = layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv4_2"
    )(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu", name="act4_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(0.35, name="drop4")(x)

    # =========================================================================
    # GLOBAL AVERAGE POOLING (Drastic parameter reduction & spatial invariance)
    # Output: 14x14x256 -> 256 vector
    # =========================================================================
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # =========================================================================
    # CLASSIFIER HEAD (Dense representation & regularization)
    # Output: 256 -> 256 -> num_classes
    # =========================================================================
    x = layers.Dense(
        units=dense_units,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_features"
    )(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="act_dense")(x)
    x = layers.Dropout(dropout_rate, name="drop_dense")(x)

    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="kuchipudi_predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Configure Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Configure Loss and Metrics
    if loss_type == "categorical_crossentropy":
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    elif loss_type == "sparse_categorical_crossentropy":
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'categorical_crossentropy' or 'sparse_categorical_crossentropy'.")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def get_callbacks(
    checkpoint_path: Union[str, Path] = BEST_MODEL_PATH,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    reduce_lr_patience: int = REDUCE_LR_PATIENCE,
    reduce_lr_factor: float = REDUCE_LR_FACTOR,
    min_lr: float = MIN_LEARNING_RATE,
    monitor: str = "val_loss",
    mode: str = "min"
) -> List[tf.keras.callbacks.Callback]:
    """
    Constructs configurable training callbacks:
    1. EarlyStopping: Prevents overfitting by halting training when validation loss stops improving.
    2. ModelCheckpoint: Saves ONLY the best model weights based on validation performance.
    3. ReduceLROnPlateau: Dynamically decays learning rate when validation loss plateaus.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callback_list = [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode=mode
        ),
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            mode=mode
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
            mode=mode
        )
    ]
    return callback_list

def compute_balanced_class_weights(
    labels: Union[List[int], np.ndarray],
    num_classes: int = NUM_CLASSES
) -> Dict[int, float]:
    """
    Computes balanced class weights inversely proportional to class frequencies:
      weight = total_samples / (num_classes * class_count)
      
    Mitigates severe class imbalance between video-dense classes and minority image classes
    without corrupting the unaugmented validation/test sets.
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels_arr = np.array(labels, dtype=int)
    present_classes = np.unique(labels_arr)
    weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=labels_arr)

    # Initialize all num_classes with 1.0 default
    class_weight_dict = {i: 1.0 for i in range(num_classes)}
    for cls_idx, w in zip(present_classes, weights):
        class_weight_dict[int(cls_idx)] = float(w)

    return class_weight_dict
