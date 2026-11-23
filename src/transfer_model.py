import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union
import numpy as np

try:
    from config import (
        PROJECT_ROOT,
        TRANSFER_MODEL_NAME,
        PRETRAINED_BACKBONE,
        NUM_CLASSES,
        CNN_INPUT_SHAPE,
        LEARNING_RATE_STAGE1,
        LEARNING_RATE_STAGE2,
        DENSE_UNITS,
        DROPOUT_RATE,
        L2_REGULARIZATION,
        FINE_TUNE_PERCENTAGE,
        LOSS_FUNCTION,
        TRANSFER_BEST_MODEL_PATH,
        RANDOM_SEED
    )
except ImportError:
    from src.config import (
        PROJECT_ROOT,
        TRANSFER_MODEL_NAME,
        PRETRAINED_BACKBONE,
        NUM_CLASSES,
        CNN_INPUT_SHAPE,
        LEARNING_RATE_STAGE1,
        LEARNING_RATE_STAGE2,
        DENSE_UNITS,
        DROPOUT_RATE,
        L2_REGULARIZATION,
        FINE_TUNE_PERCENTAGE,
        LOSS_FUNCTION,
        TRANSFER_BEST_MODEL_PATH,
        RANDOM_SEED
    )

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks

def get_backbone(backbone_name: str, input_shape: Tuple[int, int, int]):
    """
    Instantiates the designated pre-trained CNN backbone with ImageNet weights.
    """
    name_clean = backbone_name.lower().replace("-", "").replace("_", "")
    if "mobilenetv3large" in name_clean:
        return tf.keras.applications.MobileNetV3Large(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
    elif "mobilenetv3small" in name_clean:
        return tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
    elif "efficientnetb0" in name_clean:
        return tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
    elif "resnet50" in name_clean:
        return tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Choose 'MobileNetV3Large', 'MobileNetV3Small', or 'EfficientNetB0'.")

def build_transfer_model(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE,
    backbone_name: str = PRETRAINED_BACKBONE,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE_STAGE1,
    l2_reg: float = L2_REGULARIZATION,
    loss_type: str = LOSS_FUNCTION,
    model_name: str = TRANSFER_MODEL_NAME
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds and compiles the Stage-1 Transfer Learning CNN (Backbone Frozen).
    
    Architecture:
      Input (224x224x3)
        ↓
      Pretrained Backbone (ImageNet weights, Frozen)
        ↓
      Global Average Pooling 2D
        ↓
      Dense (256, He-normal, L2)
        ↓
      Batch Normalization
        ↓
      Activation (ReLU)
        ↓
      Dropout (0.40)
        ↓
      Dense (num_classes, Softmax)
    """
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # 1. Instantiate pretrained backbone
    backbone = get_backbone(backbone_name, input_shape)

    # 2. Freeze backbone for Stage 1 (Feature Extraction)
    backbone.trainable = False

    # 3. Construct Model
    inputs = layers.Input(shape=input_shape, name="dance_input")

    # Pass inputs through backbone (training=False preserves inference BN statistics)
    x = backbone(inputs, training=False)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense Classification Head
    x = layers.Dense(
        units=dense_units,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_features"
    )(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="act_dense")(x)
    x = layers.Dropout(dropout_rate, name="drop_dense")(x)

    # 75-class softmax output
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="kuchipudi_predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Configure Optimizer and Loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if loss_type == "categorical_crossentropy":
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy")]
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model, backbone

def configure_stage2_finetuning(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    fine_tune_percentage: float = FINE_TUNE_PERCENTAGE,
    learning_rate: float = LEARNING_RATE_STAGE2,
    loss_type: str = LOSS_FUNCTION
) -> Dict[str, Any]:
    """
    Unfreezes the top 20-30% of backbone layers for controlled Stage-2 fine-tuning.
    Freezes earlier layers to preserve basic low-level edge/texture primitives.
    Keeps Batch Normalization layers in the backbone frozen to prevent statistic drift on small batches.
    """
    # Unfreeze backbone
    backbone.trainable = True

    total_backbone_layers = len(backbone.layers)
    num_unfreeze = int(np.ceil(total_backbone_layers * fine_tune_percentage))
    freeze_until = total_backbone_layers - num_unfreeze

    unfrozen_count = 0
    frozen_count = 0
    bn_kept_frozen = 0

    for idx, layer in enumerate(backbone.layers):
        if idx < freeze_until:
            layer.trainable = False
            frozen_count += 1
        else:
            # Best practice for fine-tuning on small datasets: keep BN layers frozen
            if isinstance(layer, (layers.BatchNormalization, tf.keras.layers.Layer)) and "batch_norm" in layer.name.lower():
                layer.trainable = False
                bn_kept_frozen += 1
            else:
                layer.trainable = True
                unfrozen_count += 1

    # Re-compile with low learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if loss_type == "categorical_crossentropy":
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy")]
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    total_params = model.count_params()
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    non_trainable_params = sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)

    return {
        "total_backbone_layers": total_backbone_layers,
        "freeze_threshold_index": freeze_until,
        "frozen_backbone_layers": frozen_count,
        "unfrozen_backbone_layers": unfrozen_count,
        "bn_layers_kept_frozen": bn_kept_frozen,
        "stage2_learning_rate": learning_rate,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params
    }

def get_transfer_callbacks(
    checkpoint_path: Union[str, Path] = TRANSFER_BEST_MODEL_PATH,
    monitor: str = "val_accuracy",
    mode: str = "max",
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 3,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6
) -> List[tf.keras.callbacks.Callback]:
    """
    Returns callbacks optimized for validation accuracy maximization.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            mode=mode,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode=mode,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            mode="min",
            verbose=1
        )
    ]
