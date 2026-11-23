import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

try:
    from config import (
        PROJECT_ROOT,
        DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        NUM_CLASSES,
        CNN_INPUT_SHAPE,
        PRETRAINED_BACKBONE,
        LEARNING_RATE_STAGE1,
        DENSE_UNITS,
        DROPOUT_RATE,
        L2_REGULARIZATION,
        LOSS_FUNCTION,
        RANDOM_SEED,
        NUM_CATEGORIES,
        CATEGORY_NAMES,
        CATEGORY_DISPLAY_NAMES,
        CATEGORY_CLASS_COUNTS,
        HIERARCHICAL_MODEL_NAME,
        HIERARCHICAL_REPORT_TXT,
        HIERARCHICAL_MAPPING_JSON
    )
except ImportError:
    from src.config import (
        PROJECT_ROOT,
        DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        NUM_CLASSES,
        CNN_INPUT_SHAPE,
        PRETRAINED_BACKBONE,
        LEARNING_RATE_STAGE1,
        DENSE_UNITS,
        DROPOUT_RATE,
        L2_REGULARIZATION,
        LOSS_FUNCTION,
        RANDOM_SEED,
        NUM_CATEGORIES,
        CATEGORY_NAMES,
        CATEGORY_DISPLAY_NAMES,
        CATEGORY_CLASS_COUNTS,
        HIERARCHICAL_MODEL_NAME,
        HIERARCHICAL_REPORT_TXT,
        HIERARCHICAL_MAPPING_JSON
    )

import tensorflow as tf
from tensorflow.keras import layers, regularizers


# ==============================================================================
# 1. HIERARCHICAL CLASS MAPPING GENERATOR
# ==============================================================================

def build_hierarchical_mapping(save_json: bool = True) -> Dict[str, Any]:
    """
    Builds a robust, collision-free hierarchical mapping:
      category -> category_id -> class_name -> local_class_id & global_class_id
    
    Extracts mappings strictly from existing data/processed/class_mapping.json
    and preserves exact spelling, casing, and folder names.
    
    Disambiguates duplicate class names across categories:
      - '2. Shiro Bhedhas (HEAD MOVEMENTS)' -> '1. Sama' (Global ID: 52, Local ID: 0)
      - '3. Drishti Bhedhas(EYE MOVEMENTS)' -> '1. Sama' (Global ID: 61, Local ID: 0)
      - '4. Pada Karma(Foot movements)'     -> '2. Sama' (Global ID: 70, Local ID: 1)
    """
    source_mapping_path = PROCESSED_DATA_DIR / "class_mapping.json"
    if not source_mapping_path.exists():
        raise FileNotFoundError(f"Missing master class mapping: {source_mapping_path}")

    with open(source_mapping_path, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    classes_dict = master_data["classes"]

    # Categories list matching the canonical dataset order
    canonical_categories = [
        "1. Hasthas",
        "2. Shiro Bhedhas (HEAD MOVEMENTS)",
        "3. Drishti Bhedhas(EYE MOVEMENTS)",
        "4. Pada Karma(Foot movements)"
    ]

    category_to_id = {cat: idx for idx, cat in enumerate(canonical_categories)}
    id_to_category = {idx: cat for idx, cat in enumerate(canonical_categories)}

    category_display_map = {
        "1. Hasthas": "Hasthas",
        "2. Shiro Bhedhas (HEAD MOVEMENTS)": "Shiro Bhedhas",
        "3. Drishti Bhedhas(EYE MOVEMENTS)": "Drishti Bhedhas",
        "4. Pada Karma(Foot movements)": "Pada Karma"
    }

    # Group classes by category preserving global class ID ordering
    category_classes_grouped: Dict[str, List[Dict[str, Any]]] = {
        cat: [] for cat in canonical_categories
    }

    for global_id_str, info in sorted(classes_dict.items(), key=lambda x: int(x[0])):
        global_id = int(global_id_str)
        cat = info["category"]
        cls_name = info["class_name"]

        if cat not in category_classes_grouped:
            raise ValueError(f"Encountered unexpected category '{cat}' for class '{cls_name}'")

        category_classes_grouped[cat].append({
            "global_class_id": global_id,
            "class_name": cls_name,
            "full_name": info["full_name"],
            "composite_key": f"{cat}::{cls_name}"
        })

    # Build hierarchical expert metadata
    experts_meta: Dict[str, Any] = {}
    global_id_to_hierarchical: Dict[int, Dict[str, Any]] = {}
    total_classes_counted = 0

    for cat in canonical_categories:
        cat_id = category_to_id[cat]
        class_list = category_classes_grouped[cat]
        num_classes = len(class_list)
        total_classes_counted += num_classes

        local_classes = []
        class_to_local_id = {}
        local_id_to_class = {}
        local_to_global = {}
        global_to_local = {}

        for local_id, item in enumerate(class_list):
            cls_name = item["class_name"]
            gid = item["global_class_id"]
            comp_key = item["composite_key"]

            class_info = {
                "local_class_id": local_id,
                "global_class_id": gid,
                "class_name": cls_name,
                "composite_key": comp_key
            }
            local_classes.append(class_info)
            class_to_local_id[cls_name] = local_id
            local_id_to_class[local_id] = cls_name
            local_to_global[local_id] = gid
            global_to_local[gid] = local_id

            global_id_to_hierarchical[gid] = {
                "category": cat,
                "category_id": cat_id,
                "local_class_id": local_id,
                "class_name": cls_name,
                "composite_key": comp_key
            }

        experts_meta[cat] = {
            "category_id": cat_id,
            "display_name": category_display_map[cat],
            "num_classes": num_classes,
            "classes": local_classes,
            "class_to_local_id": class_to_local_id,
            "local_id_to_class": local_id_to_class,
            "local_to_global": local_to_global,
            "global_to_local": global_to_local
        }

    # Verify counts
    expected_counts = {"1. Hasthas": 52, "2. Shiro Bhedhas (HEAD MOVEMENTS)": 9, "3. Drishti Bhedhas(EYE MOVEMENTS)": 8, "4. Pada Karma(Foot movements)": 6}
    for cat, exp_cnt in expected_counts.items():
        actual_cnt = experts_meta[cat]["num_classes"]
        assert actual_cnt == exp_cnt, f"Mismatch in {cat}: expected {exp_cnt}, got {actual_cnt}"

    assert total_classes_counted == 75, f"Expected 75 total classes, found {total_classes_counted}"

    hierarchical_mapping = {
        "num_categories": len(canonical_categories),
        "total_classes": total_classes_counted,
        "categories": canonical_categories,
        "category_to_id": category_to_id,
        "id_to_category": {str(k): v for k, v in id_to_category.items()},
        "category_display_map": category_display_map,
        "experts": experts_meta,
        "global_id_to_hierarchical": {str(k): v for k, v in global_id_to_hierarchical.items()}
    }

    if save_json:
        out_path = HIERARCHICAL_MAPPING_JSON
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hierarchical_mapping, f, indent=2, ensure_ascii=False)

    return hierarchical_mapping


# ==============================================================================
# 2. BACKBONE / FEATURE EXTRACTOR FACTORY
# ==============================================================================

def get_backbone(
    backbone_name: str = PRETRAINED_BACKBONE,
    input_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE,
    frozen: bool = True
) -> tf.keras.Model:
    """
    Instantiates MobileNetV3Large backbone with official ImageNet weights.
    When frozen=True, backbone parameters are locked for pure feature extraction.
    """
    name_clean = backbone_name.lower().replace("-", "").replace("_", "")
    if "mobilenetv3large" in name_clean:
        backbone = tf.keras.applications.MobileNetV3Large(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
    else:
        raise ValueError(f"Backbone '{backbone_name}' not supported. Expected 'MobileNetV3Large'.")

    if frozen:
        backbone.trainable = False

    return backbone


# ==============================================================================
# 3. STAGE 1 CATEGORY ROUTER CNN
# ==============================================================================

def build_category_router(
    backbone: Optional[tf.keras.Model] = None,
    input_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE,
    num_categories: int = NUM_CATEGORIES,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE_STAGE1,
    l2_reg: float = L2_REGULARIZATION,
    loss_type: str = LOSS_FUNCTION
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds the Stage-1 Category Router CNN (Backbone Frozen).
    
    Architecture:
      Input (224x224x3)
        ↓
      MobileNetV3Large Backbone (ImageNet weights, Frozen, training=False)
        ↓
      GlobalAveragePooling2D
        ↓
      Dense (256, ReLU, He-normal, L2=1e-4)
        ↓
      BatchNormalization
        ↓
      Dropout (0.40)
        ↓
      Dense (4 categories, Softmax)
      
    Outputs:
      0 = 1. Hasthas
      1 = 2. Shiro Bhedhas (HEAD MOVEMENTS)
      2 = 3. Drishti Bhedhas(EYE MOVEMENTS)
      3 = 4. Pada Karma(Foot movements)
    """
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    if backbone is None:
        backbone = get_backbone(PRETRAINED_BACKBONE, input_shape, frozen=True)
    else:
        backbone.trainable = False

    inputs = layers.Input(shape=input_shape, name="router_image_input")
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="router_global_pool")(x)
    x = layers.Dense(
        units=dense_units,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="router_dense"
    )(x)
    x = layers.BatchNormalization(name="router_bn")(x)
    x = layers.Activation("relu", name="router_relu")(x)
    x = layers.Dropout(dropout_rate, name="router_dropout")(x)

    category_output = layers.Dense(
        units=num_categories,
        activation="softmax",
        name="category_predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=category_output, name="Stage1_Category_Router")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if loss_type == "categorical_crossentropy":
        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )

    return model, backbone


# ==============================================================================
# 4. STAGE 2 CATEGORY EXPERT CNN FACTORY
# ==============================================================================

def build_category_expert(
    category_name: str,
    num_classes: int,
    backbone: Optional[tf.keras.Model] = None,
    input_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE_STAGE1,
    l2_reg: float = L2_REGULARIZATION,
    loss_type: str = LOSS_FUNCTION
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds a Category-Specific Expert CNN (Backbone Frozen).
    
    Architecture:
      Input (224x224x3)
        ↓
      MobileNetV3Large Backbone (ImageNet weights, Frozen, training=False)
        ↓
      GlobalAveragePooling2D
        ↓
      Dense (256, ReLU, He-normal, L2=1e-4)
        ↓
      BatchNormalization
        ↓
      Dropout (0.40)
        ↓
      Dense (num_classes, Softmax)
    """
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # Sanitize prefix for layer names
    clean_prefix = category_name.split(".")[1].strip().split("(")[0].strip().replace(" ", "_")

    if backbone is None:
        backbone = get_backbone(PRETRAINED_BACKBONE, input_shape, frozen=True)
    else:
        backbone.trainable = False

    inputs = layers.Input(shape=input_shape, name=f"{clean_prefix}_image_input")
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name=f"{clean_prefix}_global_pool")(x)
    x = layers.Dense(
        units=dense_units,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{clean_prefix}_dense"
    )(x)
    x = layers.BatchNormalization(name=f"{clean_prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{clean_prefix}_relu")(x)
    x = layers.Dropout(dropout_rate, name=f"{clean_prefix}_dropout")(x)

    expert_output = layers.Dense(
        units=num_classes,
        activation="softmax",
        name=f"{clean_prefix}_predictions"
    )(x)

    model_name = f"Stage2_Expert_{clean_prefix}"
    model = tf.keras.Model(inputs=inputs, outputs=expert_output, name=model_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if loss_type == "categorical_crossentropy":
        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # Top-k metric: for small heads (< 5 classes), k must be <= num_classes
    top_k = min(5, num_classes)
    metrics = ["accuracy"]
    if top_k > 1 and num_classes > 5:
        if loss_type == "categorical_crossentropy":
            metrics.append(tf.keras.metrics.TopKCategoricalAccuracy(k=top_k, name=f"top_{top_k}_accuracy"))
        else:
            metrics.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(k=top_k, name=f"top_{top_k}_accuracy"))

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model, backbone


def build_all_hierarchical_models(
    shared_backbone: bool = False
) -> Dict[str, Any]:
    """
    Builds the Stage-1 Router and all four Stage-2 Expert models.
    Returns dictionary with all instantiated models.
    """
    mapping = build_hierarchical_mapping(save_json=True)

    backbone_shared = get_backbone(PRETRAINED_BACKBONE, CNN_INPUT_SHAPE, frozen=True) if shared_backbone else None

    # Stage 1 Router (4 categories)
    router_model, router_backbone = build_category_router(
        backbone=backbone_shared,
        input_shape=CNN_INPUT_SHAPE,
        num_categories=mapping["num_categories"]
    )

    # Stage 2 Experts
    experts = {}
    for cat_name, info in mapping["experts"].items():
        exp_model, exp_backbone = build_category_expert(
            category_name=cat_name,
            num_classes=info["num_classes"],
            backbone=backbone_shared if shared_backbone else None,
            input_shape=CNN_INPUT_SHAPE
        )
        experts[cat_name] = exp_model

    return {
        "mapping": mapping,
        "router": router_model,
        "experts": experts,
        "shared_backbone": shared_backbone
    }


# ==============================================================================
# 5. HIERARCHICAL INFERENCE PIPELINE
# ==============================================================================

class HierarchicalKuchipudiClassifier:
    """
    End-to-End Hierarchical Kuchipudi Dance Classifier.
    
    Inference Flow:
      predict(image):
        1. category_id, category_name, category_confidence = Router.predict(image)
        2. Select Expert for category_name:
             if Hasthas       -> HasthasExpert (52 classes)
             if Shiro Bhedhas -> ShiroExpert (9 classes)
             if Drishti       -> DrishtiExpert (8 classes)
             if Pada Karma    -> PadaKarmaExpert (6 classes)
        3. local_class_id, class_name, expert_confidence = Expert.predict(image)
        4. Resolve global_class_id and composite key
        5. Return category, final_class, confidence (and full diagnostic details)
    """

    def __init__(
        self,
        router_model: tf.keras.Model,
        experts: Dict[str, tf.keras.Model],
        mapping: Optional[Dict[str, Any]] = None
    ):
        self.router = router_model
        self.experts = experts
        self.mapping = mapping if mapping is not None else build_hierarchical_mapping(save_json=False)

    def _prepare_input(self, image: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
        """
        Validates and standardizes input tensor to shape (Batch, 224, 224, 3) in [0.0, 1.0].
        """
        if isinstance(image, tf.Tensor):
            arr = image.numpy()
        else:
            arr = np.array(image, dtype=np.float32)

        # Handle single unbatched image (224, 224, 3)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)

        if arr.shape[1:] != CNN_INPUT_SHAPE:
            raise ValueError(f"Input shape {arr.shape[1:]} must match CNN_INPUT_SHAPE {CNN_INPUT_SHAPE}")

        # Ensure normalized to [0, 1] if values appear to be in [0, 255]
        if np.max(arr) > 1.5:
            arr = arr / 255.0

        return arr

    def predict(
        self,
        image: Union[np.ndarray, tf.Tensor],
        return_details: bool = False
    ) -> Union[Tuple[str, str, float], Dict[str, Any]]:
        """
        Hierarchical inference on input image.
        
        Args:
          image: ndarray of shape (1, 224, 224, 3) or (224, 224, 3)
          return_details: If True, returns rich diagnostic dict with raw probability vectors
        
        Returns:
          If return_details=False:
            Tuple of (category_name, final_class_name, expert_confidence)
          If return_details=True:
            Dictionary containing:
              - category_id
              - category_name
              - category_display_name
              - category_confidence (Stage 1 probability)
              - stage1_probabilities (raw 4-element probability vector)
              - local_class_id
              - final_class (class_name)
              - global_class_id
              - expert_confidence (Stage 2 probability)
              - stage2_probabilities (raw probability vector for the expert)
              - composite_key
              - confidence (expert_confidence)
        """
        x = self._prepare_input(image)

        # Stage 1: Route Category
        stage1_probs = self.router.predict(x, verbose=0)[0]
        cat_id = int(np.argmax(stage1_probs))
        cat_confidence = float(stage1_probs[cat_id])
        cat_name = self.mapping["categories"][cat_id]
        cat_display = self.mapping["category_display_map"][cat_name]

        # Stage 2: Route to Category-Specific Expert
        if cat_name not in self.experts:
            raise KeyError(f"No expert model registered for category: '{cat_name}'")

        expert_model = self.experts[cat_name]
        stage2_probs = expert_model.predict(x, verbose=0)[0]
        local_id = int(np.argmax(stage2_probs))
        expert_confidence = float(stage2_probs[local_id])

        # Resolve Class Details
        expert_info = self.mapping["experts"][cat_name]
        class_meta = expert_info["classes"][local_id]
        final_class = class_meta["class_name"]
        global_class_id = class_meta["global_class_id"]
        composite_key = class_meta["composite_key"]

        if return_details:
            return {
                "category_id": cat_id,
                "category_name": cat_name,
                "category_display_name": cat_display,
                "category_confidence": cat_confidence,
                "stage1_probabilities": stage1_probs.tolist(),
                "local_class_id": local_id,
                "final_class": final_class,
                "global_class_id": global_class_id,
                "expert_confidence": expert_confidence,
                "stage2_probabilities": stage2_probs.tolist(),
                "composite_key": composite_key,
                "confidence": expert_confidence
            }

        return cat_display, final_class, expert_confidence


# ==============================================================================
# 6. CLASS WEIGHT CALCULATOR FOR HIERARCHICAL EXPERTS
# ==============================================================================

def compute_expert_class_weights(
    train_df: pd.DataFrame,
    category_name: str,
    mapping: Optional[Dict[str, Any]] = None
) -> Dict[int, float]:
    """
    Computes balanced class weights ONLY from the training partition for a specific category.
    Strictly forbids looking at validation or test partitions.
    """
    if mapping is None:
        mapping = build_hierarchical_mapping(save_json=False)

    expert_meta = mapping["experts"][category_name]
    num_classes = expert_meta["num_classes"]
    class_to_local = expert_meta["class_to_local_id"]

    # Filter training partition to this category
    cat_df = train_df[train_df["category"] == category_name]

    if len(cat_df) == 0:
        # Fallback if partition has no samples for this category
        return {i: 1.0 for i in range(num_classes)}

    local_labels = cat_df["class_name"].map(class_to_local).dropna().astype(int).values

    unique_classes = np.unique(local_labels)
    all_classes = np.arange(num_classes)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=local_labels
    )
    weight_dict = {cls: float(w) for cls, w in zip(unique_classes, weights)}

    # Ensure all local classes have a defined weight
    for c in all_classes:
        if c not in weight_dict:
            weight_dict[c] = float(np.max(weights) if len(weights) > 0 else 1.0)

    return weight_dict
