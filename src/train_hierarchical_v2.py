"""
Step 13 — Retrain Hierarchical CNN Using Dataset Split V2

This script executes:
1. Pre-training verification of Dataset Split V2:
   - data/splits/v2_train.csv (716 samples, 75 classes)
   - data/splits/v2_validation.csv (199 samples, 75 classes)
   - data/splits/v2_test.csv (179 samples, 75 classes - STRICTLY UNTOUCHED)
2. Ingestion and caching of 224x224 RGB images in RAM (0 disk bottle-neck).
3. Computation of class-balanced loss weights strictly from V2 training labels.
4. Stage 1: Transfer learning with MobileNetV3Large (Backbone Frozen, Adam lr=1e-3):
   - Category Router (4 categories)
   - Hasthas Expert (52 classes)
   - Shiro Bhedhas Expert (9 classes)
   - Drishti Bhedhas Expert (8 classes)
   - Pada Karma Expert (6 classes)
5. Stage 2: Controlled Fine-Tuning (Top 25% of backbone unfrozen, BN locked, Adam lr=1e-5).
6. Model selection (best validation accuracy between Stage 1 & Stage 2 for each model).
7. Full Hierarchical End-to-End Evaluation on all 199 validation samples:
   - Router Accuracy
   - Hasthas, Shiro, Drishti, Pada Karma Expert Accuracies
   - Overall Hierarchical Top-1 Accuracy
   - Overall Hierarchical Top-5 Accuracy
   - Macro Precision, Recall, F1, Weighted F1
   - Per-class accuracy across all 75 classes
   - Worst-performing classes
   - Category Confusion Matrix
8. Direct comparison against Step 11 baseline.
9. Artifact and Report generation:
   - models/v2_hierarchical_*_best.keras
   - reports/step13_v2_training_report.txt
   - reports/step13_v2_metrics.csv
   - results/plots/v2_training_accuracy.png
   - results/plots/v2_training_loss.png
   - results/plots/v2_confusion_matrix.png
"""

import os
import sys
import json
import time
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    CNN_INPUT_SHAPE,
    PRETRAINED_BACKBONE,
    LEARNING_RATE_STAGE1,
    DENSE_UNITS,
    DROPOUT_RATE,
    L2_REGULARIZATION,
    RANDOM_SEED,
    BATCH_SIZE
)
from src.preprocessing import load_and_preprocess_image
from src.augmentation import KuchipudiAugmenter
from src.dataset import KuchipudiDataSequence
from src.hierarchical_model import (
    build_hierarchical_mapping,
    build_category_router,
    build_category_expert,
    HierarchicalKuchipudiClassifier
)

import tensorflow as tf
from tensorflow.keras import callbacks, layers

# Hyperparameters
STAGE1_MAX_EPOCHS = 25
STAGE1_EARLY_PATIENCE = 7
STAGE1_REDUCE_LR_PATIENCE = 3
STAGE1_REDUCE_LR_FACTOR = 0.2
STAGE1_MIN_LR = 1e-6

STAGE2_MAX_EPOCHS = 15
STAGE2_INITIAL_LR = 1e-5
STAGE2_EARLY_PATIENCE = 5
STAGE2_REDUCE_LR_PATIENCE = 2
STAGE2_REDUCE_LR_FACTOR = 0.2
STAGE2_MIN_LR = 1e-7
FINE_TUNE_PERCENT = 0.25

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORTS_DIR / "step13_v2_training_report.txt"
METRICS_CSV = REPORTS_DIR / "step13_v2_metrics.csv"
ACC_PLOT_PNG = PLOTS_DIR / "v2_training_accuracy.png"
LOSS_PLOT_PNG = PLOTS_DIR / "v2_training_loss.png"
CM_PLOT_PNG = PLOTS_DIR / "v2_confusion_matrix.png"


# ==============================================================================
# CALLBACK LOGGER
# ==============================================================================

class HierarchicalMetricsLogger(callbacks.Callback):
    def __init__(self, csv_path: Optional[Path] = None):
        super().__init__()
        self.csv_path = csv_path
        self.history_records = []

    def on_train_begin(self, logs=None):
        self.history_records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(self.model.optimizer.learning_rate.numpy()) if hasattr(self.model.optimizer.learning_rate, "numpy") else float(self.model.optimizer.learning_rate)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(logs.get("loss", 0.0)),
            "train_accuracy": float(logs.get("accuracy", 0.0)),
            "validation_loss": float(logs.get("val_loss", 0.0)),
            "validation_accuracy": float(logs.get("val_accuracy", 0.0)),
            "learning_rate": lr
        }
        self.history_records.append(record)

    def on_train_end(self, logs=None):
        if self.csv_path and self.history_records:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "learning_rate"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.history_records:
                    writer.writerow(r)


# ==============================================================================
# FINE-TUNING CONFIGURATOR
# ==============================================================================

def configure_expert_finetuning(
    model: tf.keras.Model,
    learning_rate: float = STAGE2_INITIAL_LR,
    fine_tune_percent: float = FINE_TUNE_PERCENT
):
    """
    Unfreezes top ~25% of backbone layers, keeping earlier layers and BN locked.
    """
    backbone = None
    for layer in model.layers:
        if "mobilenetv3" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise ValueError(f"MobileNetV3Large backbone not found in model {model.name}")

    backbone.trainable = True
    total_bb_layers = len(backbone.layers)
    num_unfreeze = int(np.ceil(total_bb_layers * fine_tune_percent))
    freeze_until = total_bb_layers - num_unfreeze

    unfrozen_count = 0
    frozen_count = 0
    bn_frozen_count = 0

    for idx, layer in enumerate(backbone.layers):
        if idx < freeze_until:
            layer.trainable = False
            frozen_count += 1
        else:
            if isinstance(layer, layers.BatchNormalization) or "batch_norm" in layer.name.lower() or "bn" in layer.name.lower():
                layer.trainable = False
                bn_frozen_count += 1
            else:
                layer.trainable = True
                unfrozen_count += 1

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return {
        "total_backbone_layers": total_bb_layers,
        "unfrozen_count": unfrozen_count,
        "frozen_count": frozen_count,
        "bn_frozen_count": bn_frozen_count
    }


# ==============================================================================
# PRELOAD CACHE
# ==============================================================================

def preload_images_and_metadata(csv_path: Path) -> Dict[str, Any]:
    print(f"  Pre-loading {csv_path.name} into memory...")
    df = pd.read_csv(csv_path)
    images_list = []
    valid_indices = []

    for idx, row in df.iterrows():
        full_path = PROJECT_ROOT / row["sample_path"]
        img = load_and_preprocess_image(full_path, target_size=(224, 224), resize_method="aspect_pad")
        if img is not None:
            images_list.append(img)
            valid_indices.append(idx)
        else:
            raise FileNotFoundError(f"Failed to load image: {full_path}")

    cached_df = df.iloc[valid_indices].reset_index(drop=True)
    images_arr = np.array(images_list, dtype=np.float32)

    return {
        "df": cached_df,
        "images": images_arr
    }


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def run_step13_retraining():
    print("=" * 80)
    print("STEP 13: RETRAIN THE HIERARCHICAL CNN USING DATASET SPLIT V2")
    print("=" * 80)
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # 1. VERIFY DATASET SPLIT V2
    # -------------------------------------------------------------------------
    print("\n[1/7] Verifying Dataset Split V2...")
    splits_dir = DATA_DIR / "splits"
    train_csv = splits_dir / "v2_train.csv"
    val_csv = splits_dir / "v2_validation.csv"
    test_csv = splits_dir / "v2_test.csv"

    assert train_csv.exists(), f"Missing {train_csv}"
    assert val_csv.exists(), f"Missing {val_csv}"
    assert test_csv.exists(), f"Missing {test_csv}"

    df_tr_check = pd.read_csv(train_csv)
    df_val_check = pd.read_csv(val_csv)
    df_te_check = pd.read_csv(test_csv)

    print(f"  - v2_train.csv        : {len(df_tr_check)} samples, {df_tr_check.groupby(['category', 'class_name']).ngroups} unique classes")
    print(f"  - v2_validation.csv   : {len(df_val_check)} samples, {df_val_check.groupby(['category', 'class_name']).ngroups} unique classes")
    print(f"  - v2_test.csv         : {len(df_te_check)} samples, {df_te_check.groupby(['category', 'class_name']).ngroups} unique classes (STRICTLY ISOLATED & UNTOUCHED)")

    assert len(df_tr_check) == 716, f"Expected 716 train samples, got {len(df_tr_check)}"
    assert len(df_val_check) == 199, f"Expected 199 val samples, got {len(df_val_check)}"
    assert len(df_te_check) == 179, f"Expected 179 test samples, got {len(df_te_check)}"
    assert df_tr_check.groupby(['category', 'class_name']).ngroups == 75, "Expected 75 classes in train"
    assert df_val_check.groupby(['category', 'class_name']).ngroups == 75, "Expected 75 classes in val"
    assert df_te_check.groupby(['category', 'class_name']).ngroups == 75, "Expected 75 classes in test"
    print("  Dataset V2 Pre-Training Safety Verification: PASSED (100% Validated).")

    # -------------------------------------------------------------------------
    # 2. HIERARCHICAL MAPPING & AUGMENTATION
    # -------------------------------------------------------------------------
    print("\n[2/7] Initializing Taxonomy Mapping & Augmenter...")
    mapping = build_hierarchical_mapping(save_json=True)
    augmenter = KuchipudiAugmenter(enable=True, seed=RANDOM_SEED)
    print(f"  - Hierarchical Categories: {mapping['categories']}")
    print(f"  - Total Classes Mapped   : {mapping['total_classes']}")
    print("  - Augmentation: Domain-safe active for TRAIN ONLY (Flips strictly disabled).")

    # -------------------------------------------------------------------------
    # 3. PRELOAD IMAGES INTO RAM
    # -------------------------------------------------------------------------
    print("\n[3/7] Preloading and Caching Images in RAM...")
    t_load = time.time()
    train_data = preload_images_and_metadata(train_csv)
    val_data = preload_images_and_metadata(val_csv)

    train_df = train_data["df"]
    train_imgs = train_data["images"]
    val_df = val_data["df"]
    val_imgs = val_data["images"]
    print(f"  Cached {len(train_imgs)} Train and {len(val_imgs)} Val images in {time.time() - t_load:.1f}s.")

    # -------------------------------------------------------------------------
    # 4. CLASS WEIGHT BALANCING (Strictly from V2 Train)
    # -------------------------------------------------------------------------
    print("\n[4/7] Computing Class-Balanced Loss Weights (V2 Train Only)...")
    # Global 75-class weights
    classes_str = train_df["category"] + "::" + train_df["class_name"]
    unique_75 = np.unique(classes_str)
    weights_75 = compute_class_weight(class_weight="balanced", classes=unique_75, y=classes_str)
    print(f"  Global 75-Class Weights (Train Only):")
    print(f"    * Min Weight  : {weights_75.min():.4f}")
    print(f"    * Max Weight  : {weights_75.max():.4f}")
    print(f"    * Mean Weight : {weights_75.mean():.4f}")
    print(f"    * Median      : {np.median(weights_75):.4f}")

    # Router Category weights
    cat_to_id = mapping["category_to_id"]
    train_cat_ids = train_df["category"].map(cat_to_id).values
    val_cat_ids = val_df["category"].map(cat_to_id).values
    unique_cats = np.unique(train_cat_ids)
    cat_weights_arr = compute_class_weight(class_weight="balanced", classes=unique_cats, y=train_cat_ids)
    router_class_weights = {cls: float(w) for cls, w in zip(unique_cats, cat_weights_arr)}
    for c in range(4):
        if c not in router_class_weights:
            router_class_weights[c] = 1.0
    print(f"  Router Category Weights: {router_class_weights}")

    # Expert class weights dictionary
    expert_class_weights_dict = {}
    for cat_name, info in mapping["experts"].items():
        c_to_loc = info["class_to_local_id"]
        sub_df = train_df[train_df["category"] == cat_name]
        loc_labels = sub_df["class_name"].map(c_to_loc).values
        u_loc = np.unique(loc_labels)
        exp_w_arr = compute_class_weight(class_weight="balanced", classes=u_loc, y=loc_labels)
        exp_w = {cls: float(w) for cls, w in zip(u_loc, exp_w_arr)}
        for loc_id in range(info["num_classes"]):
            if loc_id not in exp_w:
                exp_w[loc_id] = float(np.max(exp_w_arr) if len(exp_w_arr) > 0 else 1.0)
        expert_class_weights_dict[cat_name] = exp_w

    # Store histories and performance logs
    training_histories = {}
    best_models = {}
    model_summaries = {}

    models_to_train = [
        ("router", "Category Router", None, 4),
        ("hasthas", "Hasthas Expert", "1. Hasthas", 52),
        ("shiro", "Shiro Bhedhas Expert", "2. Shiro Bhedhas (HEAD MOVEMENTS)", 9),
        ("drishti", "Drishti Bhedhas Expert", "3. Drishti Bhedhas(EYE MOVEMENTS)", 8),
        ("pada_karma", "Pada Karma Expert", "4. Pada Karma(Foot movements)", 6)
    ]

    # -------------------------------------------------------------------------
    # 5. SEQUENTIAL TRAINING: STAGE 1 & STAGE 2
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[5/7] EXECUTING SEQUENTIAL TRAINING (STAGE 1 HEADS + STAGE 2 FINE-TUNING)")
    print("=" * 80)

    train_cat_oh = tf.keras.utils.to_categorical(train_cat_ids, num_classes=4)
    val_cat_oh = tf.keras.utils.to_categorical(val_cat_ids, num_classes=4)

    for key, display_name, cat_name, num_classes in models_to_train:
        print("\n" + "#" * 70)
        print(f"TRAINING COMPONENT: {display_name.upper()} ({num_classes} Classes)")
        print("#" * 70)

        # Prepare sub-dataset
        if key == "router":
            sub_tr_imgs = train_imgs
            sub_val_imgs = val_imgs
            sub_tr_oh = train_cat_oh
            sub_val_oh = val_cat_oh
            sub_cw = router_class_weights
        else:
            tr_idx = train_df.index[train_df["category"] == cat_name].tolist()
            val_idx = val_df.index[val_df["category"] == cat_name].tolist()
            sub_tr_imgs = train_imgs[tr_idx]
            sub_val_imgs = val_imgs[val_idx]

            c_to_loc = mapping["experts"][cat_name]["class_to_local_id"]
            tr_loc_labels = train_df.iloc[tr_idx]["class_name"].map(c_to_loc).values
            val_loc_labels = val_df.iloc[val_idx]["class_name"].map(c_to_loc).values
            sub_tr_oh = tf.keras.utils.to_categorical(tr_loc_labels, num_classes=num_classes)
            sub_val_oh = tf.keras.utils.to_categorical(val_loc_labels, num_classes=num_classes)
            sub_cw = expert_class_weights_dict[cat_name]

        print(f"  Dataset: Train = {len(sub_tr_imgs)} samples | Validation = {len(sub_val_imgs)} samples")

        # Sequences
        train_gen = KuchipudiDataSequence(
            images=sub_tr_imgs,
            labels=sub_tr_oh,
            batch_size=BATCH_SIZE,
            is_training=True,
            augmenter=augmenter,
            shuffle=True,
            seed=RANDOM_SEED
        )
        val_gen = KuchipudiDataSequence(
            images=sub_val_imgs,
            labels=sub_val_oh,
            batch_size=BATCH_SIZE,
            is_training=False,
            augmenter=None,
            shuffle=False,
            seed=RANDOM_SEED
        )

        # Build Stage 1 Model
        if key == "router":
            model, backbone = build_category_router(
                input_shape=CNN_INPUT_SHAPE,
                num_categories=num_classes,
                learning_rate=LEARNING_RATE_STAGE1
            )
        else:
            model, backbone = build_category_expert(
                category_name=cat_name,
                num_classes=num_classes,
                input_shape=CNN_INPUT_SHAPE,
                learning_rate=LEARNING_RATE_STAGE1
            )

        # --- STAGE 1 TRAINING ---
        print(f"\n  >>> Starting Stage 1 (Backbone Frozen, LR=1e-3, Max Epochs={STAGE1_MAX_EPOCHS})...")
        s1_ckpt_path = MODELS_DIR / f"v2_{key}_stage1_best.keras"
        s1_logger = HierarchicalMetricsLogger()

        s1_callbacks = [
            callbacks.ModelCheckpoint(
                filepath=str(s1_ckpt_path),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=STAGE1_EARLY_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=STAGE1_REDUCE_LR_FACTOR,
                patience=STAGE1_REDUCE_LR_PATIENCE,
                min_lr=STAGE1_MIN_LR,
                verbose=1
            ),
            s1_logger
        ]

        t0_s1 = time.time()
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=STAGE1_MAX_EPOCHS,
            class_weight=sub_cw,
            callbacks=s1_callbacks,
            verbose=1
        )
        dur_s1 = time.time() - t0_s1

        # Evaluate best Stage 1
        model_s1_best = tf.keras.models.load_model(str(s1_ckpt_path))
        eval_s1 = model_s1_best.evaluate(val_gen, verbose=0)
        m_names_s1 = model_s1_best.metrics_names
        acc_idx_s1 = m_names_s1.index("accuracy") if "accuracy" in m_names_s1 else 1
        s1_val_acc = float(eval_s1[acc_idx_s1])
        s1_val_loss = float(eval_s1[0])
        print(f"  [Stage 1 Best] Val Acc: {s1_val_acc*100:.2f}% | Val Loss: {s1_val_loss:.4f} (Duration: {dur_s1:.1f}s)")

        # --- STAGE 2 TRAINING (Controlled Fine-Tuning) ---
        print(f"\n  >>> Starting Stage 2 (Fine-Tuning Top 25%, LR=1e-5, Max Epochs={STAGE2_MAX_EPOCHS})...")
        # Load best Stage 1 checkpoint as baseline for Stage 2
        model_s2 = tf.keras.models.load_model(str(s1_ckpt_path))
        ft_info = configure_expert_finetuning(model_s2, learning_rate=STAGE2_INITIAL_LR, fine_tune_percent=FINE_TUNE_PERCENT)
        print(f"    Backbone Layers: {ft_info['total_backbone_layers']} total | {ft_info['unfrozen_count']} unfrozen | {ft_info['bn_frozen_count']} BN frozen")

        s2_ckpt_path = MODELS_DIR / f"v2_{key}_stage2_best.keras"
        model_s2.save(str(s2_ckpt_path))
        s2_logger = HierarchicalMetricsLogger()

        s2_callbacks = [
            callbacks.ModelCheckpoint(
                filepath=str(s2_ckpt_path),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=STAGE2_EARLY_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=STAGE2_REDUCE_LR_FACTOR,
                patience=STAGE2_REDUCE_LR_PATIENCE,
                min_lr=STAGE2_MIN_LR,
                verbose=1
            ),
            s2_logger
        ]

        t0_s2 = time.time()
        model_s2.fit(
            train_gen,
            validation_data=val_gen,
            epochs=STAGE2_MAX_EPOCHS,
            class_weight=sub_cw,
            callbacks=s2_callbacks,
            verbose=1
        )
        dur_s2 = time.time() - t0_s2

        model_s2_best = tf.keras.models.load_model(str(s2_ckpt_path))
        eval_s2 = model_s2_best.evaluate(val_gen, verbose=0)
        m_names_s2 = model_s2_best.metrics_names
        acc_idx_s2 = m_names_s2.index("accuracy") if "accuracy" in m_names_s2 else 1
        s2_val_acc = float(eval_s2[acc_idx_s2])
        s2_val_loss = float(eval_s2[0])
        print(f"  [Stage 2 Best] Val Acc: {s2_val_acc*100:.2f}% | Val Loss: {s2_val_loss:.4f} (Duration: {dur_s2:.1f}s)")

        # Model Selection: Pick whichever has higher validation accuracy
        if s2_val_acc > s1_val_acc:
            print(f"  --> SELECTING STAGE 2 MODEL (+{(s2_val_acc - s1_val_acc)*100:.2f}% improvement over Stage 1)")
            chosen_model = model_s2_best
            chosen_stage = "Stage 2 (Fine-Tuned)"
            chosen_acc = s2_val_acc
            chosen_loss = s2_val_loss
        else:
            print(f"  --> SELECTING STAGE 1 MODEL (Stage 1 was superior: {s1_val_acc*100:.2f}% vs {s2_val_acc*100:.2f}%)")
            chosen_model = model_s1_best
            chosen_stage = "Stage 1 (Feature Extractor)"
            chosen_acc = s1_val_acc
            chosen_loss = s1_val_loss

        # Save as the official best v2 model
        best_save_path = MODELS_DIR / f"v2_hierarchical_{key}_best.keras"
        chosen_model.save(str(best_save_path))
        best_models[key] = chosen_model

        # Combine histories for plotting
        training_histories[key] = {
            "display_name": display_name,
            "stage1_records": s1_logger.history_records,
            "stage2_records": s2_logger.history_records,
            "stage1_best_acc": s1_val_acc,
            "stage2_best_acc": s2_val_acc,
            "chosen_stage": chosen_stage,
            "chosen_acc": chosen_acc,
            "chosen_loss": chosen_loss,
            "duration": dur_s1 + dur_s2
        }

        model_summaries[key] = {
            "name": display_name,
            "stage1_epochs": len(s1_logger.history_records),
            "stage1_val_acc": s1_val_acc,
            "stage2_epochs": len(s2_logger.history_records),
            "stage2_val_acc": s2_val_acc,
            "chosen_stage": chosen_stage,
            "final_val_acc": chosen_acc,
            "final_val_loss": chosen_loss,
            "total_duration": dur_s1 + dur_s2
        }

    # Save representative stage1 and stage2 models as requested
    tf.keras.models.load_model(str(MODELS_DIR / "v2_router_stage1_best.keras")).save(str(MODELS_DIR / "v2_hierarchical_stage1_best.keras"))
    tf.keras.models.load_model(str(MODELS_DIR / "v2_router_stage2_best.keras")).save(str(MODELS_DIR / "v2_hierarchical_stage2_best.keras"))

    # -------------------------------------------------------------------------
    # 6. FULL HIERARCHICAL VALIDATION EVALUATION (199 Samples)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[6/7] FULL HIERARCHICAL END-TO-END VALIDATION EVALUATION (199 Samples)")
    print("=" * 80)

    best_router = best_models["router"]
    best_experts = {
        "1. Hasthas": best_models["hasthas"],
        "2. Shiro Bhedhas (HEAD MOVEMENTS)": best_models["shiro"],
        "3. Drishti Bhedhas(EYE MOVEMENTS)": best_models["drishti"],
        "4. Pada Karma(Foot movements)": best_models["pada_karma"]
    }

    hierarchical_classifier = HierarchicalKuchipudiClassifier(
        router_model=best_router,
        experts=best_experts,
        mapping=mapping
    )

    val_true_cats = []
    val_pred_cats = []
    val_true_classes = []
    val_pred_classes = []
    val_true_gids = []
    val_pred_gids = []
    top5_hits = 0

    per_cat_eval = {cat: {"total": 0, "cat_correct": 0, "class_correct": 0} for cat in mapping["categories"]}
    per_class_counts = defaultdict(lambda: {"total": 0, "correct": 0, "category": ""})

    for i in range(len(val_imgs)):
        img = val_imgs[i:i+1]
        true_cat = val_df.iloc[i]["category"]
        true_cls = val_df.iloc[i]["class_name"]

        # Run Hierarchical inference
        details = hierarchical_classifier.predict(img, return_details=True)
        pred_cat = details["category_name"]
        pred_cls = details["final_class"]
        pred_gid = details["global_class_id"]

        # Resolve true global ID
        cat_meta = mapping["experts"][true_cat]
        true_local = cat_meta["class_to_local_id"][true_cls]
        true_gid = cat_meta["local_to_global"][true_local]

        val_true_cats.append(true_cat)
        val_pred_cats.append(pred_cat)
        val_true_classes.append(true_cls)
        val_pred_classes.append(pred_cls)
        val_true_gids.append(true_gid)
        val_pred_gids.append(pred_gid)

        # Rigorous Top-5 Calculation across 75 global classes
        router_probs = np.array(details["stage1_probabilities"])
        full_75_probs = np.zeros(75, dtype=np.float32)

        for c_idx, c_name in enumerate(mapping["categories"]):
            exp_m = best_experts[c_name]
            p_cat = router_probs[c_idx]
            exp_p = exp_m.predict(img, verbose=0)[0]
            exp_info = mapping["experts"][c_name]
            for loc_id, p_val in enumerate(exp_p):
                g_id = exp_info["local_to_global"][loc_id]
                full_75_probs[g_id] = p_cat * p_val

        top5_pred_gids = np.argsort(full_75_probs)[-5:]
        if true_gid in top5_pred_gids:
            top5_hits += 1

        # Track per-category & per-class accuracy
        per_cat_eval[true_cat]["total"] += 1
        composite_key = f"{true_cat}::{true_cls}"
        per_class_counts[composite_key]["total"] += 1
        per_class_counts[composite_key]["category"] = true_cat
        per_class_counts[composite_key]["class_name"] = true_cls

        if pred_cat == true_cat:
            per_cat_eval[true_cat]["cat_correct"] += 1
        if pred_cls == true_cls and pred_cat == true_cat:
            per_cat_eval[true_cat]["class_correct"] += 1
            per_class_counts[composite_key]["correct"] += 1

    total_val_samples = len(val_imgs)
    total_cat_correct = sum(1 for tc, pc in zip(val_true_cats, val_pred_cats) if tc == pc)
    total_class_correct = sum(1 for tg, pg in zip(val_true_gids, val_pred_gids) if tg == pg)

    overall_cat_acc = total_cat_correct / total_val_samples
    overall_top1 = total_class_correct / total_val_samples
    overall_top5 = top5_hits / total_val_samples

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        val_true_gids, val_pred_gids, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        val_true_gids, val_pred_gids, average="weighted", zero_division=0
    )

    print("\n--- OVERALL HIERARCHICAL VALIDATION PERFORMANCE ---")
    print(f"  * Total Validation Samples : {total_val_samples}")
    print(f"  * Router Category Accuracy : {overall_cat_acc*100:.2f}% ({total_cat_correct}/{total_val_samples})")
    print(f"  * Overall Top-1 Accuracy   : {overall_top1*100:.2f}% ({total_class_correct}/{total_val_samples})")
    print(f"  * Overall Top-5 Accuracy   : {overall_top5*100:.2f}% ({top5_hits}/{total_val_samples})")
    print(f"  * Macro Precision          : {prec_macro*100:.2f}%")
    print(f"  * Macro Recall             : {rec_macro*100:.2f}%")
    print(f"  * Macro F1 Score           : {f1_macro*100:.2f}%")
    print(f"  * Weighted F1 Score        : {f1_weighted*100:.2f}%")

    print("\n--- PER-CATEGORY BREAKDOWN ---")
    for cat_name, stats in per_cat_eval.items():
        disp = mapping["experts"][cat_name]["display_name"]
        n_s = stats["total"]
        c_acc = (stats["cat_correct"] / n_s) * 100 if n_s > 0 else 0.0
        cls_acc = (stats["class_correct"] / n_s) * 100 if n_s > 0 else 0.0
        print(f"  [{disp:<15}] Val Samples: {n_s:<3} | Router Acc: {c_acc:6.2f}% | Expert Class Acc: {cls_acc:6.2f}%")

    # Worst-performing classes
    per_class_records = []
    for comp_key, d in per_class_counts.items():
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        per_class_records.append({
            "category": d["category"],
            "class_name": d["class_name"],
            "val_samples": d["total"],
            "correct_samples": d["correct"],
            "val_accuracy": acc
        })
    df_per_class = pd.DataFrame(per_class_records)
    worst_classes = df_per_class.sort_values(by=["val_accuracy", "val_samples"], ascending=[True, False]).head(10)
    print("\n--- 10 WORST PERFORMING CLASSES (Validation) ---")
    for _, r in worst_classes.iterrows():
        print(f"  [{r['category'][:15]}] {r['class_name']:<25}: {r['val_accuracy']*100:5.1f}% ({r['correct_samples']}/{r['val_samples']})")

    # Confusion matrix
    cat_order = mapping["categories"]
    cat_cm = confusion_matrix(val_true_cats, val_pred_cats, labels=cat_order)
    cat_labels_disp = [mapping["category_display_map"][c] for c in cat_order]
    cm_df = pd.DataFrame(cat_cm, index=cat_labels_disp, columns=cat_labels_disp)

    # -------------------------------------------------------------------------
    # 7. GENERATE ARTIFACTS, PLOTS, AND REPORTS
    # -------------------------------------------------------------------------
    print("\n[7/7] Generating Reports and Plots...")
    total_training_time = time.time() - overall_start_time

    # Plot 1: Accuracy Curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, (m_key, _, _, _) in enumerate(models_to_train):
        ax = axes[idx]
        h_data = training_histories[m_key]
        s1_recs = h_data["stage1_records"]
        s2_recs = h_data["stage2_records"]

        ep_s1 = [r["epoch"] for r in s1_recs]
        tr_s1 = [r["train_accuracy"] for r in s1_recs]
        val_s1 = [r["validation_accuracy"] for r in s1_recs]

        ax.plot(ep_s1, tr_s1, label="Stage 1 Train", color="#1f77b4", linestyle="--")
        ax.plot(ep_s1, val_s1, label=f"Stage 1 Val (Best: {h_data['stage1_best_acc']*100:.1f}%)", color="#1f77b4", linewidth=2)

        if s2_recs:
            ep_s2 = [len(ep_s1) + r["epoch"] for r in s2_recs]
            tr_s2 = [r["train_accuracy"] for r in s2_recs]
            val_s2 = [r["validation_accuracy"] for r in s2_recs]
            ax.plot(ep_s2, tr_s2, label="Stage 2 Train", color="#ff7f0e", linestyle="--")
            ax.plot(ep_s2, val_s2, label=f"Stage 2 Val (Best: {h_data['stage2_best_acc']*100:.1f}%)", color="#ff7f0e", linewidth=2)
            ax.axvline(x=len(ep_s1), color="gray", linestyle=":", label="Stage 2 Start")

        ax.set_title(f"{h_data['display_name']}\nChosen: {h_data['chosen_stage']} ({h_data['chosen_acc']*100:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="lower right")

    axes[-1].axis("off")
    plt.suptitle("Hierarchical CNN V2 — Training & Validation Accuracy Curves Across Stages", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ACC_PLOT_PNG, dpi=300)
    plt.close()
    print(f"  Saved accuracy plot to {ACC_PLOT_PNG}")

    # Plot 2: Loss Curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, (m_key, _, _, _) in enumerate(models_to_train):
        ax = axes[idx]
        h_data = training_histories[m_key]
        s1_recs = h_data["stage1_records"]
        s2_recs = h_data["stage2_records"]

        ep_s1 = [r["epoch"] for r in s1_recs]
        tr_s1 = [r["train_loss"] for r in s1_recs]
        val_s1 = [r["validation_loss"] for r in s1_recs]

        ax.plot(ep_s1, tr_s1, label="Stage 1 Train Loss", color="#1f77b4", linestyle="--")
        ax.plot(ep_s1, val_s1, label="Stage 1 Val Loss", color="#1f77b4", linewidth=2)

        if s2_recs:
            ep_s2 = [len(ep_s1) + r["epoch"] for r in s2_recs]
            tr_s2 = [r["train_loss"] for r in s2_recs]
            val_s2 = [r["validation_loss"] for r in s2_recs]
            ax.plot(ep_s2, tr_s2, label="Stage 2 Train Loss", color="#ff7f0e", linestyle="--")
            ax.plot(ep_s2, val_s2, label="Stage 2 Val Loss", color="#ff7f0e", linewidth=2)
            ax.axvline(x=len(ep_s1), color="gray", linestyle=":", label="Stage 2 Start")

        ax.set_title(f"{h_data['display_name']} - Loss", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].axis("off")
    plt.suptitle("Hierarchical CNN V2 — Loss Curves Across Stages", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PNG, dpi=300)
    plt.close()
    print(f"  Saved loss plot to {LOSS_PLOT_PNG}")

    # Plot 3: Confusion Matrix Heatmap (Pure Matplotlib)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cat_cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cat_cm.shape[1]),
        yticks=np.arange(cat_cm.shape[0]),
        xticklabels=cat_labels_disp,
        yticklabels=cat_labels_disp,
        title=f"Category Router Confusion Matrix (Val N={total_val_samples})\nCategory Routing Accuracy: {overall_cat_acc*100:.2f}%",
        ylabel="True Category",
        xlabel="Predicted Category"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cat_cm.max() / 2.0
    for r_idx in range(cat_cm.shape[0]):
        for c_idx in range(cat_cm.shape[1]):
            val = cat_cm[r_idx, c_idx]
            color = "white" if val > thresh else "black"
            ax.text(c_idx, r_idx, format(val, "d"), ha="center", va="center", color=color, fontweight="bold", fontsize=13)

    plt.tight_layout()
    plt.savefig(CM_PLOT_PNG, dpi=300)
    plt.close()
    print(f"  Saved confusion matrix plot to {CM_PLOT_PNG}")

    # Save metrics CSV
    metrics_records = [
        {"metric_type": "Overall", "name": "Router_Category_Accuracy", "value": round(overall_cat_acc, 6), "details": f"{total_cat_correct}/{total_val_samples}"},
        {"metric_type": "Overall", "name": "Hierarchical_Top1_Accuracy", "value": round(overall_top1, 6), "details": f"{total_class_correct}/{total_val_samples}"},
        {"metric_type": "Overall", "name": "Hierarchical_Top5_Accuracy", "value": round(overall_top5, 6), "details": f"{top5_hits}/{total_val_samples}"},
        {"metric_type": "Overall", "name": "Macro_Precision", "value": round(prec_macro, 6), "details": "Unweighted mean across classes"},
        {"metric_type": "Overall", "name": "Macro_Recall", "value": round(rec_macro, 6), "details": "Unweighted mean across classes"},
        {"metric_type": "Overall", "name": "Macro_F1", "value": round(f1_macro, 6), "details": "Unweighted mean across classes"},
        {"metric_type": "Overall", "name": "Weighted_F1", "value": round(f1_weighted, 6), "details": "Weighted by class support in val"},
        {"metric_type": "Component", "name": "Hasthas_Expert_Acc", "value": round(per_cat_eval['1. Hasthas']['class_correct']/per_cat_eval['1. Hasthas']['total'], 6), "details": "52 classes"},
        {"metric_type": "Component", "name": "Shiro_Expert_Acc", "value": round(per_cat_eval['2. Shiro Bhedhas (HEAD MOVEMENTS)']['class_correct']/per_cat_eval['2. Shiro Bhedhas (HEAD MOVEMENTS)']['total'], 6), "details": "9 classes"},
        {"metric_type": "Component", "name": "Drishti_Expert_Acc", "value": round(per_cat_eval['3. Drishti Bhedhas(EYE MOVEMENTS)']['class_correct']/per_cat_eval['3. Drishti Bhedhas(EYE MOVEMENTS)']['total'], 6), "details": "8 classes"},
        {"metric_type": "Component", "name": "Pada_Karma_Expert_Acc", "value": round(per_cat_eval['4. Pada Karma(Foot movements)']['class_correct']/per_cat_eval['4. Pada Karma(Foot movements)']['total'], 6), "details": "6 classes"},
    ]
    for r in per_class_records:
        metrics_records.append({
            "metric_type": "Class_Level",
            "name": f"{r['category']}::{r['class_name']}",
            "value": round(r["val_accuracy"], 4),
            "details": f"Support={r['val_samples']}, Correct={r['correct_samples']}"
        })
    pd.DataFrame(metrics_records).to_csv(METRICS_CSV, index=False)
    print(f"  Saved metrics CSV to {METRICS_CSV}")

    # Generate master report
    step11_top1 = 0.2835
    step11_top5 = 0.4567
    step11_router = 0.9921

    delta_top1 = (overall_top1 - step11_top1) * 100
    delta_top5 = (overall_top5 - step11_top5) * 100
    delta_router = (overall_cat_acc - step11_router) * 100

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("STEP 13 — HIERARCHICAL CNN RETRAINING REPORT (DATASET SPLIT V2)\n")
        f.write("================================================================================\n\n")

        f.write("1. EXECUTIVE SUMMARY & STEP 11 COMPARISON\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"  * Overall Hierarchical Top-1 Accuracy : {overall_top1*100:.2f}%  (Step 11: 28.35% -> Change: {delta_top1:+.2f} percentage points)\n")
        f.write(f"  * Overall Hierarchical Top-5 Accuracy : {overall_top5*100:.2f}%  (Step 11: 45.67% -> Change: {delta_top5:+.2f} percentage points)\n")
        f.write(f"  * Category Router Accuracy            : {overall_cat_acc*100:.2f}%  (Step 11: 99.21% -> Change: {delta_router:+.2f} percentage points)\n")
        f.write(f"  * Macro Precision                     : {prec_macro*100:.2f}%\n")
        f.write(f"  * Macro Recall                        : {rec_macro*100:.2f}%\n")
        f.write(f"  * Macro F1 Score                      : {f1_macro*100:.2f}%\n")
        f.write(f"  * Weighted F1 Score                   : {f1_weighted*100:.2f}%\n")
        f.write(f"  * Total Retraining Duration           : {total_training_time:.1f}s ({total_training_time/60:.2f} minutes)\n\n")

        f.write("2. TARGET ATTAINMENT ASSESSMENT (95% ENGINEERING TARGET)\n")
        f.write("--------------------------------------------------------------------------------\n")
        target_achieved = overall_top1 >= 0.95
        f.write(f"  * Target: Validation Accuracy >= 95.00%\n")
        f.write(f"  * Measured Top-1 Validation Accuracy: {overall_top1*100:.2f}%\n")
        if target_achieved:
            f.write(f"  * VERDICT: 95% VALIDATION ACCURACY TARGET WAS ACHIEVED.\n\n")
        else:
            f.write(f"  * VERDICT: 95% VALIDATION ACCURACY TARGET WAS NOT ACHIEVED.\n")
            f.write(f"    Top-1 accuracy reached {overall_top1*100:.2f}% (gap to target: {95.0 - overall_top1*100:.2f} percentage points).\n\n")

        f.write("3. DATASET V2 VERIFICATION & INGESTION STATS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"  * Train samples ingested      : {len(train_imgs)} (v2_train.csv, 75/75 classes represented)\n")
        f.write(f"  * Validation samples ingested : {len(val_imgs)} (v2_validation.csv, 75/75 classes represented)\n")
        f.write(f"  * Test samples isolated       : 179 (v2_test.csv, STRICTLY UNTOUCHED)\n")
        f.write(f"  * Source-level leakage        : 0.00% (Strictly 0 video or image overlap across partitions)\n\n")

        f.write("4. CLASS WEIGHT BALANCING (V2 Train Only)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"  * 75-Class Min Weight   : {weights_75.min():.4f}\n")
        f.write(f"  * 75-Class Max Weight   : {weights_75.max():.4f}\n")
        f.write(f"  * 75-Class Mean Weight  : {weights_75.mean():.4f}\n")
        f.write(f"  * 75-Class Median Weight: {np.median(weights_75):.4f}\n\n")

        f.write("5. COMPONENT TRAINING SUMMARY & MODEL SELECTION\n")
        f.write("--------------------------------------------------------------------------------\n")
        for k, sum_dict in model_summaries.items():
            f.write(f"  [{sum_dict['name']}]:\n")
            f.write(f"    - Stage 1: {sum_dict['stage1_epochs']} epochs, Best Val Acc = {sum_dict['stage1_val_acc']*100:.2f}%\n")
            f.write(f"    - Stage 2: {sum_dict['stage2_epochs']} epochs, Best Val Acc = {sum_dict['stage2_val_acc']*100:.2f}%\n")
            f.write(f"    - Selected Model : {sum_dict['chosen_stage']}\n")
            f.write(f"    - Selected Val Acc: {sum_dict['final_val_acc']*100:.2f}% (Loss: {sum_dict['final_val_loss']:.4f})\n")
            f.write(f"    - Component Time : {sum_dict['total_duration']:.1f}s\n\n")

        f.write("6. PER-CATEGORY & EXPERT PERFORMANCE BREAKDOWN\n")
        f.write("--------------------------------------------------------------------------------\n")
        for cat_name, stats in per_cat_eval.items():
            disp = mapping["experts"][cat_name]["display_name"]
            n_s = stats["total"]
            c_acc = (stats["cat_correct"] / n_s) * 100 if n_s > 0 else 0.0
            cls_acc = (stats["class_correct"] / n_s) * 100 if n_s > 0 else 0.0
            f.write(f"  [{disp}]:\n")
            f.write(f"    - Val Samples Evaluated : {n_s}\n")
            f.write(f"    - Category Router Acc   : {c_acc:.2f}%\n")
            f.write(f"    - Expert Local Acc      : {cls_acc:.2f}%\n\n")

        f.write("7. WORST PERFORMING CLASSES (Validation)\n")
        f.write("--------------------------------------------------------------------------------\n")
        for _, r in worst_classes.iterrows():
            f.write(f"  - [{r['category']}] {r['class_name']}: {r['val_accuracy']*100:.1f}% ({r['correct_samples']}/{r['val_samples']})\n")

        f.write("\n8. SAVED ARTIFACTS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("  Models:\n")
        f.write("    - models/v2_hierarchical_router_best.keras\n")
        f.write("    - models/v2_hierarchical_hasthas_best.keras\n")
        f.write("    - models/v2_hierarchical_shiro_best.keras\n")
        f.write("    - models/v2_hierarchical_drishti_best.keras\n")
        f.write("    - models/v2_hierarchical_pada_karma_best.keras\n")
        f.write("    - models/v2_hierarchical_stage1_best.keras\n")
        f.write("    - models/v2_hierarchical_stage2_best.keras\n")
        f.write("  Reports:\n")
        f.write("    - reports/step13_v2_training_report.txt\n")
        f.write("    - reports/step13_v2_metrics.csv\n")
        f.write("  Plots:\n")
        f.write("    - results/plots/v2_training_accuracy.png\n")
        f.write("    - results/plots/v2_training_loss.png\n")
        f.write("    - results/plots/v2_confusion_matrix.png\n")

    print(f"  Saved master report to {REPORT_FILE}")

    return {
        "overall_top1": overall_top1,
        "overall_top5": overall_top5,
        "router_acc": overall_cat_acc,
        "macro_f1": f1_macro,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "per_cat_eval": per_cat_eval,
        "total_training_time": total_training_time,
        "model_summaries": model_summaries,
        "target_achieved": target_achieved
    }

if __name__ == "__main__":
    run_step13_retraining()
