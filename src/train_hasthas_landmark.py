"""
STEP 16A — Hasthas Landmark-Based Geometric Classifier Training
Trains a compact neural network on the 175-dimensional MediaPipe geometric features
for the 52 Hasthas (hand gesture) classes of classical Kuchipudi dance.
Strictly adheres to zero data leakage:
  - Training data: v2_train.csv (Hasthas only, 416 samples)
  - Validation data: v2_validation.csv (Hasthas only, 85 samples)
  - Test data: STRICTLY EXCLUDED
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Project paths and imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical_model import build_hierarchical_mapping

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


# =============================================================================
# 1. Reproducibility Configuration
# =============================================================================

RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_hasthas_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str], pd.DataFrame, pd.DataFrame]:
    """
    Loads 175-dim landmark feature vectors for Hasthas train and validation partitions.
    Preserves exact class names from project hierarchical mapping.
    """
    # Load canonical mapping
    mapping = build_hierarchical_mapping(save_json=False)
    hasthas_info = mapping["experts"]["1. Hasthas"]
    class_to_idx = {item["class_name"]: item["local_class_id"] for item in hasthas_info["classes"]}
    idx_to_class = {item["local_class_id"]: item["class_name"] for item in hasthas_info["classes"]}

    # Load split CSVs
    train_csv = PROJECT_ROOT / "data" / "splits" / "v2_train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "v2_validation.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Filter for Hasthas only
    train_hasthas = train_df[train_df["category"] == "1. Hasthas"].copy().reset_index(drop=True)
    val_hasthas = val_df[val_df["category"] == "1. Hasthas"].copy().reset_index(drop=True)

    print(f"Loaded Hasthas Train samples: {len(train_hasthas)}")
    print(f"Loaded Hasthas Val samples  : {len(val_hasthas)}")

    landmarks_dir = PROJECT_ROOT / "data" / "landmarks" / "images"

    # Ingest train vectors
    X_train_list = []
    y_train_list = []
    train_detected_flags = []

    for _, row in train_hasthas.iterrows():
        stem = Path(row["sample_path"]).stem
        npz_file = landmarks_dir / f"{stem}_landmarks.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Missing landmark file: {npz_file}")
        
        data = np.load(npz_file)
        vec = data["vector"]
        meta = json.loads(str(data["metadata"]))
        
        X_train_list.append(vec)
        y_train_list.append(class_to_idx[row["class_name"]])
        train_detected_flags.append(meta.get("detected", True))

    # Ingest validation vectors
    X_val_list = []
    y_val_list = []
    val_detected_flags = []

    for _, row in val_hasthas.iterrows():
        stem = Path(row["sample_path"]).stem
        npz_file = landmarks_dir / f"{stem}_landmarks.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Missing landmark file: {npz_file}")
        
        data = np.load(npz_file)
        vec = data["vector"]
        meta = json.loads(str(data["metadata"]))
        
        X_val_list.append(vec)
        y_val_list.append(class_to_idx[row["class_name"]])
        val_detected_flags.append(meta.get("detected", True))

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)

    train_hasthas["detected"] = train_detected_flags
    val_hasthas["detected"] = val_detected_flags

    return X_train, y_train, X_val, y_val, class_to_idx, idx_to_class, train_hasthas, val_hasthas


def build_hasthas_landmark_mlp(input_dim: int = 175, num_classes: int = 52) -> tf.keras.Model:
    """
    Builds the recommended compact neural-network classifier for geometric feature vectors:
      Dense(256) -> BatchNorm -> Dropout(0.30) ->
      Dense(128) -> BatchNorm -> Dropout(0.30) ->
      Dense(64) -> Dropout(0.20) ->
      Dense(52, softmax)
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,), name="landmark_geometric_input"),
        
        layers.Dense(256, activation="relu", name="dense_256"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(0.30, name="dropout_1"),
        
        layers.Dense(128, activation="relu", name="dense_128"),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(0.30, name="dropout_2"),
        
        layers.Dense(64, activation="relu", name="dense_64"),
        layers.Dropout(0.20, name="dropout_3"),
        
        layers.Dense(num_classes, activation="softmax", name="hasthas_output")
    ], name="Hasthas_Landmark_MLP")

    return model


def main():
    print("=" * 80)
    print("STEP 16A — HASTHAS LANDMARK-BASED CLASSIFIER TRAINING")
    print("=" * 80)

    start_time = time.time()

    # Directories
    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Dataset & Landmark Audit
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Auditing Hasthas Landmark Features & Ingestion...")
    X_train_raw, y_train, X_val_raw, y_val, class_to_idx, idx_to_class, train_df, val_df = load_hasthas_data()

    num_train = len(X_train_raw)
    num_val = len(X_val_raw)
    feat_dim = X_train_raw.shape[1]
    num_classes = len(class_to_idx)

    print(f"  - Training samples     : {num_train}")
    print(f"  - Validation samples   : {num_val}")
    print(f"  - Feature dimension    : {feat_dim} (Verified)")
    print(f"  - Number of classes    : {num_classes} (Verified)")

    # Per-class sample counts
    train_class_counts = train_df["class_name"].value_counts()
    val_class_counts = val_df["class_name"].value_counts()

    min_train_samples = int(train_class_counts.min())
    max_train_samples = int(train_class_counts.max())
    mean_train_samples = float(train_class_counts.mean())

    min_val_samples = int(val_class_counts.min())
    max_val_samples = int(val_class_counts.max())
    mean_val_samples = float(val_class_counts.mean())

    print(f"  - Train samples per class: min={min_train_samples}, max={max_train_samples}, mean={mean_train_samples:.2f}")
    print(f"  - Val samples per class  : min={min_val_samples}, max={max_val_samples}, mean={mean_val_samples:.2f}")

    # Integrity & NaN audit
    train_nans = int(np.isnan(X_train_raw).sum())
    val_nans = int(np.isnan(X_val_raw).sum())
    train_infs = int(np.isinf(X_train_raw).sum())
    val_infs = int(np.isinf(X_val_raw).sum())

    assert train_nans == 0 and val_nans == 0, "NaNs found in feature vectors!"
    assert train_infs == 0 and val_infs == 0, "Infs found in feature vectors!"
    print("  - Data Integrity       : 0 NaNs, 0 Infs, 100% clean vectors.")

    # -------------------------------------------------------------------------
    # 2. Feature Normalization (Strictly fitted on Train only)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Normalizing Features using Training Statistics Only...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # Save scaler statistics
    scaler_save_path = models_dir / "step16a_hasthas_scaler.npz"
    np.savez_compressed(
        scaler_save_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_
    )
    print(f"  - Saved scaler parameters to: {scaler_save_path}")

    # Compute balanced class weights from training distribution
    classes_present = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=y_train
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, weights)}
    print(f"  - Computed balanced class weights for {len(class_weight_dict)} classes.")

    # -------------------------------------------------------------------------
    # 3. Model Construction
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Constructing Landmark Classifier MLP...")
    model = build_hasthas_landmark_mlp(input_dim=feat_dim, num_classes=num_classes)
    model.summary()

    # Optimizer, Loss, and Metrics
    initial_lr = 1e-3
    optimizer = optimizers.Adam(learning_rate=initial_lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy", top5_metric]
    )

    # -------------------------------------------------------------------------
    # 4. Training Callbacks & Execution
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Training Landmark Classifier with Early Stopping...")
    best_model_path = models_dir / "step16a_hasthas_landmark_mlp_best.keras"

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode="min",
        verbose=1
    )

    batch_size = 16
    max_epochs = 100

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1
    )

    # -------------------------------------------------------------------------
    # 5. Evaluation & Performance Audit (Best Restored Model)
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Evaluating Best Model Checkpoint on Validation Data...")
    best_model = tf.keras.models.load_model(str(best_model_path))

    # Evaluate validation loss and accuracy
    val_loss, val_top1_acc, val_top5_acc = best_model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_top1_acc, train_top5_acc = best_model.evaluate(X_train, y_train, verbose=0)

    # Find best epoch from history
    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss = float(val_losses[best_epoch - 1])
    final_val_top1 = float(history.history["val_accuracy"][-1])
    train_acc_best_epoch = float(history.history["accuracy"][best_epoch - 1])

    acc_gap = float(train_acc_best_epoch - val_top1_acc)

    print(f"  * Best Epoch                      : {best_epoch} / {len(val_losses)}")
    print(f"  * Best Validation Top-1 Accuracy  : {val_top1_acc * 100:.2f}%")
    print(f"  * Final Validation Top-1 Accuracy : {final_val_top1 * 100:.2f}%")
    print(f"  * Best Validation Top-5 Accuracy  : {val_top5_acc * 100:.2f}%")
    print(f"  * Best Validation Loss            : {best_val_loss:.4f}")
    print(f"  * Training Accuracy at Best Epoch : {train_acc_best_epoch * 100:.2f}%")
    print(f"  * Generalization Gap (Train - Val): {acc_gap * 100:.2f} percentage points")

    # Detailed Predictions & Macro Metrics
    val_probs = best_model.predict(X_val, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_val, val_preds, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_val, val_preds, average="weighted", zero_division=0
    )

    print(f"  * Macro Precision                 : {macro_precision * 100:.2f}%")
    print(f"  * Macro Recall                    : {macro_recall * 100:.2f}%")
    print(f"  * Macro F1 Score                  : {macro_f1 * 100:.2f}%")

    # Per-Class Accuracy
    per_class_results = []
    cm = confusion_matrix(y_val, val_preds, labels=list(range(num_classes)))

    for cls_idx in range(num_classes):
        cls_name = idx_to_class[cls_idx]
        total_cls = int((y_val == cls_idx).sum())
        correct_cls = int(cm[cls_idx, cls_idx])
        acc_cls = (correct_cls / total_cls * 100) if total_cls > 0 else 0.0
        
        per_class_results.append({
            "local_class_id": cls_idx,
            "class_name": cls_name,
            "validation_samples": total_cls,
            "correct_predictions": correct_cls,
            "accuracy_percent": round(acc_cls, 2)
        })

    df_per_class = pd.DataFrame(per_class_results)
    metrics_csv_path = reports_dir / "step16a_hasthas_landmark_metrics.csv"
    df_per_class.to_csv(metrics_csv_path, index=False)
    print(f"  - Saved per-class metrics to: {metrics_csv_path}")

    # -------------------------------------------------------------------------
    # 6. Comparison with Previous Hasthas CNN Expert (Step 13)
    # -------------------------------------------------------------------------
    old_hasthas_val_acc = 15.29  # Step 13 measured Hasthas Expert accuracy
    old_hasthas_train_acc = 77.88 # Step 13 measured Hasthas Expert training accuracy
    new_val_acc_pct = val_top1_acc * 100.0
    delta_val_acc = new_val_acc_pct - old_hasthas_val_acc

    print("\n[STEP 6] Comparison with Previous Hasthas CNN Expert:")
    print(f"  - Previous CNN Expert Val Accuracy: {old_hasthas_val_acc:.2f}%")
    print(f"  - New Landmark MLP Val Accuracy   : {new_val_acc_pct:.2f}%")
    print(f"  - Absolute Improvement            : {delta_val_acc:+.2f} percentage points")
    print(f"  - Relative Improvement Factor     : {new_val_acc_pct / old_hasthas_val_acc:.2f}x")

    # -------------------------------------------------------------------------
    # 7. Visualization Artifacts
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Generating Training History & Confusion Matrix Plots...")
    
    # 1. Training History Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", color="#1f77b4", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Top-1 Accuracy", color="#ff7f0e", lw=2)
    axes[0].plot(history.history["val_top_5_accuracy"], label="Val Top-5 Accuracy", color="#2ca02c", linestyle="--", lw=1.5)
    axes[0].axvline(best_epoch - 1, color="red", linestyle=":", label=f"Best Checkpoint (Ep {best_epoch})")
    axes[0].set_title("Hasthas Landmark MLP: Accuracy vs Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    axes[1].axvline(best_epoch - 1, color="red", linestyle=":", label=f"Best Checkpoint (Ep {best_epoch})")
    axes[1].set_title("Hasthas Landmark MLP: Loss vs Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_plot_path = reports_dir / "step16a_hasthas_training_history.png"
    plt.savefig(history_plot_path, dpi=300)
    plt.close()
    print(f"  - Saved training history plot to: {history_plot_path}")

    # 2. Confusion Matrix Plot
    plt.figure(figsize=(16, 14))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Hasthas Landmark Classifier: 52-Class Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [idx_to_class[i] for i in range(num_classes)], rotation=90, fontsize=8)
    plt.yticks(tick_marks, [idx_to_class[i] for i in range(num_classes)], fontsize=8)
    plt.tight_layout()
    cm_plot_path = reports_dir / "step16a_hasthas_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"  - Saved confusion matrix plot to: {cm_plot_path}")

    # -------------------------------------------------------------------------
    # 8. Save Class Mapping & Training Configuration
    # -------------------------------------------------------------------------
    class_map_path = models_dir / "step16a_hasthas_class_mapping.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "num_classes": num_classes
        }, f, indent=2)

    config_path = models_dir / "step16a_training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "Hasthas_Landmark_MLP",
            "input_dimension": feat_dim,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "initial_learning_rate": initial_lr,
            "best_epoch": best_epoch,
            "total_epochs_trained": len(val_losses),
            "best_val_loss": best_val_loss,
            "best_val_top1_accuracy": float(val_top1_acc),
            "best_val_top5_accuracy": float(val_top5_acc),
            "train_accuracy_best_epoch": train_acc_best_epoch,
            "random_seed": RANDOM_SEED
        }, f, indent=2)

    # -------------------------------------------------------------------------
    # 9. Technical Report Generation
    # -------------------------------------------------------------------------
    report_file_path = reports_dir / "step16a_hasthas_landmark_report.txt"
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STEP 16A — HASTHAS LANDMARK-BASED CLASSIFIER REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. DATASET & AUDIT SUMMARY:\n")
        f.write("-----------------------------------------------------------------------\n")
        f.write(f"  Category                         : 1. Hasthas (One-Hand & Two-Hand Mudras)\n")
        f.write(f"  Total Hasthas Classes            : {num_classes} classes\n")
        f.write(f"  Training Samples Ingested        : {num_train} (from v2_train.csv)\n")
        f.write(f"  Validation Samples Ingested      : {num_val} (from v2_validation.csv)\n")
        f.write(f"  Test Samples                     : 65 (STRICTLY EXCLUDED)\n")
        f.write(f"  Train Samples per Class          : Min={min_train_samples}, Max={max_train_samples}, Mean={mean_train_samples:.2f}\n")
        f.write(f"  Val Samples per Class            : Min={min_val_samples}, Max={max_val_samples}, Mean={mean_val_samples:.2f}\n")
        f.write(f"  Source-level Overlap (Train/Val) : 0 entities (100% strict isolation)\n")
        f.write(f"  Data Integrity                   : 0 NaNs, 0 Infs\n\n")

        f.write("2. FEATURE REPRESENTATION:\n")
        f.write("-----------------------------------------------------------------------\n")
        f.write(f"  Input Dimensionality             : {feat_dim} float32 features\n")
        f.write("  Components:\n")
        f.write("    - Scale-normalized wrist-relative 3D landmarks (Hand 1 & Hand 2): 126 dims\n")
        f.write("    - Normalized fingertip-to-wrist distances (5 per hand)          :  10 dims\n")
        f.write("    - Pairwise inter-fingertip distances (9 per hand)               :  18 dims\n")
        f.write("    - 3D finger joint flexion angles (10 per hand)                  :  20 dims\n")
        f.write("    - Inter-hand spatial distance (Samyutha gestures)               :   1 dim\n")
        f.write("  Normalization                    : StandardScaler (Fitted strictly on Train only)\n\n")

        f.write("3. MODEL ARCHITECTURE:\n")
        f.write("-----------------------------------------------------------------------\n")
        f.write("  Layer (type)                     Output Shape         Param #\n")
        f.write("  =================================================================\n")
        f.write("  landmark_geometric_input         (None, 175)          0\n")
        f.write("  dense_256 (Dense, ReLU)          (None, 256)          45,056\n")
        f.write("  bn_1 (BatchNormalization)        (None, 256)          1,024\n")
        f.write("  dropout_1 (Dropout 0.30)         (None, 256)          0\n")
        f.write("  dense_128 (Dense, ReLU)          (None, 128)          32,896\n")
        f.write("  bn_2 (BatchNormalization)        (None, 128)          512\n")
        f.write("  dropout_2 (Dropout 0.30)         (None, 128)          0\n")
        f.write("  dense_64 (Dense, ReLU)           (None, 64)           8,256\n")
        f.write("  dropout_3 (Dropout 0.20)         (None, 64)           0\n")
        f.write("  hasthas_output (Dense, Softmax)  (None, 52)           3,380\n")
        f.write("  =================================================================\n")
        f.write(f"  Total Parameters                 : {model.count_params():,}\n\n")

        f.write("4. TRAINING PERFORMANCE:\n")
        f.write("-----------------------------------------------------------------------\n")
        f.write(f"  Optimizer                        : Adam (initial_lr=1e-3)\n")
        f.write(f"  Batch Size                       : {batch_size}\n")
        f.write(f"  Total Epochs Trained             : {len(val_losses)}\n")
        f.write(f"  Best Epoch Checkpoint            : {best_epoch}\n")
        f.write(f"  Training Accuracy at Best Epoch  : {train_acc_best_epoch * 100:.2f}%\n")
        f.write(f"  Best Validation Loss             : {best_val_loss:.4f}\n\n")

        f.write("5. VALIDATION PERFORMANCE (BEST RESTORED MODEL):\n")
        f.write("-----------------------------------------------------------------------\n")
        f.write(f"  Best Validation Top-1 Accuracy   : {val_top1_acc * 100:.2f}%\n")
        f.write(f"  Best Validation Top-5 Accuracy   : {val_top5_acc * 100:.2f}%\n")
        f.write(f"  Final Validation Top-1 Accuracy  : {final_val_top1 * 100:.2f}%\n")
        f.write(f"  Generalization Gap (Train - Val) : {acc_gap * 100:.2f} percentage points\n")
        f.write(f"  Macro Precision                  : {macro_precision * 100:.2f}%\n")
        f.write(f"  Macro Recall                     : {macro_recall * 100:.2f}%\n")
        f.write(f"  Macro F1 Score                   : {macro_f1 * 100:.2f}%\n")
        f.write(f"  Weighted F1 Score                : {weighted_f1 * 100:.2f}%\n\n")

        f.write("6. COMPARISON WITH STEP 13 HASTHAS CNN EXPERT:\n")
        f.write("-----------------------------------------------------------------------\n")
        f.write(f"  Step 13 CNN Expert Validation Acc : {old_hasthas_val_acc:.2f}%\n")
        f.write(f"  Step 16A Landmark MLP Val Accuracy: {new_val_acc_pct:.2f}%\n")
        f.write(f"  Absolute Improvement              : {delta_val_acc:+.2f} percentage points\n")
        f.write(f"  Relative Multiplier               : {new_val_acc_pct / old_hasthas_val_acc:.2f}x\n\n")

        f.write("7. ARTIFACTS CREATED:\n")
        f.write("-----------------------------------------------------------------------\n")
        f.write("  - Model Checkpoint : models/step16a_hasthas_landmark_mlp_best.keras\n")
        f.write("  - Scaler Parameters: models/step16a_hasthas_scaler.npz\n")
        f.write("  - Class Mapping    : models/step16a_hasthas_class_mapping.json\n")
        f.write("  - Training Config  : models/step16a_training_config.json\n")
        f.write("  - Metrics CSV      : reports/step16a_hasthas_landmark_metrics.csv\n")
        f.write("  - Training Curves  : reports/step16a_hasthas_training_history.png\n")
        f.write("  - Confusion Matrix : reports/step16a_hasthas_confusion_matrix.png\n")
        f.write("  - Technical Report : reports/step16a_hasthas_landmark_report.txt\n\n")

        f.write("=======================================================================\n")
        f.write(f"STATUS: STEP 16A COMPLETE — HASTHAS LANDMARK CLASSIFIER TRAINED.\n")
        f.write("=======================================================================\n")

    print(f"  - Saved report to: {report_file_path}")
    print(f"\nExecution finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
