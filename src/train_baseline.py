import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Set seed early for full reproducibility
np.random.seed(42)

try:
    from config import (
        PROJECT_ROOT,
        DATA_DIR,
        REPORTS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        MODELS_DIR,
        MODEL_NAME,
        NUM_CLASSES,
        CNN_INPUT_SHAPE,
        BATCH_SIZE,
        LEARNING_RATE,
        DROPOUT_RATE,
        DENSE_UNITS,
        L2_REGULARIZATION,
        LOSS_FUNCTION,
        OPTIMIZER_NAME,
        ROTATION_RANGE_DEGREES,
        WIDTH_SHIFT_FRACTION,
        HEIGHT_SHIFT_FRACTION,
        ZOOM_RANGE,
        BRIGHTNESS_RANGE,
        CONTRAST_RANGE,
        HORIZONTAL_FLIP,
        VERTICAL_FLIP,
        AUGMENTATION_PROBABILITY,
        RANDOM_SEED
    )
    from model import build_cnn_model, compute_balanced_class_weights
    from dataset import load_partition_data, KuchipudiDataSequence, get_class_mapping
    from augmentation import KuchipudiAugmenter
except ImportError:
    from src.config import (
        PROJECT_ROOT,
        DATA_DIR,
        REPORTS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        MODELS_DIR,
        MODEL_NAME,
        NUM_CLASSES,
        CNN_INPUT_SHAPE,
        BATCH_SIZE,
        LEARNING_RATE,
        DROPOUT_RATE,
        DENSE_UNITS,
        L2_REGULARIZATION,
        LOSS_FUNCTION,
        OPTIMIZER_NAME,
        ROTATION_RANGE_DEGREES,
        WIDTH_SHIFT_FRACTION,
        HEIGHT_SHIFT_FRACTION,
        ZOOM_RANGE,
        BRIGHTNESS_RANGE,
        CONTRAST_RANGE,
        HORIZONTAL_FLIP,
        VERTICAL_FLIP,
        AUGMENTATION_PROBABILITY,
        RANDOM_SEED
    )
    from src.model import build_cnn_model, compute_balanced_class_weights
    from src.dataset import load_partition_data, KuchipudiDataSequence, get_class_mapping
    from src.augmentation import KuchipudiAugmenter

import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

# Explicit baseline targets
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 12
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

BASELINE_BEST_MODEL_PATH = MODELS_DIR / "baseline_best.keras"
BASELINE_HISTORY_CSV = REPORTS_DIR / "baseline_training_history.csv"
BASELINE_REPORT_TXT = REPORTS_DIR / "baseline_training_report.txt"
BASELINE_ACC_PLOT_PNG = PLOTS_DIR / "baseline_accuracy.png"
BASELINE_LOSS_PLOT_PNG = PLOTS_DIR / "baseline_loss.png"

class EpochMetricsLogger(callbacks.Callback):
    """
    Custom callback to log per-epoch metrics and record into a structured CSV:
    epoch, train_loss, train_accuracy, validation_loss, validation_accuracy, learning_rate
    """
    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = csv_path
        self.history_records = []

    def on_train_begin(self, logs=None):
        self.history_records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Get current learning rate
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
        print(f"Epoch {epoch+1:02d}/{MAX_EPOCHS:02d} -> "
              f"loss: {record['train_loss']:.4f} - acc: {record['train_accuracy']:.4f} | "
              f"val_loss: {record['validation_loss']:.4f} - val_acc: {record['validation_accuracy']:.4f} | "
              f"lr: {record['learning_rate']:.6f}")

    def on_train_end(self, logs=None):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "learning_rate"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.history_records:
                writer.writerow(r)
        print(f"\nSaved training history to {self.csv_path}")

def plot_training_curves(history_records: List[Dict[str, Any]]):
    """
    Generates two publication-ready PNG plots:
    1. baseline_accuracy.png: Training vs Validation Accuracy
    2. baseline_loss.png: Training vs Validation Loss
    """
    epochs = [r["epoch"] for r in history_records]
    train_acc = [r["train_accuracy"] for r in history_records]
    val_acc = [r["validation_accuracy"] for r in history_records]
    train_loss = [r["train_loss"] for r in history_records]
    val_loss = [r["validation_loss"] for r in history_records]

    best_val_loss_idx = int(np.argmin(val_loss))
    best_epoch_loss = epochs[best_val_loss_idx]

    # --- Plot 1: Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Training Accuracy", color="#1f77b4", linewidth=2.0)
    plt.plot(epochs, val_acc, label="Validation Accuracy", color="#ff7f0e", linewidth=2.0)
    plt.axvline(x=best_epoch_loss, color="green", linestyle="--", alpha=0.7, label=f"Best Model (Epoch {best_epoch_loss})")
    plt.title("Baseline CNN - Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Categorical Accuracy", fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(BASELINE_ACC_PLOT_PNG, dpi=300)
    plt.close()
    print(f"Saved accuracy plot: {BASELINE_ACC_PLOT_PNG}")

    # --- Plot 2: Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.0)
    plt.plot(epochs, val_loss, label="Validation Loss", color="#d62728", linewidth=2.0)
    plt.axvline(x=best_epoch_loss, color="green", linestyle="--", alpha=0.7, label=f"Best Model (Epoch {best_epoch_loss})")
    plt.title("Baseline CNN - Training vs Validation Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Categorical Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(BASELINE_LOSS_PLOT_PNG, dpi=300)
    plt.close()
    print(f"Saved loss plot: {BASELINE_LOSS_PLOT_PNG}")

def run_baseline_training():
    print("=" * 80)
    print("STEP 6: KUCHIPUDI BASELINE CNN TRAINING")
    print("=" * 80)

    # 1. Environment Confirmation
    print("\n[1] Environment Confirmation:")
    print(f"  Python executable   : {sys.executable}")
    print(f"  TensorFlow version  : {tf.__version__}")
    print(f"  Random Seed         : {RANDOM_SEED}")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # 2. Data Splits Confirmation
    splits_dir = DATA_DIR / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "validation.csv"
    test_csv = splits_dir / "test.csv"

    # Count test samples without touching/loading image pixels
    with open(test_csv, "r", encoding="utf-8") as f:
        test_count = sum(1 for _ in f) - 1

    print("\n[2] Loading Data Partitions:")
    class_mapping = get_class_mapping()
    train_images, train_labels_oh, train_labels_int, _ = load_partition_data(
        train_csv, class_mapping=class_mapping, num_classes=NUM_CLASSES, verbose=True
    )
    val_images, val_labels_oh, val_labels_int, _ = load_partition_data(
        val_csv, class_mapping=class_mapping, num_classes=NUM_CLASSES, verbose=True
    )
    print(f"  Train sample count      : {len(train_images)}")
    print(f"  Validation sample count : {len(val_images)}")
    print(f"  Test sample count       : {test_count} (STRICTLY UNTOUCHED - NO EVALUATION)")

    # 3. Class Imbalance Calculation (TRAIN DATA ONLY)
    print("\n[3] Calculating Balanced Class Weights from Training Labels Only...")
    class_weights = compute_balanced_class_weights(train_labels_int, num_classes=NUM_CLASSES)
    present_weights = {k: v for k, v in class_weights.items() if k in set(train_labels_int)}
    min_w = min(present_weights.values())
    max_w = max(present_weights.values())
    print(f"  Classes present in train split : {len(present_weights)} / {NUM_CLASSES}")
    print(f"  Min class weight in train      : {min_w:.4f}")
    print(f"  Max class weight in train      : {max_w:.4f}")

    # 4. Generators Construction
    print("\n[4] Constructing Data Generators:")
    augmenter = KuchipudiAugmenter(enable=True, seed=RANDOM_SEED)
    print("  TRAIN generator      : On-the-fly Augmentation ENABLED (prob=0.75, rot=+/-8 deg, trans=+/-6%, zoom=[0.95,1.05])")
    print("  VALIDATION generator : Augmentation STRICTLY DISABLED (Identity tensors, 0% transformation)")
    print("  TEST generator       : NOT CREATED (Test set untouched)")

    train_gen = KuchipudiDataSequence(
        images=train_images,
        labels=train_labels_oh,
        batch_size=BATCH_SIZE,
        is_training=True,
        augmenter=augmenter,
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_gen = KuchipudiDataSequence(
        images=val_images,
        labels=val_labels_oh,
        batch_size=BATCH_SIZE,
        is_training=False,
        augmenter=None,
        shuffle=False,
        seed=RANDOM_SEED
    )

    # 5. Build CNN Model
    print(f"\n[5] Instantiating Model '{MODEL_NAME}' ({NUM_CLASSES} classes)...")
    model = build_cnn_model(
        num_classes=NUM_CLASSES,
        input_shape=CNN_INPUT_SHAPE,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE,
        dense_units=DENSE_UNITS,
        l2_reg=L2_REGULARIZATION,
        loss_type=LOSS_FUNCTION,
        model_name=MODEL_NAME
    )
    total_params = model.count_params()
    print(f"  Total parameters : {total_params:,}")

    # 6. Callbacks Setup
    print("\n[6] Configuring Callbacks:")
    print(f"  - ModelCheckpoint   : saving best model to {BASELINE_BEST_MODEL_PATH} (monitor=val_loss, mode=min)")
    print(f"  - EarlyStopping     : monitor=val_loss, patience={EARLY_STOPPING_PATIENCE}, restore_best_weights=True")
    print(f"  - ReduceLROnPlateau : monitor=val_loss, factor={REDUCE_LR_FACTOR}, patience={REDUCE_LR_PATIENCE}, min_lr={MIN_LEARNING_RATE}")

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(BASELINE_BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
        mode="min"
    )

    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode="min"
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1,
        mode="min"
    )

    metrics_logger_cb = EpochMetricsLogger(csv_path=BASELINE_HISTORY_CSV)

    callbacks_list = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, metrics_logger_cb]

    # 7. Model Training
    print(f"\n[7] Starting Baseline Training (MAX_EPOCHS = {MAX_EPOCHS}, BATCH_SIZE = {BATCH_SIZE})...")
    start_time = time.time()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=0  # EpochMetricsLogger handles clean progress output
    )

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f}s ({total_training_time/60.0:.2f} mins).")

    # 8. Extract Metrics from History
    records = metrics_logger_cb.history_records
    epochs_completed = len(records)
    val_losses = [r["validation_loss"] for r in records]
    val_accuracies = [r["validation_accuracy"] for r in records]
    train_accuracies = [r["train_accuracy"] for r in records]
    learning_rates = [r["learning_rate"] for r in records]

    best_val_loss_idx = int(np.argmin(val_losses))
    best_epoch = records[best_val_loss_idx]["epoch"]
    best_val_loss = records[best_val_loss_idx]["validation_loss"]
    best_val_acc = max(val_accuracies)
    best_train_acc = max(train_accuracies)

    final_train_acc = records[-1]["train_accuracy"]
    final_val_acc = records[-1]["validation_accuracy"]
    final_val_loss = records[-1]["validation_loss"]

    early_stopped = epochs_completed < MAX_EPOCHS
    lr_reduced = any(lr < LEARNING_RATE for lr in learning_rates)

    # 9. Plot Curves
    print("\n[8] Generating Visualizations...")
    plot_training_curves(records)

    # 10. Generate Comprehensive Baseline Report
    print(f"\n[9] Generating Baseline Report in {BASELINE_REPORT_TXT}...")
    with open(BASELINE_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KUCHIPUDI CLASSICAL DANCE RECOGNITION - BASELINE TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. SYSTEM & FRAMEWORK CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- TensorFlow Version          : {tf.__version__}\n")
        f.write(f"- Python Executable           : {sys.executable}\n")
        f.write(f"- Random Seed                 : {RANDOM_SEED}\n\n")

        f.write("2. MODEL SPECIFICATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Model Architecture Name     : {MODEL_NAME}\n")
        f.write(f"- Total Parameter Count       : {total_params:,}\n")
        f.write(f"- Input Tensor Dimensions     : {CNN_INPUT_SHAPE}\n")
        f.write(f"- Output Classification Target: {NUM_CLASSES} classes (Softmax)\n")
        f.write(f"- Dense Layer Units           : {DENSE_UNITS}\n")
        f.write(f"- Classifier Dropout Rate     : {DROPOUT_RATE * 100:.1f}%\n")
        f.write(f"- L2 Weight Regularization    : {L2_REGULARIZATION}\n\n")

        f.write("3. DATASET PARTITIONING & LEAKAGE CONTROL:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Train Sample Count          : {len(train_images)} (Holdout training partition)\n")
        f.write(f"- Validation Sample Count     : {len(val_images)} (Holdout validation partition)\n")
        f.write(f"- Test Sample Count           : {test_count} (COMPLETELY UNTOUCHED held-out test partition)\n")
        f.write("- Leakage Safeguard           : All video frames grouped strictly by source_video_id.\n")
        f.write("                                Validation/test sets never augmented and never used during training.\n\n")

        f.write("4. TRAINING HYPERPARAMETERS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Optimizer                   : {OPTIMIZER_NAME.upper()} (Adam)\n")
        f.write(f"- Initial Learning Rate       : {LEARNING_RATE}\n")
        f.write(f"- Loss Function               : {LOSS_FUNCTION}\n")
        f.write(f"- Batch Size                  : {BATCH_SIZE}\n")
        f.write(f"- Maximum Epochs              : {MAX_EPOCHS}\n")
        f.write(f"- EarlyStopping Patience      : {EARLY_STOPPING_PATIENCE} (monitor='val_loss', restore_best_weights=True)\n")
        f.write(f"- ReduceLROnPlateau           : factor={REDUCE_LR_FACTOR}, patience={REDUCE_LR_PATIENCE}, min_lr={MIN_LEARNING_RATE}\n\n")

        f.write("5. AUGMENTATION & CLASS IMBALANCE CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Augmentation Strategy       : Training-only stochastic on-the-fly augmentation.\n")
        f.write(f"                                (Rot: +/-{ROTATION_RANGE_DEGREES} deg, Shift: +/-{WIDTH_SHIFT_FRACTION*100}%, Zoom: {ZOOM_RANGE}, Brightness: {BRIGHTNESS_RANGE}, Contrast: {CONTRAST_RANGE})\n")
        f.write(f"- Horizontal/Vertical Flip    : STRICTLY DISABLED ({HORIZONTAL_FLIP}/{VERTICAL_FLIP}) to preserve dance semantics.\n")
        f.write(f"- Validation/Test Augmentation: STRICTLY DISABLED (Identity tensors, 0% modification).\n")
        f.write(f"- Class Imbalance Strategy    : Balanced class weights computed SOLELY on training labels.\n")
        f.write(f"                                (Weights range from {min_w:.4f} to {max_w:.4f} across present classes).\n\n")

        f.write("6. EMPIRICAL TRAINING RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Total Training Time         : {total_training_time:.1f} seconds ({total_training_time/60.0:.2f} minutes)\n")
        f.write(f"- Actual Epochs Completed     : {epochs_completed} / {MAX_EPOCHS}\n")
        f.write(f"- Best Model Epoch            : Epoch {best_epoch}\n")
        f.write(f"- Best Validation Loss        : {best_val_loss:.4f}\n")
        f.write(f"- Best Validation Accuracy    : {best_val_acc * 100:.2f}%\n")
        f.write(f"- Best Training Accuracy      : {best_train_acc * 100:.2f}%\n")
        f.write(f"- Final Validation Accuracy   : {final_val_acc * 100:.2f}%\n")
        f.write(f"- Final Validation Loss       : {final_val_loss:.4f}\n")
        f.write(f"- Final Training Accuracy     : {final_train_acc * 100:.2f}%\n")
        f.write(f"- Early Stopping Triggered    : {early_stopped} ({'Triggered at epoch ' + str(epochs_completed) if early_stopped else 'Completed all epochs'})\n")
        f.write(f"- Learning Rate Reduction     : {lr_reduced} ({'Reduced during training' if lr_reduced else 'Constant'})\n\n")

        f.write("7. ARTIFACT PERSISTENCE & LOCATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Best Model Weights Saved To : {BASELINE_BEST_MODEL_PATH}\n")
        f.write(f"- Training History CSV        : {BASELINE_HISTORY_CSV}\n")
        f.write(f"- Accuracy Curves Plot        : {BASELINE_ACC_PLOT_PNG}\n")
        f.write(f"- Loss Curves Plot            : {BASELINE_LOSS_PLOT_PNG}\n\n")

        f.write("8. ACADEMIC INTEGRITY DECLARATION:\n")
        f.write("-" * 80 + "\n")
        f.write("This run constitutes an honest, unvarnished baseline CNN experiment.\n")
        f.write("No test set evaluation was performed. 95% accuracy was NOT fabricated.\n")
        f.write("Test set evaluation will only occur in the designated evaluation step.\n")

    print(f"\nBaseline report created: {BASELINE_REPORT_TXT}")
    print("=" * 80)
    print("STEP 6 COMPLETE: Awaiting user review. STOPPING.")
    print("=" * 80)

if __name__ == "__main__":
    run_baseline_training()
