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
        PRETRAINED_BACKBONE,
        TRANSFER_MODEL_NAME,
        NUM_CLASSES,
        CNN_INPUT_SHAPE,
        BATCH_SIZE,
        LEARNING_RATE_STAGE1,
        DENSE_UNITS,
        DROPOUT_RATE,
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
    from transfer_model import build_transfer_model
    from model import compute_balanced_class_weights
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
        PRETRAINED_BACKBONE,
        TRANSFER_MODEL_NAME,
        NUM_CLASSES,
        CNN_INPUT_SHAPE,
        BATCH_SIZE,
        LEARNING_RATE_STAGE1,
        DENSE_UNITS,
        DROPOUT_RATE,
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
    from src.transfer_model import build_transfer_model
    from src.model import compute_balanced_class_weights
    from src.dataset import load_partition_data, KuchipudiDataSequence, get_class_mapping
    from src.augmentation import KuchipudiAugmenter

import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

# Stage 1 Parameters
STAGE1_MAX_EPOCHS = 30
STAGE1_EARLY_PATIENCE = 7
STAGE1_LR_PATIENCE = 3
STAGE1_LR_FACTOR = 0.2
MIN_LEARNING_RATE = 1e-6

STAGE1_BEST_MODEL_PATH = MODELS_DIR / "transfer_stage1_best.keras"
STAGE1_HISTORY_CSV = REPORTS_DIR / "transfer_stage1_history.csv"
STAGE1_REPORT_TXT = REPORTS_DIR / "transfer_stage1_report.txt"
STAGE1_ACC_PLOT_PNG = PLOTS_DIR / "transfer_stage1_accuracy.png"
STAGE1_LOSS_PLOT_PNG = PLOTS_DIR / "transfer_stage1_loss.png"

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
        print(f"Epoch {epoch+1:02d}/{STAGE1_MAX_EPOCHS:02d} -> "
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
        print(f"\nSaved Stage 1 history to {self.csv_path}")

def plot_training_curves(history_records: List[Dict[str, Any]]):
    """
    Generates two publication-ready PNG plots:
    1. transfer_stage1_accuracy.png: Training vs Validation Accuracy
    2. transfer_stage1_loss.png: Training vs Validation Loss
    """
    epochs = [r["epoch"] for r in history_records]
    train_acc = [r["train_accuracy"] for r in history_records]
    val_acc = [r["validation_accuracy"] for r in history_records]
    train_loss = [r["train_loss"] for r in history_records]
    val_loss = [r["validation_loss"] for r in history_records]

    best_val_acc_idx = int(np.argmax(val_acc))
    best_epoch_acc = epochs[best_val_acc_idx]

    # --- Plot 1: Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Training Accuracy", color="#1f77b4", linewidth=2.0)
    plt.plot(epochs, val_acc, label="Validation Accuracy", color="#ff7f0e", linewidth=2.0)
    plt.axvline(x=best_epoch_acc, color="green", linestyle="--", alpha=0.7, label=f"Best Val Acc (Epoch {best_epoch_acc}: {val_acc[best_val_acc_idx]*100:.1f}%)")
    plt.title(f"Transfer Learning Stage 1 ({PRETRAINED_BACKBONE}) - Accuracy", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Categorical Accuracy", fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(STAGE1_ACC_PLOT_PNG, dpi=300)
    plt.close()
    print(f"Saved accuracy plot: {STAGE1_ACC_PLOT_PNG}")

    # --- Plot 2: Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.0)
    plt.plot(epochs, val_loss, label="Validation Loss", color="#d62728", linewidth=2.0)
    plt.axvline(x=best_epoch_acc, color="green", linestyle="--", alpha=0.7, label=f"Best Model (Epoch {best_epoch_acc})")
    plt.title(f"Transfer Learning Stage 1 ({PRETRAINED_BACKBONE}) - Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Categorical Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(STAGE1_LOSS_PLOT_PNG, dpi=300)
    plt.close()
    print(f"Saved loss plot: {STAGE1_LOSS_PLOT_PNG}")

def run_stage1_training():
    print("=" * 80)
    print("STEP 8C: TRANSFER LEARNING STAGE 1 TRAINING (BACKBONE FROZEN)")
    print("=" * 80)

    # 1. Environment Confirmation
    print("\n[1] Environment Confirmation:")
    print(f"  Python executable   : {sys.executable}")
    print(f"  TensorFlow version  : {tf.__version__}")
    print(f"  Pretrained Backbone : {PRETRAINED_BACKBONE}")
    print(f"  Random Seed         : {RANDOM_SEED}")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # 2. Data Splits Confirmation
    splits_dir = DATA_DIR / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "validation.csv"
    test_csv = splits_dir / "test.csv"

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
    print(f"  Class weight range in train    : {min_w:.4f} to {max_w:.4f}")

    # 4. Generators Construction
    print("\n[4] Constructing Data Generators:")
    augmenter = KuchipudiAugmenter(enable=True, seed=RANDOM_SEED)
    print("  TRAIN generator      : On-the-fly Augmentation ENABLED")
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

    # 5. Build Stage-1 Transfer Model
    print(f"\n[5] Instantiating Stage-1 Transfer Model '{TRANSFER_MODEL_NAME}' ({NUM_CLASSES} classes)...")
    model, backbone = build_transfer_model(
        num_classes=NUM_CLASSES,
        input_shape=CNN_INPUT_SHAPE,
        backbone_name=PRETRAINED_BACKBONE,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE_STAGE1,
        loss_type=LOSS_FUNCTION,
        model_name=TRANSFER_MODEL_NAME
    )
    total_params = model.count_params()
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    non_trainable_params = sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)
    print(f"  Backbone state           : FROZEN (backbone.trainable = False)")
    print(f"  Total parameters         : {total_params:,}")
    print(f"  Trainable parameters     : {trainable_params:,} (Classification Head only)")
    print(f"  Non-trainable parameters : {non_trainable_params:,} (Pretrained ImageNet weights)")

    # 6. Callbacks Setup
    print("\n[6] Configuring Callbacks for Stage 1:")
    print(f"  - ModelCheckpoint   : saving best model to {STAGE1_BEST_MODEL_PATH} (monitor=val_accuracy, mode=max)")
    print(f"  - EarlyStopping     : monitor=val_loss, patience={STAGE1_EARLY_PATIENCE}, restore_best_weights=True, mode=min")
    print(f"  - ReduceLROnPlateau : monitor=val_loss, factor={STAGE1_LR_FACTOR}, patience={STAGE1_LR_PATIENCE}, min_lr={MIN_LEARNING_RATE}, mode=min")

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(STAGE1_BEST_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max"
    )

    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=STAGE1_EARLY_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode="min"
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=STAGE1_LR_FACTOR,
        patience=STAGE1_LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1,
        mode="min"
    )

    metrics_logger_cb = EpochMetricsLogger(csv_path=STAGE1_HISTORY_CSV)

    callbacks_list = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, metrics_logger_cb]

    # 7. Model Training (Stage 1)
    print(f"\n[7] Starting Stage 1 Training (MAX_EPOCHS = {STAGE1_MAX_EPOCHS}, BATCH_SIZE = {BATCH_SIZE}, LR = {LEARNING_RATE_STAGE1})...")
    start_time = time.time()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=STAGE1_MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=0
    )

    total_training_time = time.time() - start_time
    print(f"\nStage 1 training completed in {total_training_time:.1f}s ({total_training_time/60.0:.2f} mins).")

    # 8. Extract Metrics from History
    records = metrics_logger_cb.history_records
    epochs_completed = len(records)
    val_losses = [r["validation_loss"] for r in records]
    val_accuracies = [r["validation_accuracy"] for r in records]
    train_accuracies = [r["train_accuracy"] for r in records]
    learning_rates = [r["learning_rate"] for r in records]

    best_val_acc_idx = int(np.argmax(val_accuracies))
    best_epoch_acc = records[best_val_acc_idx]["epoch"]
    best_val_acc = val_accuracies[best_val_acc_idx]
    best_train_acc = max(train_accuracies)
    best_val_loss = min(val_losses)

    final_train_acc = records[-1]["train_accuracy"]
    final_val_acc = records[-1]["validation_accuracy"]
    final_val_loss = records[-1]["validation_loss"]

    early_stopped = epochs_completed < STAGE1_MAX_EPOCHS
    lr_reductions = sum(1 for i in range(1, len(learning_rates)) if learning_rates[i] < learning_rates[i-1])

    # 9. Plot Curves
    print("\n[8] Generating Visualizations...")
    plot_training_curves(records)

    # 10. Generate Stage 1 Report
    print(f"\n[9] Generating Stage 1 Report in {STAGE1_REPORT_TXT}...")
    with open(STAGE1_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KUCHIPUDI CLASSICAL DANCE RECOGNITION - TRANSFER LEARNING STAGE 1 REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. MODEL ARCHITECTURE & PRETRAINED BACKBONE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Pretrained Backbone         : {PRETRAINED_BACKBONE} (ImageNet-1k)\n")
        f.write("- Backbone Status             : FROZEN (backbone.trainable = False throughout Stage 1)\n")
        f.write(f"- Total Parameters            : {total_params:,}\n")
        f.write(f"- Trainable Parameters        : {trainable_params:,} (Classification Head only)\n")
        f.write(f"- Non-trainable Parameters    : {non_trainable_params:,} (Pretrained ImageNet weights)\n")
        f.write(f"- Head Architecture           : GAP -> Dense({DENSE_UNITS}) -> BatchNorm -> ReLU -> Dropout({DROPOUT_RATE}) -> Dense({NUM_CLASSES}, Softmax)\n\n")

        f.write("2. DATASET PARTITIONING & LEAKAGE CONTROL:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Training Samples            : {len(train_images)} (Holdout training partition)\n")
        f.write(f"- Validation Samples          : {len(val_images)} (Holdout validation partition)\n")
        f.write(f"- Test Samples                : {test_count} (COMPLETELY UNTOUCHED held-out test partition)\n")
        f.write("- Validation Augmentation     : STRICTLY DISABLED (Identity tensors, 0% modification)\n")
        f.write("- Test Set Status             : STRICTLY ISOLATED & UNTOUCHED (Zero evaluation performed)\n\n")

        f.write("3. HYPERPARAMETERS & CALLBACK SETTINGS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Optimizer                   : {OPTIMIZER_NAME.upper()} (Adam)\n")
        f.write(f"- Initial Learning Rate       : {LEARNING_RATE_STAGE1}\n")
        f.write(f"- Batch Size                  : {BATCH_SIZE}\n")
        f.write(f"- Loss Function               : {LOSS_FUNCTION}\n")
        f.write(f"- Augmentation Configuration  : Training-only stochastic (Rot: +/-{ROTATION_RANGE_DEGREES} deg, Shift: +/-{WIDTH_SHIFT_FRACTION*100}%, Zoom: {ZOOM_RANGE}, Brightness: {BRIGHTNESS_RANGE}, Contrast: {CONTRAST_RANGE}, Flips: False)\n")
        f.write(f"- Class Weights Range         : {min_w:.4f} to {max_w:.4f} (computed SOLELY from training data)\n")
        f.write(f"- Maximum Epochs Configured   : {STAGE1_MAX_EPOCHS}\n")
        f.write(f"- ModelCheckpoint Criterion   : monitor='val_accuracy', mode='max', save_best_only=True\n")
        f.write(f"- EarlyStopping Criterion     : monitor='val_loss', mode='min', patience={STAGE1_EARLY_PATIENCE}, restore_best_weights=True\n")
        f.write(f"- ReduceLROnPlateau Criterion : monitor='val_loss', mode='min', factor={STAGE1_LR_FACTOR}, patience={STAGE1_LR_PATIENCE}, min_lr={MIN_LEARNING_RATE}\n\n")

        f.write("4. EMPIRICAL RESULTS (STAGE 1):\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Total Training Time         : {total_training_time:.1f} seconds ({total_training_time/60.0:.2f} minutes)\n")
        f.write(f"- Actual Epochs Completed     : {epochs_completed} / {STAGE1_MAX_EPOCHS}\n")
        f.write(f"- Best Model Epoch (val_acc)  : Epoch {best_epoch_acc}\n")
        f.write(f"- Best Validation Accuracy    : {best_val_acc * 100:.2f}%\n")
        f.write(f"- Best Training Accuracy      : {best_train_acc * 100:.2f}%\n")
        f.write(f"- Best Validation Loss        : {best_val_loss:.4f}\n")
        f.write(f"- Final Validation Accuracy   : {final_val_acc * 100:.2f}%\n")
        f.write(f"- Final Validation Loss       : {final_val_loss:.4f}\n")
        f.write(f"- Final Training Accuracy     : {final_train_acc * 100:.2f}%\n")
        f.write(f"- Early Stopping Triggered    : {early_stopped} ({'Triggered at epoch ' + str(epochs_completed) if early_stopped else 'Completed all epochs'})\n")
        f.write(f"- Learning Rate Reductions    : {lr_reductions} times\n\n")

        f.write("5. ARTIFACT PERSISTENCE LOCATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Best Stage-1 Model Saved To : {STAGE1_BEST_MODEL_PATH}\n")
        f.write(f"- Training History CSV        : {STAGE1_HISTORY_CSV}\n")
        f.write(f"- Accuracy Curves Plot        : {STAGE1_ACC_PLOT_PNG}\n")
        f.write(f"- Loss Curves Plot            : {STAGE1_LOSS_PLOT_PNG}\n\n")

        f.write("6. ACADEMIC INTEGRITY COMMITMENT:\n")
        f.write("-" * 80 + "\n")
        f.write("Stage 1 training was performed strictly on the classification head with the backbone frozen.\n")
        f.write("No test set evaluation was performed. Stage 2 fine-tuning was NOT started.\n")
        f.write("The 95% target is an engineering goal to be evaluated following Stage 2.\n")

    print(f"\nStage 1 report created: {STAGE1_REPORT_TXT}")
    print("=" * 80)
    print("STEP 8C COMPLETE: Stage 1 finished. Awaiting user review before Stage 2.")
    print("=" * 80)

if __name__ == "__main__":
    run_stage1_training()
