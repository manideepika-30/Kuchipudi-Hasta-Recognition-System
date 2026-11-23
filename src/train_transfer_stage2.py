import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Fixed random seed
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
        LEARNING_RATE_STAGE2,
        LOSS_FUNCTION,
        OPTIMIZER_NAME,
        ROTATION_RANGE_DEGREES,
        WIDTH_SHIFT_FRACTION,
        HEIGHT_SHIFT_FRACTION,
        ZOOM_RANGE,
        BRIGHTNESS_RANGE,
        CONTRAST_RANGE,
        AUGMENTATION_PROBABILITY,
        RANDOM_SEED
    )
    from transfer_model import configure_stage2_finetuning
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
        LEARNING_RATE_STAGE2,
        LOSS_FUNCTION,
        OPTIMIZER_NAME,
        ROTATION_RANGE_DEGREES,
        WIDTH_SHIFT_FRACTION,
        HEIGHT_SHIFT_FRACTION,
        ZOOM_RANGE,
        BRIGHTNESS_RANGE,
        CONTRAST_RANGE,
        AUGMENTATION_PROBABILITY,
        RANDOM_SEED
    )
    from src.transfer_model import configure_stage2_finetuning
    from src.model import compute_balanced_class_weights
    from src.dataset import load_partition_data, KuchipudiDataSequence, get_class_mapping
    from src.augmentation import KuchipudiAugmenter

import tensorflow as tf
from tensorflow.keras import callbacks, layers
import matplotlib.pyplot as plt

# Stage 2 Hyperparameters
STAGE2_MAX_EPOCHS = 35
STAGE2_INITIAL_LR = 1e-4
STAGE2_EARLY_PATIENCE = 8
STAGE2_LR_PATIENCE = 3
STAGE2_LR_FACTOR = 0.2
STAGE2_MIN_LR = 1e-7
FINE_TUNE_PERCENT = 0.25

STAGE1_BEST_MODEL_PATH = MODELS_DIR / "transfer_stage1_best.keras"
STAGE2_BEST_MODEL_PATH = MODELS_DIR / "transfer_stage2_best.keras"
STAGE2_HISTORY_CSV = REPORTS_DIR / "transfer_stage2_history.csv"
STAGE2_REPORT_TXT = REPORTS_DIR / "transfer_stage2_report.txt"
STAGE2_ACC_PLOT_PNG = PLOTS_DIR / "transfer_stage2_accuracy.png"
STAGE2_LOSS_PLOT_PNG = PLOTS_DIR / "transfer_stage2_loss.png"

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
        print(f"Epoch {epoch+1:02d}/{STAGE2_MAX_EPOCHS:02d} -> "
              f"loss: {record['train_loss']:.4f} - acc: {record['train_accuracy']:.4f} | "
              f"val_loss: {record['validation_loss']:.4f} - val_acc: {record['validation_accuracy']:.4f} | "
              f"lr: {record['learning_rate']:.7f}")

    def on_train_end(self, logs=None):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "learning_rate"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.history_records:
                writer.writerow(r)
        print(f"\nSaved Stage 2 history to {self.csv_path}")

def plot_training_curves(history_records: List[Dict[str, Any]]):
    """
    Generates publication-ready PNG plots for Stage 2 fine-tuning.
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
    plt.title(f"Transfer Learning Stage 2 Fine-Tuning ({PRETRAINED_BACKBONE}) - Accuracy", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Categorical Accuracy", fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(STAGE2_ACC_PLOT_PNG, dpi=300)
    plt.close()
    print(f"Saved accuracy plot: {STAGE2_ACC_PLOT_PNG}")

    # --- Plot 2: Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.0)
    plt.plot(epochs, val_loss, label="Validation Loss", color="#d62728", linewidth=2.0)
    plt.axvline(x=best_epoch_acc, color="green", linestyle="--", alpha=0.7, label=f"Best Model (Epoch {best_epoch_acc})")
    plt.title(f"Transfer Learning Stage 2 Fine-Tuning ({PRETRAINED_BACKBONE}) - Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Categorical Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(STAGE2_LOSS_PLOT_PNG, dpi=300)
    plt.close()
    print(f"Saved loss plot: {STAGE2_LOSS_PLOT_PNG}")

def run_stage2_finetuning():
    print("=" * 80)
    print("STEP 8D: CONTROLLED FINE-TUNING OF MOBILENETV3LARGE (STAGE 2)")
    print("=" * 80)

    # 1. Environment & Reproducibility
    print("\n[1] Environment Confirmation:")
    print(f"  Python executable   : {sys.executable}")
    print(f"  TensorFlow version  : {tf.__version__}")
    print(f"  Pretrained Backbone : {PRETRAINED_BACKBONE}")
    print(f"  Random Seed         : {RANDOM_SEED}")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # 2. Load BEST Stage-1 Checkpoint (NOT the final epoch)
    print(f"\n[2] Loading BEST Stage-1 Checkpoint from: {STAGE1_BEST_MODEL_PATH}...")
    if not STAGE1_BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing Stage 1 best model: {STAGE1_BEST_MODEL_PATH}")

    model = tf.keras.models.load_model(STAGE1_BEST_MODEL_PATH)
    print("  Successfully loaded BEST Stage-1 model weights!")

    # 3. Configure Controlled Fine-Tuning
    print(f"\n[3] Configuring Backbone for Stage-2 Fine-Tuning (Top {FINE_TUNE_PERCENT * 100:.0f}% Unfrozen):")
    backbone = model.get_layer("MobileNetV3Large")
    backbone.trainable = True

    total_backbone_layers = len(backbone.layers)
    num_unfreeze = int(np.ceil(total_backbone_layers * FINE_TUNE_PERCENT))
    freeze_until = total_backbone_layers - num_unfreeze

    frozen_backbone_layers = 0
    trainable_backbone_layers = 0
    bn_layers_kept_frozen = 0

    for idx, layer in enumerate(backbone.layers):
        if idx < freeze_until:
            layer.trainable = False
            frozen_backbone_layers += 1
        else:
            # Safest BatchNorm strategy: keep BN layers frozen to preserve ImageNet running statistics
            if isinstance(layer, layers.BatchNormalization) or "batch_norm" in layer.name.lower() or "bn" in layer.name.lower():
                layer.trainable = False
                bn_layers_kept_frozen += 1
                frozen_backbone_layers += 1
            else:
                layer.trainable = True
                trainable_backbone_layers += 1

    # Re-compile with small fine-tuning learning rate (1e-4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=STAGE2_INITIAL_LR)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy")]
    )

    total_params = model.count_params()
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    non_trainable_params = sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)

    print(f"  Total Backbone Layers        : {total_backbone_layers}")
    print(f"  Frozen Backbone Layers       : {frozen_backbone_layers} ({freeze_until} base + {bn_layers_kept_frozen} BN)")
    print(f"  Trainable Backbone Layers    : {trainable_backbone_layers}")
    print(f"  BN Layers Kept Frozen        : {bn_layers_kept_frozen} (Strictly preserved to prevent covariate shift)")
    print(f"  Total Parameters             : {total_params:,}")
    print(f"  Total Trainable Parameters   : {trainable_params:,}")
    print(f"  Total Non-trainable Params   : {non_trainable_params:,}")
    print(f"  Fine-Tuning Learning Rate    : {STAGE2_INITIAL_LR} (Adam)")

    # 4. Load Dataset Partitions
    splits_dir = DATA_DIR / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "validation.csv"
    test_csv = splits_dir / "test.csv"

    with open(test_csv, "r", encoding="utf-8") as f:
        test_count = sum(1 for _ in f) - 1

    print("\n[4] Loading Data Partitions:")
    class_mapping = get_class_mapping()
    train_images, train_labels_oh, train_labels_int, _ = load_partition_data(
        train_csv, class_mapping=class_mapping, num_classes=NUM_CLASSES, verbose=True
    )
    val_images, val_labels_oh, val_labels_int, _ = load_partition_data(
        val_csv, class_mapping=class_mapping, num_classes=NUM_CLASSES, verbose=True
    )
    print(f"  Train sample count      : {len(train_images)}")
    print(f"  Validation sample count : {len(val_images)}")
    print(f"  Test sample count       : {test_count} (COMPLETELY UNTOUCHED - NO EVALUATION)")

    # 5. Class Weights (TRAIN LABELS ONLY)
    print("\n[5] Calculating Balanced Class Weights from Training Labels Only...")
    class_weights = compute_balanced_class_weights(train_labels_int, num_classes=NUM_CLASSES)
    present_weights = {k: v for k, v in class_weights.items() if k in set(train_labels_int)}
    min_w = min(present_weights.values())
    max_w = max(present_weights.values())
    print(f"  Class weight range in train : {min_w:.4f} to {max_w:.4f}")

    # 6. Data Generators
    print("\n[6] Constructing Data Generators:")
    augmenter = KuchipudiAugmenter(enable=True, seed=RANDOM_SEED)
    print("  TRAIN generator      : On-the-fly Augmentation ENABLED")
    print("  VALIDATION generator : Augmentation STRICTLY DISABLED (Identity tensors, 0% modification)")
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

    # 7. Callbacks Setup
    print("\n[7] Configuring Callbacks for Stage 2:")
    print(f"  - ModelCheckpoint   : saving best model to {STAGE2_BEST_MODEL_PATH} (monitor=val_accuracy, mode=max)")
    print(f"  - EarlyStopping     : monitor=val_loss, patience={STAGE2_EARLY_PATIENCE}, restore_best_weights=True, mode=min")
    print(f"  - ReduceLROnPlateau : monitor=val_loss, factor={STAGE2_LR_FACTOR}, patience={STAGE2_LR_PATIENCE}, min_lr={STAGE2_MIN_LR}, mode=min")

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(STAGE2_BEST_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max"
    )

    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=STAGE2_EARLY_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode="min"
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=STAGE2_LR_FACTOR,
        patience=STAGE2_LR_PATIENCE,
        min_lr=STAGE2_MIN_LR,
        verbose=1,
        mode="min"
    )

    metrics_logger_cb = EpochMetricsLogger(csv_path=STAGE2_HISTORY_CSV)

    callbacks_list = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, metrics_logger_cb]

    # 8. Run Stage 2 Fine-Tuning
    print(f"\n[8] Starting Stage 2 Fine-Tuning (MAX_EPOCHS = {STAGE2_MAX_EPOCHS}, BATCH_SIZE = {BATCH_SIZE}, LR = {STAGE2_INITIAL_LR})...")
    start_time = time.time()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=STAGE2_MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=0
    )

    total_training_time = time.time() - start_time
    print(f"\nStage 2 fine-tuning completed in {total_training_time:.1f}s ({total_training_time/60.0:.2f} mins).")

    # 9. Extract Metrics from History
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

    early_stopped = epochs_completed < STAGE2_MAX_EPOCHS
    lr_reductions = sum(1 for i in range(1, len(learning_rates)) if learning_rates[i] < learning_rates[i-1])

    # 10. Plot Curves
    print("\n[9] Generating Visualizations...")
    plot_training_curves(records)

    # 11. Generate Stage 2 Report
    print(f"\n[10] Generating Stage 2 Report in {STAGE2_REPORT_TXT}...")
    with open(STAGE2_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KUCHIPUDI CLASSICAL DANCE RECOGNITION - STAGE 2 FINE-TUNING REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. MODEL ARCHITECTURE & FINE-TUNING SPECIFICATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Pretrained Backbone          : {PRETRAINED_BACKBONE} (ImageNet-1k)\n")
        f.write(f"- Starting Model Checkpoint    : {STAGE1_BEST_MODEL_PATH} (Stage 1 Best Epoch 22)\n")
        f.write(f"- Total Backbone Layers        : {total_backbone_layers}\n")
        f.write(f"- Frozen Backbone Layers       : {frozen_backbone_layers} ({freeze_until} base + {bn_layers_kept_frozen} BN)\n")
        f.write(f"- Trainable Backbone Layers    : {trainable_backbone_layers} (Top {FINE_TUNE_PERCENT*100:.0f}% Inverted Residuals)\n")
        f.write(f"- BatchNorm Freeze Strategy    : All {bn_layers_kept_frozen} BN layers in backbone frozen (ImageNet statistics preserved)\n")
        f.write(f"- Total Parameters             : {total_params:,}\n")
        f.write(f"- Trainable Parameters         : {trainable_params:,}\n")
        f.write(f"- Non-trainable Parameters     : {non_trainable_params:,}\n")
        f.write(f"- Initial Fine-Tuning LR       : {STAGE2_INITIAL_LR} (Adam)\n\n")

        f.write("2. DATASET PARTITIONING & LEAKAGE CONTROL:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Training Samples             : {len(train_images)} (Holdout training partition)\n")
        f.write(f"- Validation Samples           : {len(val_images)} (Holdout validation partition)\n")
        f.write(f"- Test Samples                 : {test_count} (COMPLETELY UNTOUCHED held-out test partition)\n")
        f.write("- Validation Augmentation      : STRICTLY DISABLED (Identity tensors, 0% modification)\n")
        f.write("- Test Set Status              : STRICTLY ISOLATED & UNTOUCHED (Zero evaluation performed)\n\n")

        f.write("3. EMPIRICAL RESULTS (STAGE 2 FINE-TUNING):\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Total Training Time          : {total_training_time:.1f} seconds ({total_training_time/60.0:.2f} minutes)\n")
        f.write(f"- Actual Epochs Completed      : {epochs_completed} / {STAGE2_MAX_EPOCHS}\n")
        f.write(f"- Best Model Epoch (val_acc)   : Epoch {best_epoch_acc}\n")
        f.write(f"- Best Validation Accuracy     : {best_val_acc * 100:.2f}%\n")
        f.write(f"- Best Training Accuracy       : {best_train_acc * 100:.2f}%\n")
        f.write(f"- Best Validation Loss         : {best_val_loss:.4f}\n")
        f.write(f"- Final Validation Accuracy    : {final_val_acc * 100:.2f}%\n")
        f.write(f"- Final Validation Loss        : {final_val_loss:.4f}\n")
        f.write(f"- Final Training Accuracy      : {final_train_acc * 100:.2f}%\n")
        f.write(f"- Early Stopping Triggered     : {early_stopped} ({'Triggered at epoch ' + str(epochs_completed) if early_stopped else 'Completed all epochs'})\n")
        f.write(f"- Learning Rate Reductions     : {lr_reductions} times\n\n")

        f.write("4. SYSTEMATIC PROGRESSION COMPARISON:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Experiment Phase':<32} | {'Validation Accuracy':<22} | {'Validation Loss':<18}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'1. Scratch CNN Baseline':<32} | {'9.45%':<22} | {'5.4429':<18}\n")
        f.write(f"{'2. Transfer Stage 1 (Head-only)':<32} | {'26.77%':<22} | {'2.8075':<18}\n")
        f.write(f"{'3. Transfer Stage 2 (Fine-tune)':<32} | {f'{best_val_acc * 100:.2f}%':<22} | {f'{best_val_loss:.4f}':<18}\n")
        f.write("-" * 80 + "\n\n")

        f.write("5. ARTIFACT PERSISTENCE LOCATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- Best Fine-Tuned Model Saved  : {STAGE2_BEST_MODEL_PATH}\n")
        f.write(f"- Training History CSV         : {STAGE2_HISTORY_CSV}\n")
        f.write(f"- Accuracy Curves Plot         : {STAGE2_ACC_PLOT_PNG}\n")
        f.write(f"- Loss Curves Plot             : {STAGE2_LOSS_PLOT_PNG}\n\n")

        f.write("6. ACADEMIC INTEGRITY DECLARATION:\n")
        f.write("-" * 80 + "\n")
        f.write("This report presents the actual measured performance of Stage 2 fine-tuning.\n")
        f.write("The 95% target is an engineering goal and was not fabricated.\n")
        f.write("The test dataset remains 100% untouched and un-evaluated.\n")

    print(f"\nStage 2 report created: {STAGE2_REPORT_TXT}")
    print("=" * 80)
    print("STEP 8D COMPLETE: Stage 2 fine-tuning finished. STOPPING.")
    print("=" * 80)

if __name__ == "__main__":
    run_stage2_finetuning()
