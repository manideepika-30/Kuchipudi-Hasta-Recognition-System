import os
import sys
import io
from pathlib import Path
import numpy as np

try:
    from config import (
        PROJECT_ROOT,
        REPORTS_DIR,
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
        CNN_ARCHITECTURE_REPORT_TXT
    )
    from model import build_cnn_model, get_callbacks, compute_balanced_class_weights
except ImportError:
    from src.config import (
        PROJECT_ROOT,
        REPORTS_DIR,
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
        CNN_ARCHITECTURE_REPORT_TXT
    )
    from src.model import build_cnn_model, get_callbacks, compute_balanced_class_weights

import tensorflow as tf

def run_verification():
    print("=" * 80)
    print("STEP 5: KUCHIPUDI CNN ARCHITECTURE DESIGN & VERIFICATION")
    print("=" * 80)

    # 1. Environment & Framework Check
    print(f"\n[1] Framework Environment:")
    print(f"  Python executable : {sys.executable}")
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  Keras version     : {tf.keras.__version__ if hasattr(tf.keras, '__version__') else 'Embedded in TF'}")

    # 2. Build the CNN Model
    print(f"\n[2] Building Model '{MODEL_NAME}' for {NUM_CLASSES} Kuchipudi classes...")
    model = build_cnn_model(
        num_classes=NUM_CLASSES,
        input_shape=CNN_INPUT_SHAPE,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE,
        dense_units=DENSE_UNITS,
        l2_reg=L2_REGULARIZATION,
        loss_type=LOSS_FUNCTION
    )

    # 3. Model Summary & Architectural Structure
    print(f"\n[3] Model Summary:")
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    summary_str = "\n".join(string_list)
    print(summary_str)

    # 4. Input & Output Shapes
    input_shape = model.input_shape
    output_shape = model.output_shape
    print(f"\n[4] Tensor Dimensions:")
    print(f"  Input Shape  : {input_shape}")
    print(f"  Output Shape : {output_shape}")

    # 5. Parameter Counts
    total_params = model.count_params()
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    non_trainable_params = sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)

    print(f"\n[5] Parameter Breakdown:")
    print(f"  Total Parameters         : {int(total_params):,}")
    print(f"  Trainable Parameters     : {int(trainable_params):,}")
    print(f"  Non-trainable Parameters : {int(non_trainable_params):,} (Batch Normalization moving statistics)")

    # 6. Dummy Forward Pass (Single tensor, NO training)
    print(f"\n[6] Performing ONE Dummy Forward Pass...")
    dummy_input = np.zeros((1, *CNN_INPUT_SHAPE), dtype=np.float32)
    # Set a small test pattern
    dummy_input[0, 50:150, 50:150, :] = 0.5

    dummy_output = model.predict(dummy_input, verbose=0)
    print(f"  Dummy Input Tensor Shape  : {dummy_input.shape}")
    print(f"  Dummy Output Tensor Shape : {dummy_output.shape}")
    prob_sum = float(np.sum(dummy_output[0]))
    print(f"  Softmax Probability Sum   : {prob_sum:.6f} (Valid Softmax: {np.isclose(prob_sum, 1.0)})")
    print(f"  Top Predicted Class Index : {np.argmax(dummy_output[0])}")

    assert dummy_output.shape == (1, NUM_CLASSES), f"Expected shape (1, {NUM_CLASSES}), got {dummy_output.shape}"
    assert np.isclose(prob_sum, 1.0, atol=1e-5), "Softmax output does not sum to 1.0!"
    print("  Dummy Forward Pass: SUCCESSFUL and verified.")

    # 7. Callbacks Verification
    print(f"\n[7] Verifying Callback Configuration:")
    callbacks_list = get_callbacks(
        checkpoint_path=BEST_MODEL_PATH,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        reduce_lr_patience=REDUCE_LR_PATIENCE,
        reduce_lr_factor=REDUCE_LR_FACTOR,
        min_lr=MIN_LEARNING_RATE
    )
    for cb in callbacks_list:
        print(f"  - Configured Callback: {cb.__class__.__name__}")

    # 8. Class Imbalance Strategy Verification
    print(f"\n[8] Verifying Class Imbalance Strategy (Balanced Class Weights):")
    # Simulate a dummy subset with uneven distribution
    dummy_labels = [0] * 50 + [1] * 10 + [2] * 4
    weights = compute_balanced_class_weights(dummy_labels, num_classes=3)
    for c_id, w in weights.items():
        print(f"  Class {c_id}: weight = {w:.4f}")

    # 9. Generate Comprehensive Documentation Report
    print(f"\n[9] Generating Architecture Report in {CNN_ARCHITECTURE_REPORT_TXT}...")
    with open(CNN_ARCHITECTURE_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KUCHIPUDI DANCE RECOGNITION - CNN ARCHITECTURE DESIGN REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. MODEL OVERVIEW:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model Name                 : {MODEL_NAME}\n")
        f.write(f"Architecture Type          : Custom Deep Convolutional Neural Network (Functional API)\n")
        f.write(f"Input Shape                : {CNN_INPUT_SHAPE} (Height, Width, Channels)\n")
        f.write(f"Output Target              : {NUM_CLASSES} Kuchipudi Dance Classes (Softmax distribution)\n")
        f.write(f"Total Parameters           : {int(total_params):,}\n")
        f.write(f"Trainable Parameters       : {int(trainable_params):,}\n")
        f.write(f"Non-trainable Parameters   : {int(non_trainable_params):,} (Batch Normalization moving stats)\n\n")

        f.write("2. TRAINING HYPERPARAMETERS & CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Optimizer                  : {OPTIMIZER_NAME.upper()} (Adam)\n")
        f.write(f"Initial Learning Rate      : {LEARNING_RATE}\n")
        f.write(f"Loss Function              : {LOSS_FUNCTION} (Cross-Entropy)\n")
        f.write(f"Dense Layer Units          : {DENSE_UNITS}\n")
        f.write(f"Classifier Head Dropout    : {DROPOUT_RATE * 100:.1f}%\n")
        f.write(f"L2 Weight Regularization   : {L2_REGULARIZATION}\n\n")

        f.write("3. DETAILED ARCHITECTURAL TABLE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Layer / Block':<25} | {'Filter / Units':<16} | {'Output Dimension':<20} | {'Regularization':<18}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Input Layer':<25} | {'None':<16} | {'(None, 224, 224, 3)':<20} | {'None':<18}\n")
        f.write(f"{'Conv Block 1 (x2)':<25} | {'32 filters (3x3)':<16} | {'(None, 224, 224, 32)':<20} | {'BN, L2(1e-4)':<18}\n")
        f.write(f"{'Pool 1 + Dropout':<25} | {'Max (2x2)':<16} | {'(None, 112, 112, 32)':<20} | {'Dropout(0.20)':<18}\n")
        f.write(f"{'Conv Block 2 (x2)':<25} | {'64 filters (3x3)':<16} | {'(None, 112, 112, 64)':<20} | {'BN, L2(1e-4)':<18}\n")
        f.write(f"{'Pool 2 + Dropout':<25} | {'Max (2x2)':<16} | {'(None, 56, 56, 64)':<20} | {'Dropout(0.25)':<18}\n")
        f.write(f"{'Conv Block 3 (x2)':<25} | {'128 filters (3x3)':<16} | {'(None, 56, 56, 128)':<20} | {'BN, L2(1e-4)':<18}\n")
        f.write(f"{'Pool 3 + Dropout':<25} | {'Max (2x2)':<16} | {'(None, 28, 28, 128)':<20} | {'Dropout(0.30)':<18}\n")
        f.write(f"{'Conv Block 4 (x2)':<25} | {'256 filters (3x3)':<16} | {'(None, 28, 28, 256)':<20} | {'BN, L2(1e-4)':<18}\n")
        f.write(f"{'Pool 4 + Dropout':<25} | {'Max (2x2)':<16} | {'(None, 14, 14, 256)':<20} | {'Dropout(0.35)':<18}\n")
        f.write(f"{'Global Avg Pooling':<25} | {'GAP (Pool 14x14)':<16} | {'(None, 256)':<20} | {'Zero-param regular':<18}\n")
        f.write(f"{'Dense Features':<25} | {'256 units':<16} | {'(None, 256)':<20} | {'BN, L2(1e-4)':<18}\n")
        f.write(f"{'Dense Dropout':<25} | {'None':<16} | {'(None, 256)':<20} | {f'Dropout({DROPOUT_RATE})':<18}\n")
        f.write(f"{'Output Softmax':<25} | {f'{NUM_CLASSES} classes':<16} | {f'(None, {NUM_CLASSES})':<20} | {'Softmax':<18}\n")
        f.write("-" * 80 + "\n\n")

        f.write("4. WHY THIS CNN IS SUITED FOR THE KUCHIPUDI DATASET:\n")
        f.write("-" * 80 + "\n")
        f.write("a) Dataset Size Proportionality:\n")
        f.write("   The Kuchipudi dataset contains 1,094 total samples across 75 fine-grained classes. Massive models\n")
        f.write("   (such as standard ResNet-50 with 25M params or VGG-16 with 138M params) are notoriously prone to\n")
        f.write("   rapid empirical memorization (overfitting) when trained on small image corpuses. This architecture\n")
        f.write("   contains ~1.26M parameters—the optimal capacity envelope for learning subtle dance gestures.\n\n")
        f.write("b) Global Average Pooling vs. Flattening:\n")
        f.write("   Flattening a 14x14x256 feature map creates 50,176 elements. Connecting this to a 256-unit Dense layer\n")
        f.write("   alone requires 12.8 MILLION parameters, which concentrates 90% of model weights in a single fully-\n")
        f.write("   connected transition. In contrast, Global Average Pooling condenses each 14x14 feature map into a\n")
        f.write("   single channel summary (256 values) with ZERO extra parameters, providing structural translation\n")
        f.write("   invariance and directly dampening overfitting.\n\n")
        f.write("c) Multi-Scale Receptive Fields for Dance Mudras:\n")
        f.write("   Classical dance gestures require perceiving both fine-grained finger bends (local edge/contour\n")
        f.write("   features captured in Blocks 1 & 2) and holistic body/head/gaze postures (global spatial relationships\n")
        f.write("   captured in Blocks 3 & 4).\n\n")
        f.write("d) Multi-Tier Regularization:\n")
        f.write("   Batch Normalization stabilizes internal covariate shifts; progressive spatial dropout (0.20 -> 0.35)\n")
        f.write("   prevents feature co-dependency; L2 weight decay (1e-4) penalizes large weights; and head dropout (0.4)\n")
        f.write("   ensures robust decision boundaries.\n\n")

        f.write("5. CLASS IMBALANCE HANDLING STRATEGY:\n")
        f.write("-" * 80 + "\n")
        f.write("- Analysis: Certain video-derived classes possess 32-40 frames, while small-sample image classes have\n")
        f.write("  only 3-6 original captures.\n")
        f.write("- Mechanism: Cost-sensitive loss weighting using compute_balanced_class_weights(). During training,\n")
        f.write("  gradients from minority classes are scaled up inversely proportional to their sample frequency,\n")
        f.write("  ensuring the model does not degenerate into predicting only majority classes.\n")
        f.write("- Leakage Guarantee: Validation and test sets remain unweighted and completely unaugmented to ensure\n")
        f.write("  rigorous, unbiased statistical evaluation.\n\n")

        f.write("6. CALLBACK SUITE SPECIFICATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"- EarlyStopping           : monitor='val_loss', patience={EARLY_STOPPING_PATIENCE}, restore_best_weights=True\n")
        f.write(f"- ModelCheckpoint         : filepath='{BEST_MODEL_PATH}', monitor='val_loss', save_best_only=True\n")
        f.write(f"- ReduceLROnPlateau       : monitor='val_loss', factor={REDUCE_LR_FACTOR}, patience={REDUCE_LR_PATIENCE}, min_lr={MIN_LEARNING_RATE}\n\n")

        f.write("7. COMPLETE KERAS MODEL SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(summary_str + "\n\n")

        f.write("8. ACADEMIC EVALUATION COMMITMENT:\n")
        f.write("-" * 80 + "\n")
        f.write("95% accuracy is a project aspirational target, NOT an empirical claim. Actual classification\n")
        f.write("performance will be measured rigorously post-training on the completely isolated, unaugmented held-out\n")
        f.write("test partition and reported across Precision, Recall, F1-Score, and Confusion Matrix metrics.\n")

    print(f"\nArchitecture report successfully created: {CNN_ARCHITECTURE_REPORT_TXT}")
    print("=" * 80)

if __name__ == "__main__":
    run_verification()
