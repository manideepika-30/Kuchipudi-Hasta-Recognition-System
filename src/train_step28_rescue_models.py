"""
STEP 28 — Final Recognition Rescue Model Trainer & Benchmark
============================================================
Comprehensive benchmark of candidate models across all 4 categories:
  1. Hasthas (52 classes)
  2. Shiro Bhedhas (9 classes)
  3. Drishti Bhedhas (8 classes)
  4. Pada Karma (6 classes)

Candidate Classifiers Evaluated per Category:
  1. RBF SVM (class-balanced)
  2. Random Forest (class-balanced)
  3. Compact MLP
  4. k-NN / Cosine Nearest Neighbors (k=1, 3, 5)
  5. Landmark-space data augmentation (Hasthas, Train-only)
  6. Multimodal fusion (Visual CNN embedding + Landmark / Temporal)

Strict Integrity:
  - TRAIN data ONLY for fitting
  - VALIDATION data ONLY for comparison & model selection
  - Locked test set (v2_test.csv) STRICTLY EXCLUDED
"""

import os
import sys
import pickle
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# Realistic Landmark-Space Augmentation for Hasthas (Train Only)
# =============================================================================

def augment_hasthas_landmarks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    multiplier: int = 2,
    noise_std: float = 0.015,
    max_angle_deg: float = 8.0,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    trans_range: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies realistic 3D landmark geometric augmentation strictly to training data:
      - Coordinate jitter
      - Small 3D planar rotation (around z-axis)
      - Small uniform scaling
      - Small translation
    Preserves distance positivity and normalized angle bounds in [0, 1].
    NO horizontal flipping.
    """
    X_aug_list = [X_train]
    y_aug_list = [y_train]

    N, D = X_train.shape
    rad_max = np.radians(max_angle_deg)

    for m in range(multiplier):
        X_syn = np.copy(X_train)
        for i in range(N):
            theta = np.random.uniform(-rad_max, rad_max)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = np.array([
                [cos_t, -sin_t, 0.0],
                [sin_t,  cos_t, 0.0],
                [0.0,    0.0,   1.0]
            ], dtype=np.float32)

            scale = np.random.uniform(scale_range[0], scale_range[1])
            trans = np.random.uniform(-trans_range, trans_range, size=(3,)).astype(np.float32)

            # Hand 1 Coords: 0..63 (21 joints * 3)
            h1 = X_syn[i, 0:63].reshape(21, 3)
            if np.any(h1 != 0):
                h1_aug = (h1 @ R.T) * scale + trans
                h1_aug += np.random.normal(0, noise_std, size=h1_aug.shape)
                X_syn[i, 0:63] = h1_aug.flatten()

            # Hand 1 Distances: 63..77 (14 distances)
            d1 = X_syn[i, 63:77]
            if np.any(d1 != 0):
                d1_aug = d1 * scale + np.random.normal(0, noise_std * 0.5, size=d1.shape)
                X_syn[i, 63:77] = np.maximum(0.0, d1_aug)

            # Hand 1 Angles: 77..87 (10 normalized angles in [0, 1])
            a1 = X_syn[i, 77:87]
            if np.any(a1 != 0):
                a1_aug = a1 + np.random.normal(0, 0.02, size=a1.shape)
                X_syn[i, 77:87] = np.clip(a1_aug, 0.0, 1.0)

            # Hand 2 Coords: 87..150
            h2 = X_syn[i, 87:150].reshape(21, 3)
            if np.any(h2 != 0):
                h2_aug = (h2 @ R.T) * scale + trans
                h2_aug += np.random.normal(0, noise_std, size=h2_aug.shape)
                X_syn[i, 87:150] = h2_aug.flatten()

            # Hand 2 Distances: 150..164
            d2 = X_syn[i, 150:164]
            if np.any(d2 != 0):
                d2_aug = d2 * scale + np.random.normal(0, noise_std * 0.5, size=d2.shape)
                X_syn[i, 150:164] = np.maximum(0.0, d2_aug)

            # Hand 2 Angles: 164..174
            a2 = X_syn[i, 164:174]
            if np.any(a2 != 0):
                a2_aug = a2 + np.random.normal(0, 0.02, size=a2.shape)
                X_syn[i, 164:174] = np.clip(a2_aug, 0.0, 1.0)

        X_aug_list.append(X_syn)
        y_aug_list.append(y_train)

    return np.vstack(X_aug_list), np.concatenate(y_aug_list)


# =============================================================================
# Model Evaluation Helper
# =============================================================================

def evaluate_model(
    model: Any,
    scaler: StandardScaler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int
) -> Dict[str, Any]:
    """
    Fits scaler on train, scales train and val, fits model, evaluates on val.
    """
    X_tr_s = scaler.fit_transform(X_train)
    X_vl_s = scaler.transform(X_val)

    t0 = time.time()
    model.fit(X_tr_s, y_train)
    train_time = time.time() - t0

    preds = model.predict(X_vl_s)
    top1 = accuracy_score(y_val, preds) * 100.0

    # Top-5 accuracy
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vl_s)
        # Handle cases where model only saw a subset of classes in train
        if probs.shape[1] < num_classes:
            full_probs = np.zeros((len(y_val), num_classes), dtype=np.float32)
            classes_seen = model.classes_
            for col_idx, cls in enumerate(classes_seen):
                full_probs[:, cls] = probs[:, col_idx]
            probs = full_probs
        k_val = min(5, num_classes)
        top5 = top_k_accuracy_score(y_val, probs, k=k_val, labels=np.arange(num_classes)) * 100.0
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X_vl_s)
        # Convert decision function to ranked top-5
        if dec.ndim == 1:
            top5 = top1
            probs = np.vstack([1.0 - dec, dec]).T
        else:
            k_val = min(5, num_classes)
            top5 = top_k_accuracy_score(y_val, dec, k=k_val, labels=np.arange(num_classes)) * 100.0
            probs = dec
    else:
        top5 = top1
        probs = None

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, preds, average='macro', zero_division=0
    )
    f1_macro *= 100.0
    r_macro *= 100.0

    # Count parameters
    params_count = "N/A"
    if isinstance(model, SVC):
        params_count = f"{len(model.support_)} SVs"
    elif isinstance(model, RandomForestClassifier):
        total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
        params_count = f"{total_nodes} nodes"
    elif isinstance(model, MLPClassifier):
        weights_count = sum(coef.size for coef in model.coefs_) + sum(b.size for b in model.intercepts_)
        params_count = f"{weights_count} weights"
    elif isinstance(model, KNeighborsClassifier):
        params_count = f"{len(X_train)} points"

    return {
        "model": model,
        "scaler": scaler,
        "top1": top1,
        "top5": top5,
        "f1_macro": f1_macro,
        "r_macro": r_macro,
        "train_time": train_time,
        "params": params_count,
        "preds": preds,
        "probs": probs
    }


# =============================================================================
# Benchmark Suite
# =============================================================================

def run_step28_rescue_benchmark():
    cache_path = PROJECT_ROOT / "scratch" / "step28_features_cache.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}. Run prepare_step28_data.py first.")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    print("=" * 80)
    print("STEP 28 — FINAL RECOGNITION RESCUE MODEL BENCHMARK")
    print("=" * 80)

    comparison_results = []
    selected_models = {}

    # -------------------------------------------------------------------------
    # 1. HASTHAS (52 classes)
    # -------------------------------------------------------------------------
    h_data = cache["1. Hasthas"]
    print("\n" + "=" * 80)
    print("PART 1: HASTHAS BENCHMARK (52 Classes, 416 Train, 85 Val)")
    print("=" * 80)

    X_lm_tr = h_data["X_lm_train"]
    X_vis_tr = h_data["X_vis_train"]
    y_tr = h_data["y_train"]

    X_lm_vl = h_data["X_lm_val"]
    X_vis_vl = h_data["X_vis_val"]
    y_vl = h_data["y_val"]
    num_classes = h_data["num_classes"]

    # Augmented landmarks
    X_lm_tr_aug, y_tr_aug = augment_hasthas_landmarks(X_lm_tr, y_tr, multiplier=2)
    print(f"Augmented Hasthas Landmark Train: {X_lm_tr.shape} -> {X_lm_tr_aug.shape}")

    # Multimodal features: 960 visual + 175 landmarks
    X_multi_tr = np.hstack([X_vis_tr, X_lm_tr])
    X_multi_vl = np.hstack([X_vis_vl, X_lm_vl])

    hasthas_experiments = [
        # Landmark Candidates
        ("Hasthas", "RBF SVM (C=2.0, balanced)", "Landmark (175)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_lm_tr, y_tr, X_lm_vl, y_vl),
        ("Hasthas", "RBF SVM (C=5.0, balanced)", "Landmark (175)",
         SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_lm_tr, y_tr, X_lm_vl, y_vl),
        ("Hasthas", "Random Forest (300 trees, balanced)", "Landmark (175)",
         RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1),
         X_lm_tr, y_tr, X_lm_vl, y_vl),
        ("Hasthas", "Compact MLP (128-64, L2=1e-3)", "Landmark (175)",
         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=42),
         X_lm_tr, y_tr, X_lm_vl, y_vl),
        ("Hasthas", "k-NN (Cosine, k=1)", "Landmark (175)",
         KNeighborsClassifier(n_neighbors=1, metric='cosine', weights='distance'),
         X_lm_tr, y_tr, X_lm_vl, y_vl),
        ("Hasthas", "k-NN (Cosine, k=3)", "Landmark (175)",
         KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance'),
         X_lm_tr, y_tr, X_lm_vl, y_vl),
        ("Hasthas", "k-NN (Cosine, k=5)", "Landmark (175)",
         KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='distance'),
         X_lm_tr, y_tr, X_lm_vl, y_vl),

        # Landmark + Augmentation Candidates
        ("Hasthas", "RBF SVM (C=2.0, balanced, Aug)", "Landmark+Aug (175)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_lm_tr_aug, y_tr_aug, X_lm_vl, y_vl),
        ("Hasthas", "RBF SVM (C=5.0, balanced, Aug)", "Landmark+Aug (175)",
         SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_lm_tr_aug, y_tr_aug, X_lm_vl, y_vl),
        ("Hasthas", "Random Forest (300 trees, Aug)", "Landmark+Aug (175)",
         RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1),
         X_lm_tr_aug, y_tr_aug, X_lm_vl, y_vl),
        ("Hasthas", "Compact MLP (128-64, Aug)", "Landmark+Aug (175)",
         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=42),
         X_lm_tr_aug, y_tr_aug, X_lm_vl, y_vl),

        # Multimodal Rescue Candidates (Part 5)
        ("Hasthas", "Multimodal RBF SVM (C=2.0)", "Visual+LM (1135)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_multi_tr, y_tr, X_multi_vl, y_vl),
        ("Hasthas", "Multimodal Random Forest (300 trees)", "Visual+LM (1135)",
         RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1),
         X_multi_tr, y_tr, X_multi_vl, y_vl),
        ("Hasthas", "Multimodal Compact MLP (256-128)", "Visual+LM (1135)",
         MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=300, random_state=42),
         X_multi_tr, y_tr, X_multi_vl, y_vl)
    ]

    best_hasthas_res = None
    best_hasthas_name = None
    best_hasthas_top1 = -1.0

    for cat, name, ftype, model, X_tr, y_tr_curr, X_vl, y_vl_curr in hasthas_experiments:
        res = evaluate_model(model, StandardScaler(), X_tr, y_tr_curr, X_vl, y_vl_curr, num_classes)
        print(f"  {name:42s} | Top-1: {res['top1']:5.2f}% | Top-5: {res['top5']:5.2f}% | F1: {res['f1_macro']:5.2f}% | Params: {res['params']}")
        row_dict = {
            "CATEGORY": cat,
            "MODEL": name,
            "FEATURE_TYPE": ftype,
            "VALIDATION_TOP1": f"{res['top1']:.2f}%",
            "VALIDATION_TOP5": f"{res['top5']:.2f}%",
            "MACRO_F1": f"{res['f1_macro']:.2f}%",
            "PARAMETERS": res["params"],
            "SELECTED": "NO"
        }
        comparison_results.append((row_dict, res))

        # Selection criterion: Highest Top-1 accuracy, tie-break on Macro F1
        if res['top1'] > best_hasthas_top1 or (res['top1'] == best_hasthas_top1 and best_hasthas_res and res['f1_macro'] > best_hasthas_res['f1_macro']):
            best_hasthas_top1 = res['top1']
            best_hasthas_res = res
            best_hasthas_name = name

    print(f"\n>>> Selected Hasthas Rescue Model: {best_hasthas_name} (Top-1: {best_hasthas_top1:.2f}%)")
    selected_models["1. Hasthas"] = {
        "name": best_hasthas_name,
        "res": best_hasthas_res,
        "data": h_data
    }

    # -------------------------------------------------------------------------
    # 2. SHIRO BHEDHAS (9 classes)
    # -------------------------------------------------------------------------
    s_data = cache["2. Shiro Bhedhas (HEAD MOVEMENTS)"]
    print("\n" + "=" * 80)
    print("PART 2: SHIRO BHEDHAS BENCHMARK (9 Classes, 132 Train, 37 Val)")
    print("=" * 80)

    X_lm_tr = s_data["X_lm_train"]
    X_temp_tr = s_data["X_temp_train"]
    X_vis_tr = s_data["X_vis_train"]
    y_tr = s_data["y_train"]

    X_lm_vl = s_data["X_lm_val"]
    X_temp_vl = s_data["X_temp_val"]
    X_vis_vl = s_data["X_vis_val"]
    y_vl = s_data["y_val"]
    num_classes = s_data["num_classes"]

    X_nonvis_tr = np.hstack([X_lm_tr, X_temp_tr])
    X_nonvis_vl = np.hstack([X_lm_vl, X_temp_vl])

    X_multi_tr = np.hstack([X_vis_tr, X_nonvis_tr])
    X_multi_vl = np.hstack([X_vis_vl, X_nonvis_vl])

    shiro_experiments = [
        ("Shiro Bhedhas", "RBF SVM (LM only, C=2.0)", "Landmark (32)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_lm_tr, y_tr, X_lm_vl, y_vl),
        ("Shiro Bhedhas", "RBF SVM (LM+Temp, C=2.0)", "LM+Temp (288)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Shiro Bhedhas", "Random Forest (LM+Temp, 300 trees)", "LM+Temp (288)",
         RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Shiro Bhedhas", "Compact MLP (LM+Temp, 128-64)", "LM+Temp (288)",
         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=42),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Shiro Bhedhas", "k-NN (LM+Temp, Cosine, k=3)", "LM+Temp (288)",
         KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance'),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Shiro Bhedhas", "Multimodal RBF SVM (C=2.0)", "Visual+LM+Temp (1248)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_multi_tr, y_tr, X_multi_vl, y_vl),
        ("Shiro Bhedhas", "Multimodal Random Forest (300 trees)", "Visual+LM+Temp (1248)",
         RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
         X_multi_tr, y_tr, X_multi_vl, y_vl),
        ("Shiro Bhedhas", "Multimodal Compact MLP (128-64)", "Visual+LM+Temp (1248)",
         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=42),
         X_multi_tr, y_tr, X_multi_vl, y_vl)
    ]

    best_shiro_res = None
    best_shiro_name = None
    best_shiro_top1 = -1.0

    for cat, name, ftype, model, X_tr, y_tr_curr, X_vl, y_vl_curr in shiro_experiments:
        res = evaluate_model(model, StandardScaler(), X_tr, y_tr_curr, X_vl, y_vl_curr, num_classes)
        print(f"  {name:42s} | Top-1: {res['top1']:5.2f}% | Top-5: {res['top5']:5.2f}% | F1: {res['f1_macro']:5.2f}% | Params: {res['params']}")
        row_dict = {
            "CATEGORY": cat,
            "MODEL": name,
            "FEATURE_TYPE": ftype,
            "VALIDATION_TOP1": f"{res['top1']:.2f}%",
            "VALIDATION_TOP5": f"{res['top5']:.2f}%",
            "MACRO_F1": f"{res['f1_macro']:.2f}%",
            "PARAMETERS": res["params"],
            "SELECTED": "NO"
        }
        comparison_results.append((row_dict, res))

        if res['top1'] > best_shiro_top1 or (res['top1'] == best_shiro_top1 and best_shiro_res and res['f1_macro'] > best_shiro_res['f1_macro']):
            best_shiro_top1 = res['top1']
            best_shiro_res = res
            best_shiro_name = name

    print(f"\n>>> Selected Shiro Rescue Model: {best_shiro_name} (Top-1: {best_shiro_top1:.2f}%)")
    selected_models["2. Shiro Bhedhas (HEAD MOVEMENTS)"] = {
        "name": best_shiro_name,
        "res": best_shiro_res,
        "data": s_data
    }

    # -------------------------------------------------------------------------
    # 3. DRISHTI BHEDHAS (8 classes)
    # -------------------------------------------------------------------------
    d_data = cache["3. Drishti Bhedhas(EYE MOVEMENTS)"]
    print("\n" + "=" * 80)
    print("PART 3: DRISHTI BHEDHAS BENCHMARK (8 Classes, 80 Train, 29 Val)")
    print("=" * 80)

    X_lm_tr = d_data["X_lm_train"]
    X_temp_tr = d_data["X_temp_train"]
    X_vis_tr = d_data["X_vis_train"]
    y_tr = d_data["y_train"]

    X_lm_vl = d_data["X_lm_val"]
    X_temp_vl = d_data["X_temp_val"]
    X_vis_vl = d_data["X_vis_val"]
    y_vl = d_data["y_val"]
    num_classes = d_data["num_classes"]

    X_nonvis_tr = np.hstack([X_lm_tr, X_temp_tr])
    X_nonvis_vl = np.hstack([X_lm_vl, X_temp_vl])

    X_multi_tr = np.hstack([X_vis_tr, X_nonvis_tr])
    X_multi_vl = np.hstack([X_vis_vl, X_nonvis_vl])

    drishti_experiments = [
        ("Drishti Bhedhas", "RBF SVM (Gaze only, C=2.0)", "Gaze (44)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_lm_tr, y_tr, X_lm_vl, y_vl),
        ("Drishti Bhedhas", "RBF SVM (Gaze+Temp, C=2.0)", "Gaze+Temp (396)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Drishti Bhedhas", "Random Forest (Gaze+Temp, 300 trees)", "Gaze+Temp (396)",
         RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Drishti Bhedhas", "Compact MLP (Gaze+Temp, 128-64)", "Gaze+Temp (396)",
         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=42),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Drishti Bhedhas", "k-NN (Gaze+Temp, Cosine, k=3)", "Gaze+Temp (396)",
         KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance'),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Drishti Bhedhas", "Multimodal RBF SVM (C=2.0)", "Visual+Gaze+Temp (1356)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_multi_tr, y_tr, X_multi_vl, y_vl),
        ("Drishti Bhedhas", "Multimodal Random Forest (300 trees)", "Visual+Gaze+Temp (1356)",
         RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
         X_multi_tr, y_tr, X_multi_vl, y_vl),
        ("Drishti Bhedhas", "Multimodal Compact MLP (128-64)", "Visual+Gaze+Temp (1356)",
         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=42),
         X_multi_tr, y_tr, X_multi_vl, y_vl)
    ]

    best_drishti_res = None
    best_drishti_name = None
    best_drishti_top1 = -1.0

    for cat, name, ftype, model, X_tr, y_tr_curr, X_vl, y_vl_curr in drishti_experiments:
        res = evaluate_model(model, StandardScaler(), X_tr, y_tr_curr, X_vl, y_vl_curr, num_classes)
        print(f"  {name:42s} | Top-1: {res['top1']:5.2f}% | Top-5: {res['top5']:5.2f}% | F1: {res['f1_macro']:5.2f}% | Params: {res['params']}")
        row_dict = {
            "CATEGORY": cat,
            "MODEL": name,
            "FEATURE_TYPE": ftype,
            "VALIDATION_TOP1": f"{res['top1']:.2f}%",
            "VALIDATION_TOP5": f"{res['top5']:.2f}%",
            "MACRO_F1": f"{res['f1_macro']:.2f}%",
            "PARAMETERS": res["params"],
            "SELECTED": "NO"
        }
        comparison_results.append((row_dict, res))

        if res['top1'] > best_drishti_top1 or (res['top1'] == best_drishti_top1 and best_drishti_res and res['f1_macro'] > best_drishti_res['f1_macro']):
            best_drishti_top1 = res['top1']
            best_drishti_res = res
            best_drishti_name = name

    print(f"\n>>> Selected Drishti Rescue Model: {best_drishti_name} (Top-1: {best_drishti_top1:.2f}%)")
    selected_models["3. Drishti Bhedhas(EYE MOVEMENTS)"] = {
        "name": best_drishti_name,
        "res": best_drishti_res,
        "data": d_data
    }

    # -------------------------------------------------------------------------
    # 4. PADA KARMA (6 classes)
    # -------------------------------------------------------------------------
    p_data = cache["4. Pada Karma(Foot movements)"]
    print("\n" + "=" * 80)
    print("PART 4: PADA KARMA BENCHMARK (6 Classes, 88 Train, 48 Val)")
    print("=" * 80)

    X_lm_tr = p_data["X_lm_train"]
    X_temp_tr = p_data["X_temp_train"]
    X_vis_tr = p_data["X_vis_train"]
    y_tr = p_data["y_train"]

    X_lm_vl = p_data["X_lm_val"]
    X_temp_vl = p_data["X_temp_val"]
    X_vis_vl = p_data["X_vis_val"]
    y_vl = p_data["y_val"]
    num_classes = p_data["num_classes"]

    X_nonvis_tr = np.hstack([X_lm_tr, X_temp_tr])
    X_nonvis_vl = np.hstack([X_lm_vl, X_temp_vl])

    X_multi_tr = np.hstack([X_vis_tr, X_nonvis_tr])
    X_multi_vl = np.hstack([X_vis_vl, X_nonvis_vl])

    pada_experiments = [
        ("Pada Karma", "RBF SVM (Lower-body only, C=2.0)", "Landmark (40)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_lm_tr, y_tr, X_lm_vl, y_vl),
        ("Pada Karma", "RBF SVM (LM+Temp, C=2.0)", "LM+Temp (360)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Pada Karma", "Random Forest (LM+Temp, 300 trees)", "LM+Temp (360)",
         RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Pada Karma", "Compact MLP (LM+Temp, 128-64)", "LM+Temp (360)",
         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=42),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Pada Karma", "k-NN (LM+Temp, Cosine, k=3)", "LM+Temp (360)",
         KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance'),
         X_nonvis_tr, y_tr, X_nonvis_vl, y_vl),
        ("Pada Karma", "Multimodal RBF SVM (C=2.0)", "Visual+LM+Temp (1320)",
         SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
         X_multi_tr, y_tr, X_multi_vl, y_vl),
        ("Pada Karma", "Multimodal Random Forest (300 trees)", "Visual+LM+Temp (1320)",
         RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
         X_multi_tr, y_tr, X_multi_vl, y_vl),
        ("Pada Karma", "Multimodal Compact MLP (128-64)", "Visual+LM+Temp (1320)",
         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=42),
         X_multi_tr, y_tr, X_multi_vl, y_vl)
    ]

    best_pada_res = None
    best_pada_name = None
    best_pada_top1 = -1.0

    for cat, name, ftype, model, X_tr, y_tr_curr, X_vl, y_vl_curr in pada_experiments:
        res = evaluate_model(model, StandardScaler(), X_tr, y_tr_curr, X_vl, y_vl_curr, num_classes)
        print(f"  {name:42s} | Top-1: {res['top1']:5.2f}% | Top-5: {res['top5']:5.2f}% | F1: {res['f1_macro']:5.2f}% | Params: {res['params']}")
        row_dict = {
            "CATEGORY": cat,
            "MODEL": name,
            "FEATURE_TYPE": ftype,
            "VALIDATION_TOP1": f"{res['top1']:.2f}%",
            "VALIDATION_TOP5": f"{res['top5']:.2f}%",
            "MACRO_F1": f"{res['f1_macro']:.2f}%",
            "PARAMETERS": res["params"],
            "SELECTED": "NO"
        }
        comparison_results.append((row_dict, res))

        if res['top1'] > best_pada_top1 or (res['top1'] == best_pada_top1 and best_pada_res and res['f1_macro'] > best_pada_res['f1_macro']):
            best_pada_top1 = res['top1']
            best_pada_res = res
            best_pada_name = name

    print(f"\n>>> Selected Pada Rescue Model: {best_pada_name} (Top-1: {best_pada_top1:.2f}%)")
    selected_models["4. Pada Karma(Foot movements)"] = {
        "name": best_pada_name,
        "res": best_pada_res,
        "data": p_data
    }

    # Mark SELECTED in comparison_results
    for row, res in comparison_results:
        cat_key = row["CATEGORY"]
        if "Hasthas" in cat_key and row["MODEL"] == best_hasthas_name:
            row["SELECTED"] = "YES (BEST)"
        elif "Shiro" in cat_key and row["MODEL"] == best_shiro_name:
            row["SELECTED"] = "YES (BEST)"
        elif "Drishti" in cat_key and row["MODEL"] == best_drishti_name:
            row["SELECTED"] = "YES (BEST)"
        elif "Pada" in cat_key and row["MODEL"] == best_pada_name:
            row["SELECTED"] = "YES (BEST)"

    # Save Selected Models to models/
    models_dir = PROJECT_ROOT / "models"
    for cat_name, sel_info in selected_models.items():
        prefix = sel_info["data"]["prefix"]
        model_file = models_dir / f"rescue_{prefix}_best.joblib"
        scaler_file = models_dir / f"rescue_{prefix}_scaler.joblib"
        joblib.dump(sel_info["res"]["model"], model_file)
        joblib.dump(sel_info["res"]["scaler"], scaler_file)
        print(f"Saved rescue model: {model_file}")

    # -------------------------------------------------------------------------
    # PART 9: CRITICAL KNOWN-GESTURE CHECK
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 9: CRITICAL KNOWN-GESTURE EVALUATION ON VALIDATION SAMPLES")
    print("=" * 80)

    target_checks = {
        "1. Hasthas": ["1. Pathaka", "1. Anjali", "2. Kapothamu", "2. Thripathaka", "17.Matsyamu"],
        "2. Shiro Bhedhas (HEAD MOVEMENTS)": ["1. Sama", "2. Udvahitam", "3. Adhomukham", "4. Alolitam", "5. Dhutam", "6. Kampitam"],
        "3. Drishti Bhedhas(EYE MOVEMENTS)": ["1. Sama", "2. Alokita", "3. Saachi", "4. Pralokita", "5. Nimilita", "7. Anuvritta"],
        "4. Pada Karma(Foot movements)": ["1. Udghattita", "2. Sama", "3. Agratalasanchara", "4. Anchita Pada", "5. Kunchita", "6. Suchi Pada"]
    }

    known_gesture_records = []
    overall_correct = 0
    overall_total = 0

    for cat_key, target_list in target_checks.items():
        sel_info = selected_models[cat_key]
        res = sel_info["res"]
        cat_data = sel_info["data"]
        vl_sub = cat_data["vl_sub"]
        idx_to_class = cat_data["idx_to_class"]
        preds = res["preds"]
        probs = res["probs"]

        print(f"\n--- {cat_key} Known Gestures ---")
        cat_correct = 0
        cat_total = 0

        for target_cname in target_list:
            matches = vl_sub[vl_sub["class_name"] == target_cname]
            if len(matches) == 0:
                print(f"  Warning: No validation sample for {target_cname}")
                continue
            
            # Check the first representative sample of this class
            val_idx = matches.index[0]
            true_cls = target_cname
            pred_id = preds[val_idx]
            pred_cls = idx_to_class[pred_id]
            conf = probs[val_idx, pred_id] if probs is not None else 1.0
            is_pass = (true_cls == pred_cls)
            result_str = "PASS" if is_pass else "FAIL"

            cat_total += 1
            overall_total += 1
            if is_pass:
                cat_correct += 1
                overall_correct += 1

            rec = {
                "category": cat_key,
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "confidence": conf * 100.0,
                "result": result_str
            }
            known_gesture_records.append(rec)
            print(f"  TRUE: {true_cls:22s} -> PREDICTED: {pred_cls:22s} | Conf: {conf*100:5.1f}% | {result_str}")

        print(f"  Category Summary: {cat_correct}/{cat_total} correct ({cat_correct/cat_total*100:.1f}%)" if cat_total > 0 else "")

    print(f"\n>>> OVERALL KNOWN GESTURE RESULT: {overall_correct}/{overall_total} correct ({overall_correct/overall_total*100:.1f}%)")

    # -------------------------------------------------------------------------
    # PART 12 & 13: GENERATE DETAILED TECHNICAL REPORT
    # -------------------------------------------------------------------------
    report_file = PROJECT_ROOT / "reports" / "step28_recognition_rescue_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STEP 28 — FINAL RECOGNITION RESCUE MODEL TECHNICAL REPORT\n")
        f.write("KUCHIPUDI DANCE RECOGNITION SYSTEM\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. EXISTING MODEL PROBLEM (DIAGNOSED IN STEP 27)\n")
        f.write("-" * 80 + "\n")
        f.write("In Step 27 emergency diagnostics, the existing deployed models in src/final_inference.py\n")
        f.write("exhibited severe recognition degradation on known validation development samples:\n")
        f.write("  - Hasthas (52 classes)        : 1 / 7 correct = 14.3%\n")
        f.write("  - Shiro Bhedhas (9 classes)   : 2 / 6 correct = 33.3%\n")
        f.write("  - Drishti Bhedhas (8 classes) : 3 / 6 correct = 50.0%\n")
        f.write("  - Pada Karma (6 classes)      : 0 / 6 correct =  0.0%\n")
        f.write("  - Overall Known-Sample Check  : 6 / 25 correct = 24.0%\n")
        f.write("This confirmed that live recognition failures were caused by model misclassification\n")
        f.write("rather than camera pipeline or UI defects.\n\n")

        f.write("2. FEATURES USED ACROSS CANDIDATES\n")
        f.write("-" * 80 + "\n")
        f.write("Strictly disjoint source-isolated partitions (v2_train.csv and v2_validation.csv):\n")
        f.write("  1. Hasthas:\n")
        f.write("     - MediaPipe 175-dim geometric hand landmark vector (joint coordinates, normalized\n")
        f.write("       fingertip-wrist distances, inter-fingertip distances, finger flexion angles).\n")
        f.write("     - Landmark-space data augmentation (coordinate jitter, rotation, scale, translation).\n")
        f.write("     - Multimodal fusion with 960-dim MobileNetV3Large CNN visual embedding.\n")
        f.write("  2. Shiro Bhedhas:\n")
        f.write("     - 32-dim head pose / facial geometry landmarks.\n")
        f.write("     - 256-dim temporal trajectory statistics (mean, std, displacement, velocity).\n")
        f.write("     - Multimodal fusion with 960-dim CNN visual embedding.\n")
        f.write("  3. Drishti Bhedhas:\n")
        f.write("     - 44-dim eye & gaze metrics (EAR, iris displacement, horizontal/vertical gaze ratios).\n")
        f.write("     - 352-dim temporal trajectory statistics.\n")
        f.write("     - Multimodal fusion with 960-dim CNN visual embedding.\n")
        f.write("  4. Pada Karma:\n")
        f.write("     - 40-dim lower-body pose (knee/ankle angles, heel elevation, stance width).\n")
        f.write("     - 320-dim temporal trajectory statistics.\n")
        f.write("     - Multimodal fusion with 960-dim CNN visual embedding.\n\n")

        f.write("3. CANDIDATE MODELS EVALUATED\n")
        f.write("-" * 80 + "\n")
        f.write("Lightweight classifiers evaluated on TRAIN only and tested on VALIDATION only:\n")
        f.write("  - RBF Support Vector Machines (class-balanced, probability calibration)\n")
        f.write("  - Random Forest Classifiers (300 estimators, balanced class weights)\n")
        f.write("  - Compact Multi-Layer Perceptrons (128-64 or 256-128 units, L2 regularization)\n")
        f.write("  - k-Nearest Neighbors / Cosine Nearest Neighbors (k in {1, 3, 5})\n\n")

        f.write("4. VALIDATION RESULTS COMPARISON TABLE (PART 8)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'CATEGORY':<16} | {'MODEL':<38} | {'TOP-1':<7} | {'TOP-5':<7} | {'MACRO F1':<8} | {'PARAMETERS':<14} | {'SELECTED?'}\n")
        f.write("-" * 110 + "\n")
        for row, _ in comparison_results:
            f.write(f"{row['CATEGORY']:<16} | {row['MODEL']:<38} | {row['VALIDATION_TOP1']:<7} | {row['VALIDATION_TOP5']:<7} | {row['MACRO_F1']:<8} | {row['PARAMETERS']:<14} | {row['SELECTED']}\n")
        f.write("-" * 110 + "\n\n")

        f.write("5. BEST RESCUE MODEL SELECTED PER CATEGORY\n")
        f.write("-" * 80 + "\n")
        for cat_name, sel_info in selected_models.items():
            r = sel_info["res"]
            f.write(f"Category: {cat_name}\n")
            f.write(f"  Selected Model  : {sel_info['name']}\n")
            f.write(f"  Validation Top-1: {r['top1']:.2f}%\n")
            f.write(f"  Validation Top-5: {r['top5']:.2f}%\n")
            f.write(f"  Macro F1 Score  : {r['f1_macro']:.2f}%\n")
            f.write(f"  Macro Recall    : {r['r_macro']:.2f}%\n\n")

        f.write("6. CRITICAL KNOWN-GESTURE VERIFICATION RESULTS (PART 9)\n")
        f.write("-" * 80 + "\n")
        for rec in known_gesture_records:
            f.write(f"[{rec['category']}] TRUE: {rec['true_class']:<22} | PRED: {rec['predicted_class']:<22} | CONF: {rec['confidence']:5.1f}% | RESULT: {rec['result']}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Known Validation Gesture Accuracy: {overall_correct}/{overall_total} = {overall_correct/overall_total*100:.1f}%\n")
        f.write(f"Comparison with Step 27 Baseline        : 6/25 (24.0%) -> {overall_correct}/{overall_total} ({overall_correct/overall_total*100:.1f}%)\n\n")

        f.write("7. LIVE CAMERA INTEGRATION FEASIBILITY (PART 10)\n")
        f.write("-" * 80 + "\n")
        f.write("The selected rescue models have been saved to models/rescue_*.joblib.\n")
        f.write("A dedicated standalone runner src/rescue_inference.py has been developed.\n")
        f.write("Preserved components:\n")
        f.write("  1. Interactive Category Selection (1: Hasthas, 2: Shiro, 3: Drishti, 4: Pada)\n")
        f.write("  2. Live Camera Feed with Aspect-Ratio Preserving Letterbox\n")
        f.write("  3. MediaPipe Real-time Landmark Extraction Pipeline\n")
        f.write("  4. Direct Category Expert Dispatch (zero Router dependency)\n")
        f.write("  5. Step 26B Stability Gate (Window=10, MinStable=7, ConfThreshold=0.55)\n")
        f.write("  6. Output formatting: Exactly ONE gesture name displayed once stabilized.\n")
        f.write("Integrity guarantee: final_inference.py and original Step 18 models are completely UNTOUCHED.\n\n")

        f.write("8. HONEST LIMITATIONS & BOTTLENECKS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Fine-Grained Intra-Class Hand Ambiguity:\n")
        f.write("   - 52 Hasthas classes with only ~8 training samples per class.\n")
        f.write("   - Subtle finger curvatures (e.g. Pathaka vs Arala vs Sukatunda) are extremely sensitive\n")
        f.write("     to small landmark detection errors and camera orientation.\n")
        f.write("2. Movement Dynamics & Static Sequence Expansion:\n")
        f.write("   - Categories with mixed static images and video sequences (Shiro, Drishti, Pada) suffer\n")
        f.write("     from temporal domain mismatch when stationary images are expanded to 16 frames.\n")
        f.write("3. Limited Per-Class Validation Representation:\n")
        f.write("   - With source-disjoint splits, several classes have only 1 validation sample.\n")
        f.write("   - A single misclassification impacts class recall by 100 percentage points.\n\n")

        f.write("9. EXACT FILES CREATED\n")
        f.write("-" * 80 + "\n")
        f.write("  - src/prepare_step28_data.py\n")
        f.write("  - src/train_step28_rescue_models.py\n")
        f.write("  - src/rescue_inference.py\n")
        f.write("  - models/rescue_Hasthas_best.joblib & rescue_Hasthas_scaler.joblib\n")
        f.write("  - models/rescue_Shiro_Bhedhas_best.joblib & rescue_Shiro_Bhedhas_scaler.joblib\n")
        f.write("  - models/rescue_Drishti_Bhedhas_best.joblib & rescue_Drishti_Bhedhas_scaler.joblib\n")
        f.write("  - models/rescue_Pada_Karma_best.joblib & rescue_Pada_Karma_scaler.joblib\n")
        f.write("  - reports/step28_recognition_rescue_report.txt\n\n")

        f.write("10. BLUNT DECISION RULE (PART 13)\n")
        f.write("=" * 80 + "\n")
        if (overall_correct / overall_total) >= 0.60:
            f.write("CONCLUSION:\n")
            f.write('"Use the rescue model for the demonstration."\n\n')
            f.write(f"Rationale: The rescue models improved known validation gesture recognition from\n")
            f.write(f"24.0% (Step 27) to {overall_correct/overall_total*100:.1f}%, providing a substantial and reliable boost.\n")
        else:
            f.write("CONCLUSION:\n")
            f.write('"The existing dataset/model cannot reliably support exact real-time recognition; further data collection is required."\n\n')
            f.write(f"Rationale: Despite exhaustive benchmarking across RBF SVM, Random Forest, compact MLP, and\n")
            f.write(f"multimodal fusion, known validation gesture recognition reached {overall_correct/overall_total*100:.1f}%, which\n")
            f.write("remains insufficient for reliable, error-free demonstration across all 75 classes.\n")
        f.write("=" * 80 + "\n")

    print(f"\nReport written to: {report_file}")
    return comparison_results, selected_models, known_gesture_records

if __name__ == "__main__":
    run_step28_rescue_benchmark()
