"""
STEP 28 — Feature Ingestion & Caching Script
Ingests train and validation features for all 4 categories:
  1. Hasthas (52 classes)
  2. Shiro Bhedhas (9 classes)
  3. Drishti Bhedhas (8 classes)
  4. Pada Karma (6 classes)
Strictly adheres to:
  - Train: v2_train.csv
  - Validation: v2_validation.csv
  - Test: STRICTLY EXCLUDED (zero access to v2_test.csv)
Caches arrays to scratch/step28_features_cache.pkl
"""

import os
import sys
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_hybrid_models import load_category_hybrid_data

CATEGORIES_CONFIG = [
    {
        "name": "1. Hasthas",
        "prefix": "Hasthas",
        "expert_model": "v2_hierarchical_hasthas_best.keras",
        "pool_layer": "Hasthas_global_pool"
    },
    {
        "name": "2. Shiro Bhedhas (HEAD MOVEMENTS)",
        "prefix": "Shiro_Bhedhas",
        "expert_model": "v2_hierarchical_shiro_best.keras",
        "pool_layer": "Shiro_Bhedhas_global_pool"
    },
    {
        "name": "3. Drishti Bhedhas(EYE MOVEMENTS)",
        "prefix": "Drishti_Bhedhas",
        "expert_model": "v2_hierarchical_drishti_best.keras",
        "pool_layer": "Drishti_Bhedhas_global_pool"
    },
    {
        "name": "4. Pada Karma(Foot movements)",
        "prefix": "Pada_Karma",
        "expert_model": "v2_hierarchical_pada_karma_best.keras",
        "pool_layer": "Pada_Karma_global_pool"
    }
]

def main():
    print("=" * 80)
    print("STEP 28: INGESTING AND CACHING FEATURES FOR ALL 4 CATEGORIES")
    print("=" * 80)
    cache_path = PROJECT_ROOT / "scratch" / "step28_features_cache.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {}
    total_start = time.time()

    for cfg in CATEGORIES_CONFIG:
        cat_name = cfg["name"]
        print(f"\nProcessing category: {cat_name}...")
        t0 = time.time()
        data = load_category_hybrid_data(
            cat_name,
            cfg["prefix"],
            cfg["expert_model"],
            cfg["pool_layer"]
        )
        print(f"Loaded {cat_name} in {time.time() - t0:.2f}s")
        print(f"  Train: X_lm={data['X_lm_train'].shape}, X_temp={data['X_temp_train'].shape}, X_vis={data['X_vis_train'].shape}, y={data['y_train'].shape}")
        print(f"  Val  : X_lm={data['X_lm_val'].shape}, X_temp={data['X_temp_val'].shape}, X_vis={data['X_vis_val'].shape}, y={data['y_val'].shape}")
        
        # Store clean serializable dict
        cache[cat_name] = {
            "category_name": cat_name,
            "prefix": cfg["prefix"],
            "num_classes": data["num_classes"],
            "class_to_idx": data["class_to_idx"],
            "idx_to_class": data["idx_to_class"],
            "X_lm_train": data["X_lm_train"],
            "X_temp_train": data["X_temp_train"],
            "X_vis_train": data["X_vis_train"],
            "y_train": data["y_train"],
            "X_lm_val": data["X_lm_val"],
            "X_temp_val": data["X_temp_val"],
            "X_vis_val": data["X_vis_val"],
            "y_val": data["y_val"],
            "tr_sub": data["tr_sub"],
            "vl_sub": data["vl_sub"]
        }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n" + "=" * 80)
    print(f"CACHE SAVED SUCCESSFULLY: {cache_path} ({cache_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"TOTAL TIME: {time.time() - total_start:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
