import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hierarchical_model import build_hierarchical_mapping
from src.preprocessing import load_and_preprocess_image, resize_with_aspect_ratio_pad, normalize_pixels
from src.landmark_extraction import KuchipudiLandmarkPipeline
from src.train_hybrid_models import build_cnn_feature_extractor
from src.final_inference import KuchipudiInferenceEngine

def run_diagnostic():
    print("=" * 80)
    print("STEP 25: CONTROLLED LOCAL COMPARISON DIAGNOSTIC")
    print("=" * 80)

    # 1. Load validation sample
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "v2_validation.csv")
    hasthas_val = val_df[val_df["category"] == "1. Hasthas"].reset_index(drop=True)
    sample_row = hasthas_val.iloc[0]
    sample_path = PROJECT_ROOT / sample_row["sample_path"]
    true_class = sample_row["class_name"]

    print(f"Sample Path: {sample_path}")
    print(f"True Class: {true_class}")
    print(f"File exists: {sample_path.exists()}")

    # -------------------------------------------------------------------------
    # PART A: EXACT STEP 18 PIPELINE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("PART A: STEP 18 GROUND TRUTH PIPELINE")
    print("-" * 40)

    # A1. Step 18 Image Preprocessing
    # Uses load_and_preprocess_image with aspect_pad and rescale_0_1
    img_step18 = load_and_preprocess_image(sample_path)
    print(f"Step 18 image shape: {img_step18.shape}, dtype: {img_step18.dtype}, range: [{img_step18.min():.4f}, {img_step18.max():.4f}]")

    # A2. Step 18 Landmark Extraction
    # In Step 18, landmarks were loaded from pre-extracted cache: data/landmarks/images/<stem>_landmarks.npz
    stem = sample_path.stem
    lm_cache_file = PROJECT_ROOT / "data" / "landmarks" / "images" / f"{stem}_landmarks.npz"
    saved_lm_data = np.load(lm_cache_file)
    lm_step18_cached = saved_lm_data["vector"].astype(np.float32)
    print(f"Step 18 cached landmark vector shape: {lm_step18_cached.shape}, non-zero: {np.count_nonzero(lm_step18_cached)}")

    # A3. Step 18 Visual Feature Extractor
    models_dir = PROJECT_ROOT / "models"
    vis_extractor_s18 = build_cnn_feature_extractor(
        models_dir / "v2_hierarchical_hasthas_best.keras", "Hasthas_global_pool"
    )
    vis_emb_s18 = vis_extractor_s18.predict(np.expand_dims(img_step18, axis=0), verbose=0)[0]
    print(f"Step 18 visual embedding shape: {vis_emb_s18.shape}, norm: {np.linalg.norm(vis_emb_s18):.4f}")

    # A4. Step 18 Scalers & Model Inference
    s_h17 = np.load(str(models_dir / "step17_hasthas_scaler.npz"))
    s_h16 = np.load(str(models_dir / "step16a_hasthas_scaler.npz"))

    m_h17 = tf.keras.models.load_model(str(models_dir / "step17_hasthas_hybrid_best.keras"))
    m_h16 = tf.keras.models.load_model(str(models_dir / "step16a_hasthas_landmark_mlp_best.keras"))

    lm_s17_scaled = (lm_step18_cached - s_h17["mean"]) / s_h17["scale"]
    lm_s16_scaled = (lm_step18_cached - s_h16["mean"]) / s_h16["scale"]

    p_h17_s18 = m_h17.predict([np.expand_dims(vis_emb_s18, 0), np.expand_dims(lm_s17_scaled, 0)], verbose=0)[0]
    p_h16_s18 = m_h16.predict(np.expand_dims(lm_s16_scaled, 0), verbose=0)[0]

    probs_step18 = 0.80 * p_h17_s18 + 0.20 * p_h16_s18
    pred_id_s18 = int(np.argmax(probs_step18))

    # Mapping
    mapping = build_hierarchical_mapping(save_json=False)
    cat_info = mapping["experts"]["1. Hasthas"]
    idx_to_class = {item["local_class_id"]: item["class_name"] for item in cat_info["classes"]}
    pred_name_s18 = idx_to_class[pred_id_s18]

    top5_ids_s18 = np.argsort(probs_step18)[-5:][::-1]
    print(f"Step 18 Predicted: {pred_name_s18} ({probs_step18[pred_id_s18]*100:.2f}%)")
    print("Step 18 Top 5:")
    for r, cid in enumerate(top5_ids_s18, 1):
        print(f"  {r}. {idx_to_class[cid]:<30} {probs_step18[cid]*100:5.2f}%")

    # -------------------------------------------------------------------------
    # PART B: CURRENT FINAL_INFERENCE.PY PIPELINE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("PART B: CURRENT FINAL_INFERENCE.PY PIPELINE")
    print("-" * 40)

    engine = KuchipudiInferenceEngine(models_dir=models_dir)

    # Load image via cv2 as done in final_inference.py
    frame_bgr = cv2.imread(str(sample_path))
    res_final = engine.process_category(frame_bgr, "1. Hasthas", apply_smoothing=False)

    print(f"final_inference Predicted: {res_final['gesture_name']} ({res_final['confidence']:.2f}%)")
    print("final_inference Top 5:")
    for r, (name, pct) in enumerate(res_final["top5"], 1):
        print(f"  {r}. {name:<30} {pct:5.2f}%")

    # -------------------------------------------------------------------------
    # PART C: DETAILED COMPONENT-WISE COMPARISON & DIFFERENCE BREAKDOWN
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART C: DETAILED COMPONENT-WISE DIFFERENCE ANALYSIS")
    print("=" * 80)

    # C1: Image Preprocessing Comparison
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_final_cv2 = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img_diff = np.max(np.abs(img_step18 - img_final_cv2))
    img_mse = np.mean((img_step18 - img_final_cv2) ** 2)
    print(f"C1. Image Preprocessing Difference:")
    print(f"    Max Absolute Diff: {img_diff:.6f}")
    print(f"    Mean Squared Error: {img_mse:.6f}")
    orig_h, orig_w = frame_bgr.shape[:2]
    print(f"    Original aspect ratio: {orig_w}x{orig_h} (Ratio: {orig_w/orig_h:.3f})")
    if orig_w != orig_h:
        print(f"    [MISMATCH DETECTED]: cv2.resize distorted aspect ratio from {orig_w/orig_h:.2f} to 1.00, whereas Step 18 used letterbox padding!")

    # C2: Visual Feature Extraction Comparison
    vis_emb_final = engine.vis_extractors["1. Hasthas"](np.expand_dims(img_final_cv2, 0), training=False).numpy()[0]
    vis_diff = np.max(np.abs(vis_emb_s18 - vis_emb_final))
    vis_cos_sim = np.dot(vis_emb_s18, vis_emb_final) / (np.linalg.norm(vis_emb_s18) * np.linalg.norm(vis_emb_final) + 1e-8)
    print(f"\nC2. Visual Embedding Difference (Due to Preprocessing):")
    print(f"    Max Absolute Diff: {vis_diff:.6f}")
    print(f"    Cosine Similarity: {vis_cos_sim:.6f}")

    # C3: Landmark Extraction Comparison
    # Test static mode pipeline vs live mode pipeline vs saved cache
    pipe_static = KuchipudiLandmarkPipeline(static_image_mode=True)
    pipe_live = KuchipudiLandmarkPipeline(static_image_mode=False)

    lm_live = pipe_live.extract_from_frame(frame_rgb, "1. Hasthas")["vector"]
    lm_static = pipe_static.extract_from_frame(frame_rgb, "1. Hasthas")["vector"]

    diff_cached_static = np.max(np.abs(lm_step18_cached - lm_static))
    diff_cached_live = np.max(np.abs(lm_step18_cached - lm_live))
    diff_static_live = np.max(np.abs(lm_static - lm_live))

    print(f"\nC3. Landmark Extraction Difference:")
    print(f"    Cached vs Static Pipeline Max Diff: {diff_cached_static:.6f}")
    print(f"    Cached vs Live Pipeline Max Diff  : {diff_cached_live:.6f}")
    print(f"    Static vs Live Pipeline Max Diff  : {diff_static_live:.6f}")

    # C4: What if final_inference used the EXACT Step 18 preprocessing?
    vis_emb_fixed = engine.vis_extractors["1. Hasthas"](np.expand_dims(img_step18, 0), training=False).numpy()[0]
    lm_s17_fixed = (lm_live - s_h17["mean"]) / s_h17["scale"]
    lm_s16_fixed = (lm_live - s_h16["mean"]) / s_h16["scale"]

    p_h17_fixed = engine.m_hasthas([np.expand_dims(vis_emb_fixed, 0), np.expand_dims(lm_s17_fixed, 0)], training=False).numpy()[0]
    p_h16_fixed = engine.m_hasthas_lm(np.expand_dims(lm_s16_fixed, 0), training=False).numpy()[0]
    probs_fixed = 0.80 * p_h17_fixed + 0.20 * p_h16_fixed
    pred_id_fixed = int(np.argmax(probs_fixed))

    print(f"\nC4. Simulated Output IF Aspect-Ratio Preprocessing is Applied:")
    print(f"    Predicted: {idx_to_class[pred_id_fixed]} ({probs_fixed[pred_id_fixed]*100:.2f}%)")
    top5_fixed = np.argsort(probs_fixed)[-5:][::-1]
    print(f"    Top 5 with Fixed Preprocessing:")
    for r, cid in enumerate(top5_fixed, 1):
        print(f"      {r}. {idx_to_class[cid]:<30} {probs_fixed[cid]*100:5.2f}%")

    # C5: What if both EXACT Step 18 preprocessing AND exact cached landmark are used?
    p_h17_exact = engine.m_hasthas([np.expand_dims(vis_emb_s18, 0), np.expand_dims(lm_s17_scaled, 0)], training=False).numpy()[0]
    p_h16_exact = engine.m_hasthas_lm(np.expand_dims(lm_s16_scaled, 0), training=False).numpy()[0]
    probs_exact = 0.80 * p_h17_exact + 0.20 * p_h16_exact
    prob_diff = np.max(np.abs(probs_step18 - probs_exact))
    print(f"\nC5. Pipeline Reproducibility with Identical Inputs:")
    print(f"    Max Diff between Step 18 calculation and engine calculation: {prob_diff:.8f}")

    pipe_static.close()
    pipe_live.close()
    engine.close()

if __name__ == "__main__":
    run_diagnostic()
