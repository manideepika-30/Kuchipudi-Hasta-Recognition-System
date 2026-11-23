"""
STEP 26 — Comprehensive Functional Verification Script for Final Demo Mode
Tests:
  A. Hasthas: Missing hand -> 'Show a Hasta' (Zero name displayed)
  B. Hasthas: Fleeting / Unstable -> 'Hold the Hasta clearly' (Zero name displayed)
  C. Hasthas: Stable Held Gesture (7+ frames) -> 'RECOGNIZED' -> Single Name '1. ANJALI'
  D. Hasthas: Gesture Shift -> Previous name immediately removed -> 'Hold the Hasta clearly'
  E. Category Switching -> All buffers & gate reset cleanly
  F. Shiro Bhedhas: Stability gate & missing/unstable prompts
  G. Drishti Bhedhas: Stability gate & eye mesh prompts
  H. Pada Karma: Stability gate & leg/pose prompts
"""

import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.final_inference import (
    KuchipudiInferenceEngine,
    GestureDisplayGate,
    CATEGORIES,
    CATEGORY_BY_ID
)

def run_step26_functional_tests():
    print("=" * 80)
    print("STEP 26B: FUNCTIONAL DEMO MODE & STABILITY-FIRST GATE VERIFICATION")
    print("=" * 80)

    engine = KuchipudiInferenceEngine()
    gate = GestureDisplayGate(window_size=8, min_stable_frames=6, min_confidence=0.20)

    sample_anjali = PROJECT_ROOT / "Data set/1. Hasthas/2. Samyutha Hasthas (two-hand gestures)/1. Anjali/WhatsApp Image 2026-09-11 at 10.46.05 PM (3).jpeg"
    frame_anjali = cv2.imread(str(sample_anjali))

    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # -------------------------------------------------------------------------
    # TEST 1: HASTHAS — NO HAND DETECTED
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: HASTHAS (NO HAND PRESENT) ---")
    gate.reset("1. Hasthas")
    res_blank = engine.process_category(blank_frame, "1. Hasthas", apply_smoothing=False)
    gate_out = gate.update(res_blank, "1. Hasthas")

    print(f"  Hand Detected   : {res_blank['detected']}")
    print(f"  Gate Decision   : {'RECOGNIZED' if gate_out['is_recognized'] else 'NOT RECOGNIZED'}")
    print(f"  Displayed Name  : {gate_out['recognized_name']}")
    print(f"  Status Prompt   : '{gate_out['status_prompt']}'")
    assert not gate_out["is_recognized"], "Should not be recognized when hand is missing!"
    assert gate_out["recognized_name"] is None, "Name must be None when hand is missing!"
    assert gate_out["status_prompt"] == "Show a Hasta", f"Expected 'Show a Hasta', got '{gate_out['status_prompt']}'"
    print("  --> TEST 1 PASSED: Zero random class names displayed.")

    # -------------------------------------------------------------------------
    # TEST 2: HASTHAS — FLEETING / UNSTABLE FRAMES (1 to 5 frames)
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: HASTHAS (FLEETING / UNSTABLE GESTURE < 6 FRAMES) ---")
    gate.reset("1. Hasthas")
    res_anjali = engine.process_category(frame_anjali, "1. Hasthas", apply_smoothing=False)

    for f_idx in range(1, 6):
        gate_out = gate.update(res_anjali, "1. Hasthas")
        print(f"  Frame {f_idx:02d}: Gate='{gate_out['status_prompt']}', Displayed Name={gate_out['recognized_name']}")
        assert not gate_out["is_recognized"], f"Frame {f_idx} must not be recognized before stability threshold!"
        assert gate_out["recognized_name"] is None, f"Frame {f_idx} must not display any name!"
        assert gate_out["status_prompt"] == "Hold the Hasta clearly"
    print("  --> TEST 2 PASSED: Fleeting/unstable frames never display premature class names.")

    # -------------------------------------------------------------------------
    # TEST 3: HASTHAS — STABLE RECOGNITION (>= 6 of 8 frames)
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: HASTHAS (STABLE HELD GESTURE >= 6 FRAMES) ---")
    gate_out = gate.update(res_anjali, "1. Hasthas")  # Frame 6
    print(f"  Frame 06: is_recognized={gate_out['is_recognized']}, Displayed Name='{gate_out['recognized_name']}'")
    assert gate_out["is_recognized"], "Frame 6 should be stably recognized!"
    assert gate_out["recognized_name"] == "1. Anjali", f"Expected '1. Anjali', got '{gate_out['recognized_name']}'"

    # Sustained holding
    for f_idx in range(7, 11):
        gate_out = gate.update(res_anjali, "1. Hasthas")
        assert gate_out["is_recognized"] and gate_out["recognized_name"] == "1. Anjali"
    print(f"  Frame 07-10: Consistently recognized as '{gate_out['recognized_name']}'")
    print("  --> TEST 3 PASSED: Single gesture name displayed clearly once stable.")

    # -------------------------------------------------------------------------
    # TEST 4: HASTHAS — GESTURE SHIFT (IMMEDIATE REMOVAL OF OLD NAME)
    # -------------------------------------------------------------------------
    print("\n--- TEST 4: HASTHAS (GESTURE POSITION SHIFT) ---")
    # Simulate user shifting hand position (e.g. hand shifts, producing a different prediction)
    fake_shift_res = dict(res_anjali)
    fake_shift_res["gesture_name"] = "2. Thripathaka"
    fake_shift_res["confidence"] = 52.0

    gate_out_shifted = gate.update(fake_shift_res, "1. Hasthas")
    print(f"  Shift Frame 01: is_recognized={gate_out_shifted['is_recognized']}, Displayed Name={gate_out_shifted['recognized_name']}, Prompt='{gate_out_shifted['status_prompt']}'")
    assert not gate_out_shifted["is_recognized"], "Must immediately revoke recognition when hand shifts!"
    assert gate_out_shifted["recognized_name"] is None, "Old gesture name '1. Anjali' must disappear immediately!"
    assert gate_out_shifted["status_prompt"] == "Hold the Hasta clearly"
    print("  --> TEST 4 PASSED: Old prediction disappears instantly upon movement.")

    # -------------------------------------------------------------------------
    # TEST 4B: ALTERNATING / UNSTABLE CLASSES (NO RANDOM NAME)
    # -------------------------------------------------------------------------
    print("\n--- TEST 4B: ALTERNATING / UNSTABLE CLASSES (PREVENT RANDOM NAMES) ---")
    gate.reset("1. Hasthas")
    alternating_classes = ["Pathaka", "Pathaka", "Matsyamu", "Pathaka", "Thripathaka", "Matsyamu", "Pathaka", "Thripathaka"]
    for idx, cname in enumerate(alternating_classes):
        alt_res = dict(res_anjali)
        alt_res["gesture_name"] = cname
        alt_res["confidence"] = 40.0
        alt_out = gate.update(alt_res, "1. Hasthas")
        assert not alt_out["is_recognized"], f"Alternating class {cname} at frame {idx+1} must not be recognized!"
        assert alt_out["recognized_name"] is None, "Must not display any name during alternating predictions!"
        assert alt_out["status_prompt"] == "Hold the Hasta clearly"
    print("  --> TEST 4B PASSED: Alternating predictions never display a class name.")

    # -------------------------------------------------------------------------
    # TEST 5: CATEGORY SWITCHING AND BUFFER RESET
    # -------------------------------------------------------------------------
    print("\n--- TEST 5: CATEGORY SWITCHING & TOTAL STATE RESET ---")
    # Switch from Hasthas -> Shiro
    gate.reset("2. Shiro Bhedhas (HEAD MOVEMENTS)")
    engine.reset_category("2. Shiro Bhedhas (HEAD MOVEMENTS)")
    assert len(gate.history) == 0, "Gate history must be 0 after reset!"
    assert gate.current_recognized_name is None, "Recognized name must be None after reset!"
    assert gate.current_status_prompt == "Show a Shiro Bhedha"
    print("  --> TEST 5 PASSED: Total state reset verified.")

    # -------------------------------------------------------------------------
    # TEST 6: SHIRO BHEDHAS SAFEGUARDS & STABILITY
    # -------------------------------------------------------------------------
    print("\n--- TEST 6: SHIRO BHEDHAS SAFEGUARDS & PROMPTS ---")
    res_shiro_blank = engine.process_category(blank_frame, "2. Shiro Bhedhas (HEAD MOVEMENTS)", apply_smoothing=False)
    gate_shiro_out = gate.update(res_shiro_blank, "2. Shiro Bhedhas (HEAD MOVEMENTS)")
    print(f"  Blank Frame Status: '{gate_shiro_out['status_prompt']}', Displayed Name={gate_shiro_out['recognized_name']}")
    assert not gate_shiro_out["is_recognized"]
    assert gate_shiro_out["recognized_name"] is None
    print("  --> TEST 6 PASSED: Shiro Bhedhas prompts correctly.")

    # -------------------------------------------------------------------------
    # TEST 7: DRISHTI BHEDHAS SAFEGUARDS & PROMPTS
    # -------------------------------------------------------------------------
    print("\n--- TEST 7: DRISHTI BHEDHAS SAFEGUARDS & PROMPTS ---")
    gate.reset("3. Drishti Bhedhas(EYE MOVEMENTS)")
    engine.reset_category("3. Drishti Bhedhas(EYE MOVEMENTS)")
    res_drishti_blank = engine.process_category(blank_frame, "3. Drishti Bhedhas(EYE MOVEMENTS)", apply_smoothing=False)
    gate_drishti_out = gate.update(res_drishti_blank, "3. Drishti Bhedhas(EYE MOVEMENTS)")
    print(f"  Blank Frame Status: '{gate_drishti_out['status_prompt']}', Displayed Name={gate_drishti_out['recognized_name']}")
    assert not gate_drishti_out["is_recognized"]
    assert gate_drishti_out["status_prompt"] == "Look at the camera"
    assert gate_drishti_out["recognized_name"] is None
    print("  --> TEST 7 PASSED: Drishti Bhedhas prompts 'Look at the camera' when face is missing.")

    # -------------------------------------------------------------------------
    # TEST 8: PADA KARMA SAFEGUARDS & PROMPTS
    # -------------------------------------------------------------------------
    print("\n--- TEST 8: PADA KARMA SAFEGUARDS & PROMPTS ---")
    gate.reset("4. Pada Karma(Foot movements)")
    engine.reset_category("4. Pada Karma(Foot movements)")
    res_pada_blank = engine.process_category(blank_frame, "4. Pada Karma(Foot movements)", apply_smoothing=False)
    gate_pada_out = gate.update(res_pada_blank, "4. Pada Karma(Foot movements)")
    print(f"  Blank Frame Status: '{gate_pada_out['status_prompt']}', Displayed Name={gate_pada_out['recognized_name']}")
    assert not gate_pada_out["is_recognized"]
    assert gate_pada_out["status_prompt"] == "Show your feet/legs"
    assert gate_pada_out["recognized_name"] is None
    print("  --> TEST 8 PASSED: Pada Karma prompts 'Show your feet/legs' when legs are missing.")

    print("\n" + "=" * 80)
    print("ALL 8 STEP 26 FUNCTIONAL VERIFICATION TESTS COMPLETED AND PASSED!")
    print("=" * 80)
    engine.close()

if __name__ == "__main__":
    run_step26_functional_tests()
