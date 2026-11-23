import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
CLEAN_VENV_PKGS = os.path.join(PROJECT_ROOT, "deployment_clean", ".venv", "Lib", "site-packages")

if os.path.exists(CLEAN_VENV_PKGS) and CLEAN_VENV_PKGS not in sys.path:
    sys.path.insert(0, CLEAN_VENV_PKGS)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from test_stage4_curriculum import run_stage4_tests
from test_stage5_assignments import run_stage5_tests
from test_stage6_ai import run_stage6_tests
from test_stage7_e2e_system import run_stage7_e2e_test_suite

def main():
    print("=" * 75)
    print("NRITYA AI: STAGE 9 COMPREHENSIVE AUTOMATED AUDIT SUITE")
    print("=" * 75)
    
    results = {}
    
    # 1. Stage 4: Curriculum & Learning Modules
    print("\n>>> AUDIT 1: SEDS Stage 4 Curriculum (7 Modules, 92 Lessons, Telugu Texts)")
    t0 = time.time()
    try:
        run_stage4_tests()
        results["Stage 4 Curriculum"] = {"status": "PASSED", "duration": round(time.time() - t0, 2)}
    except Exception as e:
        results["Stage 4 Curriculum"] = {"status": "FAILED", "error": str(e)}

    # 2. Stage 5: Assignments & Examinations
    print("\n>>> AUDIT 2: SEDS Stage 5 Assignments (Theory MCQ, Image ID, Anti-Cheat)")
    t0 = time.time()
    try:
        run_stage5_tests()
        results["Stage 5 Assignments"] = {"status": "PASSED", "duration": round(time.time() - t0, 2)}
    except Exception as e:
        results["Stage 5 Assignments"] = {"status": "FAILED", "error": str(e)}

    # 3. Stage 6: Existing AI Model Integration
    print("\n>>> AUDIT 3: SEDS Stage 6 AI Integration (nritya_ai_weights.keras, 69 Classes)")
    t0 = time.time()
    try:
        run_stage6_tests()
        results["Stage 6 AI Engine"] = {"status": "PASSED", "duration": round(time.time() - t0, 2)}
    except Exception as e:
        results["Stage 6 AI Engine"] = {"status": "FAILED", "error": str(e)}

    # 4. Stage 7: Full Platform End-to-End System
    print("\n>>> AUDIT 4: SEDS Stage 7 E2E Platform (Auth, RBAC, Progress, Anti-Cheat, AI)")
    t0 = time.time()
    try:
        run_stage7_e2e_test_suite()
        results["Stage 7 Full E2E"] = {"status": "PASSED", "duration": round(time.time() - t0, 2)}
    except Exception as e:
        results["Stage 7 Full E2E"] = {"status": "FAILED", "error": str(e)}

    print("\n" + "=" * 75)
    print("STAGE 9 AUDIT RESULTS SUMMARY:")
    print("=" * 75)
    all_passed = True
    for suite_name, res in results.items():
        st = res["status"]
        if st == "PASSED":
            print(f"  [PASS] {suite_name:<30} Duration: {res['duration']}s")
        else:
            print(f"  [FAIL] {suite_name:<30} Error: {res.get('error')}")
            all_passed = False
    
    print("=" * 75)
    if all_passed:
        print("ALL AUDIT SUITES PASSED (100% SUCCESS RATE)!")
    else:
        print("SOME AUDIT SUITES FAILED. PLEASE REVIEW LOGS.")
    print("=" * 75)

if __name__ == "__main__":
    main()
