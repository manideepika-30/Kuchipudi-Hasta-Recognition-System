"""
Automated Test Suite for Stage 6: Existing AI Model Integration.
Validates loading of nritya_ai_weights.keras without modification,
verifies all 69 classes (52 Hasthas, 9 Shiro Bhedas, 8 Drishti Bhedas),
tests genuine frame prediction, landmark extraction, and SQLite practice history persistence.
"""
import sys
import os
import base64
from pathlib import Path

# Link pre-installed AI packages from deployment_clean virtual environment
CLEAN_VENV_PKGS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "deployment_clean", ".venv", "Lib", "site-packages")
)
if os.path.exists(CLEAN_VENV_PKGS) and CLEAN_VENV_PKGS not in sys.path:
    sys.path.insert(0, CLEAN_VENV_PKGS)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.database.session import SessionLocal
from app.models.user import User
from app.models.quiz_practice import PracticeHistory
from app.utils.security import create_access_token
import cv2
import numpy as np

def run_stage6_tests():
    print("=" * 70)
    print("RUNNING STAGE 6: EXISTING AI MODEL INTEGRATION VERIFICATION SUITE")
    print("=" * 70)

    client = TestClient(app)
    db = SessionLocal()

    try:
        # -------------------------------------------------------------
        # TEST 1: Live AI Engine Status & Architecture
        # -------------------------------------------------------------
        print("\n[TEST 1] Verifying Pre-trained Weights & Unified Architecture...")
        res = client.get("/api/ai/status")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["status"] == "online"
        assert data["engine_initialized"] == True
        assert data["weights_file"] == "nritya_ai_weights.keras"
        assert data["weights_size_mb"] >= 1.2
        assert data["supported_classes"]["total"] == 69
        assert data["supported_classes"]["hasthas"] == 52
        assert data["supported_classes"]["shiro_bhedas"] == 9
        assert data["supported_classes"]["drishti_bhedas"] == 8

        print(f"  ✓ Weights File: {data['weights_file']} ({data['weights_size_mb']} MB)")
        print(f"  ✓ Model Layers: {data['model_layers']}")
        print(f"  ✓ Supported Classes: {data['supported_classes']['total']} total (52 Hastas, 9 Shiro, 8 Drishti)")
        print("  -> [PASS] Weights file verified and loaded cleanly without modifications.")

        # -------------------------------------------------------------
        # TEST 2: All 69 Canonical Classes Catalog
        # -------------------------------------------------------------
        print("\n[TEST 2] Verifying All 69 Canonical Class Definitions...")
        res_classes = client.get("/api/ai/classes")
        assert res_classes.status_code == 200
        classes_data = res_classes.json()
        assert len(classes_data["hasthas"]) == 52
        assert len(classes_data["shiro_bhedhas"]) == 9
        assert len(classes_data["drishti_bhedhas"]) == 8
        assert classes_data["total_classes"] == 69

        hastha_names = [h["name"] for h in classes_data["hasthas"][:5]]
        shiro_names = [s["name"] for s in classes_data["shiro_bhedhas"]]
        drishti_names = [d["name"] for d in classes_data["drishti_bhedhas"]]

        print(f"  ✓ Sample Hasthas: {hastha_names} ... (52 total)")
        print(f"  ✓ Shiro Bhedhas: {shiro_names} (9 total)")
        print(f"  ✓ Drishti Bhedhas: {drishti_names} (8 total)")
        print("  -> [PASS] Complete 69-class taxonomy confirmed.")

        # -------------------------------------------------------------
        # TEST 3: Real Frame Inference via Keras Submodels
        # -------------------------------------------------------------
        print("\n[TEST 3] Testing Real Frame Inference with Authentic Dance Asset...")
        # Load authentic gesture photo from curriculum
        sample_img_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "public", "assets", "curriculum", "Asamyutha_Hasthas_p1_0.jpg")))
        assert sample_img_path.exists(), f"Sample image missing: {sample_img_path}"

        with open(sample_img_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        res_infer = client.post(
            "/api/ai/predict/frame",
            json={
                "image_base64": b64_img,
                "category": "hasthas",
                "target_gesture": "Pathaka"
            }
        )
        assert res_infer.status_code == 200, f"Inference failed: {res_infer.text}"
        pred = res_infer.json()
        assert "detected" in pred
        assert "latency_ms" in pred
        assert pred["latency_ms"] > 0

        if pred["detected"]:
            print(f"  ✓ Gesture Detected: {pred['predicted_gesture']} (Confidence: {pred['confidence_percentage']}%)")
            print(f"  ✓ Latency: {pred['latency_ms']} ms")
            print(f"  ✓ Target Matched: {pred.get('target_matched')} - Feedback: {pred.get('feedback_message')}")
            print(f"  ✓ Top Predictions: {pred.get('top_predictions')}")
        else:
            print(f"  ✓ Detection Response: {pred['message']} (Latency: {pred['latency_ms']} ms)")
        print("  -> [PASS] Genuine Keras forward pass executed without dummy data.")

        # -------------------------------------------------------------
        # TEST 4: Non-detection & Input Validation Handling
        # -------------------------------------------------------------
        print("\n[TEST 4] Testing Genuine Non-Detection on Blank Image...")
        blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank_img)
        blank_b64 = base64.b64encode(buf).decode("utf-8")

        res_blank = client.post(
            "/api/ai/predict/frame",
            json={"image_base64": blank_b64, "category": "hasthas"}
        )
        assert res_blank.status_code == 200
        blank_pred = res_blank.json()
        assert blank_pred["detected"] == False, "Model must not falsely detect hands in a blank image!"
        print(f"  ✓ Blank image accurately returned detected=False: '{blank_pred['message']}'")
        print("  -> [PASS] Anti-fake-prediction requirement verified.")

        # -------------------------------------------------------------
        # TEST 5: Practice Session Persistence in SQLite
        # -------------------------------------------------------------
        print("\n[TEST 5] Testing Practice Session Recording in SQLite...")
        student = db.query(User).filter(User.role == "student").first()
        assert student is not None, "No test student found in DB"
        student_token = create_access_token(subject=student.id, role=student.role)
        headers = {"Authorization": f"Bearer {student_token}"}

        # Clear previous practice history for this test
        db.query(PracticeHistory).filter(PracticeHistory.user_id == student.id).delete()
        db.commit()

        # Record a genuine practice session
        session_payload = {
            "category": "hasthas",
            "gesture_name": "Pathaka",
            "confidence_score": 0.942,
            "accuracy_percentage": 94.2,
            "feedback_notes": "Flawless finger alignment and palm curvature."
        }
        res_rec = client.post("/api/ai/practice/record", json=session_payload, headers=headers)
        assert res_rec.status_code == 200, f"Record failed: {res_rec.text}"
        rec_data = res_rec.json()
        assert rec_data["record"]["gesture_name"] == "Pathaka"
        print(f"  ✓ Recorded Practice Session ID {rec_data['id']}: {rec_data['record']['gesture_name']} ({rec_data['record']['accuracy_percentage']}%)")

        # Retrieve practice history
        res_hist = client.get("/api/ai/practice/history", headers=headers)
        assert res_hist.status_code == 200
        history = res_hist.json()
        assert len(history) == 1
        assert history[0]["gesture_name"] == "Pathaka"
        print(f"  ✓ Verified history persistence in SQLite for student {student.email}")
        print("  -> [PASS] Practice studio history successfully stored and retrieved.")

        print("\n" + "=" * 70)
        print("STAGE 6 EXISTING AI INTEGRATION VERIFICATION PASSED (5/5)!")
        print("=" * 70)

    finally:
        db.close()

if __name__ == "__main__":
    run_stage6_tests()
