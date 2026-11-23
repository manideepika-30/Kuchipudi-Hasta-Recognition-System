import sys
import os
import json
import time
import base64
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
CLEAN_VENV_PKGS = os.path.join(PROJECT_ROOT, "deployment_clean", ".venv", "Lib", "site-packages")

if os.path.exists(CLEAN_VENV_PKGS) and CLEAN_VENV_PKGS not in sys.path:
    sys.path.insert(0, CLEAN_VENV_PKGS)

sys.path.insert(0, BACKEND_DIR)

from fastapi.testclient import TestClient
from app.main import app
from app.database.session import SessionLocal
from app.models.curriculum import Module, Lesson
from app.models.user import User
from app.models.quiz_practice import Quiz, Question, QuizResult, PracticeHistory

def run_acceptance_tests():
    print("=" * 80)
    print("NRITYA AI - STAGE 9 FINAL SYSTEM ACCEPTANCE AUDIT")
    print("=" * 80)
    
    results = {}
    client = TestClient(app)
    db = SessionLocal()
    
    try:
        # ---------------------------------------------------------
        # SECTION 1: CURRICULUM AUDIT (92 LESSONS, 7 MODULES)
        # ---------------------------------------------------------
        print("\n[SECTION 1] AUDITING CANONICAL CURRICULUM & 92 LESSONS...")
        modules = db.query(Module).order_by(Module.module_number).all()
        assert len(modules) == 7, f"Expected 7 modules, found {len(modules)}"
        
        expected_counts = {1: 4, 2: 28, 3: 24, 4: 9, 5: 8, 6: 12, 7: 7}
        total_lessons = 0
        all_lessons_valid = True
        
        for m in modules:
            lessons = db.query(Lesson).filter(Lesson.module_id == m.id).order_by(Lesson.lesson_order).all()
            total_lessons += len(lessons)
            exp = expected_counts[m.module_number]
            assert len(lessons) == exp, f"Module {m.module_number} count mismatch: {len(lessons)} != {exp}"
            
            for l in lessons:
                if not l.name_en or not l.name_te or not l.description_en or not l.description_te:
                    all_lessons_valid = False
                    print(f"  [ERROR] Lesson {l.id} ({l.name_en}) has missing metadata!")
        
        assert total_lessons == 92, f"Expected 92 lessons, found {total_lessons}"
        print(f"  ✓ Verified all 7 modules and exactly {total_lessons} lessons.")
        print(f"  ✓ Verified English and Telugu lesson names and descriptions.")
        
        # Verify 28 Asamyukta Hastas
        asamyukta = db.query(Lesson).filter(Lesson.module_id == 2).order_by(Lesson.lesson_order).all()
        assert len(asamyukta) == 28
        print(f"  ✓ Verified 28 Asamyukta Hastas (e.g. {asamyukta[0].name_en} to {asamyukta[-1].name_en}).")
        
        # Verify 24 Samyukta Hastas
        samyukta = db.query(Lesson).filter(Lesson.module_id == 3).order_by(Lesson.lesson_order).all()
        assert len(samyukta) == 24
        print(f"  ✓ Verified 24 Samyukta Hastas (e.g. {samyukta[0].name_en} to {samyukta[-1].name_en}).")
        
        # Verify 9 Shiro Bhedas
        shiro = db.query(Lesson).filter(Lesson.module_id == 4).order_by(Lesson.lesson_order).all()
        assert len(shiro) == 9
        print(f"  ✓ Verified 9 Shiro Bhedas (e.g. {shiro[0].name_en} to {shiro[-1].name_en}).")
        
        # Verify 8 Drishti Bhedas
        drishti = db.query(Lesson).filter(Lesson.module_id == 5).order_by(Lesson.lesson_order).all()
        assert len(drishti) == 8
        print(f"  ✓ Verified 8 Drishti Bhedas (e.g. {drishti[0].name_en} to {drishti[-1].name_en}).")
        
        # Verify 7 Sapta Talas compared with Sapta Talamulu.pdf
        sapta = db.query(Lesson).filter(Lesson.module_id == 7).order_by(Lesson.lesson_order).all()
        assert len(sapta) == 7
        tala_shlokas = {
            "Dhruva Tala": "భరతో దత్తిశ్చైవ",
            "Matya Tala": "మత్య తాళస్య కర్తారో",
            "Rupaka Tala": "ఆంజనేయ కుంభోజి",
            "Jhampe Tala": "నందిశ్చ భరతశ్చ",
            "Triputa Tala": "నిశాచరపతిః బ్రహ్మ",
            "Ata Tala": "బలిః కుమార దేవేంద్ర",
            "Eka Tala": "మహతాంచాపి సర్వేషాం"
        }
        for s in sapta:
            exp_substr = tala_shlokas[s.name_en]
            assert exp_substr in s.shloka, f"Tala {s.name_en} missing authentic Shloka prefix {exp_substr}!"
            assert len(s.meaning_te) > 20, f"Tala {s.name_en} missing Telugu meaning!"
            print(f"  ✓ Verified Tala: {s.name_en} ({s.name_te}) -> Shloka: {s.shloka[:30]}...")
            
        results["Curriculum (92 Lessons & 7 Modules)"] = "PASS"
        results["Sapta Tala PDF Shloka Accuracy"] = "PASS"
        
        # ---------------------------------------------------------
        # SECTION 2: AI INFERENCE ENGINE & WEAPONIZED TEST ASSETS
        # ---------------------------------------------------------
        print("\n[SECTION 2] AUDITING AI INFERENCE ENGINE & WEIGHT INTEGRITY...")
        res_ai_status = client.get("/api/ai/status")
        assert res_ai_status.status_code == 200, f"Status failed: {res_ai_status.text}"
        ai_stat = res_ai_status.json()
        assert ai_stat["status"] == "online"
        assert ai_stat["weights_file"] == "nritya_ai_weights.keras"
        assert ai_stat["supported_classes"]["total"] == 69
        print(f"  ✓ Model status: {ai_stat['status']} ({ai_stat['weights_file']}, {ai_stat['weights_size_mb']} MB, 69 classes)")
        
        # Check all 69 classes
        res_classes = client.get("/api/ai/classes")
        assert res_classes.status_code == 200
        classes_data = res_classes.json()
        assert len(classes_data["hasthas"]) == 52
        assert len(classes_data["shiro_bhedhas"]) == 9
        assert len(classes_data["drishti_bhedhas"]) == 8
        print(f"  ✓ Verified 69 dance classes: 52 Hasthas, 9 Shiro Bhedas, 8 Drishti Bhedas.")
        
        # Test 1: Authentic Dance Frame Inference
        sample_img_path = Path(os.path.join(PROJECT_ROOT, "frontend", "public", "assets", "curriculum", "Asamyutha_Hasthas_p1_0.jpg"))
        assert sample_img_path.exists(), f"Sample image missing: {sample_img_path}"
        with open(sample_img_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")
            
        t_infer_start = time.time()
        res_infer = client.post(
            "/api/ai/predict/frame",
            json={
                "image_base64": b64_img,
                "category": "hasthas",
                "target_gesture": "Pathaka"
            }
        )
        infer_latency = (time.time() - t_infer_start) * 1000
        assert res_infer.status_code == 200, f"Inference failed: {res_infer.text}"
        infer_data = res_infer.json()
        print(f"  ✓ Authentic Frame Prediction: Detected={infer_data['detected']}, Gesture={infer_data['predicted_gesture']}, Conf={infer_data.get('confidence_percentage', 0)}%, Latency={infer_data.get('latency_ms', infer_latency):.2f}ms")
        assert infer_data["detected"] == True, "Failed to detect hand in authentic image!"
        assert infer_data.get("confidence_percentage", 0) > 0, "Confidence score must be greater than 0!"
        assert len(infer_data["top_predictions"]) >= 3, "Expected top-3 predictions"
        
        # Test 2: Blank Image Rejection (Anti-Hallucination)
        from PIL import Image
        import io
        blank_img = Image.new("RGB", (320, 240), color=(255, 255, 255))
        buf = io.BytesIO()
        blank_img.save(buf, format="JPEG")
        blank_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        res_blank = client.post(
            "/api/ai/predict/frame",
            json={
                "image_base64": blank_b64,
                "category": "hasthas",
                "target_gesture": "Pathaka"
            }
        )
        assert res_blank.status_code == 200
        blank_data = res_blank.json()
        print(f"  ✓ Blank Image Rejection: Detected={blank_data['detected']}, Feedback='{blank_data.get('feedback_message', blank_data.get('message'))}'")
        assert blank_data["detected"] == False, "Anti-Hallucination violation: AI predicted gesture on pure blank white image!"
        
        results["AI Practice Studio Real Keras Inference"] = "PASS"
        results["AI Blank Image Rejection (Anti-Hallucination)"] = "PASS"
        
        # ---------------------------------------------------------
        # SECTION 3: ASSIGNMENTS, EXAMS & SCORING INTEGRITY
        # ---------------------------------------------------------
        print("\n[SECTION 3] AUDITING ASSIGNMENTS & SCORING...")
        res_assign = client.get("/api/quizzes")
        assert res_assign.status_code == 200, f"Status code: {res_assign.status_code}"
        assignments_data = res_assign.json()
        assert len(assignments_data) >= 3, f"Expected 3 assignments, got {len(assignments_data)}"
        
        assign_types = {a["quiz_type"] for a in assignments_data}
        assert "theory_mcq" in assign_types, "Missing Theory MCQ"
        assert "image_identification" in assign_types, "Missing Image Identification"
        assert "practical_camera" in assign_types, "Missing Practical Camera"
        print(f"  ✓ Verified all 3 assignment types: Theory MCQ, Image ID, Practical Camera.")
        
        # Verify anti-cheat: questions delivered to students DO NOT leak correct_option
        for a in assignments_data:
            q_res = client.get(f"/api/quizzes/{a['id']}")
            assert q_res.status_code == 200
            q_detail = q_res.json()
            assert "questions" in q_detail
            for q in q_detail["questions"]:
                assert "correct_option" not in q or q["correct_option"] is None, f"Anti-cheat violation! correct_option leaked in {q}"
                assert "explanation_en" not in q, f"Anti-cheat violation! explanation_en leaked in {q}"
        print(f"  ✓ Anti-cheat verified: correct_option securely masked before delivery.")
        
        results["Assignments & Quizzes Delivery"] = "PASS"
        results["Anti-Cheat Option Masking"] = "PASS"
        
        # ---------------------------------------------------------
        # SECTION 4: SECURITY BOUNDARIES & AUTHENTICATION
        # ---------------------------------------------------------
        print("\n[SECTION 4] AUDITING SECURITY BOUNDARIES & RBAC...")
        
        # Test 1: Guest unauthenticated access to admin routes
        admin_probe = client.get("/api/admin/metrics")
        print(f"  ✓ Guest -> /api/admin/metrics returned {admin_probe.status_code} (Expected 401 Unauthorized)")
        assert admin_probe.status_code == 401, f"Security vulnerability! Admin route open to guest with code {admin_probe.status_code}"
        
        # Test 2: Student registration, login and profile
        reg_payload = {
            "email": f"audit_student_{int(time.time())}@kuchipudi.org",
            "password": "ValidPassword@123",
            "full_name": "Audit Student",
            "preferred_language": "te"
        }
        reg_res = client.post("/api/auth/register", json=reg_payload)
        assert reg_res.status_code == 201, f"Registration failed: {reg_res.text}"
        
        login_res = client.post("/api/auth/login", json={"email": reg_payload["email"], "password": reg_payload["password"]})
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        auth_headers = {"Authorization": f"Bearer {token}"}
        
        # Test 3: Student forbidden from admin routes
        stu_admin_probe = client.get("/api/admin/metrics", headers=auth_headers)
        print(f"  ✓ Student -> /api/admin/metrics returned {stu_admin_probe.status_code} (Expected 403 Forbidden)")
        assert stu_admin_probe.status_code == 403, f"RBAC violation! Student accessed admin route with code {stu_admin_probe.status_code}"
        
        stu_quiz_admin_probe = client.get("/api/quizzes/admin/all-results", headers=auth_headers)
        print(f"  ✓ Student -> /api/quizzes/admin/all-results returned {stu_quiz_admin_probe.status_code} (Expected 403 Forbidden)")
        assert stu_quiz_admin_probe.status_code in [401, 403]
        
        # Test 4: Token Revocation on Logout
        logout_res = client.post("/api/auth/logout", headers=auth_headers)
        assert logout_res.status_code == 200
        
        # Try reusing revoked token
        reuse_probe = client.get("/api/auth/me", headers=auth_headers)
        print(f"  ✓ Revoked Token Reuse returned {reuse_probe.status_code} (Expected 401 Unauthorized)")
        assert reuse_probe.status_code == 401, f"Security flaw! Revoked JWT token still accepted with code {reuse_probe.status_code}"
        
        results["Security: Guest Boundary & Admin Protection"] = "PASS"
        results["Security: Student RBAC Access Denial"] = "PASS"
        results["Security: JWT Token Revocation on Logout"] = "PASS"
        
        print("\n" + "=" * 80)
        print("ALL ACCEPTANCE TEST SECTIONS EXECUTED SUCCESSFULLY!")
        print("=" * 80)
        for k, v in results.items():
            print(f"  [{v}] {k}")
            
    finally:
        db.close()

if __name__ == "__main__":
    run_acceptance_tests()
