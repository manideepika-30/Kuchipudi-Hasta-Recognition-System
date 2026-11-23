import sys
import os

# Link pre-installed AI packages from deployment_clean virtual environment FIRST
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
CLEAN_VENV_PKGS = os.path.join(PROJECT_ROOT, "deployment_clean", ".venv", "Lib", "site-packages")

if os.path.exists(CLEAN_VENV_PKGS) and CLEAN_VENV_PKGS not in sys.path:
    sys.path.insert(0, CLEAN_VENV_PKGS)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Now import libraries safely
import uuid
import time
import json
import base64
import numpy as np
import cv2
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app
from app.database.session import get_db, SessionLocal
from app.models.user import User, StudentProfile
from app.models.curriculum import Module, Lesson, StudentLessonProgress
from app.models.quiz_practice import Quiz, Question, QuizResult, PracticeHistory
from app.utils.security import get_password_hash, verify_password

def run_stage7_e2e_test_suite():
    print("=" * 70)
    print("STAGE 7: FULL PLATFORM END-TO-END SYSTEM TEST SUITE")
    print("=" * 70)

    client = TestClient(app)
    db = SessionLocal()

    # Ensure Admin account exists for RBAC tests
    admin_user = db.query(User).filter(User.role == "admin").first()
    if not admin_user:
        admin_user = User(
            email="stage7_admin@kuchipudi.org",
            hashed_password=get_password_hash("AdminSecurePass123!"),
            role="admin",
            is_active=True
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        admin_profile = StudentProfile(user_id=admin_user.id, full_name="Guru Admin", preferred_language="en")
        db.add(admin_profile)
        db.commit()

    test_student_email = f"student_{uuid.uuid4().hex[:6]}@kuchipudi.org"
    test_student_password = "StudentStrongPass123!"

    passed_suites = 0
    total_suites = 6

    # -------------------------------------------------------------
    # TEST SUITE 1: AUTHENTICATION & TOKEN REVOCATION LIFECYCLE
    # -------------------------------------------------------------
    print("\n[SUITE 1/6] Testing Authentication, Profile & Token Revocation...")
    
    # 1.1 Register
    reg_res = client.post("/api/auth/register", json={
        "full_name": "Pragathi Dance Scholar",
        "email": test_student_email,
        "password": test_student_password,
        "preferred_language": "en"
    })
    assert reg_res.status_code in [200, 201], f"Registration failed: {reg_res.text}"
    student_token_data = reg_res.json()
    student_user_data = student_token_data["user"]
    print("  ✓ Student registration successful:", student_user_data["email"])

    # 1.2 Login
    login_res = client.post("/api/auth/login", json={
        "email": test_student_email,
        "password": test_student_password
    })
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token_data = login_res.json()
    student_token = token_data["access_token"]
    student_headers = {"Authorization": f"Bearer {student_token}"}
    print("  ✓ Login successful, JWT token obtained.")

    # 1.3 Verify Profile & Language Preference
    profile_res = client.get("/api/auth/me", headers=student_headers)
    assert profile_res.status_code == 200
    assert profile_res.json()["email"] == test_student_email
    assert profile_res.json()["profile"]["preferred_language"] == "en"
    print("  ✓ Profile authenticated and matches session.")

    # 1.4 Update Language Preference
    lang_res = client.put("/api/users/language-preference", json={"preferred_language": "te"}, headers=student_headers)
    assert lang_res.status_code == 200
    assert lang_res.json()["preferred_language"] == "te"
    print("  ✓ Language preference updated to Telugu (te) in SQLite.")

    # 1.5 Logout & Token Revocation
    logout_res = client.post("/api/auth/logout", headers=student_headers)
    assert logout_res.status_code == 200
    
    # Invalidate check: Subsequent request with revoked token must fail with 401
    revoked_res = client.get("/api/auth/me", headers=student_headers)
    assert revoked_res.status_code == 401, f"Revoked token was not rejected: {revoked_res.status_code}"
    print("  ✓ Token revocation verified: revoked token rejected with 401 Unauthorized.")

    passed_suites += 1
    print(">>> SUITE 1 PASSED!")

    # -------------------------------------------------------------
    # TEST SUITE 2: ROLE-BASED ACCESS CONTROL (RBAC) SECURITY
    # -------------------------------------------------------------
    print("\n[SUITE 2/6] Testing RBAC Security & Privilege Boundaries...")
    
    # Re-login student to obtain fresh token
    student_login = client.post("/api/auth/login", json={
        "email": test_student_email,
        "password": test_student_password
    })
    active_student_token = student_login.json()["access_token"]
    student_auth = {"Authorization": f"Bearer {active_student_token}"}

    # Student attempting to access admin routes
    blocked_metrics = client.get("/api/admin/metrics", headers=student_auth)
    assert blocked_metrics.status_code in [401, 403], f"Student accessed admin metrics: {blocked_metrics.status_code}"

    blocked_students = client.get("/api/admin/students", headers=student_auth)
    assert blocked_students.status_code in [401, 403], f"Student accessed admin student list: {blocked_students.status_code}"

    blocked_submissions = client.get("/api/quizzes/admin/all-results", headers=student_auth)
    assert blocked_submissions.status_code in [401, 403], f"Student accessed quiz submissions: {blocked_submissions.status_code}"
    print("  ✓ Student securely denied access to all admin routes (403/401).")

    # Admin Login
    admin_login = client.post("/api/auth/login", json={
        "email": admin_user.email,
        "password": "AdminSecurePass123!" if admin_user.email == "stage7_admin@kuchipudi.org" else "admin123"
    })
    if admin_login.status_code != 200:
        admin_user.hashed_password = get_password_hash("AdminSecurePass123!")
        db.commit()
        admin_login = client.post("/api/auth/login", json={
            "email": admin_user.email,
            "password": "AdminSecurePass123!"
        })
    assert admin_login.status_code == 200
    admin_token = admin_login.json()["access_token"]
    admin_auth = {"Authorization": f"Bearer {admin_token}"}

    # Admin accessing admin routes
    admin_metrics = client.get("/api/admin/metrics", headers=admin_auth)
    assert admin_metrics.status_code == 200
    assert "total_students" in admin_metrics.json()
    print("  ✓ Admin authorized to view system metrics (Total students:", admin_metrics.json()["total_students"], ")")

    admin_students_res = client.get("/api/admin/students", headers=admin_auth)
    assert admin_students_res.status_code == 200
    assert len(admin_students_res.json()) >= 1
    print("  ✓ Admin authorized to view student roster.")

    passed_suites += 1
    print(">>> SUITE 2 PASSED!")

    # -------------------------------------------------------------
    # TEST SUITE 3: CURRICULUM, MODULES & PROGRESS TRACKING
    # -------------------------------------------------------------
    print("\n[SUITE 3/6] Testing 7 Authentic Modules & 92 Lessons...")
    
    modules_res = client.get("/api/curriculum/modules")
    assert modules_res.status_code == 200
    modules = modules_res.json()
    assert len(modules) == 7, f"Expected 7 modules, found {len(modules)}"
    print("  ✓ All 7 canonical modules retrieved.")

    # Check lessons for each module
    total_lessons_found = 0
    sample_module = None
    sample_lesson = None
    for mod in modules:
        mod_lessons = mod.get("lessons", [])
        total_lessons_found += len(mod_lessons)
        if not sample_lesson and len(mod_lessons) > 0:
            sample_module = mod
            sample_lesson = mod_lessons[0]

    assert total_lessons_found == 92, f"Expected 92 lessons across curriculum, found {total_lessons_found}"
    print(f"  ✓ Total lessons verified: {total_lessons_found} lessons across 7 modules.")

    # Inspect sample lesson details
    assert sample_lesson["name_en"] is not None
    assert sample_lesson["shloka"] is not None
    assert sample_lesson["meaning_te"] is not None
    print(f"  ✓ Lesson '{sample_lesson['name_en']}' loaded with authentic Sanskrit shloka and Telugu explanation.")

    # Toggle lesson completion for student
    toggle_res = client.post(
        "/api/curriculum/progress/toggle",
        json={
            "module_id": sample_module["id"],
            "lesson_id": sample_lesson["id"],
            "is_completed": True
        },
        headers=student_auth
    )
    assert toggle_res.status_code == 200
    assert toggle_res.json()["is_completed"] is True
    print("  ✓ Lesson completion successfully recorded in SQLite.")

    # Check student overall curriculum progress
    progress_res = client.get("/api/curriculum/progress", headers=student_auth)
    assert progress_res.status_code == 200
    progress_info = progress_res.json()
    assert progress_info["total_completed_lessons"] >= 1
    assert progress_info["overall_progress_percentage"] > 0
    print(f"  ✓ Student overall progress tracked: {progress_info['overall_progress_percentage']}% completed.")

    passed_suites += 1
    print(">>> SUITE 3 PASSED!")

    # -------------------------------------------------------------
    # TEST SUITE 4: ASSIGNMENTS, ANTI-CHEAT & EVALUATION
    # -------------------------------------------------------------
    print("\n[SUITE 4/6] Testing Assignments, Anti-Cheat Masking & Grading...")

    quizzes_res = client.get("/api/quizzes")
    assert quizzes_res.status_code == 200
    quizzes = quizzes_res.json()
    assert len(quizzes) >= 3, f"Expected at least 3 assignment types, found {len(quizzes)}"
    
    quiz_types = [q["quiz_type"] for q in quizzes]
    assert "theory_mcq" in quiz_types
    assert "image_identification" in quiz_types
    assert "practical_camera" in quiz_types
    print("  ✓ All 3 assignment types present (Theory MCQ, Image ID, Practical Camera).")

    # Anti-cheat check on student question delivery
    theory_quiz = [q for q in quizzes if q["quiz_type"] == "theory_mcq"][0]
    theory_detail_res = client.get(f"/api/quizzes/{theory_quiz['id']}", headers=student_auth)
    assert theory_detail_res.status_code == 200
    delivered_questions = theory_detail_res.json()["questions"]
    for q in delivered_questions:
        assert "correct_option" not in q, f"SECURITY LEAK: correct_option was exposed to student in question {q['id']}!"
    print("  ✓ Anti-cheat verification passed: correct_option is stripped before delivery.")

    # Student submission
    sample_answers = {}
    for q in delivered_questions:
        sample_answers[str(q["id"])] = "A"  # Select option "A" for all
    
    submit_res = client.post(
        f"/api/quizzes/{theory_quiz['id']}/submit",
        json={"answers": sample_answers},
        headers=student_auth
    )
    assert submit_res.status_code == 200, f"Submit failed: {submit_res.text}"
    result_data = submit_res.json()
    assert "score" in result_data
    assert "percentage" in result_data
    assert "passed" in result_data
    print(f"  ✓ Submission evaluated: Score {result_data['score']}/{result_data['total_questions']} ({result_data['percentage']}%) - Passed: {result_data['passed']}")

    # Verify student history contains this result
    history_res = client.get("/api/quizzes/results/history", headers=student_auth)
    assert history_res.status_code == 200
    assert len(history_res.json()) >= 1
    print("  ✓ Student assignment history retrieved successfully.")

    # Admin oversight: Admin views submissions
    admin_subs = client.get("/api/quizzes/admin/all-results", headers=admin_auth)
    assert admin_subs.status_code == 200
    assert len(admin_subs.json()) >= 1
    print(f"  ✓ Admin successfully retrieved {len(admin_subs.json())} submitted exam records.")

    passed_suites += 1
    print(">>> SUITE 4 PASSED!")

    # -------------------------------------------------------------
    # TEST SUITE 5: REAL AI INFERENCE ENGINE & PRACTICE PERSISTENCE
    # -------------------------------------------------------------
    print("\n[SUITE 5/6] Testing Real Keras AI Inference & Practice Persistence...")

    status_res = client.get("/api/ai/status")
    assert status_res.status_code == 200
    ai_status = status_res.json()
    assert ai_status["supported_classes"]["total"] == 69
    assert ai_status["supported_classes"]["hasthas"] == 52
    assert ai_status["supported_classes"]["shiro_bhedas"] == 9
    assert ai_status["supported_classes"]["drishti_bhedas"] == 8
    print("  ✓ AI Inference Engine is fully online with 69 classes (Hastha: 52, Shiro: 9, Drishti: 8).")

    # Blank frame (no hands/face) -> detected: False, no synthetic hallucination
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', blank_img)
    blank_b64 = base64.b64encode(buffer).decode('utf-8')
    
    predict_res = client.post("/api/ai/predict/frame", json={
        "image_base64": blank_b64,
        "category": "hasthas"
    })
    assert predict_res.status_code == 200, f"Blank predict failed: {predict_res.text}"
    assert predict_res.json()["detected"] is False
    assert "No hands detected" in predict_res.json().get("message", "")
    print("  ✓ Blank image accurately returned detected=False without hallucinations.")

    # Real frame test with authentic image asset
    sample_img_path = Path(PROJECT_ROOT) / "frontend" / "public" / "assets" / "curriculum" / "Asamyutha_Hasthas_p1_0.jpg"
    if sample_img_path.exists():
        real_img = cv2.imread(str(sample_img_path))
        if real_img is not None:
            _, buffer_real = cv2.imencode('.jpg', real_img)
            real_b64 = base64.b64encode(buffer_real).decode('utf-8')
            t0 = time.time()
            real_res = client.post("/api/ai/predict/frame", json={
                "image_base64": real_b64,
                "category": "hasthas"
            })
            dt = (time.time() - t0) * 1000
            assert real_res.status_code == 200
            print(f"  ✓ Authentic frame inference completed in {dt:.1f}ms. Detected: {real_res.json()['detected']}")

    # Record practice session
    practice_payload = {
        "category": "hasthas",
        "gesture_name": "Pataka",
        "confidence_score": 0.925,
        "accuracy_percentage": 92.5,
        "feedback_notes": "Excellent thumb tuck and flat palm posture maintained."
    }
    rec_res = client.post("/api/ai/practice/record", json=practice_payload, headers=student_auth)
    assert rec_res.status_code == 200, f"Record practice failed: {rec_res.text}"
    assert rec_res.json()["record"]["gesture_name"] == "Pataka"
    print("  ✓ Student practice session recorded in SQLite.")

    # Fetch practice history
    p_hist_res = client.get("/api/ai/practice/history", headers=student_auth)
    assert p_hist_res.status_code == 200
    assert len(p_hist_res.json()) >= 1
    assert p_hist_res.json()[0]["gesture_name"] == "Pataka"
    print("  ✓ Student practice history verified in SQLite database.")

    passed_suites += 1
    print(">>> SUITE 5 PASSED!")

    # -------------------------------------------------------------
    # TEST SUITE 6: DATA INTEGRITY & SECURITY CHECKS
    # -------------------------------------------------------------
    print("\n[SUITE 6/6] Testing Security, Password Hashing & Anti-Tampering...")

    # Verify password hash in DB is not plaintext
    student_record = db.query(User).filter(User.email == test_student_email).first()
    assert student_record.hashed_password != test_student_password
    assert verify_password(test_student_password, student_record.hashed_password)
    assert not verify_password("WrongPassword123!", student_record.hashed_password)
    print("  ✓ Password securely hashed (bcrypt) and verified in SQLite.")

    # Verify student cannot forge their own score
    tamper_res = client.post(
        f"/api/quizzes/{theory_quiz['id']}/submit",
        json={"answers": {}, "score": 100, "percentage": 100.0, "passed": True},
        headers=student_auth
    )
    assert tamper_res.status_code == 200
    # Server calculates score from answers dictionary, ignoring injected 'score' fields
    assert tamper_res.json()["score"] == 0
    print("  ✓ Score tampering prevention verified: server strictly recalculates from answer map.")

    passed_suites += 1
    print(">>> SUITE 6 PASSED!")

    print("\n" + "=" * 70)
    print(f"ALL {passed_suites}/{total_suites} END-TO-END SYSTEM TEST SUITES PASSED PERFECTLY!")
    print("=" * 70)
    db.close()

if __name__ == "__main__":
    run_stage7_e2e_test_suite()
