"""
Enhanced Comprehensive Verification Script for Stage 3 Post-Review Verification.
Tests:
1. Student registration directly via API (underlying React frontend calls).
2. Login and logout with SQLite Token Revocation check (tokens rejected after logout).
3. Attempt by student to access admin routes (403 Forbidden).
4. Attempt by student to promote account (rejected / ignored, role remains student).
5. Data persistence in SQLite across DB reloads.
6. Real student progress retrieval from SQLite without relying on localStorage.
7. AI model readiness preservation.
"""
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app, init_db_and_seed
from app.database.session import SessionLocal, engine
from app.models.user import User, RevokedToken
from app.models.curriculum import Module, Lesson, StudentLessonProgress
from app.utils.security import verify_password
from scripts.create_admin import create_or_promote_admin

def run_post_review_verification():
    print("=" * 70)
    print("RUNNING STAGE 3 POST-REVIEW VERIFICATION SUITE")
    print("=" * 70)

    # Initialize tables (including revoked_tokens)
    init_db_and_seed()
    client = TestClient(app)
    db = SessionLocal()

    # -------------------------------------------------------------
    # CHECK 1 & 2: Registration, Login and Secure Token Revocation
    # -------------------------------------------------------------
    print("\n[CHECK 1 & 2] Testing Student Registration, Login & Logout...")
    test_email = "dance_learner@kuchipudi.edu"
    test_pwd = "StudentSecurePass2026"
    test_name = "Kavya Natarajan"

    # Clean existing test user if present
    existing = db.query(User).filter(User.email == test_email).first()
    if existing:
        db.delete(existing)
        db.commit()

    # Register
    res_reg = client.post("/api/auth/register", json={
        "email": test_email,
        "password": test_pwd,
        "full_name": test_name,
        "preferred_language": "te"
    })
    assert res_reg.status_code == 201, f"Registration failed: {res_reg.text}"
    reg_data = res_reg.json()
    student_token = reg_data["access_token"]
    student_id = reg_data["user"]["id"]
    print(f"  -> Student registered successfully (ID: {student_id}, Role: {reg_data['user']['role']})")
    assert reg_data["user"]["role"] == "student"

    # Login
    res_login = client.post("/api/auth/login", json={
        "email": test_email,
        "password": test_pwd
    })
    assert res_login.status_code == 200, f"Login failed: {res_login.text}"
    active_token = res_login.json()["access_token"]
    print("  -> Login succeeded, active JWT issued.")

    # Access protected route with active token
    auth_header = {"Authorization": f"Bearer {active_token}"}
    res_me = client.get("/api/auth/me", headers=auth_header)
    assert res_me.status_code == 200
    print(f"  -> Authenticated /api/auth/me response: {res_me.json()['email']}")

    # -------------------------------------------------------------
    # CHECK 3: Secure Token Revocation Verification
    # -------------------------------------------------------------
    print("\n[CHECK 3] Testing JWT Revocation After Logout...")
    res_logout = client.post("/api/auth/logout", headers=auth_header)
    assert res_logout.status_code == 200, f"Logout failed: {res_logout.text}"
    print(f"  -> Logout endpoint response: {res_logout.json()['message']}")

    # Verify token is recorded in SQLite revoked_tokens table
    revoked_in_db = db.query(RevokedToken).filter(RevokedToken.token == active_token).first()
    assert revoked_in_db is not None, "Token was not recorded in revoked_tokens table!"
    print(f"  -> [VERIFIED] Token persisted in SQLite revoked_tokens (Revoked at: {revoked_in_db.revoked_at})")

    # Attempt to reuse the revoked token -> MUST RETURN 401
    res_reused = client.get("/api/auth/me", headers=auth_header)
    assert res_reused.status_code == 401, f"Expected 401 for revoked token, got {res_reused.status_code}"
    print(f"  -> [PASS] Access denied with revoked token: HTTP 401 ('{res_reused.json()['detail']}')")

    # Log in again to get fresh token for remaining checks
    res_fresh = client.post("/api/auth/login", json={"email": test_email, "password": test_pwd})
    fresh_token = res_fresh.json()["access_token"]
    auth_header = {"Authorization": f"Bearer {fresh_token}"}

    # -------------------------------------------------------------
    # CHECK 4: SQLite Persistence Across Database Sessions
    # -------------------------------------------------------------
    print("\n[CHECK 4] Testing SQLite Data Persistence Across Database Sessions...")
    # Update language preference
    client.put("/api/users/language-preference", json={"preferred_language": "te"}, headers=auth_header)

    # Toggle lesson progress
    res_toggle = client.post("/api/curriculum/progress/toggle", json={
        "module_id": 2,
        "lesson_id": 5,
        "is_completed": True
    }, headers=auth_header)
    print(f"  -> Toggle response ({res_toggle.status_code}): {res_toggle.text}")

    # Close DB session completely, reopen new session to simulate process restart
    db.close()
    new_db = SessionLocal()
    persisted_user = new_db.query(User).filter(User.email == test_email).first()
    assert persisted_user is not None
    assert persisted_user.profile.preferred_language == "te"
    print(f"  -> [VERIFIED] User profile persisted: {persisted_user.profile.full_name}, Lang: {persisted_user.profile.preferred_language}")

    persisted_progress = new_db.query(StudentLessonProgress).filter(
        StudentLessonProgress.user_id == persisted_user.id,
        StudentLessonProgress.is_completed == True
    ).first()
    assert persisted_progress is not None
    assert persisted_progress.module_id == 2 and persisted_progress.lesson_id == 5
    print(f"  -> [VERIFIED] Lesson progress persisted: Module {persisted_progress.module_id}, Lesson {persisted_progress.lesson_id}")
    new_db.close()

    # -------------------------------------------------------------
    # CHECK 5: Student Cannot Access Admin Routes or Self-Promote
    # -------------------------------------------------------------
    print("\n[CHECK 5] Testing RBAC Security: Student Access Restrictions & Self-Promotion Prevention...")
    # Student accessing admin routes
    res_admin_metrics = client.get("/api/admin/metrics", headers=auth_header)
    assert res_admin_metrics.status_code == 403, f"Expected 403, got {res_admin_metrics.status_code}"
    print(f"  -> [PASS] Student blocked from /api/admin/metrics: HTTP 403 ('{res_admin_metrics.json()['detail']}')")

    res_admin_students = client.get("/api/admin/students", headers=auth_header)
    assert res_admin_students.status_code == 403, f"Expected 403, got {res_admin_students.status_code}"
    print(f"  -> [PASS] Student blocked from /api/admin/students: HTTP 403 ('{res_admin_students.json()['detail']}')")

    # Attempt self-promotion via profile update
    res_hack = client.put("/api/users/profile", json={
        "full_name": "Hacked Admin",
        "role": "admin"  # Attempt to inject role
    }, headers=auth_header)
    assert res_hack.status_code == 200

    # Verify role in DB remains 'student'
    db_check = SessionLocal()
    u_check = db_check.query(User).filter(User.id == student_id).first()
    assert u_check.role == "student", f"CRITICAL: User role was promoted to {u_check.role}!"
    print(f"  -> [PASS] Self-promotion exploit prevented. User role remains: '{u_check.role}'")
    db_check.close()

    # -------------------------------------------------------------
    # CHECK 6: Real Student Progress Retrieval from FastAPI
    # -------------------------------------------------------------
    print("\n[CHECK 6] Testing Real Student Progress Retrieval from FastAPI...")
    res_prog = client.get("/api/curriculum/progress", headers=auth_header)
    assert res_prog.status_code == 200
    prog = res_prog.json()
    assert 5 in prog["completed_lesson_ids"], "Lesson 5 not found in completed lessons!"
    print(f"  -> Total completed lessons returned by API: {prog['total_completed_lessons']}")
    print(f"  -> Completed lesson IDs: {prog['completed_lesson_ids']}")
    print(f"  -> Next incomplete module: {prog['next_incomplete_module']}")
    print("  -> [PASS] FastAPI returns true relational progress from SQLite.")

    # -------------------------------------------------------------
    # CHECK 7 & 8: AI Architecture & Independent Virtual Env
    # -------------------------------------------------------------
    print("\n[CHECK 7 & 8] Testing AI Model Architecture Preservation & Virtual Environment...")
    res_ai = client.get("/api/ai/health")
    assert res_ai.status_code == 200
    ai_info = res_ai.json()
    assert ai_info["weights_file_exists"] is True
    assert ai_info["supported_recognition_classes"]["total_classes"] == 69
    print(f"  -> [PASS] Model file preserved ({ai_info['weights_file_size_mb']} MB, 69 classes).")
    print(f"  -> Python interpreter: {sys.executable}")
    assert ".venv" in sys.executable, "Not running in isolated backend virtual environment!"
    print("  -> [PASS] Running strictly in isolated virtual environment backend/.venv.")

    print("\n" + "=" * 70)
    print("ALL POST-REVIEW VERIFICATION CHECKS PASSED (8/8)!")
    print("=" * 70)

if __name__ == "__main__":
    run_post_review_verification()
