"""
Automated Test Suite for Stage 5: Assignments and Examinations.
Validates Type A (Theory MCQ), Type B (Image Identification), and Type C (Practical Camera Exam).
Tests server-side scoring, anti-tampering, student history, and admin oversight.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.database.session import SessionLocal
from app.models.user import User
from app.models.quiz_practice import Quiz, Question, QuizResult
from app.utils.security import create_access_token

def run_stage5_tests():
    print("=" * 70)
    print("RUNNING STAGE 5: ASSIGNMENTS & EXAMINATIONS VERIFICATION SUITE")
    print("=" * 70)

    client = TestClient(app)
    db = SessionLocal()

    try:
        # -------------------------------------------------------------
        # TEST 1: Public Quiz Catalog
        # -------------------------------------------------------------
        print("\n[TEST 1] Verifying All 3 Canonical Assignment Types in Catalog...")
        res = client.get("/api/quizzes")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        quizzes = res.json()
        assert len(quizzes) == 3, f"Expected 3 quizzes, got {len(quizzes)}"

        quiz_types = {q["quiz_type"]: q for q in quizzes}
        assert "theory_mcq" in quiz_types, "Missing Type A: theory_mcq"
        assert "image_identification" in quiz_types, "Missing Type B: image_identification"
        assert "practical_camera" in quiz_types, "Missing Type C: practical_camera"

        q_a = quiz_types["theory_mcq"]
        q_b = quiz_types["image_identification"]
        q_c = quiz_types["practical_camera"]

        print(f"  ✓ Type A: {q_a['title_en']} ({q_a['total_questions']} questions, Pass: {q_a['passing_score_pct']}%)")
        print(f"  ✓ Type B: {q_b['title_en']} ({q_b['total_questions']} questions, Pass: {q_b['passing_score_pct']}%)")
        print(f"  ✓ Type C: {q_c['title_en']} ({q_c['total_questions']} tasks, Pass: {q_c['passing_score_pct']}%)")
        assert q_a["total_questions"] == 10
        assert q_b["total_questions"] == 10
        assert q_c["total_questions"] == 3
        print("  -> [PASS] All 3 assignment types correctly structured in SQLite.")

        # -------------------------------------------------------------
        # TEST 2: Anti-Cheating Question Delivery (No correct_option)
        # -------------------------------------------------------------
        print("\n[TEST 2] Testing Question Delivery & Anti-Cheating Sanitization...")
        res_q = client.get(f"/api/quizzes/{q_b['id']}")
        assert res_q.status_code == 200
        q_detail = res_q.json()
        assert len(q_detail["questions"]) == 10

        # Verify correct_option and explanation are NOT returned in public question payload
        for quest in q_detail["questions"]:
            assert "correct_option" not in quest, f"Security leak: correct_option exposed in question {quest['id']}"
            assert "explanation_en" not in quest, f"Security leak: explanation_en exposed in question {quest['id']}"
            assert "explanation_te" not in quest, f"Security leak: explanation_te exposed in question {quest['id']}"
            assert quest["image_url"] is not None, f"Image identification question {quest['id']} missing image_url!"
        print("  ✓ Public question endpoint safely delivers questions without answer leaks.")
        print("  ✓ Type B questions properly bind authentic image assets.")
        print("  -> [PASS] Anti-cheating delivery confirmed.")

        # -------------------------------------------------------------
        # TEST 3: Student Submission & Server-Side Scoring
        # -------------------------------------------------------------
        print("\n[TEST 3] Testing Student Quiz Submission & Server-Side Evaluation...")
        student = db.query(User).filter(User.role == "student").first()
        assert student is not None, "No test student found in DB"
        student_token = create_access_token(subject=student.id, role=student.role)
        headers = {"Authorization": f"Bearer {student_token}"}

        # Retrieve actual questions for Quiz 1 to submit perfect answers
        quiz_1_questions = db.query(Question).filter(Question.quiz_id == q_a["id"]).all()
        answers_payload = {str(q.id): q.correct_option for q in quiz_1_questions}

        res_sub = client.post(
            f"/api/quizzes/{q_a['id']}/submit",
            json={"answers": answers_payload},
            headers=headers
        )
        assert res_sub.status_code == 200, f"Submit failed: {res_sub.text}"
        sub_data = res_sub.json()
        assert sub_data["score"] == 10
        assert sub_data["total_questions"] == 10
        assert sub_data["percentage"] == 100.0
        assert sub_data["passed"] == True
        assert len(sub_data["review_details"]) == 10
        print(f"  ✓ Perfect score evaluated server-side: {sub_data['score']}/{sub_data['total_questions']} (100%) - Passed: {sub_data['passed']}")

        # Test partial score on Quiz 2 (Type B image identification) with 8 correct, 2 wrong
        quiz_2_questions = db.query(Question).filter(Question.quiz_id == q_b["id"]).all()
        answers_payload_2 = {}
        for idx, q in enumerate(quiz_2_questions):
            if idx < 8:
                answers_payload_2[str(q.id)] = q.correct_option
            else:
                answers_payload_2[str(q.id)] = "D" if q.correct_option != "D" else "C"

        res_sub_2 = client.post(
            f"/api/quizzes/{q_b['id']}/submit",
            json={"answers": answers_payload_2},
            headers=headers
        )
        assert res_sub_2.status_code == 200
        sub_data_2 = res_sub_2.json()
        assert sub_data_2["score"] == 8
        assert sub_data_2["total_questions"] == 10
        assert sub_data_2["percentage"] == 80.0
        assert sub_data_2["passed"] == True
        print(f"  ✓ Partial score evaluated server-side: {sub_data_2['score']}/{sub_data_2['total_questions']} (80%) - Passed: {sub_data_2['passed']}")
        print("  -> [PASS] Server-side calculation and grading verified.")

        # -------------------------------------------------------------
        # TEST 4: Student Assessment History
        # -------------------------------------------------------------
        print("\n[TEST 4] Testing Student Assessment History Retrieval...")
        res_hist = client.get("/api/quizzes/results/history", headers=headers)
        assert res_hist.status_code == 200
        history = res_hist.json()
        assert len(history) >= 2, f"Expected at least 2 history records, found {len(history)}"
        print(f"  ✓ Found {len(history)} persistent attempt records for Student {student.email}")
        for h in history[:2]:
            print(f"    - [{h['quiz_type']}] {h['quiz_title_en']}: {h['score']}/{h['total_questions']} ({h['percentage']}%) Passed: {h['passed']}")
        print("  -> [PASS] Student assessment history persists in SQLite.")

        # -------------------------------------------------------------
        # TEST 5: Admin Assignment Review & Authorization Security
        # -------------------------------------------------------------
        print("\n[TEST 5] Testing Admin Assignment Review & Security Boundary...")
        # A student should be rejected with 403 Forbidden
        res_admin_student = client.get("/api/quizzes/admin/all-results", headers=headers)
        assert res_admin_student.status_code == 403, f"Expected 403 Forbidden for student, got {res_admin_student.status_code}"
        print("  ✓ Security Check: Student prevented from accessing admin assignment submissions (HTTP 403).")

        # Admin user access
        admin = db.query(User).filter(User.role == "admin").first()
        assert admin is not None, "No admin user found in DB"
        admin_token = create_access_token(subject=admin.id, role=admin.role)
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        res_admin = client.get("/api/quizzes/admin/all-results", headers=admin_headers)
        assert res_admin.status_code == 200
        all_results = res_admin.json()
        assert len(all_results) >= 2
        print(f"  ✓ Admin successfully retrieved {len(all_results)} global assignment records.")
        print("  -> [PASS] Role-based access control and admin oversight verified.")

        print("\n" + "=" * 70)
        print("STAGE 5 ASSIGNMENTS & EXAMINATIONS VERIFICATION PASSED (5/5)!")
        print("=" * 70)

    finally:
        db.close()

if __name__ == "__main__":
    run_stage5_tests()
