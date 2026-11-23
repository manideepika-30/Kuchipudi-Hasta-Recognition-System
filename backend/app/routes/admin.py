from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.user import User, StudentProfile
from app.models.curriculum import Module, Lesson, StudentLessonProgress
from app.models.quiz_practice import QuizResult, PracticeHistory
from app.services.auth_service import require_admin

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get("/metrics")
def get_system_metrics(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Administrator-only system metrics: total students, completions, AI practice count."""
    total_students = db.query(User).filter(User.role == "student").count()
    total_completed_lessons = db.query(StudentLessonProgress).filter(
        StudentLessonProgress.is_completed == True,
        StudentLessonProgress.lesson_id.isnot(None)
    ).count()
    total_quizzes_taken = db.query(QuizResult).count()
    total_practice_sessions = db.query(PracticeHistory).count()

    return {
        "total_students": total_students,
        "total_completed_lessons": total_completed_lessons,
        "total_quizzes_taken": total_quizzes_taken,
        "total_practice_sessions": total_practice_sessions,
        "database_status": "Healthy (SQLite kuchipudi.db)"
    }

@router.get("/students")
def list_students(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Administrator-only student list with profile information and activity."""
    students = db.query(User).filter(User.role == "student").all()
    results = []
    for s in students:
        completed_count = db.query(StudentLessonProgress).filter(
            StudentLessonProgress.user_id == s.id,
            StudentLessonProgress.is_completed == True
        ).count()
        results.append({
            "id": s.id,
            "email": s.email,
            "full_name": s.profile.full_name if s.profile else "N/A",
            "preferred_language": s.profile.preferred_language if s.profile else "en",
            "experience_level": s.profile.experience_level if s.profile else "Beginner",
            "completed_items": completed_count,
            "created_at": s.created_at.isoformat() if s.created_at else None
        })
    return results
