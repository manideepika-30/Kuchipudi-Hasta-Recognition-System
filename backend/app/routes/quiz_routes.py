import json
import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.user import User
from app.models.quiz_practice import Quiz, Question, QuizResult
from app.services.auth_service import get_current_user, get_optional_current_user, require_admin

router = APIRouter(prefix="/quizzes", tags=["Quizzes & Assignments"])

# =====================================================================
# Pydantic Schemas
# =====================================================================
class QuizSubmissionPayload(BaseModel):
    answers: Dict[str, str]  # { "question_id": "A" }

class QuestionPublicOut(BaseModel):
    id: int
    order_num: int
    image_url: Optional[str]
    question_text_en: str
    question_text_te: str
    option_a_en: str
    option_a_te: str
    option_b_en: str
    option_b_te: str
    option_c_en: str
    option_c_te: str
    option_d_en: str
    option_d_te: str

class QuizDetailOut(BaseModel):
    id: int
    module_id: Optional[int]
    quiz_type: str
    title_en: str
    title_te: str
    description_en: Optional[str]
    description_te: Optional[str]
    time_limit_minutes: int
    passing_score_pct: float
    total_questions: int
    questions: List[QuestionPublicOut]

class QuizSummaryOut(BaseModel):
    id: int
    module_id: Optional[int]
    quiz_type: str
    title_en: str
    title_te: str
    description_en: Optional[str]
    description_te: Optional[str]
    time_limit_minutes: int
    passing_score_pct: float
    total_questions: int
    user_best_score: Optional[float] = None
    user_passed: Optional[bool] = None
    attempts_count: int = 0

# =====================================================================
# API Endpoints
# =====================================================================

@router.get("", response_model=List[QuizSummaryOut])
def list_quizzes(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    """
    List all available quizzes and assignments.
    If authenticated, includes the student's best score and pass status.
    """
    quizzes = db.query(Quiz).order_by(Quiz.id).all()
    results = []

    for q in quizzes:
        best_score = None
        user_passed = None
        attempts = 0

        if current_user:
            user_results = db.query(QuizResult).filter(
                QuizResult.quiz_id == q.id,
                QuizResult.user_id == current_user.id
            ).all()
            attempts = len(user_results)
            if attempts > 0:
                best_score = max(r.percentage for r in user_results)
                user_passed = any(r.passed for r in user_results)

        results.append(QuizSummaryOut(
            id=q.id,
            module_id=q.module_id,
            quiz_type=q.quiz_type,
            title_en=q.title_en,
            title_te=q.title_te,
            description_en=q.description_en,
            description_te=q.description_te,
            time_limit_minutes=q.time_limit_minutes,
            passing_score_pct=q.passing_score_pct,
            total_questions=len(q.questions),
            user_best_score=best_score,
            user_passed=user_passed,
            attempts_count=attempts
        ))

    return results


@router.get("/{quiz_id}", response_model=QuizDetailOut)
def get_quiz_details(
    quiz_id: int,
    db: Session = Depends(get_db)
):
    """
    Fetch quiz questions for taking the test.
    Note: correct_option and explanations are strictly omitted to prevent client tampering.
    """
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quiz with ID {quiz_id} not found."
        )

    public_questions = [
        QuestionPublicOut(
            id=q.id,
            order_num=q.order_num,
            image_url=q.image_url,
            question_text_en=q.question_text_en,
            question_text_te=q.question_text_te,
            option_a_en=q.option_a_en,
            option_a_te=q.option_a_te,
            option_b_en=q.option_b_en,
            option_b_te=q.option_b_te,
            option_c_en=q.option_c_en,
            option_c_te=q.option_c_te,
            option_d_en=q.option_d_en,
            option_d_te=q.option_d_te
        )
        for q in quiz.questions
    ]

    return QuizDetailOut(
        id=quiz.id,
        module_id=quiz.module_id,
        quiz_type=quiz.quiz_type,
        title_en=quiz.title_en,
        title_te=quiz.title_te,
        description_en=quiz.description_en,
        description_te=quiz.description_te,
        time_limit_minutes=quiz.time_limit_minutes,
        passing_score_pct=quiz.passing_score_pct,
        total_questions=len(public_questions),
        questions=public_questions
    )


@router.post("/{quiz_id}/submit")
def submit_quiz_answers(
    quiz_id: int,
    payload: QuizSubmissionPayload,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
):
    """
    Server-side evaluation of student answers.
    Calculates verified score in SQLite, marks pass/fail, and returns detailed explanations.
    Tamper-proof: client cannot specify score directly.
    Supports authenticated students (persists to SQLite) and guest learners (evaluates safely without fake records).
    """
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    questions = quiz.questions
    if not questions:
        raise HTTPException(status_code=400, detail="Quiz has no questions configured.")

    correct_count = 0
    total_count = len(questions)
    review_details = []

    for q in questions:
        q_id_str = str(q.id)
        student_choice = payload.answers.get(q_id_str, "").strip().upper()
        is_correct = (student_choice == q.correct_option.strip().upper())

        if is_correct:
            correct_count += 1

        review_details.append({
            "question_id": q.id,
            "order_num": q.order_num,
            "image_url": q.image_url,
            "question_text_en": q.question_text_en,
            "question_text_te": q.question_text_te,
            "student_choice": student_choice,
            "correct_option": q.correct_option,
            "is_correct": is_correct,
            "explanation_en": q.explanation_en,
            "explanation_te": q.explanation_te
        })

    percentage = round((correct_count / total_count) * 100.0, 2)
    has_passed = (percentage >= quiz.passing_score_pct)

    completed_iso = datetime.datetime.utcnow().isoformat()
    result_id = None

    # Persist verified result in SQLite only if authenticated student
    if current_user:
        result_record = QuizResult(
            user_id=current_user.id,
            quiz_id=quiz.id,
            score=correct_count,
            total_questions=total_count,
            percentage=percentage,
            passed=has_passed,
            details_json=json.dumps(review_details, ensure_ascii=False),
            completed_at=datetime.datetime.utcnow()
        )
        db.add(result_record)
        db.commit()
        db.refresh(result_record)
        result_id = result_record.id
        completed_iso = result_record.completed_at.isoformat()

    return {
        "result_id": result_id,
        "quiz_id": quiz.id,
        "quiz_title_en": quiz.title_en,
        "quiz_title_te": quiz.title_te,
        "score": correct_count,
        "total_questions": total_count,
        "percentage": percentage,
        "passing_score_pct": quiz.passing_score_pct,
        "passed": has_passed,
        "completed_at": completed_iso,
        "review_details": review_details,
        "guest_mode": (current_user is None)
    }


@router.get("/results/history")
def get_user_quiz_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve logged-in student's complete assessment history.
    """
    records = db.query(QuizResult).filter(
        QuizResult.user_id == current_user.id
    ).order_by(QuizResult.completed_at.desc()).all()

    output = []
    for r in records:
        output.append({
            "id": r.id,
            "quiz_id": r.quiz_id,
            "quiz_title_en": r.quiz.title_en if r.quiz else "N/A",
            "quiz_title_te": r.quiz.title_te if r.quiz else "N/A",
            "quiz_type": r.quiz.quiz_type if r.quiz else "theory_mcq",
            "score": r.score,
            "total_questions": r.total_questions,
            "percentage": r.percentage,
            "passed": r.passed,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None
        })
    return output


@router.get("/admin/all-results")
def get_admin_quiz_results(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Administrator-only view of all student assignment submissions.
    """
    records = db.query(QuizResult).order_by(QuizResult.completed_at.desc()).all()
    output = []
    for r in records:
        output.append({
            "id": r.id,
            "student_id": r.user_id,
            "student_email": r.user.email if r.user else "N/A",
            "student_name": r.user.profile.full_name if r.user and r.user.profile else "N/A",
            "quiz_id": r.quiz_id,
            "quiz_title_en": r.quiz.title_en if r.quiz else "N/A",
            "quiz_type": r.quiz.quiz_type if r.quiz else "theory_mcq",
            "score": r.score,
            "total_questions": r.total_questions,
            "percentage": r.percentage,
            "passed": r.passed,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None
        })
    return output
