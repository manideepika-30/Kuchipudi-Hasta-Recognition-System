from typing import List, Optional
import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.user import User
from app.models.curriculum import Module, Lesson, StudentLessonProgress
from app.schemas.curriculum import ModuleResponse, ProgressToggleRequest, ProgressResponse
from app.services.auth_service import get_current_user, get_optional_current_user

router = APIRouter(prefix="/curriculum", tags=["Curriculum"])

@router.get("/modules", response_model=List[ModuleResponse])
def get_all_modules(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve all 7 Kuchipudi learning modules with lessons and completion status."""
    modules = db.query(Module).order_by(Module.module_number).all()
    user_id = current_user.id if current_user else None

    # Fetch user's completed records if authenticated
    completed_module_ids = set()
    completed_lesson_ids = set()
    if user_id:
        progress_records = db.query(StudentLessonProgress).filter(
            StudentLessonProgress.user_id == user_id,
            StudentLessonProgress.is_completed == True
        ).all()
        for r in progress_records:
            if r.lesson_id is None:
                completed_module_ids.add(r.module_id)
            else:
                completed_lesson_ids.add(r.lesson_id)

    results = []
    for m in modules:
        m_lessons = []
        completed_lessons_count = 0
        for les in m.lessons:
            is_comp = les.id in completed_lesson_ids
            if is_comp:
                completed_lessons_count += 1
            m_lessons.append({
                "id": les.id,
                "module_id": les.module_id,
                "lesson_order": les.lesson_order,
                "name_en": les.name_en,
                "name_te": les.name_te,
                "sanskrit_name": les.sanskrit_name,
                "category": les.category,
                "description_en": les.description_en,
                "description_te": les.description_te,
                "shloka": les.shloka,
                "meaning_en": les.meaning_en,
                "meaning_te": les.meaning_te,
                "image_url": les.image_url,
                "is_completed": is_comp
            })

        is_mod_completed = (m.id in completed_module_ids) or (len(m_lessons) > 0 and completed_lessons_count == len(m_lessons))
        results.append({
            "id": m.id,
            "module_number": m.module_number,
            "title_en": m.title_en,
            "title_te": m.title_te,
            "sanskrit_title": m.sanskrit_title,
            "source_pdf": m.source_pdf,
            "total_classes": m.total_classes,
            "description_en": m.description_en,
            "description_te": m.description_te,
            "completed_classes": completed_lessons_count,
            "is_completed": is_mod_completed,
            "lessons": m_lessons
        })

    return results

@router.get("/progress", response_model=ProgressResponse)
def get_student_progress(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve full progress summary for student dashboard."""
    modules = db.query(Module).order_by(Module.module_number).all()
    progress_records = db.query(StudentLessonProgress).filter(
        StudentLessonProgress.user_id == current_user.id,
        StudentLessonProgress.is_completed == True
    ).all()

    completed_mod_ids = set()
    completed_les_ids = set()
    for r in progress_records:
        if r.lesson_id is None:
            completed_mod_ids.add(r.module_id)
        else:
            completed_les_ids.add(r.lesson_id)

    total_lessons = db.query(Lesson).count()
    total_completed_lessons = len(completed_les_ids)
    overall_percentage = (total_completed_lessons / max(total_lessons, 1)) * 100

    # Determine next incomplete module
    next_incomplete_module = None
    next_incomplete_lesson = None

    for m in modules:
        if m.id not in completed_mod_ids:
            # Check if has incomplete lessons
            for les in m.lessons:
                if les.id not in completed_les_ids:
                    next_incomplete_module = m.module_number
                    next_incomplete_lesson = f"{m.title_en} — {les.name_en}"
                    break
            if next_incomplete_module is not None:
                break
            if len(m.lessons) == 0:
                next_incomplete_module = m.module_number
                next_incomplete_lesson = m.title_en
                break

    return ProgressResponse(
        user_id=current_user.id,
        completed_module_ids=list(completed_mod_ids),
        completed_lesson_ids=list(completed_les_ids),
        total_completed_lessons=total_completed_lessons,
        overall_progress_percentage=round(overall_percentage, 1),
        next_incomplete_module=next_incomplete_module,
        next_incomplete_lesson=next_incomplete_lesson
    )

@router.post("/progress/toggle")
def toggle_lesson_progress(
    req: ProgressToggleRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle completion status for a lesson or full module in SQLite."""
    record = db.query(StudentLessonProgress).filter(
        StudentLessonProgress.user_id == current_user.id,
        StudentLessonProgress.module_id == req.module_id,
        StudentLessonProgress.lesson_id == req.lesson_id
    ).first()

    if not record:
        record = StudentLessonProgress(
            user_id=current_user.id,
            module_id=req.module_id,
            lesson_id=req.lesson_id,
            is_completed=req.is_completed,
            completed_at=datetime.datetime.utcnow() if req.is_completed else None
        )
        db.add(record)
    else:
        record.is_completed = req.is_completed
        record.completed_at = datetime.datetime.utcnow() if req.is_completed else None

    db.commit()
    return {
        "status": "success",
        "module_id": req.module_id,
        "lesson_id": req.lesson_id,
        "is_completed": req.is_completed
    }
