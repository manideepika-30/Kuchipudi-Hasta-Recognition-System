import os
import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.user import User
from app.models.quiz_practice import PracticeHistory
from app.services.auth_service import get_current_user, get_optional_current_user
from app.services.ai_service import get_ai_engine, WEIGHTS_PATH

router = APIRouter(prefix="/ai", tags=["AI & Practice Studio"])

# =====================================================================
# Pydantic Schemas
# =====================================================================

class FramePredictionPayload(BaseModel):
    image_base64: str
    category: str = "hasthas"  # "hasthas", "shiro", "drishti"
    target_gesture: Optional[str] = None

class PracticeRecordPayload(BaseModel):
    category: str
    gesture_name: str
    confidence_score: float
    accuracy_percentage: float
    feedback_notes: Optional[str] = None

# =====================================================================
# Endpoints
# =====================================================================

@router.get("/status")
def get_ai_status():
    """
    Check the live status of the consolidated AI recognition engine.
    Verifies weights file and loaded architecture without modifying weights.
    """
    engine = get_ai_engine()
    weights_exist = WEIGHTS_PATH.exists()
    weights_size_mb = round(os.path.getsize(WEIGHTS_PATH) / (1024 * 1024), 2) if weights_exist else 0.0

    if not getattr(engine, "enabled", True):
        return {
            "status": "standby",
            "ai_enabled": False,
            "engine_initialized": False,
            "weights_file": WEIGHTS_PATH.name if weights_exist else "nritya_ai_weights.keras",
            "weights_path": str(WEIGHTS_PATH),
            "weights_size_mb": weights_size_mb,
            "message": "AI real-time inference is in standby on free-tier 512MB RAM cloud hosting. Full 7 curriculum modules, 92 lessons, and quizzes are 100% operational.",
            "supported_classes": {
                "hasthas": 52,
                "shiro_bhedas": 9,
                "drishti_bhedas": 8,
                "total": 69
            },
            "model_layers": []
        }

    return {
        "status": "online",
        "ai_enabled": True,
        "engine_initialized": engine._initialized,
        "weights_file": WEIGHTS_PATH.name,
        "weights_path": str(WEIGHTS_PATH),
        "weights_size_mb": weights_size_mb,
        "supported_classes": {
            "hasthas": 52,
            "shiro_bhedas": 9,
            "drishti_bhedas": 8,
            "total": 69
        },
        "model_layers": [l.name for l in engine.unified_model.layers] if engine.unified_model else []
    }


@router.get("/classes")
def get_supported_classes():
    """
    Retrieve all 69 supported Kuchipudi recognition classes organized by category.
    """
    engine = get_ai_engine()
    return engine.get_supported_classes()


@router.post("/predict/frame")
def predict_camera_frame(payload: FramePredictionPayload):
    """
    Execute real-time classification on a camera frame using the authentic AI model.
    Never outputs fake or dummy data.
    """
    engine = get_ai_engine()
    cat = payload.category.lower().strip()

    try:
        frame_rgb = engine.decode_image_frame(payload.image_base64)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image data: {str(e)}"
        )

    if cat in ("hasthas", "hastha", "hastas", "hasta"):
        prediction = engine.predict_hastha(frame_rgb)
    elif cat in ("shiro", "shiro_bhedhas", "head"):
        prediction = engine.predict_shiro(frame_rgb)
    elif cat in ("drishti", "drishti_bhedhas", "eyes"):
        prediction = engine.predict_drishti(frame_rgb)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown category: {cat}. Must be 'hasthas', 'shiro', or 'drishti'."
        )

    # Optional feedback generation against target gesture
    if payload.target_gesture and prediction.get("detected"):
        pred_name = prediction.get("predicted_gesture") or prediction.get("predicted_movement") or prediction.get("predicted_glance") or ""
        target = payload.target_gesture.strip()
        matched = (pred_name.lower() == target.lower())
        prediction["target_gesture"] = target
        prediction["target_matched"] = matched

        conf = prediction.get("confidence_percentage", 0.0)
        if matched and conf >= 70.0:
            prediction["feedback_message"] = f"Excellent! {target} gesture recognized with high confidence ({conf}%)."
        elif matched:
            prediction["feedback_message"] = f"Good attempt at {target}. Hold your fingers steady to improve stability ({conf}%)."
        else:
            prediction["feedback_message"] = f"Detected {pred_name} instead of {target}. Check finger alignment and posture."

    return prediction


@router.post("/practice/record")
def record_practice_session(
    payload: PracticeRecordPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Persist student's AI practice session metrics into SQLite practice_history table.
    """
    record = PracticeHistory(
        user_id=current_user.id,
        category=payload.category,
        gesture_name=payload.gesture_name,
        confidence_score=payload.confidence_score,
        accuracy_percentage=payload.accuracy_percentage,
        feedback_notes=payload.feedback_notes,
        created_at=datetime.datetime.utcnow()
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "message": "Practice session recorded successfully in SQLite database.",
        "record": {
            "id": record.id,
            "category": record.category,
            "gesture_name": record.gesture_name,
            "confidence_score": record.confidence_score,
            "accuracy_percentage": record.accuracy_percentage,
            "feedback_notes": record.feedback_notes,
            "created_at": record.created_at.isoformat()
        }
    }


@router.get("/practice/history")
def get_student_practice_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve logged-in student's complete practice history.
    """
    records = db.query(PracticeHistory).filter(
        PracticeHistory.user_id == current_user.id
    ).order_by(PracticeHistory.created_at.desc()).limit(50).all()

    return [
        {
            "id": r.id,
            "category": r.category,
            "gesture_name": r.gesture_name,
            "confidence_score": r.confidence_score,
            "accuracy_percentage": r.accuracy_percentage,
            "feedback_notes": r.feedback_notes,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in records
    ]
