from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.user import User, StudentProfile
from app.schemas.user import UserResponse, StudentProfileResponse, LanguagePreferenceUpdate, UserProfileUpdate
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/profile", response_model=StudentProfileResponse)
def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get the current student's profile."""
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return profile

@router.put("/profile", response_model=StudentProfileResponse)
def update_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update student profile details."""
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    if not profile:
        profile = StudentProfile(user_id=current_user.id, full_name="Student")
        db.add(profile)

    if profile_data.full_name is not None:
        profile.full_name = profile_data.full_name
    if profile_data.preferred_language is not None:
        profile.preferred_language = profile_data.preferred_language
    if profile_data.experience_level is not None:
        profile.experience_level = profile_data.experience_level
    if profile_data.bio is not None:
        profile.bio = profile_data.bio

    db.commit()
    db.refresh(profile)
    return profile

@router.put("/language-preference")
def update_language(
    pref: LanguagePreferenceUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update student's preferred UI language (English/Telugu) in SQLite."""
    profile = db.query(StudentProfile).filter(StudentProfile.user_id == current_user.id).first()
    if not profile:
        profile = StudentProfile(user_id=current_user.id, full_name=current_user.email, preferred_language=pref.preferred_language)
        db.add(profile)
    else:
        profile.preferred_language = pref.preferred_language
    db.commit()
    return {"message": "Language preference updated", "preferred_language": pref.preferred_language}
