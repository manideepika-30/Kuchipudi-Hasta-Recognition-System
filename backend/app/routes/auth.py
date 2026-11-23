import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.user import User, StudentProfile, RevokedToken
from app.schemas.user import UserRegister, UserLogin, Token, UserResponse
from app.utils.security import verify_password, get_password_hash, create_access_token, decode_access_token
from app.services.auth_service import get_current_user, oauth2_scheme

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new student account with hashed password and profile."""
    # Check if user already exists
    existing = db.query(User).filter(User.email == user_data.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists"
        )

    # Hash the password securely
    hashed_pwd = get_password_hash(user_data.password)

    # Create new student user (public registration always creates role="student")
    new_user = User(
        email=user_data.email.lower(),
        hashed_password=hashed_pwd,
        role="student",
        is_active=True
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Create associated student profile
    profile = StudentProfile(
        user_id=new_user.id,
        full_name=user_data.full_name,
        preferred_language=user_data.preferred_language or "en",
        experience_level="Beginner"
    )
    db.add(profile)
    db.commit()
    db.refresh(new_user)

    # Generate JWT
    token_str = create_access_token(subject=new_user.id, role=new_user.role)
    return Token(access_token=token_str, token_type="bearer", user=new_user)

@router.post("/login", response_model=Token)
def login(login_data: UserLogin, db: Session = Depends(get_db)):
    """Authenticate student or admin and issue JWT token."""
    user = db.query(User).filter(User.email == login_data.email.lower()).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )

    token_str = create_access_token(subject=user.id, role=user.role)
    return Token(access_token=token_str, token_type="bearer", user=user)

@router.get("/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Retrieve currently authenticated user and student profile."""
    return current_user

@router.post("/logout")
def logout(
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Secure token revocation endpoint: adds JWT to revoked_tokens table in SQLite."""
    if token:
        already_revoked = db.query(RevokedToken).filter(RevokedToken.token == token).first()
        if not already_revoked:
            payload = decode_access_token(token)
            exp_ts = payload.get("exp") if payload else None
            exp_dt = datetime.datetime.utcfromtimestamp(exp_ts) if exp_ts else None
            revoked_entry = RevokedToken(
                token=token,
                revoked_at=datetime.datetime.utcnow(),
                expires_at=exp_dt
            )
            db.add(revoked_entry)
            db.commit()
    return {"message": "Successfully logged out and token invalidated"}
