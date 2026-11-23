from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

class StudentProfileBase(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    preferred_language: str = Field(default="en", pattern="^(en|te)$")
    experience_level: Optional[str] = "Beginner"
    bio: Optional[str] = None

class StudentProfileResponse(StudentProfileBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserRegister(BaseModel):
    email: str = Field(..., pattern=EMAIL_REGEX)
    password: str = Field(..., min_length=6, max_length=100)
    full_name: str = Field(..., min_length=2, max_length=100)
    preferred_language: Optional[str] = Field(default="en", pattern="^(en|te)$")

class UserLogin(BaseModel):
    email: str = Field(..., pattern=EMAIL_REGEX)
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool
    created_at: datetime
    profile: Optional[StudentProfileResponse] = None

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[str] = None

class LanguagePreferenceUpdate(BaseModel):
    preferred_language: str = Field(..., pattern="^(en|te)$")

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    preferred_language: Optional[str] = Field(None, pattern="^(en|te)$")
    experience_level: Optional[str] = None
    bio: Optional[str] = None
