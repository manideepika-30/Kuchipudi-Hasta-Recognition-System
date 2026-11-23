from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

class LessonResponse(BaseModel):
    id: int
    module_id: int
    lesson_order: int
    name_en: str
    name_te: str
    sanskrit_name: Optional[str] = None
    category: Optional[str] = None
    description_en: Optional[str] = None
    description_te: Optional[str] = None
    shloka: Optional[str] = None
    meaning_en: Optional[str] = None
    meaning_te: Optional[str] = None
    image_url: Optional[str] = None
    is_completed: bool = False

    class Config:
        from_attributes = True

class ModuleResponse(BaseModel):
    id: int
    module_number: int
    title_en: str
    title_te: str
    sanskrit_title: Optional[str] = None
    source_pdf: Optional[str] = None
    total_classes: int
    description_en: Optional[str] = None
    description_te: Optional[str] = None
    completed_classes: int = 0
    is_completed: bool = False
    lessons: List[LessonResponse] = []

    class Config:
        from_attributes = True

class ProgressToggleRequest(BaseModel):
    module_id: int
    lesson_id: Optional[int] = None
    is_completed: bool

class ProgressResponse(BaseModel):
    user_id: int
    completed_module_ids: List[int]
    completed_lesson_ids: List[int]
    total_completed_lessons: int
    overall_progress_percentage: float
    next_incomplete_module: Optional[int] = None
    next_incomplete_lesson: Optional[str] = None
