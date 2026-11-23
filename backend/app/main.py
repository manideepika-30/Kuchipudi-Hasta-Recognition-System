import os
import datetime
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.config import settings
from app.database.session import engine, SessionLocal
from app.database.base import Base
from app.models.curriculum import Module, Lesson
from app.routes.auth import router as auth_router
from app.routes.users import router as users_router
from app.routes.curriculum import router as curriculum_router
from app.routes.admin import router as admin_router
from app.routes.ai_routes import router as ai_router
from app.routes.quiz_routes import router as quiz_router

from app.database.seed_data import get_authentic_modules_data
from app.database.seed_quizzes import seed_quizzes_data

def init_db_and_seed():
    """Create all tables in SQLite and seed full authentic curriculum modules & quizzes if needed."""
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        total_lessons = db.query(Lesson).count()
        if total_lessons < 90:
            # Clear old partial data to load full authentic canon from PDFs
            db.query(Lesson).delete()
            db.query(Module).delete()
            db.commit()

            modules_data = get_authentic_modules_data()
            for m_info in modules_data:
                lessons_data = m_info.pop("lessons", [])
                mod = Module(**m_info)
                db.add(mod)
                db.commit()
                db.refresh(mod)
                for l_info in lessons_data:
                    lesson = Lesson(module_id=mod.id, **l_info)
                    db.add(lesson)
                db.commit()

        # Seed quizzes & questions
        seed_quizzes_data(db)
    finally:
        db.close()

# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API supporting the AI-Powered Kuchipudi Learning and Practice Platform."
)

# Exception handler for cleaner validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_messages = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        error_messages.append(f"{loc}: {msg}")
    return JSONResponse(
        status_code=422,
        content={"detail": "; ".join(error_messages)}
    )

# Setup CORS Middleware to allow localhost, Vercel deployments, and Render backends
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|.*\.vercel\.app|.*\.onrender\.com)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
def on_startup():
    init_db_and_seed()

# Register Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(users_router, prefix=settings.API_V1_STR)
app.include_router(curriculum_router, prefix=settings.API_V1_STR)
app.include_router(admin_router, prefix=settings.API_V1_STR)
app.include_router(ai_router, prefix=settings.API_V1_STR)
app.include_router(quiz_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/docs",
        "supported_classes": 69
    }

@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
