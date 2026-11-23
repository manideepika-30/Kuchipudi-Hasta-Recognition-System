from app.routes.auth import router as auth_router
from app.routes.users import router as users_router
from app.routes.curriculum import router as curriculum_router
from app.routes.admin import router as admin_router
from app.routes.ai_routes import router as ai_router

__all__ = [
    "auth_router",
    "users_router",
    "curriculum_router",
    "admin_router",
    "ai_router"
]
