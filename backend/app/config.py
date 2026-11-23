import os
from typing import List
from pydantic_settings import BaseSettings

# Calculate absolute path to backend directory so sqlite DB is consistently located
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB_PATH = os.path.join(BACKEND_DIR, "kuchipudi.db").replace("\\", "/")

class Settings(BaseSettings):
    PROJECT_NAME: str = "Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # AI Engine toggle (True for local execution with TensorFlow; False on free 512MB RAM cloud tiers)
    ENABLE_AI: bool = os.getenv("ENABLE_AI", "true").lower() in ("true", "1", "yes")
    
    # Absolute Database URL
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")
    
    # JWT Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "kuchipudi-classical-dance-secret-key-change-in-prod-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # CORS
    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174"
    )
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    class Config:
        env_file = os.path.join(BACKEND_DIR, ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()
