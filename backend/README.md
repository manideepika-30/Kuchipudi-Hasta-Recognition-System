# Kuchipudi Backend API

FastAPI backend with SQLite database and JWT authentication for the Kuchipudi Classical Dance Learning and Practice Platform.

## Prerequisites
- Python 3.11 (Bundled in `../Python311` or system Python 3.11)
- Virtual environment at `backend/.venv`

## Architecture & Features
- **FastAPI 0.115+** with automatic OpenAPI documentation at `/docs`
- **SQLite Database** (`kuchipudi.db`) with SQLAlchemy 2.0 ORM
- **JWT Authentication** (Bearer tokens with HS256 algorithm)
- **Role-Based Access Control**:
  - `student`: Standard access to curriculum, progress tracking, and AI practice. Public registration.
  - `admin`: Restricted administrative access to user metrics and student lists. Requires CLI script creation.
- **Bilingual Support**: Student language preference (`en`/`te`) persisted in SQLite.
- **AI Architecture Prepared**: `/api/ai/health` endpoint verifies weights file existence and 69-class architecture for Stage 6 TensorFlow/Keras connection.

## How to Run

### 1. Activate Virtual Environment
PowerShell:
```powershell
.\backend\.venv\Scripts\Activate.ps1
```

### 2. Start Uvicorn Server
```powershell
.\backend\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload
```
API runs at: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

### 3. Create Admin User
```powershell
.\backend\.venv\Scripts\python.exe backend\scripts\create_admin.py --email admin@kuchipudi.edu --password AdminSecretPass2026 --name "Kuchipudi Admin"
```
