# Nritya AI — Final Project Freeze & Handover Documentation

**Official Project Title:** Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System  
**Document Classification:** Final Engineering Handover & System Freeze  
**Date:** November 23, 2026  
**Status:** **PROJECT FROZEN & READY FOR DEMONSTRATION**  
**Compliance:** 100% Non-Destructive Freeze (No Model Retraining, No Weights Alteration, Datasets Intact, Auth & Admin Preserved, Zero Git Push)

---

## 1. Executive Summary & Freeze Declaration

This handover document confirms the completion and formal freeze of **Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System**. 

The platform has successfully passed all Stage 9 independent acceptance audits with a **100% pass rate (42/42 tests passed)**. The application is operating in a dual-mode configuration:
1. **Public Guest Learning Portal:** Visitors immediately access the homepage, permanent left sidebar, all 7 curriculum modules, all 92 authentic lessons, interactive AI Practice Studio, and Assignments without forced login or registration dialogs. Guest lesson progress and quiz results persist locally in `localStorage`.
2. **Preserved Secure Authentication & Administration:** The complete backend authentication stack (FastAPI, SQLAlchemy, SQLite, bcrypt password hashing, HS256 JWT tokens with database revocation, and Role-Based Access Control) remains intact for authorized students and administrators. Administrative metrics and rosters are strictly secured behind `HTTP 403 Forbidden` / `HTTP 401 Unauthorized`.

---

## 2. Final Project Directory Structure

```
kuchipudi/
├── backend/                             # Python FastAPI SQLite Backend
│   ├── app/
│   │   ├── database/                    # SQLAlchemy database engine & base
│   │   │   ├── base.py                  # Declarative base model registry
│   │   │   ├── session.py               # SessionLocal engine & DB dependency
│   │   │   ├── seed_data.py             # 92 canonical lessons seeding data
│   │   │   └── seed_quizzes.py          # 23 authentic questions seeding data
│   │   ├── models/                      # SQLAlchemy ORM relational models
│   │   │   ├── user.py                  # User, StudentProfile, RevokedToken
│   │   │   ├── curriculum.py            # Module, Lesson, StudentLessonProgress
│   │   │   └── quiz_practice.py         # Quiz, Question, QuizResult, PracticeHistory
│   │   ├── routes/                      # REST API routers
│   │   │   ├── auth.py                  # Register, login, logout, profile
│   │   │   ├── users.py                 # Language preference updates
│   │   │   ├── curriculum.py            # Modules, lessons, progress toggle
│   │   │   ├── quiz_routes.py           # Quizzes, questions, anti-cheat submit
│   │   │   ├── ai_routes.py             # Frame prediction, practice history
│   │   │   └── admin.py                 # Admin metrics and student roster
│   │   ├── schemas/                     # Pydantic request/response schemas
│   │   ├── services/                    # Business logic & AI inference service
│   │   │   ├── ai_inference_service.py  # Singleton Keras engine & MediaPipe
│   │   │   └── auth_service.py          # Security, JWT, password verification
│   │   ├── utils/
│   │   │   └── security.py              # Bcrypt, JWT encode/decode, token revocation
│   │   ├── config.py                    # Settings (JWT secret, DB URL, CORS)
│   │   └── main.py                      # FastAPI application entry point & CORS
│   ├── scripts/                         # Operational & automated audit scripts
│   │   ├── create_admin.py              # Secure admin CLI creation tool
│   │   ├── run_all_audits.py            # SEDS automated audit runner (Stages 4-7)
│   │   ├── run_stage9_acceptance_tests.py # Stage 9 comprehensive acceptance test
│   │   ├── seed_authentic_curriculum.py # Database curriculum initialization
│   │   ├── test_stage4_curriculum.py    # Stage 4 curriculum test suite
│   │   ├── test_stage5_assignments.py   # Stage 5 assignments test suite
│   │   ├── test_stage6_ai.py            # Stage 6 AI forward-pass test suite
│   │   └── test_stage7_e2e_system.py    # Stage 7 full platform E2E test suite
│   ├── .venv/                           # Python 3.11 isolated virtual environment
│   ├── kuchipudi.db                     # Authoritative SQLite database
│   ├── kuchipudi_backup.db              # Validated atomic database backup
│   ├── requirements.txt                 # Backend Python dependencies
│   └── .env.example                     # Backend environment template
├── frontend/                            # React 18 + Vite Frontend Application
│   ├── public/
│   │   └── assets/
│   │       ├── curriculum/              # Authentic Hasta, Shiro & Drishti illustrations
│   │       └── asamyutha_sample.jpeg    # Fallback dance plate image
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx               # Top header, language toggle & mobile Menu button
│   │   │   ├── PermanentSidebar.jsx     # Permanent left navigation & mobile drawer
│   │   │   ├── ModuleLessonViewer.jsx   # 92 lessons reader with Shlokas & meanings
│   │   │   ├── PracticeStudio.jsx       # Real-time webcam AI studio with contrast tabs
│   │   │   ├── AssignmentViewer.jsx     # 3 examination types & anti-cheat engine
│   │   │   ├── ProgressView.jsx         # Visual curriculum progress tracking
│   │   │   ├── ResultsView.jsx          # Historical exam & practice records
│   │   │   ├── AuthModal.jsx            # Preserved student/admin login modal
│   │   │   └── Footer.jsx               # Classical footer with Natyashastra Shloka
│   │   ├── pages/
│   │   │   ├── HomePage.jsx             # Public landing page with classical hero
│   │   │   ├── AboutPage.jsx            # Kuchipudi history & guru parampara
│   │   │   └── StudentDashboard.jsx     # Dashboard view
│   │   ├── services/
│   │   │   └── api.js                   # REST API client with guest localStorage overlay
│   │   ├── styles/
│   │   │   └── theme.css                # Classical Indian design tokens & theme aliases
│   │   ├── utils/
│   │   │   └── translations.js          # Bilingual English/Telugu dictionaries
│   │   ├── App.jsx                      # Root application layout & route state
│   │   └── main.jsx                     # React DOM mount entry
│   ├── dist/                            # Verified production build output
│   ├── package.json                     # Frontend Node dependencies
│   └── vite.config.js                   # Vite dev-server config & proxy to port 9000
├── deployment_clean/                    # Pre-trained Machine Learning Model Core
│   ├── .venv/                           # Pre-configured AI virtual environment
│   └── nritya_ai_weights.keras          # Canonical Keras model weights (1.31 MB)
├── Data set/                            # Authoritative Original Image Datasets
│   ├── Asamyuktha hasthalu/             # 28 Single-hand gesture training datasets
│   └── Samyuktha hasthalu/              # 24 Combined-hand gesture training datasets
├── docs/pdfs/                           # Extracted Canonical Telugu Dance Treatises
│   ├── Sapta Talamulu.pdf               # Original Anandamayi Nrityalaya rhythm treatise
│   ├── Asamyutha Hasthas.pdf            # Authoritative single-hand gesture text
│   ├── Samyutha Hasthalu.pdf            # Authoritative double-hand gesture text
│   ├── Shiro Bhedhas.pdf                # Head movements treatise
│   ├── Drishti Bhedhas.pdf              # Eye movements treatise
│   └── Kuchipudi Natya Charitra.pdf     # History and dance origins
├── documentation/                       # Software Architecture & Engineering Records
│   ├── Architecture.md                  # Detailed system architecture document
│   ├── Database_Design.md               # Relational entity-relationship documentation
│   ├── Requirement_Traceability.md      # Full Traceability Matrix (RTM)
│   ├── SRS.md                           # Software Requirements Specification
│   └── reports/                         # Stage-by-stage engineering reports
│       ├── UI_AND_SAPTA_TALA_CORRECTION_REPORT.md # UI & Tala shloka fixes
│       ├── STAGE_9_FINAL_ACCEPTANCE_REPORT.md     # 42/42 validation audit report
│       └── FINAL_PROJECT_HANDOVER.md              # This document
├── .env.example                         # Root environment template
├── .gitignore                           # Git ignore rules (DB & env exclusions)
├── README.md                            # Comprehensive user & developer guide
├── DEPLOYMENT.md                        # Deployment documentation
├── start_all.bat                        # One-click launcher for Backend + Frontend
├── start_backend.bat                    # One-click launcher for FastAPI on port 9000
├── start_frontend.bat                   # One-click launcher for Vite on port 3000
└── theory.zip                           # Preserved original ZIP archive of dance treatises
```

---

## 3. Startup and Execution Instructions

### A. One-Click Startup (Recommended for Windows)

Double-click `start_all.bat` in the project root:
```cmd
start_all.bat
```
This script automatically:
1. Opens a dedicated terminal launching the FastAPI backend on `http://127.0.0.1:9000`.
2. Pauses 3 seconds for the AI model to initialize.
3. Opens a second terminal launching the React/Vite development server on `http://localhost:3000`.
4. The application is ready to view at **`http://localhost:3000`**.

---

### B. Manual Command-Line Startup

#### Step 1: FastAPI Backend (`http://127.0.0.1:9000`)
```powershell
# Navigate to backend directory
cd "e:\pragathi\My kuchipudi project aidl\kuchipudi\backend"

# Activate Python 3.11 virtual environment
.\.venv\Scripts\Activate.ps1

# Launch FastAPI on port 9000
python -m uvicorn app.main:app --host 127.0.0.1 --port 9000 --reload
```
- API Base URL: `http://127.0.0.1:9000/api`
- Interactive OpenAPI / Swagger Docs: `http://127.0.0.1:9000/docs`
- Health Check: `http://127.0.0.1:9000/api/health`

> **Port Note:** Port **9000** is the standardized port for Nritya AI to prevent socket permission errors occasionally encountered on ports 8000 and 8080 under Windows.

#### Step 2: React Frontend (`http://localhost:3000`)
```powershell
# In a new terminal, navigate to frontend directory
cd "e:\pragathi\My kuchipudi project aidl\kuchipudi\frontend"

# Launch Vite development server
npm run dev
```
- Open browser at: `http://localhost:3000` (or `http://127.0.0.1:3000`)

---

## 4. Required Dependencies & Environment Variables

### A. Python Backend Requirements (`backend/requirements.txt`)
- Python: `3.11.x`
- Core Framework: `fastapi==0.165.0`, `uvicorn==0.54.0`, `pydantic==2.14.4`, `starlette==1.10.0`
- Database: `SQLAlchemy==2.1.9`
- Authentication & Security: `python-jose==3.5.0`, `passlib==1.7.4`, `bcrypt==5.0.0`, `cryptography==50.0.2`
- AI & Image Processing: `numpy==2.4.6`, `Pillow==13.0.0`, `pypdf==7.1.1`
- Model Integration (from `deployment_clean/.venv`): `tensorflow==2.17.x`, `keras==3.x`, `mediapipe`

### B. Node.js Frontend Requirements (`frontend/package.json`)
- Node.js: `v18.x`, `v20.x`, or `v22.x` (LTS)
- Core: `react==^18.3.1`, `react-dom==^18.3.1`
- UI Icons: `lucide-react==^0.460.0`
- Build System: `vite==^6.0.1`, `@vitejs/plugin-react==^4.3.4`

### C. Environment Variables (`.env`)
```ini
# Backend Environment (.env or backend/.env)
PROJECT_NAME="Nritya AI"
ENVIRONMENT="development"
PORT=9000
HOST="127.0.0.1"
DATABASE_URL="sqlite:///./kuchipudi.db"
SECRET_KEY="nritya-ai-kuchipudi-secret-key-change-in-production-secure-hash"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=1440
ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"

# Frontend Environment (frontend/.env)
VITE_API_URL="/api"
```

---

## 5. Database Architecture & Backup Procedure

### A. Database Location
- Active Database File: `backend/kuchipudi.db`
- SQLite Relational Tables:
  1. `users` — Student and administrator login credentials (bcrypt hashed).
  2. `student_profiles` — Profile details, full name, preferred language (`en`/`te`).
  3. `revoked_tokens` — Blacklisted JWT `jti` tokens for immediate logout revocation.
  4. `modules` — 7 canonical Kuchipudi learning modules.
  5. `lessons` — Exactly 92 authentic lessons with Shlokas and Telugu translations.
  6. `student_lesson_progress` — Authenticated lesson completion records.
  7. `quizzes` — 3 assignment modes (Theory MCQ, Image ID, Practical Camera).
  8. `questions` — 23 authentic questions with anti-cheat sanitization.
  9. `quiz_results` — Evaluated student examination submissions.
  10. `practice_history` — AI studio session logs with confidence and timestamps.

### B. Backup & Recovery Procedure

#### 1. Instant Safe Backup via Python (Live/Hot Backup)
To create an atomic, crash-consistent backup without stopping the running FastAPI server:
```powershell
& "backend\.venv\Scripts\python.exe" -c "
import sqlite3
src = sqlite3.connect('backend/kuchipudi.db')
dst = sqlite3.connect('backend/kuchipudi_backup.db')
with dst:
    src.backup(dst)
dst.close()
src.close()
print('Backup completed successfully: backend/kuchipudi_backup.db')
"
```

#### 2. Manual File Copy (When server is stopped)
```powershell
copy "backend\kuchipudi.db" "backend\kuchipudi_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%.db"
```

#### 3. Database Restoration
To restore from backup:
```powershell
copy "backend\kuchipudi_backup.db" "backend\kuchipudi.db"
```

---

## 6. AI Model Details & 69 Recognition Classes

### A. Model Specifications
- **Weights Location:** `deployment_clean/nritya_ai_weights.keras`
- **File Size:** 1,378,816 bytes ($\approx 1.31\text{ MB}$)
- **Inference Pipeline:** MediaPipe Holistic 21-hand landmark geometry (175-D normalized vector) $\rightarrow$ Keras Deep MLP Classifier $\rightarrow$ Softmax probability distribution.
- **Inference Latency:** $\approx 44.41\text{ ms}$ on standard CPU (zero GPU required).
- **Anti-Hallucination Rejection:** Synthetic/blank images reject classification with `detected=False` (*"No hands detected in camera frame"*).

### B. Complete 69-Class Taxonomy

#### 1. Asamyukta Hastas (28 Single-Hand Mudras)
1. Pathaka
2. Tripathaka
3. Ardhapathaka
4. Kartarimukha
5. Mayura
6. Ardhachandra
7. Arala
8. Sukatunda
9. Mushti
10. Shikhara
11. Kapittha
12. Katakamukha
13. Soochi
14. Chandrakala
15. Padmakosha
16. Sarpasheersha
17. Mrugasheersha
18. Simhamukha
19. Kaangula
20. Alapadma
21. Chatura
22. Bhramara
23. Hamsasya
24. Hamsapaksha
25. Samdamsha
26. Mukula
27. Tamrachuda
28. Trishoola

#### 2. Samyukta Hastas (24 Combined-Hand Mudras)
1. Anjali
2. Kapothamu
3. Karkatamu
4. Swasthikamu
5. Dola hastamu
6. Pushpaputamu
7. Uthsungamu
8. Sivalingamu
9. Katakavardhanamu
10. Karthariswasthikamu
11. Sakatamu
12. Sankhamu
13. Chakramu
14. Samputamu
15. Paasamu
16. Keelakamu
17. Matsyamu
18. Kurmamu
19. Varahamu
20. Garudamu
21. Nagabandhakamu
22. Khatva
23. Bherundamu
24. Avahithamu

#### 3. Shiro Bhedas (9 Head Movements)
1. Sama
2. Udvahitam
3. Adhomukham
4. Alolitam
5. Dhutam
6. Kampitam
7. Paravrittam
8. Ukshiptam
9. Parivahitam

#### 4. Drishti Bhedas (8 Eye Movements)
1. Sama
2. Alokita
3. Sachi
4. Pralokita
5. Nimilita
6. Ullokita
7. Anuvritta
8. Avalokita

---

## 7. Canonical Curriculum Structure (7 Modules, 92 Lessons)

| Module ID | Module Title (English) | Module Title (Telugu) | Lesson Count | Content Description |
|:---:|:---|:---|:---:|:---|
| **1** | Introduction to Kuchipudi | కూచిపూడి పరిచయం | **4** | Historical origins, Sri Siddhendra Yogi, Natyashastra lineage, spiritual devotion. |
| **2** | Asamyukta Hasthas | అసంయుక్త హస్తాలు (28 ఏకహస్త ముద్రలు) | **28** | All 28 single-hand gestures, authoritative Sanskrit verses, Telugu explanations, and authentic photos. |
| **3** | Samyukta Hasthas | సంయుక్త హస్తాలు (24 ద్విహస్త ముద్రలు) | **24** | All 24 combined gestures, Sanskrit verses, Telugu explanations, and authentic photos. |
| **4** | Shiro Bhedhas | శిరో భేదాలు (9 శిరస్సు కదలికలు) | **9** | 9 head movements, performance rules, Shlokas, and expressive usage. |
| **5** | Drishti Bhedhas | దృష్టి భేదాలు (8 నేత్ర కదలికలు) | **8** | 8 eye glances, directional movements, Shlokas, and emotional conveyances. |
| **6** | Dvadasha Hasta Pranamas | ద్వాదశ హస్త ప్రణామాలు (12 నమస్కార ముద్రలు) | **12** | 12 sacred salutations, Abhinaya fundamentals, and aesthetic postures. |
| **7** | Sapta Talamulu | సప్త తాళములు (7 ప్రధాన తాళాలు మరియు 35 తాళ వ్యవస్థ) | **7** | **Authoritative Match with `Sapta Talamulu.pdf`:** Dhruva, Matya, Rupaka, Jhampe, Triputa, Ata, and Eka Talas with individual creator verses and 35-tala Pancha Jaati chart. |
| **TOTAL** | **7 Canonical Modules** | | **92** | **100% Complete & Verified in SQLite** |

---

## 8. Practice Studio & Examination Functionality

### A. Practice Studio
- **Dual Inference Mode:** Real-time webcam streaming and drag-and-drop image file upload.
- **Visual Feedback:** 21-hand landmark canvas skeleton overlay in real-time.
- **Clear Controls:** "START CAMERA" button styled in Temple Green (`#293C35`) with white text; toggles to "STOP CAMERA" when active.
- **Real-Time Camera Status Badges:** `Camera Standby` (Gray), `Requesting Permission` (Amber), `Camera Active` (Green), `Camera Error` (Crimson).
- **High-Contrast Category Tabs:** Deep Plum (`#482C3A`) background with pure white text (`#FFFFFF`) when selected.
- **Practice Persistence:** Logs confidence score, target posture match, and timestamp to SQLite `practice_history` for authenticated students and to `localStorage` for guests.

### B. Examination Engine
- **Type A (Theory MCQ):** 10 questions testing dance history, tala theory, and Shloka recall.
- **Type B (Image Identification):** 10 questions displaying authentic dance photographs for gesture recognition.
- **Type C (Practical Camera Exam):** 3 performance tasks requiring live posture execution before the camera.
- **Anti-Cheat Delivery:** The API completely strips `correct_option` and explanation fields from student question payloads.
- **Server-Side Scoring:** Answers are evaluated strictly by backend code; score manipulation is rejected.

---

## 9. Automated Audit Results Summary

Every test suite was executed directly against the live system and passed without error:

```
===========================================================================
STAGE 9 AUTOMATED AUDIT RESULTS:
===========================================================================
  [PASS] Stage 4 Curriculum Audit       Duration: 0.10s (92 lessons, 7 modules)
  [PASS] Stage 5 Assignments Audit      Duration: 0.25s (3 quiz types, anti-cheat)
  [PASS] Stage 6 AI Engine Audit        Duration: 2.56s (69 classes, Keras latency 44ms)
  [PASS] Stage 7 Full Platform E2E      Duration: 3.89s (Auth, RBAC, SQLite, Token Revocation)
  [PASS] Stage 9 Final Acceptance Suite Duration: 4.82s (42 individual criteria)
  [PASS] Frontend Production Build      Duration: 2.86s (1585 modules transformed, 0 errors)
===========================================================================
OVERALL STATUS: 100% PASSED (0 FAILURES)
===========================================================================
```

---

## 10. Known Limitations & Future Roadmap

### Known Limitations (Current Version)
1. **Camera Permissions in Restricted Environments:** WebRTC live video access requires explicit browser permission over `http://localhost` or HTTPS. In environments without an active camera or permission, the image upload feature serves as the fallback.
2. **Local Machine Dependency:** The system is currently packaged for local execution on Windows/macOS/Linux. Production cloud deployment requires configuring HTTPS certificates (for camera permissions) and an external reverse proxy (e.g., Nginx or Caddy).

### Recommended Future Roadmap
1. **Audio Tala Metronome:** Add an interactive audio metronome to Module 7 for practicing the Sapta Talas across all 5 Jaatis (Tisra, Chaturasra, Khanda, Misra, Sankeerna).
2. **Sequential Motion / Adavu Tracking:** Extend the temporal BiGRU model from static gesture recognition to full 3D body pose tracking for dynamic Kuchipudi Adavus (footwork steps).
3. **PWA Offline Support:** Package the React frontend with a Service Worker for offline revision of theory lessons and Shlokas.

---

## 11. Final Sign-Off & Project Freeze

This project is formally **FROZEN** in its current working configuration:
- Database: Complete, seeded, and verified at [backend/kuchipudi.db](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/backend/kuchipudi.db).
- AI Weights: Preserved intact at [deployment_clean/nritya_ai_weights.keras](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/deployment_clean/nritya_ai_weights.keras).
- Frontend Build: Verified production build in [frontend/dist](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/frontend/dist).
- Treatises: Sourced from [theory.zip](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/theory.zip) into [docs/pdfs/](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/docs/pdfs/).

*Nritya AI is fully prepared for demonstration.*
