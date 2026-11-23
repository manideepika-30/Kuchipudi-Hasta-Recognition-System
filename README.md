# Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_0.165-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_18_Vite-61DAFB.svg?style=flat&logo=react)](https://react.dev)
[![TensorFlow](https://img.shields.io/badge/AI_Engine-TensorFlow_2.17_Keras_3-FF6F00.svg?style=flat&logo=tensorflow)](https://tensorflow.org)
[![SQLite](https://img.shields.io/badge/Database-SQLite_3-003B57.svg?style=flat&logo=sqlite)](https://sqlite.org)
[![Bilingual](https://img.shields.io/badge/Language-English_%26_Telugu-8E24AA.svg)](#bilingual-interface)
[![Audits](https://img.shields.io/badge/Acceptance_Audits-100%25_PASS-success.svg)](#automated-test-suites)

> **Nritya AI** is an authentic, AI-powered learning, recognition, and analysis platform dedicated to the traditional Indian classical dance form of **Kuchipudi**. The system features a public learning portal with a permanent sidebar, 7 structured curriculum modules containing 92 authentic lessons with canonical Sanskrit Shlokas and Telugu meanings, 3 comprehensive examination types, and real-time computer vision inference across **69 recognition classes** using pre-trained neural network weights.

---

## 🏛️ Platform Highlights

- **Public Learning Portal**: Direct open access without forced authentication or registration modals.
- **Permanent Left Sidebar**: Permanent desktop navigation (with mobile off-canvas drawer) providing direct access to all modules, Practice Studio, and Quizzes.
- **Authentic Kuchipudi Curriculum (7 Modules, 92 Lessons)**:
  - Preserved traditional Shlokas, Telugu meanings, and performance guidance.
  - Sourced directly from authentic dance treatises in `docs/pdfs/` (including `Sapta Talamulu.pdf` by Anandamayi Nrityalaya).
- **Real-Time 69-Class AI Practice Studio**:
  - **52 Hasthas** (28 Asamyukta single-hand mudras + 24 Samyukta combined-hand mudras) via MediaPipe 21-hand landmark geometry and deep MLP.
  - **9 Shiro Bhedas** (head movements) via kinematic temporal sequence recognition.
  - **8 Drishti Bhedas** (eye glances) via facial gaze analysis.
  - **Zero Fake Predictions**: Genuine Keras forward-pass inference (~44ms on CPU). Blank/non-dance frames are rejected with `detected=False`.
- **Examination & Assessment Engine**:
  - **Type A**: Theory MCQ Examination (10 questions).
  - **Type B**: Image Identification Test (10 authentic visual plate questions).
  - **Type C**: Practical Camera Performance Examination (3 real-time posture tasks).
  - **Anti-Cheat Hardened**: Correct options are permanently stripped before question delivery; scoring is strictly evaluated server-side.
- **Bilingual Experience**: Instant toggle between English and Telugu (`తెలుగు`).
- **Preserved Security Stack**: Bcrypt password hashing, HS256 JWT tokens with database revocation on logout, and role-based access control (RBAC).

---

## 🚀 Quickstart Guide

### Option 1: One-Click Startup (Windows)

Double-click the launcher script in the project root:
```cmd
start_all.bat
```
This script launches:
1. **Backend**: FastAPI server on `http://127.0.0.1:9000`
2. **Frontend**: Vite server on `http://localhost:3000`

Open your web browser at: **`http://localhost:3000`**

---

### Option 2: Manual Terminal Startup

#### 1. Start the FastAPI Backend (Port 9000)

```powershell
# Navigate to backend directory
cd backend

# Activate Python 3.11 virtual environment
.\.venv\Scripts\Activate.ps1

# Launch uvicorn on port 9000
python -m uvicorn app.main:app --host 127.0.0.1 --port 9000 --reload
```
- API Base URL: `http://127.0.0.1:9000/api`
- Interactive OpenAPI / Swagger Docs: `http://127.0.0.1:9000/docs`

> **Note on Port 9000:** The backend uses port **9000** to avoid socket permission errors on ports 8000/8080 under Windows.

#### 2. Start the React Frontend (Port 3000)

```powershell
# Open a new terminal and navigate to frontend
cd frontend

# Launch Vite development server
npm run dev
```
- Web Application: `http://localhost:3000`

---

## 🗂️ Project Directory Structure

```
kuchipudi/
├── backend/                             # Python FastAPI Backend
│   ├── app/                             # FastAPI application code
│   │   ├── database/                    # SQLAlchemy engine & seed data
│   │   ├── models/                      # User, Curriculum, Quiz models
│   │   ├── routes/                      # API endpoint routers
│   │   ├── schemas/                     # Pydantic request/response schemas
│   │   ├── services/                    # AI inference & Auth services
│   │   └── main.py                      # FastAPI app entry point
│   ├── scripts/                         # Audit and administration scripts
│   ├── kuchipudi.db                     # Authoritative SQLite database
│   ├── requirements.txt                 # Backend Python dependencies
│   └── .venv/                           # Python 3.11 virtual environment
├── frontend/                            # React 18 + Vite Frontend
│   ├── public/assets/                   # Authentic dance illustrations
│   ├── src/                             # React components, pages, services
│   ├── package.json                     # Frontend Node dependencies
│   └── vite.config.js                   # Vite config (proxies /api to :9000)
├── deployment_clean/                    # Pre-trained AI Core
│   └── nritya_ai_weights.keras          # Canonical Keras weights (1.31 MB)
├── Data set/                            # Authoritative Original Image Datasets
│   ├── Asamyuktha hasthalu/             # 28 Single-hand classes
│   └── Samyuktha hasthalu/              # 24 Combined-hand classes
├── docs/pdfs/                           # Authoritative Dance Treatises
│   ├── Sapta Talamulu.pdf               # Rhythm system treatise (Anandamayi Nrityalaya)
│   └── Asamyutha Hasthas.pdf            # Authoritative Mudra texts
├── documentation/                       # Architecture & Engineering Reports
│   └── reports/                         # Stage audit & handover reports
├── start_all.bat                        # Launcher for Backend + Frontend
├── start_backend.bat                    # Launcher for Backend on :9000
├── start_frontend.bat                   # Launcher for Frontend on :3000
└── README.md                            # This documentation
```

---

## 📚 Canonical Curriculum (7 Modules, 92 Lessons)

| Module | Title (English / Telugu) | Lessons | Key Content |
|:---:|:---|:---:|:---|
| **1** | Introduction to Kuchipudi / కూచిపూడి పరిచయం | 4 | Origins, Sri Siddhendra Yogi, Natyashastra lineage |
| **2** | Asamyukta Hasthas / అసంయుక్త హస్తాలు | 28 | All 28 single-hand mudras with Shlokas and photos |
| **3** | Samyukta Hasthas / సంయుక్త హస్తాలు | 24 | All 24 double-hand mudras with Shlokas and photos |
| **4** | Shiro Bhedhas / శిరో భేదాలు | 9 | 9 head movements with performance rules and verses |
| **5** | Drishti Bhedhas / దృష్టి భేదాలు | 8 | 8 eye movements with directional guidance |
| **6** | Dvadasha Hasta Pranamas / ద్వాదశ హస్త ప్రణామాలు | 12 | 12 sacred salutations and Abhinaya fundamentals |
| **7** | Sapta Talamulu / సప్త తాళములు | 7 | Individual creator Shlokas and 35-tala Pancha Jaati system |
| **TOTAL** | **7 Canonical Modules** | **92** | **100% Complete & Verified** |

---

## 🤖 AI Model & Supported Classes

- **Model File:** `deployment_clean/nritya_ai_weights.keras` (1.31 MB)
- **Supported Classes (69 Total):**
  - **Asamyukta Hastas (28):** Pathaka, Tripathaka, Ardhapathaka, Kartarimukha, Mayura, Ardhachandra, Arala, Sukatunda, Mushti, Shikhara, Kapittha, Katakamukha, Soochi, Chandrakala, Padmakosha, Sarpasheersha, Mrugasheersha, Simhamukha, Kaangula, Alapadma, Chatura, Bhramara, Hamsasya, Hamsapaksha, Samdamsha, Mukula, Tamrachuda, Trishoola.
  - **Samyukta Hastas (24):** Anjali, Kapothamu, Karkatamu, Swasthikamu, Dola, Pushpaputamu, Uthsungamu, Sivalingamu, Katakavardhanamu, Karthariswasthikamu, Sakatamu, Sankhamu, Chakramu, Samputamu, Paasamu, Keelakamu, Matsyamu, Kurmamu, Varahamu, Garudamu, Nagabandhakamu, Khatva, Bherundamu, Avahithamu.
  - **Shiro Bhedas (9):** Sama, Udvahitam, Adhomukham, Alolitam, Dhutam, Kampitam, Paravrittam, Ukshiptam, Parivahitam.
  - **Drishti Bhedas (8):** Sama, Alokita, Sachi, Pralokita, Nimilita, Ullokita, Anuvritta, Avalokita.
- **Inference Speed:** ~44ms per frame on CPU.
- **Anti-Hallucination:** Blank/non-hand frames return `detected=False`.

---

## 💾 Database & Backup Procedure

- **Database File:** `backend/kuchipudi.db`

### Safe Hot Backup Command (Python):
```powershell
& "backend\.venv\Scripts\python.exe" -c "
import sqlite3
src = sqlite3.connect('backend/kuchipudi.db')
dst = sqlite3.connect('backend/kuchipudi_backup.db')
with dst:
    src.backup(dst)
dst.close()
src.close()
print('Backup created: backend/kuchipudi_backup.db')
"
```

---

## 🧪 Automated Test Suites

Run the comprehensive acceptance test suite:
```powershell
& "backend\.venv\Scripts\python.exe" "backend\scripts\run_stage9_acceptance_tests.py"
```

Or run all stage audits:
```powershell
& "backend\.venv\Scripts\python.exe" "backend\scripts\run_all_audits.py"
```

Verify frontend production build:
```powershell
cd frontend
npm run build
```

---

## 🔒 Security Configuration

- **JWT Revocation:** Calling `/api/auth/logout` invalidates tokens via the SQLite `revoked_tokens` table.
- **RBAC Boundaries:** Public visitors and student accounts cannot access `/api/admin/*` (`HTTP 403 / 401`).
- **Anti-Cheat Delivery:** The question delivery endpoint strips `correct_option` and explanations from student payloads.

---

## 📄 Documentation Links

- [Final Project Handover Report](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/documentation/reports/FINAL_PROJECT_HANDOVER.md)
- [Stage 9 Final Acceptance Report](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/documentation/reports/STAGE_9_FINAL_ACCEPTANCE_REPORT.md)
- [UI & Sapta Tala Correction Report](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/documentation/reports/UI_AND_SAPTA_TALA_CORRECTION_REPORT.md)
- [System Architecture](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/documentation/Architecture.md)
- [Software Requirements Specification (SRS)](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/documentation/SRS.md)
