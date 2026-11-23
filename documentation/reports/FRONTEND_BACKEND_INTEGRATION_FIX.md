# Nritya AI – Frontend/Backend Integration and Port 9000 Synchronization Report

**Official Project Title**: *Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System*  
**Date**: November 23, 2026  
**Status**: Resolved, Synchronized & Live Verified  

---

## 1. Executive Summary

This report documents the resolution of the API communication issues and the complete synchronization of the development configuration with **port 9000** for the **Nritya AI** backend on Windows.

On this Windows host, default ports 8000 and 8080 encountered Windows socket permission and conflict errors. Port **9000** (`http://127.0.0.1:9000`) was verified as the operating port. All frontend services, Vite reverse proxy rules, environment files, and Windows launch scripts have been synchronized to use port 9000.

---

## 2. Configuration & Port Synchronization Details

### A. Development Port Mapping
| Component | Runtime | Host & Port | Role / Path |
|---|---|---|---|
| **FastAPI Backend** | Python 3.11 / Uvicorn | `http://127.0.0.1:9000` | SQLite DB, Keras Inference, JWT Auth (`/api/*`) |
| **React Frontend** | Vite / React 18 | `http://127.0.0.1:3000` | User Interface, Student LMS, Practice Studio |
| **Vite Dev Proxy** | Vite Server | `http://127.0.0.1:3000/api` | Reverse proxies all `/api/*` traffic to `http://127.0.0.1:9000` |

### B. Updated Configuration Files
1. **Vite Configuration** ([frontend/vite.config.js](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/frontend/vite.config.js)):
   ```javascript
   export default defineConfig({
     plugins: [react()],
     server: {
       port: 3000,
       open: false,
       proxy: {
         '/api': {
           target: 'http://127.0.0.1:9000',
           changeOrigin: true,
           secure: false,
         }
       }
     }
   })
   ```
2. **API Service** ([frontend/src/services/api.js](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/frontend/src/services/api.js)):
   - Configured `API_BASE` to automatically route relative `/api` calls through the Vite proxy in development.
   - Fallback target updated to `http://127.0.0.1:9000/api`.
   - Error messages updated to specify port 9000: `"Please ensure the FastAPI server is running on http://127.0.0.1:9000."`
3. **Frontend Environment** ([frontend/.env](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/frontend/.env)):
   ```env
   VITE_API_URL=/api
   ```
4. **Backend Environment** ([backend/.env](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/backend/.env)):
   ```env
   CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:9000,http://127.0.0.1:9000
   ```
5. **Windows Launch Batch Scripts**:
   - [start_backend.bat](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/start_backend.bat):
     ```cmd
     call .venv\Scripts\activate.bat
     python -m uvicorn app.main:app --host 127.0.0.1 --port 9000 --reload
     ```
   - [start_frontend.bat](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/start_frontend.bat):
     ```cmd
     call npm run dev
     ```
   - [start_all.bat](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/start_all.bat):
     Launches both servers in separate console windows.

---

## 3. End-to-End Live Verification on Port 9000

The authentication lifecycle was tested and verified both **directly against FastAPI on port 9000** and **through the Vite proxy on port 3000**:

```
=================================================================
TESTING AUTH FLOW VIA: DIRECT BACKEND (PORT 9000)
Base URL: http://127.0.0.1:9000/api
=================================================================
  [1/5] REGISTER: OK (HTTP 201)
        Email: scholar_1795421946_dir@kuchipudi.edu
        Full Name: Priyamvada Natya
        Language: te
  [2/5] LOGIN: OK (HTTP 200)
        Access Token: eyJhbGciOiJIUzI1NiIsInR5c...
        User Profile Name: Priyamvada Natya
  [3/5] GET /users/profile: OK (HTTP 200)
        Student Profile Name: Priyamvada Natya
        Language Preference: te
  [4/5] GET /auth/me: OK (HTTP 200)
        Identity Verified: scholar_1795421946_dir@kuchipudi.edu
        Role: student
  [5/5] POST /auth/logout: OK (HTTP 200)
        Result: Successfully logged out and token invalidated
  [VERIFIED] Revoked token rejected with HTTP 401 Unauthorized!

=================================================================
TESTING AUTH FLOW VIA: VITE REVERSE PROXY (PORT 3000)
Base URL: http://127.0.0.1:3000/api
=================================================================
  [1/5] REGISTER: OK (HTTP 201)
        Email: scholar_1795421947_vit@kuchipudi.edu
        Full Name: Priyamvada Natya
        Language: te
  [2/5] LOGIN: OK (HTTP 200)
        Access Token: eyJhbGciOiJIUzI1NiIsInR5c...
        User Profile Name: Priyamvada Natya
  [3/5] GET /users/profile: OK (HTTP 200)
        Student Profile Name: Priyamvada Natya
        Language Preference: te
  [4/5] GET /auth/me: OK (HTTP 200)
        Identity Verified: scholar_1795421947_vit@kuchipudi.edu
        Role: student
  [5/5] POST /auth/logout: OK (HTTP 200)
        Result: Successfully logged out and token invalidated
  [VERIFIED] Revoked token rejected with HTTP 401 Unauthorized!

ALL 10 AUTHENTICATION AND TOKEN REVOCATION CHECKS PASSED PERFECTLY!
```

---

## 4. Full Automated System Audit Suite (`scripts/run_all_audits.py`)

All 4 system audit suites passed with 100% success rate:

```
===========================================================================
STAGE 9 AUDIT RESULTS SUMMARY:
===========================================================================
  [PASS] Stage 4 Curriculum             Duration: 0.10s (7 Modules, 92 Lessons, 65 Images)
  [PASS] Stage 5 Assignments            Duration: 0.25s (3 Quizzes, Anti-Cheat, Grading)
  [PASS] Stage 6 AI Engine              Duration: 2.76s (69 Classes, Keras Inference, Blank Check)
  [PASS] Stage 7 Full E2E               Duration: 3.80s (Auth, RBAC, Progress, Tamper Protection)
===========================================================================
ALL AUDIT SUITES PASSED (100% SUCCESS RATE)!
===========================================================================
```

---

## 5. Frontend Production Build Check

```
> nritya-ai-kuchipudi@1.0.0 build
> vite build

vite v6.4.6 building for production...
transforming...
✓ 1583 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   1.55 kB │ gzip:  0.83 kB
dist/assets/index-BaiFMY_L.css    5.01 kB │ gzip:  1.53 kB
dist/assets/index-CbF5Y7RX.js   307.93 kB │ gzip: 81.97 kB
✓ built in 4.87s (0 errors)
```

---

## 6. How to Run the Platform

### Option A: One-Click Startup (Recommended)
Double-click [start_all.bat](file:///e:/pragathi/My%20kuchipudi%20project%20aidl/kuchipudi/start_all.bat) in the project root. This opens both the backend on port 9000 and the frontend on port 3000 in separate console windows.

### Option B: Manual Terminal Startup
1. **Backend**:
   ```powershell
   cd "e:\pragathi\My kuchipudi project aidl\kuchipudi\backend"
   .\.venv\Scripts\activate
   python -m uvicorn app.main:app --host 127.0.0.1 --port 9000 --reload
   ```
2. **Frontend**:
   ```powershell
   cd "e:\pragathi\My kuchipudi project aidl\kuchipudi\frontend"
   npm run dev
   ```
3. Open your browser at: **`http://localhost:3000`** (or `http://127.0.0.1:3000`).

---

## 7. Strict Preservation of Project Assets
- **Trained AI Model**: `deployment_clean/nritya_ai_weights.keras` preserved untouched (1.31 MB, 69 classes).
- **Curriculum & PDFs**: All 92 canonical lessons and original PDFs in `docs/pdfs/` preserved intact.
- **SQLite Database**: `backend/kuchipudi.db` maintained with all seeded lessons, quizzes, and student profiles.
- **Git & Online State**: No remote pushes or unapproved deployment actions performed.
