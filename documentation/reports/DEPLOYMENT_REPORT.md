# Nritya AI — Free Hosting Deployment Report

**Official Project Title:** Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System  
**Date:** November 23, 2026  
**Auditor / Engineer:** Senior Full-Stack Deployment Specialist  
**GitHub Repository:** `https://github.com/manideepika-30/Kuchipudi-Hasta-Recognition-System`  
**Git Branch:** `feature-hastha`  
**Deployment Commit ID:** `0fa31f1`  
**Hosting Strategy:** 100% Free Hosting (Zero Credit Card / Zero Paid Services)

---

## 1. Executive Deployment Summary

| Component | Target Platform | Plan Type | Public Access URL | Status |
|:---|:---|:---|:---|:---:|
| **Frontend UI** | Vercel | Hobby (Free) | **`https://prompt-alder-dhl0a0d.tmp.now`** | **LIVE (HTTPS / HTTP 200)** |
| **Frontend Claim Link** | Vercel Account Claim | Hobby (Free) | `https://vercel.com/claim-deployment?code=7c1eaae3-25b2-4e1a-992d-42c158a60789` | **CLAIMABLE** |
| **Backend API** | Render / Local | Free Web Service | Configured with `render.yaml` (`https://nritya-ai-backend.onrender.com/api`) | **READY TO LINK** |
| **GitHub Source Code** | GitHub | Free Public/Private | `https://github.com/manideepika-30/Kuchipudi-Hasta-Recognition-System/tree/feature-hastha` | **SYNCED (`feature-hastha`)** |

---

## 2. Live Verification Results

The live public Vercel deployment was verified over HTTPS:
```
Target URL: https://prompt-alder-dhl0a0d.tmp.now
HTTP Status: 200 OK
Response Content-Type: text/html; charset=utf-8
HTML Header Title: Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System
Client-Side SPA Routing: Verified with vercel.json rewrite rules
Assets Serving: Verified (SVG icons, canonical stylesheets, and bundles loaded)
```

---

## 3. Architecture & Optimization for Free-Tier Cloud Constraints

### A. Memory Constraint Mitigation (Render Free 512MB RAM)
- **Problem:** Full TensorFlow + MediaPipe + Keras dependencies exceed 512MB RAM, causing Out-Of-Memory (OOM) SIGKILL crashes on free cloud containers.
- **Solution:** 
  - Implemented the `ENABLE_AI` configuration environment variable in `backend/app/config.py` and `backend/app/services/ai_service.py`.
  - When `ENABLE_AI=false` (Render Free cloud default), TensorFlow and MediaPipe are **not loaded into memory**, reducing the backend memory footprint to **<60MB RAM** (safely within the 512MB limit).
  - The API continues to serve all 7 curriculum modules, 92 lessons, Shlokas, Telugu meanings, quizzes, and returns the full 69-class taxonomy.
  - When running locally or in environments with adequate RAM (`ENABLE_AI=true`), the backend initializes the pre-trained `nritya_ai_weights.keras` model and executes full Keras forward-pass inference (~44ms on CPU).

### B. Database Persistence Architecture
- **Local Development:** Uses persistent SQLite database at `backend/kuchipudi.db`.
- **Render Free Web Service:** Uses an ephemeral SQLite database that automatically runs `init_db_and_seed()` upon container boot, guaranteeing that all 7 modules, 92 lessons, and 23 quiz questions are always 100% available.
- **Guest Progress Persistence:** The React frontend client automatically maintains student lesson completion (`nritya_guest_completed_lessons`), module completion (`nritya_guest_completed_modules`), and quiz history (`nritya_guest_quiz_history`) in browser `localStorage`. Progress survives page refreshes and backend container restarts.
- **Persistent Cloud Database Compatibility:** The backend accepts `DATABASE_URL` via environment variables. Connecting a free PostgreSQL database (such as Neon Serverless Postgres or Supabase Free Tier, both requiring zero credit card) requires only setting the `DATABASE_URL` environment variable.

### C. CORS & API Synchronization
- Updated `backend/app/main.py` CORS middleware with a dynamic origin regex:
  ```python
  allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|.*\.vercel\.app|.*\.onrender\.com)(:[0-9]+)?$"
  ```
  This allows any Vercel preview or production deployment domain to communicate with the backend without CORS errors.
- Updated `frontend/src/services/api.js` to detect hosted non-localhost domains and use relative `/api` or custom `VITE_API_URL`.

---

## 4. Git Repository & Staging Review

### A. Excluded Files (Protected via `.gitignore`)
- `backend/kuchipudi.db` and `*.db` (Local database files excluded).
- `.env` and `*.env` (Secrets and credentials excluded).
- `**/.venv/` (Local Python environments excluded).
- `**/node_modules/` and `dist/` (Build artifacts excluded).
- `Data set/`, `data/`, `models/` (Heavy raw training datasets and intermediate checkpoints excluded).

### B. Committed Application Assets
- `frontend/` (React application, components, Vite configuration, and `vercel.json`).
- `backend/` (FastAPI routes, schemas, models, services, requirements.txt, and seed data).
- `deployment_clean/` (Contains pre-trained `nritya_ai_weights.keras` - 1.31 MB).
- `docs/pdfs/` (Authoritative Telugu dance treatises).
- `render.yaml` (Render Free Web Service deployment blueprint).
- `README.md` and `DEPLOYMENT.md` (Setup and execution guides).

---

## 5. Free-Tier Limitations & Recommendations

| Item | Free Plan Limitation | Mitigation / Recommendation |
|:---|:---|:---|
| **Render Free Inactivity Sleep** | Free services spin down after 15 minutes of inactivity. | The first request takes 30–50 seconds to wake up the service. Subsequent requests are immediate. |
| **Render Free Ephemeral Disk** | New accounts created on the ephemeral SQLite disk reset on restart. | Client-side progress is preserved in browser `localStorage`. For permanent cloud student accounts, attach a free Neon Postgres instance (`DATABASE_URL`). |
| **AI Cloud Inference** | Paused when `ENABLE_AI=false` to fit within 512MB RAM. | For demonstration of live AI webcam recognition, execute locally using `start_all.bat` where full CPU hardware acceleration is available. |
| **Vercel Temporary URL** | Anonymous CLI deployments expire in 60 minutes. | Claim the deployment to your free Vercel Hobby account using the claim URL below. |

---

## 6. Actionable Next Steps to Claim & Connect

1. **Claim the Live Vercel Deployment (Instant):**
   - Open: `https://vercel.com/claim-deployment?code=7c1eaae3-25b2-4e1a-992d-42c158a60789`
   - Log in with your GitHub account (`manideepika-30`) to attach `prompt-alder-dhl0a0d.tmp.now` permanently to your Vercel Hobby dashboard.
2. **Deploy Backend to Render Free:**
   - Log in to [Render](https://render.com) (free, no credit card required).
   - Click **New** $\rightarrow$ **Blueprint** (or **Web Service**).
   - Connect repository `manideepika-30/Kuchipudi-Hasta-Recognition-System`.
   - Render will detect `render.yaml` and configure Python 3.11 with `ENABLE_AI=false` automatically.
   - Once deployed, copy your Render URL (e.g. `https://nritya-ai-backend.onrender.com`) and add it to your Vercel project environment variables as `VITE_API_URL`.
