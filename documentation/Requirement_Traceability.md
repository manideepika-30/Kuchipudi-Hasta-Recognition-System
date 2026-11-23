# Requirement Traceability Matrix (RTM)
## Project: Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System

**Status Legend:**
- **Completed & Verified**: Fully implemented, tested across automated test suites, and verified.

---

## 1. Traceability Matrix

| Req ID | Requirement Description | Website Module | Stage | Implementation Status | Test Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FR-DESIGN-01** | Soft classical Indian theme (Blush Ivory, Sage, Lavender, Antique Gold, Deep Plum, Temple Green) | UI / CSS Theme | Stage 2.5 | **Completed & Verified** | TC-UI-01 |
| **FR-HOME-01** | Clean public homepage with dancer hero, short intro, start learning CTA, and about section | Public Homepage | Stage 2.5 | **Completed & Verified** | TC-UI-02 |
| **FR-LANG-01** | Bilingual toggle (English & Telugu) with persistent storage and Sanskrit preservation | Global Nav & LMS | Stage 2.5 | **Completed & Verified** | TC-LANG-01 |
| **FR-LMS-01** | Permanent left sidebar navigation on desktop with slide-out drawer on mobile | Student Dashboard | Stage 2.5 | **Completed & Verified** | TC-LMS-01 |
| **FR-LMS-02** | Dynamic next incomplete lesson detection and automatic Module 1 starting point | Student Dashboard | Stage 2.5 | **Completed & Verified** | TC-LMS-02 |
| **FR-LMS-03** | Interactive lesson completion toggle updating progress bar reactively | Student Dashboard | Stage 2.5 | **Completed & Verified** | TC-LMS-03 |
| **FR-AUTH-01** | Student and Admin registration and login modal interface | Auth Module | Stage 3 | **Completed & Verified** | TC-AUTH-01 |
| **FR-DOC-01** | Software Requirements Specification, Architecture, and Database Design | `documentation/` | Stage 2.5 | **Completed & Verified** | TC-DOC-01 |
| **FR-BACK-01** | FastAPI backend with isolated virtual environment, CORS, and `/docs` | Backend API | Stage 3 | **Completed & Verified** | TC-API-01 (9/9 Passed) |
| **FR-AUTH-02** | Secure bcrypt password hashing, JWT bearer token, and RBAC authorization | Backend Auth | Stage 3 | **Completed & Verified** | TC-AUTH-02 (Bcrypt + HS256) |
| **FR-DB-01** | SQLite database schema creation for users, lessons, quiz scores, and practice history | SQLite Database | Stage 3 | **Completed & Verified** | TC-DB-01 (`kuchipudi.db`) |
| **FR-AI-ARCH-01** | AI health endpoint verifying model weights presence & 69-class architecture for Stage 6 | AI Router | Stage 3 | **Completed & Verified** | TC-AI-ARCH-01 (1.31MB Verified) |
| **FR-CURR-01** | All 7 theory modules with complete English and Telugu lesson content from original PDFs | Learning Modules | Stage 4 | **Completed & Verified** | `test_stage4_curriculum.py` (4/4 Passed) |
| **FR-EXAM-01** | Type A Theory MCQ quiz engine with automatic scoring and explanations | Assignments | Stage 5 | **Completed & Verified** | `test_stage5_assignments.py` (5/5 Passed) |
| **FR-EXAM-02** | Type B Image identification test using authentic dataset gesture photographs | Assignments | Stage 5 | **Completed & Verified** | `test_stage5_assignments.py` (5/5 Passed) |
| **FR-EXAM-03** | Type C Practical camera examination interface | Assignments | Stage 5 | **Completed & Verified** | `test_stage5_assignments.py` (5/5 Passed) |
| **FR-AI-01** | FastAPI AI adapter connecting to `deployment_clean/nritya_ai_weights.keras` | AI Engine | Stage 6 | **Completed & Verified** | `test_stage6_ai.py` (5/5 Passed) |
| **FR-AI-02** | 52-Class Hastha recognition with 175-D MediaPipe landmark extraction | Practice Studio | Stage 6 | **Completed & Verified** | `test_stage6_ai.py` (5/5 Passed) |
| **FR-AI-03** | 9-Class Shiro Bheda recognition with 102-D temporal kinematics and BiGRU | Practice Studio | Stage 6 | **Completed & Verified** | `test_stage6_ai.py` (5/5 Passed) |
| **FR-AI-04** | 8-Class Drishti Bheda recognition with 132-D facial gaze temporal sequence | Practice Studio | Stage 6 | **Completed & Verified** | `test_stage6_ai.py` (5/5 Passed) |
| **FR-TEST-01** | Complete end-to-end frontend, backend, security, and AI validation suite | System QA | Stage 7 | **Completed & Verified** | `test_stage7_e2e_system.py` (6/6 Passed) |
| **FR-DEP-01** | GitHub repository preparation with `.gitignore` and deployment guide | Deployment | Stage 8 | **In Progress (Stage 8)** | README.md / DEPLOYMENT.md |

---

## 2. Stage Verification Summary

- **Stage 1 (Codebase Inspection)**: Audited existing AI models (`deployment_clean/nritya_ai_weights.keras`), datasets, and 69-class taxonomy. Verified and preserved.
- **Stage 2 & 2.5 (Frontend Architecture & Classical Redesign)**: Implemented React 18 + Vite frontend with soft classical Indian palette, responsive drawer navigation, and bilingual English/Telugu toggle.
- **Stage 3 (FastAPI Backend, SQLite & Auth)**: FastAPI backend with SQLAlchemy SQLite models, JWT token creation with `jti` anti-collision, token revocation blocklist, and student/admin RBAC.
- **Stage 4 (Authentic Curriculum Modules)**: Sourced all 7 classical modules and 92 lessons directly from extracted PDF theory documents (`theory.zip`). Included 61 authentic photographic mudra plates, Sanskrit shlokas, and Telugu translations.
- **Stage 5 (Assignments & Anti-Cheat Quizzes)**: Implemented all 3 assignment types (Theory MCQ, Image ID, Practical Camera). Validated anti-cheat question masking (`correct_option` stripped prior to transmission) and server-side grade calculation.
- **Stage 6 (AI Inference Engine & Practice Studio)**: Successfully mounted `nritya_ai_weights.keras` (1.31 MB) into FastAPI via singleton `KuchipudiAIInferenceEngine`. Achieved ~118-135ms forward passes without modifying weights or introducing synthetic/fake predictions. Integrated webcam landmark overlay and practice history persistence.
- **Stage 7 (Full Platform System Testing)**: Created comprehensive end-to-end test suite `backend/scripts/test_stage7_e2e_system.py`. Validated all 6 platform subsystems with 100% test pass rate.
