# Software Requirement Specification (SRS)
## Project: Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System

**Document Version:** 1.0  
**Date:** November 2026  
**Project Category:** SEDS AI/ML & Full-Stack Web Application  

---

## 1. Project Overview
**Nritya AI** is an authentic educational and AI-assisted practice web platform designed for classical Indian dance enthusiasts, students, and gurus, specifically supporting the classical dance form of **Kuchipudi**.

---

## 2. Problem Statement
Traditional Indian classical dance training relies heavily on direct guru-shishya physical presence. Students practicing at home frequently lack:
1. **Immediate Feedback**: Inability to verify whether hand joint angles, head inclinations, and ocular gaze directions are accurate.
2. **Accessible Authentic Theory**: Theoretical texts (*Natya Shastra*, *Abhinaya Darpana*) and Tala mathematics are often scattered or inaccessible in bilingual formats.
3. **Structured Self-Paced Progression**: Absence of personalized LMS tracking identifying completed lessons, pending assignments, and practice session logs.

---

## 3. Project Objectives
1. **Preserve Cultural Heritage**: Digitize and present the classical Kuchipudi curriculum across seven comprehensive modules based strictly on original authoritative documents without inventing or altering traditional meanings.
2. **Empower Remote Practice**: Deliver real-time, zero-fake AI evaluation directly through standard webcams utilizing trained TensorFlow/Keras neural models and MediaPipe landmarks.
3. **Support Bilingual Pedagogy**: Provide seamless, instantaneous language toggling between English and Telugu (తెలుగు) across all interface labels, instructions, and theoretical explanations.
4. **Track Student Progress**: Implement a robust SQLite and FastAPI backend with role-based authentication, student progress management, and assignment evaluation.

---

## 4. Scope
- **In Scope**:
  - Bilingual responsive web application (React + Vite).
  - Seven structured theory modules: Introduction, Asamyukta Hasthas (28), Samyukta Hasthas (24), Shiro Bhedhas (9), Drishti Bhedhas (8), Dvadasha Hasta Pranamas (12), and Sapta Talamulu (35 systems).
  - Student and Admin authentication with JWT and bcrypt password hashing.
  - Interactive student dashboard with permanent left sidebar and progress tracking.
  - Three assignment modes: Theory MCQ Quiz, Image Identification, and Practical Camera Examination.
  - Integration of existing 69-class AI inference pipeline:
    - 52 Hasta classes (175-D MediaPipe landmark representation)
    - 9 Shiro Bheda classes (102-D temporal motion representation, 160-frame buffer)
    - 8 Drishti Bheda classes (132-D facial gaze temporal motion representation)
- **Out of Scope (Current Release)**:
  - Mobile native application compilation (iOS/Android APKs).
  - Real-time Pada Karma (footwork) camera recognition (preserved in training datasets, but excluded from live inference until full body camera setups are standard).

---

## 5. Intended Users
1. **Classical Dance Students**: Learners studying Kuchipudi theory, memorizing mudras, and practicing postures in front of their personal computer camera.
2. **Dance Instructors & Gurus**: Educators viewing student progress, reviewing assignment scores, and managing theoretical curricula.
3. **System Administrators**: Technical managers maintaining users, database integrity, and system health.

---

## 6. Functional Requirements Summary
- **FR-1**: User registration and secure authentication with role separation (Student vs. Admin).
- **FR-2**: Bilingual language switching (English / Telugu) without session disruption or navigation loss.
- **FR-3**: Left permanent sidebar LMS navigation with dynamic next-incomplete-lesson identification.
- **FR-4**: Seven structured theory modules with lesson completion tracking.
- **FR-5**: Practice studio camera feed ingestion and real-time inference via FastAPI.
- **FR-6**: Three assignment evaluation modes with database score persistence.
- **FR-7**: Student and Admin dashboard metrics and history retrieval.

---

## 7. Non-Functional Requirements
- **Performance & Latency**: AI inference latency per frame $\le 50\text{ ms}$ on standard CPU environments; web initial page load time $< 2\text{ seconds}$.
- **Security**: Passwords hashed using bcrypt; API endpoints protected via JWT tokens; student records isolated to prevent unauthorized access.
- **Reliability & Authenticity**: Zero fake AI predictions; true confidence metrics displayed only when validated by consensus logic.
- **Accessibility & Usability**: High text contrast compliant with WCAG AA using Deep Plum (`#482C3A`), Temple Green (`#293C35`), and Blush Ivory (`#F4E8E2`); fully responsive on mobile, tablet, and desktop.
- **Portability**: Modular FastAPI backend and static React build deployable on modern cloud environments or offline local workstations.

---

## 8. Hardware & Software Requirements

### Hardware Requirements
- **Client**: Any laptop or desktop with modern web browser and integrated/external USB webcam (720p at 30 fps recommended).
- **Server**: 2.0 GHz quad-core x86-64 CPU, 4 GB RAM minimum (8 GB recommended for simultaneous TensorFlow inference).

### Software Requirements
- **Frontend**: React 18, Vite 6, Lucide React, HTML5 Canvas & MediaDevices API.
- **Backend**: Python 3.11, FastAPI 0.115+, Uvicorn 0.54+, SQLAlchemy 2.1+, Pydantic 2.14+, Bcrypt 5.0+, Python-Jose 3.5+.
- **AI / ML Runtime**: TensorFlow 2.17.0, Keras 3.15.1, MediaPipe 0.10.14, OpenCV 4.11.0, NumPy 1.26.4.
- **Database**: SQLite 3 (`backend/kuchipudi.db` with full relational schema).

---

## 8.1 Implementation Status: Stages 1 Through 8 Completed & Verified
- **Stage 1 (Codebase Inspection)**: Audited existing AI models (`deployment_clean/nritya_ai_weights.keras`), datasets, and 69-class taxonomy. Verified and preserved.
- **Stage 2 & 2.5 (Frontend Architecture & Classical Redesign)**: Implemented React 18 + Vite frontend with soft classical Indian palette, responsive drawer navigation, and bilingual English/Telugu toggle.
- **Stage 3 (FastAPI Backend, SQLite & Auth)**: FastAPI backend with SQLAlchemy SQLite models, JWT token creation with `jti` anti-collision, token revocation blocklist, and student/admin RBAC.
- **Stage 4 (Authentic Curriculum Modules)**: Sourced all 7 classical modules and 92 lessons directly from extracted PDF theory documents (`theory.zip`). Included 61 authentic photographic mudra plates, Sanskrit shlokas, and Telugu translations.
- **Stage 5 (Assignments & Anti-Cheat Quizzes)**: Implemented all 3 assignment types (Theory MCQ, Image ID, Practical Camera). Validated anti-cheat question masking (`correct_option` stripped prior to transmission) and server-side grade calculation.
- **Stage 6 (AI Inference Engine & Practice Studio)**: Successfully mounted `nritya_ai_weights.keras` (1.31 MB) into FastAPI via singleton `KuchipudiAIInferenceEngine`. Achieved ~118-135ms forward passes without modifying weights or introducing synthetic/fake predictions. Integrated webcam landmark overlay and practice history persistence.
- **Stage 7 (Full Platform System Testing)**: Created comprehensive end-to-end test suite `backend/scripts/test_stage7_e2e_system.py`. Validated all 6 platform subsystems with 100% test pass rate.
- **Stage 8 (GitHub & Deployment Preparation)**: Configured comprehensive `.gitignore`, `.env.example`, beginner-friendly `README.md`, and production `DEPLOYMENT.md`. Kept repository ready for user approval before remote operations.

---

## 9. Constraints and Assumptions
- Camera access requires user permission and an HTTPS context in production.
- Model inference runs backend-side to utilize optimized TensorFlow C++ binaries and native MediaPipe libraries.
- Classical terminology and mudra names remain in canonical Sanskrit transliteration alongside localized Telugu and English descriptions.
