# Functional Requirements Specification
## Project: Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System

---

## 1. User Registration & Authentication (FR-AUTH)

### FR-AUTH-01: Student Registration
- **Description**: Students must be able to register with Full Name, Email Address, Password, and Preferred Language (English / Telugu).
- **Validation**: Email uniqueness check, password minimum length $\ge 8$ characters. Passwords must be hashed using bcrypt prior to database storage.
- **Output**: JSON Web Token (JWT) and student profile record creation.

### FR-AUTH-02: User Login & Session Handling
- **Description**: Registered users (Students & Admins) must authenticate using email and password.
- **Output**: Return access token with embedded user ID, role claim (`student` or `admin`), and expiration time.
- **Rule**: Plain-text passwords must never be stored, logged, or returned by any API.

### FR-AUTH-03: Role-Based Authorization
- **Description**: API endpoints and views must enforce role checks:
  - Students can only access their own profile, progress, quiz submissions, and practice logs.
  - Admins can view aggregate student rosters, modify quiz questions, and oversee system metrics. Public admin registration is strictly disallowed.

---

## 2. Bilingual Language Support (FR-LANG)

### FR-LANG-01: Dual Language Selection
- **Description**: The platform must provide a visible language toggle between **English** and **తెలుగు (Telugu)** accessible from the global navigation bar and student portal.
- **Scope of Translation**:
  - All navigation buttons, action CTAs, instructional text, error messages, and module descriptions.
  - Sanskrit technical names (e.g., *Pathaka*, *Alapadma*, *Anjali*, *Sama*) must be preserved in their authentic classical form in both languages.
- **Persistence**: User selection must be persisted in browser `localStorage` and synchronized with the user profile upon login.

---

## 3. Learning Management System & Left Sidebar (FR-LMS)

### FR-LMS-01: Permanent Left Sidebar
- **Description**: On desktop resolutions ($\ge 900\text{px}$), the student dashboard must feature a fixed left navigation sidebar containing:
  1. Dashboard Overview
  2. Module 1: Introduction to Kuchipudi
  3. Module 2: Asamyukta Hasthas (28 Gestures)
  4. Module 3: Samyukta Hasthas (24 Gestures)
  5. Module 4: Shiro Bhedhas (9 Movements)
  6. Module 5: Drishti Bhedhas (8 Glances)
  7. Module 6: Dvadasha Hasta Pranamas (12 Salutations)
  8. Module 7: Sapta Talamulu (35 Tala System)
  9. Practice Studio
  10. Assignments
  11. My Progress
  12. Student Profile
  13. Logout
- **Mobile Behavior**: The sidebar must transform into a smooth slide-out drawer triggered by a header hamburger button.

### FR-LMS-02: Dynamic Next Incomplete Lesson Identification
- **Description**: The LMS must inspect the student's lesson completion records:
  - If no lessons are completed: Module 1 is highlighted as the starting point.
  - If modules $1 \dots k$ are completed: Module $k+1$ is automatically presented as the "Next Incomplete Lesson" with a one-click "Continue Learning" button.
  - If all 7 modules are completed: The dashboard displays a completion banner and directs the student to the AI Practice Studio.

### FR-LMS-03: Lesson Completion Tracking
- **Description**: Students can click "Mark Lesson Complete" to record lesson status in the database. The progress percentage ($0\% - 100\%$) must update reactively.

---

## 4. Theory Curriculum Modules (FR-CURR)

### FR-CURR-01: Seven Canonical Theory Modules
- **Module 1**: Historical roots in Kuchelapuram, Siddhendra Yogi, Nritta, Nritya, and Natya.
- **Module 2**: 28 Asamyukta Hasthas (Pathaka, Thripathaka, Ardhapathaka, Kartarimukha, Mayura, Ardhachandra, Arala, Sukatunda, Mushti, Shikhara, Kapittha, Katakamukha, Soochi, Chandrakala, Padmakosha, Sarpasheersha, Mrugasheersha, Simhamukha, Kaangula, Alapadma, Chatura, Bhramara, Hamsasya, Hamsapaksha, Samdamsha, Mukula, Tamrachuda, Trishoola).
- **Module 3**: 24 Samyukta Hasthas (Anjali, Kapothamu, Karkatamu, Swasthikamu, Dola, Pushpaputamu, Uthsungamu, Sivalingamu, Katakavardhanamu, Karthariswasthikamu, Sakatamu, Sankhamu, Chakramu, Samputamu, Paasamu, Keelakamu, Matsyamu, Kurmamu, Varahamu, Garudamu, Nagabandhakamu, Khatva, Bherundamu, Avahithamu).
- **Module 4**: 9 Shiro Bhedhas (Sama, Udvahitam, Adhomukham, Alolitam, Dhutam, Kampitam, Paravrittam, Ukshiptam, Parivahittam).
- **Module 5**: 8 Drishti Bhedhas (Sama, Alokita, Saachi, Pralokita, Nimilita, Ullokita, Anuvritta, Avalokita).
- **Module 6**: 12 Dvadasha Hasta Pranamas (Prasaranam, Kunchitam, Rechitam, Pumkhitam, Apaveshtitam, Preritam, Udveshtitam, Vyavrittam, Parivrittam, Sanketam, Chihnam, Padardha Tika).
- **Module 7**: Sapta Talamulu (Dhruva, Matya, Rupaka, Jhampe, Triputa, Ata, Eka Tala; Laghu, Dhrutam, Anudhrutam; 5 Jaatis; 35 Tala Matrix).

---

## 5. AI Practice Studio (FR-AI)

### FR-AI-01: Real-Time Camera Stream
- **Description**: The browser accesses the user's camera via `navigator.mediaDevices.getUserMedia` with video dimensions $1280 \times 720$ or $640 \times 480$.
- **Error Handling**: Graceful messaging if camera permission is denied, camera is occupied, or no video device is detected.

### FR-AI-02: Zero Fake Predictions Requirement
- **Rule**: All inference outputs displayed to the user must originate from real model execution on incoming frames. Fake random predictions or mock timeouts are prohibited.

### FR-AI-03: Multi-Category Recognition Pipeline
- **Category A (Hastas)**:
  - 175-D feature extraction using MediaPipe Hands.
  - Normalization via embedded `HASTHA_SCALER_MEAN` and `HASTHA_SCALER_SCALE`.
  - Inference through `Personalized_52Hasthas_MLP` (52 classes).
  - 8-frame temporal deque with 6/8 consensus voting and $\ge 0.20$ confidence gating.
- **Category B (Shiro Bhedas)**:
  - 102-D temporal kinematics using MediaPipe Pose (lateral shift, tilt, pitch, yaw, velocity, acceleration).
  - 160-frame buffer fed to `Shiro_BiGRU_Classifier` (9 classes).
- **Category C (Drishti Bhedas)**:
  - 132-D ocular gaze metrics using MediaPipe FaceMesh with refined iris landmarks.
  - 160-frame buffer fed to `Drishti_BiGRU_Classifier` (8 classes).

---

## 6. Assignments & Examination System (FR-EXAM)

### FR-EXAM-01: Type A — Theory MCQ Quiz
- Multiple-choice questions randomly sampled across modules.
- Instant automated grading, score calculation ($0-100\%$), correct answer display, and detailed explanations.

### FR-EXAM-02: Type B — Image Identification
- High-resolution classical dance gesture image presented to the student.
- Identification of correct mudra name with verified ground truth from the authentic dataset.

### FR-EXAM-03: Type C — Practical Camera Exam
- Randomized prompt (e.g., *"Perform Kapittha Hasta"*).
- Student performs the gesture in front of the camera.
- AI model validates posture stability and records score directly to the student's database record.
