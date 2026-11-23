# Use Case Specifications & Workflows
## Project: Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System

---

## 1. High-Level Use Case Diagram

```mermaid
flowchart LR
  subgraph Students["Dance Student"]
    S1[Register Account]
    S2[Login / Logout]
    S3[Select Language EN / TE]
    S4[Study Theory Modules 1-7]
    S5[Track Next Incomplete Lesson]
    S6[Practice via Webcam in AI Studio]
    S7[Take Theory MCQ Quiz]
    S8[Take Image ID Quiz]
    S9[Take Practical Camera Exam]
    S10[Review Past Scores & History]
  end

  subgraph System["Kuchipudi Platform Engine"]
    SE1[(SQLite Database)]
    SE2[FastAPI Auth & REST API]
    SE3[MediaPipe Feature Extractor]
    SE4[Keras Consolidated Model 69 Classes]
  end

  subgraph Admin["Dance Instructor / Admin"]
    A1[Login as Admin]
    A2[Manage Learning Content]
    A3[Create / Edit Quiz Questions]
    A4[View Student Roster & Scores]
    A5[Monitor Platform Metrics]
  end

  S1 & S2 --> SE2
  S3 --> SE2
  S4 & S5 --> SE2 --> SE1
  S6 --> SE3 --> SE4
  S7 & S8 & S9 --> SE2 --> SE1
  S10 --> SE2 --> SE1

  A1 & A2 & A3 & A4 & A5 --> SE2 --> SE1
```

---

## 2. Primary Use Case Flows

### UC-01: Student Registration & Login Flow
- **Actor**: Dance Student
- **Precondition**: User has access to web browser and internet connection.
- **Workflow**:
  1. Student navigates to the platform homepage.
  2. Student clicks "Login / Register" in the top navbar.
  3. Student enters Full Name, Email, Password, and selects preferred language (English or Telugu).
  4. System validates email uniqueness, hashes password with bcrypt, and stores record in SQLite.
  5. System generates JWT access token and initializes student session.
  6. Student is redirected to the Student Dashboard with Module 1 set as the starting point.

```mermaid
sequenceDiagram
  autonumber
  actor Student
  participant UI as React Frontend
  participant API as FastAPI Backend
  participant DB as SQLite Database

  Student->>UI: Enter Email & Password
  UI->>API: POST /api/auth/login
  API->>DB: Query User by Email
  DB-->>API: Return User Record (Hashed Password)
  API->>API: Verify Password with bcrypt
  alt Credentials Valid
    API->>API: Generate JWT Access Token
    API-->>UI: 200 OK + JWT + User Profile
    UI->>UI: Store Token in localStorage
    UI-->>Student: Display Student LMS Dashboard
  else Credentials Invalid
    API-->>UI: 401 Unauthorized (Error Message)
    UI-->>Student: Display "Invalid Email or Password"
  end
```

---

### UC-02: Learning and Dynamic Progress Flow
- **Actor**: Dance Student
- **Workflow**:
  1. Student logs into dashboard.
  2. System queries `StudentLessonProgress` table for all completed module IDs.
  3. System evaluates:
     $$\text{Next Module} = \min \{ m \in \{1 \dots 7\} \mid m \notin \text{Completed} \}$$
  4. If next module exists: System highlights it on the dashboard overview with "Resume Lesson".
  5. If all 7 are completed: System displays completion celebration and recommends AI practice.
  6. Student reviews lesson text, mudra illustrations, and Sanskrit definitions.
  7. Student clicks "Mark Lesson Complete".
  8. System writes record to SQLite, updates progress bar, and re-evaluates next incomplete lesson.

```mermaid
stateDiagram-v2
  [*] --> Module1_Unstarted
  Module1_Unstarted --> Module1_InProgress: Start Lesson 1
  Module1_InProgress --> Module1_Completed: Mark Complete
  Module1_Completed --> Module2_InProgress: Auto-Advance
  Module2_InProgress --> Module2_Completed: Mark Complete
  Module2_Completed --> NextModules: Repeat for Modules 3-7
  NextModules --> All_Completed: Module 7 Complete
  All_Completed --> Practice_Recommended: Enter AI Studio
```

---

### UC-03: Real-Time AI Camera Practice Flow
- **Actor**: Dance Student
- **Precondition**: Webcam connected and browser permission granted.
- **Workflow**:
  1. Student opens Practice Studio and selects category:
     - **A. Hasthas (52 classes)**
     - **B. Shiro Bhedhas (9 classes)**
     - **C. Drishti Bhedhas (8 classes)**
  2. Student selects target mudra or movement (e.g. *Pathaka Hasta*).
  3. Video feed renders in the viewport with alignment guides.
  4. Video frames are captured and transferred to the backend inference service.
  5. Feature extractor generates normalized vectors:
     - 175-D for Hastas (MediaPipe Hands)
     - 102-D for Shiro (MediaPipe Pose velocity & angles)
     - 132-D for Drishti (MediaPipe FaceMesh gaze kinematics)
  6. Model performs forward pass through `nritya_ai_weights.keras`.
  7. Stabilization gate aggregates temporal consensus (6 out of 8 frames).
  8. Recognized gesture name, confidence, and match feedback appear on the HUD.

```mermaid
sequenceDiagram
  autonumber
  actor Student
  participant Web as Browser Webcam
  participant UI as Studio HUD
  participant AI as FastAPI AI Adapter
  participant Model as Keras Neural Network

  Student->>UI: Select Category (e.g., Hasthas) & Target
  UI->>Web: Request Camera Access (getUserMedia)
  Web-->>UI: Live Video Stream Active
  loop Every 60ms (Frame Capture)
    UI->>AI: Send Mirrored Video Frame
    AI->>AI: MediaPipe 3D Landmark Extraction
    AI->>AI: Standardize with Scaler Mean & Std
    AI->>Model: Forward Pass (1, 175)
    Model-->>AI: Softmax Class Probabilities (52)
    AI->>AI: Temporal Buffer 6/8 Majority Consensus
    AI-->>UI: Result: { gesture: "PATHAKA", confidence: 0.98, stable: true }
    UI-->>Student: Green HUD Feedback "Match Confirmed!"
  end
```

---

### UC-04: Assignment & Examination Flow
- **Actor**: Dance Student
- **Workflow**:
  1. Student selects assignment: Theory MCQ, Image ID, or Practical Exam.
  2. System loads questions or authentic test images from the curriculum database.
  3. Student submits answers.
  4. Server calculates score and writes to `StudentQuizResults` table.
  5. Student receives score percentage, correct answer review, and explanations.
  6. Student dashboard metrics update automatically.
