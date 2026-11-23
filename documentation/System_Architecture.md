# System Architecture Document
## Project: Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System

---

## 1. High-Level Architectural Decomposition

The system is architected as a clean, decoupled 3-tier architecture:
1. **Client Tier**: React 18 + Vite SPA delivering a bilingual UI/UX with responsive mobile/desktop LMS layouts.
2. **Application Tier**: Python 3.11 FastAPI asynchronous server delivering REST endpoints, JWT authorization, and WebSocket/REST AI inference streaming.
3. **AI & Data Tier**:
   - Consolidated TensorFlow/Keras neural network (`nritya_ai_weights.keras`).
   - MediaPipe geometric landmark extractors (Hands, Pose, FaceMesh).
   - Embedded SQLite relational database.

```mermaid
graph TD
  subgraph Client_Tier["Client Tier (React 18 + Vite)"]
    UI_Home[Clean Public Homepage]
    UI_LMS[Student LMS Dashboard]
    UI_Nav[Permanent Left Sidebar]
    UI_Studio[AI Practice Studio HUD]
    UI_Exam[Assignments & Quizzes]
    UI_Lang[Bilingual Engine EN / TE]
    UI_Webcam[Webcam MediaDevices Stream]
  end

  subgraph API_Tier["Application Tier (FastAPI + Uvicorn)"]
    API_Gateway[FastAPI Application Gateway /api/v1]
    Auth_Service[Auth & JWT Middleware]
    LMS_Service[Curriculum & Lesson Progress Service]
    Exam_Service[Quiz & Scoring Service]
    AI_Adapter[AI Stream Inference Adapter]
  end

  subgraph AI_Engine["AI & Computer Vision Core"]
    MP_Hands[MediaPipe Hands 175-D Landmarks]
    MP_Pose[MediaPipe Pose 102-D Kinematics]
    MP_Face[MediaPipe FaceMesh 132-D Gaze]
    Weights[nritya_ai_weights.keras 1.37MB]
    subgraph Models["Consolidated Submodels"]
      M_Hastha[Personalized_52Hasthas_MLP]
      M_Shiro[Shiro_BiGRU_Classifier]
      M_Drishti[Drishti_BiGRU_Classifier]
    end
  end

  subgraph Data_Tier["Persistence Tier (SQLite)"]
    DB[(kuchipudi.db)]
  end

  UI_Webcam -->|Base64 / Binary Frames| AI_Adapter
  UI_LMS & UI_Exam -->|REST JSON / Bearer JWT| API_Gateway
  API_Gateway --> Auth_Service & LMS_Service & Exam_Service & AI_Adapter
  Auth_Service & LMS_Service & Exam_Service --> DB

  AI_Adapter --> MP_Hands & MP_Pose & MP_Face
  MP_Hands --> M_Hastha
  MP_Pose --> M_Shiro
  MP_Face --> M_Drishti
  Weights -.-> Models
```

---

## 2. Component Details

### A. Frontend Layer (React + Vite)
- **Vite Bundler**: Hot module replacement (HMR), sub-second build times, and tree-shaken production bundles.
- **State Management**:
  - `currentLang`: Persistent bilingual dictionary manager (`en` / `te`).
  - `user`: Authenticated JWT and user profile token.
  - `completedModules`: Reactive lesson progress tracking state.
- **Styling Architecture**: Custom CSS custom-property variables adhering to the Stage 2.5 refined soft palette (Blush Ivory `#F4E8E2`, Sage Green `#D9E5D5`, Lavender `#E8DDF0`, Antique Gold `#E8D1A7`, Deep Plum `#482C3A`, Temple Green `#293C35`).

### B. Backend Layer (FastAPI)
- **Asynchronous Execution**: Native async/await event loop in Python 3.11 for concurrent API requests.
- **CORS Middleware**: Explicit origin whitelisting allowing secure communication between frontend (`localhost:3000`) and backend (`localhost:8000`).
- **Pydantic Schemas**: Strict schema typing for incoming requests (registration, login, lesson progress, quiz submissions).

### C. AI Inference Adapter (`ai_engine/`)
- Isolates inference execution from user camera devices.
- Reuses the exact mathematical feature vectors, scalers, and consensus deque gates from `deployment_clean/test.py`.
- **Pre-Compiled Graph Execution**: Employs `@tf.function(reduce_retracing=True)` graph optimization to minimize CPU inference latency ($< 25\text{ ms}$ per forward pass).

---

## 3. Real-Time AI Inference Pipeline

```mermaid
flowchart TD
  Frame[Client Camera Frame RGB] --> Router{Category Selected?}
  
  Router -->|Hasthas| H1[MediaPipe Hands Detection]
  H1 --> H2[Mirror Right Hand to Canonical Left]
  H2 --> H3[Extract 87-D Joint Angles & Tip Distances]
  H3 --> H4[Concatenate Inter-wrist Distance -> 175-D]
  H4 --> H5[Standardize with HASTHA_SCALER_MEAN & SCALE]
  H5 --> H6[Personalized_52Hasthas_MLP Forward Pass]
  H6 --> H7[8-Frame Deque Buffer & 6/8 Consensus Vote]
  H7 --> Res1[52-Class Output + Confidence]

  Router -->|Shiro Bhedas| S1[MediaPipe Pose Detection]
  S1 --> S2[32-D Base Upper Body Coordinates]
  S2 --> S3[Velocity dt & Acceleration dt -> 64-D]
  S3 --> S4[Head Tilt Roll, Pitch, Yaw & Neutral Deviations -> 6-D]
  S4 --> S5[102-D Feature Vector Append to 160-Frame Temporal Buffer]
  S5 --> S6[Shiro_BiGRU_Classifier Forward Pass]
  S6 --> Res2[9-Class Output + Confidence]

  Router -->|Drishti Bhedas| D1[MediaPipe FaceMesh with Irises]
  D1 --> D2[44-D Base Gaze, EAR & Geometric Ratios]
  D2 --> D3[Velocity dt & Acceleration dt -> 88-D]
  D3 --> D4[132-D Feature Vector Append to 160-Frame Temporal Buffer]
  D4 --> D5[Drishti_BiGRU_Classifier Forward Pass]
  D5 --> Res3[8-Class Output + Confidence]
```
