# Database Design Specification
## Project: Nritya AI – AI-Based Kuchipudi Dance Learning, Recognition & Analysis System

---

## 1. Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    USERS ||--o{ STUDENT_PROFILES : has
    USERS ||--o{ STUDENT_LESSON_PROGRESS : records
    USERS ||--o{ QUIZ_RESULTS : achieves
    USERS ||--o{ PRACTICE_HISTORY : logs

    MODULES ||--o{ LESSONS : contains
    MODULES ||--o{ QUIZZES : evaluated_by
    MODULES ||--o{ STUDENT_LESSON_PROGRESS : tracks

    QUIZZES ||--o{ QUESTIONS : includes
    QUIZZES ||--o{ QUIZ_RESULTS : produces
    QUESTIONS ||--o{ QUESTION_OPTIONS : provides

    USERS {
        int id PK
        string email UK
        string hashed_password
        string role "student | admin"
        boolean is_active
        datetime created_at
    }

    STUDENT_PROFILES {
        int id PK
        int user_id FK
        string full_name
        string preferred_language "en | te"
        string experience_level
        datetime updated_at
    }

    MODULES {
        int id PK
        int module_number UK
        string title_en
        string title_te
        string sanskrit_title
        string source_pdf
        int total_classes
    }

    LESSONS {
        int id PK
        int module_id FK
        int lesson_order
        string name_en
        string name_te
        string sanskrit_name
        string image_url
        text description_en
        text description_te
    }

    STUDENT_LESSON_PROGRESS {
        int id PK
        int user_id FK
        int module_id FK
        int lesson_id FK
        boolean is_completed
        datetime completed_at
    }

    QUIZZES {
        int id PK
        int module_id FK
        string quiz_type "theory_mcq | image_id | practical_exam"
        string title_en
        string title_te
        int total_questions
        int passing_percentage
    }

    QUESTIONS {
        int id PK
        int quiz_id FK
        text question_text_en
        text question_text_te
        string image_reference_url
        string explanation_en
        string explanation_te
    }

    QUESTION_OPTIONS {
        int id PK
        int question_id FK
        string option_text_en
        string option_text_te
        boolean is_correct
    }

    QUIZ_RESULTS {
        int id PK
        int user_id FK
        int quiz_id FK
        int score_obtained
        int total_score
        float percentage
        boolean passed
        datetime submitted_at
    }

    PRACTICE_HISTORY {
        int id PK
        int user_id FK
        string category "hasthas | shiro | drishti"
        string target_class_name
        string recognized_class_name
        float confidence
        int duration_seconds
        boolean is_match
        datetime practiced_at
    }
```

---

## 2. Table Schemas & Data Dictionaries

### 2.1 `users` Table
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Unique user identifier |
| `email` | `VARCHAR(255)` | `UNIQUE NOT NULL` | Login email address |
| `hashed_password` | `VARCHAR(255)` | `NOT NULL` | bcrypt-hashed password hash |
| `role` | `VARCHAR(32)` | `DEFAULT 'student'` | Access role (`student` or `admin`) |
| `is_active` | `BOOLEAN` | `DEFAULT TRUE` | Account activation flag |
| `created_at` | `DATETIME` | `DEFAULT CURRENT_TIMESTAMP` | Account creation timestamp |

### 2.2 `student_profiles` Table
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Profile record identifier |
| `user_id` | `INTEGER` | `FOREIGN KEY (users.id) ON DELETE CASCADE` | Associated user reference |
| `full_name` | `VARCHAR(255)` | `NOT NULL` | Student full legal/preferred name |
| `preferred_language` | `VARCHAR(10)` | `DEFAULT 'en'` | Default language preference (`en` or `te`) |
| `experience_level` | `VARCHAR(64)` | `DEFAULT 'beginner'` | Dance experience background |
| `updated_at` | `DATETIME` | `DEFAULT CURRENT_TIMESTAMP` | Last profile update timestamp |

### 2.3 `modules` Table
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Module database ID |
| `module_number` | `INTEGER` | `UNIQUE NOT NULL` | Number ($1 \dots 7$) |
| `title_en` | `VARCHAR(255)` | `NOT NULL` | English module title |
| `title_te` | `VARCHAR(255)` | `NOT NULL` | Telugu module title |
| `sanskrit_title` | `VARCHAR(255)` | `NOT NULL` | Sanskrit title |
| `source_pdf` | `VARCHAR(255)` | `NOT NULL` | Reference PDF filename |
| `total_classes` | `INTEGER` | `NOT NULL` | Number of gestures/movements |

### 2.4 `student_lesson_progress` Table
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Progress record identifier |
| `user_id` | `INTEGER` | `FOREIGN KEY (users.id) ON DELETE CASCADE` | Student reference |
| `module_id` | `INTEGER` | `FOREIGN KEY (modules.id) ON DELETE CASCADE` | Module reference |
| `lesson_id` | `INTEGER` | `NULLABLE` | Optional specific lesson reference |
| `is_completed` | `BOOLEAN` | `DEFAULT FALSE` | Completion flag |
| `completed_at` | `DATETIME` | `DEFAULT CURRENT_TIMESTAMP` | Timestamp of completion |

### 2.5 `quiz_results` Table
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Result record identifier |
| `user_id` | `INTEGER` | `FOREIGN KEY (users.id) ON DELETE CASCADE` | Student reference |
| `quiz_id` | `INTEGER` | `FOREIGN KEY (quizzes.id) ON DELETE CASCADE` | Associated assignment |
| `score_obtained` | `INTEGER` | `NOT NULL` | Raw score achieved |
| `total_score` | `INTEGER` | `NOT NULL` | Total score possible |
| `percentage` | `FLOAT` | `NOT NULL` | Score percentage |
| `passed` | `BOOLEAN` | `NOT NULL` | Passing status ($\ge 70\%$) |
| `submitted_at` | `DATETIME` | `DEFAULT CURRENT_TIMESTAMP` | Submission timestamp |

### 2.6 `practice_history` Table
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Session log ID |
| `user_id` | `INTEGER` | `FOREIGN KEY (users.id) ON DELETE CASCADE` | Student reference |
| `category` | `VARCHAR(32)` | `NOT NULL` | `hasthas`, `shiro`, or `drishti` |
| `target_class_name` | `VARCHAR(128)` | `NOT NULL` | Gesture student attempted |
| `recognized_class_name` | `VARCHAR(128)` | `NOT NULL` | Model prediction output |
| `confidence` | `FLOAT` | `NOT NULL` | Model confidence metric |
| `duration_seconds` | `INTEGER` | `DEFAULT 0` | Duration of practice |
| `is_match` | `BOOLEAN` | `NOT NULL` | Prediction matches target |
| `practiced_at` | `DATETIME` | `DEFAULT CURRENT_TIMESTAMP` | Timestamp of practice |
