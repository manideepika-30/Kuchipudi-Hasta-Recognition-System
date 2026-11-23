/**
 * API Service for connecting the React Frontend with the FastAPI SQLite Backend.
 * Includes intelligent API base URL detection, automatic dev-proxy routing,
 * and user-friendly error formatting.
 */

// Dynamically determine the best API base URL:
// 1. If VITE_API_URL is configured in environment, use it.
// 2. If running in browser with Vite proxy (e.g. port 3000, 5173, etc.), use relative '/api'
//    to benefit from Vite's zero-CORS reverse proxy to FastAPI.
// 3. Otherwise, connect to http://<hostname>:8000/api.
export const API_BASE = (function resolveApiBase() {
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) {
    return import.meta.env.VITE_API_URL.replace(/\/+$/, '');
  }
  if (typeof window !== 'undefined') {
    const port = window.location.port;
    const host = window.location.hostname || '127.0.0.1';
    const isViteDev = port === '3000' || port === '5173' || port === '5174' || port === '3001';
    if (isViteDev) {
      return '/api';
    }
    if (host === 'localhost' || host === '127.0.0.1') {
      return `http://${host}:9000/api`;
    }
    // Hosted production default (e.g. Vercel deployment with relative or proxy route)
    return '/api';
  }
  return 'http://127.0.0.1:9000/api';
})();

/**
 * Helper to retrieve stored JWT access token.
 */
export function getAuthToken() {
  return localStorage.getItem('kuchipudi_token');
}

/**
 * Helper for authenticated headers.
 */
function authHeaders() {
  const token = getAuthToken();
  const headers = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

/**
 * Safe fetch wrapper that handles network disconnects and provides clear messages.
 */
async function safeFetch(url, options = {}) {
  try {
    return await fetch(url, options);
  } catch (err) {
    console.error('API Network Error connecting to:', url, err);
    throw new Error(
      `Unable to connect to the Nritya AI backend (${url}). Please ensure the FastAPI server is running on http://127.0.0.1:9000.`
    );
  }
}

/**
 * Handle API responses and extract meaningful, user-friendly error messages.
 */
async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = 'Request failed';
    try {
      const errorData = await response.json();
      if (typeof errorData.detail === 'string') {
        errorDetail = errorData.detail;
      } else if (Array.isArray(errorData.detail)) {
        errorDetail = errorData.detail
          .map(d => (d.msg ? (d.loc ? `${d.loc.slice(-1)[0]}: ${d.msg}` : d.msg) : JSON.stringify(d)))
          .join('; ');
      } else if (errorData.message) {
        errorDetail = errorData.message;
      } else {
        errorDetail = JSON.stringify(errorData);
      }
    } catch {
      errorDetail = response.statusText || `HTTP Error ${response.status}`;
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

// ----------------- AUTHENTICATION -----------------

export async function loginUser(email, password) {
  const response = await safeFetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
  });
  const data = await handleResponse(response);
  if (data.access_token) {
    localStorage.setItem('kuchipudi_token', data.access_token);
    localStorage.setItem('kuchipudi_user', JSON.stringify(data.user));
  }
  return data;
}

export async function registerUser({ email, password, full_name, preferred_language = 'en' }) {
  const response = await safeFetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: email.trim().toLowerCase(),
      password,
      full_name: full_name.trim(),
      preferred_language,
    }),
  });
  const data = await handleResponse(response);
  if (data.access_token) {
    localStorage.setItem('kuchipudi_token', data.access_token);
    localStorage.setItem('kuchipudi_user', JSON.stringify(data.user));
  }
  return data;
}

export async function fetchCurrentUser() {
  const response = await safeFetch(`${API_BASE}/auth/me`, {
    method: 'GET',
    headers: authHeaders(),
  });
  return handleResponse(response);
}

export async function logoutUser() {
  const token = getAuthToken();
  if (token) {
    try {
      await safeFetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        headers: authHeaders(),
      });
    } catch (e) {
      console.warn('Backend token revocation failed:', e);
    }
  }
  localStorage.removeItem('kuchipudi_token');
  localStorage.removeItem('kuchipudi_user');
}

// ----------------- PROFILE & PREFERENCES -----------------

export async function fetchProfile() {
  const response = await safeFetch(`${API_BASE}/users/profile`, {
    method: 'GET',
    headers: authHeaders(),
  });
  return handleResponse(response);
}

export async function updateLanguagePreference(preferredLanguage) {
  const response = await safeFetch(`${API_BASE}/users/language-preference`, {
    method: 'PUT',
    headers: authHeaders(),
    body: JSON.stringify({ preferred_language: preferredLanguage }),
  });
  return handleResponse(response);
}

// ----------------- CURRICULUM & PROGRESS -----------------

export async function fetchModules() {
  const response = await safeFetch(`${API_BASE}/curriculum/modules`, {
    method: 'GET',
    headers: authHeaders(),
  });
  const modules = await handleResponse(response);

  // If unauthenticated guest, overlay browser-local completed lessons
  if (!getAuthToken() && Array.isArray(modules)) {
    try {
      const guestDoneLessons = JSON.parse(localStorage.getItem('nritya_guest_completed_lessons') || '[]');
      return modules.map(m => {
        let completedInMod = 0;
        const lessons = (m.lessons || []).map(les => {
          const isDone = guestDoneLessons.includes(les.id);
          if (isDone) completedInMod++;
          return { ...les, is_completed: isDone };
        });
        const isModDone = lessons.length > 0 && completedInMod === lessons.length;
        return {
          ...m,
          lessons,
          completed_classes: completedInMod,
          is_completed: isModDone,
        };
      });
    } catch {
      return modules;
    }
  }

  return modules;
}

export async function fetchProgress() {
  const token = getAuthToken();
  if (!token) {
    // Return browser-persisted guest progress
    try {
      const guestDoneLessons = JSON.parse(localStorage.getItem('nritya_guest_completed_lessons') || '[]');
      const guestDoneModules = JSON.parse(localStorage.getItem('nritya_guest_completed_modules') || '[]');
      const totalLessons = 92;
      const totalDone = guestDoneLessons.length;
      const percentage = Math.round((totalDone / totalLessons) * 100);

      return {
        user_id: 0,
        completed_module_ids: guestDoneModules,
        completed_lesson_ids: guestDoneLessons,
        total_completed_lessons: totalDone,
        overall_progress_percentage: percentage,
        next_incomplete_module: 1,
        next_incomplete_lesson: 'Module 1: Introduction to Kuchipudi',
        is_guest: true
      };
    } catch {
      return {
        user_id: 0,
        completed_module_ids: [],
        completed_lesson_ids: [],
        total_completed_lessons: 0,
        overall_progress_percentage: 0,
        next_incomplete_module: 1,
        next_incomplete_lesson: 'Module 1: Introduction to Kuchipudi',
        is_guest: true
      };
    }
  }

  const response = await safeFetch(`${API_BASE}/curriculum/progress`, {
    method: 'GET',
    headers: authHeaders(),
  });
  return handleResponse(response);
}

export async function toggleLessonProgress(moduleId, lessonId, isCompleted) {
  const token = getAuthToken();
  if (!token) {
    // Persist in localStorage for guest student
    try {
      let guestDone = JSON.parse(localStorage.getItem('nritya_guest_completed_lessons') || '[]');
      if (isCompleted && !guestDone.includes(lessonId)) {
        guestDone.push(lessonId);
      } else if (!isCompleted) {
        guestDone = guestDone.filter(id => id !== lessonId);
      }
      localStorage.setItem('nritya_guest_completed_lessons', JSON.stringify(guestDone));
      return { success: true, is_completed: isCompleted, guest: true };
    } catch (e) {
      console.warn('Failed to save guest progress in localStorage:', e);
      return { success: true, is_completed: isCompleted, guest: true };
    }
  }

  const response = await safeFetch(`${API_BASE}/curriculum/progress/toggle`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({
      module_id: moduleId,
      lesson_id: lessonId,
      is_completed: isCompleted,
    }),
  });
  return handleResponse(response);
}

// ----------------- ADMIN ROUTES -----------------

export async function fetchAdminMetrics() {
  const response = await safeFetch(`${API_BASE}/admin/metrics`, {
    method: 'GET',
    headers: authHeaders(),
  });
  return handleResponse(response);
}

export async function fetchAdminStudents() {
  const response = await safeFetch(`${API_BASE}/admin/students`, {
    method: 'GET',
    headers: authHeaders(),
  });
  return handleResponse(response);
}

// ----------------- QUIZZES & ASSIGNMENTS (STAGE 5) -----------------

export async function fetchQuizzes() {
  const response = await safeFetch(`${API_BASE}/quizzes`, {
    method: 'GET',
    headers: authHeaders(),
  });
  const qList = await handleResponse(response);

  // If unauthenticated guest, attach local best scores
  if (!getAuthToken() && Array.isArray(qList)) {
    try {
      const guestHistory = JSON.parse(localStorage.getItem('nritya_guest_quiz_history') || '[]');
      return qList.map(q => {
        const attempts = guestHistory.filter(h => h.quiz_id === q.id);
        const bestScore = attempts.length > 0 ? Math.max(...attempts.map(a => a.percentage)) : null;
        const userPassed = attempts.some(a => a.passed);
        return {
          ...q,
          user_best_score: bestScore,
          user_passed: userPassed,
          attempts_count: attempts.length
        };
      });
    } catch {
      return qList;
    }
  }

  return qList;
}

export async function fetchQuizDetail(quizId) {
  const response = await safeFetch(`${API_BASE}/quizzes/${quizId}`, {
    method: 'GET',
    headers: authHeaders(),
  });
  return handleResponse(response);
}

export async function submitQuiz(quizId, answers) {
  const response = await safeFetch(`${API_BASE}/quizzes/${quizId}/submit`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ answers }),
  });
  const data = await handleResponse(response);

  // If guest, save authenticated result locally
  if (!getAuthToken() || data.guest_mode) {
    try {
      const history = JSON.parse(localStorage.getItem('nritya_guest_quiz_history') || '[]');
      history.unshift({
        id: Date.now(),
        quiz_id: data.quiz_id,
        quiz_title_en: data.quiz_title_en,
        quiz_title_te: data.quiz_title_te,
        score: data.score,
        total_questions: data.total_questions,
        percentage: data.percentage,
        passed: data.passed,
        completed_at: data.completed_at,
        review_details: data.review_details
      });
      localStorage.setItem('nritya_guest_quiz_history', JSON.stringify(history.slice(0, 50)));
    } catch (e) {
      console.warn('Failed to save guest quiz history:', e);
    }
  }

  return data;
}

export async function fetchQuizHistory() {
  if (!getAuthToken()) {
    try {
      return JSON.parse(localStorage.getItem('nritya_guest_quiz_history') || '[]');
    } catch {
      return [];
    }
  }
  const response = await safeFetch(`${API_BASE}/quizzes/results/history`, {
    method: 'GET',
    headers: authHeaders(),
  });
  return handleResponse(response);
}

export async function fetchAdminQuizResults() {
  const response = await safeFetch(`${API_BASE}/quizzes/admin/all-results`, {
    method: 'GET',
    headers: authHeaders(),
  });
  return handleResponse(response);
}

// ----------------- AI PRACTICE STUDIO (STAGE 6) -----------------

export async function fetchAIStatus() {
  const response = await safeFetch(`${API_BASE}/ai/status`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });
  return handleResponse(response);
}

export async function fetchSupportedClasses() {
  const response = await safeFetch(`${API_BASE}/ai/classes`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });
  return handleResponse(response);
}

export async function predictCameraFrame(imageBase64, category = 'hasthas', targetGesture = null) {
  const response = await safeFetch(`${API_BASE}/ai/predict/frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: imageBase64,
      category,
      target_gesture: targetGesture,
    }),
  });
  return handleResponse(response);
}

export async function recordPracticeSession(payload) {
  if (!getAuthToken()) {
    try {
      const record = {
        id: Date.now(),
        ...payload,
        created_at: new Date().toISOString()
      };
      const history = JSON.parse(localStorage.getItem('nritya_guest_practice_history') || '[]');
      history.unshift(record);
      localStorage.setItem('nritya_guest_practice_history', JSON.stringify(history.slice(0, 50)));
      return record;
    } catch (e) {
      console.warn('Failed to save guest practice history:', e);
      return payload;
    }
  }

  const response = await safeFetch(`${API_BASE}/ai/practice/record`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(payload),
  });
  return handleResponse(response);
}

export async function fetchPracticeHistory() {
  if (!getAuthToken()) {
    try {
      return JSON.parse(localStorage.getItem('nritya_guest_practice_history') || '[]');
    } catch {
      return [];
    }
  }

  const response = await safeFetch(`${API_BASE}/ai/practice/history`, {
    method: 'GET',
    headers: authHeaders(),
  });
  return handleResponse(response);
}
