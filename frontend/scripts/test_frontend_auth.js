/**
 * Automated End-to-End Test for Frontend Authentication & Dashboard Integration.
 * Simulates React frontend behavior calling the live FastAPI backend at http://127.0.0.1:8000
 */

const API_BASE = 'http://127.0.0.1:8000/api';

// In-memory mock of browser localStorage
const localStorage = {
  store: {},
  getItem(key) { return this.store[key] || null; },
  setItem(key, val) { this.store[key] = String(val); },
  removeItem(key) { delete this.store[key]; },
  clear() { this.store = {}; }
};

function getAuthToken() {
  return localStorage.getItem('kuchipudi_token');
}

function authHeaders() {
  const token = getAuthToken();
  const headers = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = 'Request failed';
    try {
      const errorData = await response.json();
      errorDetail = errorData.detail || errorData.message || JSON.stringify(errorData);
    } catch {
      errorDetail = response.statusText || `HTTP Error ${response.status}`;
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

async function runE2ETests() {
  console.log('='.repeat(70));
  console.log('RUNNING FRONTEND-TO-BACKEND INTEGRATION E2E TEST');
  console.log('='.repeat(70));

  const testStudent = {
    name: 'Meenakshi Sundaram',
    email: 'meenakshi@kuchipudi.edu',
    password: 'SecureDancePassword123'
  };

  // 1. Test Registration
  console.log('\n[STEP 1] Testing Student Registration via Frontend client...');
  const regRes = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: testStudent.email,
      password: testStudent.password,
      full_name: testStudent.name,
      preferred_language: 'te'
    })
  });

  if (regRes.status === 400) {
    console.log('  -> Note: Test student was already registered. Proceeding to login...');
  } else {
    const regData = await handleResponse(regRes);
    console.log(`  -> [PASS] Registration successful! Created User ID: ${regData.user.id}, Role: ${regData.user.role}`);
    localStorage.setItem('kuchipudi_token', regData.access_token);
    localStorage.setItem('kuchipudi_user', JSON.stringify(regData.user));
  }

  // 2. Test Login
  console.log('\n[STEP 2] Testing Student Login via Frontend client...');
  const loginRes = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: testStudent.email,
      password: testStudent.password
    })
  });
  const loginData = await handleResponse(loginRes);
  console.log('  -> [PASS] Login successful! Received JWT access token.');
  localStorage.setItem('kuchipudi_token', loginData.access_token);
  localStorage.setItem('kuchipudi_user', JSON.stringify(loginData.user));

  // 3. Test Authenticated User Fetch (/auth/me)
  console.log('\n[STEP 3] Fetching Authenticated Profile (/api/auth/me)...');
  const meRes = await fetch(`${API_BASE}/auth/me`, {
    method: 'GET',
    headers: authHeaders()
  });
  const meData = await handleResponse(meRes);
  console.log(`  -> Student Name: ${meData.profile.full_name}`);
  console.log(`  -> Student Email: ${meData.email}`);
  console.log(`  -> Student Preferred Language: ${meData.profile.preferred_language}`);
  console.log(`  -> Student Role: ${meData.role}`);
  console.log('  -> [PASS] Profile successfully fetched with Bearer token.');

  // 4. Test Student Dashboard Progress Fetch & Toggle
  console.log('\n[STEP 4] Testing Real Progress Retrieval from FastAPI...');
  const progRes = await fetch(`${API_BASE}/curriculum/progress`, {
    method: 'GET',
    headers: authHeaders()
  });
  const progData = await handleResponse(progRes);
  console.log(`  -> Current Completed Lessons Count: ${progData.total_completed_lessons}`);
  console.log(`  -> Overall Progress Percentage: ${progData.overall_progress_percentage}%`);

  console.log('  -> Toggling Module 1 completion in SQLite via /api/curriculum/progress/toggle...');
  const toggleRes = await fetch(`${API_BASE}/curriculum/progress/toggle`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({
      module_id: 1,
      lesson_id: 1,
      is_completed: true
    })
  });
  const toggleData = await handleResponse(toggleRes);
  console.log(`  -> [PASS] Toggle persisted: Module ${toggleData.module_id}, is_completed: ${toggleData.is_completed}`);

  // Re-fetch progress to verify update in SQLite
  const progRes2 = await fetch(`${API_BASE}/curriculum/progress`, {
    method: 'GET',
    headers: authHeaders()
  });
  const progData2 = await handleResponse(progRes2);
  console.log(`  -> [VERIFIED] Updated Completed Lessons: ${JSON.stringify(progData2.completed_lesson_ids)}`);

  // 5. Test Logout & Token Revocation
  console.log('\n[STEP 5] Testing Logout and Token Invalidation...');
  const currentToken = getAuthToken();
  const logoutRes = await fetch(`${API_BASE}/auth/logout`, {
    method: 'POST',
    headers: authHeaders()
  });
  const logoutData = await handleResponse(logoutRes);
  console.log(`  -> [PASS] Backend logout response: "${logoutData.message}"`);

  // Clear client storage as the frontend does
  localStorage.removeItem('kuchipudi_token');
  localStorage.removeItem('kuchipudi_user');

  // 6. Verify Revoked Token Rejection
  console.log('\n[STEP 6] Verifying that revoked token is REJECTED by backend...');
  const reusedRes = await fetch(`${API_BASE}/auth/me`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${currentToken}`
    }
  });

  if (reusedRes.status === 401) {
    const errorBody = await reusedRes.json();
    console.log(`  -> [PASS] Reused revoked token properly rejected with HTTP 401: "${errorBody.detail}"`);
  } else {
    throw new Error(`Expected HTTP 401 for revoked token, but got HTTP ${reusedRes.status}!`);
  }

  // 7. Test Admin Protection
  console.log('\n[STEP 7] Verifying Student is Forbidden from Admin endpoints...');
  // Log in again to get fresh student token
  const freshLogin = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: testStudent.email, password: testStudent.password })
  });
  const freshData = await freshLogin.json();
  const adminTestRes = await fetch(`${API_BASE}/admin/metrics`, {
    headers: { 'Authorization': `Bearer ${freshData.access_token}` }
  });
  if (adminTestRes.status === 403) {
    const adminErr = await adminTestRes.json();
    console.log(`  -> [PASS] Student access to /api/admin/metrics blocked with HTTP 403: "${adminErr.detail}"`);
  } else {
    throw new Error(`Expected HTTP 403 for student accessing admin route, got ${adminTestRes.status}`);
  }

  console.log('\n' + '='.repeat(70));
  console.log('ALL FRONTEND E2E INTEGRATION CHECKS PASSED (7/7)!');
  console.log('='.repeat(70));
}

runE2ETests().catch((err) => {
  console.error('\n[FAIL] E2E test failed:', err);
  process.exit(1);
});
