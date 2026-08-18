const DEFAULT_API_BASE = '/api/v1';
const API_BASE = process.env.REACT_APP_API_BASE || DEFAULT_API_BASE;

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const token = localStorage.getItem('access_token');
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Server error: ${response.status}`);
  }

  return response.json();
}

export async function getReport(reportId) {
  const token = localStorage.getItem('access_token');
  const response = await fetch(`${API_BASE}/report/${reportId}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) {
    throw new Error('Report not found');
  }
  return response.json();
}

export async function listReports(limit = 50) {
  const token = localStorage.getItem('access_token');
  const response = await fetch(`${API_BASE}/reports?limit=${limit}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) {
    throw new Error('Failed to fetch reports');
  }
  return response.json();
}

export async function healthCheck() {
  const response = await fetch('/health');
  return response.json();
}

// ── Auth ──
// These go through the CRA dev proxy (package.json "proxy") so the browser
// sees them as same-origin. That avoids CORS entirely and means we never
// hard-code a host/port — the #1 cause of "Failed to fetch" during local dev.

export async function registerUser({ username, email, password }) {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || 'Registration failed. Please try again.');
  }
  return data;
}

export async function loginUser({ email, password }) {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || 'Login failed. Please check your credentials.');
  }
  return data;
}
