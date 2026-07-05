const API_BASE = '/api/v1';

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
