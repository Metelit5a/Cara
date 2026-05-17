const API_BASE = '/api/v1';

export async function detectFaces(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/detect-face`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Server error: ${response.status}`);
  }

  return response.json();
}

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Server error: ${response.status}`);
  }

  return response.json();
}

export async function getReport(reportId) {
  const response = await fetch(`${API_BASE}/report/${reportId}`);
  if (!response.ok) {
    throw new Error('Report not found');
  }
  return response.json();
}

export async function listReports(limit = 50) {
  const response = await fetch(`${API_BASE}/reports?limit=${limit}`);
  if (!response.ok) {
    throw new Error('Failed to fetch reports');
  }
  return response.json();
}

export async function healthCheck() {
  const response = await fetch('/health');
  return response.json();
}
