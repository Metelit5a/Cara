export function getStoredToken() {
  return localStorage.getItem('access_token');
}

export function isAuthenticated() {
  return Boolean(getStoredToken());
}

export function saveAuthToken(token) {
  localStorage.setItem('access_token', token);
}

export function clearAuthToken() {
  localStorage.removeItem('access_token');
}

export function getCurrentUsername() {
  return localStorage.getItem('username') || '';
}

export function saveUsername(username) {
  if (username) {
    localStorage.setItem('username', username);
  }
}

export function clearUsername() {
  localStorage.removeItem('username');
}
