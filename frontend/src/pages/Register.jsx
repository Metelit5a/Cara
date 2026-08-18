import React, { useState } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import { registerUser, loginUser } from '../api';

function Register({ authValue }) {
  const navigate = useNavigate();
  const location = useLocation();
  const redirectTo = location.state?.from?.pathname || '/';
  const [formData, setFormData] = useState({ username: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setLoading(true);

    try {
      await registerUser(formData);
      const loginData = await loginUser({ email: formData.email, password: formData.password });
      
      authValue.login(loginData.access_token, formData.username);
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setError(err.message || 'Unexpected error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card auth-card">
      <h2>Create your account</h2>
      <p className="auth-subtitle">Join Cara and get personalized skincare insights.</p>

      {error && <div className="status-message status-error">{error}</div>}

      <form onSubmit={handleSubmit} className="auth-form">
        <label className="auth-label" htmlFor="username">
          Username
        </label>
        <input
          id="username"
          name="username"
          type="text"
          value={formData.username}
          onChange={handleChange}
          required
          placeholder="Choose a username"
        />

        <label className="auth-label" htmlFor="email">
          Email
        </label>
        <input
          id="email"
          name="email"
          type="email"
          value={formData.email}
          onChange={handleChange}
          required
          placeholder="you@example.com"
        />

        <label className="auth-label" htmlFor="password">
          Password
        </label>
        <input
          id="password"
          name="password"
          type="password"
          value={formData.password}
          onChange={handleChange}
          required
          placeholder="Create a secure password"
        />

        <button className="btn btn-primary auth-submit" type="submit" disabled={loading}>
          {loading ? 'Creating account...' : 'Register'}
        </button>
      </form>

      <p className="auth-link-row">
        Already have an account?{' '}
        <Link to="/login" state={location.state}>Log in</Link>
      </p>
    </div>
  );
}

export default Register;