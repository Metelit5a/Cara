import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import HomePage from './pages/HomePage';
import AnalyzePage from './pages/AnalyzePage';
import ResultsPage from './pages/ResultsPage';
import HistoryPage from './pages/HistoryPage';
import Login from './pages/Login';
import Register from './pages/Register';

function App() {
  return (
    <Router>
      <div className="app-container">
        <header className="header">
          <div className="header-inner">
            <NavLink to="/"><h1>Cara</h1></NavLink>
            <nav>
              <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''}>Home</NavLink>
              <NavLink to="/analyze" className={({ isActive }) => isActive ? 'active' : ''}>Analyze</NavLink>
              <NavLink to="/history" className={({ isActive }) => isActive ? 'active' : ''}>History</NavLink>
              <NavLink to="/login" className={({ isActive }) => isActive ? 'active' : ''}>Login</NavLink>
            </nav>
          </div>
        </header>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/results/:reportId" element={<ResultsPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
          </Routes>
        </main>

        <footer className="disclaimer">
          Cara provides cosmetic skincare insights for educational purposes only.
          This is not a medical diagnosis tool. Consult a dermatologist for medical advice.
        </footer>
      </div>
    </Router>
  );
}

export default App;
