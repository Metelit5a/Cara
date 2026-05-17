import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listReports } from '../api';

function HistoryPage() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    listReports()
      .then(setReports)
      .catch(() => setReports([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="card loading-container">
        <div className="spinner" />
        <p>Loading history...</p>
      </div>
    );
  }

  return (
    <div>
      <div className="card">
        <h2 style={{ marginBottom: '16px' }}>Analysis History</h2>

        {reports.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '32px', color: 'var(--text-light)' }}>
            <p>No analyses yet.</p>
            <button
              className="btn btn-primary"
              style={{ marginTop: '16px' }}
              onClick={() => navigate('/analyze')}
            >
              Start Your First Analysis
            </button>
          </div>
        ) : (
          reports.map((report) => (
            <div
              key={report.id}
              className="history-item"
              onClick={() => navigate(`/results/${report.id}`)}
            >
              <div>
                <span
                  className={`severity-badge severity-${report.acne_severity || 'clear'}`}
                  style={{ fontSize: '0.75rem', padding: '3px 10px' }}
                >
                  {report.status === 'success' ? report.acne_severity : report.status}
                </span>
                <span style={{ marginLeft: '12px', color: 'var(--text-light)', fontSize: '0.85rem' }}>
                  {new Date(report.created_at).toLocaleString()}
                </span>
              </div>
              <span style={{ color: 'var(--text-light)', fontSize: '0.9rem' }}>
                {report.confidence ? `${Math.round(report.confidence * 100)}%` : '—'}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default HistoryPage;
