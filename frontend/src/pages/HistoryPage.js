import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { listReports } from '../api';

function confidenceLabel(confidence) {
  if (!confidence) return '\u2014';
  if (confidence >= 0.6) return 'High';
  if (confidence >= 0.35) return 'Medium';
  return 'Low';
}

function severityToValue(severity) {
  switch (severity) {
    case 'clear':
      return 1;
    case 'mild':
      return 2;
    case 'moderate':
      return 3;
    case 'severe':
      return 4;
    default:
      return 1;
  }
}

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

  const chartData = [...reports]
    .filter((report) => report.created_at)
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    .map((report) => ({
      id: report.id,
      date: new Date(report.created_at).toLocaleDateString(undefined, {
        month: '2-digit',
        day: '2-digit',
      }),
      acneLevel: severityToValue(report.acne_severity),
      label: report.acne_severity || 'clear',
    }));

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
          <>
            <div style={{ width: '100%', height: 260, marginBottom: '24px' }}>
              <ResponsiveContainer>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={[1, 4]} ticks={[1, 2, 3, 4]} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="acneLevel"
                    stroke="var(--primary)"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {reports.map((report) => (
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
                  {confidenceLabel(report.acne_confidence)}
                </span>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

export default HistoryPage;
