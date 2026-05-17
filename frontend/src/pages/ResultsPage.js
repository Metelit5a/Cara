import React, { useEffect, useState } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { getReport } from '../api';

const severityColors = {
  clear: '#48BB78',
  minimal: '#48BB78',
  mild: '#ECC94B',
  moderate: '#ED8936',
  severe: '#F56565',
};

function ResultsPage() {
  const { reportId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [report, setReport] = useState(location.state?.report || null);
  const [loading, setLoading] = useState(!report);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!report) {
      getReport(reportId)
        .then(setReport)
        .catch(() => setError('Report not found.'))
        .finally(() => setLoading(false));
    }
  }, [reportId, report]);

  if (loading) {
    return (
      <div className="card loading-container">
        <div className="spinner" />
        <p>Loading report...</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="card status-message status-error">
        <h3>Error</h3>
        <p>{error || 'Report data unavailable.'}</p>
        <button className="btn btn-primary" style={{ marginTop: '12px' }} onClick={() => navigate('/analyze')}>
          Try Again
        </button>
      </div>
    );
  }

  // Handle non-success statuses
  if (report.status !== 'success') {
    return (
      <div className="card">
        <div className={`status-message ${report.status === 'no_face_detected' ? 'status-warning' : 'status-error'}`}>
          <h3>{report.status === 'low_confidence' ? 'Low Confidence' : report.status === 'no_face_detected' ? 'No Face Detected' : 'Analysis Error'}</h3>
          <p style={{ marginTop: '8px' }}>{report.message}</p>
        </div>
        <button className="btn btn-primary" style={{ width: '100%', marginTop: '12px' }} onClick={() => navigate('/analyze')}>
          Try Again
        </button>
      </div>
    );
  }

  const confidencePercent = Math.round(report.confidence * 100);
  const color = severityColors[report.acne_severity] || '#A0AEC0';
  const poreColor = severityColors[report.pore_severity] || '#A0AEC0';
  const generalAcneColor = severityColors[report.general_acne_severity] || '#A0AEC0';

  return (
    <div>
      {/* Acne Severity & Confidence */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2>Acne Analysis</h2>
          <span className={`severity-badge severity-${report.acne_severity}`}>
            {report.acne_severity}
          </span>
        </div>

        <div style={{ marginBottom: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem' }}>
            <span style={{ color: 'var(--text-light)' }}>Confidence</span>
            <span style={{ fontWeight: '600' }}>{confidencePercent}%</span>
          </div>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{ width: `${confidencePercent}%`, background: color }}
            />
          </div>
        </div>
      </div>

      {/* General Acne Analysis */}
      {report.general_acne_severity && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h2>General Acne Analysis</h2>
            <span className={`severity-badge severity-${report.general_acne_severity}`}>
              {report.general_acne_severity}
            </span>
          </div>

          {report.general_acne_confidence && (
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem' }}>
                <span style={{ color: 'var(--text-light)' }}>Confidence</span>
                <span style={{ fontWeight: '600' }}>{Math.round(report.general_acne_confidence * 100)}%</span>
              </div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${Math.round(report.general_acne_confidence * 100)}%`, background: generalAcneColor }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Pores Analysis */}
      {report.pore_severity && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h2>Pores Analysis</h2>
            <span className={`severity-badge severity-${report.pore_severity}`}>
              {report.pore_severity}
            </span>
          </div>

          {report.pore_confidence && (
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem' }}>
                <span style={{ color: 'var(--text-light)' }}>Confidence</span>
                <span style={{ fontWeight: '600' }}>{Math.round(report.pore_confidence * 100)}%</span>
              </div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${Math.round(report.pore_confidence * 100)}%`, background: poreColor }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Combined Explanation */}
      <div className="card">
        <p style={{ color: 'var(--text-light)', lineHeight: '1.7' }}>
          {report.explanation}
        </p>
      </div>

      {/* Recommendations */}
      {report.recommendations?.length > 0 && (
        <div className="card">
          <h3 style={{ marginBottom: '16px' }}>Recommended Ingredients</h3>
          {report.recommendations.map((rec, i) => (
            <div key={i} className="recommendation-item">
              <h4>{rec.ingredient}</h4>
              <p>{rec.reason}</p>
              {rec.category && (
                <span className="recommendation-category">{rec.category.replace('_', ' ')}</span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Educational Note */}
      {report.educational_note && (
        <div className="card status-info">
          <strong>Educational Note</strong>
          <p style={{ marginTop: '8px', fontSize: '0.9rem' }}>{report.educational_note}</p>
        </div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: '12px' }}>
        <button className="btn btn-primary" style={{ flex: 1 }} onClick={() => navigate('/analyze')}>
          New Analysis
        </button>
        <button className="btn btn-outline" style={{ flex: 1 }} onClick={() => navigate('/history')}>
          View History
        </button>
      </div>
    </div>
  );
}

export default ResultsPage;
