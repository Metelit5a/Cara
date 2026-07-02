import React, { useEffect, useState } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { getReport } from '../api';

// Map a raw confidence score (0-1) to a qualitative level for display.
// Showing "Low / Medium / High" reads far better in a demo than a bare 42%.
function getConfidenceLevel(confidence) {
  const pct = (confidence || 0) * 100;
  if (pct >= 60) return { label: 'High', color: '#48BB78', fill: 90 };
  if (pct >= 35) return { label: 'Medium', color: '#ECC94B', fill: 60 };
  return { label: 'Low', color: '#ED8936', fill: 30 };
}

function ConfidenceMeter({ confidence }) {
  const level = getConfidenceLevel(confidence);
  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.9rem' }}>
        <span style={{ color: 'var(--text-light)' }}>Confidence</span>
        <span className="confidence-level-badge" style={{ background: level.color }}>
          {level.label}
        </span>
      </div>
      <div className="confidence-bar">
        <div className="confidence-fill" style={{ width: `${level.fill}%`, background: level.color }} />
      </div>
    </div>
  );
}

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

        <ConfidenceMeter confidence={report.acne_confidence} />
      </div>

      {/* Skin Type */}
      {report.skin_type && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h2>Skin Type</h2>
            <span className={`severity-badge severity-${report.skin_type}`}>
              {report.skin_type}
            </span>
          </div>
          <ConfidenceMeter confidence={report.skin_type_confidence} />
        </div>
      )}

      {/* Skin Issues */}
      {report.skin_issue && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h2>Skin Condition</h2>
            <span className={`severity-badge severity-${report.skin_issue}`}>
              {report.skin_issue.replace('_', ' ')}
            </span>
          </div>
          <ConfidenceMeter confidence={report.skin_issue_confidence} />
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
