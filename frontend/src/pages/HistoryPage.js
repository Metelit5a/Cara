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
  ReferenceLine,
} from 'recharts';
import { listReports } from '../api';

function confidenceLabel(confidence) {
  if (!confidence) return '\u2014';
  if (confidence >= 0.6) return 'High';
  if (confidence >= 0.35) return 'Medium';
  return 'Low';
}

function severityToValue(severity) {
  const normalizedSeverity = typeof severity === 'string' ? severity.toLowerCase().trim() : '';

  switch (normalizedSeverity) {
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

function severityToLabel(severity) {
  const normalizedSeverity = typeof severity === 'string' ? severity.toLowerCase().trim() : '';

  switch (normalizedSeverity) {
    case 'clear':
      return 'Clear';
    case 'mild':
      return 'Mild';
    case 'moderate':
      return 'Moderate';
    case 'severe':
      return 'Severe';
    default:
      return 'Clear';
  }
}

function formatChartDate(date) {
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');

  return `${month}/${day}, ${hours}:${minutes}`;
}

function formatConfidence(confidence) {
  if (typeof confidence !== 'number' || Number.isNaN(confidence)) {
    return 'N/A';
  }

  return `${Math.round(confidence * 100)}%`;
}

function formatSkinType(skinType) {
  if (typeof skinType !== 'string' || skinType.trim() === '') {
    return 'Unknown';
  }

  return skinType.trim().charAt(0).toUpperCase() + skinType.trim().slice(1);
}

function formatConditions(conditions) {
  if (!Array.isArray(conditions)) {
    return 'None';
  }

  const parsedConditions = conditions
    .map((condition) => {
      if (typeof condition === 'string') {
        return condition.trim();
      }

      if (condition && typeof condition === 'object') {
        return condition.label || condition.name || '';
      }

      return '';
    })
    .filter(Boolean);

  return parsedConditions.length > 0 ? parsedConditions.join(', ') : 'None';
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const entry = payload[0]?.payload;
  const tooltipDate = entry?.rawDate || label || 'Unknown';
  const severityLabelText = entry?.label || 'Clear';
  const confidenceText = entry?.confidence || 'N/A';
  const skinTypeText = entry?.skinType || 'Unknown';
  const conditionsText = entry?.conditions || 'None';

  return (
    <div
      style={{
        backgroundColor: 'white',
        border: '1px solid #e5e7eb',
        borderRadius: '12px',
        boxShadow: '0 10px 24px rgba(0, 0, 0, 0.12)',
        padding: '12px 14px',
        minWidth: '210px',
      }}
    >
      <p style={{ margin: '0 0 6px', fontSize: '0.82rem', color: 'var(--text-light)' }}>
        {tooltipDate}
      </p>
      <p style={{ margin: '0 0 6px', fontWeight: 600, color: 'var(--text-dark)' }}>
        Severity: {severityLabelText}
      </p>
      <p style={{ margin: '0 0 6px', fontSize: '0.9rem', color: 'var(--text-dark)' }}>
        Model Confidence: {confidenceText}
      </p>
      <p style={{ margin: '0 0 6px', fontSize: '0.9rem', color: 'var(--text-dark)' }}>
        Skin Type: {skinTypeText}
      </p>
      <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-dark)' }}>
        Detected Conditions: {conditionsText}
      </p>
    </div>
  );
}

function ClickableDot({ cx, cy, payload, radius = 4, onClick }) {
  if (cx == null || cy == null) {
    return null;
  }

  return (
    <circle
      cx={cx}
      cy={cy}
      r={radius}
      fill="white"
      stroke="var(--primary)"
      strokeWidth={2}
      style={{ cursor: 'pointer' }}
      onClick={() => onClick?.(payload?.id)}
    />
  );
}

function HistoryPage() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      navigate('/login');
      return;
    }

    listReports()
      .then(setReports)
      .catch(() => setReports([]))
      .finally(() => setLoading(false));
  }, [navigate]);

  if (loading) {
    return (
      <div className="card loading-container">
        <div className="spinner" />
        <p>Loading history...</p>
      </div>
    );
  }

  const severityLabels = { 1: 'Clear', 2: 'Mild', 3: 'Moderate', 4: 'Severe' };

  const chartData = [...reports]
    .filter((report) => report?.created_at)
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    .map((report) => {
      const createdAt = new Date(report.created_at);
      const severity = report?.acne_severity ?? 'clear';
      const normalizedSeverity = typeof severity === 'string' ? severity.toLowerCase().trim() : '';
      const resolvedSeverity = ['clear', 'mild', 'moderate', 'severe'].includes(normalizedSeverity)
        ? normalizedSeverity
        : 'clear';

      return {
        id: report.id,
        date: formatChartDate(createdAt),
        rawDate: createdAt.toLocaleString(undefined, {
          month: 'short',
          day: 'numeric',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        }),
        acneLevel: severityToValue(resolvedSeverity),
        label: severityToLabel(resolvedSeverity),
        confidence: formatConfidence(report?.acne_confidence),
        skinType: formatSkinType(report?.skin_type),
        conditions: formatConditions(report?.skin_conditions),
      };
    });

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
            <div style={{ width: '100%', height: 300, marginBottom: '24px' }}>
              <ResponsiveContainer>
                <LineChart data={chartData} margin={{ left: 30, right: 20, top: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="date" axisLine={false} tickLine={false} minTickGap={16} />
                  <YAxis
                    domain={[1, 4]}
                    ticks={[1, 2, 3, 4]}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(value) => severityLabels[value] || value}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine
                    y={3}
                    stroke="red"
                    strokeDasharray="3 3"
                    label={{ value: 'Needs Attention', position: 'insideTopRight', fill: 'red', fontSize: 12 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="acneLevel"
                    stroke="var(--primary)"
                    strokeWidth={3}
                    dot={(props) => (
                      <ClickableDot
                        {...props}
                        onClick={(id) => { if (id) navigate(`/results/${id}`); }}
                      />
                    )}
                    activeDot={(props) => (
                      <ClickableDot
                        {...props}
                        radius={6}
                        onClick={(id) => { if (id) navigate(`/results/${id}`); }}
                      />
                    )}
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
