import React from 'react';
import { useNavigate } from 'react-router-dom';

function HomePage() {
  const navigate = useNavigate();

  return (
    <div>
      <div className="card" style={{ textAlign: 'center', padding: '48px 24px' }}>
        <h2 style={{ fontSize: '2rem', marginBottom: '12px' }}>
          AI-Powered Skincare Analysis
        </h2>
        <p style={{ color: 'var(--text-light)', fontSize: '1.1rem', maxWidth: '500px', margin: '0 auto 32px' }}>
          Get personalized, explainable skincare insights powered by computer vision.
          Upload a photo and receive educational ingredient recommendations.
        </p>
        <button className="btn btn-primary" onClick={() => navigate('/analyze')}>
          Start Analysis
        </button>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: '16px' }}>How It Works</h3>
        <div style={{ display: 'grid', gap: '16px' }}>
          {[
            { step: '1', title: 'Upload', desc: 'Take or upload a clear, front-facing photo of your face. A single face is required.' },
            { step: '2', title: 'Face Detection', desc: 'We verify exactly one face is detected in your image for accuracy.' },
            { step: '3', title: 'Analyze', desc: 'Our AI processes the image through preprocessing and AI analysis.' },
            { step: '4', title: 'Results', desc: 'Receive an explainable report with severity assessment and ingredient recommendations.' },
          ].map((item) => (
            <div key={item.step} style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
              <div style={{
                width: '36px', height: '36px', borderRadius: '50%',
                background: 'var(--primary)', color: 'white',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontWeight: '700', flexShrink: 0
              }}>
                {item.step}
              </div>
              <div>
                <strong>{item.title}</strong>
                <p style={{ color: 'var(--text-light)', fontSize: '0.9rem' }}>{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card status-info">
        <strong>Educational Use Only</strong>
        <p style={{ marginTop: '4px', fontSize: '0.9rem' }}>
          Cara provides cosmetic skincare insights and educational ingredient recommendations.
          It is not a medical device and should not be used for medical diagnosis.
        </p>
      </div>
    </div>
  );
}

export default HomePage;
