import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyzeImage } from '../api';

function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleFileSelect = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!allowed.includes(selected.type)) {
      setError('Please select a valid image file (JPEG, PNG, WebP, or BMP).');
      return;
    }

    if (selected.size > 10 * 1024 * 1024) {
      setError('File is too large. Maximum size is 10MB.');
      return;
    }

    setFile(selected);
    setError(null);
    setPreview(URL.createObjectURL(selected));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      const fakeEvent = { target: { files: e.dataTransfer.files } };
      handleFileSelect(fakeEvent);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const data = await analyzeImage(file);
      navigate(`/results/${data.report.id}`, { state: { report: data.report } });
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.');
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  if (loading) {
    return (
      <div className="card">
        <div className="loading-container">
          <div className="spinner" />
          <h3>Analyzing your image...</h3>
          <p style={{ color: 'var(--text-light)' }}>
            Running face detection, preprocessing, and AI analysis.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="card">
        <h2 style={{ marginBottom: '16px' }}>Upload Your Photo</h2>
        <p style={{ color: 'var(--text-light)', marginBottom: '20px' }}>
          For best results, use a well-lit, front-facing photo of your face.
        </p>

        <div className="photo-tips">
          <h4 className="photo-tips-title">Tips for the best results</h4>
          <ul className="photo-tips-list">
            <li><span className="tip-icon">💡</span> Use bright, even lighting — avoid harsh shadows</li>
            <li><span className="tip-icon">🙂</span> Face the camera directly, fill the frame with your face</li>
            <li><span className="tip-icon">👓</span> Remove glasses, hats, and pull back hair</li>
            <li><span className="tip-icon">🧼</span> Use a clean, makeup-free face for accurate analysis</li>
            <li><span className="tip-icon">📷</span> Keep the photo sharp and in focus</li>
          </ul>
        </div>

        {error && (
          <div className="status-message status-error" style={{ marginBottom: '16px' }}>
            {error}
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        <div
          className={`upload-area ${preview ? 'has-image' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
        >
          {preview ? (
            <img src={preview} alt="Preview" className="preview-image" />
          ) : (
            <>
              <h3>Tap to upload or drag & drop</h3>
              <p>JPEG, PNG, WebP, or BMP (max 10MB)</p>
            </>
          )}
        </div>

        <div style={{ display: 'flex', gap: '12px', marginTop: '20px' }}>
          <button
            className="btn btn-primary"
            style={{ flex: 1 }}
            onClick={handleAnalyze}
            disabled={!file}
          >
            Analyze
          </button>
          {file && (
            <button className="btn btn-outline" onClick={handleReset}>
              Clear
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default AnalyzePage;
