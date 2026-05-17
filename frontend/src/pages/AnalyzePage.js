import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { detectFaces, analyzeImage } from '../api';

function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [error, setError] = useState(null);
  const [faceStatus, setFaceStatus] = useState(null); // {faces: 0/1/2+, message: string}
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleFileSelect = async (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!allowed.includes(selected.type)) {
      setError('Please select a valid image file (JPEG, PNG, WebP, or BMP).');
      setFile(null);
      setFaceStatus(null);
      return;
    }

    if (selected.size > 10 * 1024 * 1024) {
      setError('File is too large. Maximum size is 10MB.');
      setFile(null);
      setFaceStatus(null);
      return;
    }

    setFile(selected);
    setError(null);
    setPreview(URL.createObjectURL(selected));

    // Run face detection
    setDetecting(true);
    try {
      const result = await detectFaces(selected);
      setFaceStatus({
        faces: result.face_count,
        message: result.message,
        bbox: result.bbox,
      });
      if (result.face_count !== 1) {
        setError(result.message);
      } else {
        setError(null);
      }
    } catch (err) {
      setError(err.message || 'Failed to detect face. Please try another image.');
      setFaceStatus(null);
    } finally {
      setDetecting(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      const fakeEvent = { target: { files: e.dataTransfer.files } };
      handleFileSelect(fakeEvent);
    }
  };

  const handleAnalyze = async () => {
    if (!file || !faceStatus || faceStatus.faces !== 1) return;
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
    setFaceStatus(null);
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

        {error && (
          <div className="status-message status-error" style={{ marginBottom: '16px' }}>
            {error}
          </div>
        )}

        {faceStatus && faceStatus.faces === 1 && !error && (
          <div className="status-message status-success" style={{ marginBottom: '16px' }}>
            ✓ {faceStatus.message}
          </div>
        )}

        {detecting && (
          <div className="status-message" style={{ marginBottom: '16px', color: 'var(--primary)' }}>
            🔍 Analyzing image for faces...
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
            disabled={!file || !faceStatus || faceStatus.faces !== 1 || detecting}
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
