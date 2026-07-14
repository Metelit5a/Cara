import React, { useCallback, useEffect, useRef, useState } from 'react';

const HOLD_STEADY_MS = 1200;

const MEDIAPIPE_VERSION = '0.10.14';
const MEDIAPIPE_ESM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
const MEDIAPIPE_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MEDIAPIPE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';

// Unified detector interface: detect(canvas) → [{ boundingBox, landmarks }]
async function initFaceDetector() {
  // Prefer the browser's native Shape Detection API when available.
  if ('FaceDetector' in window) {
    try {
      // eslint-disable-next-line no-undef
      const native = new window.FaceDetector({ fastMode: false, maxDetectedFaces: 1 });
      return {
        kind: 'native',
        detect: async (canvas) => {
          const faces = await native.detect(canvas);
          return faces.map((f) => ({ boundingBox: f.boundingBox, landmarks: f.landmarks || [] }));
        },
      };
    } catch {
      // fall through to MediaPipe
    }
  }

  // MediaPipe fallback — works on Safari/Firefox/etc.
  try {
    const vision = await import(/* webpackIgnore: true */ MEDIAPIPE_ESM);
    const fileset = await vision.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
    const mp = await vision.FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MEDIAPIPE_MODEL, delegate: 'GPU' },
      runningMode: 'IMAGE',
      minDetectionConfidence: 0.5,
    });
    return {
      kind: 'mediapipe',
      detect: async (canvas) => {
        const result = mp.detect(canvas);
        const W = canvas.width;
        const H = canvas.height;
        return (result.detections || []).map((d) => {
          const bb = d.boundingBox || {};
          const kps = d.keypoints || [];
          // MediaPipe key-point order: 0 right eye, 1 left eye, 2 nose,
          // 3 mouth, 4 right ear, 5 left ear. Coordinates are normalised.
          const denorm = (k) => (k ? { x: k.x * W, y: k.y * H } : null);
          const landmarks = [];
          const re = denorm(kps[0]);
          const le = denorm(kps[1]);
          const ns = denorm(kps[2]);
          const mo = denorm(kps[3]);
          if (re) landmarks.push({ type: 'eye', locations: [re] });
          if (le) landmarks.push({ type: 'eye', locations: [le] });
          if (ns) landmarks.push({ type: 'nose', locations: [ns] });
          if (mo) landmarks.push({ type: 'mouth', locations: [mo] });
          return {
            boundingBox: {
              x: bb.originX ?? 0,
              y: bb.originY ?? 0,
              width: bb.width ?? 0,
              height: bb.height ?? 0,
            },
            landmarks,
          };
        });
      },
    };
  } catch (e) {
    console.warn('Failed to initialise MediaPipe face detector', e);
    return null;
  }
}

function CameraCapture({ onCapture, onCancel }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const analysisCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const detectorRef = useRef(null);
  const readySinceRef = useRef(null);
  const autoCaptureRef = useRef(false);

  const [error, setError] = useState(null);
  const [ready, setReady] = useState(false);
  const [countdown, setCountdown] = useState(null);
  const [detectorState, setDetectorState] = useState('loading'); // 'loading' | 'ready' | 'unavailable'
  const [checks, setChecks] = useState({
    lighting: { status: 'checking', message: 'Checking lighting…' },
    face: { status: 'checking', message: 'Looking for your face…' },
    position: { status: 'checking', message: 'Center your face in the oval' },
    angle: { status: 'checking', message: 'Face the camera straight on' },
  });
  const [primaryHint, setPrimaryHint] = useState('Align your face inside the oval');

  const stopStream = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  const capture = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    // Mirror horizontally so the saved photo matches what the user saw.
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
        stopStream();
        onCapture(file, URL.createObjectURL(blob));
      },
      'image/jpeg',
      0.92,
    );
  }, [onCapture, stopStream]);

  const runAnalysis = useCallback(async () => {
    const video = videoRef.current;
    const analysisCanvas = analysisCanvasRef.current;
    if (!video || !analysisCanvas || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(runAnalysis);
      return;
    }

    const W = 160;
    const H = Math.round((video.videoHeight / video.videoWidth) * W) || 120;
    analysisCanvas.width = W;
    analysisCanvas.height = H;
    const ctx = analysisCanvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(video, 0, 0, W, H);

    // Lighting: mean luminance + rough uniformity of the central region.
    const cx = Math.floor(W * 0.25);
    const cy = Math.floor(H * 0.2);
    const cw = W - cx * 2;
    const ch = H - cy * 2;
    const img = ctx.getImageData(cx, cy, cw, ch).data;
    let sum = 0;
    let sumSq = 0;
    const px = img.length / 4;
    for (let i = 0; i < img.length; i += 4) {
      const y = 0.2126 * img[i] + 0.7152 * img[i + 1] + 0.0722 * img[i + 2];
      sum += y;
      sumSq += y * y;
    }
    const mean = sum / px;
    const variance = sumSq / px - mean * mean;
    const std = Math.sqrt(Math.max(variance, 0));

    let lighting;
    if (mean < 55) lighting = { status: 'bad', message: 'Too dark — find brighter light' };
    else if (mean > 215) lighting = { status: 'bad', message: 'Too bright — reduce backlight' };
    else if (std > 70) lighting = { status: 'warn', message: 'Uneven light — face a soft light source' };
    else lighting = { status: 'good', message: 'Lighting looks good' };

    // Face detection via the browser's FaceDetector API when available.
    let face = { status: 'checking', message: 'Looking for your face…' };
    let position = { status: 'checking', message: 'Center your face in the oval' };
    let angle = { status: 'checking', message: 'Face the camera straight on' };
    let hint = 'Align your face inside the oval';

    if (detectorRef.current && detectorRef.current.detect) {
      try {
        const faces = await detectorRef.current.detect(analysisCanvas);
        if (!faces || faces.length === 0) {
          face = { status: 'bad', message: 'No face detected — face the camera' };
          position = { status: 'checking', message: 'Center your face in the oval' };
          angle = { status: 'checking', message: 'Face the camera straight on' };
          hint = 'Show your full face to the camera';
        } else {
          const f = faces[0].boundingBox;
          const fx = f.x + f.width / 2;
          const fy = f.y + f.height / 2;
          // Preview is mirrored horizontally, so we flip x-based directions
          // for on-screen advice ("left/right" reflect what the user sees).
          const dx = (fx - W / 2) / W;
          const dy = (fy - H / 2) / H;
          const rel = f.width / W;

          face = { status: 'good', message: 'Face detected' };

          // Framing: is the face touching / running off the frame edges?
          const margin = 0.03;
          const cutLeft = f.x < W * margin;
          const cutRight = f.x + f.width > W * (1 - margin);
          const cutTop = f.y < H * margin;
          const cutBottom = f.y + f.height > H * (1 - margin);
          const cropped = cutLeft || cutRight || cutTop || cutBottom;

          // Position guidance — distance and centering.
          if (rel < 0.28) {
            position = { status: 'warn', message: 'Move closer to the camera' };
            hint = 'Move closer — fill more of the oval';
          } else if (rel > 0.72 || cropped) {
            const parts = [];
            if (cutLeft) parts.push('right'); // mirrored preview
            if (cutRight) parts.push('left');
            if (cutTop) parts.push('down');
            if (cutBottom) parts.push('up');
            if (parts.length) {
              const dir = parts.join(' and ');
              position = { status: 'bad', message: `Face cut off — move ${dir}` };
              hint = `Move ${dir} so your whole face fits`;
            } else {
              position = { status: 'warn', message: 'Move a little farther back' };
              hint = 'Move back — face is too close';
            }
          } else if (Math.abs(dx) > 0.10 || Math.abs(dy) > 0.12) {
            const parts = [];
            if (dx > 0.10) parts.push('right'); // mirrored
            else if (dx < -0.10) parts.push('left');
            if (dy > 0.12) parts.push('up');
            else if (dy < -0.12) parts.push('down');
            const dir = parts.join(' and ');
            position = { status: 'warn', message: `Move ${dir}` };
            hint = `Move ${dir} to center your face`;
          } else {
            position = { status: 'good', message: 'Face is well positioned' };
          }

          // Angle guidance from landmarks (roll from eye line, yaw from eye
          // symmetry around the nose, pitch from eye vertical position).
          const lms = faces[0].landmarks || [];
          const eyes = lms
            .filter((l) => l.type === 'eye')
            .map((l) => (l.locations && l.locations[0]) || l);
          const nose = (lms.find((l) => l.type === 'nose') || {}).locations?.[0];
          const mouth = (lms.find((l) => l.type === 'mouth') || {}).locations?.[0];

          if (eyes.length === 2) {
            const [a, b] = eyes[0].x <= eyes[1].x ? [eyes[0], eyes[1]] : [eyes[1], eyes[0]];
            const rollDeg = (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI;

            const eyeMidX = (a.x + b.x) / 2;
            const eyeMidY = (a.y + b.y) / 2;
            const eyeSpan = Math.hypot(b.x - a.x, b.y - a.y) || 1;

            // Yaw: nose horizontal offset from midpoint between the eyes,
            // normalised by eye distance.
            let yaw = 0;
            if (nose) yaw = (nose.x - eyeMidX) / eyeSpan;

            // Pitch: where the eye line sits inside the face box.
            // Upright, forward-facing eyes sit around 40% down the box.
            const pitchRel = (eyeMidY - f.y) / f.height;

            if (Math.abs(rollDeg) > 8) {
              const dir = rollDeg > 0 ? 'right' : 'left'; // mirrored: user sees inverted tilt
              const userDir = rollDeg > 0 ? 'left' : 'right';
              angle = { status: 'warn', message: `Tilt your head ${userDir} to level` };
              hint = `Tilt your head ${userDir} — keep it level`;
              void dir;
            } else if (Math.abs(yaw) > 0.18) {
              // yaw > 0 → nose right of eye midpoint (in image coords).
              // Preview is mirrored, so user needs to turn the other way.
              const userDir = yaw > 0 ? 'right' : 'left';
              angle = { status: 'warn', message: `Turn your head slightly ${userDir}` };
              hint = `Turn slightly ${userDir} to face the camera`;
            } else if (pitchRel < 0.30) {
              angle = { status: 'warn', message: 'Lower your chin slightly' };
              hint = 'Lower your chin slightly';
            } else if (pitchRel > 0.50) {
              angle = { status: 'warn', message: 'Raise your chin slightly' };
              hint = 'Raise your chin slightly';
            } else {
              angle = { status: 'good', message: 'Head angle looks good' };
            }
          } else if (mouth && nose) {
            // Only one eye visible → strong yaw.
            angle = { status: 'warn', message: 'Face the camera — one eye is hidden' };
            hint = 'Face the camera — we can only see one eye';
          } else {
            angle = { status: 'checking', message: 'Face the camera straight on' };
          }

          if (
            position.status === 'good' &&
            angle.status === 'good' &&
            lighting.status === 'good'
          ) {
            hint = 'Hold still…';
          }
        }
      } catch {
        // FaceDetector may throw transiently; keep previous state.
      }
    } else {
      // No detector at all — capture is disabled below; still show status.
      face = { status: 'bad', message: 'Face detector unavailable' };
      position = { status: 'checking', message: '—' };
      angle = { status: 'checking', message: '—' };
      hint = 'Face detection is required. Please upload a photo instead.';
    }

    setChecks({ lighting, face, position, angle });
    setPrimaryHint(hint);

    // Face detection is REQUIRED — never auto-capture without a verified face.
    const allGood =
      !!detectorRef.current &&
      lighting.status === 'good' &&
      face.status === 'good' &&
      position.status === 'good' &&
      angle.status === 'good';

    const now = performance.now();
    if (allGood) {
      if (readySinceRef.current == null) readySinceRef.current = now;
      const held = now - readySinceRef.current;
      const secondsLeft = Math.max(0, Math.ceil((HOLD_STEADY_MS - held) / 400));
      setCountdown(secondsLeft > 0 ? secondsLeft : null);
      if (held >= HOLD_STEADY_MS && !autoCaptureRef.current) {
        autoCaptureRef.current = true;
        capture();
        return;
      }
    } else {
      readySinceRef.current = null;
      setCountdown(null);
    }

    rafRef.current = requestAnimationFrame(runAnalysis);
  }, [capture]);

  useEffect(() => {
    let cancelled = false;

    async function start() {
      try {
        const detector = await initFaceDetector();
        if (cancelled) return;
        detectorRef.current = detector;
        setDetectorState(detector ? 'ready' : 'unavailable');

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 1280 },
            height: { ideal: 960 },
          },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
        setReady(true);
        rafRef.current = requestAnimationFrame(runAnalysis);
      } catch (e) {
        if (e && e.name === 'NotAllowedError') {
          setError('Camera permission denied. Please allow camera access and try again.');
        } else if (e && e.name === 'NotFoundError') {
          setError('No camera found on this device.');
        } else {
          setError('Could not open the camera. ' + (e?.message || ''));
        }
      }
    }

    start();
    return () => {
      cancelled = true;
      stopStream();
    };
  }, [runAnalysis, stopStream]);

  const handleCancel = () => {
    stopStream();
    onCancel();
  };

  const detectorAvailable = detectorState === 'ready';
  const requiredChecks = ['lighting', 'face', 'position', 'angle'];
  const goodCount = detectorAvailable
    ? requiredChecks.filter((k) => checks[k].status === 'good').length
    : 0;
  const totalCount = requiredChecks.length;
  const overallReady = detectorAvailable && goodCount === totalCount;

  const handleManualCapture = () => {
    if (autoCaptureRef.current) return;
    if (!overallReady) return;
    autoCaptureRef.current = true;
    capture();
  };

  return (
    <div className="camera-capture">
      {error ? (
        <div className="status-message status-error">{error}</div>
      ) : (
        <>
          <div className="camera-stage">
            <video
              ref={videoRef}
              className="camera-video"
              autoPlay
              muted
              playsInline
            />
            <div className={`camera-oval ${overallReady ? 'oval-ready' : ''}`} />
            <div className="camera-crosshair" aria-hidden="true">
              <span className="ch-h" />
              <span className="ch-v" />
            </div>
            {countdown != null && <div className="camera-countdown">{countdown}</div>}
            {!ready && <div className="camera-loading">Starting camera…</div>}
            {ready && (
              <div className="camera-hint-banner" role="status">
                {primaryHint}
              </div>
            )}
          </div>

          <div className="camera-progress">
            <div className="camera-progress-header">
              <span>Ready to capture</span>
              <strong>{goodCount} / {totalCount}</strong>
            </div>
            <div className="camera-progress-bar">
              <div
                className={`camera-progress-fill ${overallReady ? 'is-ready' : ''}`}
                style={{ width: `${(goodCount / totalCount) * 100}%` }}
              />
            </div>
          </div>

          {detectorState === 'loading' && (
            <div className="status-message" style={{ background: 'rgba(66,153,225,0.12)', color: 'var(--info)', marginTop: '12px' }}>
              Loading face detector…
            </div>
          )}
          {detectorState === 'unavailable' && (
            <div className="status-message status-error" style={{ marginTop: '12px' }}>
              Could not load the face detector. Please check your network and reload,
              or use "Choose file" instead.
            </div>
          )}

          <ul className="camera-checks">
            <CheckRow label="Lighting" check={checks.lighting} />
            <CheckRow label="Face" check={checks.face} />
            <CheckRow label="Position" check={checks.position} />
            <CheckRow label="Angle" check={checks.angle} />
          </ul>

          <div style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
            <button
              className={`btn ${overallReady ? 'btn-primary' : 'btn-outline'}`}
              style={{ flex: 1 }}
              onClick={handleManualCapture}
              disabled={!ready || !overallReady}
              title={overallReady ? 'Capture now' : 'Fix the checks above to enable capture'}
            >
              {overallReady ? '📸 Capture now' : `Waiting — ${totalCount - goodCount} to fix`}
            </button>
            <button className="btn btn-outline" onClick={handleCancel}>
              Cancel
            </button>
          </div>

          {!('FaceDetector' in window) && (
            <p className="camera-hint">
              Tip: your browser doesn't support automatic face detection. Align your face
              inside the oval and tap Capture when ready.
            </p>
          )}
        </>
      )}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <canvas ref={analysisCanvasRef} style={{ display: 'none' }} />
    </div>
  );
}

function CheckRow({ label, check }) {
  const icon =
    check.status === 'good' ? '✅' : check.status === 'warn' ? '⚠️' : check.status === 'bad' ? '❌' : '⏳';
  return (
    <li className={`camera-check camera-check-${check.status}`}>
      <span className="camera-check-icon">{icon}</span>
      <span className="camera-check-label">{label}:</span>
      <span className="camera-check-msg">{check.message}</span>
    </li>
  );
}

export default CameraCapture;
