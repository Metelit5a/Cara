<p align="center">
  <h1 align="center">CARA — Computer-Aided Review & Analysis</h1>
  <p align="center">
    AI-powered skincare analysis assistant using deep learning and computer vision
    <br />
    <a href="docs/ARCHITECTURE.md"><strong>Architecture »</strong></a>
    &nbsp;&middot;&nbsp;
    <a href="docs/API.md"><strong>API Docs »</strong></a>
  </p>
</p>

---

## About

**CARA** is a production-grade web application that analyzes facial skin images and provides a comprehensive skin report with personalized, evidence-based skincare recommendations.

A user uploads a photo, the system runs 3 deep learning models in parallel (acne severity, skin type, skin issues), combines their outputs through a rule-based engine, and returns an actionable report — all within seconds.

> **Disclaimer:** CARA is **not** a medical diagnostic tool. It provides cosmetic skincare insights and educational ingredient recommendations only.

### Key Features

| Feature | Description |
|---------|-------------|
| **Acne Severity Classification** | EfficientNetB0 — clear/mild/moderate/severe (69.6% test accuracy) |
| **Skin Type Detection** | EfficientNetB0 — oily/dry/normal/combination (77.8% test accuracy) |
| **Skin Issues Detection** | EfficientNetB0 — blackheads/dark spots/pores/wrinkles/healthy (98.3% test accuracy) |
| **Recommendation Engine** | Rule-based BLP engine mapping predictions to evidence-based skincare advice |
| **Report Persistence** | Full analysis history with JSON storage (MongoDB-ready) |
| **Responsive UI** | Mobile-first React frontend with image upload, results, and history pages |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      React Frontend                          │
│           Upload Image  ·  View Results  ·  History          │
└────────────────────────────┬─────────────────────────────────┘
                             │  REST API (JSON)
┌────────────────────────────▼─────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌──────────┐  ┌───────────────┐  ┌────────────────────────┐ │
│  │  Routes   │→│ Analysis Svc  │→│  Report Builder         │ │
│  └──────────┘  └──────┬────────┘  └────────────────────────┘ │
│                       │                                      │
│         ┌─────────────▼──────────────┐                       │
│         │  Inference Orchestrator    │                        │
│         │  ┌───────────────────────┐ │                       │
│         │  │ Preprocessing Pipeline│ │                       │
│         │  │  Face Detect → CLAHE  │ │                       │
│         │  └───────────┬───────────┘ │                       │
│         │  ┌───────────▼───────────┐ │                       │
│         │  │ EfficientNet-B0 Model │ │                       │
│         │  │ 4-class classifier    │ │                       │
│         │  └───────────────────────┘ │                       │
│         └────────────────────────────┘                       │
│                       │                                      │
│         ┌─────────────▼──────────────┐                       │
│         │   BLP Rule Engine          │                        │
│         │   Severity → Advice        │                        │
│         └────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, React Router 6 |
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Deep Learning | PyTorch, torchvision, EfficientNet-B0 |
| Computer Vision | OpenCV (Haar Cascade, CLAHE) |
| Dataset | [ACNE04](https://www.kaggle.com/datasets/manuelhettich/acne04) (1 393 images, 4 classes) |
| Storage | JSON file storage (MongoDB-ready via Motor) |
| DevOps | Docker, Docker Compose, Nginx |
| Testing | pytest, pytest-asyncio, httpx |

---

## Project Structure

```
cara/
├── backend/                    # FastAPI application
│   ├── api/                    #   REST API routes
│   ├── services/               #   Business services & orchestration
│   ├── blp/                    #   Rule-based recommendation engine
│   ├── database/               #   Storage abstraction layer
│   └── report_generation/      #   Report builder
│
├── model_service/              # ML pipeline
│   ├── acne_model/             #   EfficientNet-B0 model definition
│   ├── preprocessing/          #   Face detection & image pipeline
│   ├── inference/              #   Model registry & prediction
│   ├── training/               #   Training & dataset scripts
│   └── checkpoints/            #   Saved model weights (.gitignored)
│
├── frontend/                   # React SPA
│   └── src/
│       └── pages/              #   Home, Analyze, Results, History
│
├── shared/                     # Shared config & Pydantic schemas
├── storage/                    # JSON report persistence
├── tests/                      # Full test suite
└── docs/                       # Architecture & API documentation
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** & npm
- **NVIDIA GPU** with CUDA (recommended for training; CPU works for inference)
- **Kaggle account** (for dataset download — `kagglehub` handles authentication)

### 1. Clone the Repository

```bash
git clone https://github.com/Metelit5a/Cara.git
cd Cara
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS

# Install Python dependencies
pip install -r requirements.txt

# For GPU training, install CUDA-enabled PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy environment config
copy .env.example .env           # Windows
# cp .env.example .env           # Linux / macOS
```

### 3. Train the Models (or use pretrained weights)

```bash
# Train all 3 models from scratch
python -m model_service.training.train --model all

# Or train individually
python -m model_service.training.train --model acne
python -m model_service.training.train --model skin_type
python -m model_service.training.train --model skin_issues

# Resume training from checkpoint (e.g., on GPU)
python -m model_service.training.train --model acne --resume --epochs 30
```

Trained models are saved to `model_service/checkpoints/`.
See `model_service/training/README.md` for full training documentation.

---

## Testing

> **⚠️ IMPORTANT: Run all tests before pushing to main.**
>
> The test suite includes algorithm accuracy tests that catch model regressions.
> A passing test suite means the pipeline produces correct predictions.

```bash
# Run the full test suite (56 tests)
python -m pytest tests/ -v

# Quick check (just logic tests, no model inference)
python -m pytest tests/test_blp.py tests/test_api.py -v

# Model accuracy tests only (requires checkpoints + data)
python -m pytest tests/test_model_accuracy.py -v

# End-to-end pipeline tests
python -m pytest tests/test_e2e_pipeline.py -v
```

### Test Categories

| File | Tests | What it verifies |
|------|:-----:|------------------|
| `test_blp.py` | 18 | BLP rule engine logic, all rules reachable |
| `test_api.py` | 4 | API endpoints, request/response format |
| `test_integration.py` | 8 | Component connections, report building |
| `test_model_accuracy.py` | 11 | Per-model predictions on ground truth images |
| `test_e2e_pipeline.py` | 10 | Full image→report pipeline with real images |

If any `test_model_accuracy` or `test_e2e_pipeline` test fails after a code change,
**you have introduced a regression** — do not push until fixed.

### 4. Start the Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Frontend Setup

```bash
cd frontend
npm install
npm start
```

The app opens at **http://localhost:3000** and proxies API calls to port 8000.

### 6. Docker (Alternative)

```bash
docker-compose up --build
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze` | Upload a face image → receive a full analysis report |
| `GET` | `/api/v1/report/{id}` | Retrieve a previously generated report by ID |
| `GET` | `/api/v1/reports` | List all saved reports |
| `GET` | `/health` | Service health check (model status, version) |

### Example: Analyze an Image

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@photo.jpg"
```

For detailed request/response schemas, see [docs/API.md](docs/API.md).

---

## Model Details

### Architecture

- **Backbone:** EfficientNet-B0 pretrained on ImageNet
- **Head:** Custom classifier — `AdaptiveAvgPool → Dropout(0.3) → Linear(1280, 512) → ReLU → Dropout(0.2) → Linear(512, 4)`

### Training Pipeline

| Phase | Epochs | Learning Rate | Description |
|-------|--------|---------------|-------------|
| Phase 1 — Frozen Backbone | 10 | 1e-3 | Train classification head only |
| Phase 2 — Fine-Tuning | 5 | 1e-4 | Unfreeze upper backbone layers |

### Preprocessing Pipeline

1. **Face Detection** — OpenCV Haar Cascade with 20% bounding-box padding
2. **Crop** — Extract face region
3. **CLAHE** — Contrast-limited adaptive histogram equalization for lighting normalization
4. **Resize** — 224 × 224 pixels
5. **Normalize** — ImageNet mean/std normalization

### Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | ~73% |
| Test Accuracy | ~70% |
| Classes | Clear, Mild, Moderate, Severe |

---

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run by category
pytest tests/test_api.py -v              # API endpoints
pytest tests/test_preprocessing.py -v    # Image pipeline
pytest tests/test_inference.py -v        # Model inference
pytest tests/test_blp.py -v              # Recommendation engine
pytest tests/test_storage.py -v          # Storage layer
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_HOST` | `0.0.0.0` | Server bind address |
| `BACKEND_PORT` | `8000` | Server port |
| `DEBUG` | `true` | Debug mode |
| `MODEL_WEIGHTS_PATH` | `model_service/checkpoints/acne_model_best.pth` | Path to trained model |
| `CONFIDENCE_THRESHOLD` | `0.4` | Minimum confidence for predictions |
| `STORAGE_BACKEND` | `json` | Storage type (`json` or `mongodb`) |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |

See [.env.example](.env.example) for the full template.

---

## Future Roadmap

- [ ] Additional skin analysis models (redness, wrinkles, pores, pigmentation)
- [ ] MongoDB integration for production persistence
- [ ] User authentication & personal history
- [ ] Progressive Web App (PWA) support
- [ ] Expanded dataset with more diverse skin tones
- [ ] Model interpretability with Grad-CAM visualizations

---

## License

This project was developed as a final-year academic project. All rights reserved.

---

<p align="center">
  Built with PyTorch · FastAPI · React
</p>

## Future Extensibility
The architecture supports plugging in additional models:
- Skin type classification
- Redness/inflammation detection
- Wrinkle estimation
- Pore visibility estimation
- Pigmentation analysis

Each model is an independent service behind the inference orchestrator.

## License
Proprietary - Academic Project
