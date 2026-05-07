# Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│   Home │ Analyze │ Results │ History                      │
└────────────────────┬────────────────────────────────────┘
                     │ REST API
┌────────────────────▼────────────────────────────────────┐
│              FastAPI Backend Gateway                     │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────────┐ │
│   │ API      │  │ Analysis     │  │ Database         │ │
│   │ Routes   │→ │ Service      │  │ Repository       │ │
│   └──────────┘  └──────┬───────┘  └──────────────────┘ │
└─────────────────────────┼───────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ Preprocessing│ │  Inference   │ │  BLP Engine       │
│ Pipeline     │ │ Orchestrator │ │  (Rule-based)     │
│              │ │              │ │                    │
│ Face Detect  │ │ Model        │ │ rules.json        │
│ Crop/Resize  │ │ Registry     │ │ → Recommendations │
│ Normalize    │ │              │ │ → Explanations     │
└──────────────┘ └──────┬───────┘ └──────────────────┘
                        │
              ┌─────────▼─────────┐
              │  Acne Severity    │
              │  Model            │
              │  (EfficientNetB0) │
              └───────────────────┘
```

## Module Responsibilities

### Frontend
- React SPA with mobile-first responsive design
- Image upload/camera capture
- Results visualization
- History browsing

### Backend Gateway (FastAPI)
- REST API endpoints
- Request validation (Pydantic)
- CORS handling
- Routes to Analysis Service

### Analysis Service
Orchestrates the pipeline:
1. Receives image bytes
2. Calls preprocessing
3. Calls inference orchestrator
4. Checks confidence threshold
5. Calls BLP engine
6. Builds report
7. Persists to storage

### Preprocessing Pipeline
- MediaPipe face detection
- Face cropping with padding
- CLAHE brightness normalization
- Resize to 224×224
- ImageNet tensor normalization

### Inference Orchestrator
- Model Registry pattern for plug-in models
- Loads weights once at startup
- Runs prediction with confidence scoring
- Returns structured `ModelPrediction`

### BLP Engine
- Data-driven rules from `rules.json`
- Maps severity → recommendations
- Generates explanations
- Confidence-level context

### Database Repository
- Interface: `StorageRepository` (abstract)
- POC: `JsonStorageRepository` (JSON files)
- Future: `MongoStorageRepository` (prepared)
- Seamless switching via `STORAGE_BACKEND` env var

## Adding a New Model

1. Create `model_service/new_model/model.py` with the model class
2. Add a `register_*` method to `ModelRegistry`
3. Add a `predict_*` method to `InferenceOrchestrator`
4. Add rules to `backend/blp/rules.json`
5. Update `AnalysisService` to call the new model
6. No frontend changes needed if results follow the same schema
