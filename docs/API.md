# Cara API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
No authentication required for the POC.

---

## Endpoints

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0"
}
```

---

### Analyze Image
```
POST /api/v1/analyze
Content-Type: multipart/form-data
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file  | File | Yes      | Face image (JPEG, PNG, WebP, BMP). Max 10MB. |

**Success Response (200):**
```json
{
  "report": {
    "id": "uuid-string",
    "created_at": "2024-01-01T00:00:00",
    "status": "success",
    "severity": "mild",
    "confidence": 0.85,
    "explanation": "Mild acne has been detected...",
    "recommendations": [
      {
        "ingredient": "Salicylic Acid (BHA) 0.5-2%",
        "reason": "A beta-hydroxy acid that penetrates into pores...",
        "category": "exfoliation"
      }
    ],
    "educational_note": "Mild acne is very common..."
  }
}
```

**Possible Status Values:**
| Status | Description |
|--------|-------------|
| `success` | Analysis completed successfully |
| `low_confidence` | Model confidence below threshold |
| `no_face_detected` | No face found in uploaded image |
| `error` | Processing error |

**Error Responses:**
- `400` - Invalid file type, empty file, or file too large
- `422` - Missing required file parameter

---

### Get Report
```
GET /api/v1/report/{report_id}
```

**Response (200):** Same `AnalysisReport` schema as above.

**Error Responses:**
- `404` - Report not found

---

### List Reports
```
GET /api/v1/reports?limit=50
```

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| limit | int  | 50      | Max reports to return (max 100) |

**Response (200):** Array of `AnalysisReport` objects.

---

## Schemas

### AcneSeverity (enum)
`"clear"` | `"mild"` | `"moderate"` | `"severe"`

### Recommendation
```json
{
  "ingredient": "string",
  "reason": "string",
  "category": "string"
}
```

### AnalysisReport
```json
{
  "id": "string (UUID)",
  "created_at": "datetime",
  "status": "success | low_confidence | no_face_detected | error",
  "severity": "clear | mild | moderate | severe | null",
  "confidence": "float 0-1 | null",
  "explanation": "string | null",
  "recommendations": "Recommendation[] | []",
  "educational_note": "string | null",
  "message": "string | null"
}
```
