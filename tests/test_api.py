"""Tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_jpeg_bytes():
    """Generate a valid JPEG image for testing."""
    img = Image.new("RGB", (300, 300), color=(200, 150, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes():
    """Generate a valid PNG image."""
    img = Image.new("RGB", (300, 300), color=(200, 150, 130))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestAnalyzeEndpoint:
    def test_analyze_accepts_jpeg(self, client, sample_jpeg_bytes):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("face.jpg", sample_jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "report" in data
        assert "id" in data["report"]
        assert "status" in data["report"]

    def test_analyze_accepts_png(self, client, sample_png_bytes):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("face.png", sample_png_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_analyze_rejects_invalid_content_type(self, client):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("file.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_analyze_rejects_empty_file(self, client):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("face.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    def test_analyze_report_has_correct_structure(self, client, sample_jpeg_bytes):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("face.jpg", sample_jpeg_bytes, "image/jpeg")},
        )
        report = response.json()["report"]
        assert "id" in report
        assert "status" in report
        assert "created_at" in report
        assert report["status"] in ["success", "low_confidence", "error", "no_face_detected"]


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "version" in data


class TestReportEndpoints:
    def test_get_nonexistent_report_returns_404(self, client):
        response = client.get("/api/v1/report/nonexistent-id")
        assert response.status_code == 404

    def test_list_reports_returns_list(self, client):
        response = client.get("/api/v1/reports")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
