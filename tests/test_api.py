"""Tests for the FastAPI endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image
import numpy as np
import io


@pytest.fixture
def test_app():
    from backend.main import app
    return app


@pytest.fixture
def sample_jpeg():
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


async def test_health_endpoint(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "version" in data


async def test_analyze_no_file(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/analyze")
        assert response.status_code == 422  # Missing required file


async def test_analyze_wrong_content_type(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analyze",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400


async def test_analyze_valid_image(test_app, sample_jpeg):
    """A valid JPEG should get a response (may be no_face_detected for random noise)."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analyze",
            files={"file": ("test.jpg", sample_jpeg, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "report" in data
        assert "id" in data["report"]
        assert "status" in data["report"]


async def test_report_not_found(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/report/nonexistent-id")
        assert response.status_code == 404


async def test_list_reports(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/reports")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
