"""Tests for the API endpoints."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    client.post(
        "/api/v1/auth/register",
        json={
            "username": "authuser",
            "email": "auth@example.com",
            "password": "secret123",
        },
    )
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "auth@example.com", "password": "secret123"},
    )
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_jpeg_bytes():
    """Generate a valid JPEG image for testing."""
    img = Image.new("RGB", (300, 300), color=(200, 150, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_user_store():
    users_file = Path("storage/users.json")
    users_file.parent.mkdir(parents=True, exist_ok=True)
    users_file.write_text("[]", encoding="utf-8")

    reports_dir = Path("storage/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    for report_file in reports_dir.glob("*.json"):
        report_file.unlink()

    yield
    users_file.write_text("[]", encoding="utf-8")
    for report_file in reports_dir.glob("*.json"):
        report_file.unlink()


@pytest.fixture
def sample_png_bytes():
    """Generate a valid PNG image."""
    img = Image.new("RGB", (300, 300), color=(200, 150, 130))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestAnalyzeEndpoint:
    def test_analyze_accepts_jpeg(self, client, sample_jpeg_bytes, auth_headers):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("face.jpg", sample_jpeg_bytes, "image/jpeg")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "report" in data
        assert "id" in data["report"]
        assert "status" in data["report"]

    def test_analyze_accepts_png(self, client, sample_png_bytes, auth_headers):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("face.png", sample_png_bytes, "image/png")},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_analyze_rejects_invalid_content_type(self, client, auth_headers):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("file.txt", b"not an image", "text/plain")},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_analyze_rejects_empty_file(self, client, auth_headers):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("face.jpg", b"", "image/jpeg")},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    def test_analyze_report_has_correct_structure(self, client, sample_jpeg_bytes, auth_headers):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("face.jpg", sample_jpeg_bytes, "image/jpeg")},
            headers=auth_headers,
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


class TestAuthEndpoints:
    def test_register_creates_user(self, client):
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "secret123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "User registered successfully"

    def test_login_returns_access_token(self, client):
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "secret123",
            },
        )

        response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "secret123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["token_type"] == "bearer"
        assert "access_token" in data


class TestReportEndpoints:
    def test_get_nonexistent_report_returns_404(self, client, auth_headers):
        response = client.get("/api/v1/report/nonexistent-id", headers=auth_headers)
        assert response.status_code == 404

    def test_list_reports_returns_list(self, client, auth_headers):
        response = client.get("/api/v1/reports", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_reports_filters_by_authenticated_user(self, client):
        first_user = client.post(
            "/api/v1/auth/register",
            json={
                "username": "first",
                "email": "first@example.com",
                "password": "secret123",
            },
        )
        second_user = client.post(
            "/api/v1/auth/register",
            json={
                "username": "second",
                "email": "second@example.com",
                "password": "secret123",
            },
        )

        first_login = client.post(
            "/api/v1/auth/login",
            json={"email": "first@example.com", "password": "secret123"},
        )
        second_login = client.post(
            "/api/v1/auth/login",
            json={"email": "second@example.com", "password": "secret123"},
        )

        assert first_user.status_code == 200
        assert second_user.status_code == 200
        assert first_login.status_code == 200
        assert second_login.status_code == 200

        first_token = first_login.json()["access_token"]
        second_token = second_login.json()["access_token"]

        response_one = client.post(
            "/api/v1/analyze",
            files={"file": ("face.jpg", b"fake-image-bytes", "image/jpeg")},
            headers={"Authorization": f"Bearer {first_token}"},
        )
        response_two = client.post(
            "/api/v1/analyze",
            files={"file": ("face.jpg", b"fake-image-bytes", "image/jpeg")},
            headers={"Authorization": f"Bearer {second_token}"},
        )

        assert response_one.status_code == 200
        assert response_two.status_code == 200

        filtered_response = client.get(
            "/api/v1/reports",
            headers={"Authorization": f"Bearer {first_token}"},
        )

        assert filtered_response.status_code == 200
        data = filtered_response.json()
        assert len(data) == 1
        assert data[0]["user_id"] == "1"
