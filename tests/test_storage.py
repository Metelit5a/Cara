"""Tests for the storage repository."""

import pytest
import tempfile
import os
from shared.schemas import AnalysisReport, AnalysisStatus, AcneSeverity


class TestJsonStorageRepository:
    """Test JSON file-based storage."""

    @pytest.fixture
    def repo(self):
        from backend.database.repository import JsonStorageRepository
        with tempfile.TemporaryDirectory() as tmpdir:
            yield JsonStorageRepository(storage_path=tmpdir)

    @pytest.fixture
    def sample_report(self):
        return AnalysisReport(
            id="test-123",
            status=AnalysisStatus.SUCCESS,
            acne_severity=AcneSeverity.MILD,
            confidence=0.85,
            explanation="Test explanation",
            recommendations=[],
            educational_note="Test note",
        )

    async def test_save_and_retrieve(self, repo, sample_report):
        await repo.save_report(sample_report)
        retrieved = await repo.get_report("test-123")

        assert retrieved is not None
        assert retrieved.id == "test-123"
        assert retrieved.acne_severity == AcneSeverity.MILD
        assert retrieved.confidence == 0.85

    async def test_get_nonexistent(self, repo):
        result = await repo.get_report("does-not-exist")
        assert result is None

    async def test_list_reports(self, repo, sample_report):
        await repo.save_report(sample_report)
        reports = await repo.list_reports()
        assert len(reports) == 1
        assert reports[0].id == "test-123"

    async def test_list_empty(self, repo):
        reports = await repo.list_reports()
        assert len(reports) == 0
