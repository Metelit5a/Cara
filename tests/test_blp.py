"""Tests for the BLP engine and report generation."""

import pytest
from shared.schemas import ModelPrediction, AcneSeverity, PoreSeverity


class TestBLPEngine:
    """Test business logic processing."""

    def test_process_clear(self):
        from backend.blp.engine import BLPEngine

        engine = BLPEngine()
        prediction = ModelPrediction(
            model_name="acne_severity",
            predicted_class=0,
            predicted_label="clear",
            confidence=0.92,
            all_probabilities=[0.92, 0.05, 0.02, 0.01],
        )
        result = engine.process({"acne": prediction})

        assert result.acne_severity == AcneSeverity.CLEAR
        assert len(result.recommendations) > 0
        assert "clear" in result.explanation.lower()

    def test_process_severe(self):
        from backend.blp.engine import BLPEngine

        engine = BLPEngine()
        prediction = ModelPrediction(
            model_name="acne_severity",
            predicted_class=3,
            predicted_label="severe",
            confidence=0.85,
            all_probabilities=[0.02, 0.05, 0.08, 0.85],
        )
        result = engine.process({"acne": prediction})

        assert result.acne_severity == AcneSeverity.SEVERE
        assert len(result.recommendations) > 0
        assert "dermatologist" in result.educational_note.lower()

    def test_all_severities_have_rules(self):
        from backend.blp.engine import BLPEngine

        engine = BLPEngine()
        for severity in ["clear", "mild", "moderate", "severe"]:
            prediction = ModelPrediction(
                model_name="acne_severity",
                predicted_class=0,
                predicted_label=severity,
                confidence=0.8,
                all_probabilities=[0.8, 0.1, 0.05, 0.05],
            )
            result = engine.process({"acne": prediction})
            assert result.acne_severity.value == severity
            assert len(result.recommendations) >= 1
            assert result.explanation
            assert result.educational_note

    def test_process_pores(self):
        from backend.blp.engine import BLPEngine

        engine = BLPEngine()
        prediction = ModelPrediction(
            model_name="pores_detection",
            predicted_class=2,
            predicted_label="moderate",
            confidence=0.75,
            all_probabilities=[0.0, 0.0, 0.75, 0.25],
        )
        result = engine.process({"pores": prediction})

        assert result.pore_severity == PoreSeverity.MODERATE
        assert len(result.recommendations) > 0

    def test_process_both_models(self):
        from backend.blp.engine import BLPEngine

        engine = BLPEngine()
        acne_pred = ModelPrediction(
            model_name="acne_severity",
            predicted_class=1,
            predicted_label="mild",
            confidence=0.8,
            all_probabilities=[0.1, 0.8, 0.05, 0.05],
        )
        pore_pred = ModelPrediction(
            model_name="pores_detection",
            predicted_class=1,
            predicted_label="mild",
            confidence=0.7,
            all_probabilities=[0.0, 0.7, 0.2, 0.1],
        )
        result = engine.process({"acne": acne_pred, "pores": pore_pred})

        assert result.acne_severity == AcneSeverity.MILD
        assert result.pore_severity == PoreSeverity.MILD
        assert len(result.recommendations) > 0
        # Should have merged recommendations from both
        assert result.explanation
        assert result.educational_note

    def test_low_confidence_message(self):
        from backend.blp.engine import BLPEngine

        engine = BLPEngine()
        assert engine.low_confidence_message
        assert engine.confidence_threshold > 0


class TestReportBuilder:
    """Test report generation."""

    def test_success_report(self):
        from backend.report_generation.builder import ReportBuilder
        from shared.schemas import BLPResult, Recommendation

        prediction = ModelPrediction(
            model_name="acne_severity",
            predicted_class=1,
            predicted_label="mild",
            confidence=0.78,
            all_probabilities=[0.1, 0.78, 0.1, 0.02],
        )
        blp_result = BLPResult(
            acne_severity=AcneSeverity.MILD,
            recommendations=[Recommendation(ingredient="Test", reason="Test reason", category="test")],
            explanation="Test explanation",
            educational_note="Test note",
        )
        report = ReportBuilder.build_success_report({"acne": prediction}, blp_result)

        assert report.status.value == "success"
        assert report.acne_severity == AcneSeverity.MILD
        assert report.confidence == 0.78
        assert len(report.recommendations) == 1
        assert report.id

    def test_no_face_report(self):
        from backend.report_generation.builder import ReportBuilder

        report = ReportBuilder.build_no_face_report("No face detected")
        assert report.status.value == "no_face_detected"
        assert report.message == "No face detected"
        assert report.acne_severity is None

    def test_error_report(self):
        from backend.report_generation.builder import ReportBuilder

        report = ReportBuilder.build_error_report("Something went wrong")
        assert report.status.value == "error"
        assert report.message == "Something went wrong"
