"""
Real-World Regression Tests — the tests we WISH we had before.

These tests exercise the models on the same kind of images real users
upload (well-lit adult portraits/selfies), not on training data.

Why this file exists:
  `test_model_accuracy.py` pulls samples straight from the training
  folders and just verifies the model remembers them. It passes even
  when the model is catastrophically overfit and never predicts some
  classes on real inputs (the bug that motivated this suite).

Three layers of protection:
  1. Per-image assertions against hand-labelled ground truth in
     `tests/fixtures/real_world_labels.json`. `null` fields are skipped.
  2. Population-level sanity checks over the whole real-world folder
     (catches class-collapse — e.g. "always predicts wrinkles").
  3. Multi-label specific assertions on the skin_conditions model
     (empty findings on clean faces, per-condition thresholds).

If the skin_conditions checkpoint isn't present yet (e.g. before the
first multi-label training run) the multi-label assertions skip cleanly.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pytest


# ── Locations ────────────────────────────────────────────────────────────────
REAL_WORLD_DIR = Path("image tests people")
FIXTURE_PATH = Path("tests/fixtures/real_world_labels.json")
CHECKPOINT_DIR = Path("model_service/checkpoints")

ACNE_CKPT = CHECKPOINT_DIR / "acne_model_best.pth"
SKIN_TYPE_CKPT = CHECKPOINT_DIR / "skin_type_model_best.pth"
SKIN_CONDITIONS_CKPT = CHECKPOINT_DIR / "skin_conditions_model_best.pth"

requires_core_checkpoints = pytest.mark.skipif(
    not ACNE_CKPT.exists() or not SKIN_TYPE_CKPT.exists(),
    reason="Acne or skin_type checkpoint missing",
)
requires_multi_label = pytest.mark.skipif(
    not SKIN_CONDITIONS_CKPT.exists(),
    reason="skin_conditions checkpoint not trained yet",
)
requires_real_images = pytest.mark.skipif(
    not REAL_WORLD_DIR.exists() or not any(REAL_WORLD_DIR.glob("*.jpg")),
    reason="No real-world test images in `image tests people/`",
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _orchestrator():
    from model_service.inference.orchestrator import InferenceOrchestrator

    return InferenceOrchestrator()


def _pipeline():
    """Face-crop preprocessing pipeline — same one the app uses."""
    from model_service.preprocessing.pipeline import PreprocessingPipeline

    return PreprocessingPipeline(require_face=True, apply_mask=False)


def _predict(orchestrator, pipeline, path: Path) -> dict:
    """Run the SAME inference path the app uses: face-crop → model.

    Returns {} if face detection fails.
    """
    with path.open("rb") as f:
        image_bytes = f.read()
    pre, tensor = pipeline.process(image_bytes)
    if not pre.success:
        return {}
    return orchestrator.predict_all(tensor)


def _load_fixture() -> Dict[str, dict]:
    if not FIXTURE_PATH.exists():
        return {}
    with FIXTURE_PATH.open() as f:
        data = json.load(f)
    return data.get("images", {})


def _real_images() -> List[Path]:
    if not REAL_WORLD_DIR.exists():
        return []
    return sorted(
        p for p in REAL_WORLD_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _finding_labels(multi_label_pred) -> List[str]:
    """Extract the string labels from a `MultiLabelPrediction`."""
    from shared.schemas import MultiLabelPrediction

    if not isinstance(multi_label_pred, MultiLabelPrediction):
        return []
    return [f.label.value for f in multi_label_pred.findings]


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 — per-image ground-truth assertions
# ═══════════════════════════════════════════════════════════════════════════════


@requires_core_checkpoints
@requires_real_images
class TestLabelledRealWorldImages:
    """Assert predictions match human labels for real-world reference photos.

    A field set to `null` in the fixture is skipped (so partially-labelled
    images are OK).
    """

    @pytest.fixture(autouse=True)
    def _bind(self):
        # Fresh orchestrator + pipeline per test method for isolation.
        self.orchestrator = _orchestrator()
        self.pipeline = _pipeline()

    def _get_pred(self, filename: str, model_key: str):
        img_path = REAL_WORLD_DIR / filename
        if not img_path.exists():
            pytest.skip(f"{img_path} not in workspace")
        preds = _predict(self.orchestrator, self.pipeline, img_path)
        if not preds:
            pytest.skip(f"Face not detected in {filename}")
        pred = preds.get(model_key)
        if pred is None:
            pytest.skip(f"Model `{model_key}` not loaded")
        return pred

    @pytest.mark.parametrize("filename, expected", list(_load_fixture().items()))
    def test_acne_matches_label(self, filename, expected):
        if expected.get("acne") is None:
            pytest.skip("acne label not provided")
        pred = self._get_pred(filename, "acne")
        assert pred.predicted_label == expected["acne"], (
            f"[{filename}] acne: expected `{expected['acne']}`, "
            f"got `{pred.predicted_label}` (conf={pred.confidence:.2f}). "
            f"Notes: {expected.get('notes', '')}"
        )

    @pytest.mark.parametrize("filename, expected", list(_load_fixture().items()))
    def test_skin_type_matches_label(self, filename, expected):
        if expected.get("skin_type") is None:
            pytest.skip("skin_type label not provided")
        pred = self._get_pred(filename, "skin_type")
        assert pred.predicted_label == expected["skin_type"], (
            f"[{filename}] skin_type: expected `{expected['skin_type']}`, "
            f"got `{pred.predicted_label}` (conf={pred.confidence:.2f}). "
            f"Notes: {expected.get('notes', '')}"
        )

    @requires_multi_label
    @pytest.mark.parametrize("filename, expected", list(_load_fixture().items()))
    def test_skin_conditions_matches_label(self, filename, expected):
        """Multi-label: findings on the photo must equal the expected set.

        A missing `skin_conditions` key skips. An explicit `[]` means the
        model MUST report no findings — the "clean skin" case that used
        to fail because the old model always hallucinated a class.
        """
        if "skin_conditions" not in expected or expected["skin_conditions"] is None:
            pytest.skip("skin_conditions label not provided")
        expected_set = set(expected["skin_conditions"])
        pred = self._get_pred(filename, "skin_conditions")
        got_set = set(_finding_labels(pred))
        assert got_set == expected_set, (
            f"[{filename}] skin_conditions: expected "
            f"{sorted(expected_set) or 'no findings'}, "
            f"got {sorted(got_set) or 'no findings'}. "
            f"Raw scores: {getattr(pred, 'all_scores', {})}. "
            f"Notes: {expected.get('notes', '')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — population sanity checks (catch class collapse)
# ═══════════════════════════════════════════════════════════════════════════════


@requires_core_checkpoints
@requires_real_images
class TestRealWorldPopulationSanity:
    """Population-level checks over ALL real-world images."""

    @pytest.fixture(autouse=True, scope="class")
    def _distribution(self, request):
        from shared.schemas import MultiLabelPrediction, ModelPrediction

        orch = _orchestrator()
        pipeline = _pipeline()
        single_label_buckets: Dict[str, Counter] = {
            "acne": Counter(),
            "skin_type": Counter(),
        }
        confidences: Dict[str, list] = {"acne": [], "skin_type": []}
        multi_label_findings: Counter = Counter()
        images_with_any_finding = 0
        images_with_no_finding = 0
        images_with_both_findings = 0
        face_failures = 0
        skin_conditions_available = False

        for img_path in _real_images():
            preds = _predict(orch, pipeline, img_path)
            if not preds:
                face_failures += 1
                continue
            for key, pred in preds.items():
                if isinstance(pred, ModelPrediction) and key in single_label_buckets:
                    single_label_buckets[key][pred.predicted_label] += 1
                    confidences[key].append(pred.confidence)
                elif isinstance(pred, MultiLabelPrediction) and key == "skin_conditions":
                    skin_conditions_available = True
                    labels = _finding_labels(pred)
                    if labels:
                        images_with_any_finding += 1
                        for lbl in labels:
                            multi_label_findings[lbl] += 1
                        if len(labels) >= 2:
                            images_with_both_findings += 1
                    else:
                        images_with_no_finding += 1

        request.cls._buckets = single_label_buckets
        request.cls._confidences = confidences
        request.cls._multi_label_findings = multi_label_findings
        request.cls._images_with_any_finding = images_with_any_finding
        request.cls._images_with_no_finding = images_with_no_finding
        request.cls._images_with_both_findings = images_with_both_findings
        request.cls._skin_conditions_available = skin_conditions_available
        request.cls._n_images = len(_real_images())
        request.cls._face_failures = face_failures

    def test_face_detection_covers_most_photos(self):
        """Face-crop pipeline should succeed on at least 80% of real photos."""
        assert self._face_failures / max(self._n_images, 1) <= 0.20, (
            f"Face detection failed on {self._face_failures}/{self._n_images} real photos. "
            "Check rotation-fallback and EXIF handling in preprocessing/pipeline.py."
        )

    def test_acne_does_not_overpredict_severe(self):
        """Adult real-world portraits should rarely be `severe` (≤30%)."""
        total = sum(self._buckets["acne"].values())
        severe = self._buckets["acne"].get("severe", 0)
        ratio = severe / total if total else 0
        assert ratio <= 0.30, (
            f"acne model predicted `severe` on {severe}/{total} "
            f"({ratio:.0%}) real-world photos. Distribution: {dict(self._buckets['acne'])}."
        )

    def test_confidence_distribution_is_reasonable(self):
        """Mean confidence should be within (0.30, 0.98)."""
        for key, confs in self._confidences.items():
            if not confs:
                continue
            mean_conf = sum(confs) / len(confs)
            assert 0.30 < mean_conf < 0.98, (
                f"{key} mean confidence {mean_conf:.2f} on real photos "
                "is outside the reasonable range (0.30, 0.98)."
            )

    @requires_multi_label
    def test_skin_conditions_reports_empty_findings_sometimes(self):
        """The clean-skin case (empty findings) MUST occur on real portraits.

        This is the exact bug that motivated the refactor: the old
        single-label model never predicted `healthy` on any real photo.
        """
        if not self._skin_conditions_available:
            pytest.skip("skin_conditions model output not present")
        assert self._images_with_no_finding > 0, (
            "skin_conditions model flagged something on EVERY real photo — "
            "'no notable conditions' should occur on at least some clean faces. "
            f"Per-condition tallies: {dict(self._multi_label_findings)}."
        )

    @requires_multi_label
    def test_skin_conditions_does_not_flag_everything(self):
        """Model shouldn't flag both conditions on more than 60% of photos."""
        if not self._skin_conditions_available:
            pytest.skip("skin_conditions model output not present")
        total = self._n_images - self._face_failures
        ratio = self._images_with_both_findings / max(total, 1)
        assert ratio <= 0.60, (
            f"Model flagged BOTH conditions on {self._images_with_both_findings}/{total} "
            f"({ratio:.0%}) real photos. Consider raising "
            "`skin_conditions_threshold` in shared/config.py."
        )
