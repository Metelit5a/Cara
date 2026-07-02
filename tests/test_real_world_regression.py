"""
Real-World Regression Tests — the tests we WISH we had before.

These tests exercise the models on the same kind of images real users
upload (well-lit adult portraits/selfies), not on training data.

Why this file exists:
  `test_model_accuracy.py` pulls samples straight from the training
  folders and just verifies the model remembers them. It passes even
  when the model is catastrophically overfit and never predicts some
  classes on real inputs (e.g. `healthy` for skin_issues).

Two layers:
  1. Per-image assertions using hand-labelled ground truth from
     `tests/fixtures/real_world_labels.json`.
  2. Population-level sanity checks over the whole real-world folder
     (guards against class-collapse — e.g. "never predicts healthy").
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from PIL import Image


# ── Locations ────────────────────────────────────────────────────────────────
REAL_WORLD_DIR = Path("image tests people")
FIXTURE_PATH = Path("tests/fixtures/real_world_labels.json")
CHECKPOINT_DIR = Path("model_service/checkpoints")

requires_checkpoints = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "acne_model_best.pth").exists()
    or not (CHECKPOINT_DIR / "skin_type_model_best.pth").exists()
    or not (CHECKPOINT_DIR / "skin_issues_model_best.pth").exists(),
    reason="One or more model checkpoints missing",
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

    # Fresh instance per test class so we don't share state with the app singleton.
    return PreprocessingPipeline(require_face=True, apply_mask=False)


def _predict(orchestrator, pipeline, path: Path) -> dict:
    """Run the SAME inference path the app uses: face-crop → model.

    If face detection fails, returns {} — callers should skip.
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
    return sorted(p for p in REAL_WORLD_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 — per-image ground-truth assertions
# ═══════════════════════════════════════════════════════════════════════════════


@requires_checkpoints
@requires_real_images
class TestLabelledRealWorldImages:
    """Assert predictions match human labels for real-world reference photos.

    Each labelled image contributes independent assertions. A field labelled
    `null` in the fixture is skipped (so partially-labelled images are OK).
    """

    @pytest.fixture(autouse=True, scope="class")
    def orchestrator(self):
        # Cache one orchestrator per class run — model loading is slow.
        request = getattr(self, "request", None)
        return _orchestrator()

    def _get_pred(self, filename: str, model_key: str) -> Optional[dict]:
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

    # Populate one parametrised test per fixture entry so failures name the image.
    @pytest.fixture(autouse=True)
    def _bind_orch(self):
        self.orchestrator = _orchestrator()
        self.pipeline = _pipeline()

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

    @pytest.mark.parametrize("filename, expected", list(_load_fixture().items()))
    def test_skin_issues_in_accepted_set(self, filename, expected):
        accepted = expected.get("skin_issues")
        if not accepted:
            pytest.skip("skin_issues label not provided")
        pred = self._get_pred(filename, "skin_issues")
        assert pred.predicted_label in accepted, (
            f"[{filename}] skin_issues: expected one of {accepted}, "
            f"got `{pred.predicted_label}` (conf={pred.confidence:.2f}). "
            f"Notes: {expected.get('notes', '')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — population sanity checks (catch class collapse)
# ═══════════════════════════════════════════════════════════════════════════════


@requires_checkpoints
@requires_real_images
class TestRealWorldPopulationSanity:
    """Population-level checks over ALL real-world images.

    These guard against silent regressions where the model works on
    training data but never predicts a class on real inputs (the bug
    that motivated this suite).
    """

    @pytest.fixture(autouse=True, scope="class")
    def _distribution(self, request):
        orch = _orchestrator()
        pipeline = _pipeline()
        buckets: Dict[str, Counter] = {"acne": Counter(), "skin_type": Counter(), "skin_issues": Counter()}
        confidences: Dict[str, list] = {"acne": [], "skin_type": [], "skin_issues": []}
        face_failures = 0
        for img_path in _real_images():
            preds = _predict(orch, pipeline, img_path)
            if not preds:
                face_failures += 1
                continue
            for key, pred in preds.items():
                buckets[key][pred.predicted_label] += 1
                confidences[key].append(pred.confidence)
        request.cls._buckets = buckets
        request.cls._confidences = confidences
        request.cls._n_images = len(_real_images())
        request.cls._face_failures = face_failures

    def test_skin_issues_predicts_healthy_sometimes(self):
        """On a set of predominantly healthy real faces, `healthy` must appear.

        Catches the exact bug this suite was built for: the model achieves
        ~98% on training data but has never predicted `healthy` on any
        real-world image.
        """
        healthy_count = self._buckets["skin_issues"].get("healthy", 0)
        total = sum(self._buckets["skin_issues"].values())
        assert healthy_count > 0, (
            f"skin_issues model NEVER predicts `healthy` on {total} real-world "
            f"photos. Distribution: {dict(self._buckets['skin_issues'])}. "
            "This indicates severe overfitting or class collapse — the training "
            "data likely doesn't match real user selfies."
        )

    def test_acne_does_not_overpredict_severe(self):
        """Adult real-world portraits should rarely be `severe`.

        Even in stock photos where some acne is visible, `severe` is a
        specific clinical label. If >30% of arbitrary real photos come
        back as severe, the model is biased toward the rare class
        (likely from class-weighting during training).
        """
        total = sum(self._buckets["acne"].values())
        severe = self._buckets["acne"].get("severe", 0)
        ratio = severe / total if total else 0
        assert ratio <= 0.30, (
            f"acne model predicted `severe` on {severe}/{total} "
            f"({ratio:.0%}) real-world photos. Class weighting has "
            f"over-corrected. Distribution: {dict(self._buckets['acne'])}."
        )

    def test_skin_issues_uses_more_than_two_classes(self):
        """Model shouldn't collapse to only 2 classes on real photos.

        With 5 classes and diverse real inputs, at least 3 different
        labels should appear (otherwise the model has effectively
        forgotten most of its output vocabulary).
        """
        n_classes_used = len(self._buckets["skin_issues"])
        assert n_classes_used >= 3, (
            f"skin_issues model only used {n_classes_used} distinct labels "
            f"across {self._n_images} real photos: "
            f"{dict(self._buckets['skin_issues'])}. Model has collapsed."
        )

    def test_confidence_distribution_is_reasonable(self):
        """Mean confidence shouldn't be pathologically low or unrealistically high."""
        for key, confs in self._confidences.items():
            if not confs:
                continue
            mean_conf = sum(confs) / len(confs)
            # Below 0.30 → model is guessing.
            # Above 0.98 → model is dangerously overconfident on OOD data.
            assert 0.30 < mean_conf < 0.98, (
                f"{key} mean confidence {mean_conf:.2f} on real photos "
                "is outside the reasonable range (0.30, 0.98)."
            )
