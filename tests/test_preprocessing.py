"""Tests for the preprocessing pipeline."""

import numpy as np
import pytest
from PIL import Image
import io
import torch


class TestPreprocessingPipeline:
    """Test preprocessing pipeline components."""

    def test_valid_image_produces_tensor(self, sample_face_image_bytes):
        """A valid image should produce a tensor (face detection may not find a face in random noise)."""
        from model_service.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(require_face=False)
        result, tensor = pipeline.process(sample_face_image_bytes)

        assert result.success is True
        assert tensor is not None
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_invalid_image_returns_failure(self, invalid_image_bytes):
        """Non-image bytes should fail gracefully."""
        from model_service.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(require_face=False)
        result, tensor = pipeline.process(invalid_image_bytes)

        assert result.success is False
        assert tensor is None

    def test_require_face_no_face(self, sample_face_image_bytes):
        """With require_face=True and random noise (no face), should fail."""
        from model_service.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(require_face=True)
        result, tensor = pipeline.process(sample_face_image_bytes)

        # Random noise won't have a detectable face
        assert result.success is False
        assert result.face_detected is False
        assert tensor is None

    def test_brightness_normalization(self):
        """CLAHE normalization should not crash on valid images."""
        from model_service.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(require_face=False)

        # Create a very dark image
        dark_img = Image.fromarray(np.full((224, 224, 3), 20, dtype=np.uint8))
        buf = io.BytesIO()
        dark_img.save(buf, format="JPEG")
        dark_bytes = buf.getvalue()

        result, tensor = pipeline.process(dark_bytes)
        assert result.success is True
        assert tensor is not None

    def test_output_tensor_normalized(self, sample_face_image_bytes):
        """Output tensor should be normalized (values around -2 to +3 due to ImageNet normalization)."""
        from model_service.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(require_face=False)
        _, tensor = pipeline.process(sample_face_image_bytes)

        # ImageNet-normalized tensors should not have values in raw 0-255 range
        assert tensor.max().item() < 10.0
        assert tensor.min().item() > -10.0
