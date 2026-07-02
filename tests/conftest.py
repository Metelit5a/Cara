"""Shared test fixtures."""

import pytest
import numpy as np
from PIL import Image
import io


@pytest.fixture
def sample_face_image_bytes():
    """Generate a synthetic test image (300x300 RGB face-like)."""
    img = Image.fromarray(np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def invalid_image_bytes():
    """Return non-image bytes."""
    return b"this is not an image"


@pytest.fixture
def empty_bytes():
    return b""
