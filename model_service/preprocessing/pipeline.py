"""
Image Preprocessing Pipeline

Modular preprocessing pipeline for skin analysis:
1. Face detection (OpenCV Haar Cascade)
2. Face cropping
3. Resize
4. Brightness normalization
5. Tensor normalization

Designed to be reusable across all future model services.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Optional
from shared.schemas import PreprocessingResult

# Standard image size for EfficientNetB0
IMAGE_SIZE = (224, 224)

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FaceDetector:
    """Face detection using OpenCV Haar Cascade."""

    def __init__(self, min_confidence: float = 0.5):
        # OpenCV ships with pre-trained Haar cascades
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.min_confidence = min_confidence

    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face and return bounding box (x, y, w, h) or None."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) == 0:
            return None

        # Use the largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        # Add padding around the face
        img_h, img_w = image.shape[:2]
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(img_w - x, w + 2 * pad_x)
        h = min(img_h - y, h + 2 * pad_y)

        return (x, y, w, h)

    def detect_all(self, image: np.ndarray) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
        """Detect all faces and return count + bbox of largest face.
        
        Returns:
            Tuple of (face_count, bbox) where bbox is (x, y, w, h) or None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        face_count = len(faces)
        if face_count == 0:
            return 0, None

        # Use the largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        # Add padding around the face
        img_h, img_w = image.shape[:2]
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(img_w - x, w + 2 * pad_x)
        h = min(img_h - y, h + 2 * pad_y)

        return face_count, (x, y, w, h)


class PreprocessingPipeline:
    """Complete preprocessing pipeline for skin analysis images."""

    def __init__(self, require_face: bool = True):
        self.face_detector = FaceDetector()
        self.require_face = require_face

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for brightness normalization."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def process(self, image_bytes: bytes) -> Tuple[PreprocessingResult, Optional[torch.Tensor]]:
        """Run full preprocessing pipeline on raw image bytes.

        Returns:
            Tuple of (PreprocessingResult, tensor or None)
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return PreprocessingResult(
                success=False, face_detected=False, message="Invalid image data"
            ), None

        # Face detection
        bbox = self.face_detector.detect(image)
        face_detected = bbox is not None

        if self.require_face and not face_detected:
            return PreprocessingResult(
                success=False,
                face_detected=False,
                message="No face detected in image. Please upload a clear face photo.",
            ), None

        # Crop to face if detected
        if face_detected:
            x, y, w, h = bbox
            cropped = image[y:y + h, x:x + w]
            if cropped.size > 0:
                image = cropped

        # Brightness normalization
        image = self._normalize_brightness(image)

        # Convert BGR to RGB and to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Apply torchvision transforms
        tensor = self.transform(pil_image)

        return PreprocessingResult(
            success=True,
            face_detected=face_detected,
            message="Preprocessing complete",
        ), tensor


# Module-level singleton
_pipeline: Optional[PreprocessingPipeline] = None


def get_pipeline() -> PreprocessingPipeline:
    """Get or create the preprocessing pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PreprocessingPipeline()
    return _pipeline
