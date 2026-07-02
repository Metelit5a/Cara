"""
Image Preprocessing Pipeline

Shared by both inference AND training so the distributions match:
1. Face detection (OpenCV Haar Cascade) — tight bounding box, NO padding
2. Elliptical face mask — zero out the corners (background) so the model
   only sees face skin (approximates face contour without landmark deps)
3. Resize to 224x224
4. ImageNet normalization
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Optional
from shared.schemas import PreprocessingResult

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FaceDetector:
    """Tight Haar-cascade face detector (no artificial padding)."""

    def __init__(self, min_size: int = 60):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.min_size = min_size

    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(self.min_size, self.min_size),
        )
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        return (int(x), int(y), int(w), int(h))


def apply_face_mask(face_bgr: np.ndarray) -> np.ndarray:
    """Mask the bounding-box corners with the face mean colour so only the
    face contour (inscribed ellipse) remains visible. Removes background
    pixels that confuse the classifier without needing facial landmarks."""
    h, w = face_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center=(w // 2, h // 2),
        axes=(int(w * 0.48), int(h * 0.55)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )
    mean_color = face_bgr.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    background = np.full_like(face_bgr, mean_color)
    return np.where(mask[..., None] == 255, face_bgr, background)


def preprocess_face_array(
    image_bgr: np.ndarray,
    detector: Optional[FaceDetector] = None,
    apply_mask: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Core pipeline shared by training and inference.

    Returns (RGB uint8 224x224 numpy array, face_detected).
    """
    detector = detector or FaceDetector()
    bbox = detector.detect(image_bgr)
    face_detected = bbox is not None

    if face_detected:
        x, y, w, h = bbox
        cropped = image_bgr[y:y + h, x:x + w]
        if cropped.size == 0:
            cropped = image_bgr
            face_detected = False
        elif apply_mask:
            cropped = apply_face_mask(cropped)
    else:
        cropped = image_bgr

    resized = cv2.resize(cropped, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), face_detected


class PreprocessingPipeline:
    """Inference-time preprocessing pipeline."""

    def __init__(self, require_face: bool = True, apply_mask: bool = True):
        self.face_detector = FaceDetector()
        self.require_face = require_face
        self.apply_mask = apply_mask
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def process(self, image_bytes: bytes) -> Tuple[PreprocessingResult, Optional[torch.Tensor]]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return PreprocessingResult(
                success=False, face_detected=False, message="Invalid image data"
            ), None

        rgb, face_detected = preprocess_face_array(
            image, detector=self.face_detector, apply_mask=self.apply_mask
        )

        if self.require_face and not face_detected:
            return PreprocessingResult(
                success=False,
                face_detected=False,
                message="No face detected in image. Please upload a clear face photo.",
            ), None

        pil_image = Image.fromarray(rgb)
        tensor = self.transform(pil_image)
        return PreprocessingResult(
            success=True,
            face_detected=face_detected,
            message="Preprocessing complete" if face_detected else "Processed without face crop",
        ), tensor


_pipeline: Optional[PreprocessingPipeline] = None


def get_pipeline() -> PreprocessingPipeline:
    """Return the shared preprocessing pipeline.

    Face-crop is required (`require_face=True`) so we never silently feed a
    full selfie with background/hair/clothing into models that were trained
    on face-centric close-ups.

    Mask is disabled (`apply_mask=False`): the elliptical mean-colour fill
    introduces an artefact the models were never trained on. Empirically,
    pure face-crop matches the training distribution better than crop+mask.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = PreprocessingPipeline(require_face=True, apply_mask=False)
    return _pipeline
