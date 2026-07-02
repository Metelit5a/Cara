"""
Image Preprocessing Pipeline

Shared by both inference AND training so the distributions match:
1. Face detection (OpenCV Haar Cascade + rotation-fallback) — tight
   bounding box, NO padding. Haar can only find upright frontal faces,
   so we retry at ±15°, ±30° when the 0° attempt fails.
2. Elliptical face mask (optional) — zeros out corners so the model
   only sees face skin.
3. Resize to 224x224
4. ImageNet normalization
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from typing import Tuple, Optional
from shared.schemas import PreprocessingResult

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Angles (deg) to try when the 0° detection fails. 0° is always tried
# first. Small angles come first so we prefer the least rotation.
# ±90 covers phones held sideways, ±180 covers upside-down selfies.
_ROTATION_FALLBACK_ANGLES = (-15, 15, -30, 30, -45, 45, -90, 90, 180)


class FaceDetector:
    """Tight Haar-cascade face detector (no artificial padding).

    Includes a rotation-fallback: if the upright detector fails, tries
    several rotations of the image and keeps the largest face found.
    """

    def __init__(self, min_size: int = 60):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.min_size = min_size

    def _detect_upright(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect a frontal face in the image as-provided (no rotation)."""
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

    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Backwards-compat: return bbox in the original (non-rotated) frame.

        Prefer `detect_with_rotation` — it returns both the bbox AND the
        rotation angle that made detection succeed, which the caller
        needs to crop the actual face.
        """
        bbox_and_angle = self.detect_with_rotation(image)
        return bbox_and_angle[0] if bbox_and_angle else None

    def detect_with_rotation(
        self, image: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """Detect a face, retrying with rotations if the upright attempt fails.

        Returns ((x, y, w, h), angle) where the bbox is in the coordinate
        space of the image *rotated by `angle`*. The caller should rotate
        the image by the same angle before cropping.
        """
        # Try 0° first — most photos are upright.
        bbox = self._detect_upright(image)
        if bbox is not None:
            return bbox, 0.0

        # Rotation fallback for tilted / sideways photos.
        h, w = image.shape[:2]
        centre = (w // 2, h // 2)
        for angle in _ROTATION_FALLBACK_ANGLES:
            M = cv2.getRotationMatrix2D(centre, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            bbox = self._detect_upright(rotated)
            if bbox is not None:
                return bbox, float(angle)
        return None


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


def _rotate_bgr(image_bgr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the whole image around its centre — no cropping."""
    h, w = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def preprocess_face_array(
    image_bgr: np.ndarray,
    detector: Optional[FaceDetector] = None,
    apply_mask: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Core pipeline shared by training and inference.

    Returns (RGB uint8 224x224 numpy array, face_detected).

    Face detection is rotation-aware — if the head is tilted (as in
    lying-down selfies), the image is rotated so the crop is upright,
    which matches the training distribution of frontal-face samples.
    """
    detector = detector or FaceDetector()
    result = detector.detect_with_rotation(image_bgr)
    face_detected = result is not None

    if face_detected:
        bbox, angle = result
        x, y, w, h = bbox
        # Crop from the image AT the rotation angle where detection worked,
        # so the crop is an upright face (matches training distribution).
        source = _rotate_bgr(image_bgr, angle) if angle != 0.0 else image_bgr
        cropped = source[y:y + h, x:x + w]
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

    def _load_bgr_with_exif(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Decode image bytes into BGR, honouring EXIF orientation.

        cv2.imdecode ignores EXIF, which causes iPhone photos (stored
        landscape with an orientation tag) to load sideways. We route
        through PIL to normalise orientation first.
        """
        try:
            pil = Image.open(io.BytesIO(image_bytes))
            pil = ImageOps.exif_transpose(pil)  # apply rotation tag
            pil = pil.convert("RGB")
        except Exception:
            return None
        arr = np.array(pil)  # RGB
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def process(self, image_bytes: bytes) -> Tuple[PreprocessingResult, Optional[torch.Tensor]]:
        image = self._load_bgr_with_exif(image_bytes)
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
