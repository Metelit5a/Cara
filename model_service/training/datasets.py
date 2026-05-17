"""Shared training dataset utilities.

Wraps a PIL/image source so the SAME face-detection + elliptical mask
preprocessing used at inference is also applied during training. This
eliminates the train/inference distribution mismatch that was causing
the low-confidence predictions on real-world photos.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model_service.preprocessing.pipeline import (
    FaceDetector,
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_face_array,
)


def build_train_augment() -> Callable:
    """Augmentations applied AFTER face-crop+mask, on the 224x224 RGB PIL image."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform() -> Callable:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FaceCropImageDataset(Dataset):
    """Dataset that takes (image_path, label) samples and applies the shared
    face-detect + elliptical-mask preprocessing identically to inference.

    If face detection fails on a training image, the full image is used
    (still resized to 224). This matches inference fallback behavior.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[Callable] = None,
        apply_mask: bool = True,
    ):
        self.samples = samples
        self.transform = transform or build_eval_transform()
        self.apply_mask = apply_mask
        # Detector is created lazily per worker process (cv2.CascadeClassifier
        # is not picklable, so we can't create it in __init__ when using
        # num_workers > 0 on Windows / spawn).
        self._detector: Optional[FaceDetector] = None

    def _get_detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            # Corrupt / unreadable: return a neutral grey image
            image_bgr = np.full((224, 224, 3), 127, dtype=np.uint8)

        rgb, _ = preprocess_face_array(
            image_bgr, detector=self._get_detector(), apply_mask=self.apply_mask
        )
        pil_image = Image.fromarray(rgb)
        return self.transform(pil_image), label

    @property
    def targets(self) -> List[int]:
        return [label for _, label in self.samples]
