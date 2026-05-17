"""Robust General Acne Severity Model.

EfficientNetB0-based 4-class severity classifier trained on the Roboflow
"Acne new data v1i COCO" dataset (~5944 face/skin photos with 83k+
lesion bounding boxes spanning 15 lesion types).

Severity is derived from total lesion count per image:
    clear     : 0 lesions
    mild      : 1-5 lesions
    moderate  : 6-15 lesions
    severe    : 16+ lesions

This complements the acne04 severity model, which was trained on a much
smaller dataset, and is intended to be the primary "general acne" model
for real-world photos.
"""

from typing import List

import torch
import torch.nn as nn
import torchvision.models as models


GENERAL_ACNE_CLASSES = {
    0: "clear",
    1: "mild",
    2: "moderate",
    3: "severe",
}
NUM_CLASSES = 4

# Lesion-count thresholds used to bin COCO annotations into severity classes.
LESION_COUNT_THRESHOLDS: List[int] = [0, 5, 15]  # <=0 clear, 1-5 mild, 6-15 mod, >15 severe


def lesion_count_to_class(count: int) -> int:
    if count <= LESION_COUNT_THRESHOLDS[0]:
        return 0
    if count <= LESION_COUNT_THRESHOLDS[1]:
        return 1
    if count <= LESION_COUNT_THRESHOLDS[2]:
        return 2
    return 3


class GeneralAcneModel(nn.Module):
    """EfficientNetB0 transfer-learning classifier for general acne severity."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False

    def unfreeze_upper_layers(self):
        for name, p in self.backbone.features.named_parameters():
            if any(f"{i}" in name.split(".")[0] for i in [6, 7, 8]):
                p.requires_grad = True

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_general_acne_model(pretrained: bool = True, freeze: bool = True) -> GeneralAcneModel:
    model = GeneralAcneModel(pretrained=pretrained)
    if freeze:
        model.freeze_backbone()
    return model
