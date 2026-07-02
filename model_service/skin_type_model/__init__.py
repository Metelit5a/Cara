"""
Skin Type Classification Model

EfficientNetB0-based classifier for skin type detection.
Classes: Oily (0), Dry (1), Normal (2), Combination (3)

Trained on the Facial Skin Analysis and Type Classification dataset
(4,093 face images, Apache 2.0 license).
"""

import torch
import torch.nn as nn
import torchvision.models as models


SKIN_TYPE_CLASSES = {
    0: "oily",
    1: "dry",
    2: "normal",
    3: "combination",
}

NUM_CLASSES = 4


class SkinTypeModel(nn.Module):
    """EfficientNetB0-based skin type classifier."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_upper_layers(self):
        for name, param in self.backbone.features.named_parameters():
            if any(f"{i}" in name.split(".")[0] for i in [6, 7, 8]):
                param.requires_grad = True

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_skin_type_model(pretrained: bool = True, freeze: bool = True) -> SkinTypeModel:
    model = SkinTypeModel(pretrained=pretrained)
    if freeze:
        model.freeze_backbone()
    return model
