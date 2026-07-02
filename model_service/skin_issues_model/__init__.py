"""
Skin Issues Detection Model

EfficientNetB0-based classifier for detecting specific skin issues.
Classes: Healthy (0), Blackheads (1), Dark Spots (2), Pores (3), Wrinkles (4)

Trained on the Skin Issues v2 dataset (9,770 images) + healthy class
from ACNE04 clear images. The "acne" class is excluded because Model 1
handles acne severity separately.
"""

import torch
import torch.nn as nn
import torchvision.models as models


SKIN_ISSUE_CLASSES = {
    0: "healthy",
    1: "blackheads",
    2: "dark_spots",
    3: "pores",
    4: "wrinkles",
}

NUM_CLASSES = 5


class SkinIssuesModel(nn.Module):
    """EfficientNetB0-based skin issues classifier."""

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


def build_skin_issues_model(pretrained: bool = True, freeze: bool = True) -> SkinIssuesModel:
    model = SkinIssuesModel(pretrained=pretrained)
    if freeze:
        model.freeze_backbone()
    return model
