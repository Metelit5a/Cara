"""
Pores Severity Classification Model

EfficientNetB0-based classifier for pore severity detection.
Classes: Minimal (0), Mild (1), Moderate (2), Severe (3)

Trained on COCO pore-detection data converted to classification
by binning annotation counts per image into severity classes.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Label mapping
PORE_CLASSES = {
    0: "minimal",
    1: "mild",
    2: "moderate",
    3: "severe",
}

NUM_CLASSES = 4

# Pore count thresholds for severity classification
# Based on dataset distribution: Q1=12, median=16, Q3=22
PORE_COUNT_THRESHOLDS = [8, 16, 26]  # <=8 minimal, 9-16 mild, 17-26 moderate, >26 severe


class PoreSeverityModel(nn.Module):
    """EfficientNetB0-based pore severity classifier.

    Uses transfer learning from ImageNet weights.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.efficientnet_b0(weights=weights)

        # Replace classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all backbone layers, keep classifier trainable."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_upper_layers(self):
        """Unfreeze upper convolutional blocks (6, 7, 8) for fine-tuning."""
        for name, param in self.backbone.features.named_parameters():
            if any(f"{i}" in name.split(".")[0] for i in [6, 7, 8]):
                param.requires_grad = True

    def get_trainable_params(self):
        """Return only parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]


def build_pore_model(pretrained: bool = True, freeze: bool = True) -> PoreSeverityModel:
    """Factory function to build the pores model."""
    model = PoreSeverityModel(pretrained=pretrained)
    if freeze:
        model.freeze_backbone()
    return model
