"""
Acne Severity Classification Model

EfficientNetB0-based classifier for acne severity detection.
Classes: Clear (0), Mild (1), Moderate (2), Severe (3)

Architecture adapted from validated academic transfer learning approach.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Label mapping
ACNE_CLASSES = {
    0: "clear",
    1: "mild",
    2: "moderate",
    3: "severe",
}

NUM_CLASSES = 4


class AcneSeverityModel(nn.Module):
    """EfficientNetB0-based acne severity classifier.

    Uses transfer learning from ImageNet weights.
    The backbone can be frozen for initial training, then upper layers
    unfrozen for fine-tuning with a smaller learning rate.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.efficientnet_b0(weights=weights)

        # Replace classification head for our classes
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


def build_model(pretrained: bool = True, freeze: bool = True) -> AcneSeverityModel:
    """Factory function to build the acne model."""
    model = AcneSeverityModel(pretrained=pretrained)
    if freeze:
        model.freeze_backbone()
    return model
