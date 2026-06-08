"""
Skin Issues Type Classification Model

EfficientNetB0-based classifier for skin issue type detection.
Classes: Acne (0), Blackheads (1), Dark Spots (2), Pores (3), Wrinkles (4)

Trained on the Skin Issues v2 dataset (folder-based, ~2000 images per class).
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Label mapping
SKIN_ISSUE_CLASSES = {
    0: "acne",
    1: "blackheads",
    2: "dark_spots",
    3: "pores",
    4: "wrinkles",
}

NUM_CLASSES = 5


class SkinIssuesModel(nn.Module):
    """EfficientNetB0-based skin issue type classifier.

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


def build_skin_issues_model(pretrained: bool = True, freeze: bool = True) -> SkinIssuesModel:
    """Factory function to build the skin issues model."""
    model = SkinIssuesModel(pretrained=pretrained)
    if freeze:
        model.freeze_backbone()
    return model
