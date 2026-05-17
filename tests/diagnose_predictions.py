"""Diagnostic test: run all loaded models on a folder of real photos and
print per-image preprocessing results and prediction distributions.

Usage:
    python -m tests.diagnose_predictions "image tests people"
"""

import sys
from pathlib import Path

import torch

from model_service.preprocessing.pipeline import get_pipeline
from model_service.inference.orchestrator import get_orchestrator, get_registry
from shared.config import settings


def main(folder: str):
    registry = get_registry()
    registry.register_acne_model(settings.model_weights_path)
    registry.register_pores_model(settings.pores_model_weights_path)
    registry.register_general_acne_model(settings.general_acne_model_weights_path)
    print(f"Loaded models: {registry.loaded_models}")

    orchestrator = get_orchestrator()
    pipeline = get_pipeline()
    pipeline.require_face = False  # diagnose, don't reject

    folder_path = Path(folder)
    images = sorted(
        p for p in folder_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    print(f"\nFound {len(images)} images in {folder_path}\n")

    for img_path in images:
        data = img_path.read_bytes()
        result, tensor = pipeline.process(data)
        print(f"--- {img_path.name} ---")
        print(f"  face_detected={result.face_detected}  msg={result.message}")
        if tensor is None:
            print("  (no tensor)\n")
            continue
        preds = orchestrator.predict_all(tensor)
        for name, p in preds.items():
            probs_str = ", ".join(f"{v:.2f}" for v in p.all_probabilities)
            print(f"  [{name:13s}] {p.predicted_label:9s} conf={p.confidence:.3f}  probs=[{probs_str}]")
        print()


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "image tests people"
    main(folder)
