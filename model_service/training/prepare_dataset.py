"""
ACNE04 Dataset Preparation

Converts ACNE04 annotations into simplified severity classification labels.
Severity is derived heuristically from lesion counts:

    0 lesions     → Clear   (0)
    1–5 lesions   → Mild    (1)
    6–20 lesions  → Moderate (2)
    21+ lesions   → Severe  (3)

This script expects the ACNE04 dataset to be placed in data/acne04/ with
the standard directory structure.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET


# Severity mapping based on lesion count
def lesion_count_to_severity(count: int) -> int:
    if count == 0:
        return 0  # Clear
    elif count <= 5:
        return 1  # Mild
    elif count <= 20:
        return 2  # Moderate
    else:
        return 3  # Severe


SEVERITY_NAMES = {0: "clear", 1: "mild", 2: "moderate", 3: "severe"}


def parse_acne04_annotations(data_dir: str) -> List[Dict]:
    """Parse ACNE04 dataset and create classification labels.

    Supports multiple annotation formats:
    - XML (Pascal VOC style)
    - Direct folder-based severity levels
    """
    data_path = Path(data_dir)
    samples = []

    # Strategy 1: Check for XML annotations
    annotations_dir = data_path / "annotations"
    images_dir = data_path / "images"

    if annotations_dir.exists() and images_dir.exists():
        for xml_file in annotations_dir.glob("*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename = root.find("filename").text
            objects = root.findall("object")
            lesion_count = len(objects)
            severity = lesion_count_to_severity(lesion_count)

            image_path = images_dir / filename
            if image_path.exists():
                samples.append({
                    "image_path": str(image_path),
                    "lesion_count": lesion_count,
                    "severity": severity,
                    "severity_name": SEVERITY_NAMES[severity],
                })
        return samples

    # Strategy 2: Folder-based levels (e.g., level_0, level_1, level_2, level_3)
    for level_dir in sorted(data_path.iterdir()):
        if not level_dir.is_dir():
            continue
        name = level_dir.name.lower()

        # Try to extract severity from folder name
        severity = None
        if "clear" in name or "level_0" in name or "0" == name:
            severity = 0
        elif "mild" in name or "level_1" in name or "1" == name:
            severity = 1
        elif "moderate" in name or "level_2" in name or "2" == name:
            severity = 2
        elif "severe" in name or "level_3" in name or "3" == name:
            severity = 3

        if severity is not None:
            for img_file in level_dir.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    samples.append({
                        "image_path": str(img_file),
                        "lesion_count": -1,
                        "severity": severity,
                        "severity_name": SEVERITY_NAMES[severity],
                    })

    return samples


def organize_into_folders(samples: List[Dict], output_dir: str):
    """Organize images into train/test folders by severity class."""
    output_path = Path(output_dir)
    from sklearn.model_selection import train_test_split

    # Split 80/20
    train_samples, test_samples = train_test_split(
        samples, test_size=0.2, stratify=[s["severity"] for s in samples], random_state=42
    )

    for split_name, split_samples in [("train", train_samples), ("test", test_samples)]:
        for severity_name in SEVERITY_NAMES.values():
            (output_path / split_name / severity_name).mkdir(parents=True, exist_ok=True)

        for sample in split_samples:
            src = sample["image_path"]
            dst_dir = output_path / split_name / sample["severity_name"]
            dst = dst_dir / Path(src).name
            shutil.copy2(src, dst)

    # Save metadata
    meta = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "class_distribution": {},
    }
    for s in SEVERITY_NAMES.values():
        meta["class_distribution"][s] = sum(1 for x in samples if x["severity_name"] == s)

    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset organized: {len(train_samples)} train, {len(test_samples)} test")
    print(f"Distribution: {meta['class_distribution']}")


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/acne04"
    output_dir = "data/acne04_processed"

    print(f"Parsing ACNE04 dataset from: {data_dir}")
    samples = parse_acne04_annotations(data_dir)

    if not samples:
        print("No samples found. Check the dataset path and format.")
        sys.exit(1)

    print(f"Found {len(samples)} samples")
    organize_into_folders(samples, output_dir)
    print(f"Processed dataset saved to: {output_dir}")
