"""Dataset constants for the EARTHVISION 2021 FloodNet Track 1 release."""

from __future__ import annotations

TRACK1_ROOT_NAME = "FloodNet Challenge @ EARTHVISION 2021 - Track 1"
SUPERVISED_ROOT_NAME = "FloodNet-Supervised_v1.0"

CLASS_NAMES = (
    "Background",
    "Building-flooded",
    "Building-non-flooded",
    "Road-flooded",
    "Road-non-flooded",
    "Water",
    "Tree",
    "Vehicle",
    "Pool",
    "Grass",
)
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

EXPECTED_COUNTS = {
    "labeled_flooded": 51,
    "labeled_non_flooded": 347,
    "unlabeled": 1047,
    "validation": 450,
    "test": 448,
}


SUPERVISED_EXPECTED_COUNTS = {
    "train": 1445,
    "validation": 450,
    "test": 448,
}

SCENE_DIRECTORIES = {
    "Flooded": "Train/Labeled/Flooded",
    "Non-Flooded": "Train/Labeled/Non-Flooded",
}
