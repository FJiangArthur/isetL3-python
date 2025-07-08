"""L3 ISP: Modular Image Signal Processing Pipeline with Local Linear Learning.

This package provides a complete ISP pipeline that combines:
- L3 (Local, Linear, Learned) algorithm for adaptive image processing
- InfiniteISP integration for traditional ISP stages
- MIT-Adobe FiveK dataset preprocessing
- BT.2020 color space support
- Synthetic data generation
"""

from .core import L3Pipeline, L3Config
from .data import (
    L3DataCamera,
    L3DataSimulation,
    L3DataISET,
    FiveKDataset,
    SyntheticDataGenerator,
)
from .classify import L3Classifier, AdaptiveClassifier
from .train import L3TrainOLS, L3TrainRidge, L3TrainS
from .render import L3Render, L3ISPRender
from .color import ColorSpaceConverter, BT2020Converter
from .utils import l3_root_path, create_bayer_pattern, extract_patches

__version__ = "2.0.0"

__all__ = [
    "L3Pipeline",
    "L3Config",
    "L3DataCamera",
    "L3DataSimulation",
    "L3DataISET",
    "FiveKDataset",
    "SyntheticDataGenerator",
    "L3Classifier",
    "AdaptiveClassifier",
    "L3TrainOLS",
    "L3TrainRidge",
    "L3TrainS",
    "L3Render",
    "L3ISPRender",
    "ColorSpaceConverter",
    "BT2020Converter",
    "l3_root_path",
    "create_bayer_pattern",
    "extract_patches",
]
