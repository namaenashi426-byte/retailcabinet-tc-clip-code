"""RetailCabinet preprocessing transforms for MMAction2 baselines."""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
from mmcv.transforms import BaseTransform
from PIL import Image
import torchvision.transforms as tv_transforms

from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RetailColorJitter(BaseTransform):
    """Color jitter with the same probability semantics as the TC-CLIP pipeline."""

    def __init__(
        self,
        p: float = 0.8,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
    ) -> None:
        self.p = float(p)
        self.worker = tv_transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def transform(self, results: Dict) -> Dict:
        if random.random() >= self.p:
            return results
        results["imgs"] = [
            np.asarray(self.worker(Image.fromarray(img))) for img in results["imgs"]
        ]
        return results


@TRANSFORMS.register_module()
class RetailGrayScale(BaseTransform):
    """Random grayscale with three output channels, matching TC-CLIP."""

    def __init__(self, p: float = 0.2) -> None:
        self.p = float(p)
        self.worker = tv_transforms.Grayscale(num_output_channels=3)

    def transform(self, results: Dict) -> Dict:
        if random.random() >= self.p:
            return results
        results["imgs"] = [
            np.asarray(self.worker(Image.fromarray(img))) for img in results["imgs"]
        ]
        return results
