from __future__ import annotations

import numpy as np


def mask_to_color(mask: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_id, rgb in palette.items():
        color[mask == cls_id] = rgb
    return color


def make_overlay(image_rgb: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(image_rgb * (1 - alpha) + color_mask * alpha, 0, 255).astype(np.uint8)

