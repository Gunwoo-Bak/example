from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def read_image_rgb(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def list_images(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() not in VALID_IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {p}")
        return [p]
    if not p.exists():
        raise FileNotFoundError(p)
    return sorted(
        x for x in p.rglob("*")
        if x.is_file() and x.suffix.lower() in VALID_IMAGE_EXTS
    )

