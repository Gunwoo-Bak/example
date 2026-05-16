from __future__ import annotations

from typing import Any

import numpy as np


def summarize_pixels(mask: np.ndarray, class_names: dict[int, str]) -> list[dict[str, Any]]:
    total = int(mask.size)
    rows: list[dict[str, Any]] = []
    for cls_id, cls_name in class_names.items():
        count = int((mask == cls_id).sum())
        ratio = (count / total * 100.0) if total > 0 else 0.0
        rows.append(
            {
                "class_id": int(cls_id),
                "class_name": cls_name,
                "pixel_count": count,
                "pixel_ratio": round(count / total, 8) if total > 0 else 0.0,
                "area_percent": round(ratio, 6),
            }
        )
    return rows


def foreground_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float((mask != 0).sum() / mask.size)
