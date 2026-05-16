from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CrackSegConfig:
    classes: list[str]
    num_classes: int
    model: str
    encoder_name: str
    img_size: int
    stride: int
    palette: dict[int, tuple[int, int, int]]

    @property
    def class_names(self) -> dict[int, str]:
        return {idx: name for idx, name in enumerate(self.classes)}


def load_config(path: str | Path) -> CrackSegConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    palette = {
        int(cls_id): tuple(int(v) for v in rgb)
        for cls_id, rgb in raw["palette"].items()
    }

    return CrackSegConfig(
        classes=list(raw["classes"]),
        num_classes=int(raw["num_classes"]),
        model=str(raw["model"]),
        encoder_name=str(raw["encoder_name"]),
        img_size=int(raw["img_size"]),
        stride=int(raw.get("stride", int(raw["img_size"]) // 2)),
        palette=palette,
    )

