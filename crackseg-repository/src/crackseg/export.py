from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import CrackSegConfig
from .instances import summarize_instances
from .metrics import summarize_pixels
from .visualization import make_overlay, mask_to_color


def save_prediction_outputs(
    image: np.ndarray,
    pred_mask: np.ndarray,
    confidence: np.ndarray,
    output_dir: str | Path,
    stem: str,
    config: CrackSegConfig,
    alpha: float = 0.45,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    confidences_dir = output_dir / "confidence"
    summaries_dir = output_dir / "summaries"
    instances_dir = output_dir / "instances"

    for directory in [masks_dir, overlays_dir, confidences_dir, summaries_dir, instances_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    color_mask = mask_to_color(pred_mask, config.palette)
    overlay = make_overlay(image, color_mask, alpha=alpha)
    conf_u8 = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)

    mask_path = masks_dir / f"{stem}_mask.png"
    overlay_path = overlays_dir / f"{stem}_overlay.png"
    confidence_path = confidences_dir / f"{stem}_confidence.png"
    pixel_csv_path = summaries_dir / f"{stem}_pixel_summary.csv"
    instance_csv_path = instances_dir / f"{stem}_instances.csv"
    instance_summary_csv_path = instances_dir / f"{stem}_instance_summary.csv"
    metadata_path = summaries_dir / f"{stem}_metadata.json"

    Image.fromarray(color_mask).save(mask_path)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(conf_u8).save(confidence_path)

    pixel_rows = summarize_pixels(pred_mask, config.class_names)
    write_dict_rows(pixel_csv_path, pixel_rows)

    instance_rows, instance_summary_rows = summarize_instances(pred_mask, config.class_names)
    write_dict_rows(instance_csv_path, instance_rows)
    write_dict_rows(instance_summary_csv_path, instance_summary_rows)

    metadata: dict[str, Any] = {
        "image_height": int(image.shape[0]),
        "image_width": int(image.shape[1]),
        "classes": config.class_names,
        "note": "No ground-truth mask was provided. Pixel and instance summaries are computed from model predictions only.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "mask": mask_path,
        "overlay": overlay_path,
        "confidence": confidence_path,
        "pixel_summary": pixel_csv_path,
        "instances": instance_csv_path,
        "instance_summary": instance_summary_csv_path,
        "metadata": metadata_path,
    }


def write_batch_summary(rows: list[dict[str, Any]], output_csv: str | Path) -> None:
    write_dict_rows(Path(output_csv), rows)


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_mask_for_overlay(mask_path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)
