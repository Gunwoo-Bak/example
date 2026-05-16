from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import CrackSegConfig
from .geotiff import (
    add_geo_to_instance_rows,
    add_geo_to_instance_summary_rows,
    add_geo_to_pixel_rows,
    read_geotiff_info,
)
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
    source_path: str | Path | None = None,
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

    geotiff_info = read_geotiff_info(source_path) if source_path is not None else None

    pixel_rows = add_geo_to_pixel_rows(summarize_pixels(pred_mask, config.class_names), geotiff_info)
    write_dict_rows(pixel_csv_path, pixel_rows)

    instance_rows, instance_summary_rows = summarize_instances(pred_mask, config.class_names)
    instance_rows = add_geo_to_instance_rows(instance_rows, geotiff_info)
    instance_summary_rows = add_geo_to_instance_summary_rows(instance_summary_rows, geotiff_info)
    write_dict_rows(instance_csv_path, instance_rows)
    write_dict_rows(instance_summary_csv_path, instance_summary_rows)

    metadata: dict[str, Any] = {
        "image_height": int(image.shape[0]),
        "image_width": int(image.shape[1]),
        "classes": config.class_names,
        "geotiff": geotiff_info.to_dict() if geotiff_info is not None else None,
        "note": "No ground-truth mask was provided. Pixel and instance summaries are computed from model predictions only.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    paths = {
        "mask": mask_path,
        "overlay": overlay_path,
        "confidence": confidence_path,
        "pixel_summary": pixel_csv_path,
        "instances": instance_csv_path,
        "instance_summary": instance_summary_csv_path,
        "metadata": metadata_path,
    }
    paths.update(
        save_damage_type_outputs(
            image=image,
            pred_mask=pred_mask,
            output_dir=output_dir,
            stem=stem,
            config=config,
            pixel_rows=pixel_rows,
            instance_rows=instance_rows,
            instance_summary_rows=instance_summary_rows,
            alpha=alpha,
        )
    )
    return paths


def save_damage_type_outputs(
    image: np.ndarray,
    pred_mask: np.ndarray,
    output_dir: str | Path,
    stem: str,
    config: CrackSegConfig,
    pixel_rows: list[dict[str, Any]],
    instance_rows: list[dict[str, Any]],
    instance_summary_rows: list[dict[str, Any]],
    alpha: float = 0.45,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    saved: dict[str, Path] = {}
    row_by_class = {int(row["class_id"]): row for row in pixel_rows}
    summary_by_class = {int(row["class_id"]): row for row in instance_summary_rows}

    for cls_id, cls_name in config.class_names.items():
        if int(cls_id) == 0:
            continue

        damage_dir = output_dir / "by_damage" / f"{int(cls_id):02d}_{safe_name(cls_name)}"
        masks_dir = damage_dir / "masks"
        overlays_dir = damage_dir / "overlays"
        summaries_dir = damage_dir / "summaries"
        instances_dir = damage_dir / "instances"
        for directory in [masks_dir, overlays_dir, summaries_dir, instances_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        binary_mask = (pred_mask == int(cls_id)).astype(np.uint8)
        binary_mask_u8 = binary_mask * 255
        class_color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        class_color_mask[binary_mask == 1] = config.palette.get(int(cls_id), (255, 255, 255))
        class_overlay = make_overlay(image, class_color_mask, alpha=alpha)

        class_stem = f"{stem}_{safe_name(cls_name)}"
        binary_mask_path = masks_dir / f"{class_stem}_binary_mask.png"
        color_mask_path = masks_dir / f"{class_stem}_color_mask.png"
        overlay_path = overlays_dir / f"{class_stem}_overlay.png"
        pixel_summary_path = summaries_dir / f"{class_stem}_pixel_summary.csv"
        instance_path = instances_dir / f"{class_stem}_instances.csv"
        instance_summary_path = instances_dir / f"{class_stem}_instance_summary.csv"

        Image.fromarray(binary_mask_u8).save(binary_mask_path)
        Image.fromarray(class_color_mask).save(color_mask_path)
        Image.fromarray(class_overlay).save(overlay_path)

        write_dict_rows(pixel_summary_path, [row_by_class.get(int(cls_id), empty_pixel_row(int(cls_id), cls_name))])
        write_dict_rows(
            instance_path,
            [row for row in instance_rows if int(row["class_id"]) == int(cls_id)],
        )
        write_dict_rows(
            instance_summary_path,
            [summary_by_class.get(int(cls_id), empty_instance_summary_row(int(cls_id), cls_name))],
        )

        key_prefix = safe_name(cls_name).lower()
        saved[f"{key_prefix}_binary_mask"] = binary_mask_path
        saved[f"{key_prefix}_color_mask"] = color_mask_path
        saved[f"{key_prefix}_overlay"] = overlay_path
        saved[f"{key_prefix}_pixel_summary"] = pixel_summary_path
        saved[f"{key_prefix}_instances"] = instance_path
        saved[f"{key_prefix}_instance_summary"] = instance_summary_path

    return saved


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


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_")


def empty_pixel_row(cls_id: int, cls_name: str) -> dict[str, Any]:
    return {
        "class_id": int(cls_id),
        "class_name": cls_name,
        "pixel_count": 0,
        "pixel_ratio": 0.0,
        "area_percent": 0.0,
    }


def empty_instance_summary_row(cls_id: int, cls_name: str) -> dict[str, Any]:
    return {
        "class_id": int(cls_id),
        "class_name": cls_name,
        "instance_count": 0,
        "total_area_px": 0,
    }


def read_mask_for_overlay(mask_path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)
