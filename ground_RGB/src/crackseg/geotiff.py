from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


MODEL_PIXEL_SCALE_TAG = 33550
MODEL_TIEPOINT_TAG = 33922
GEO_KEY_DIRECTORY_TAG = 34735
GEO_ASCII_PARAMS_TAG = 34737


@dataclass(frozen=True)
class GeoTiffInfo:
    pixel_size_x: float
    pixel_size_y: float
    tiepoint_col: float
    tiepoint_row: float
    tiepoint_x: float
    tiepoint_y: float
    crs_text: str | None

    @property
    def pixel_area(self) -> float:
        return abs(self.pixel_size_x * self.pixel_size_y)

    def pixel_to_world(self, col: float, row: float) -> tuple[float, float]:
        x = self.tiepoint_x + (col - self.tiepoint_col) * self.pixel_size_x
        y = self.tiepoint_y - (row - self.tiepoint_row) * self.pixel_size_y
        return x, y

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_size_x": self.pixel_size_x,
            "pixel_size_y": self.pixel_size_y,
            "pixel_area": self.pixel_area,
            "tiepoint_col": self.tiepoint_col,
            "tiepoint_row": self.tiepoint_row,
            "tiepoint_x": self.tiepoint_x,
            "tiepoint_y": self.tiepoint_y,
            "crs_text": self.crs_text,
        }


def read_geotiff_info(path: str | Path) -> GeoTiffInfo | None:
    image_path = Path(path)
    if image_path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    try:
        with Image.open(image_path) as img:
            tags = getattr(img, "tag_v2", {})
            scale = tags.get(MODEL_PIXEL_SCALE_TAG)
            tiepoint = tags.get(MODEL_TIEPOINT_TAG)
            crs_text = tags.get(GEO_ASCII_PARAMS_TAG)
    except Exception:
        return None

    if scale is None or tiepoint is None or len(scale) < 2 or len(tiepoint) < 6:
        return None

    return GeoTiffInfo(
        pixel_size_x=float(scale[0]),
        pixel_size_y=float(scale[1]),
        tiepoint_col=float(tiepoint[0]),
        tiepoint_row=float(tiepoint[1]),
        tiepoint_x=float(tiepoint[3]),
        tiepoint_y=float(tiepoint[4]),
        crs_text=str(crs_text) if crs_text is not None else None,
    )


def add_geo_to_pixel_rows(rows: list[dict[str, Any]], info: GeoTiffInfo | None) -> list[dict[str, Any]]:
    if info is None:
        return rows
    enriched = []
    for row in rows:
        item = dict(row)
        item["area_m2"] = round(float(row["pixel_count"]) * info.pixel_area, 10)
        item["pixel_size_x_m"] = info.pixel_size_x
        item["pixel_size_y_m"] = info.pixel_size_y
        enriched.append(item)
    return enriched


def add_geo_to_instance_rows(rows: list[dict[str, Any]], info: GeoTiffInfo | None) -> list[dict[str, Any]]:
    if info is None:
        return rows
    enriched = []
    for row in rows:
        item = dict(row)
        cx, cy = info.pixel_to_world(float(row["centroid_x_px"]) + 0.5, float(row["centroid_y_px"]) + 0.5)
        xmin, ymax = info.pixel_to_world(float(row["bbox_xmin_px"]), float(row["bbox_ymin_px"]))
        xmax, ymin = info.pixel_to_world(float(row["bbox_xmax_px"]), float(row["bbox_ymax_px"]))
        item.update(
            {
                "centroid_x_m": round(cx, 8),
                "centroid_y_m": round(cy, 8),
                "bbox_xmin_m": round(min(xmin, xmax), 8),
                "bbox_ymin_m": round(min(ymin, ymax), 8),
                "bbox_xmax_m": round(max(xmin, xmax), 8),
                "bbox_ymax_m": round(max(ymin, ymax), 8),
                "area_m2": round(float(row["area_px"]) * info.pixel_area, 10),
            }
        )
        enriched.append(item)
    return enriched


def add_geo_to_instance_summary_rows(rows: list[dict[str, Any]], info: GeoTiffInfo | None) -> list[dict[str, Any]]:
    if info is None:
        return rows
    enriched = []
    for row in rows:
        item = dict(row)
        item["total_area_m2"] = round(float(row["total_area_px"]) * info.pixel_area, 10)
        enriched.append(item)
    return enriched
