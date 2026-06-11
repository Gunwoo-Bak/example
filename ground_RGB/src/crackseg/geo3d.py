from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODEL_PIXEL_SCALE_TAG = 33550
MODEL_TIEPOINT_TAG = 33922
GEO_ASCII_PARAMS_TAG = 34737


@dataclass(frozen=True)
class OrthoReference:
    path: str
    width: int
    height: int
    pixel_size_x: float
    pixel_size_y: float
    tiepoint_col: float
    tiepoint_row: float
    tiepoint_x: float
    tiepoint_y: float
    crs_text: str | None = None

    @classmethod
    def from_geotiff(cls, path: str | Path) -> OrthoReference:
        image_path = Path(path)
        with Image.open(image_path) as img:
            tags = getattr(img, "tag_v2", {})
            scale = tags.get(MODEL_PIXEL_SCALE_TAG)
            tiepoint = tags.get(MODEL_TIEPOINT_TAG)
            crs_text = tags.get(GEO_ASCII_PARAMS_TAG)
            width, height = img.size

        if scale is None or tiepoint is None or len(scale) < 2 or len(tiepoint) < 6:
            raise ValueError(f"GeoTIFF scale/tiepoint tags were not found in {image_path}")

        return cls(
            path=str(image_path),
            width=int(width),
            height=int(height),
            pixel_size_x=float(scale[0]),
            pixel_size_y=float(scale[1]),
            tiepoint_col=float(tiepoint[0]),
            tiepoint_row=float(tiepoint[1]),
            tiepoint_x=float(tiepoint[3]),
            tiepoint_y=float(tiepoint[4]),
            crs_text=str(crs_text) if crs_text is not None else None,
        )

    @property
    def pixel_area(self) -> float:
        return abs(self.pixel_size_x * self.pixel_size_y)

    def pixel_to_world(self, col: float, row: float) -> tuple[float, float]:
        x = self.tiepoint_x + (col - self.tiepoint_col) * self.pixel_size_x
        y = self.tiepoint_y - (row - self.tiepoint_row) * self.pixel_size_y
        return x, y

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        col = self.tiepoint_col + (x - self.tiepoint_x) / self.pixel_size_x
        row = self.tiepoint_row - (y - self.tiepoint_y) / self.pixel_size_y
        return col, row

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "image_width": self.width,
            "image_height": self.height,
            "pixel_size_x": self.pixel_size_x,
            "pixel_size_y": self.pixel_size_y,
            "pixel_area": self.pixel_area,
            "tiepoint_col": self.tiepoint_col,
            "tiepoint_row": self.tiepoint_row,
            "tiepoint_x": self.tiepoint_x,
            "tiepoint_y": self.tiepoint_y,
            "crs_text": self.crs_text,
        }


@dataclass(frozen=True)
class GcpPoint:
    name: str
    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class HeightModel:
    gcps: list[GcpPoint]
    method: str = "idw"
    power: float = 2.0
    plane_coefficients: tuple[float, float, float] | None = None

    @classmethod
    def from_csv(cls, path: str | Path, method: str = "idw", power: float = 2.0) -> HeightModel:
        gcp_path = Path(path)
        points: list[GcpPoint] = []
        with gcp_path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is None:
                raise ValueError(f"GCP CSV has no header: {gcp_path}")
            name_field = reader.fieldnames[0]
            for row in reader:
                points.append(
                    GcpPoint(
                        name=str(row[name_field]),
                        x=float(row["X"]),
                        y=float(row["Y"]),
                        z=float(row["Z"]),
                    )
                )

        if not points:
            raise ValueError(f"No GCP rows were found in {gcp_path}")

        plane_coefficients = None
        if method == "plane":
            if len(points) < 3:
                raise ValueError("At least three GCP points are required for plane height interpolation")
            a = np.array([[p.x, p.y, 1.0] for p in points], dtype=np.float64)
            b = np.array([p.z for p in points], dtype=np.float64)
            coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
            plane_coefficients = tuple(float(v) for v in coeffs)
        elif method != "idw":
            raise ValueError(f"Unknown GCP height interpolation method: {method}")

        return cls(gcps=points, method=method, power=float(power), plane_coefficients=plane_coefficients)

    def height_at(self, x: float, y: float) -> float:
        if self.method == "plane":
            if self.plane_coefficients is None:
                raise ValueError("Plane coefficients are missing")
            a, b, c = self.plane_coefficients
            return float(a * x + b * y + c)

        coords = np.array([[p.x, p.y] for p in self.gcps], dtype=np.float64)
        zs = np.array([p.z for p in self.gcps], dtype=np.float64)
        query = np.array([x, y], dtype=np.float64)
        distances = np.linalg.norm(coords - query, axis=1)
        nearest = np.where(distances < 1e-9)[0]
        if nearest.size:
            return float(zs[int(nearest[0])])
        weights = 1.0 / np.power(distances, self.power)
        return float(np.sum(weights * zs) / np.sum(weights))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "power": self.power,
            "gcp_count": len(self.gcps),
            "gcps": [p.to_dict() for p in self.gcps],
            "plane_coefficients": self.plane_coefficients,
        }


@dataclass(frozen=True)
class HomographyResult:
    matrix: np.ndarray
    source_path: str | None = None
    orthophoto_path: str | None = None
    detector: str | None = None
    total_matches: int | None = None
    good_matches: int | None = None
    inliers: int | None = None
    reprojection_rmse_px: float | None = None

    def project(self, x: float, y: float) -> tuple[float, float]:
        point = np.array([x, y, 1.0], dtype=np.float64)
        projected = self.matrix @ point
        if abs(float(projected[2])) < 1e-12:
            raise ValueError("Homography projection produced a point at infinity")
        return float(projected[0] / projected[2]), float(projected[1] / projected[2])

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "orthophoto_path": self.orthophoto_path,
            "detector": self.detector,
            "matrix": self.matrix.tolist(),
            "total_matches": self.total_matches,
            "good_matches": self.good_matches,
            "inliers": self.inliers,
            "reprojection_rmse_px": self.reprojection_rmse_px,
        }

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> HomographyResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            matrix=np.array(data["matrix"], dtype=np.float64),
            source_path=data.get("source_path"),
            orthophoto_path=data.get("orthophoto_path"),
            detector=data.get("detector"),
            total_matches=data.get("total_matches"),
            good_matches=data.get("good_matches"),
            inliers=data.get("inliers"),
            reprojection_rmse_px=data.get("reprojection_rmse_px"),
        )


@dataclass(frozen=True)
class Geo3DContext:
    orthophoto: OrthoReference
    height_model: HeightModel | None = None
    homography: HomographyResult | None = None

    def image_pixel_to_ortho_pixel(self, x: float, y: float) -> tuple[float, float]:
        if self.homography is None:
            return x, y
        return self.homography.project(x, y)

    def image_pixel_to_world3d(self, x: float, y: float) -> tuple[float, float, float | None, float, float]:
        ortho_x, ortho_y = self.image_pixel_to_ortho_pixel(x, y)
        world_x, world_y = self.orthophoto.pixel_to_world(ortho_x, ortho_y)
        world_z = self.height_model.height_at(world_x, world_y) if self.height_model is not None else None
        return world_x, world_y, world_z, ortho_x, ortho_y

    def to_dict(self) -> dict[str, Any]:
        return {
            "orthophoto": self.orthophoto.to_dict(),
            "height_model": self.height_model.to_dict() if self.height_model is not None else None,
            "homography": self.homography.to_dict() if self.homography is not None else None,
            "coordinate_note": (
                "world_x_m and world_y_m use the orthophoto CRS. world_z_m is interpolated from GCP Z values "
                "when a GCP CSV is provided."
            ),
        }


def build_geo3d_context(
    source_path: str | Path,
    orthophoto_path: str | Path,
    gcp_csv: str | Path | None = None,
    homography_path: str | Path | None = None,
    save_homography_path: str | Path | None = None,
    height_method: str = "idw",
) -> Geo3DContext:
    orthophoto = OrthoReference.from_geotiff(orthophoto_path)
    height_model = HeightModel.from_csv(gcp_csv, method=height_method) if gcp_csv is not None else None

    homography = None
    if homography_path is not None:
        homography = HomographyResult.load(homography_path)
    elif Path(source_path).resolve() != Path(orthophoto_path).resolve():
        homography = estimate_image_to_ortho_homography(source_path, orthophoto_path)
        if save_homography_path is not None:
            homography.save(save_homography_path)

    return Geo3DContext(orthophoto=orthophoto, height_model=height_model, homography=homography)


def add_geo3d_to_instance_rows(
    rows: list[dict[str, Any]],
    context: Any,
    pred_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if pred_mask is not None and hasattr(context, "image_regions_to_world3d"):
        results = context.image_regions_to_world3d(rows, pred_mask)
        enriched: list[dict[str, Any]] = []
        for row, result in zip(rows, results):
            item = dict(row)
            item.update(result)
            enriched.append(item)
        return enriched

    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if pred_mask is not None and hasattr(context, "image_region_to_world3d"):
            result = context.image_region_to_world3d(row, pred_mask)
            item.update(result)
            enriched.append(item)
            continue

        result = context.image_pixel_to_world3d(
            float(row["centroid_x_px"]),
            float(row["centroid_y_px"]),
        )
        if isinstance(result, dict):
            item.update(result)
            enriched.append(item)
            continue

        world_x, world_y, world_z, ortho_x, ortho_y = result
        item["ortho_pixel_x"] = round(ortho_x, 4)
        item["ortho_pixel_y"] = round(ortho_y, 4)
        item["world_x_m"] = round(world_x, 8)
        item["world_y_m"] = round(world_y, 8)
        item["world_z_m"] = round(world_z, 8) if world_z is not None else None
        item["geo3d_source"] = "homography_to_orthophoto" if context.homography is not None else "orthophoto_geotiff"
        enriched.append(item)
    return enriched


def build_geo3d_damage_pixel_rows(
    pred_mask: np.ndarray,
    class_names: dict[int, str],
    context: Any,
    stride: int = 1,
) -> list[dict[str, Any]]:
    if stride < 1:
        raise ValueError("stride must be >= 1")

    ys, xs = np.nonzero(pred_mask != 0)
    rows: list[dict[str, Any]] = []
    for index in range(0, len(xs), stride):
        x = float(xs[index])
        y = float(ys[index])
        class_id = int(pred_mask[int(y), int(x)])
        result = context.image_pixel_to_world3d(x, y)
        if isinstance(result, dict):
            spatial = dict(result)
        else:
            world_x, world_y, world_z, ortho_x, ortho_y = result
            spatial = {
                "ortho_pixel_x": round(ortho_x, 4),
                "ortho_pixel_y": round(ortho_y, 4),
                "world_x_m": round(world_x, 8),
                "world_y_m": round(world_y, 8),
                "world_z_m": round(world_z, 8) if world_z is not None else None,
                "geo3d_source": "homography_to_orthophoto" if getattr(context, "homography", None) is not None else "orthophoto_geotiff",
            }

        rows.append(
            {
                "pixel_x": int(x),
                "pixel_y": int(y),
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                **spatial,
            }
        )
    return rows


def estimate_image_to_ortho_homography(
    source_path: str | Path,
    orthophoto_path: str | Path,
    max_dim: int = 2200,
    max_features: int = 12000,
    ratio: float = 0.75,
    ransac_reproj_threshold: float = 5.0,
    min_inliers: int = 12,
) -> HomographyResult:
    import cv2

    source = _read_gray_resized(source_path, max_dim=max_dim)
    ortho = _read_gray_resized(orthophoto_path, max_dim=max_dim)
    detector_name, detector, norm_type = _create_feature_detector(max_features=max_features)

    source_kp, source_desc = detector.detectAndCompute(source["image"], None)
    ortho_kp, ortho_desc = detector.detectAndCompute(ortho["image"], None)
    if source_desc is None or ortho_desc is None or not source_kp or not ortho_kp:
        raise ValueError("Could not extract enough image features to estimate homography")

    matcher = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(source_desc, ortho_desc, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good_matches.append(first)

    if len(good_matches) < 4:
        raise ValueError(f"Not enough feature matches for homography: {len(good_matches)}")

    src_pts = np.float32([source_kp[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([ortho_kp[m.trainIdx].pt for m in good_matches])
    src_pts[:, 0] /= source["scale_x"]
    src_pts[:, 1] /= source["scale_y"]
    dst_pts[:, 0] /= ortho["scale_x"]
    dst_pts[:, 1] /= ortho["scale_y"]

    matrix, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
    if matrix is None or inlier_mask is None:
        raise ValueError("OpenCV could not estimate a homography")

    inliers = int(inlier_mask.ravel().sum())
    if inliers < min_inliers:
        raise ValueError(f"Homography has too few inliers: {inliers} < {min_inliers}")

    rmse = _homography_rmse(matrix, src_pts, dst_pts, inlier_mask.ravel().astype(bool))
    return HomographyResult(
        matrix=matrix.astype(np.float64),
        source_path=str(source_path),
        orthophoto_path=str(orthophoto_path),
        detector=detector_name,
        total_matches=len(raw_matches),
        good_matches=len(good_matches),
        inliers=inliers,
        reprojection_rmse_px=round(rmse, 4),
    )


def _read_gray_resized(path: str | Path, max_dim: int) -> dict[str, Any]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image for homography: {path}")
    height, width = image.shape[:2]
    scale = min(float(max_dim) / max(width, height), 1.0)
    if scale < 1.0:
        resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    else:
        resized = image
    return {
        "image": resized,
        "scale_x": resized.shape[1] / width,
        "scale_y": resized.shape[0] / height,
    }


def _create_feature_detector(max_features: int) -> tuple[str, Any, int]:
    import cv2

    if hasattr(cv2, "SIFT_create"):
        return "SIFT", cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    return "ORB", cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING


def _homography_rmse(matrix: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, inliers: np.ndarray) -> float:
    src_in = src_pts[inliers]
    dst_in = dst_pts[inliers]
    if len(src_in) == 0:
        return float("nan")
    points = np.concatenate([src_in, np.ones((len(src_in), 1), dtype=np.float32)], axis=1)
    projected = (matrix @ points.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - dst_in, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))
