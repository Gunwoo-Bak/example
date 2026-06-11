from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    focal_length_px: float
    cx: float
    cy: float
    dewarp_data: str | None = None

    def ray_camera(self, pixel_x: float, pixel_y: float) -> np.ndarray:
        x = (float(pixel_x) - self.cx) / self.focal_length_px
        y = (float(pixel_y) - self.cy) / self.focal_length_px
        ray = np.array([x, y, 1.0], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "focal_length_px": self.focal_length_px,
            "cx": self.cx,
            "cy": self.cy,
            "dewarp_data": self.dewarp_data,
            "distortion_note": "DewarpData is preserved in metadata; ray projection currently uses calibrated pinhole intrinsics.",
        }


@dataclass(frozen=True)
class CameraPose:
    source_path: str
    center_xyz: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    position_source: str
    orientation_source: str
    rtk_std_lon_m: float | None = None
    rtk_std_lat_m: float | None = None
    rtk_std_hgt_m: float | None = None

    def ray_world(self, camera_ray: np.ndarray) -> np.ndarray:
        rotation = camera_to_world_matrix(self.yaw_deg, self.pitch_deg, self.roll_deg)
        ray = rotation @ camera_ray
        return ray / np.linalg.norm(ray)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "center_x_m": self.center_xyz[0],
            "center_y_m": self.center_xyz[1],
            "center_z_m": self.center_xyz[2],
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "position_source": self.position_source,
            "orientation_source": self.orientation_source,
            "rtk_std_lon_m": self.rtk_std_lon_m,
            "rtk_std_lat_m": self.rtk_std_lat_m,
            "rtk_std_hgt_m": self.rtk_std_hgt_m,
            "orientation_note": "DJI gimbal yaw/pitch/roll are interpreted as camera optical-axis orientation in an ENU world frame.",
        }


@dataclass(frozen=True)
class LasHeader:
    path: str
    version: str
    point_format: int
    point_record_length: int
    point_count: int
    offset_to_points: int
    scale: tuple[float, float, float]
    offset: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    crs_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "version": self.version,
            "point_format": self.point_format,
            "point_record_length": self.point_record_length,
            "point_count": self.point_count,
            "scale": self.scale,
            "offset": self.offset,
            "bounds_min": self.bounds_min,
            "bounds_max": self.bounds_max,
            "crs_text": self.crs_text,
        }


class LasSurfaceIndex:
    def __init__(self, path: str | Path):
        self.header = read_las_header(path)
        self.points = read_las_points(self.header)
        self.tree = cKDTree(self.points)

    def intersect_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        step_m: float = 0.25,
        max_ray_distance_m: float = 0.20,
    ) -> dict[str, Any]:
        t_min, t_max = ray_box_interval(origin, direction, self.header.bounds_min, self.header.bounds_max)
        if t_min is None or t_max is None:
            return empty_hit("ray_misses_las_bounds")

        t0 = max(0.0, t_min)
        t1 = max(t0 + step_m, t_max)
        sample_count = max(2, int(math.ceil((t1 - t0) / step_m)) + 1)
        samples_t = np.linspace(t0, t1, sample_count)
        sample_points = origin[None, :] + samples_t[:, None] * direction[None, :]
        _, sample_indices = self.tree.query(sample_points, k=1)

        candidate_indices = np.unique(sample_indices)
        candidates = self.points[candidate_indices]
        to_points = candidates - origin[None, :]
        t = to_points @ direction
        in_front = t >= 0.0
        if not np.any(in_front):
            return empty_hit("no_las_points_in_front_of_camera")

        candidates = candidates[in_front]
        candidate_indices = candidate_indices[in_front]
        t = t[in_front]
        closest_on_ray = origin[None, :] + t[:, None] * direction[None, :]
        distances = np.linalg.norm(candidates - closest_on_ray, axis=1)
        best = int(np.argmin(distances))
        best_point = candidates[best]
        best_distance = float(distances[best])
        return {
            "world_x_m": round(float(best_point[0]), 8),
            "world_y_m": round(float(best_point[1]), 8),
            "world_z_m": round(float(best_point[2]), 8),
            "geo3d_source": "dji_pose_las_ray",
            "las_point_index": int(candidate_indices[best]),
            "ray_t_m": round(float(t[best]), 6),
            "ray_surface_distance_m": round(best_distance, 6),
            "ray_hit": bool(best_distance <= max_ray_distance_m),
            "ray_miss_reason": None,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.header.to_dict()


@dataclass
class LasRayGeo3DContext:
    intrinsics: CameraIntrinsics
    pose: CameraPose
    surface: LasSurfaceIndex
    ray_step_m: float = 0.25
    max_ray_distance_m: float = 0.20

    def image_pixel_to_world3d(self, x: float, y: float) -> dict[str, Any]:
        ray_camera = self.intrinsics.ray_camera(x, y)
        ray_world = self.pose.ray_world(ray_camera)
        hit = self.surface.intersect_ray(
            np.array(self.pose.center_xyz, dtype=np.float64),
            ray_world,
            step_m=self.ray_step_m,
            max_ray_distance_m=self.max_ray_distance_m,
        )
        hit["image_pixel_x"] = round(float(x), 4)
        hit["image_pixel_y"] = round(float(y), 4)
        return hit

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "dji_pose_las_ray",
            "intrinsics": self.intrinsics.to_dict(),
            "pose": self.pose.to_dict(),
            "las": self.surface.to_dict(),
            "ray_step_m": self.ray_step_m,
            "max_ray_distance_m": self.max_ray_distance_m,
            "accuracy_note": (
                "This path uses DJI/PPK camera position and DJI gimbal orientation without manual pose refinement. "
                "ray_hit=false means the nearest LAS point is farther than max_ray_distance_m from the projected ray."
            ),
        }


@dataclass
class ImageXYZMap:
    world_x: np.ndarray
    world_y: np.ndarray
    world_z: np.ndarray
    depth_m: np.ndarray
    distance_px: np.ndarray
    source: np.ndarray
    metadata: dict[str, Any]

    @property
    def height(self) -> int:
        return int(self.world_x.shape[0])

    @property
    def width(self) -> int:
        return int(self.world_x.shape[1])

    def lookup(self, x: float, y: float) -> dict[str, Any]:
        col = int(round(float(x)))
        row = int(round(float(y)))
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return empty_xyz_map_lookup("pixel_outside_image", x, y)

        source = int(self.source[row, col])
        if source == 0 or not np.isfinite(self.world_x[row, col]):
            return empty_xyz_map_lookup("no_xyz_map_value", x, y)

        return {
            "world_x_m": round(float(self.world_x[row, col]), 8),
            "world_y_m": round(float(self.world_y[row, col]), 8),
            "world_z_m": round(float(self.world_z[row, col]), 8),
            "geo3d_source": "las_image_xyz_map",
            "xyz_valid": True,
            "xyz_map_source": "projected_las" if source == 1 else "nearest_fill",
            "xyz_distance_px": round(float(self.distance_px[row, col]), 4),
            "xyz_depth_m": round(float(self.depth_m[row, col]), 6),
            "xyz_miss_reason": None,
            "image_pixel_x": round(float(x), 4),
            "image_pixel_y": round(float(y), 4),
        }

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            world_x=self.world_x,
            world_y=self.world_y,
            world_z=self.world_z,
            depth_m=self.depth_m,
            distance_px=self.distance_px,
            source=self.source,
            metadata=json.dumps(self.metadata, ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: str | Path) -> ImageXYZMap:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            return cls(
                world_x=data["world_x"],
                world_y=data["world_y"],
                world_z=data["world_z"],
                depth_m=data["depth_m"],
                distance_px=data["distance_px"],
                source=data["source"],
                metadata=metadata,
            )


@dataclass
class XYZMapGeo3DContext:
    intrinsics: CameraIntrinsics
    pose: CameraPose
    surface_header: LasHeader
    xyz_map: ImageXYZMap
    cache_path: str
    fill_distance_px: float
    instance_padding_px: int = 0

    def image_pixel_to_world3d(self, x: float, y: float) -> dict[str, Any]:
        return self.xyz_map.lookup(x, y)

    def image_region_to_world3d(self, instance_row: dict[str, Any], pred_mask: np.ndarray) -> dict[str, Any]:
        class_id = int(instance_row["class_id"])
        xmin = max(0, int(instance_row["bbox_xmin_px"]) - self.instance_padding_px)
        ymin = max(0, int(instance_row["bbox_ymin_px"]) - self.instance_padding_px)
        xmax = min(self.xyz_map.width, int(instance_row["bbox_xmax_px"]) + self.instance_padding_px)
        ymax = min(self.xyz_map.height, int(instance_row["bbox_ymax_px"]) + self.instance_padding_px)
        if xmin >= xmax or ymin >= ymax:
            return empty_instance_xyz("empty_instance_bbox")

        submask = pred_mask[ymin:ymax, xmin:xmax] == class_id
        valid = submask & (self.xyz_map.source[ymin:ymax, xmin:xmax] > 0)
        if not np.any(valid):
            centroid = self.image_pixel_to_world3d(
                float(instance_row["centroid_x_px"]),
                float(instance_row["centroid_y_px"]),
            )
            centroid.update(
                {
                    "xyz_valid": False,
                    "xyz_miss_reason": "no_valid_xyz_in_instance_region",
                    "instance_xyz_sample_count": 0,
                    "instance_xyz_projected_count": 0,
                    "instance_xyz_nearest_count": 0,
                    "instance_xyz_distance_px_median": None,
                    "instance_xyz_distance_px_max": None,
                }
            )
            return centroid

        wx = self.xyz_map.world_x[ymin:ymax, xmin:xmax][valid].astype(np.float64)
        wy = self.xyz_map.world_y[ymin:ymax, xmin:xmax][valid].astype(np.float64)
        wz = self.xyz_map.world_z[ymin:ymax, xmin:xmax][valid].astype(np.float64)
        distances = self.xyz_map.distance_px[ymin:ymax, xmin:xmax][valid].astype(np.float64)
        sources = self.xyz_map.source[ymin:ymax, xmin:xmax][valid]
        return {
            "world_x_m": round(float(np.median(wx)), 8),
            "world_y_m": round(float(np.median(wy)), 8),
            "world_z_m": round(float(np.median(wz)), 8),
            "geo3d_source": "las_image_xyz_map_instance_median",
            "xyz_valid": True,
            "xyz_map_source": "instance_median",
            "xyz_distance_px": round(float(np.median(distances)), 4),
            "xyz_depth_m": None,
            "xyz_miss_reason": None,
            "image_pixel_x": round(float(instance_row["centroid_x_px"]), 4),
            "image_pixel_y": round(float(instance_row["centroid_y_px"]), 4),
            "instance_xyz_sample_count": int(len(wx)),
            "instance_xyz_projected_count": int(np.count_nonzero(sources == 1)),
            "instance_xyz_nearest_count": int(np.count_nonzero(sources == 2)),
            "instance_xyz_distance_px_median": round(float(np.median(distances)), 4),
            "instance_xyz_distance_px_max": round(float(np.max(distances)), 4),
        }

    def to_dict(self) -> dict[str, Any]:
        valid_count = int(np.count_nonzero(self.xyz_map.source))
        projected_count = int(np.count_nonzero(self.xyz_map.source == 1))
        filled_count = int(np.count_nonzero(self.xyz_map.source == 2))
        return {
            "method": "las_image_xyz_map",
            "intrinsics": self.intrinsics.to_dict(),
            "pose": self.pose.to_dict(),
            "las": self.surface_header.to_dict(),
            "xyz_cache_path": self.cache_path,
            "fill_distance_px": self.fill_distance_px,
            "instance_padding_px": self.instance_padding_px,
            "valid_pixel_count": valid_count,
            "projected_las_pixel_count": projected_count,
            "filled_pixel_count": filled_count,
            "valid_pixel_ratio": round(valid_count / float(self.intrinsics.width * self.intrinsics.height), 8),
            "accuracy_note": (
                "LAS points are projected into the DJI image with a z-buffer. Damage coordinates are looked up from "
                "the cached XYZ map. xyz_map_source=nearest_fill means no LAS point landed exactly on that image pixel."
            ),
        }


def build_las_ray_context(
    image_path: str | Path,
    las_path: str | Path,
    mrk_path: str | Path | None = None,
    ray_step_m: float = 0.25,
    max_ray_distance_m: float = 0.20,
) -> LasRayGeo3DContext:
    surface = LasSurfaceIndex(las_path)
    return build_las_ray_context_with_surface(
        image_path=image_path,
        surface=surface,
        mrk_path=mrk_path,
        ray_step_m=ray_step_m,
        max_ray_distance_m=max_ray_distance_m,
    )


def build_las_ray_context_with_surface(
    image_path: str | Path,
    surface: LasSurfaceIndex,
    mrk_path: str | Path | None = None,
    ray_step_m: float = 0.25,
    max_ray_distance_m: float = 0.20,
) -> LasRayGeo3DContext:
    xmp = read_dji_xmp(image_path)
    intrinsics = intrinsics_from_xmp(image_path, xmp)
    pose = pose_from_xmp(image_path, xmp, mrk_path=mrk_path)
    return LasRayGeo3DContext(
        intrinsics=intrinsics,
        pose=pose,
        surface=surface,
        ray_step_m=ray_step_m,
        max_ray_distance_m=max_ray_distance_m,
    )


def build_xyz_map_context_with_surface(
    image_path: str | Path,
    surface: LasSurfaceIndex,
    cache_dir: str | Path,
    mrk_path: str | Path | None = None,
    fill_distance_px: float = 3.0,
    instance_padding_px: int = 0,
    rebuild_cache: bool = False,
) -> XYZMapGeo3DContext:
    xmp = read_dji_xmp(image_path)
    intrinsics = intrinsics_from_xmp(image_path, xmp)
    pose = pose_from_xmp(image_path, xmp, mrk_path=mrk_path)
    cache_path = Path(cache_dir) / f"{Path(image_path).stem}_xyz_map.npz"

    if cache_path.exists() and not rebuild_cache:
        xyz_map = ImageXYZMap.load(cache_path)
    else:
        xyz_map = build_image_xyz_map(
            surface=surface,
            intrinsics=intrinsics,
            pose=pose,
            fill_distance_px=fill_distance_px,
        )
        xyz_map.save(cache_path)

    return XYZMapGeo3DContext(
        intrinsics=intrinsics,
        pose=pose,
        surface_header=surface.header,
        xyz_map=xyz_map,
        cache_path=str(cache_path),
        fill_distance_px=fill_distance_px,
        instance_padding_px=int(instance_padding_px),
    )


def build_xyz_map_context(
    image_path: str | Path,
    las_path: str | Path,
    cache_dir: str | Path,
    mrk_path: str | Path | None = None,
    fill_distance_px: float = 3.0,
    instance_padding_px: int = 0,
    rebuild_cache: bool = False,
) -> XYZMapGeo3DContext:
    surface = LasSurfaceIndex(las_path)
    return build_xyz_map_context_with_surface(
        image_path=image_path,
        surface=surface,
        cache_dir=cache_dir,
        mrk_path=mrk_path,
        fill_distance_px=fill_distance_px,
        instance_padding_px=instance_padding_px,
        rebuild_cache=rebuild_cache,
    )


def build_image_xyz_map(
    surface: LasSurfaceIndex,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    fill_distance_px: float = 3.0,
) -> ImageXYZMap:
    height = intrinsics.height
    width = intrinsics.width
    world_x = np.full((height, width), np.nan, dtype=np.float32)
    world_y = np.full((height, width), np.nan, dtype=np.float32)
    world_z = np.full((height, width), np.nan, dtype=np.float32)
    depth_m = np.full((height, width), np.nan, dtype=np.float32)
    distance_px = np.full((height, width), np.inf, dtype=np.float32)
    source = np.zeros((height, width), dtype=np.uint8)

    rotation = camera_to_world_matrix(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    origin = np.array(pose.center_xyz, dtype=np.float64)
    rel_world = surface.points - origin[None, :]
    camera_points = rel_world @ rotation
    z = camera_points[:, 2]
    in_front = z > 1e-6
    if not np.any(in_front):
        return ImageXYZMap(world_x, world_y, world_z, depth_m, distance_px, source, xyz_map_metadata(surface, intrinsics, pose, 0, 0, fill_distance_px))

    points = surface.points[in_front]
    camera_points = camera_points[in_front]
    z = z[in_front]
    u = intrinsics.focal_length_px * camera_points[:, 0] / z + intrinsics.cx
    v = intrinsics.focal_length_px * camera_points[:, 1] / z + intrinsics.cy
    cols = np.rint(u).astype(np.int64)
    rows = np.rint(v).astype(np.int64)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    if not np.any(inside):
        return ImageXYZMap(world_x, world_y, world_z, depth_m, distance_px, source, xyz_map_metadata(surface, intrinsics, pose, 0, 0, fill_distance_px))

    points = points[inside]
    z = z[inside]
    cols = cols[inside]
    rows = rows[inside]
    linear = rows * width + cols
    order = np.lexsort((z, linear))
    sorted_linear = linear[order]
    first = np.concatenate(([True], sorted_linear[1:] != sorted_linear[:-1]))
    chosen = order[first]

    flat_idx = linear[chosen]
    flat_x = world_x.ravel()
    flat_y = world_y.ravel()
    flat_z = world_z.ravel()
    flat_depth = depth_m.ravel()
    flat_distance = distance_px.ravel()
    flat_source = source.ravel()
    flat_x[flat_idx] = points[chosen, 0].astype(np.float32)
    flat_y[flat_idx] = points[chosen, 1].astype(np.float32)
    flat_z[flat_idx] = points[chosen, 2].astype(np.float32)
    flat_depth[flat_idx] = z[chosen].astype(np.float32)
    flat_distance[flat_idx] = 0.0
    flat_source[flat_idx] = 1

    projected_count = int(len(chosen))
    fill_count = 0
    if fill_distance_px > 0 and projected_count > 0:
        fill_count = fill_nearest_xyz(
            world_x=world_x,
            world_y=world_y,
            world_z=world_z,
            depth_m=depth_m,
            distance_px=distance_px,
            source=source,
            max_distance_px=fill_distance_px,
        )

    metadata = xyz_map_metadata(surface, intrinsics, pose, projected_count, fill_count, fill_distance_px)
    return ImageXYZMap(world_x, world_y, world_z, depth_m, distance_px, source, metadata)


def fill_nearest_xyz(
    world_x: np.ndarray,
    world_y: np.ndarray,
    world_z: np.ndarray,
    depth_m: np.ndarray,
    distance_px: np.ndarray,
    source: np.ndarray,
    max_distance_px: float,
) -> int:
    from scipy.ndimage import distance_transform_edt

    projected = source == 1
    if not np.any(projected):
        return 0

    missing = ~projected
    distances, indices = distance_transform_edt(missing, return_indices=True)
    fill_mask = missing & (distances <= float(max_distance_px))
    if not np.any(fill_mask):
        return 0

    nearest_rows = indices[0][fill_mask]
    nearest_cols = indices[1][fill_mask]
    world_x[fill_mask] = world_x[nearest_rows, nearest_cols]
    world_y[fill_mask] = world_y[nearest_rows, nearest_cols]
    world_z[fill_mask] = world_z[nearest_rows, nearest_cols]
    depth_m[fill_mask] = depth_m[nearest_rows, nearest_cols]
    distance_px[fill_mask] = distances[fill_mask].astype(np.float32)
    source[fill_mask] = 2
    return int(np.count_nonzero(fill_mask))


def xyz_map_metadata(
    surface: LasSurfaceIndex,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    projected_count: int,
    fill_count: int,
    fill_distance_px: float,
) -> dict[str, Any]:
    return {
        "method": "las_image_xyz_map",
        "image_width": intrinsics.width,
        "image_height": intrinsics.height,
        "projected_las_pixel_count": int(projected_count),
        "filled_pixel_count": int(fill_count),
        "fill_distance_px": float(fill_distance_px),
        "intrinsics": intrinsics.to_dict(),
        "pose": pose.to_dict(),
        "las": surface.header.to_dict(),
    }


def read_dji_xmp(path: str | Path) -> dict[str, str]:
    data = Path(path).read_bytes()
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta>")
    if start < 0 or end < 0:
        raise ValueError(f"DJI XMP metadata was not found in {path}")
    text = data[start : end + len(b"</x:xmpmeta>")].decode("utf-8", errors="ignore")
    return {key: value for key, value in re.findall(r"drone-dji:([A-Za-z0-9_]+)=\"([^\"]*)\"", text)}


def intrinsics_from_xmp(image_path: str | Path, xmp: dict[str, str]) -> CameraIntrinsics:
    with Image.open(image_path) as img:
        width, height = img.size
    try:
        focal = float(xmp["CalibratedFocalLength"])
        cx = float(xmp["CalibratedOpticalCenterX"])
        cy = float(xmp["CalibratedOpticalCenterY"])
    except KeyError as exc:
        raise ValueError(f"Missing calibrated camera intrinsic in DJI XMP: {exc}") from exc
    return CameraIntrinsics(
        width=int(width),
        height=int(height),
        focal_length_px=focal,
        cx=cx,
        cy=cy,
        dewarp_data=xmp.get("DewarpData"),
    )


def pose_from_xmp(image_path: str | Path, xmp: dict[str, str], mrk_path: str | Path | None = None) -> CameraPose:
    lat = float(xmp["GpsLatitude"])
    lon = float(xmp["GpsLongitude"])
    alt = float(xmp["AbsoluteAltitude"])
    position_source = "jpg_xmp"
    if mrk_path is not None:
        photo_index = image_index_from_name(Path(image_path).name)
        mrk_positions = read_mrk_positions(mrk_path)
        if photo_index in mrk_positions:
            lat, lon, alt = mrk_positions[photo_index]
            position_source = "timestamp_mrk"

    center_x, center_y = wgs84_to_epsg5186(lon, lat)
    yaw = float(xmp.get("GimbalYawDegree", xmp.get("FlightYawDegree", 0.0)))
    pitch = float(xmp.get("GimbalPitchDegree", xmp.get("FlightPitchDegree", 0.0)))
    roll = float(xmp.get("GimbalRollDegree", xmp.get("FlightRollDegree", 0.0)))
    return CameraPose(
        source_path=str(image_path),
        center_xyz=(float(center_x), float(center_y), float(alt)),
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        position_source=position_source,
        orientation_source="jpg_xmp_gimbal",
        rtk_std_lon_m=parse_optional_float(xmp.get("RtkStdLon")),
        rtk_std_lat_m=parse_optional_float(xmp.get("RtkStdLat")),
        rtk_std_hgt_m=parse_optional_float(xmp.get("RtkStdHgt")),
    )


def read_mrk_positions(path: str | Path) -> dict[int, tuple[float, float, float]]:
    positions: dict[int, tuple[float, float, float]] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 12:
            continue
        try:
            index = int(parts[0])
            lat_i = parts.index("Lat")
            lon_i = parts.index("Lon")
            ellh_i = parts.index("Ellh")
            lat = float(parts[lat_i - 1])
            lon = float(parts[lon_i - 1])
            ellh = float(parts[ellh_i - 1])
        except (ValueError, IndexError):
            continue
        positions[index] = (lat, lon, ellh)
    return positions


def image_index_from_name(name: str) -> int:
    match = re.search(r"_(\d{4})_", name)
    if match is None:
        raise ValueError(f"Could not parse DJI image index from {name}")
    return int(match.group(1))


def wgs84_to_epsg5186(lon: float, lat: float) -> tuple[float, float]:
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError("pyproj is required for WGS84 to EPSG:5186 conversion. Install requirements.txt.") from exc

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def camera_to_world_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    forward = np.array(
        [
            math.cos(pitch) * math.sin(yaw),
            math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)

    if abs(roll) > 1e-12:
        cr = math.cos(roll)
        sr = math.sin(roll)
        right, down = right * cr + down * sr, -right * sr + down * cr

    return np.column_stack([right, down, forward])


def read_las_header(path: str | Path) -> LasHeader:
    las_path = Path(path)
    with las_path.open("rb") as fp:
        header = fp.read(227)
        if header[:4] != b"LASF":
            raise ValueError(f"Not a LAS file: {las_path}")
        version = f"{header[24]}.{header[25]}"
        offset_to_points = struct.unpack_from("<I", header, 96)[0]
        num_vlrs = struct.unpack_from("<I", header, 100)[0]
        point_format = int(header[104])
        point_record_length = struct.unpack_from("<H", header, 105)[0]
        point_count = struct.unpack_from("<I", header, 107)[0]
        scale = struct.unpack_from("<3d", header, 131)
        offset = struct.unpack_from("<3d", header, 155)
        max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<6d", header, 179)

        crs_parts: list[str] = []
        fp.seek(227)
        for _ in range(num_vlrs):
            vlr_header = fp.read(54)
            if len(vlr_header) < 54:
                break
            _, user_id_raw, record_id, payload_len, _ = struct.unpack("<H16sHH32s", vlr_header)
            user_id = user_id_raw.split(b"\0", 1)[0].decode("ascii", errors="ignore")
            payload = fp.read(payload_len)
            if user_id == "LASF_Projection" and record_id == 34737:
                crs_parts.append(payload.decode("ascii", errors="ignore").strip("\0"))

    return LasHeader(
        path=str(las_path),
        version=version,
        point_format=point_format,
        point_record_length=point_record_length,
        point_count=point_count,
        offset_to_points=offset_to_points,
        scale=tuple(float(v) for v in scale),
        offset=tuple(float(v) for v in offset),
        bounds_min=(float(min_x), float(min_y), float(min_z)),
        bounds_max=(float(max_x), float(max_y), float(max_z)),
        crs_text=" ".join(crs_parts) if crs_parts else None,
    )


def read_las_points(header: LasHeader) -> np.ndarray:
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<i4", "<i4", "<i4"],
            "offsets": [0, 4, 8],
            "itemsize": header.point_record_length,
        }
    )
    with Path(header.path).open("rb") as fp:
        fp.seek(header.offset_to_points)
        records = np.fromfile(fp, dtype=dtype, count=header.point_count)

    points = np.empty((len(records), 3), dtype=np.float64)
    points[:, 0] = records["x"].astype(np.float64) * header.scale[0] + header.offset[0]
    points[:, 1] = records["y"].astype(np.float64) * header.scale[1] + header.offset[1]
    points[:, 2] = records["z"].astype(np.float64) * header.scale[2] + header.offset[2]
    return points


def ray_box_interval(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
) -> tuple[float | None, float | None]:
    t_min = -math.inf
    t_max = math.inf
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-12:
            if origin[axis] < bounds_min[axis] or origin[axis] > bounds_max[axis]:
                return None, None
            continue
        t1 = (bounds_min[axis] - origin[axis]) / direction[axis]
        t2 = (bounds_max[axis] - origin[axis]) / direction[axis]
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return None, None
    return float(t_min), float(t_max)


def empty_hit(reason: str) -> dict[str, Any]:
    return {
        "world_x_m": None,
        "world_y_m": None,
        "world_z_m": None,
        "geo3d_source": "dji_pose_las_ray",
        "las_point_index": None,
        "ray_t_m": None,
        "ray_surface_distance_m": None,
        "ray_hit": False,
        "ray_miss_reason": reason,
    }


def empty_xyz_map_lookup(reason: str, x: float, y: float) -> dict[str, Any]:
    return {
        "world_x_m": None,
        "world_y_m": None,
        "world_z_m": None,
        "geo3d_source": "las_image_xyz_map",
        "xyz_valid": False,
        "xyz_map_source": None,
        "xyz_distance_px": None,
        "xyz_depth_m": None,
        "xyz_miss_reason": reason,
        "image_pixel_x": round(float(x), 4),
        "image_pixel_y": round(float(y), 4),
    }


def empty_instance_xyz(reason: str) -> dict[str, Any]:
    return {
        "world_x_m": None,
        "world_y_m": None,
        "world_z_m": None,
        "geo3d_source": "las_image_xyz_map_instance_median",
        "xyz_valid": False,
        "xyz_map_source": None,
        "xyz_distance_px": None,
        "xyz_depth_m": None,
        "xyz_miss_reason": reason,
        "image_pixel_x": None,
        "image_pixel_y": None,
        "instance_xyz_sample_count": 0,
        "instance_xyz_projected_count": 0,
        "instance_xyz_nearest_count": 0,
        "instance_xyz_distance_px_median": None,
        "instance_xyz_distance_px_max": None,
    }


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def write_camera_pose_json(context: LasRayGeo3DContext, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(context.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
