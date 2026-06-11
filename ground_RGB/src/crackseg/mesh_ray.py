from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from skimage.measure import find_contours

from .las_ray import (
    CameraIntrinsics,
    CameraPose,
    camera_to_world_matrix,
    intrinsics_from_xmp,
    pose_from_xmp,
    read_dji_xmp,
)


class MeshSurfaceIndex:
    def __init__(self, path: str | Path, ray_backend: str = "trimesh", warp_device: str = "cuda:0"):
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("trimesh and rtree are required for mesh ray intersection. Install requirements.txt.") from exc

        ray_backend = str(ray_backend)
        if ray_backend not in {"trimesh", "warp", "auto"}:
            raise ValueError("ray_backend must be 'trimesh', 'warp', or 'auto'")

        self.path = str(path)
        self.ray_backend_requested = ray_backend
        self.warp_device = str(warp_device)
        self.ray_backend = "trimesh"
        self.ray_backend_error: str | None = None
        self._warp_raycaster = None

        loaded = trimesh.load_mesh(self.path, process=False)
        if hasattr(loaded, "geometry"):
            loaded = loaded.dump(concatenate=True)
        if loaded.faces is None or len(loaded.faces) == 0:
            raise ValueError(f"Mesh has no faces: {self.path}")

        self.mesh = loaded
        self.bounds = np.asarray(self.mesh.bounds, dtype=np.float64)
        self.vertex_count = int(len(self.mesh.vertices))
        self.face_count = int(len(self.mesh.faces))
        if ray_backend in {"warp", "auto"}:
            try:
                from .warp_ray import WarpMeshRaycaster

                self._warp_raycaster = WarpMeshRaycaster(
                    vertices=np.asarray(self.mesh.vertices),
                    faces=np.asarray(self.mesh.faces),
                    device=self.warp_device,
                    origin_offset=0.5 * (self.bounds[0] + self.bounds[1]),
                )
                self.ray_backend = "warp"
            except Exception as exc:
                self.ray_backend_error = f"{type(exc).__name__}: {exc}"
                if ray_backend == "warp":
                    raise RuntimeError(f"Failed to initialize Warp mesh ray backend: {self.ray_backend_error}") from exc

    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray) -> dict[str, Any]:
        hits = self.intersect_rays(origin[None, :], direction[None, :])
        if not hits:
            return empty_mesh_hit("mesh_ray_no_hit", None, None)
        return hits[0]

    def intersect_rays(self, origins: np.ndarray, directions: np.ndarray) -> list[dict[str, Any]]:
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        if self._warp_raycaster is not None:
            return self._intersect_rays_warp(origins, directions)

        locations, ray_indices, face_indices = self.mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
        )
        by_ray: dict[int, tuple[np.ndarray, int, float]] = {}
        for location, ray_index, face_index in zip(locations, ray_indices, face_indices):
            ray_index = int(ray_index)
            t = float(np.dot(location - origins[ray_index], directions[ray_index]))
            current = by_ray.get(ray_index)
            if current is None or t < current[2]:
                by_ray[ray_index] = (np.asarray(location, dtype=np.float64), int(face_index), t)

        rows: list[dict[str, Any]] = []
        for ray_index in range(len(origins)):
            hit = by_ray.get(ray_index)
            if hit is None:
                row = empty_mesh_hit("mesh_ray_no_hit", None, None)
                row["mesh_ray_backend"] = self.ray_backend
                rows.append(row)
                continue
            location, face_index, t = hit
            rows.append(
                {
                    "world_x_m": round(float(location[0]), 8),
                    "world_y_m": round(float(location[1]), 8),
                    "world_z_m": round(float(location[2]), 8),
                    "geo3d_source": "dji_pose_mesh_ray",
                    "xyz_valid": True,
                    "mesh_ray_hit": True,
                    "mesh_face_index": int(face_index),
                    "mesh_ray_t_m": round(float(t), 6),
                    "mesh_ray_backend": self.ray_backend,
                    "xyz_miss_reason": None,
                }
            )
        return rows

    def _intersect_rays_warp(self, origins: np.ndarray, directions: np.ndarray) -> list[dict[str, Any]]:
        if self._warp_raycaster is None:
            raise RuntimeError("Warp raycaster is not initialized.")

        hit_flags, hit_t, hit_faces, hit_points = self._warp_raycaster.intersect_rays(origins, directions)
        rows: list[dict[str, Any]] = []
        for ray_index in range(len(origins)):
            if int(hit_flags[ray_index]) == 0:
                row = empty_mesh_hit("mesh_ray_no_hit", None, None)
                row["mesh_ray_backend"] = self.ray_backend
                rows.append(row)
                continue

            location = np.asarray(hit_points[ray_index], dtype=np.float64)
            rows.append(
                {
                    "world_x_m": round(float(location[0]), 8),
                    "world_y_m": round(float(location[1]), 8),
                    "world_z_m": round(float(location[2]), 8),
                    "geo3d_source": "dji_pose_mesh_ray",
                    "xyz_valid": True,
                    "mesh_ray_hit": True,
                    "mesh_face_index": int(hit_faces[ray_index]),
                    "mesh_ray_t_m": round(float(hit_t[ray_index]), 6),
                    "mesh_ray_backend": self.ray_backend,
                    "xyz_miss_reason": None,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        data = {
            "path": self.path,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "bounds_min": tuple(float(v) for v in self.bounds[0]),
            "bounds_max": tuple(float(v) for v in self.bounds[1]),
            "ray_backend_requested": self.ray_backend_requested,
            "ray_backend": self.ray_backend,
            "warp_device": self.warp_device if self.ray_backend_requested in {"warp", "auto"} else None,
            "ray_backend_error": self.ray_backend_error,
        }
        if self._warp_raycaster is not None:
            data["warp"] = self._warp_raycaster.to_dict()
        return data


@dataclass
class MeshRayGeo3DContext:
    intrinsics: CameraIntrinsics
    pose: CameraPose
    surface: MeshSurfaceIndex
    instance_sample_count: int = 128
    node_sample_count: int = 256
    ray_batch_size: int = 128
    representative_mode: str = "centroid"
    profile: bool = False
    timings: dict[str, float] = field(default_factory=dict)

    def image_pixel_to_world3d(self, x: float, y: float) -> dict[str, Any]:
        hit = self._intersect_pixel_batch(np.array([[float(x), float(y)]], dtype=np.float64), desc="centroid mesh rays")[0]
        hit["image_pixel_x"] = round(float(x), 4)
        hit["image_pixel_y"] = round(float(y), 4)
        return hit

    def image_regions_to_world3d(self, instance_rows: list[dict[str, Any]], pred_mask: np.ndarray) -> list[dict[str, Any]]:
        total_start = time.perf_counter()
        prepare_start = time.perf_counter()
        prepared: list[dict[str, Any]] = []
        sample_xy_parts: list[np.ndarray] = []
        node_xy_parts: list[np.ndarray] = []

        for instance_index, instance_row in enumerate(instance_rows):
            class_id = int(instance_row["class_id"])
            xmin = max(0, int(instance_row["bbox_xmin_px"]))
            ymin = max(0, int(instance_row["bbox_ymin_px"]))
            xmax = min(self.intrinsics.width, int(instance_row["bbox_xmax_px"]))
            ymax = min(self.intrinsics.height, int(instance_row["bbox_ymax_px"]))
            item: dict[str, Any] = {
                "instance_index": instance_index,
                "row": instance_row,
                "sample_start": 0,
                "sample_count": 0,
                "node_start": 0,
                "node_count": 0,
                "fallback_reason": None,
            }
            if xmin >= xmax or ymin >= ymax:
                item["fallback_reason"] = "empty_instance_bbox"
                prepared.append(item)
                continue

            submask = pred_mask[ymin:ymax, xmin:xmax] == class_id
            ys, xs = np.nonzero(submask)
            if len(xs) == 0:
                item["fallback_reason"] = "empty_instance_mask"
                prepared.append(item)
                continue

            sample_xy = self._representative_pixels(instance_row, xs, ys, xmin, ymin)
            node_xy = extract_contour_nodes(submask, offset_x=xmin, offset_y=ymin, max_count=self.node_sample_count)

            item["sample_start"] = sum(len(part) for part in sample_xy_parts)
            item["sample_count"] = len(sample_xy)
            item["node_start"] = sum(len(part) for part in node_xy_parts)
            item["node_count"] = len(node_xy)
            sample_xy_parts.append(sample_xy)
            if len(node_xy) > 0:
                node_xy_parts.append(node_xy)
            prepared.append(item)
        self._add_timing("mesh_prepare_regions_sec", time.perf_counter() - prepare_start)

        sample_xy_all = np.vstack(sample_xy_parts) if sample_xy_parts else np.empty((0, 2), dtype=np.float64)
        node_xy_all = np.vstack(node_xy_parts) if node_xy_parts else np.empty((0, 2), dtype=np.float64)
        self.timings["mesh_instance_ray_count"] = float(len(sample_xy_all))
        self.timings["mesh_node_ray_count"] = float(len(node_xy_all))
        self.timings["mesh_total_ray_count"] = float(len(sample_xy_all) + len(node_xy_all))

        sample_start = time.perf_counter()
        sample_hits = self._intersect_pixel_batch(sample_xy_all, desc="instance mesh rays")
        self._add_timing("mesh_instance_intersection_sec", time.perf_counter() - sample_start)

        node_start = time.perf_counter()
        node_hits = self._intersect_pixel_batch(node_xy_all, desc="node mesh rays")
        self._add_timing("mesh_node_intersection_sec", time.perf_counter() - node_start)

        assemble_start = time.perf_counter()
        results: list[dict[str, Any]] = []
        for item in prepared:
            row = item["row"]
            if item["fallback_reason"] is not None:
                results.append(self._centroid_fallback(row, str(item["fallback_reason"])))
                continue

            sample_start = int(item["sample_start"])
            sample_count = int(item["sample_count"])
            node_start = int(item["node_start"])
            node_count = int(item["node_count"])
            sample_slice = sample_hits[sample_start : sample_start + sample_count]
            node_xy = node_xy_all[node_start : node_start + node_count]
            node_slice = node_hits[node_start : node_start + node_count]
            results.append(self._result_from_hits(row, sample_slice, node_xy, node_slice))

        self._add_timing("mesh_assemble_results_sec", time.perf_counter() - assemble_start)
        self._add_timing("mesh_total_sec", time.perf_counter() - total_start)
        return results

    def image_region_to_world3d(self, instance_row: dict[str, Any], pred_mask: np.ndarray) -> dict[str, Any]:
        class_id = int(instance_row["class_id"])
        xmin = max(0, int(instance_row["bbox_xmin_px"]))
        ymin = max(0, int(instance_row["bbox_ymin_px"]))
        xmax = min(self.intrinsics.width, int(instance_row["bbox_xmax_px"]))
        ymax = min(self.intrinsics.height, int(instance_row["bbox_ymax_px"]))
        if xmin >= xmax or ymin >= ymax:
            return self._centroid_fallback(instance_row, "empty_instance_bbox")

        submask = pred_mask[ymin:ymax, xmin:xmax] == class_id
        ys, xs = np.nonzero(submask)
        if len(xs) == 0:
            return self._centroid_fallback(instance_row, "empty_instance_mask")

        sample_xy = self._representative_pixels(instance_row, xs, ys, xmin, ymin)
        node_xy = extract_contour_nodes(submask, offset_x=xmin, offset_y=ymin, max_count=self.node_sample_count)
        hits = self._intersect_pixel_batch(sample_xy, desc="instance mesh rays")
        node_hits = self._intersect_pixel_batch(node_xy, desc="node mesh rays")
        return self._result_from_hits(instance_row, hits, node_xy, node_hits)

    def _representative_pixels(
        self,
        instance_row: dict[str, Any],
        xs: np.ndarray,
        ys: np.ndarray,
        xmin: int,
        ymin: int,
    ) -> np.ndarray:
        if self.representative_mode == "centroid":
            return np.array(
                [[float(instance_row["centroid_x_px"]), float(instance_row["centroid_y_px"])]],
                dtype=np.float64,
            )
        if self.representative_mode != "median":
            raise ValueError("representative_mode must be 'centroid' or 'median'")

        xs = xs.astype(np.float64) + float(xmin)
        ys = ys.astype(np.float64) + float(ymin)
        selected = sample_indices(len(xs), self.instance_sample_count)
        return np.column_stack((xs[selected], ys[selected])).astype(np.float64)

    def _result_from_hits(
        self,
        instance_row: dict[str, Any],
        hits: list[dict[str, Any]],
        node_xy: np.ndarray,
        node_hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        valid_hits = [hit for hit in hits if hit.get("mesh_ray_hit")]
        if not valid_hits:
            result = self._centroid_fallback(instance_row, "mesh_ray_no_hit_in_instance_samples")
            result.update(
                {
                    "instance_mesh_sample_count": int(len(hits)),
                    "instance_mesh_hit_count": 0,
                    "instance_mesh_hit_ratio": 0.0,
                }
            )
            return result

        wx = np.array([float(hit["world_x_m"]) for hit in valid_hits], dtype=np.float64)
        wy = np.array([float(hit["world_y_m"]) for hit in valid_hits], dtype=np.float64)
        wz = np.array([float(hit["world_z_m"]) for hit in valid_hits], dtype=np.float64)
        t = np.array([float(hit["mesh_ray_t_m"]) for hit in valid_hits], dtype=np.float64)
        face_indices = [int(hit["mesh_face_index"]) for hit in valid_hits]
        if self.representative_mode == "centroid":
            representative_hit = valid_hits[0]
            world_x = float(representative_hit["world_x_m"])
            world_y = float(representative_hit["world_y_m"])
            world_z = float(representative_hit["world_z_m"])
            ray_t = float(representative_hit["mesh_ray_t_m"])
            mesh_face_index = int(representative_hit["mesh_face_index"])
            geo3d_source = "dji_pose_mesh_ray_centroid"
        else:
            world_x = float(np.median(wx))
            world_y = float(np.median(wy))
            world_z = float(np.median(wz))
            ray_t = float(np.median(t))
            mesh_face_index = None
            geo3d_source = "dji_pose_mesh_ray_instance_median"
        result = {
            "world_x_m": round(world_x, 8),
            "world_y_m": round(world_y, 8),
            "world_z_m": round(world_z, 8),
            "geo3d_source": geo3d_source,
            "xyz_valid": True,
            "mesh_ray_hit": True,
            "mesh_face_index": mesh_face_index,
            "mesh_ray_t_m": round(ray_t, 6),
            "mesh_ray_backend": self.surface.ray_backend,
            "xyz_miss_reason": None,
            "image_pixel_x": round(float(instance_row["centroid_x_px"]), 4),
            "image_pixel_y": round(float(instance_row["centroid_y_px"]), 4),
            "instance_mesh_sample_count": int(len(hits)),
            "instance_mesh_hit_count": int(len(valid_hits)),
            "instance_mesh_hit_ratio": round(float(len(valid_hits) / len(hits)), 6),
            "instance_mesh_unique_face_count": int(len(set(face_indices))),
            "instance_mesh_ray_t_min_m": round(float(np.min(t)), 6),
            "instance_mesh_ray_t_max_m": round(float(np.max(t)), 6),
        }
        result.update(self._nodes_to_world3d(node_xy, node_hits))
        return result

    def _nodes_to_world3d(self, node_xy: np.ndarray, hits: list[dict[str, Any]]) -> dict[str, Any]:
        if len(node_xy) == 0:
            return empty_mesh_nodes("no_instance_contour_nodes")

        image_nodes: list[list[float]] = []
        world_nodes: list[list[float | None]] = []
        valid_count = 0
        for (x, y), hit in zip(node_xy, hits):
            image_nodes.append([round(float(x), 4), round(float(y), 4)])
            if hit.get("mesh_ray_hit"):
                valid_count += 1
                world_nodes.append(
                    [
                        round(float(hit["world_x_m"]), 8),
                        round(float(hit["world_y_m"]), 8),
                        round(float(hit["world_z_m"]), 8),
                    ]
                )
            else:
                world_nodes.append([None, None, None])

        return {
            "node_count": int(len(node_xy)),
            "node_xyz_valid_count": int(valid_count),
            "node_xyz_hit_ratio": round(float(valid_count / len(node_xy)), 6),
            "nodes_image_xy_json": json.dumps(image_nodes, separators=(",", ":")),
            "nodes_world_xyz_json": json.dumps(world_nodes, separators=(",", ":")),
            "nodes_geo3d_source": "dji_pose_mesh_ray_contour_nodes",
            "nodes_miss_reason": None if valid_count > 0 else "mesh_ray_no_hit_for_nodes",
        }

    def _intersect_pixel_batch(self, xy: np.ndarray, desc: str = "mesh rays") -> list[dict[str, Any]]:
        if len(xy) == 0:
            return []
        xy = np.asarray(xy, dtype=np.float64)
        ray_start = time.perf_counter()
        directions = pixels_to_world_rays(self.intrinsics, self.pose, xy)
        self._add_timing("mesh_ray_direction_sec", time.perf_counter() - ray_start)
        origin = np.asarray(self.pose.center_xyz, dtype=np.float64)
        results: list[dict[str, Any]] = []
        batch_size = max(1, int(self.ray_batch_size))
        ranges = range(0, len(xy), batch_size)
        if self.profile:
            from tqdm import tqdm

            ranges = tqdm(ranges, desc=desc, unit="chunk", leave=False)
        for start in ranges:
            end = min(start + batch_size, len(xy))
            origins = np.repeat(origin[None, :], end - start, axis=0)
            results.extend(self.surface.intersect_rays(origins, directions[start:end]))
        return results

    def _add_timing(self, key: str, elapsed_sec: float) -> None:
        self.timings[key] = self.timings.get(key, 0.0) + float(elapsed_sec)

    def _centroid_fallback(self, instance_row: dict[str, Any], reason: str) -> dict[str, Any]:
        result = self.image_pixel_to_world3d(float(instance_row["centroid_x_px"]), float(instance_row["centroid_y_px"]))
        if not result.get("mesh_ray_hit"):
            result["xyz_miss_reason"] = reason
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "dji_pose_mesh_ray",
            "intrinsics": self.intrinsics.to_dict(),
            "pose": self.pose.to_dict(),
            "mesh": self.surface.to_dict(),
            "instance_sample_count": self.instance_sample_count,
            "node_sample_count": self.node_sample_count,
            "ray_batch_size": self.ray_batch_size,
            "representative_mode": self.representative_mode,
            "profile_timings_sec": self.timings if self.timings else None,
            "accuracy_note": (
                "Damage instance coordinates are estimated by intersecting DJI camera rays with a triangulated mesh. "
                "By default, the reported instance coordinate is the mesh hit at the 2D instance centroid. "
                "representative_mode=median uses the median of sampled mesh hits inside the damage mask. "
                "nodes_world_xyz_json stores sampled contour node coordinates as [[x,y,z], ...]."
            ),
        }


def build_mesh_ray_context(
    image_path: str | Path,
    mesh_path: str | Path,
    mrk_path: str | Path | None = None,
    instance_sample_count: int = 128,
    node_sample_count: int = 256,
    ray_batch_size: int = 128,
    representative_mode: str = "centroid",
    ray_backend: str = "trimesh",
    warp_device: str = "cuda:0",
    profile: bool = False,
) -> MeshRayGeo3DContext:
    surface = MeshSurfaceIndex(mesh_path, ray_backend=ray_backend, warp_device=warp_device)
    return build_mesh_ray_context_with_surface(
        image_path=image_path,
        surface=surface,
        mrk_path=mrk_path,
        instance_sample_count=instance_sample_count,
        node_sample_count=node_sample_count,
        ray_batch_size=ray_batch_size,
        representative_mode=representative_mode,
        profile=profile,
    )


def build_mesh_ray_context_with_surface(
    image_path: str | Path,
    surface: MeshSurfaceIndex,
    mrk_path: str | Path | None = None,
    instance_sample_count: int = 128,
    node_sample_count: int = 256,
    ray_batch_size: int = 128,
    representative_mode: str = "centroid",
    profile: bool = False,
) -> MeshRayGeo3DContext:
    xmp = read_dji_xmp(image_path)
    intrinsics = intrinsics_from_xmp(image_path, xmp)
    pose = pose_from_xmp(image_path, xmp, mrk_path=mrk_path)
    return MeshRayGeo3DContext(
        intrinsics=intrinsics,
        pose=pose,
        surface=surface,
        instance_sample_count=int(instance_sample_count),
        node_sample_count=int(node_sample_count),
        ray_batch_size=int(ray_batch_size),
        representative_mode=str(representative_mode),
        profile=bool(profile),
    )


def pixels_to_world_rays(intrinsics: CameraIntrinsics, pose: CameraPose, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    camera_rays = np.empty((len(xy), 3), dtype=np.float64)
    camera_rays[:, 0] = (xy[:, 0] - intrinsics.cx) / intrinsics.focal_length_px
    camera_rays[:, 1] = (xy[:, 1] - intrinsics.cy) / intrinsics.focal_length_px
    camera_rays[:, 2] = 1.0
    camera_rays /= np.linalg.norm(camera_rays, axis=1)[:, None]
    rotation = camera_to_world_matrix(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    world_rays = camera_rays @ rotation.T
    return world_rays / np.linalg.norm(world_rays, axis=1)[:, None]


def sample_indices(length: int, max_count: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.int64)
    if max_count <= 0 or length <= max_count:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, int(max_count), dtype=np.int64))


def extract_contour_nodes(submask: np.ndarray, offset_x: int, offset_y: int, max_count: int) -> np.ndarray:
    if submask.shape[0] < 2 or submask.shape[1] < 2:
        contours = []
    else:
        contours = find_contours(submask.astype(np.uint8), 0.5)
    if contours:
        contour = max(contours, key=len)
        nodes = np.column_stack((contour[:, 1] + float(offset_x), contour[:, 0] + float(offset_y)))
    else:
        ys, xs = np.nonzero(submask)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float64)
        nodes = np.column_stack((xs.astype(np.float64) + float(offset_x), ys.astype(np.float64) + float(offset_y)))

    selected = sample_indices(len(nodes), max_count)
    return nodes[selected].astype(np.float64)


def empty_mesh_nodes(reason: str) -> dict[str, Any]:
    return {
        "node_count": 0,
        "node_xyz_valid_count": 0,
        "node_xyz_hit_ratio": 0.0,
        "nodes_image_xy_json": "[]",
        "nodes_world_xyz_json": "[]",
        "nodes_geo3d_source": "dji_pose_mesh_ray_contour_nodes",
        "nodes_miss_reason": reason,
    }


def empty_mesh_hit(reason: str, x: float | None, y: float | None) -> dict[str, Any]:
    result = {
        "world_x_m": None,
        "world_y_m": None,
        "world_z_m": None,
        "geo3d_source": "dji_pose_mesh_ray",
        "xyz_valid": False,
        "mesh_ray_hit": False,
        "mesh_face_index": None,
        "mesh_ray_t_m": None,
        "xyz_miss_reason": reason,
    }
    if x is not None and y is not None:
        result["image_pixel_x"] = round(float(x), 4)
        result["image_pixel_y"] = round(float(y), 4)
    return result
