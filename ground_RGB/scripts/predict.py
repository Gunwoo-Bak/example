#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crack/damage segmentation for one RGB image.")
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path, for example outputs/dachung_finetune/best.pt.")
    parser.add_argument("--config", default="configs/dachung.yaml", help="Config YAML path.")
    parser.add_argument("--output-dir", default="outputs/predictions", help="Directory for prediction outputs.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omitted for automatic selection.")
    parser.add_argument("--tile-size", type=int, default=None, help="Override inference tile size.")
    parser.add_argument("--stride", type=int, default=None, help="Override inference stride.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for mask color.")
    parser.add_argument("--orthophoto", default=None, help="Reference GeoTIFF orthophoto used to project image pixels to map coordinates.")
    parser.add_argument("--gcp-csv", default=None, help="CSV containing GCP X/Y/Z values used to interpolate world_z_m.")
    parser.add_argument("--homography", default=None, help="Optional precomputed image-to-orthophoto homography JSON.")
    parser.add_argument("--save-homography", default=None, help="Optional path for saving the estimated homography JSON.")
    parser.add_argument("--las", default=None, help="LAS point cloud used for DJI camera ray intersection and world XYZ extraction.")
    parser.add_argument("--mesh", default=None, help="OBJ/PLY mesh surface used for DJI camera ray intersection and world XYZ extraction.")
    parser.add_argument("--mrk", default=None, help="Optional DJI Timestamp.MRK file used to override JPG XMP camera positions.")
    parser.add_argument(
        "--geo3d-mode",
        default="xyz-map",
        choices=["xyz-map", "ray", "mesh-ray"],
        help="3D extraction mode. xyz-map/ray use LAS; mesh-ray uses a triangulated OBJ/PLY mesh.",
    )
    parser.add_argument("--xyz-cache-dir", default=None, help="Directory for cached per-image XYZ maps.")
    parser.add_argument("--xyz-fill-distance-px", type=float, default=3.0, help="Nearest-fill radius for empty XYZ map pixels.")
    parser.add_argument("--instance-xyz-padding-px", type=int, default=0, help="Padding around each damage instance when computing median XYZ.")
    parser.add_argument("--rebuild-xyz-cache", action="store_true", help="Rebuild XYZ map cache even if it already exists.")
    parser.add_argument("--ray-step-m", type=float, default=0.25, help="Sampling step along each camera ray for LAS intersection.")
    parser.add_argument("--max-ray-distance-m", type=float, default=0.20, help="Maximum accepted LAS point distance from a projected ray.")
    parser.add_argument("--mesh-instance-samples", type=int, default=128, help="Maximum damage-mask pixels sampled per instance when --mesh-representative-mode median is used.")
    parser.add_argument("--mesh-node-samples", type=int, default=256, help="Maximum contour nodes stored per damage instance for mesh-ray coordinates.")
    parser.add_argument("--mesh-ray-batch-size", type=int, default=128, help="Number of mesh rays intersected per chunk in mesh-ray mode.")
    parser.add_argument(
        "--mesh-ray-backend",
        default="trimesh",
        choices=["trimesh", "warp", "auto"],
        help="Ray intersection backend for mesh-ray mode. trimesh is CPU, warp is GPU/CPU via NVIDIA Warp, auto tries warp then falls back.",
    )
    parser.add_argument("--warp-device", default="cuda:0", help="Warp device for --mesh-ray-backend warp/auto, for example cuda:0 or cpu.")
    parser.add_argument(
        "--mesh-representative-mode",
        default="centroid",
        choices=["centroid", "median"],
        help="How to compute instance world_x/y/z. centroid uses one ray at the 2D centroid; median samples damage pixels.",
    )
    parser.add_argument("--profile", action="store_true", help="Print stage timing and show mesh-ray progress bars.")
    parser.add_argument("--save-damage-pixels-3d", action="store_true", help="Save foreground damage mask pixels with world XYZ values.")
    parser.add_argument("--pixel-xyz-stride", type=int, default=1, help="Stride for exported damage pixel XYZ rows.")
    parser.add_argument(
        "--gcp-height-method",
        default="idw",
        choices=["idw", "plane"],
        help="Method for interpolating Z from GCP points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    from crackseg.export import save_prediction_outputs
    from crackseg.geo3d import build_geo3d_context
    from crackseg.las_ray import build_las_ray_context, build_xyz_map_context
    from crackseg.mesh_ray import build_mesh_ray_context
    from crackseg.inference import load_model_bundle, sliding_window_inference
    from crackseg.io import read_image_rgb

    start = time.perf_counter()
    bundle = load_model_bundle(
        checkpoint=args.checkpoint,
        config_path=args.config,
        device=args.device,
        tile_size=args.tile_size,
        stride=args.stride,
    )
    record_timing(timings, "script_load_model_sec", start)

    start = time.perf_counter()
    image = read_image_rgb(args.image)
    record_timing(timings, "script_read_image_sec", start)

    start = time.perf_counter()
    pred_mask, confidence, _ = sliding_window_inference(image, bundle)
    record_timing(timings, "script_inference_sec", start)
    stem = Path(args.image).stem
    geo3d_context = None
    geo3d_mode = args.geo3d_mode
    if args.mesh is not None and args.las is None and geo3d_mode == "xyz-map":
        geo3d_mode = "mesh-ray"

    if geo3d_mode == "mesh-ray":
        if args.mesh is None:
            raise ValueError("--mesh is required when --geo3d-mode mesh-ray")
        start = time.perf_counter()
        geo3d_context = build_mesh_ray_context(
            image_path=args.image,
            mesh_path=args.mesh,
            mrk_path=args.mrk,
            instance_sample_count=args.mesh_instance_samples,
            node_sample_count=args.mesh_node_samples,
            ray_batch_size=args.mesh_ray_batch_size,
            representative_mode=args.mesh_representative_mode,
            ray_backend=args.mesh_ray_backend,
            warp_device=args.warp_device,
            profile=args.profile,
        )
        record_timing(timings, "script_build_mesh_context_sec", start)
    elif args.las is not None:
        if geo3d_mode == "xyz-map":
            xyz_cache_dir = args.xyz_cache_dir or str(Path(args.output_dir) / "xyz_maps")
            start = time.perf_counter()
            geo3d_context = build_xyz_map_context(
                image_path=args.image,
                las_path=args.las,
                cache_dir=xyz_cache_dir,
                mrk_path=args.mrk,
                fill_distance_px=args.xyz_fill_distance_px,
                instance_padding_px=args.instance_xyz_padding_px,
                rebuild_cache=args.rebuild_xyz_cache,
            )
            record_timing(timings, "script_build_las_xyz_context_sec", start)
        elif geo3d_mode == "ray":
            start = time.perf_counter()
            geo3d_context = build_las_ray_context(
                image_path=args.image,
                las_path=args.las,
                mrk_path=args.mrk,
                ray_step_m=args.ray_step_m,
                max_ray_distance_m=args.max_ray_distance_m,
            )
            record_timing(timings, "script_build_las_ray_context_sec", start)
    elif args.orthophoto is not None:
        save_homography = args.save_homography
        if save_homography is None and args.homography is None:
            save_homography = str(Path(args.output_dir) / "homographies" / f"{stem}_image_to_orthophoto.json")
        start = time.perf_counter()
        geo3d_context = build_geo3d_context(
            source_path=args.image,
            orthophoto_path=args.orthophoto,
            gcp_csv=args.gcp_csv,
            homography_path=args.homography,
            save_homography_path=save_homography,
            height_method=args.gcp_height_method,
        )
        record_timing(timings, "script_build_orthophoto_context_sec", start)

    start = time.perf_counter()
    paths = save_prediction_outputs(
        image=image,
        pred_mask=pred_mask,
        confidence=confidence,
        output_dir=args.output_dir,
        stem=stem,
        config=bundle.config,
        alpha=args.alpha,
        source_path=args.image,
        geo3d_context=geo3d_context,
        save_damage_pixels_3d=args.save_damage_pixels_3d,
        pixel_xyz_stride=args.pixel_xyz_stride,
        profile_timings=timings if args.profile else None,
    )
    record_timing(timings, "script_save_outputs_sec", start)
    if geo3d_context is not None and hasattr(geo3d_context, "timings"):
        timings.update(getattr(geo3d_context, "timings"))
    timings["script_total_sec"] = time.perf_counter() - total_start

    print("Saved prediction outputs:")
    for key, path in paths.items():
        print(f"{key}: {path}")
    if args.profile:
        print_profile(timings)


def record_timing(timings: dict[str, float], key: str, start_time: float) -> None:
    timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start_time)


def print_profile(timings: dict[str, float]) -> None:
    print("\n[PROFILE] stage timings")
    for key, value in sorted(timings.items(), key=lambda item: item[1], reverse=True):
        unit = "count" if key.endswith("_count") else "sec"
        if unit == "count":
            print(f"  {key}: {int(value)}")
        else:
            print(f"  {key}: {value:.3f}s")


if __name__ == "__main__":
    main()
