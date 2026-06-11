import numpy as np
import json
import pytest

from crackseg.las_ray import CameraIntrinsics, CameraPose
from crackseg.mesh_ray import MeshRayGeo3DContext, MeshSurfaceIndex, extract_contour_nodes, pixels_to_world_rays, sample_indices


def test_sample_indices_limits_deterministically():
    assert sample_indices(3, 10).tolist() == [0, 1, 2]
    assert sample_indices(10, 3).tolist() == [0, 4, 9]


def test_mesh_ray_intersects_triangle(tmp_path):
    obj = tmp_path / "surface.obj"
    obj.write_text(
        "\n".join(
            [
                "v -2 10 -2",
                "v 2 10 -2",
                "v 0 10 2",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )

    surface = MeshSurfaceIndex(obj)
    intrinsics = CameraIntrinsics(width=11, height=11, focal_length_px=10.0, cx=5.0, cy=5.0)
    pose = CameraPose(
        source_path="image.jpg",
        center_xyz=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        position_source="test",
        orientation_source="test",
    )
    context = MeshRayGeo3DContext(intrinsics=intrinsics, pose=pose, surface=surface, instance_sample_count=2)

    hit = context.image_pixel_to_world3d(5, 5)
    assert hit["mesh_ray_hit"] is True
    assert np.isclose(hit["world_x_m"], 0.0)
    assert np.isclose(hit["world_y_m"], 10.0)
    assert np.isclose(hit["world_z_m"], 0.0)

    pred_mask = np.zeros((11, 11), dtype=np.uint8)
    pred_mask[5, 5] = 1
    row = {
        "class_id": 1,
        "centroid_x_px": 5.0,
        "centroid_y_px": 5.0,
        "bbox_xmin_px": 5,
        "bbox_ymin_px": 5,
        "bbox_xmax_px": 6,
        "bbox_ymax_px": 6,
    }
    region = context.image_region_to_world3d(row, pred_mask)
    assert region["xyz_valid"] is True
    assert region["geo3d_source"] == "dji_pose_mesh_ray_centroid"
    assert region["mesh_ray_backend"] == "trimesh"
    assert region["instance_mesh_sample_count"] == 1
    assert region["instance_mesh_hit_count"] == 1
    assert region["node_count"] >= 1
    assert region["node_xyz_valid_count"] >= 1
    assert json.loads(region["nodes_world_xyz_json"])[0][1] == 10.0

    batch_region = context.image_regions_to_world3d([row], pred_mask)[0]
    assert batch_region["xyz_valid"] is True
    assert batch_region["world_y_m"] == region["world_y_m"]
    assert batch_region["node_xyz_valid_count"] == region["node_xyz_valid_count"]

    median_context = MeshRayGeo3DContext(
        intrinsics=intrinsics,
        pose=pose,
        surface=surface,
        instance_sample_count=2,
        representative_mode="median",
    )
    median_region = median_context.image_region_to_world3d(row, pred_mask)
    assert median_region["geo3d_source"] == "dji_pose_mesh_ray_instance_median"


def test_warp_mesh_ray_backend_intersects_triangle(tmp_path):
    pytest.importorskip("warp")

    obj = tmp_path / "surface.obj"
    obj.write_text(
        "\n".join(
            [
                "v 100000 -2 200000",
                "v 100004 -2 200000",
                "v 100002 2 200010",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )

    surface = MeshSurfaceIndex(obj, ray_backend="warp", warp_device="cpu")
    hits = surface.intersect_rays(
        origins=np.array([[100002.0, 0.0, 199990.0]], dtype=np.float64),
        directions=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
    )
    assert surface.ray_backend == "warp"
    assert hits[0]["mesh_ray_backend"] == "warp"
    assert hits[0]["mesh_ray_hit"] is True
    assert np.isclose(hits[0]["world_x_m"], 100002.0, atol=1e-3)
    assert np.isclose(hits[0]["world_z_m"], 200005.0, atol=1e-3)


def test_extract_contour_nodes_returns_limited_xy_nodes():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    nodes = extract_contour_nodes(mask, offset_x=10, offset_y=20, max_count=5)
    assert nodes.shape == (5, 2)
    assert np.all(nodes[:, 0] >= 10)
    assert np.all(nodes[:, 1] >= 20)


def test_pixels_to_world_rays_vectorizes_camera_model():
    intrinsics = CameraIntrinsics(width=11, height=11, focal_length_px=10.0, cx=5.0, cy=5.0)
    pose = CameraPose(
        source_path="image.jpg",
        center_xyz=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        position_source="test",
        orientation_source="test",
    )
    rays = pixels_to_world_rays(intrinsics, pose, np.array([[5.0, 5.0], [6.0, 5.0]]))
    assert np.allclose(rays[0], [0.0, 1.0, 0.0])
    assert rays.shape == (2, 3)
