import numpy as np

from crackseg.las_ray import (
    CameraIntrinsics,
    CameraPose,
    ImageXYZMap,
    LasHeader,
    XYZMapGeo3DContext,
    camera_to_world_matrix,
    build_image_xyz_map,
    image_index_from_name,
    ray_box_interval,
    read_mrk_positions,
)


def test_camera_to_world_yaw_zero_pitch_zero_points_north():
    rotation = camera_to_world_matrix(yaw_deg=0, pitch_deg=0, roll_deg=0)
    forward_ray = rotation @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(forward_ray, [0.0, 1.0, 0.0])


def test_camera_to_world_pitch_down_points_to_negative_z():
    rotation = camera_to_world_matrix(yaw_deg=0, pitch_deg=-90, roll_deg=0)
    forward_ray = rotation @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(forward_ray, [0.0, 0.0, -1.0], atol=1e-8)


def test_ray_box_interval_hits_box():
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    t_min, t_max = ray_box_interval(origin, direction, (2.0, -1.0, -1.0), (4.0, 1.0, 1.0))
    assert t_min == 2.0
    assert t_max == 4.0


def test_read_mrk_positions(tmp_path):
    mrk = tmp_path / "Timestamp.MRK"
    mrk.write_text(
        "5\t285641.872696\t[2373]\t27,N\t54,E\t76,V\t36.47863646,Lat\t127.48026261,Lon\t84.789,Ellh\n",
        encoding="utf-8",
    )
    positions = read_mrk_positions(mrk)
    assert positions[5] == (36.47863646, 127.48026261, 84.789)


def test_image_index_from_name():
    assert image_index_from_name("DJI_20250702162023_0005_V.JPG") == 5


def test_build_image_xyz_map_projects_las_point_to_pixel_center():
    class Surface:
        points = np.array([[0.0, 10.0, 0.0]], dtype=np.float64)
        header = LasHeader(
            path="fake.las",
            version="1.2",
            point_format=2,
            point_record_length=26,
            point_count=1,
            offset_to_points=0,
            scale=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            bounds_min=(0.0, 10.0, 0.0),
            bounds_max=(0.0, 10.0, 0.0),
        )

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
    xyz_map = build_image_xyz_map(Surface(), intrinsics, pose, fill_distance_px=0)
    result = xyz_map.lookup(5, 5)
    assert result["xyz_valid"] is True
    assert result["world_x_m"] == 0.0
    assert result["world_y_m"] == 10.0
    assert result["world_z_m"] == 0.0
    assert result["xyz_map_source"] == "projected_las"


def test_xyz_map_context_uses_instance_region_median():
    intrinsics = CameraIntrinsics(width=4, height=4, focal_length_px=10.0, cx=2.0, cy=2.0)
    pose = CameraPose(
        source_path="image.jpg",
        center_xyz=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        position_source="test",
        orientation_source="test",
    )
    header = LasHeader(
        path="fake.las",
        version="1.2",
        point_format=2,
        point_record_length=26,
        point_count=2,
        offset_to_points=0,
        scale=(1.0, 1.0, 1.0),
        offset=(0.0, 0.0, 0.0),
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(20.0, 20.0, 20.0),
    )
    nan = np.nan
    xyz_map = ImageXYZMap(
        world_x=np.array([[nan, nan, nan, nan], [nan, 10, 20, nan], [nan, nan, nan, nan], [nan, nan, nan, nan]], dtype=np.float32),
        world_y=np.array([[nan, nan, nan, nan], [nan, 30, 40, nan], [nan, nan, nan, nan], [nan, nan, nan, nan]], dtype=np.float32),
        world_z=np.array([[nan, nan, nan, nan], [nan, 50, 70, nan], [nan, nan, nan, nan], [nan, nan, nan, nan]], dtype=np.float32),
        depth_m=np.array([[nan, nan, nan, nan], [nan, 1, 2, nan], [nan, nan, nan, nan], [nan, nan, nan, nan]], dtype=np.float32),
        distance_px=np.array([[np.inf, np.inf, np.inf, np.inf], [np.inf, 0, 2, np.inf], [np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf]], dtype=np.float32),
        source=np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
        metadata={},
    )
    context = XYZMapGeo3DContext(
        intrinsics=intrinsics,
        pose=pose,
        surface_header=header,
        xyz_map=xyz_map,
        cache_path="cache.npz",
        fill_distance_px=3.0,
    )
    pred_mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    row = {
        "class_id": 1,
        "centroid_x_px": 1.5,
        "centroid_y_px": 1.0,
        "bbox_xmin_px": 1,
        "bbox_ymin_px": 1,
        "bbox_xmax_px": 3,
        "bbox_ymax_px": 2,
    }
    result = context.image_region_to_world3d(row, pred_mask)
    assert result["xyz_valid"] is True
    assert result["world_x_m"] == 15.0
    assert result["world_y_m"] == 35.0
    assert result["world_z_m"] == 60.0
    assert result["instance_xyz_sample_count"] == 2
