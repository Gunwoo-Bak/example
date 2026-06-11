import numpy as np

from crackseg.geo3d import (
    Geo3DContext,
    GcpPoint,
    HeightModel,
    HomographyResult,
    OrthoReference,
    add_geo3d_to_instance_rows,
)


def test_ortho_reference_world_pixel_roundtrip():
    ref = OrthoReference(
        path="orthophoto.tif",
        width=100,
        height=100,
        pixel_size_x=0.5,
        pixel_size_y=0.5,
        tiepoint_col=0,
        tiepoint_row=0,
        tiepoint_x=1000,
        tiepoint_y=2000,
    )
    x, y = ref.pixel_to_world(10, 20)
    assert (x, y) == (1005, 1990)
    col, row = ref.world_to_pixel(x, y)
    assert (col, row) == (10, 20)


def test_geo3d_adds_world_coordinates_with_homography_and_gcp_height():
    ref = OrthoReference(
        path="orthophoto.tif",
        width=100,
        height=100,
        pixel_size_x=1,
        pixel_size_y=1,
        tiepoint_col=0,
        tiepoint_row=0,
        tiepoint_x=500,
        tiepoint_y=900,
    )
    height = HeightModel(
        gcps=[
            GcpPoint("a", 500, 900, 10),
            GcpPoint("b", 510, 900, 20),
            GcpPoint("c", 500, 890, 30),
        ]
    )
    homography = HomographyResult(matrix=np.array([[2, 0, 1], [0, 2, 2], [0, 0, 1]], dtype=float))
    context = Geo3DContext(orthophoto=ref, height_model=height, homography=homography)
    rows = add_geo3d_to_instance_rows(
        [
            {
                "instance_id": 1,
                "class_id": 1,
                "class_name": "CRC",
                "centroid_x_px": 4,
                "centroid_y_px": 5,
            }
        ],
        context,
    )
    assert rows[0]["ortho_pixel_x"] == 9
    assert rows[0]["ortho_pixel_y"] == 12
    assert rows[0]["world_x_m"] == 509
    assert rows[0]["world_y_m"] == 888
    assert rows[0]["world_z_m"] is not None
    assert rows[0]["geo3d_source"] == "homography_to_orthophoto"
