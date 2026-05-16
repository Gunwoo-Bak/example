import numpy as np

from crackseg.instances import summarize_instances
from crackseg.metrics import summarize_pixels


def test_summarize_pixels_counts_classes():
    mask = np.array(
        [
            [0, 1, 1],
            [2, 2, 0],
        ],
        dtype=np.uint8,
    )
    rows = summarize_pixels(mask, {0: "BG", 1: "CRC", 2: "DLM"})
    counts = {row["class_name"]: row["pixel_count"] for row in rows}
    assert counts == {"BG": 2, "CRC": 2, "DLM": 2}


def test_summarize_instances_ignores_background():
    mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [2, 0, 2, 0],
        ],
        dtype=np.uint8,
    )
    rows, class_rows = summarize_instances(mask, {0: "BG", 1: "CRC", 2: "DLM"})
    assert len(rows) == 3
    assert {row["class_name"] for row in rows} == {"CRC", "DLM"}
    counts = {row["class_name"]: row["instance_count"] for row in class_rows}
    assert counts == {"CRC": 1, "DLM": 2}
