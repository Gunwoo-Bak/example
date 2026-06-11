from __future__ import annotations

from typing import Any

import numpy as np
from skimage.measure import label, regionprops


def summarize_instances(
    pred_mask: np.ndarray,
    class_names: dict[int, str],
    min_area_px: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    instance_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    instance_id = 1

    for cls_id, cls_name in class_names.items():
        if int(cls_id) == 0:
            continue

        class_mask = pred_mask == int(cls_id)
        cc = label(class_mask, connectivity=2)
        props = [p for p in regionprops(cc) if int(p.area) >= min_area_px]
        total_area = 0

        for local_id, prop in enumerate(props, start=1):
            y0, x0, y1, x1 = prop.bbox
            cy, cx = prop.centroid
            area = int(prop.area)
            total_area += area
            instance_rows.append(
                {
                    "instance_id": instance_id,
                    "instance_local_id": local_id,
                    "class_id": int(cls_id),
                    "class_name": cls_name,
                    "centroid_x_px": round(float(cx), 4),
                    "centroid_y_px": round(float(cy), 4),
                    "bbox_xmin_px": int(x0),
                    "bbox_ymin_px": int(y0),
                    "bbox_xmax_px": int(x1),
                    "bbox_ymax_px": int(y1),
                    "bbox_w_px": int(x1 - x0),
                    "bbox_h_px": int(y1 - y0),
                    "area_px": area,
                }
            )
            instance_id += 1

        class_rows.append(
            {
                "class_id": int(cls_id),
                "class_name": cls_name,
                "instance_count": len(props),
                "total_area_px": int(total_area),
            }
        )

    return instance_rows, class_rows

