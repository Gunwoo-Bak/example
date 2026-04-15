"""Inference entry point for LIDAR/underwater_lidar/scour_and_erosion."""

from pathlib import Path


def infer() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[INFER] sensor=LIDAR, sub_sensor=underwater_lidar, target=scour_and_erosion")
    print(f"Project folder: {project_root}")
    print("TODO: implement inference pipeline")


if __name__ == "__main__":
    infer()
