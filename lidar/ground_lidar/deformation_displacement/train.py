"""Training entry point for LIDAR/ground_lidar/deformation_displacement."""

from pathlib import Path


def train() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[TRAIN] sensor=LIDAR, sub_sensor=ground_lidar, target=deformation_displacement")
    print(f"Project folder: {project_root}")
    print("TODO: implement training pipeline")


if __name__ == "__main__":
    train()
