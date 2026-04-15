"""Training entry point for RGB/underwater_rgb/crack."""

from pathlib import Path


def train() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[TRAIN] sensor=RGB, sub_sensor=underwater_rgb, target=crack")
    print(f"Project folder: {project_root}")
    print("TODO: implement training pipeline")


if __name__ == "__main__":
    train()
