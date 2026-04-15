"""Training entry point for RGB/ground_rgb/delamination."""

from pathlib import Path


def train() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[TRAIN] sensor=RGB, sub_sensor=ground_rgb, target=delamination")
    print(f"Project folder: {project_root}")
    print("TODO: implement training pipeline")


if __name__ == "__main__":
    train()
