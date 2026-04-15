"""Inference entry point for RGB/underwater_rgb/crack."""

from pathlib import Path


def infer() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[INFER] sensor=RGB, sub_sensor=underwater_rgb, target=crack")
    print(f"Project folder: {project_root}")
    print("TODO: implement inference pipeline")


if __name__ == "__main__":
    infer()
