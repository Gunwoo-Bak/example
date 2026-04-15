"""Inference entry point for MBES/general/scour_and_erosion."""

from pathlib import Path


def infer() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[INFER] sensor=MBES, sub_sensor=general, target=scour_and_erosion")
    print(f"Project folder: {project_root}")
    print("TODO: implement inference pipeline")


if __name__ == "__main__":
    infer()
