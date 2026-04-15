"""Training entry point for MBES/general/scour_and_erosion."""

from pathlib import Path


def train() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[TRAIN] sensor=MBES, sub_sensor=general, target=scour_and_erosion")
    print(f"Project folder: {project_root}")
    print("TODO: implement training pipeline")


if __name__ == "__main__":
    train()
