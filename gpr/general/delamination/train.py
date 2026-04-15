"""Training entry point for GPR/general/delamination."""

from pathlib import Path


def train() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[TRAIN] sensor=GPR, sub_sensor=general, target=delamination")
    print(f"Project folder: {project_root}")
    print("TODO: implement training pipeline")


if __name__ == "__main__":
    train()
