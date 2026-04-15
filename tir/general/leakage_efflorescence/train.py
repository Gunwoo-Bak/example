"""Training entry point for TIR/general/leakage_efflorescence."""

from pathlib import Path


def train() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[TRAIN] sensor=TIR, sub_sensor=general, target=leakage_efflorescence")
    print(f"Project folder: {project_root}")
    print("TODO: implement training pipeline")


if __name__ == "__main__":
    train()
