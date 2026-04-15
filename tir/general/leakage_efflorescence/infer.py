"""Inference entry point for TIR/general/leakage_efflorescence."""

from pathlib import Path


def infer() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[INFER] sensor=TIR, sub_sensor=general, target=leakage_efflorescence")
    print(f"Project folder: {project_root}")
    print("TODO: implement inference pipeline")


if __name__ == "__main__":
    infer()
