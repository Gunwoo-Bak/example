"""Inference entry point for TIR/general/corrosion_rebar_exposure."""

from pathlib import Path


def infer() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[INFER] sensor=TIR, sub_sensor=general, target=corrosion_rebar_exposure")
    print(f"Project folder: {project_root}")
    print("TODO: implement inference pipeline")


if __name__ == "__main__":
    infer()
