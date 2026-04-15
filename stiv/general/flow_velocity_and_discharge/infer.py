"""Inference entry point for STIV/general/flow_velocity_and_discharge."""

from pathlib import Path


def infer() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"[INFER] sensor=STIV, sub_sensor=general, target=flow_velocity_and_discharge")
    print(f"Project folder: {project_root}")
    print("TODO: implement inference pipeline")


if __name__ == "__main__":
    infer()
