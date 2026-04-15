"""Model definition placeholder for MBES/general/scour_and_erosion."""


class DetectionModel:
    def __init__(self, name: str = "baseline_model") -> None:
        self.name = name

    def forward(self, inputs):
        return inputs
