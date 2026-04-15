"""Model definition placeholder for RGB/ground_rgb/delamination."""


class DetectionModel:
    def __init__(self, name: str = "baseline_model") -> None:
        self.name = name

    def forward(self, inputs):
        return inputs
