"""Model definition placeholder for LIDAR/ground_lidar/deformation_displacement."""


class DetectionModel:
    def __init__(self, name: str = "baseline_model") -> None:
        self.name = name

    def forward(self, inputs):
        return inputs
