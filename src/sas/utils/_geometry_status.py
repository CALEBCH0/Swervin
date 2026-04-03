import numpy as np

# import models

class GeometryStatus():
    def __init__(self, algorithm):
        self.center_fit = None
        self.curvature = None
        self.heading: float
        self.offset: float
        self.lookahead_point: tuple[float, float]
        self.confidence: float