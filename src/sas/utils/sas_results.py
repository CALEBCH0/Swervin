import json
import numpy as np
from dataclasses import dataclass

@dataclass
class SegResult:
    mask: np.ndarray
    confidence: float

@dataclass
class GeometryResult:
    centerline: np.ndarray
    curvature: float
    heading: float
    lookahead: tuple[float, float]

@dataclass
class ControlResult:
    steering_target: float
    steering_error: float

@dataclass
class SASResults:
    frame: np.ndarray | None = None
    # frame_id: int | None = None   #TODO: add those later
    # frame_timestamp: float | None = None
    # fps: float | None = None
    # timings: dict[str, float] | None = None
    seg: SegResult | None = None
    geometry: GeometryResult | None = None
    control: ControlResult | None = None
    active: bool = False

    def update_frame(self, frame: np.ndarray):
        """Update the frame and reset processing results"""
        self.frame = frame
        self.reset()

    def reset(self):
        self.seg = None
        self.geometry = None
        self.control = None
        self.active = False

    def to_json(self) -> bytes:
        payload = {'active': self.active}
        if self.seg:
            payload['segmentation_mask'] = self.seg.mask.tolist()  # Convert numpy array to list for JSON serialization
            payload['confidence'] = self.seg.confidence
        if self.geometry:
            payload['centerline'] = self.geometry.centerline.tolist()
            payload['curvature'] = self.geometry.curvature
            payload['heading'] = self.geometry.heading
            payload['lookahead'] = self.geometry.lookahead
        if self.control:
            payload['steering_target'] = self.control.steering_target
            payload['steering_error'] = self.control.steering_error
        return json.dumps(payload).encode('utf-8')