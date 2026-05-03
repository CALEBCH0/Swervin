import os

import cv2
import numpy as np

from sas.utils.sas_results import SASResults
from sas.utils.bev_transformer import apply_bev_transform


class SASProcessClass:
    def __init__(self,
                config,
                obj_detector=None,
                lane_segmenter=None,
                tilt_detector=None
                ):
        self.config = config
        self.obj_detector = obj_detector
        self.lane_segmenter = lane_segmenter
        self.tilt_detector = tilt_detector
        self.results = SASResults()

        self.M_bev = None
        self.bev_output_size = None
        if config.bev.enabled:
            path = config.bev.homography_path
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"BEV homography not found at {path}. Run with --calibrate-bev first."
                )
            self.M_bev = np.load(path)
            self.bev_output_size = tuple(config.bev.output_size)

    def __call__(self, input_img, img_area):
        return self.forward(input_img, img_area)

    def forward(self, input_img, img_area):
        """Main process for SAS pipeline"""
        frame = input_img.copy()
        self.results.update_frame(frame)

        # Lane segmentation
        mask, confidence = self.lane_segmenter(frame)
        self.results.seg = (mask, confidence)

        # BEV transform
        if self.M_bev is not None:
            bev_mask = apply_bev_transform(mask, self.M_bev, self.bev_output_size)
            self.results.bev_mask = bev_mask

        # Lane mask points extraction in BEV
        # lane_points = self.extract_lane_points(bev_mask)

        # Fit lane centerline polynomial
        # x = ay^2 + by + c
        # lane_polynomials = self.fit_lane_polynomial(lane_points)

        # Curvature and lookahead point calculation
        # geometry = self.compute_geometry(lane_polynomials)
        # self.results.geometry = GeometryResult(
        #     centerline=geometry['centerline'],
        #     curvature=geometry['curvature'],
        #     heading=geometry['heading'],
        #     lookahead=geometry['lookahead']
        # )

        # Control math
        # steering_target = pure_pursuit(geometry)
        # self.results.control = ControlResult(
        #     steering_target=steering_target,
        #     steering_error=geometry['steering_error']
        # )

        return frame, self.results
