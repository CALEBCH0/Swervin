import os

import cv2
import numpy as np

from sas.utils.sas_results import SASResults, SegResult, BEVResult, GeometryResult
from sas.utils.bev_transformer import apply_bev_transform
from sas.utils.lane_geometry import extract_geometry


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
        mask, confidence, metadata = self.lane_segmenter(frame)
        self.results.seg = SegResult(
            mask=mask,
            confidence=confidence,
            lane_confidences=metadata.get('lane_confidences'),
        )
        dataset = metadata.get('dataset', 'culane')

        # BEV transform + geometry extraction
        if self.M_bev is not None:
            bev_mask = apply_bev_transform(mask, self.M_bev, self.bev_output_size)
            self.results.bev = BEVResult(bev_mask=bev_mask, src_points=None)

            geom = extract_geometry(bev_mask, dataset=dataset)
            if geom is not None:
                centerline, heading, curvature, lookahead = geom
                self.results.geometry = GeometryResult(
                    centerline=centerline,
                    curvature=curvature,
                    heading=heading,
                    lookahead=lookahead,
                )
                print(f"[SASProcess] Geometry extracted: curvature={curvature:.4f}, heading={heading:.2f}°, lookahead={lookahead}")
            else:
                print("[SASProcess] Geometry extraction failed, got None")


        # Control math (Pure Pursuit — next step)
        # steering_target = pure_pursuit(self.results.geometry)
        # self.results.control = ControlResult(...)

        return frame, self.results
