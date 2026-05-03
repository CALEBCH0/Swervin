import cv2
import numpy as np
from sas.utils.sas_results import SASResults

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
        
    def __call__(self, input_img, img_area):
        self.forward(input_img, img_area)

    def forward(self, input_img, img_area):
        """Main process for SAS pipeline"""
        # Camera frame
        # input:
        #   - 3 channel BGR image
        # output:
        #   - normalized frame for segmentation
        frame = input_img.copy()
        self.results.update_frame(frame)
        print("New frame received")
        
        # Lane segmentation
        # input:
        #   - preprocessed frame
        # output:
        #   - lane segmentation mask
        #   - confidence
        # TODO: binary or multi-class mask

        mask, confidence = self.lane_segmenter(frame)
        self.results.seg = SegResult(mask=mask, confidence=confidence)
        print("Lane segmentation completed")
        
        # BEV
        # input:
        #   - lane segmentation mask
        #   - perspective transformation matrix
        #   - camera calibration parameters
        # output:
        #   - BEV lane mask
        #   - top-down warped representation of lane mask
        # bev_mask, warped_lane = self.transform_to_bev(mask)

        # Lane mask points extraction in BEV
        # input:
        #   - BEV lane mask
        # output:
        #   - lane pixel coordinates
        # TODO: left/right lane + center-region point sets?
        # lane_points = self.extract_lane_points(bev_mask)

        # Fit lane centerline polynomial
        # input:
        #   - BEV lane points
        # output:
        #   - polynomial coefficents for left and right lane, centerline polynomial
        # x = ay^2 + by + c
        # lane_polynomials = self.fit_lane_polynomial(lane_points)
        
        # Curvature and lookahead point calculation
        # input:
        #   - lane polynomial coefficients
        #   - current vehicle position in BEV
        #   - vehicle speed
        #   - confidence
        # output:
        #   - lane curvature
        #   - lookahead point coordinates in BEV
        #   - heading angle
        #   - lateral offset
        # geometry = self.compute_geometry(lane_polynomials)
        # self.results.geometry = GeometryResult(
        #             centerline=geometry['centerline'],
        #             curvature=geometry['curvature'],
        #             heading=geometry['heading'],
        #             lookahead=geometry['lookahead']
        #         )
        
        # # Control math
        # steering_target = pure_pursuit(geometry)
        
        # self.results.control = ControlResult(
        #     steering_target=steering_target,
        #     steering_error=geometry['steering_error']
        # )

        return frame, self.results