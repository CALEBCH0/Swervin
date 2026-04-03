import cv2
import numpy as np
from sas.utils.utils_func import preproc_yolo, image_preprocessing
from sas.utils._geometry_status import GeometryStatus

class SASProcessClass:
    def __init__(self,
                conf,
                obj_detector=None,
                lane_detector=None,
                tilt_detector=None
                ):
        self.conf = conf
        self.obj_detector = obj_detector
        self.lane_detector = lane_detector
        self.tilt_detector = tilt_detector
        self.status = GeometryStatus(conf.algoritm)

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
        
        # Lane segmentation
        # input:
        #   - preprocessed frame
        # output:
        #   - lane segmentation mask
        #   - confidence
        # TODO: binary or multi-class mask

        mask, confidence = self.segment_lane(frame)

        # BEV
        # input:
        #   - lane segmentation mask
        #   - perspective transformation matrix
        #   - camera calibration parameters
        # output:
        #   - BEV lane mask
        #   - top-down warped representation of lane mask
        bev_mask, warped_lane = self.bev_transform(mask)

        # Lane mask points extraction in BEV
        # input:
        #   - BEV lane mask
        # output:
        #   - lane pixel coordinates
        # TODO: left/right lane + center-region point sets?
        lane_points = self.extract_lane_points(bev_mask)

        # Fit lane centerline polynomial
        # input:
        #   - BEV lane points
        # output:
        #   - polynomial coefficents for left and right lane, centerline polynomial
        # x = ay^2 + by + c
        lane_polynomials = self.fit_lane_polynomial(lane_points)
        
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
        geometry = self.compute_geometry(lane_polynomials)

        # Control math
        steering_target = pure_pursuit(geometry)