import numpy as np
import cv2

def get_src(img, vertical_shift=0):
    """
    Interactive 4 ground-lane markers selection
    click clockwise from top-left, top-right, bottom-right, bottom-left
    Returns:
        src: 4x2 array of selected points
    """
    points = []
    labels = ['top-left', 'top-right', 'bottom-right', 'bottom-left']

    for label in labels:
        print(f"Select ROI for {label} marker, then press SPACE")
        roi = cv2.selectROI(img)
        x = roi[0] + 0.5 * roi[2]
        y = roi[1] + 0.5 * roi[3] + vertical_shift
        points.append(np.float32([x, y]))
    return np.float32(points)

def get_dst(img, vertical_shift=0):
    """
    Interactive rectangular region selection for BEV output
    Returns:
        dst: 4x2 float32 array of selected points
    """
print("Select rectangular ROI for BEV output, then press SPACE")
    roi = cv2.selectROI(img)
    x, y, w, h = roi
    y += vertical_shift
    dst = np.float32([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h]
    ])
    return dst

def compute_bev_transform(src, dst):
    """
    Compute the 3x3 homography matrix M and its inverse
    Args:
        src: 4x2 array of source points
        dst: 4x2 array of destination points
        output_size: (width, height) of the BEV output
    Returns:
        M: homography matrix
        M_inv: inverse homography matrix
    """
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv

def apply_bev_transform(img, M, output_size):
    """
    Warp a perspective-view image into BEV using the homography matrix M
    Args:
        img: input perspective-view image
        M: homography matrix
        output_size: (width, height) of the BEV output image
    Returns:
        warped BEV image
    """
    return cv2.warpPerspective(img, M, output_size, flags=cv2.INTER_LINEAR)e

def draw_points()