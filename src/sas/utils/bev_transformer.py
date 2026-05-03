import numpy as np
import cv2

class BEVTransformer:
    def __init__(self, homography_path, output_size):
        self.M = np.load(homography_path)
        self.output_size = output_size

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(mask, self.M, self.output_size)


def get_src(img):
    """
    Interactive 4-point selection for road trapezoid.
    Click clockwise: top-left, top-right, bottom-right, bottom-left.
    Returns:
        src: (4, 2) float32 array of selected points
    """
    points = []
    clone = img.copy()
    win = "BEV Calibration - click TL TR BR BL then press Q"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(clone, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(clone, str(len(points)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(win, clone)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    cv2.imshow(win, clone)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4 or key == ord('q'):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return np.float32(points)


def get_dst(output_size):
    """
    Returns the 4 corners of the output BEV canvas as the destination rectangle.
    Args:
        output_size: (width, height) of the BEV output
    Returns:
        dst: (4, 2) float32 array [TL, TR, BR, BL]
    """
    W, H = output_size
    return np.float32([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ])


def compute_bev_transform(src, dst):
    """
    Compute the 3x3 homography matrix M and its inverse.
    Args:
        src: (4, 2) float32 array of source points (road trapezoid in camera view)
        dst: (4, 2) float32 array of destination points (rectangle in BEV canvas)
    Returns:
        M: homography matrix (camera → BEV)
        M_inv: inverse homography matrix (BEV → camera)
    """
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv


def apply_bev_transform(img, M, output_size):
    """
    Warp a perspective-view image into BEV using homography matrix M.
    Args:
        img: input image (BGR or single-channel mask)
        M: homography matrix
        output_size: (width, height) of the BEV output
    Returns:
        warped BEV image
    """
    return cv2.warpPerspective(img, M, output_size, flags=cv2.INTER_LINEAR)
