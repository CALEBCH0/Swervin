import numpy as np

# Minimum pixel count needed to trust a polynomial fit for a lane class
_MIN_LANE_POINTS = 15

# Default half-lane-width offset (pixels) used when only one boundary is visible.
# At a 512-wide BEV canvas, a ~3.7m lane at ~1m/8px ≈ 30px per side.
_HALF_LANE_WIDTH_PX = 30

# Per-dataset lane class preference order for ego-lane boundaries.
# CULane assigns classes 1–4 left-to-right in the camera view:
#   1=leftmost, 2=left ego boundary, 3=right ego boundary, 4=rightmost.
# Other datasets can be added here when support is needed.
DATASET_LANE_CONFIG = {
    'culane':   {'left': (2, 1), 'right': (3, 4)},
    'tusimple': {'left': (2, 1), 'right': (3, 4)},
    'llamas':   {'left': (2, 1), 'right': (3, 4)},
}
_DEFAULT_CONFIG = DATASET_LANE_CONFIG['culane']


def extract_lane_points(mask: np.ndarray) -> dict:
    """
    Extract one (y, x_centroid) point per row for each lane class.

    Using the column-wise centroid rather than raw pixels gives equal weight
    to every depth position, avoiding the bottom-heavy bias that comes from
    BEV perspective warp expanding near-vehicle lane markings into wide blobs.

    mask: (H, W) uint8 — 0=background, 1–4=lane classes
    Returns: {cls: (ys, xs)} with one point per occupied row.
    """
    result = {}
    for cls in range(1, 5):
        raw_ys, raw_xs = np.where(mask == cls)
        if len(raw_ys) < _MIN_LANE_POINTS:
            continue
        # One centroid x per unique y-row
        unique_ys = np.unique(raw_ys)
        centroid_xs = np.array([raw_xs[raw_ys == y].mean() for y in unique_ys])
        if len(unique_ys) >= _MIN_LANE_POINTS:
            result[cls] = (unique_ys, centroid_xs)
    return result


def fit_polynomial(ys: np.ndarray, xs: np.ndarray, degree: int = 2):
    """
    Fit x = f(y) polynomial with one round of outlier rejection.

    Fitting x as a function of y keeps the system well-conditioned for
    near-vertical lane lines (avoids singularities when lines are steep).

    Returns (degree+1,) coefficient array [a, b, c] or None on failure.
    """
    if len(ys) < degree + 3:
        return None
    try:
        coeffs = np.polyfit(ys, xs, degree)
        residuals = np.abs(xs - np.polyval(coeffs, ys))
        threshold = max(3.0, np.median(residuals) * 1.5)
        inliers = residuals <= threshold
        if inliers.sum() >= degree + 3:
            coeffs = np.polyfit(ys[inliers], xs[inliers], degree)
        return coeffs
    except (np.linalg.LinAlgError, TypeError):
        return None


def compute_curvature(coeffs: np.ndarray, y: float) -> float:
    """
    Signed curvature κ (px⁻¹) for x = ay² + by + c evaluated at y.

    κ = x'' / (1 + x'²)^(3/2),  where x' = dx/dy = 2ay + b,  x'' = 2a.
    Positive = curving right, negative = curving left.
    """
    a = float(coeffs[0])
    b = float(coeffs[1])
    dxdy = 2.0 * a * y + b
    d2xdy2 = 2.0 * a
    return d2xdy2 / (1.0 + dxdy ** 2) ** 1.5


def extract_geometry(
    bev_mask: np.ndarray,
    dataset: str = 'culane',
    lookahead_px: int = 80,
    half_lane_width_px: int = _HALF_LANE_WIDTH_PX,
):
    """
    Fit lane polynomials to a BEV segmentation mask and derive control geometry.

    bev_mask: (H, W) uint8 — 0=background, 1–4=lane classes
    dataset: name key into DATASET_LANE_CONFIG for boundary class preference order
    lookahead_px: Pure Pursuit lookahead distance in BEV pixels
    half_lane_width_px: lateral offset used when only one boundary is visible

    Returns (centerline, heading, curvature, lookahead) or None if no lanes found.
      centerline: (50, 2) float32 [x, y] BEV pixels ordered near→far (y decreasing)
      heading:    float — arctan(dx/dy) at ego position, radians from forward axis
      curvature:  float — signed κ at mid-range (px⁻¹); positive = right curve
      lookahead:  (float, float) — BEV (x, y) of the Pure Pursuit target point
    """
    H, W = bev_mask.shape
    cfg = DATASET_LANE_CONFIG.get(dataset, _DEFAULT_CONFIG)

    lane_pts = extract_lane_points(bev_mask)
    if not lane_pts:
        return None

    left_coeffs = None
    for cls in cfg['left']:
        if cls in lane_pts:
            left_coeffs = fit_polynomial(*lane_pts[cls])
            if left_coeffs is not None:
                break

    right_coeffs = None
    for cls in cfg['right']:
        if cls in lane_pts:
            right_coeffs = fit_polynomial(*lane_pts[cls])
            if right_coeffs is not None:
                break

    if left_coeffs is None and right_coeffs is None:
        return None

    if left_coeffs is not None and right_coeffs is not None:
        center_coeffs = (left_coeffs + right_coeffs) / 2.0
    elif left_coeffs is not None:
        center_coeffs = left_coeffs.copy()
        center_coeffs[-1] += half_lane_width_px
    else:
        center_coeffs = right_coeffs.copy()
        center_coeffs[-1] -= half_lane_width_px

    # Evaluate centerline over reliable BEV range; top 25% is unreliable at distance
    y_bot = float(H - 1)
    y_top = float(H * 0.25)
    y_range = np.linspace(y_bot, y_top, 50)
    x_range = np.polyval(center_coeffs, y_range)
    centerline = np.column_stack([x_range, y_range]).astype(np.float32)

    # Heading at ego position (bottom of BEV = closest to vehicle)
    a, b = float(center_coeffs[0]), float(center_coeffs[1])
    heading = float(np.arctan(2.0 * a * y_bot + b))

    # Curvature at mid-range where the polynomial fit is most reliable
    y_mid = (y_bot + y_top) / 2.0
    curvature = float(compute_curvature(center_coeffs, y_mid))

    # Lookahead point at fixed distance ahead of ego
    lookahead_y = float(max(y_top, y_bot - lookahead_px))
    lookahead_x = float(np.polyval(center_coeffs, lookahead_y))
    lookahead = (lookahead_x, lookahead_y)

    return centerline, heading, curvature, lookahead
