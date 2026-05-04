import numpy as np
import cv2

def get_src(img):
    """
    Drag-handle trapezoid calibration. Align the green trapezoid with lane markings.
    - Drag corners: x changes width (symmetric), y moves that edge up/down
    - Drag inside body: translates the whole trapezoid
    - Enter to confirm, Q to cancel
    Returns:
        src: (4, 2) float32 array [TL, TR, BR, BL], or empty array if cancelled
    """
    H, W = img.shape[:2]
    base = img.copy()
    win = "BEV Calibration | drag corners/body to align | Enter=confirm  Q=quit"

    s = {
        'cx':     W // 2,
        'top_y':  H // 3,
        'bot_y':  H - H // 8,
        'top_hw': W // 10,
        'bot_hw': W // 4,
    }
    drag = {'handle': None}

    def pts():
        cx, ty, by, thw, bhw = s['cx'], s['top_y'], s['bot_y'], s['top_hw'], s['bot_hw']
        return [(cx - thw, ty), (cx + thw, ty), (cx + bhw, by), (cx - bhw, by)]

    def redraw():
        clone = base.copy()
        corners = pts()
        poly = np.array(corners, np.int32).reshape((-1, 1, 2))
        cv2.polylines(clone, [poly], isClosed=True, color=(0, 220, 0), thickness=2)
        for (x, y), label in zip(corners, ('TL', 'TR', 'BR', 'BL')):
            cv2.circle(clone, (x, y), 8, (255, 255, 255), -1)
            cv2.circle(clone, (x, y), 8, (0, 220, 0), 2)
            cv2.putText(clone, label, (x + 10, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        cv2.putText(clone, "Drag corners/body to align with lanes | Enter=confirm  Q=quit",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        cv2.imshow(win, clone)

    def hit_corner(x, y):
        for i, (cx, cy) in enumerate(pts()):
            if (x - cx) ** 2 + (y - cy) ** 2 <= 14 ** 2:
                return ('tl', 'tr', 'br', 'bl')[i]
        return None

    def inside_poly(x, y):
        poly = np.array(pts(), np.int32)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h = hit_corner(x, y) or ('body' if inside_poly(x, y) else None)
            drag.update({'handle': h, 'sx': x, 'sy': y,
                         **{f's_{k}': v for k, v in s.items()}})
        elif event == cv2.EVENT_MOUSEMOVE and drag['handle']:
            dx, dy = x - drag['sx'], y - drag['sy']
            h = drag['handle']
            if h == 'tl':
                s['top_hw'] = max(10, drag['s_top_hw'] - dx)
                s['top_y']  = max(0, min(drag['s_top_y'] + dy, s['bot_y'] - 10))
            elif h == 'tr':
                s['top_hw'] = max(10, drag['s_top_hw'] + dx)
                s['top_y']  = max(0, min(drag['s_top_y'] + dy, s['bot_y'] - 10))
            elif h == 'bl':
                s['bot_hw'] = max(10, drag['s_bot_hw'] - dx)
                s['bot_y']  = max(s['top_y'] + 10, min(drag['s_bot_y'] + dy, H))
            elif h == 'br':
                s['bot_hw'] = max(10, drag['s_bot_hw'] + dx)
                s['bot_y']  = max(s['top_y'] + 10, min(drag['s_bot_y'] + dy, H))
            elif h == 'body':
                s['cx']    = drag['s_cx'] + dx
                s['top_y'] = max(0, min(drag['s_top_y'] + dy, H))
                s['bot_y'] = max(0, min(drag['s_bot_y'] + dy, H))
            redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            drag['handle'] = None

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    cancelled = False
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break
        if key == ord('q'):
            cancelled = True
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cancelled = True
            break

    cv2.destroyAllWindows()
    return np.float32([]) if cancelled else np.float32(pts())


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
