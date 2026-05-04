from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sas.utils.bev_transformer import get_src, get_dst, compute_bev_transform, apply_bev_transform

OUTPUT_SIZE = (512, 512)

imgpath = './tu.jpg'
img = cv2.imread(imgpath)
img = cv2.resize(img, OUTPUT_SIZE)

src = get_src(img)
dst = get_dst(OUTPUT_SIZE)
print("Selected source points:\n", src)
print("Destination points:\n", dst)

M, M_inv = compute_bev_transform(src, dst)
print("Homography matrix M:\n", M)

bev = apply_bev_transform(img, M, output_size=OUTPUT_SIZE)

base = imgpath.rsplit('.', 1)[0]
cv2.imwrite(base + '_transformed.jpg', bev)

# Debug: draw src points on original image
debug = img.copy()
for i, (x, y) in enumerate(src):
    cv2.circle(debug, (int(x), int(y)), 6, (0, 255, 0), -1)
    cv2.putText(debug, str(i + 1), (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imwrite(base + '_debug.jpg', debug)

np.save(base + '_homography.npy', M)
print(f"Saved: {base}_transformed.jpg, {base}_debug.jpg, {base}_homography.npy")

def show_until_closed(win, img):
    cv2.namedWindow(win)
    cv2.imshow(win, img)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

# Apply saved homography to a second image
M_loaded = np.load(base + '_homography.npy')
frame = cv2.imread('./apply.jpg')
if frame is not None:
    frame = cv2.resize(frame, OUTPUT_SIZE)
    bev_frame = apply_bev_transform(frame, M_loaded, output_size=OUTPUT_SIZE)
    show_until_closed('BEV result', bev_frame)
else:
    show_until_closed('BEV result', bev)
