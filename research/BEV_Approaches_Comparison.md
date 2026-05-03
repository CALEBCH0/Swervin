# BEV Transform Approaches for Lane Detection

Two approaches were evaluated for converting a front-facing perspective-view camera image into a Bird's Eye View (BEV):

1. **Geometric Homography (BEV-Transform)** — classical inverse perspective mapping using OpenCV
2. **Learned Perspective Transformer (Perspective-BEV-Transformer)** — deep learning regression using ResNet50 + dense heads

---

## Table of Contents

- [Approach 1: Geometric Homography](#approach-1-geometric-homography)
  - [How it works](#how-it-works)
  - [Math](#math)
  - [Full implementation](#full-implementation)
  - [Step-by-step usage](#step-by-step-usage)
  - [Strengths and limitations](#strengths-and-limitations)
- [Approach 2: Learned Perspective Transformer](#approach-2-learned-perspective-transformer)
  - [How it works](#how-it-works-1)
  - [Architecture](#architecture)
  - [Data pipeline](#data-pipeline)
  - [Full implementation](#full-implementation-1)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Strengths and limitations](#strengths-and-limitations-1)
- [Decision Guide](#decision-guide)

---

## Approach 1: Geometric Homography

**Source:** `BEV-Transform/BEV_Transform_ND.py`

### How it works

A homography is a projective transformation between two planes. When you assume the road surface is flat, you can compute a 3×3 matrix `M` that maps every pixel in the perspective view to its corresponding location in a top-down view. This is called **Inverse Perspective Mapping (IPM)**.

The matrix is computed from 4 point correspondences:

```
src  →  4 real-world markers on the road (selected in the perspective image)
dst  →  4 rectangle corners defining where those markers should appear in BEV
M    =  cv2.getPerspectiveTransform(src, dst)    # 3×3 homography matrix
BEV  =  cv2.warpPerspective(img, M, output_size) # apply the warp
```

### Math

For any pixel `p = [u, v, 1]ᵀ` in the perspective image, its BEV location `p' = [u', v', w']ᵀ` is:

```
[u']   [m00 m01 m02] [u]
[v'] = [m10 m11 m12] [v]
[w']   [m20 m21 m22] [1]

# Final pixel: (u'/w', v'/w')
```

OpenCV solves for `M` using the Direct Linear Transform (DLT) algorithm from the 4 point pairs.

### Full implementation

Dependencies: `numpy`, `opencv-python`

```python
import numpy as np
import cv2


# ── Point selection helpers ──────────────────────────────────────────────────

def get_src(img, vertical_shift=0):
    """
    Interactively select 4 ground-plane markers in the perspective image.
    Click clockwise starting from top-left: topL → topR → botR → botL.
    Returns src as float32 array of shape (4, 2).
    """
    points = []
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    for label in labels:
        print(f"Select ROI for {label} marker, then press SPACE")
        roi = cv2.selectROI(img)
        x = roi[0] + 0.5 * roi[2]
        y = roi[1] + 0.5 * roi[3] + vertical_shift
        points.append(np.float32([x, y]))
    return np.float32(points)  # shape (4, 2)


def get_dst(img, vertical_shift=0):
    """
    Interactively select a rectangular region that defines the BEV output area.
    The 4 corners of this rectangle become the dst points.
    Returns dst as float32 array of shape (4, 2).
    """
    print("Select the rectangular destination region, then press SPACE")
    roi = cv2.selectROI(img)
    x, y, w, h = roi
    y += vertical_shift
    dst = np.float32([
        [x,     y    ],   # top-left
        [x + w, y    ],   # top-right
        [x + w, y + h],   # bottom-right
        [x,     y + h],   # bottom-left
    ])
    return dst


# ── Core transform ───────────────────────────────────────────────────────────

def compute_bev_transform(src, dst):
    """
    Compute the 3x3 homography matrix M and its inverse M_inv.
    src, dst: float32 arrays of shape (4, 2).
    """
    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv


def apply_bev_transform(img, M, output_size):
    """
    Warp a perspective-view image into BEV using homography M.
    output_size: (width, height) of the output image.
    """
    return cv2.warpPerspective(img, M, output_size, flags=cv2.INTER_LINEAR)


# ── Optional: mark src/dst points for visual verification ───────────────────

def draw_points(img, pts, color):
    """Draw a 3x3 pixel dot at each point for visual verification."""
    for (x, y) in pts:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                iy, ix = int(y) + dy, int(x) + dx
                if 0 <= iy < img.shape[0] and 0 <= ix < img.shape[1]:
                    img[iy, ix] = color


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_bev_pipeline(image_path, output_size=(512, 512), vertical_shift=0):
    img = cv2.imread(image_path)
    img = cv2.resize(img, output_size)

    # Step 1: collect correspondences interactively
    src = get_src(img, vertical_shift)
    dst = get_dst(img, vertical_shift)
    print("src points:\n", src)
    print("dst points:\n", dst)

    # Step 2: compute homography
    M, M_inv = compute_bev_transform(src, dst)
    print("Homography M:\n", M)

    # Step 3: apply warp
    bev = apply_bev_transform(img, M, output_size)

    # Step 4: save outputs
    base = image_path.rsplit(".", 1)[0]
    cv2.imwrite(base + "_transformed.png", bev)

    debug = img.copy()
    draw_points(debug, src, (0, 0, 255))    # src in red
    draw_points(debug, dst, (0, 255, 0))    # dst in green
    cv2.imwrite(base + "_pts_shown.png", debug)

    cv2.destroyAllWindows()
    return bev, M, M_inv


# ── Re-apply saved M to a new frame (no re-calibration needed) ──────────────

def apply_saved_transform(img, M, output_size=(512, 512)):
    """
    Once M is calibrated for a camera, reuse it on every subsequent frame.
    This is the real-time path — pure matrix multiply, no user interaction.
    """
    return cv2.warpPerspective(img, M, output_size, flags=cv2.INTER_LINEAR)
```

### Step-by-step usage

```python
# First run: calibrate once per camera setup
bev_frame, M, M_inv = run_bev_pipeline("left_camera_frame.png", output_size=(512, 512))

# Save M for reuse
np.save("homography_M.npy", M)

# Subsequent frames: just load and apply — no user interaction
M = np.load("homography_M.npy")
cap = cv2.VideoCapture("lane_video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv2.resize(frame, (512, 512))
    bev = apply_saved_transform(frame_resized, M, output_size=(512, 512))
    cv2.imshow("BEV", bev)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Strengths and limitations

| | |
|---|---|
| **No training data needed** | Works from a single image |
| **Geometrically exact** | Pixel-perfect at the 4 calibration points |
| **Extremely fast** | Single matrix multiply per frame (~1ms) |
| **Fully interpretable** | You can inspect M directly |
| **Flat-road assumption** | Fails on hills, bumps, or any non-planar surface |
| **Manual calibration** | Requires human to select points per camera rig |
| **No adaptation** | Camera move = recalibrate from scratch |
| **No appearance reasoning** | Cannot handle lighting changes or occlusion |

---

## Approach 2: Learned Perspective Transformer

**Source:** `Perspective-BEV-Transformer/`  
**Paper:** arXiv:2311.06796

### How it works

Instead of computing geometry analytically, this approach **learns the mapping from data**. A neural network takes two inputs simultaneously:

1. The front-view image (processed by ResNet50)
2. The bounding box coordinates in the perspective image (4 normalized floats)

It outputs the corresponding bounding box coordinates in BEV space: `[x_min, x_max]` and `[y_min, y_max]`.

The key insight is that the network learns camera intrinsics and perspective geometry implicitly from paired (front-view image + BEV annotation) examples generated in the CARLA simulator.

> **Note:** This approach predicts BEV coordinates of **detected objects** (bounding boxes), not a full image warp. It answers: "given that I see a vehicle at these pixel coordinates, where is it in BEV space?"

### Architecture

```
Input A: perspective-view image (224×224×3)
    └─► ResNet50 (ImageNet weights, frozen)
        └─► layer[175].output → Reshape → [2048]  ← image feature vector

Input B: front-view bbox [x_min/W, y_min/H, x_max/W, y_max/H]  (normalized, shape: 4)
    └─► Dense(256, relu) → Dropout(0.25)
        └─► Dense(256, relu) → Dropout(0.25)
            └─► Dense(256, relu) → h_coords  ← coordinate encoding [256]

Merge: Concatenate([image_feat_2048, h_coords_256]) → merged [2304]

Output X head  (from h_coords alone, NOT merged):
    h_coords → Dense(512, relu) → Dropout(0.25)
             → Dense(128, relu) → Dropout(0.25)
             → Dense(2)          ← [x_min_bev, x_max_bev]

Output Y head  (from merged, uses both image + coords):
    merged   → Dense(1024, relu) → Dropout(0.25)
             → Dense(1024, relu) → Dropout(0.25)
             → Dense(512,  relu) → Dropout(0.25)
             → Dense(256,  relu) → Dropout(0.25)
             → Dense(128,  relu) → Dropout(0.25)
             → Dense(2)          ← [y_min_bev, y_max_bev]
```

The rationale for the split: lateral position (X in BEV) depends mainly on the horizontal position in the image (encodeable from coordinates alone), while depth/forward position (Y in BEV) requires visual context from the image to disambiguate scale and distance.

### Data pipeline

Training data is generated in **CARLA simulator** (version 0.9.13):

```
CARLA Town01_Opt
    ├── Ego vehicle with front RGB camera (1024×768, FOV=110°) at (x=1.0, z=2.0)
    ├── N NPC vehicles spawned
    └── BirdViewProducer attached to ego (200×200px, 4px/meter, front-area crop)

Per frame:
    ├── rgb_array      → front-view image (H, W, 3)
    ├── depth_array    → depth map in meters (H, W)
    ├── bounding_box   → 2D bboxes of visible vehicles [xmin, ymin, xmax, ymax]
    ├── vehicle_ids_rgb      → IDs of vehicles visible in front camera
    ├── vehicle_ids_birdview → IDs of vehicles visible in BEV
    └── bb_bv          → BEV bounding box polygon corners per vehicle

Saved to HDF5 with groups: rgb/, bounding_box/vehicles/, vehicle_ids_rgb/,
                            vehicle_ids_birdview/, bb_bv/, timestamps/
```

The critical pairing step: vehicle IDs are used to join front-view boxes with BEV boxes, so only vehicles visible in **both** views form training pairs.

### Full implementation

#### `model.py` — network definition

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Reshape, Concatenate
from tensorflow.keras.applications import ResNet50
import numpy as np


class PerspectiveTransformer:
    def __init__(self, backbone_model):
        # backbone_model: a compiled ResNet50 instance (include_top=True, weights='imagenet')
        self.backbone_model = backbone_model

    def build(self):
        input_coords = Input(shape=(4,))           # normalized front-view bbox
        input_image  = Input(shape=(224, 224, 3))  # perspective-view image

        # ── Branch 1: image features from ResNet50 ──────────────────────────
        if isinstance(self.backbone_model, ResNet50):
            # layer[175] is the final global-average-pool output → (batch, 2048)
            pv_encoded = Reshape(target_shape=(2048,))(self.backbone_model.layers[175].output)
        else:
            pv_encoded = Reshape(target_shape=(1000,))(self.backbone_model.output)

        # ── Branch 2: coordinate encoding ───────────────────────────────────
        h = Dense(256, activation='relu')(input_coords)
        h = Dropout(rate=0.25)(h)
        h = Dense(256, activation='relu')(h)
        h = Dropout(rate=0.25)(h)
        h = Dense(256, activation='relu')(h)   # h_coords: [256]

        # ── Fusion ───────────────────────────────────────────────────────────
        merged = Concatenate()([pv_encoded, h])  # [2304]

        # ── Output head X (from coords branch only) ──────────────────────────
        hx = Dense(512, activation='relu')(h)
        hx = Dropout(rate=0.25)(hx)
        hx = Dense(128, activation='relu')(hx)
        hx = Dropout(rate=0.25)(hx)
        output_coords_x = Dense(2)(hx)           # [x_min_bev, x_max_bev]

        # ── Output head Y (from merged features) ─────────────────────────────
        hy = Dense(1024, activation='relu')(merged)
        hy = Dropout(rate=0.25)(hy)
        hy = Dense(1024, activation='relu')(hy)
        hy = Dropout(rate=0.25)(hy)
        hy = Dense(512,  activation='relu')(hy)
        hy = Dropout(rate=0.25)(hy)
        hy = Dense(256,  activation='relu')(hy)
        hy = Dropout(rate=0.25)(hy)
        hy = Dense(128,  activation='relu')(hy)
        hy = Dropout(rate=0.25)(hy)
        output_coords_y = Dense(2)(hy)           # [y_min_bev, y_max_bev]

        model = Model(
            inputs=[input_coords, input_image],
            outputs=[output_coords_x, output_coords_y]
        )
        return model
```

#### `data.py` — Keras data generator

```python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, df, batch_size=32):
        """
        df columns:
            img_path          — path to front-view PNG
            front_coords      — np.array([xmin/W, ymin/H, xmax/W, ymax/H])
            birdview_coords_x — np.array([x_min_bev, x_max_bev])
            birdview_coords_y — np.array([y_min_bev, y_max_bev])
        """
        self.df = df
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        # shuffle every epoch
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.indexes = np.arange(len(self.df))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        bv_x      = [tf.convert_to_tensor(self.df.iloc[k]['birdview_coords_x'], dtype=tf.float32) for k in indexes]
        bv_y      = [tf.convert_to_tensor(self.df.iloc[k]['birdview_coords_y'], dtype=tf.float32) for k in indexes]
        rgb_coords = [tf.convert_to_tensor(self.df.iloc[k]['front_coords'],       dtype=tf.float32) for k in indexes]

        images = []
        for k in indexes:
            img = cv2.imread(self.df.iloc[k]['img_path']) / 255.0   # normalize to [0,1]
            img = cv2.resize(img, (224, 224))
            images.append(img)
        image_tensor = tf.convert_to_tensor(images, dtype=tf.float32)

        return [rgb_coords, image_tensor], [bv_x, bv_y]
```

#### `utils.py` — HDF5 reader and evaluation metrics

```python
import os
import h5py
import cv2
import numpy as np
import pandas as pd
import math
import tensorflow as tf


def read_hdf5_file(data_path, dst_path):
    """
    Reads all .hdf5 files in data_path.
    Joins front-view bboxes with BEV bboxes using shared vehicle IDs.
    Saves raw PNGs to dst_path.
    Returns a DataFrame with columns:
        id, img_path, front_coords, birdview_coords_x, birdview_coords_y
    """
    df = pd.DataFrame(columns=['id', 'img_path', 'front_coords',
                                'birdview_coords_x', 'birdview_coords_y'])
    index = 0
    os.makedirs(dst_path, exist_ok=True)

    for file_name in os.listdir(data_path):
        if not file_name.endswith('.hdf5'):
            continue
        with h5py.File(os.path.join(data_path, file_name), 'r') as f:
            rgb           = f['rgb']
            bb_vehicles   = f['bounding_box']['vehicles']
            ids_rgb       = f['vehicle_ids_rgb']
            ids_bv        = f['vehicle_ids_birdview']
            bb_bv         = f['bb_bv']
            timestamps    = f['timestamps']

            for t in timestamps['timestamps']:
                rgb_data    = np.array(rgb[str(t)])
                bb_rgb_data = np.array(bb_vehicles[str(t)])  # flattened [xmin, ymin, xmax, ymax, ...]
                id_rgb_data = np.array(ids_rgb[str(t)])
                id_bv_data  = np.array(ids_bv[str(t)])
                bb_bv_data  = np.array(bb_bv[str(t)])        # polygon corners per vehicle

                # save raw image
                img_path = os.path.join(dst_path, f'raw_img{index}.png')
                cv2.imwrite(img_path, rgb_data)

                id_bv_list  = id_bv_data.tolist()
                id_rgb_list = id_rgb_data.tolist()

                for vid in id_bv_list:
                    if vid in id_rgb_list:
                        idx_bv  = id_bv_list.index(vid)
                        idx_rgb = id_rgb_list.index(vid)

                        pts = bb_bv_data[idx_bv].tolist()   # polygon corners in BEV
                        record = {
                            'id': vid,
                            'img_path': img_path,
                            # BEV: axis-aligned bounding box from polygon corners
                            'birdview_coords_x': np.array([
                                min(pt[0] for pt in pts),
                                max(pt[0] for pt in pts)
                            ]),
                            'birdview_coords_y': np.array([
                                min(pt[1] for pt in pts),
                                max(pt[1] for pt in pts)
                            ]),
                            # Front-view: normalized [xmin/W, ymin/H, xmax/W, ymax/H]
                            'front_coords': np.array([
                                bb_rgb_data[idx_rgb * 4]     / 1024,
                                bb_rgb_data[idx_rgb * 4 + 1] / 768,
                                bb_rgb_data[idx_rgb * 4 + 2] / 1024,
                                bb_rgb_data[idx_rgb * 4 + 3] / 768,
                            ])
                        }
                        df = df.append(record, ignore_index=True)
                        index += 1
    return df


def compute_iou(box1, box2):
    """box format: (x1, y1, x2, y2)"""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter / union if union > 0 else 0.0


def compute_centroid_distance(box1, box2):
    cx1, cy1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
    cx2, cy2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2
    return math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)


def evaluate_metrics(model, test_df):
    """
    Runs inference on every sample in test_df and prints averaged:
        IoU, centroid distance, height error ratio, width error ratio, aspect ratio error
    """
    metrics = dict(iou=0, CD=0, hE=0, wE=0, arE=0)

    for i in range(len(test_df)):
        row      = test_df.iloc[i]
        img      = cv2.imread(row['img_path']) / 255.0
        img      = cv2.resize(img, (224, 224))
        coords   = tf.convert_to_tensor([row['front_coords']], dtype=tf.float32)
        img_t    = tf.convert_to_tensor([img],                 dtype=tf.float32)

        pred_x, pred_y = model.predict([coords, img_t])
        px1, px2 = pred_x[0]
        py1, py2 = pred_y[0]

        gx1, gx2 = row['birdview_coords_x']
        gy1, gy2 = row['birdview_coords_y']

        metrics['iou'] += compute_iou((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
        metrics['CD']  += compute_centroid_distance((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
        metrics['hE']  += (py2-py1) / (gy2-gy1)
        metrics['wE']  += (px2-px1) / (gx2-gx1)
        metrics['arE'] += abs((px2-px1)/(py2-py1) - (gx2-gx1)/(gy2-gy1))

    n = len(test_df)
    print(f"IoU:  {metrics['iou']/n:.4f}")
    print(f"CD:   {metrics['CD']/n:.4f}")
    print(f"hE:   {metrics['hE']/n:.4f}")
    print(f"wE:   {metrics['wE']/n:.4f}")
    print(f"arE:  {metrics['arE']/n:.4f}")
```

### Training

#### `main.py`

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

from utils import read_hdf5_file, evaluate_metrics
from data import DataGenerator
from model import PerspectiveTransformer

# ── 1. Load data ─────────────────────────────────────────────────────────────
data_path = 'data'          # folder containing .hdf5 files
img_path  = 'data/images'   # destination for extracted PNGs
df = read_hdf5_file(data_path, img_path)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ── 2. Build model ────────────────────────────────────────────────────────────
backbone = ResNet50(
    include_top=True,
    weights='imagenet',
    input_tensor=Input(shape=(224, 224, 3))
)
model = PerspectiveTransformer(backbone).build()
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
model.summary()

# ── 3. Train ──────────────────────────────────────────────────────────────────
train_gen = DataGenerator(train_df, batch_size=32)
test_gen  = DataGenerator(test_df,  batch_size=32)

model.fit(train_gen, epochs=100)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
evaluate_metrics(model, test_df)

# ── 5. Save ───────────────────────────────────────────────────────────────────
model.save('perspective_bev_transformer.h5')
```

#### Data generation with CARLA

Before training you need training data. The CARLA pipeline (`Generator/`) generates it:

```python
# Generator/main.py — run inside a CARLA 0.9.13 environment
# python main.py my_dataset -wi 1024 -he 768 -ve 50 -wa 20

from CarlaWorld import CarlaWorld
from HDF5Saver import HDF5Saver

sensor_width, sensor_height, fov = 1024, 768, 110

HDF5_file  = HDF5Saver(sensor_width, sensor_height, "data/my_dataset.hdf5")
carla_world = CarlaWorld(HDF5_file=HDF5_file)

carla_world.spawn_npcs(number_of_vehicles=50, number_of_walkers=20)

egos_to_run = 2
timestamps  = []

for weather_option in carla_world.weather_options:          # 5 weather presets
    carla_world.set_weather(weather_option)
    for _ in range(egos_to_run):                             # 2 ego vehicles
        carla_world.begin_data_acquisition(
            sensor_width, sensor_height, fov,
            frames_to_record_one_ego=2,                      # frames per ego
            timestamps=timestamps,
            egos_to_run=egos_to_run
        )

carla_world.remove_npcs()
HDF5_file.record_all_timestamps(timestamps)
HDF5_file.close_HDF5()
```

Inside `CarlaWorld`, each frame is captured in synchronous mode and saved via `HDF5Saver`:

```python
# Per-frame data capture (inside CarlaWorld.begin_data_acquisition)
_, rgb_data, depth_data = sync_mode.tick(timeout=2.0)

rgb_array, bounding_box, vehicle_ids_rgb, names, distances = \
    self.process_rgb_img(rgb_data, sensor_width, sensor_height, ego_vehicle)

# BirdView from CARLA's top-down renderer
birdview, vehicle_ids_birdview, coords = \
    self.birdview_producer.produce(agent_vehicle=ego_vehicle)

# Camera intrinsic matrix K (built during sensor setup)
calibration = np.identity(3)
calibration[0, 2] = sensor_width  / 2.0
calibration[1, 2] = sensor_height / 2.0
calibration[0, 0] = calibration[1, 1] = \
    sensor_width / (2.0 * np.tan(fov * np.pi / 360.0))
self.rgb_camera.calibration = calibration

# Save everything to HDF5
self.HDF5_file.record_data(
    rgb_array, depth_array, bounding_box, ego_speed,
    birdview, vehicle_ids_rgb, vehicle_ids_birdview,
    coords, names, distances, timestamp
)
```

The HDF5 file structure:

```
carla_dataset.hdf5
├── rgb/                       # front-view images per timestamp
│   └── <timestamp>: (H, W, 3)
├── bounding_box/
│   └── vehicles/
│       └── <timestamp>: [xmin, ymin, xmax, ymax, ...]  # flattened, 4 per vehicle
├── vehicle_ids_rgb/           # IDs of vehicles visible in front cam
├── vehicle_ids_birdview/      # IDs of vehicles visible in BEV
├── bb_bv/                     # BEV polygon corners per visible vehicle
├── depth/                     # depth map in meters
└── timestamps/
    └── timestamps: [t0, t1, ...]
```

### Inference (after training)

```python
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('perspective_bev_transformer.h5')

def predict_bev_box(model, image_path, front_bbox_normalized):
    """
    image_path: path to front-view image
    front_bbox_normalized: [xmin/W, ymin/H, xmax/W, ymax/H]

    Returns: (x_min_bev, x_max_bev, y_min_bev, y_max_bev)
    """
    img = cv2.imread(image_path) / 255.0
    img = cv2.resize(img, (224, 224))
    img_t    = tf.convert_to_tensor([img],                    dtype=tf.float32)
    coords_t = tf.convert_to_tensor([front_bbox_normalized],  dtype=tf.float32)

    pred_x, pred_y = model.predict([coords_t, img_t])
    return pred_x[0][0], pred_x[0][1], pred_y[0][0], pred_y[0][1]
```

### Evaluation metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **IoU** | intersection / union | Overlap between predicted and GT BEV box |
| **Centroid Distance (CD)** | L2 distance between box centers | Positional accuracy |
| **Height Error (hE)** | pred_h / gt_h | Scale accuracy in depth direction |
| **Width Error (wE)** | pred_w / gt_w | Scale accuracy in lateral direction |
| **Aspect Ratio Error (arE)** | \|pred_ar - gt_ar\| | Shape accuracy |

### Strengths and limitations

| | |
|---|---|
| **No manual calibration** | Learns camera geometry from data |
| **Adapts to appearance** | Handles lighting, occlusion, weather variation |
| **Generalizes across rigs** | Can fine-tune to a new camera with limited data |
| **Requires CARLA + GPU** | Heavy data generation and training infrastructure |
| **Predicts boxes, not pixels** | Does not produce a full BEV image warp |
| **Slower inference** | CNN forward pass + 8 dense layers |
| **Sim-to-real gap** | Trained on CARLA; real-world performance needs validation |
| **Flat-road assumed** | CARLA generates flat urban environments |

---

## Decision Guide

```
Is your camera position fixed and unlikely to change?
  ├─ YES → Does flat-road assumption hold for your environment?
  │          ├─ YES → Use Approach 1 (Homography). Calibrate once, save M, done.
  │          └─ NO  → Neither approach is ideal. Consider learning-based with
  │                   elevation data, or monocular depth + point cloud methods.
  └─ NO  → Do you have (or can generate) paired front-view + BEV training data?
             ├─ YES → Use Approach 2 (Learned). More robust to camera changes.
             └─ NO  → Use Approach 1 and re-calibrate when camera changes.

Do you need a full image warp (all pixels remapped to BEV)?
  ├─ YES → Use Approach 1. Approach 2 only predicts object bounding boxes.
  └─ NO  → Either works.

Do you need real-time performance on embedded hardware (Jetson, etc.)?
  ├─ YES → Use Approach 1. Matrix multiply is trivially fast.
  └─ NO  → Either works. Approach 2 runs fine on a modern GPU.

Is this a research/simulation context (CARLA available)?
  ├─ YES → Approach 2 is worth the overhead for generalizability.
  └─ NO  → Start with Approach 1; add Approach 2 only if adaptation is needed.
```

### Quick summary

| | Approach 1: Homography | Approach 2: Learned |
|---|---|---|
| **Setup time** | Minutes (manual calibration) | Days (data gen + training) |
| **Inference speed** | ~1ms | ~50ms (GPU) |
| **Output** | Full BEV image warp | Object BEV bounding boxes |
| **Accuracy** | Pixel-perfect at calibration points | Generalizes across conditions |
| **Dependencies** | `opencv-python`, `numpy` | TensorFlow, CARLA, HDF5 |
| **Camera change** | Re-calibrate manually | Fine-tune or retrain |
| **Best for** | Fixed camera, real-time, known geometry | Adaptive systems, simulation research |
