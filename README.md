# Swervin: SAS(Swerve Assist System)

Swervin implements SAS, a vision-based turn assist prototype that estimates lane geometry from a front camera feed and compares current steering input against a recommended steering target.

The goal is not full autonomy. The goal is a real-time assistance system that only activates when lane structure is detected with high confidence.

---

## Project Goal

Swervin is a perception-to-control pipeline for turn assist on clearly visible lane roads.

It is designed to:

- detect lane structure from a camera feed
- normalize lane geometry into bird's-eye view
- estimate a centerline and curvature
- compute a steering recommendation
- compare recommended steering against actual steering input
- visualize current vs desired steering
- suppress guidance when lane confidence is too low

It is **not** intended to:
- force lane predictions when lane evidence is weak
- act as a lane change assistant in MVP
- solve full 3D lane detection
- replace the driver

---

## Core Pipeline

1. Camera frame
2. Lane segmentation
3. OpenCV perspective transform to BEV
4. Extract lane mask points in BEV
5. Fit lane centerline polynomial
6. Compute curvature and lookahead point
7. Compute Pure Pursuit steering target
8. Compare desired steering against actual steering input
9. Visualize current vs desired steering

### Design Rules

- The segmentation model should be swappable through a unified input/output interface.
- Turn assist should only activate when lane detection confidence is high.
- The system should prefer disabling itself over hallucinating a lane.

---

## System Architecture

### 1. Camera Input
Front-facing camera feed captured from a fixed, calibrated setup.

Requirements:
- fixed camera mount
- fixed resolution and crop
- stable intrinsics/extrinsics for BEV

---

### 2. Lane Segmentation

Segmentation is the main perception layer for MVP.

#### Candidate Models

**ERFNet**
- best practical balance of segmentation quality, lower false positives, and manageable size
- paper comparison size: 8.07 MB
- recommended starting model

**ENet**
- lightweight and fast
- paper comparison size: 1.53 MB
- lower precision and higher false positives than ERFNet
- good fallback if speed becomes the bottleneck

**DeepLabv3+**
- stronger feature extraction in principle
- heavy for MVP
- paper comparison size: 45.21 MB
- not the preferred starting point

**Later: InSegNet-v4**
- highest reported performance in the referenced paper
- paper comparison size: 45.18 MB
- no public source code found
- would require reimplementation

#### Segmentation Model Source Candidates

Preferred pretrained model sources:
1. `voldemortX/pytorch-auto-drive`
2. `cardwing/Codes-for-Lane-Detection`
3. `Turoad/lanedet`

#### Model Interface Requirement

All segmentation backends must expose a unified interface:

```python
mask, confidence, metadata = segment_lane(frame)
```

Where:

- `mask` is a binary or class lane mask
- `confidence` is a scalar or structured confidence output
- `metadata` may include logits, class maps, timing, or debug info

### 3. BEV

Bird's-eye view is used to reduce perspective distortion and make lane geometry more usable for control.

Implementation:

OpenCV homography / perspective transform
IPM-style transform

**Placement in pipeline:**
- after segmentation
- before geometry extraction

**Reason:**
- transform the lane mask, not the raw image
- cleaner geometry, less irrelevant noise

### 4. Lane Geometry / Classical Post-Processing

After BEV, the system extracts lane structure and computes a control-ready path.

**Components:**
- contour / point extraction from mask
- optional connected-component filtering
- polynomial fitting
- optional RANSAC smoothing
- centerline generation

**Outputs:**
- left lane points
- right lane points
- centerline
- heading estimate
- curvature
- lookahead point
- lane confidence / validity
### 5. Control Math

**MVP controller:** Pure Pursuit

**Later option:** Stanley controller

Pure Pursuit is preferred initially because it is:
- simple
- interpretable
- easy to integrate with centerline geometry

**Inputs:**
- centerline
- lookahead point
- current vehicle position estimate in BEV
- current steering input
- optional vehicle speed

**Outputs:**
- desired steering target
- steering error relative to current steering

### 6. Visualization

The visualizer should show:
- detected lane mask
- BEV projection
- fitted centerline
- current steering
- desired steering
- confidence state

**Behavior:** If lane confidence is low, disable assist visualization instead of drawing a forced steering target.

---

## Dataset Strategy

The model should be focused on reliably detecting visible lanes, not aggressively detecting heavily obscured lanes.

### Candidate Public Datasets

**CurveLanes**
- 150K (mostly 2650×1440)
- 100K train
- 20K val
- 30K test
- fully annotated
- best alignment with turn-assist geometry and curved-road behavior

**CULane**
- 133,235 frames (1640x590)
- 88,880 train
- 34,680 test
- 9,675 validation
- strong mainstream lane benchmark
- useful for robustness

**LLAMAS**
- over 100K annotated images
- useful for segmentation-oriented training

**TuSimple**
- 6,408 labeled images (1280x720)
- 3,626 train
- 2,782 terst
- 358 validation
- highway-focused
- easier and cleaner
- useful for baseline startup
### Dataset Direction

**For MVP:**
1. start from a pretrained public lane model
2. then fine-tune on data collected from the exact deployment camera/setup

**Rationale:**
- lane segmentation performance drops hard when train and real environments differ
- camera position, crop, aspect ratio, FOV, and road appearance matter a lot
- target-domain fine-tuning is more valuable than chasing generic leaderboard performance
### Custom Data Collection Goals

**Collect data with:**
- exact camera setup
- visible lane roads
- curves of varying sharpness
- some shadows
- some partial occlusions
- some lower-light conditions later

**Do not optimize for:**
- always forcing detection
- every obscure scenario in MVP
---

## Confidence and Activation Logic

Swervin should behave like a conservative lane-assist system.

**Assist should activate only when:**
- lane mask confidence is high
- centerline is stable
- visible lane extent is sufficient
- BEV geometry is plausible
- curvature estimate is stable across frames

**Assist should deactivate when:**
- lane evidence is weak
- geometry becomes unstable
- segmentation confidence falls below threshold
- one or both lane boundaries become unreliable

**Principle:** Better to show no guidance than wrong guidance

---

## Initial Recommended Stack

**Segmentation**
- ERFNet first
- ENet as lightweight fallback

**BEV**
- OpenCV perspective transform

**Geometry**
- lane mask point extraction
- polynomial fit
- optional RANSAC
- centerline generation

**Control**
- Pure Pursuit

**Object Detection**
- not required for MVP turn assist
- useful later for lane change assist or context-aware suppression
---

## Later Extensions

### 1. Object Detection

Add object detection only after the lane-only turn assist pipeline works.

**Potential use cases:**
- lane change assist
- obstacle-aware suppression
- contextual warnings

### 2. Improved Segmentation with Attention

From the DeepLabv3+ attention paper:
- FCA
- CBAM
- ECA
- SE

**Reported benefit:** Large mIoU improvement in that paper's setup

**Reported drawback:** Around 10 FPS

**Conclusion:**
- only worth exploring if segmentation quality becomes the bottleneck
- do not add before the full MVP pipeline works

### 3. InSegNet Reimplementation

From the InSegNet paper:
- encoder-decoder segmentation model with 4 stages
- stages 1 and 2: convolution blocks + max pooling
- stages 3 and 4: inception blocks
- v2 adds encoder-decoder skip links
- v3 adds residual connections in encoder
- v4 modifies some filter settings vs v3

**Reported settings:**
- batch size 12
- 200 epochs
- Adam
- binary cross-entropy loss
- metrics: binary accuracy and binary IoU

Reported performance:

Model	Precision	Recall	F1	FPR	AUC
InSegNet-v3	0.959	0.940	0.949	0.0032	0.978
InSegNet-v4	0.962	0.936	0.949	0.0029	0.978

**Conclusion:**
- v4 is the strongest reported version
- no public implementation found
- should only be attempted later if needed

---

## Project Priorities

### MVP Priorities
1. Get lane segmentation running on real footage
2. Build BEV transform
3. Extract centerline geometry
4. Implement Pure Pursuit steering target
5. Add confidence-based activation gating
6. Visualize current vs desired steering

### Non-Priorities for MVP
- lane change assist
- object detection
- full 3D lane detection
- SOTA segmentation research
- domain-general lane detection everywhere

---

## Engineering Principles
- modular model swapping
- fixed camera setup
- confidence-first activation
- prefer interpretable geometry over black-box steering
- optimize for a working end-to-end system before chasing segmentation leaderboard gains
---

## Working Summary

Swervin is a segmentation-first, BEV-based, geometry-driven turn assist prototype.

The current recommended MVP stack is:
- ERFNet
- OpenCV BEV
- polynomial centerline fitting
- Pure Pursuit
- confidence-gated visualization

That is the shortest path to a serious, defensible perception-to-control system.