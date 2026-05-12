# Design Notes

## BEV Transformation

### Initial Approaches | 2026-05-12

The current pipeline applies a fixed planar homography (`cv2.warpPerspective`) to the segmentation mask, then fits independent quadratic polynomials to each lane class. Per the research survey (`research/bev-deep-research-report.md`), the primary error source is **not** the polynomial fit — it is the flat-plane road assumption embedded in the homography. Slopes, crests, dips, and camera pitch changes all distort the warped geometry before fitting even begins.

The following approaches are ordered by implementation effort. Items 1–4 are confined to `lane_geometry.py` and require no model changes. Item 5 touches `bev_transformer.py`. Item 6 is post-MVP.

---

#### 1. Confidence-weighted polynomial fitting (IRLS + Huber loss)

**Current behavior:** `np.polyfit` weights all extracted lane points equally. Noisy pixels at the edges of the BEV canvas (low detection confidence, wide blobs) pull the fit as much as clean central pixels.

**Proposed change:** Replace unweighted least squares with iteratively reweighted least squares (IRLS) using a Huber loss. Per-point weights are derived from lane confidence scores (from `SegResult.lane_confidences`) or from residuals of the previous iteration. Low-confidence or high-residual points are down-weighted automatically without hard rejection.

**Expected benefit:** Reduced heading and curvature noise on frames where one edge of the BEV mask is partially occluded or poorly warped.

**Papers:**
- BEV-LaneDet (Wang et al., CVPR 2023) — confidence + offset + embedding decoding; confidence scores propagated into geometry
- HeightLane (Park et al., arXiv 2025) — per-cell confidence used in weighted BEV accumulation
- LaneCPP (Pittner et al., CVPR 2024) — explicit robust weighting in spline control-point fitting

---

#### 2. Joint left+right boundary fitting with soft lane-width prior

**Current behavior:** Left and right polynomial coefficients are fit independently. When one boundary is partially occluded or misdetected, the centerline shifts laterally with no constraint from the other boundary.

**Proposed change:** Fit both boundaries jointly with a soft penalty on lane-width deviation: `λ · Σ |(x_right(y) − x_left(y)) − W_expected|²`, where `W_expected` is the nominal lane width in BEV pixels (derivable from calibration geometry or tuned empirically). Relax the penalty near merges and splits by gating on per-frame class confidence.

**Expected benefit:** Centerline stability when one boundary is partially occluded — the visible boundary constrains the invisible one through the width prior rather than falling back to the crude `half_lane_width_px` offset.

**Papers:**
- Anchor3DLane (Chen et al., CVPR 2023) — equal-width optimization as a post-processing step to reduce lateral error
- LaneCPP (Pittner et al., CVPR 2024) — analytical parallelism and lane-width priors applied jointly over both boundaries

---

#### 3. Piecewise cubic B-spline instead of global quadratic polynomial

**Current behavior:** A single degree-2 polynomial `x = ay² + by + c` is fit across the full BEV y-range. A single bad cluster (thick near-vehicle blob, distant noise) corrupts the entire fit because a global polynomial has no local control.

**Proposed change:** Replace `np.polyfit` / `np.polyval` with a cubic B-spline (e.g. `scipy.interpolate.make_splrep`) with 4–6 uniformly spaced knots over the reliable BEV range (`y_top` to `y_bot`). A B-spline has local support: a bad segment near the vehicle does not distort the far-range fit. Curvature and heading become well-defined at any arc-length station via the spline's analytic derivative.

**Expected benefit:** More stable curvature estimates mid-range and far-range; better-behaved derivatives for Pure Pursuit; graceful handling of local lane irregularities without global polynomial distortion.

**Papers:**
- 3D-SpLineNet (Pittner et al., WACV 2023) — first explicit comparison of polynomial vs. Bézier vs. B-spline parameterizations; B-splines win on geometric error and processing speed
- LaneCPP (Pittner et al., CVPR 2024) — extends 3D-SpLineNet to continuous 3D B-splines with curvature and surface-smoothness penalties

---

#### 4. Temporal smoothing over heading and curvature in ego frame

**Current behavior:** Heading and curvature are computed fresh each frame from the per-frame polynomial fit. Frame-to-frame jitter in these values is passed directly to the controller.

**Proposed change:** Maintain a Kalman filter (or simpler exponential smoother) over the geometry state `[centerline_lateral_offset, heading, curvature]` in the current ego frame. On each frame: (1) predict forward using odometry / IMU ego-motion, (2) update with the new fit. Smooth geometry at fixed arc-length stations, not only the final steering target. Gate the update step on per-frame lane confidence so a bad frame does not corrupt the smoothed state.

**Expected benefit:** Eliminates frame-to-frame heading jitter that is more damaging to steering than modest per-frame geometric error; provides graceful degradation during momentary occlusion.

**Papers:**
- Anchor3DLane++ / temporal Anchor3DLane (Chen et al., extensions of CVPR 2023) — temporal state tracking over 3D lane anchors in ego frame
- CaliFree3DLane (IEEE T-ITS 2024) — spatio-temporal BEV without fixed camera parameters; demonstrates stability gains from temporal ego-frame lane state

---

#### 5. Dynamic road-plane estimation for BEV homography

**Current behavior:** `compute_bev_transform` uses a single fixed homography calibrated on a flat road. Any camera pitch change (slope, crest, dip, speed bump) shifts the apparent road plane and distorts the BEV output before any fitting occurs.

**Proposed change:** Estimate camera pitch per-frame from either (a) IMU pitch measurements or (b) the vanishing point of detected lane lines in the front-view mask. Recompute the homography from a dynamic road-plane model before applying `warpPerspective`. No model retraining required — only a pitch estimator feeding into `compute_bev_transform` in `bev_transformer.py`.

**Expected benefit:** Fixes curvature and heading errors that are currently introduced by pitch variation; the most impactful single change per the research survey since the flat-plane assumption is the primary geometric error source.

**Papers:**
- HeightLane (Park et al., arXiv 2025) — learned per-frame heightmap replaces flat-plane assumption; the classical analogue is dynamic pitch estimation + homography recomputation
- LaneCPP (Pittner et al., CVPR 2024) — road-surface-hypothesis lifting: features accumulated against multiple surface hypotheses rather than a single plane

---

#### 6. Height-aware learned BEV transform (post-MVP)

**Current behavior:** `warpPerspective` with a fixed homography; see items above.

**Proposed change:** Replace the explicit warp with a learned module that predicts a dense BEV heightmap (multi-slope anchors) and uses deformable attention to sample front-view features at positions consistent with the predicted ground shape. Requires retraining and LiDAR-derived height ground truth for supervision.

**Expected benefit:** Full correction of slope-induced geometric distortion including cases where pitch changes faster than a per-frame estimator can track. Enables reliable 3D lane geometry (x, y, z) rather than planar BEV coordinates.

**Trade-off:** Significant additional complexity; LiDAR or stereo depth needed for height labels; best pursued after the classical pipeline (items 1–5) is saturated and slope errors remain the dominant failure mode.

**Papers:**
- HeightLane (Park et al., arXiv 2025) — multi-slope anchor heightmap prediction + deformable BEV attention; OpenLane val F1 62.5
- LaneCPP (Pittner et al., CVPR 2024) — road-surface-hypothesis lifting to 3D; continuous B-spline output; OpenLane val F1 60.3
