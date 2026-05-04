# Changelog

## Unreleased

### Bug Fixes

#### model_loader imported sas_process with wrong case — ModuleNotFoundError on Linux
Date: 2026-05-04
**Type:** Bug Fix
**Context:** `model_loader.py` imported `from sas.utils.sas_process import SASProcessClass`. On Windows (NTFS, case-insensitive) this resolved to `SAS_Process.py`, but on Linux/WSL2 the filesystem is case-sensitive and the module was not found, crashing startup.
**Cause:** Module name in import did not match the actual filename casing.
**Fix:** Changed import to `from sas.utils.SAS_Process import SASProcessClass`.
**Files changed:**
- `src/sas/utils/model_loader.py`
**Impact:** Pipeline starts correctly on Linux/WSL2.

#### apply_bev_transform used INTER_LINEAR on segmentation class masks
Date: 2026-05-04
**Type:** Bug Fix
**Context:** `warpPerspective` with `INTER_LINEAR` interpolates between adjacent pixel values. For a segmentation mask with discrete class labels (0–4), this produces fractional values (e.g. 1.5) at class boundaries, corrupting lane class identity and making per-class point extraction unreliable.
**Cause:** Default interpolation was `cv2.INTER_LINEAR`, which is correct for images but wrong for class masks.
**Fix:** Changed default to `cv2.INTER_NEAREST` via a `flags` parameter; callers can pass `INTER_LINEAR` for RGB images.
**Files changed:**
- `src/sas/utils/bev_transformer.py`
**Impact:** BEV mask class labels are preserved exactly; per-class lane point extraction in geometry step is now correct.

#### lane_exist output treated as probabilities but is raw logits
Date: 2026-05-03
**Type:** Bug Fix
**Context:** `ONNXERFNet._postprocess` called `.tolist()` directly on `lane_exist[0]`, producing values in the range ~2–7 instead of [0, 1]. These unbounded values were stored in `SegResult.lane_confidences` and displayed in the GUI stats panel.
**Cause:** The ERFNet CULane model outputs `lane_exist` as raw logits (pre-sigmoid binary classification scores), not probabilities.
**Fix:** Applied sigmoid (`1 / (1 + exp(-x))`) to `lane_exist[0]` before converting to a list.
**Files changed:**
- `src/sas/models/optimized_models.py`
**Impact:** Per-lane confidence values are now correctly bounded to [0, 1] and interpretable as probabilities.

#### BEV calibration image resized to wrong coordinate space
Date: 2026-05-03
**Type:** Bug Fix
**Context:** `run_bev_calibration` resized the calibration image to `output_size` (512×512) before calling `get_src`. The src points were therefore in 512×512 space, but `apply_bev_transform` at inference time receives the segmentation mask at 288×800 (the model's `INPUT_SIZE`). The homography mapped from the wrong input space, producing a black or distorted BEV output.
**Cause:** `output_size` (BEV canvas dimensions) was used where `mask_size` (seg model output dimensions) was needed.
**Fix:** Added `mask_size = [800, 288]` to `[bev]` config. `run_bev_calibration` now resizes the calibration image to `mask_size` so src points are in the correct coordinate space. Added cancellation guard so an empty `get_src` return exits cleanly.
**Files changed:**
- `src/sas/run_sas.py`
- `config.toml`
**Impact:** Homography now correctly maps from the segmentation mask space to the BEV canvas, producing a valid perspective transform.

#### BEVResult not imported in SAS_Process — silent NameError crashed runner thread
Date: 2026-05-03
**Type:** Bug Fix
**Context:** `SAS_Process.py` used `BEVResult(...)` on line 52 after setting `self.results.bev`, but only imported `SASResults` and `SegResult`. The `NameError` was unhandled inside the runner thread, silently killing it — `output_queue` stayed empty, no frames were ever sent.
**Cause:** Missing import after the `SASResults.bev_mask` → `SASResults.bev: BEVResult` schema change.
**Fix:** Added `BEVResult` to the import from `sas.utils.sas_results`.
**Files changed:**
- `src/sas/utils/SAS_Process.py`
**Impact:** Runner thread no longer crashes silently on first frame; BEV results are stored correctly.

### Features

#### Lane geometry extraction from BEV mask
Date: 2026-05-04
**Type:** Feature
**Context:** The BEV mask was produced but never analysed — the geometry step was stubbed out. This implements polynomial lane fitting, centerline generation, curvature, heading, and lookahead point extraction.
**Change:** New `src/sas/utils/lane_geometry.py` with: per-class centroid-per-row point extraction (avoids bottom-heavy bias from thick BEV blobs), quadratic polynomial fit `x = f(y)` with outlier rejection, curvature formula `κ = x''/(1+x'²)^(3/2)`, centerline from averaged left+right polynomials (with single-lane fallback using `half_lane_width_px` offset). Dataset-specific lane class preference order in `DATASET_LANE_CONFIG` dict (currently CULane/TuSimple/LLAMAS all map classes 2,3 as ego boundaries). `SAS_Process.py` wired to call `extract_geometry` after BEV transform and populate `GeometryResult`.
**Behavior:** When BEV is enabled and lanes are detected, `results.geometry` is populated with centerline (50×2 array), heading, curvature, and lookahead point.
**Files changed:**
- `src/sas/utils/lane_geometry.py` (new)
- `src/sas/utils/SAS_Process.py`
**Impact:** Pipeline now produces control-ready lane geometry. Pure Pursuit controller can be wired in next.

#### BEV side panel replaces corner inset
Date: 2026-05-04
**Type:** Feature
**Context:** The 200×200 corner inset was too small to inspect lane geometry and showed no geometry overlay. It also overwrote part of the camera frame.
**Change:** Replaced inset with a full H×H square panel hstacked to the right of the camera frame. Panel shows colorized BEV mask (green lanes), with yellow centerline polyline and red lookahead dot overlaid when geometry is available. Falls back to seg mask (no geometry) when BEV is not enabled. Implemented as `_build_bev_panel()` module-level function in `runner.py`.
**Behavior:** Transmitted frame is now `(W + H) × H`. BEV panel shows lane mask with geometry overlaid in real time.
**Files changed:**
- `src/sas/runner.py`
**Impact:** Lane geometry is visually inspectable in the GUI; camera view is no longer partially obscured.

#### GUI geometry stats section; FHD default window size; centered on screen
Date: 2026-05-04
**Type:** UX
**Context:** The StatsPanel showed only FPS and per-lane confidence. Geometry values (curvature, heading, lookahead) were computed but had no display. The window also opened at 960×720 in the top-left corner.
**Change:** Added `── Geometry ──` section to `StatsPanel` with Curvature, Heading (rad), and Lookahead (x, y) labels. Window default size changed to 1920×1080 via `resize()`; position centered on available screen via `QDesktopWidget().availableGeometry()`.
**Behavior:** Stats panel shows geometry values live when BEV and geometry extraction are active. Window opens centered on screen at FHD resolution.
**Files changed:**
- `src/sas/gui_frontend.py`
**Impact:** All pipeline outputs are visible in the GUI without manual repositioning.

#### Lane stats panel showing FPS and per-lane confidence
Date: 2026-05-03
**Type:** Feature
**Context:** Per-lane confidence values from `lane_exist` were computed but never surfaced to the user. FPS was also not visible anywhere in the GUI.
**Change:** Added `StatsPanel` widget to `gui_frontend.py`. Placed in a `QHBoxLayout` bottom row to the right of the tilt indicator (stretch 1:3). Displays `FPS: x.x` and `L1–L4: x.xx` confidence values, updated every timer tick. FPS is measured per-frame in `runner.py` using `time.monotonic()` and written to `SASResults.fps`. Per-lane confidences flow through `SegResult.lane_confidences` → `to_json()` → client.
**Behavior:** Bottom row shows tilt indicator on the left and stats panel on the right. Values show "—" until first frame arrives, then update live.
**Files changed:**
- `src/sas/gui_frontend.py`
- `src/sas/runner.py`
- `src/sas/utils/sas_results.py`
- `src/sas/utils/SAS_Process.py`
- `src/sas/models/optimized_models.py`
**Impact:** FPS and per-lane detection confidence are visible in the GUI in real time.

#### BEV lane mask inset in transmitted GUI frame
Date: 2026-05-03
**Type:** Feature
**Context:** The BEV mask was computed but never shown in the GUI. Displaying it required no new queues or protocol changes since the frame is already JPEG-encoded and sent to the client.
**Change:** After `sas_process` returns, if `results.bev` is set, the binary BEV mask is colorized (lane pixels green, background black), resized to 200×200, and composited into the top-right corner of the camera frame before it enters `send_queue`.
**Behavior:** The GUI frame now shows the camera view with a green BEV lane overlay inset in the top-right corner. When no lanes are detected the inset is dark.
**Files changed:**
- `src/sas/runner.py`
**Impact:** BEV lane visualization is visible in the GUI with no client-side changes and ~1ms added processing cost.

### Refactors

### Performance

#### Remove large numpy arrays from SASResults.to_json()
Date: 2026-05-03
**Type:** Performance
**Context:** `to_json()` serialized `seg.mask` (288×800 = 230K values) and `bev.bev_mask` + `bev.src_points` as JSON lists on every frame. At ~3 chars/value this added ~800KB to every socket transmission, severely limiting throughput.
**Change:** Removed `segmentation_mask`, `bev_mask`, and `src_points` from the JSON payload. Only scalar metadata (confidence, steering values, geometry scalars) is serialized.
**Result:** JSON payload drops from ~800KB to ~50 bytes per frame.
**Files changed:**
- `src/sas/utils/sas_results.py`
**Impact:** Eliminates the primary socket bottleneck; frame transmission is no longer dominated by JSON size.

#### ONNX CUDA kernel warmup at model load time
Date: 2026-05-03
**Type:** Performance
**Context:** `ONNXERFNet` with `device = "cuda"` compiled CUDA kernels on the first `session.run()` call inside the runner loop, silently stalling the pipeline for 30–120 seconds with no output or progress indication.
**Change:** Added an explicit dummy warmup inference (`np.zeros` tensor) in `ONNXERFNet.__init__` immediately after the session is created, gated on `device == 'cuda'`.
**Result:** Kernel compilation now happens at startup with a printed status message. Subsequent `session.run()` calls in the runner hit pre-compiled kernels and run at full speed.
**Files changed:**
- `src/sas/models/optimized_models.py`
**Impact:** Eliminates silent mid-pipeline stall; startup delay is now visible and expected.

### Tests

### Documentation

### Configuration

### Dependencies

### Data and Schema

#### dataset propagated through model metadata
Date: 2026-05-04
**Type:** Data/Schema
**Context:** `ONNXERFNet` accepted a `dataset` constructor arg but discarded it after init. The geometry module needs to know the dataset to look up the correct lane class order (CULane vs others), but had no access to it.
**Change:** Added `self.dataset = dataset` to `ONNXERFNet.__init__`. Added `'dataset': self.dataset` to the metadata dict returned by `_postprocess`. `SAS_Process.py` reads `metadata.get('dataset', 'culane')` and passes it to `extract_geometry`.
**Files changed:**
- `src/sas/models/optimized_models.py`
- `src/sas/utils/SAS_Process.py`
**Impact:** Switching datasets in config automatically uses the correct lane boundary class mapping in geometry extraction without any code changes.

#### fps and lane_confidences added to SASResults schema and JSON payload
Date: 2026-05-03
**Type:** Data/Schema
**Context:** `SASResults` had `fps` stubbed out as a comment; `SegResult` had no per-lane breakdown. Both were needed for the stats panel.
**Change:** Added `fps: float | None` to `SASResults`; added `lane_confidences: list | None` to `SegResult`. Both fields are reset in `reset()`, serialized in `to_json()` (`fps` rounded to 1 decimal, `lane_confidences` as a 4-element list). `BaseModelClass._postprocess` and `infer` signatures updated from `(mask, confidence)` to `(mask, confidence, dict)`.
**Files changed:**
- `src/sas/utils/sas_results.py`
- `src/sas/models/base_model.py`
**Impact:** Downstream consumers (GUI, future logging) receive FPS and per-lane confidence in every JSON frame without additional protocol changes.

#### BEVResult dataclass; rename bev_mask field in SASResults
Date: 2026-05-03
**Type:** Data/Schema
**Context:** `SASResults.bev_mask` was a bare `np.ndarray` field with no structured container, inconsistent with `SegResult` and `GeometryResult` which both use typed dataclasses.
**Change:** Added `BEVResult(bev_mask, src_points)` dataclass. Renamed `SASResults.bev_mask: np.ndarray` → `SASResults.bev: BEVResult | None`. Updated `reset()` and `to_json()` accordingly. `to_json()` now serializes `bev_mask` and `src_points` when present.
**Files changed:**
- `src/sas/utils/sas_results.py`
**Impact:** BEV output is now a typed container matching the rest of the result schema. Callers setting `results.bev_mask = arr` must be updated to `results.bev = BEVResult(bev_mask=arr, src_points=pts)`.

> **Warning:** `to_json()` also re-added `segmentation_mask` (full binary mask as a list) to the JSON payload. This was previously removed because serializing a 230K-element array on every frame significantly bloats the socket payload and slows transmission. Consider removing it again unless the client actively uses it.

### Security

### UX

#### Independent-corner BEV calibration trapezoid with smaller markers
Date: 2026-05-04
**Type:** UX
**Context:** The symmetric trapezoid UI forced equal left/right half-widths, making it impossible to align both sides with lane lines when the camera is offset from the road center (as with CULane). This caused angled lanes in BEV and a non-zero heading bias on straight roads. The 8px corner circles also obscured the lane markings beneath them.
**Change:** Replaced `cx/top_hw/bot_hw` state with four independent x-values (`tl_x`, `tr_x`, `bl_x`, `br_x`) plus shared `top_y`/`bot_y` per edge. Each corner now moves freely in X; dragging a corner also moves its edge's shared Y so top/bottom edges stay horizontal. Body drag translates all corners together. Corner marker radius reduced from 8px to 3px; hit radius from 14px to 6px.
**Behavior:** User can align the left trapezoid side with the left lane and the right side with the right lane independently. Horizontal edge constraint is still enforced.
**Files changed:**
- `src/sas/utils/bev_transformer.py`
**Impact:** Camera-offset setups can now produce parallel vertical lanes in BEV; heading bias on straight roads is eliminated after recalibration.

#### Drag-handle trapezoid calibration UI for BEV
Date: 2026-05-03
**Type:** UX
**Context:** The previous `get_src` required manually clicking 4 exact pixel positions for TL/TR/BR/BL. It was hard to satisfy the geometric constraint (equal-depth near pair, equal-depth far pair) by eye, producing skewed BEV output where lane lines weren't parallel.
**Change:** Replaced click-4-points with a drag-handle trapezoid overlay. A symmetric green trapezoid is shown on the calibration image at startup. Dragging a corner adjusts its edge's y-position and half-width symmetrically; dragging the body translates the whole shape. Press Enter to confirm, Q to cancel.
**Behavior:** Window stays open until Enter/Q. Cancellation returns an empty array; caller exits cleanly. The shape enforces horizontal top/bottom edges by construction, guaranteeing a valid ground-plane rectangle.
**Files changed:**
- `src/sas/utils/bev_transformer.py`
**Impact:** Calibration produces geometrically valid src points without requiring precise manual clicking; lane lines appear parallel in BEV after alignment.
