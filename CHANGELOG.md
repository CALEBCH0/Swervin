# Changelog

## Unreleased

### Bug Fixes

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

#### Drag-handle trapezoid calibration UI for BEV
Date: 2026-05-03
**Type:** UX
**Context:** The previous `get_src` required manually clicking 4 exact pixel positions for TL/TR/BR/BL. It was hard to satisfy the geometric constraint (equal-depth near pair, equal-depth far pair) by eye, producing skewed BEV output where lane lines weren't parallel.
**Change:** Replaced click-4-points with a drag-handle trapezoid overlay. A symmetric green trapezoid is shown on the calibration image at startup. Dragging a corner adjusts its edge's y-position and half-width symmetrically; dragging the body translates the whole shape. Press Enter to confirm, Q to cancel.
**Behavior:** Window stays open until Enter/Q. Cancellation returns an empty array; caller exits cleanly. The shape enforces horizontal top/bottom edges by construction, guaranteeing a valid ground-plane rectangle.
**Files changed:**
- `src/sas/utils/bev_transformer.py`
**Impact:** Calibration produces geometrically valid src points without requiring precise manual clicking; lane lines appear parallel in BEV after alignment.
