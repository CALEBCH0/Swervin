# Changelog

## Unreleased

### Bug Fixes

### Features

### Refactors

### Performance

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
