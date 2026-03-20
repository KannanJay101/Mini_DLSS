# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

```bash
# Full pipeline on a gameplay video
python main.py

# Run a single module standalone (each src/ file has an __main__ block)
python -m src.flow
python -m src.warp
python -m src.interpolate
python -m src.evaluate
python -m src.extract
python -m src.triplets
```

Standalone module runs expect frames at `data/frames/frame_00000.png`, `frame_00001.png`, `frame_00002.png`.

## Building CUDA Kernels

```bash
cd cuda
make check   # verify nvcc + GPU available
make all     # build all three .dll/.so into cuda/build/
make clean
```

Phase 2 (CUDA) integration is not yet wired into the Python pipeline — kernels exist in `cuda/` but `main.py` still calls the pure-Python/OpenCV implementations in `src/`.

## Architecture

The pipeline is a linear sequence of stages, each in its own `src/` module:

```
extract.py → triplets.py → baseline.py  ─────────────────────┐
                         → flow.py → warp.py → interpolate.py ├─► evaluate.py
```

**Key data contract between modules:**
- All frames are passed as **file paths** (strings), not numpy arrays, at the public API boundary. Internal functions load with `cv2.imread`.
- Optical flow shape is `(H, W, 2)` float32 — index `[..., 0]` = dx, `[..., 1]` = dy.
- `interpolate.py` is the top-level caller for Phase 1 interpolation — it orchestrates `flow.py` and `warp.py` internally and returns a numpy array (not a path).

**Occlusion mask logic** (`interpolate.py:compute_occlusion_mask`): forward-backward consistency check — forward flow is followed then backward flow is applied; the residual magnitude is converted to a soft mask via `exp(-error/threshold)`. Mask near 1.0 = trust both warps equally; near 0.0 = fall back to backward warp only.

**CUDA kernels** (`cuda/`) implement the same three operations as `warp.py`, `interpolate.py` (blend step), and `evaluate.py` (PSNR). The host-callable function `compute_psnr_cuda()` in `psnr_kernel.cu` is the only kernel with a full host wrapper; `warp_kernel` and `blend_kernel` are device functions only and need a ctypes/pycuda host wrapper to call from Python.

## Data Layout

- `data/videos/gameplay.mp4` — input (not tracked in git)
- `data/frames/` — extracted PNGs named `frame_NNNNN.png` (generated)
- `data/triplets/index.json` — JSON array of `{frame_a, frame_b, frame_c}` path dicts (generated)
- `outputs/results/` — predicted frames and flow visualization (generated)
