# Mini-DLSS: Frame Interpolation Pipeline

A from-scratch implementation of video frame interpolation inspired by NVIDIA's DLSS Frame Generation. Given two frames (A and C), the pipeline predicts the missing middle frame (B) using optical flow and occlusion-aware warping, then evaluates quality against the real ground truth.

Built in two phases: a pure Python/OpenCV pipeline first, followed by CUDA kernel acceleration.

---

## How It Works

```
Frame A ──┐
           ├──► Optical Flow ──► Warp A forward  ──┐
           │                                         ├──► Occlusion Blend ──► Predicted B
           ├──► Optical Flow ──► Warp C backward ──┘
Frame C ──┘

Predicted B  vs  Ground Truth B  ──►  PSNR / SSIM
```

1. **Extract** — decode a gameplay video into individual PNG frames
2. **Triplets** — group frames into (A, B, C) sets where B is the ground truth
3. **Baseline** — average A and C pixels (naive blend, no motion awareness)
4. **Optical Flow** — compute dense Farneback flow A→C and C→A
5. **Warp** — move each pixel halfway along its flow vector
6. **Occlusion Mask** — detect inconsistent regions via forward-backward flow check; weight warps accordingly
7. **Evaluate** — measure PSNR and SSIM of predicted vs ground truth, report improvement over baseline

---

## Project Structure

```
Mini_DLSS/
├── main.py                  # Pipeline entry point
├── requirements.txt
├── src/
│   ├── extract.py           # Video → PNG frames (OpenCV)
│   ├── triplets.py          # Build (A, B, C) triplet index
│   ├── baseline.py          # Naive pixel average baseline
│   ├── flow.py              # Farneback optical flow + HSV visualization
│   ├── warp.py              # Flow-based frame warping
│   ├── interpolate.py       # Occlusion-aware bidirectional blend
│   └── evaluate.py          # PSNR and SSIM metrics (implemented from scratch)
├── cuda/
│   ├── warp_kernel.cu       # GPU warp: bilinear interp per pixel
│   ├── blend_kernel.cu      # GPU blend: occlusion-masked weighted sum
│   ├── psnr_kernel.cu       # GPU PSNR: parallel reduction over squared error
│   └── Makefile             # Builds .dll/.so via nvcc (Windows + Linux)
├── data/
│   ├── videos/              # Input gameplay video goes here
│   ├── frames/              # Extracted PNG frames (generated)
│   └── triplets/            # index.json triplet manifest (generated)
└── outputs/
    └── results/             # Predicted frames + flow visualization (generated)
```

---

## Quickstart

### Requirements

- Python 3.8+
- OpenCV, NumPy (`pip install -r requirements.txt`)
- CUDA toolkit + `nvcc` (Phase 2 only)

### Install

```bash
pip install -r requirements.txt
```

### Run

Place a gameplay video at `data/videos/gameplay.mp4`, then:

```bash
python main.py
```

The pipeline will print per-triplet scores and a final averaged report:

```
============================================================
  FINAL RESULTS (averaged over 10 triplets)
============================================================
  Naive Blend:        PSNR=28.41 dB  SSIM=0.8712
  Flow Interpolation: PSNR=31.87 dB  SSIM=0.9204
  Improvement:        PSNR=+3.46 dB  SSIM=+0.0492
============================================================
```

Output images are saved to `outputs/results/`:
- `triplet_N_naive.png` — naive blend result
- `triplet_N_flow.png` — flow interpolation result
- `triplet_N_gt.png` — ground truth middle frame
- `flow_vis.png` — optical flow field (hue = direction, brightness = magnitude)

---

## CUDA Kernels (Phase 2)

The `cuda/` directory contains GPU-accelerated versions of the three core operations, callable from Python via `ctypes` or `pycuda`.

| Kernel | What it does |
|---|---|
| `warp_kernel.cu` | Per-pixel inverse warp with bilinear interpolation |
| `blend_kernel.cu` | Occlusion-masked weighted blend of two warped frames |
| `psnr_kernel.cu` | Parallel reduction over squared pixel error to compute PSNR |

**Build:**

```bash
cd cuda
make all        # builds all three shared libraries into cuda/build/
make check      # verify nvcc and GPU are available
make clean      # remove build artifacts
```

The Makefile auto-detects Windows (`.dll`) vs Linux/macOS (`.so`).

---

## Metrics

Both metrics are implemented from scratch (no scikit-image dependency in the core path):

- **PSNR** (Peak Signal-to-Noise Ratio) — measures raw pixel accuracy in dB. Higher is better; typical range 25–45 dB.
- **SSIM** (Structural Similarity Index) — measures perceived visual quality (luminance, contrast, structure). Range 0–1; >0.95 is very good.

---

## Occlusion Handling

The forward-backward consistency check detects pixels that are visible in one frame but hidden in the other (e.g. an object moving out of frame). For each pixel:

```
consistency_error = |flow_AC + warp(flow_CA, flow_AC)|
mask = exp(-error / threshold)
```

A mask near 1.0 means the pixel is reliably visible in both warped frames; near 0.0 means it's occluded and the pipeline falls back to the backward-warped frame only.
