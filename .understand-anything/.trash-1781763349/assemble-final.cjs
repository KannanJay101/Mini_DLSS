#!/usr/bin/env node
const fs = require('fs');
const INTER = '/Users/kannanjayakumar/Desktop/Mini_DLSS/.understand-anything/intermediate';
const ROOT = '/Users/kannanjayakumar/Desktop/Mini_DLSS';
const g = require(INTER + '/assembled-graph.json');

const fileId = p => 'file:' + p;
const cfg = p => 'config:' + p;
const doc = p => 'document:' + p;

const layers = [
  {
    id: 'layer:python-reference-pipeline',
    name: 'Python Reference Pipeline (Phase 1)',
    description: 'The from-scratch Python/OpenCV implementation: extract frames, build triplets, baseline blend, optical flow, occlusion-aware warp/blend, and PSNR/SSIM evaluation. This is the readable reference the native app and CUDA kernels port.',
    nodeIds: ['main.py','src/__init__.py','src/extract.py','src/triplets.py','src/baseline.py','src/flow.py','src/warp.py','src/interpolate.py','src/evaluate.py'].map(fileId),
  },
  {
    id: 'layer:native-pipeline-core',
    name: 'Native Pipeline Core (C++)',
    description: 'The C++ port of the interpolation algorithms and video I/O — optical flow, warping, occlusion mask/blend, the video reader/writer, the pipeline orchestrator, and the CUDA backend bridge that marshals data to GPU kernels.',
    nodeIds: ['app/src/flow.h','app/src/flow.cpp','app/src/warp.h','app/src/warp.cpp','app/src/blend.h','app/src/blend.cpp','app/src/video_io.h','app/src/video_io.cpp','app/src/pipeline.h','app/src/pipeline.cpp','app/src/cuda_backend.h','app/src/cuda_backend.cpp'].map(fileId),
  },
  {
    id: 'layer:application-shell-ui',
    name: 'Application Shell & UI',
    description: 'The desktop application around the pipeline: CLI/GUI entry point, the threaded App controller, the Dear ImGui front-end, and the Windows system-tray integration.',
    nodeIds: ['app/src/app.h','app/src/app.cpp','app/src/main.cpp','app/src/gui.h','app/src/gui.cpp','app/src/systray.h','app/src/systray.cpp'].map(fileId),
  },
  {
    id: 'layer:cuda-acceleration',
    name: 'CUDA Acceleration (Phase 2)',
    description: 'GPU kernels and their host wrappers: standalone teaching kernels (warp/blend/PSNR) and the app-integrated host wrappers (warp/blend/occlusion mask) callable from the C++ CudaBackend via extern-C.',
    nodeIds: ['cuda/warp_kernel.cu','cuda/blend_kernel.cu','cuda/psnr_kernel.cu','app/cuda/kernels.h','app/cuda/warp_host.cu','app/cuda/blend_host.cu','app/cuda/occlusion_kernel.cu'].map(fileId),
  },
  {
    id: 'layer:build-configuration',
    name: 'Build & Configuration',
    description: 'Build and dependency definitions: the CMake build for the native app (OpenCV/CUDA/GLFW/ImGui), the Makefile for the standalone CUDA kernels, and the Python requirements.',
    nodeIds: [cfg('app/CMakeLists.txt'), cfg('cuda/Makefile'), cfg('requirements.txt')],
  },
  {
    id: 'layer:documentation',
    name: 'Documentation',
    description: 'Project documentation: the README overview and the CLAUDE.md repository guide.',
    nodeIds: [doc('README.md'), doc('CLAUDE.md')],
  },
];

const tour = [
  { order: 1, title: 'Project Overview', description: 'Start with the README to understand the goal: predict the missing middle frame B from frames A and C using optical flow and occlusion-aware warping, then measure quality against ground truth.', nodeIds: [doc('README.md')] },
  { order: 2, title: 'Pipeline Entry Point', description: 'main.py orchestrates the whole Phase-1 run: extract -> triplets -> per-triplet baseline vs flow interpolation -> PSNR/SSIM report. Read it to see how the stages connect.', nodeIds: [fileId('main.py')] },
  { order: 3, title: 'Preparing Data: Frames & Triplets', description: 'extract.py decodes the video into PNGs; triplets.py groups them into (A,B,C) sets where B is the ground-truth midpoint.', nodeIds: [fileId('src/extract.py'), fileId('src/triplets.py')] },
  { order: 4, title: 'The Baseline to Beat', description: 'baseline.py just averages A and C. It has no motion awareness, so moving objects ghost — this is the reference the flow method must outperform.', nodeIds: [fileId('src/baseline.py')] },
  { order: 5, title: 'Estimating Motion: Optical Flow', description: 'flow.py computes dense Farneback flow between frames and can visualize it as an HSV motion field.', nodeIds: [fileId('src/flow.py')] },
  { order: 6, title: 'Moving Pixels: Warping', description: 'warp.py remaps a frame along the flow to the midpoint. interpolate_with_flow warps A forward and C backward and averages them.', nodeIds: [fileId('src/warp.py')] },
  { order: 7, title: 'Handling Occlusions (The Core Idea)', description: 'interpolate.py adds the key trick: a forward-backward consistency mask down-weights pixels where the two flows disagree, then blends the two warps accordingly.', nodeIds: [fileId('src/interpolate.py')] },
  { order: 8, title: 'Measuring Quality', description: 'evaluate.py implements PSNR and SSIM from scratch to score predictions against ground truth.', nodeIds: [fileId('src/evaluate.py')] },
  { order: 9, title: 'Porting to a Native App', description: 'pipeline.cpp is the C++ video-to-video pipeline that interpolates every consecutive pair and writes a 2x-FPS output; main.cpp chooses between a headless CLI and the GUI.', nodeIds: [fileId('app/src/pipeline.cpp'), fileId('app/src/main.cpp')] },
  { order: 10, title: 'The C++ Algorithm Ports', description: 'flow.cpp, warp.cpp, and blend.cpp mirror the Python stages using OpenCV in C++ — the CPU path of the native pipeline.', nodeIds: [fileId('app/src/flow.cpp'), fileId('app/src/warp.cpp'), fileId('app/src/blend.cpp')] },
  { order: 11, title: 'GPU Acceleration with CUDA', description: 'The CUDA kernels reimplement warp, blend, occlusion mask, and PSNR on the GPU. cuda_backend.cpp bridges OpenCV Mats to the extern-C host wrappers in app/cuda/.', nodeIds: [fileId('app/src/cuda_backend.cpp'), fileId('app/cuda/occlusion_kernel.cu'), fileId('cuda/warp_kernel.cu')] },
  { order: 12, title: 'The Desktop GUI', description: 'app.cpp runs the pipeline on a worker thread with thread-safe status; gui.cpp renders the Dear ImGui front-end with file pickers, controls, and a progress bar.', nodeIds: [fileId('app/src/app.cpp'), fileId('app/src/gui.cpp')] },
  { order: 13, title: 'Building It All', description: 'CMakeLists.txt assembles the native app (fetching GLFW/ImGui, optional CUDA); the Makefile builds the standalone CUDA kernels into shared libraries.', nodeIds: [cfg('app/CMakeLists.txt'), cfg('cuda/Makefile')] },
];

// Build final graph
const out = {
  version: '1.0.0',
  project: {
    name: 'Mini-DLSS',
    languages: ['Python','C++','CUDA','C','Markdown','Makefile'],
    frameworks: ['OpenCV','NumPy','scikit-image','PyCUDA','CUDA Toolkit','Dear ImGui','GLFW'],
    description: 'A from-scratch video frame interpolation pipeline inspired by NVIDIA DLSS Frame Generation. Predicts the missing middle frame from two frames using Farneback optical flow and occlusion-aware bidirectional warping, evaluated with PSNR/SSIM. Phase 1 is a pure Python/OpenCV pipeline; Phase 2 adds CUDA kernels and a C++ desktop GUI application.',
    analyzedAt: new Date().toISOString(),
    gitCommitHash: '0ddc7a304e295378fbad9947879d86fa93918329',
  },
  nodes: g.nodes,
  edges: g.edges,
  layers,
  tour,
};
fs.writeFileSync(INTER + '/assembled-graph.json', JSON.stringify(out, null, 2));
console.log('Final graph assembled: ' + out.nodes.length + ' nodes, ' + out.edges.length + ' edges, ' + layers.length + ' layers, ' + tour.length + ' tour steps');
fs.writeFileSync(INTER + '/layers.json', JSON.stringify(layers, null, 2));
fs.writeFileSync(INTER + '/tour.json', JSON.stringify(tour, null, 2));
