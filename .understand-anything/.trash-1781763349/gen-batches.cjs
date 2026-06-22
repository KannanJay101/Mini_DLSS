#!/usr/bin/env node
const fs = require('fs');
const INTER = '/Users/kannanjayakumar/Desktop/Mini_DLSS/.understand-anything/intermediate';

// --- batch membership (path -> batchIndex), from batches.json ---
const batches = require(INTER + '/batches.json').batches;
const pathBatch = {};
for (const b of batches) for (const f of b.files) pathBatch[f.path] = b.batchIndex;
function batchOf(filePath) { return pathBatch[filePath] || 1; }

const fileId = p => 'file:' + p;
const fn = (p, n) => 'function:' + p + ':' + n;
const cls = (p, n) => 'class:' + p + ':' + n;
const cfg = p => 'config:' + p;
const doc = p => 'document:' + p;

const nodes = [];
const edges = [];
function N(o) { nodes.push(o); }
function E(source, target, type, weight) { edges.push({ source, target, type, weight }); }

// ============ FILE / CONFIG / DOCUMENT NODES ============
const files = [
  // path, type, language, summary, tags, complexity
  ['main.py','file','python','Top-level pipeline entry point. Wires together extract -> triplets -> baseline and flow interpolation, evaluates each triplet with PSNR/SSIM, and prints an averaged comparison of naive blend vs occlusion-aware flow interpolation.',['entry-point','orchestration','pipeline','python'],'moderate'],
  ['src/__init__.py','file','python','Empty package marker that makes src/ an importable Python package.',['package-init','python'],'simple'],
  ['src/extract.py','file','python','Frame extraction stage: decodes an input video with OpenCV VideoCapture and writes every frame as a zero-padded PNG.',['video-io','opencv','extraction','python'],'simple'],
  ['src/triplets.py','file','python','Triplet builder: groups sequential frames into (A,B,C) sets where B is the ground-truth midpoint, with a configurable motion step, and persists the index as JSON.',['data-prep','triplets','json','python'],'simple'],
  ['src/baseline.py','file','python','Naive baselines: simple and weighted pixel averaging of frames A and C, used as the motion-unaware reference to beat.',['baseline','blend','opencv','python'],'simple'],
  ['src/flow.py','file','python','Optical flow stage: dense Farneback flow A->C plus an HSV visualization helper for inspecting motion fields.',['optical-flow','farneback','visualization','opencv','python'],'moderate'],
  ['src/warp.py','file','python','Flow-based warping: remaps a frame along the flow field by an interpolation factor t and a simple two-frame midpoint interpolation built on it.',['warp','remap','interpolation','opencv','python'],'moderate'],
  ['src/interpolate.py','file','python','Phase-1 interpolation core: forward-backward occlusion mask plus occlusion-weighted bidirectional warp/blend to predict the middle frame.',['interpolation','occlusion','optical-flow','core','python'],'complex'],
  ['src/evaluate.py','file','python','Quality metrics implemented from scratch: PSNR from MSE and a windowed SSIM, plus a per-triplet evaluator returning both scores.',['metrics','psnr','ssim','evaluation','python'],'moderate'],

  // C++ app core (batch 1 + 3)
  ['app/src/flow.h','file','cpp','Declares the C++ Farneback optical-flow function for the native pipeline.',['header','optical-flow','cpp'],'simple'],
  ['app/src/flow.cpp','file','cpp','C++ port of optical flow: dense Farneback A->C returning a CV_32FC2 dx/dy field.',['optical-flow','farneback','opencv','cpp'],'simple'],
  ['app/src/warp.h','file','cpp','Declares the C++ flow-based frame warp function.',['header','warp','cpp'],'simple'],
  ['app/src/warp.cpp','file','cpp','C++ port of frame warping: builds coordinate grids and inverse-warps with cv::remap bilinear interpolation.',['warp','remap','opencv','cpp'],'moderate'],
  ['app/src/blend.h','file','cpp','Declares the occlusion mask and occlusion-aware blend functions for the C++ pipeline.',['header','occlusion','blend','cpp'],'simple'],
  ['app/src/blend.cpp','file','cpp','C++ port of occlusion handling: forward-backward consistency mask and mask-weighted blend of two warped frames.',['occlusion','blend','opencv','cpp'],'complex'],
  ['app/src/video_io.h','file','cpp','Declares VideoReader/VideoWriter wrappers and the VideoInfo struct over OpenCV videoio.',['header','video-io','cpp'],'simple'],
  ['app/src/video_io.cpp','file','cpp','OpenCV video I/O: reader that probes resolution/fps/frame-count and a writer with H.264->mp4v->XVID->MJPG codec fallback.',['video-io','opencv','codec','cpp'],'moderate'],
  ['app/src/pipeline.h','file','cpp','Declares the Pipeline class plus PipelineConfig/PipelineStatus and the progress-callback type.',['header','pipeline','cpp'],'simple'],
  ['app/src/pipeline.cpp','file','cpp','Native video-to-video pipeline: reads frames, interpolates each consecutive pair (CUDA or CPU path), and writes interpolated+real frames at 2x FPS with progress reporting and cancellation.',['pipeline','orchestration','interpolation','cpp'],'complex'],
  ['app/src/cuda_backend.h','file','cpp','Declares the CudaBackend class exposing GPU warp/blend/occlusion operations and device info.',['header','cuda','gpu','cpp'],'simple'],
  ['app/src/cuda_backend.cpp','file','cpp','Optional GPU backend: detects a CUDA device and marshals cv::Mat data into the extern-C kernel host wrappers; compiles to no-ops when HAS_CUDA is off.',['cuda','gpu','backend','interop','cpp'],'complex'],
  ['app/src/app.h','file','cpp','Declares the App controller and AppState enum: owns the CUDA backend, file paths, and worker-thread state.',['header','app','threading','cpp'],'simple'],
  ['app/src/app.cpp','file','cpp','Application controller: launches the pipeline on a background std::thread, relays progress via a mutex-protected status, and supports cancellation.',['app','threading','controller','cpp'],'complex'],
  ['app/src/main.cpp','file','cpp','Native executable entry point: parses args to run either a headless CLI conversion or the ImGui/GLFW desktop GUI (with a Windows WinMain variant).',['entry-point','cli','gui','glfw','cpp'],'complex'],
  ['app/src/gui.h','file','cpp','Declares the Dear ImGui Gui class (compiled only when HAS_GUI).',['header','gui','imgui','cpp'],'simple'],
  ['app/src/gui.cpp','file','cpp','Dear ImGui front-end: file pickers (tinyfiledialogs), backend/CUDA info, start/cancel controls, and a progress + status panel bound to the App controller.',['gui','imgui','tinyfiledialogs','frontend','cpp'],'complex'],
  ['app/src/systray.h','file','cpp','Declares the Windows-only SysTray helper (compiled only on _WIN32).',['header','systray','windows','cpp'],'simple'],
  ['app/src/systray.cpp','file','cpp','Windows system-tray integration: notification icon, balloon notifications, and restore-on-double-click handling.',['systray','windows','shell','cpp'],'moderate'],

  // CUDA kernels (cuda/)
  ['cuda/warp_kernel.cu','file','cuda','Standalone teaching CUDA warp kernel: each thread inverse-warps one output pixel via bilinear interpolation of the source frame along the flow field.',['cuda','kernel','warp','bilinear','gpu'],'moderate'],
  ['cuda/blend_kernel.cu','file','cuda','Standalone CUDA blend kernel: per-pixel occlusion-weighted combination of two warped frames (uses the early mask*0.5*(a+b)+(1-mask)*b formula).',['cuda','kernel','blend','occlusion','gpu'],'moderate'],
  ['cuda/psnr_kernel.cu','file','cuda','Standalone CUDA PSNR kernel with shared-memory parallel reduction over squared error, plus a full compute_psnr_cuda host wrapper.',['cuda','kernel','psnr','reduction','gpu'],'complex'],

  // CUDA host wrappers (app/cuda/)
  ['app/cuda/kernels.h','file','cuda','extern-C declarations for the GPU host wrappers (warp/blend/occlusion-mask/psnr) called from the C++ CudaBackend.',['header','cuda','interop','extern-c'],'simple'],
  ['app/cuda/warp_host.cu','file','cuda','Host wrapper for the warp kernel: inlines warp_kernel and handles device malloc, H2D/D2H copies, and launch for the C++ app.',['cuda','host-wrapper','warp','memory','gpu'],'moderate'],
  ['app/cuda/blend_host.cu','file','cuda','Host wrapper for occlusion blend using the corrected mask*a+(1-mask)*c formula, with device allocation and transfers.',['cuda','host-wrapper','blend','occlusion','gpu'],'moderate'],
  ['app/cuda/occlusion_kernel.cu','file','cuda','GPU occlusion-mask kernel: bilinearly samples backward flow, computes forward-backward consistency error, and emits exp(-error/threshold); includes its host wrapper.',['cuda','kernel','occlusion','host-wrapper','gpu'],'complex'],

  // build / config
  ['cuda/Makefile','config','makefile','Make build for the standalone CUDA kernels: compiles each .cu into a shared library (.so/.dll) via nvcc with OS detection and a GPU/nvcc check target.',['build','make','nvcc','cuda'],'moderate'],
  ['app/CMakeLists.txt','config','cmake','CMake build for the native app: finds OpenCV, optionally enables CUDA, fetches GLFW + Dear ImGui, vendors tinyfiledialogs, and assembles the MiniDLSS executable with HAS_CUDA/HAS_GUI defines and CPack packaging.',['build','cmake','dependencies','cuda','gui'],'complex'],
  ['requirements.txt','config','txt','Python dependencies: OpenCV, NumPy, scikit-image (reference SSIM), and PyCUDA (Phase-2 GPU integration).',['dependencies','python','requirements'],'simple'],

  // docs
  ['README.md','document','markdown','Project overview: explains the A/C->B interpolation concept, the seven pipeline stages, the directory layout, and quickstart usage.',['documentation','overview','readme'],'simple'],
  ['CLAUDE.md','document','markdown','Repository guide for Claude Code: how to run the pipeline and standalone modules, build the CUDA kernels, the module data contract, and data layout.',['documentation','guide','claude'],'simple'],
];
for (const [p, type, lang, summary, tags, complexity] of files) {
  const id = type === 'config' ? cfg(p) : type === 'document' ? doc(p) : fileId(p);
  N({ id, type, name: p.split('/').pop(), filePath: p, language: lang, summary, tags, complexity });
}

// ============ FUNCTION / CLASS NODES ============
// helper to register fn/class node + contains edge
function FN(p, name, summary, tags, complexity) {
  const id = fn(p, name);
  N({ id, type: 'function', name, filePath: p, summary, tags: tags || ['function'], complexity: complexity || 'simple' });
  E(fileId(p), id, 'contains', 1.0);
  return id;
}
function CLASS(p, name, summary, tags, complexity) {
  const id = cls(p, name);
  N({ id, type: 'class', name, filePath: p, summary, tags: tags || ['class'], complexity: complexity || 'moderate' });
  E(fileId(p), id, 'contains', 1.0);
  return id;
}

// Python functions
const f_run = FN('main.py','run_pipeline','Orchestrates the full mini-DLSS run: extract frames, build triplets, then per-triplet compute naive baseline and occlusion-aware flow interpolation, accumulate PSNR/SSIM, save samples, and print an averaged report.',['orchestration','pipeline'],'complex');
const f_extract = FN('src/extract.py','extract_frames','Decodes a video with cv2.VideoCapture and writes each frame as frame_NNNNN.png; returns the number of frames written.',['video-io','opencv'],'simple');
const f_triplets = FN('src/triplets.py','create_triplets','Builds (A,B,C) triplets from sorted PNG frames with a configurable step and writes the triplet index to JSON.',['data-prep','json'],'simple');
const f_naive = FN('src/baseline.py','naive_blend','Averages frames A and C 50/50 — the motion-unaware baseline.',['baseline','blend'],'simple');
const f_weighted = FN('src/baseline.py','weighted_blend','Blends A and C with an adjustable alpha weight.',['baseline','blend'],'simple');
const f_flow = FN('src/flow.py','compute_optical_flow','Computes dense Farneback optical flow A->C and returns an (H,W,2) float32 dx/dy field.',['optical-flow','farneback'],'moderate');
const f_visflow = FN('src/flow.py','visualize_flow','Encodes a flow field as an HSV image (hue=direction, value=magnitude) and saves it.',['visualization'],'simple');
const f_warp = FN('src/warp.py','warp_frame','Inverse-warps a frame along the flow field by factor t using cv2.remap with bilinear interpolation.',['warp','remap'],'moderate');
const f_interpflow = FN('src/warp.py','interpolate_with_flow','Warps A forward and C backward to the midpoint and averages them (flow interpolation without occlusion handling).',['interpolation'],'moderate');
const f_occmask = FN('src/interpolate.py','compute_occlusion_mask','Forward-backward flow consistency check: warps backward flow by forward flow, measures residual magnitude, and maps it to a soft [0,1] mask via exp(-error/threshold).',['occlusion','optical-flow'],'complex');
const f_interpocc = FN('src/interpolate.py','interpolate_with_occlusion','Full Phase-1 prediction: bidirectional Farneback flow, occlusion mask, warp A & C to midpoint, and occlusion-weighted blend.',['interpolation','occlusion','core'],'complex');
const f_psnr = FN('src/evaluate.py','compute_psnr','Computes PSNR from mean squared error using NumPy (returns inf when identical).',['metrics','psnr'],'simple');
const f_ssim = FN('src/evaluate.py','compute_ssim','From-scratch SSIM using Gaussian-windowed luminance, contrast, and structure terms.',['metrics','ssim'],'complex');
const f_evaltrip = FN('src/evaluate.py','evaluate_triplet','Loads the ground-truth frame and returns {psnr, ssim} for a predicted frame.',['evaluation'],'simple');

// C++ classes & key functions
const c_app = CLASS('app/src/app.h','App','GUI/CLI application controller: owns the CudaBackend, runs the pipeline on a worker thread, and exposes thread-safe state, status, and file-path setters.',['app','controller','threading'],'complex');
const c_pipeline = CLASS('app/src/pipeline.h','Pipeline','Runs the full video-to-video 2x interpolation; blocks until complete or cancelled and reports progress via callback.',['pipeline'],'moderate');
CLASS('app/src/pipeline.h','PipelineConfig','Config struct: input/output paths and the useCuda flag.',['config','struct'],'simple');
CLASS('app/src/pipeline.h','PipelineStatus','Progress struct: current/total frames, fps, and a status message.',['status','struct'],'simple');
const c_cuda = CLASS('app/src/cuda_backend.h','CudaBackend','Optional GPU backend: detects a CUDA device and dispatches warp/blend/occlusion-mask kernels; degrades to no-ops without CUDA.',['cuda','gpu','backend'],'complex');
const c_reader = CLASS('app/src/video_io.h','VideoReader','OpenCV-backed reader that probes video metadata and yields frames.',['video-io'],'simple');
const c_writer = CLASS('app/src/video_io.h','VideoWriter','OpenCV-backed writer with multi-codec fallback.',['video-io','codec'],'moderate');
CLASS('app/src/video_io.h','VideoInfo','Struct holding width, height, fps, and frame count.',['struct'],'simple');
const c_gui = CLASS('app/src/gui.h','Gui','Dear ImGui front-end bound to the App controller; renders file selection, backend info, controls, progress, and status.',['gui','imgui'],'complex');
const c_systray = CLASS('app/src/systray.h','SysTray','Windows notification-area icon with balloon notifications and restore-on-double-click.',['systray','windows'],'moderate');

const cf_flow = FN('app/src/flow.cpp','computeOpticalFlow','C++ dense Farneback optical flow A->C returning a CV_32FC2 field.',['optical-flow','farneback'],'simple');
const cf_warp = FN('app/src/warp.cpp','warpFrame','C++ inverse warp: builds coordinate grids and remaps the frame by t*flow with bilinear interpolation.',['warp','remap'],'moderate');
const cf_occ = FN('app/src/blend.cpp','computeOcclusionMask','C++ forward-backward consistency mask via cv::remap and exp(-error/threshold).',['occlusion'],'complex');
const cf_blend = FN('app/src/blend.cpp','blendWithMask','C++ occlusion-weighted blend: mask*warpedA + (1-mask)*warpedC.',['blend','occlusion'],'moderate');
const cf_piperun = FN('app/src/pipeline.cpp','Pipeline::run','Opens input/output video, optionally inits CUDA, then interpolates every consecutive pair, writing interpolated+real frames at 2x FPS with progress and cancellation.',['pipeline','orchestration'],'complex');
const cf_interppair = FN('app/src/pipeline.cpp','interpolatePair','Interpolates one frame pair: bidirectional flow then occlusion-aware warp/blend, choosing the CUDA or CPU path.',['interpolation'],'complex');
const cf_appstart = FN('app/src/app.cpp','App::startProcessing','Transitions to Processing state and spawns the worker thread.',['threading','controller'],'moderate');
const cf_appworker = FN('app/src/app.cpp','App::workerThread','Worker body: configures and runs the Pipeline, relaying progress and errors through the mutex-protected status.',['threading','pipeline'],'complex');
const cf_appinit = FN('app/src/app.cpp','App::init','Initializes the CUDA backend and enables CUDA when a device is present.',['init','cuda'],'simple');
const cf_runcli = FN('app/src/main.cpp','runCli','Headless conversion path: builds a PipelineConfig and runs the Pipeline with console progress.',['cli'],'moderate');
const cf_rungui = FN('app/src/main.cpp','runGui','Boots GLFW + OpenGL3 + Dear ImGui, creates the App and Gui, and runs the render loop.',['gui','glfw','imgui'],'complex');
const cf_cwarp = FN('app/src/cuda_backend.cpp','CudaBackend::warpFrame','Copies frame+flow into the warp host wrapper, producing a GPU-warped frame (no-op without CUDA).',['cuda','warp','interop'],'moderate');
const cf_cblend = FN('app/src/cuda_backend.cpp','CudaBackend::blendFrames','Dispatches the GPU blend host wrapper over two warped frames and a mask.',['cuda','blend','interop'],'moderate');
const cf_cocc = FN('app/src/cuda_backend.cpp','CudaBackend::computeOcclusionMask','Dispatches the GPU occlusion-mask host wrapper over forward/backward flow.',['cuda','occlusion','interop'],'moderate');
const cf_cinit = FN('app/src/cuda_backend.cpp','CudaBackend::init','Queries the CUDA device count/properties and records GPU name and VRAM.',['cuda','init'],'moderate');
const cf_vread = FN('app/src/video_io.cpp','VideoReader::open','Opens a video and reads its resolution, fps, and frame count.',['video-io'],'simple');
const cf_vwrite = FN('app/src/video_io.cpp','VideoWriter::open','Opens an output video, trying codecs in preference order until one succeeds.',['video-io','codec'],'moderate');

// CUDA kernels + host wrappers
const k_warp = FN('cuda/warp_kernel.cu','warp_kernel','Device kernel: one thread per output pixel inverse-warps along the flow with bilinear interpolation and boundary clamping.',['cuda','kernel','warp'],'moderate');
const k_blend = FN('cuda/blend_kernel.cu','blend_kernel','Device kernel: per-pixel occlusion blend using mask*0.5*(a+b)+(1-mask)*b.',['cuda','kernel','blend'],'moderate');
const k_psnr = FN('cuda/psnr_kernel.cu','psnr_kernel','Device kernel: per-element squared error followed by shared-memory parallel reduction to block partial sums.',['cuda','kernel','reduction'],'complex');
const k_psnrhost = FN('cuda/psnr_kernel.cu','compute_psnr_cuda','Host wrapper: allocates device memory, launches the PSNR kernel, sums partial sums, and returns PSNR in dB.',['cuda','host-wrapper','psnr'],'complex');
const k_warphost_k = FN('app/cuda/warp_host.cu','warp_kernel','Inlined warp device kernel used by the app build (mirrors cuda/warp_kernel.cu).',['cuda','kernel','warp'],'moderate');
const k_warphost = FN('app/cuda/warp_host.cu','warp_frame_cuda','extern-C host wrapper: device alloc, H2D/D2H copies, and launch of the warp kernel.',['cuda','host-wrapper','warp','interop'],'moderate');
const k_blendhost_k = FN('app/cuda/blend_host.cu','blend_kernel_v2','Corrected occlusion blend device kernel: mask*a + (1-mask)*c.',['cuda','kernel','blend'],'moderate');
const k_blendhost = FN('app/cuda/blend_host.cu','blend_frames_cuda','extern-C host wrapper for the corrected blend kernel.',['cuda','host-wrapper','blend','interop'],'moderate');
const k_occkernel = FN('app/cuda/occlusion_kernel.cu','occlusion_mask_kernel','Device kernel: per-pixel forward-backward consistency error -> exp(-error/threshold) soft mask.',['cuda','kernel','occlusion'],'complex');
const k_occsample = FN('app/cuda/occlusion_kernel.cu','bilinearSampleFlow','Device helper: bilinearly samples a 2-channel flow field at fractional coordinates.',['cuda','device-fn','bilinear'],'moderate');
const k_occhost = FN('app/cuda/occlusion_kernel.cu','occlusion_mask_cuda','extern-C host wrapper for the occlusion-mask kernel.',['cuda','host-wrapper','occlusion','interop'],'moderate');

// ============ IMPORTS EDGES (from importMap) ============
const importMap = require(INTER + '/scan-result.json').importMap;
for (const [src, targets] of Object.entries(importMap)) {
  for (const t of targets) {
    if (src === t) continue;
    E(fileId(src), fileId(t), 'imports', 0.7);
  }
}

// ============ CALLS EDGES ============
// Python orchestration
E(f_run, f_extract, 'calls', 0.8);
E(f_run, f_triplets, 'calls', 0.8);
E(f_run, f_naive, 'calls', 0.8);
E(f_run, f_interpocc, 'calls', 0.8);
E(f_run, f_evaltrip, 'calls', 0.8);
E(f_run, f_flow, 'calls', 0.8);
E(f_run, f_visflow, 'calls', 0.8);
E(f_interpflow, f_warp, 'calls', 0.8);
E(f_interpocc, f_flow, 'calls', 0.8);
E(f_interpocc, f_occmask, 'calls', 0.8);
E(f_interpocc, f_warp, 'calls', 0.8);
E(f_evaltrip, f_psnr, 'calls', 0.8);
E(f_evaltrip, f_ssim, 'calls', 0.8);

// C++ calls
E(cf_piperun, cf_interppair, 'calls', 0.8);
E(cf_interppair, cf_flow, 'calls', 0.8);
E(cf_interppair, cf_warp, 'calls', 0.8);
E(cf_interppair, cf_occ, 'calls', 0.8);
E(cf_interppair, cf_blend, 'calls', 0.8);
E(cf_interppair, cf_cwarp, 'calls', 0.8);
E(cf_interppair, cf_cblend, 'calls', 0.8);
E(cf_interppair, cf_cocc, 'calls', 0.8);
E(cf_appworker, cf_piperun, 'calls', 0.8);
E(cf_appstart, cf_appworker, 'calls', 0.8);
E(cf_runcli, cf_piperun, 'calls', 0.8);
E(cf_rungui, cf_appinit, 'calls', 0.8);
E(cf_cwarp, k_warphost, 'calls', 0.8);
E(cf_cblend, k_blendhost, 'calls', 0.8);
E(cf_cocc, k_occhost, 'calls', 0.8);
E(k_warphost, k_warphost_k, 'calls', 0.8);
E(k_blendhost, k_blendhost_k, 'calls', 0.8);
E(k_occhost, k_occkernel, 'calls', 0.8);
E(k_occkernel, k_occsample, 'calls', 0.8);
E(k_psnrhost, k_psnr, 'calls', 0.8);

// ============ CONTAINS (class -> methods) ============
E(c_pipeline, cf_piperun, 'contains', 1.0);
E(c_app, cf_appstart, 'contains', 1.0);
E(c_app, cf_appworker, 'contains', 1.0);
E(c_app, cf_appinit, 'contains', 1.0);
E(c_cuda, cf_cwarp, 'contains', 1.0);
E(c_cuda, cf_cblend, 'contains', 1.0);
E(c_cuda, cf_cocc, 'contains', 1.0);
E(c_cuda, cf_cinit, 'contains', 1.0);
E(c_reader, cf_vread, 'contains', 1.0);
E(c_writer, cf_vwrite, 'contains', 1.0);

// App owns CudaBackend
E(c_app, c_cuda, 'depends_on', 0.6);
E(cf_appworker, c_pipeline, 'depends_on', 0.6);

// ============ SEMANTIC: cross-implementation equivalence ============
// Python <-> C++ ports of the same algorithm
E(cf_flow, f_flow, 'similar_to', 0.5);
E(cf_warp, f_warp, 'similar_to', 0.5);
E(cf_occ, f_occmask, 'similar_to', 0.5);
E(cf_blend, f_interpocc, 'similar_to', 0.5);
// CPU C++ <-> CUDA kernels (same op)
E(cf_warp, k_warphost_k, 'similar_to', 0.5);
E(cf_blend, k_blendhost_k, 'similar_to', 0.5);
E(cf_occ, k_occkernel, 'similar_to', 0.5);
// Python <-> CUDA standalone kernels
E(f_warp, k_warp, 'similar_to', 0.5);
E(f_occmask, k_occkernel, 'similar_to', 0.5);
E(f_psnr, k_psnr, 'similar_to', 0.5);
// duplicated warp kernel between standalone and app build
E(k_warp, k_warphost_k, 'similar_to', 0.5);
E(k_blend, k_blendhost_k, 'similar_to', 0.5);

// ============ CONFIGURES / DEPLOYS / DOCUMENTS ============
E(cfg('app/CMakeLists.txt'), fileId('app/src/main.cpp'), 'configures', 0.6);
E(cfg('app/CMakeLists.txt'), fileId('app/cuda/warp_host.cu'), 'configures', 0.6);
E(cfg('app/CMakeLists.txt'), fileId('app/cuda/blend_host.cu'), 'configures', 0.6);
E(cfg('app/CMakeLists.txt'), fileId('app/cuda/occlusion_kernel.cu'), 'configures', 0.6);
E(cfg('cuda/Makefile'), fileId('cuda/warp_kernel.cu'), 'configures', 0.6);
E(cfg('cuda/Makefile'), fileId('cuda/blend_kernel.cu'), 'configures', 0.6);
E(cfg('cuda/Makefile'), fileId('cuda/psnr_kernel.cu'), 'configures', 0.6);
E(cfg('requirements.txt'), fileId('main.py'), 'configures', 0.6);

E(doc('README.md'), fileId('main.py'), 'documents', 0.5);
E(doc('CLAUDE.md'), fileId('main.py'), 'documents', 0.5);
E(doc('CLAUDE.md'), cfg('cuda/Makefile'), 'documents', 0.5);

// header <-> impl (implements/related)
E(fileId('app/src/flow.cpp'), fileId('app/src/flow.h'), 'imports', 0.7);

// ============ WRITE PER-BATCH OUTPUT ============
// Assign each node to the batch of its filePath; edges to batch of source node's file.
function nodeFile(n) { return n.filePath; }
const nodeById = {}; nodes.forEach(n => nodeById[n.id] = n);
const byBatch = {1:{nodes:[],edges:[]},2:{nodes:[],edges:[]},3:{nodes:[],edges:[]},4:{nodes:[],edges:[]},5:{nodes:[],edges:[]}};
for (const n of nodes) { byBatch[batchOf(nodeFile(n))].nodes.push(n); }
for (const e of edges) {
  const src = nodeById[e.source];
  const b = src ? batchOf(src.filePath) : 1;
  byBatch[b].edges.push(e);
}
for (const b of [1,2,3,4,5]) {
  fs.writeFileSync(INTER + '/batch-' + b + '.json', JSON.stringify({ nodes: byBatch[b].nodes, edges: byBatch[b].edges }, null, 2));
}
console.log('Total nodes:', nodes.length, '| Total edges:', edges.length);
for (const b of [1,2,3,4,5]) console.log('batch-' + b + ': ' + byBatch[b].nodes.length + ' nodes, ' + byBatch[b].edges.length + ' edges');
const types = {}; nodes.forEach(n=>types[n.type]=(types[n.type]||0)+1);
console.log('node types:', JSON.stringify(types));
const etypes = {}; edges.forEach(e=>etypes[e.type]=(etypes[e.type]||0)+1);
console.log('edge types:', JSON.stringify(etypes));
