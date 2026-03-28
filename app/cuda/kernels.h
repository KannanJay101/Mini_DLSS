#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Warp a single frame on GPU using bilinear interpolation.
// All pointers are host memory; the function handles H2D/D2H transfers.
void warp_frame_cuda(
    const unsigned char* h_src,    // (H*W*3) BGR row-major
    const float* h_flow,           // (H*W*2) interleaved [dx, dy]
    unsigned char* h_output,       // (H*W*3) BGR result
    int width, int height, float t);

// Occlusion-aware blend on GPU.
// Uses correct formula: mask * a + (1 - mask) * c
void blend_frames_cuda(
    const unsigned char* h_warped_a,   // (H*W*3) BGR
    const unsigned char* h_warped_c,   // (H*W*3) BGR
    const float* h_mask,               // (H*W) float [0,1]
    unsigned char* h_output,           // (H*W*3) BGR result
    int width, int height);

// Compute occlusion mask on GPU via forward-backward consistency.
void occlusion_mask_cuda(
    const float* h_flow_ac,        // (H*W*2) forward flow
    const float* h_flow_ca,        // (H*W*2) backward flow
    float* h_mask,                 // (H*W) output mask [0,1]
    int width, int height,
    float threshold);

// PSNR computation on GPU (from existing psnr_kernel.cu)
float compute_psnr_cuda(
    const unsigned char* h_predicted,
    const unsigned char* h_ground_truth,
    int height, int width);

#ifdef __cplusplus
}
#endif
