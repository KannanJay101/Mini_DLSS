// cuda/blend_kernel.cu
//
// Per-pixel weighted blend using an occlusion mask.
// Combines warped_a and warped_c based on visibility confidence.
//
// Concepts learned: multi-input kernels, per-pixel branching, memory layout
//
// Compile: nvcc -o blend_kernel blend_kernel.cu
// Launch:
//   dim3 block(16, 16);
//   dim3 grid((width+15)/16, (height+15)/16);
//   blend_kernel<<<grid, block>>>(warped_a, warped_c, mask, output, W, H);

#include <cuda_runtime.h>
#include <math.h>

__global__ void blend_kernel(
    const unsigned char* warped_a,       // (H, W, 3) BGR
    const unsigned char* warped_c,       // (H, W, 3) BGR
    const float*         occlusion_mask, // (H, W)    one float per pixel, range [0, 1]
    unsigned char*       output,         // (H, W, 3) BGR result
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;

    // mask ~ 1.0: pixel visible in both frames -> blend equally
    // mask ~ 0.0: pixel occluded           -> favor warped_c
    float mask = occlusion_mask[pixel_idx];

    for (int c = 0; c < 3; c++) {
        int idx = pixel_idx * 3 + c;
        float a = (float)warped_a[idx];
        float b = (float)warped_c[idx];

        float blended = mask * 0.5f * (a + b) + (1.0f - mask) * b;

        output[idx] = (unsigned char)fminf(fmaxf(blended, 0.0f), 255.0f);
    }
}
