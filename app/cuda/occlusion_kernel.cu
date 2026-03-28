// GPU occlusion mask computation.
// Ports the Python compute_occlusion_mask (forward-backward consistency check):
//   1. Remap backward flow by forward flow coordinates
//   2. Compute consistency = forward + warped_backward
//   3. error = magnitude(consistency)
//   4. mask = exp(-error / threshold)

#include <cuda_runtime.h>
#include "kernels.h"

// Bilinear sample a 2-channel flow field at fractional coordinates.
__device__ void bilinearSampleFlow(
    const float* flow, int width, int height,
    float sx, float sy, float& out_dx, float& out_dy)
{
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    // Clamp to image bounds
    x0 = max(0, min(x0, width  - 1));
    x1 = max(0, min(x1, width  - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    // Sample both channels with bilinear interpolation
    for (int c = 0; c < 2; c++) {
        float v00 = flow[(y0 * width + x0) * 2 + c];
        float v01 = flow[(y0 * width + x1) * 2 + c];
        float v10 = flow[(y1 * width + x0) * 2 + c];
        float v11 = flow[(y1 * width + x1) * 2 + c];

        float val = (1.0f - fx) * (1.0f - fy) * v00
                  + fx          * (1.0f - fy) * v01
                  + (1.0f - fx) * fy          * v10
                  + fx          * fy          * v11;

        if (c == 0) out_dx = val;
        else        out_dy = val;
    }
}

__global__ void occlusion_mask_kernel(
    const float* flow_ac,   // (H, W, 2) forward flow A->C
    const float* flow_ca,   // (H, W, 2) backward flow C->A
    float*       mask,      // (H, W) output
    int width, int height,
    float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 2;

    // Forward flow at this pixel
    float fac_dx = flow_ac[idx + 0];
    float fac_dy = flow_ac[idx + 1];

    // Where does forward flow point to in frame C?
    float sx = (float)x + fac_dx;
    float sy = (float)y + fac_dy;

    // Bilinear sample backward flow at that location
    float bk_dx, bk_dy;
    bilinearSampleFlow(flow_ca, width, height, sx, sy, bk_dx, bk_dy);

    // Consistency: forward + warped_backward should be ~zero
    float cx = fac_dx + bk_dx;
    float cy = fac_dy + bk_dy;
    float error = sqrtf(cx * cx + cy * cy);

    // Soft mask
    mask[y * width + x] = expf(-error / threshold);
}

extern "C" void occlusion_mask_cuda(
    const float* h_flow_ac,
    const float* h_flow_ca,
    float* h_mask,
    int width, int height,
    float threshold)
{
    int flow_size = height * width * 2;
    int mask_size = height * width;

    float *d_flow_ac, *d_flow_ca, *d_mask;

    cudaMalloc(&d_flow_ac, flow_size * sizeof(float));
    cudaMalloc(&d_flow_ca, flow_size * sizeof(float));
    cudaMalloc(&d_mask,    mask_size * sizeof(float));

    cudaMemcpy(d_flow_ac, h_flow_ac, flow_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow_ca, h_flow_ca, flow_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    occlusion_mask_kernel<<<grid, block>>>(
        d_flow_ac, d_flow_ca, d_mask, width, height, threshold);
    cudaDeviceSynchronize();

    cudaMemcpy(h_mask, d_mask, mask_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_flow_ac);
    cudaFree(d_flow_ca);
    cudaFree(d_mask);
}
