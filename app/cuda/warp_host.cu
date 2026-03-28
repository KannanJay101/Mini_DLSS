// Host wrapper for warp_kernel — makes it callable from C++ via extern "C".
// Follows the same pattern as compute_psnr_cuda in psnr_kernel.cu.

#include <cuda_runtime.h>
#include "kernels.h"

// ---- Inline the kernel (same as cuda/warp_kernel.cu) ----
__global__ void warp_kernel(
    const unsigned char* src_frame,
    const float*         flow,
    unsigned char*       output,
    int width, int height,
    float t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int flow_idx = (y * width + x) * 2;
    float dx = flow[flow_idx + 0];
    float dy = flow[flow_idx + 1];

    float src_x = (float)x - t * dx;
    float src_y = (float)y - t * dy;

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = src_x - (float)x0;
    float fy = src_y - (float)y0;

    x0 = max(0, min(x0, width  - 1));
    x1 = max(0, min(x1, width  - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    int out_idx = (y * width + x) * 3;

    for (int c = 0; c < 3; c++) {
        float v00 = (float)src_frame[(y0 * width + x0) * 3 + c];
        float v01 = (float)src_frame[(y0 * width + x1) * 3 + c];
        float v10 = (float)src_frame[(y1 * width + x0) * 3 + c];
        float v11 = (float)src_frame[(y1 * width + x1) * 3 + c];

        float val = (1.0f - fx) * (1.0f - fy) * v00
                  + fx          * (1.0f - fy) * v01
                  + (1.0f - fx) * fy          * v10
                  + fx          * fy          * v11;

        output[out_idx + c] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
    }
}

// ---- Host wrapper ----
extern "C" void warp_frame_cuda(
    const unsigned char* h_src,
    const float* h_flow,
    unsigned char* h_output,
    int width, int height, float t)
{
    int img_size  = height * width * 3;
    int flow_size = height * width * 2;

    unsigned char *d_src, *d_output;
    float *d_flow;

    cudaMalloc(&d_src,    img_size * sizeof(unsigned char));
    cudaMalloc(&d_output, img_size * sizeof(unsigned char));
    cudaMalloc(&d_flow,   flow_size * sizeof(float));

    cudaMemcpy(d_src,  h_src,  img_size,                    cudaMemcpyHostToDevice);
    cudaMemcpy(d_flow, h_flow, flow_size * sizeof(float),   cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    warp_kernel<<<grid, block>>>(d_src, d_flow, d_output, width, height, t);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_output);
    cudaFree(d_flow);
}
