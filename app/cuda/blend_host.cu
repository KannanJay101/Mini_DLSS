// Host wrapper for occlusion-aware blend.
// Uses the CORRECT formula: mask * a + (1 - mask) * c
// (The original blend_kernel.cu used mask*0.5*(a+b) + (1-mask)*b which is wrong.)

#include <cuda_runtime.h>
#include "kernels.h"

__global__ void blend_kernel_v2(
    const unsigned char* warped_a,
    const unsigned char* warped_c,
    const float*         occlusion_mask,
    unsigned char*       output,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x;
    float mask = occlusion_mask[pixel_idx];

    for (int c = 0; c < 3; c++) {
        int idx = pixel_idx * 3 + c;
        float a = (float)warped_a[idx];
        float b = (float)warped_c[idx];

        // mask=1.0 -> trust warped_a, mask=0.0 -> trust warped_c
        float blended = mask * a + (1.0f - mask) * b;

        output[idx] = (unsigned char)fminf(fmaxf(blended, 0.0f), 255.0f);
    }
}

extern "C" void blend_frames_cuda(
    const unsigned char* h_warped_a,
    const unsigned char* h_warped_c,
    const float* h_mask,
    unsigned char* h_output,
    int width, int height)
{
    int img_size  = height * width * 3;
    int mask_size = height * width;

    unsigned char *d_a, *d_c, *d_output;
    float *d_mask;

    cudaMalloc(&d_a,      img_size * sizeof(unsigned char));
    cudaMalloc(&d_c,      img_size * sizeof(unsigned char));
    cudaMalloc(&d_output, img_size * sizeof(unsigned char));
    cudaMalloc(&d_mask,   mask_size * sizeof(float));

    cudaMemcpy(d_a,    h_warped_a, img_size,                   cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,    h_warped_c, img_size,                   cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask,     mask_size * sizeof(float),  cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    blend_kernel_v2<<<grid, block>>>(d_a, d_c, d_mask, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_c);
    cudaFree(d_output);
    cudaFree(d_mask);
}
