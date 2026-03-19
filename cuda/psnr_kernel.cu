// cuda/psnr_kernel.cu
//
// Each thread computes squared error for one pixel element (per channel value).
// Then parallel reduction sums all errors across the block.
//
// Concepts learned: thread indexing, shared memory, parallel reduction
//
// Compile: nvcc -o psnr_kernel psnr_kernel.cu
// Usage:   Called from Python via pycuda or ctypes

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void psnr_kernel(
    const unsigned char* predicted,    // (H * W * 3) flattened, row-major BGR
    const unsigned char* ground_truth,
    float* partial_sums,               // one partial sum per block
    int total_pixels                   // H * W * 3 (all channel values)
) {
    // Shared memory for block-level reduction
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread: compute squared error for its element
    float error = 0.0f;
    if (gid < total_pixels) {
        float diff = (float)predicted[gid] - (float)ground_truth[gid];
        error = diff * diff;
    }

    sdata[tid] = error;
    __syncthreads();

    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes block result to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Host helper: launch kernel and return PSNR
// Call from C or via ctypes from Python
float compute_psnr_cuda(
    const unsigned char* h_predicted,
    const unsigned char* h_ground_truth,
    int height, int width
) {
    int total = height * width * 3;

    unsigned char *d_pred, *d_gt;
    float *d_partial;

    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    cudaMalloc(&d_pred,    total * sizeof(unsigned char));
    cudaMalloc(&d_gt,      total * sizeof(unsigned char));
    cudaMalloc(&d_partial, num_blocks * sizeof(float));

    cudaMemcpy(d_pred, h_predicted,    total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gt,   h_ground_truth, total, cudaMemcpyHostToDevice);

    psnr_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_pred, d_gt, d_partial, total
    );
    cudaGetLastError();
    cudaDeviceSynchronize();

    // Sum partial results on CPU
    float* h_partial = new float[num_blocks];
    cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    double total_error = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        total_error += h_partial[i];
    }

    cudaFree(d_pred);
    cudaFree(d_gt);
    cudaFree(d_partial);
    delete[] h_partial;

    double mse = total_error / total;
    if (mse == 0.0) return INFINITY;
    return (float)(10.0 * log10(255.0 * 255.0 / mse));
}
