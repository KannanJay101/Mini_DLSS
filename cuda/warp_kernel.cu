// cuda/warp_kernel.cu
//
// Each thread warps one output pixel by looking up its source coordinate
// in the optical flow field and performing bilinear interpolation.
//
// Concepts learned: 2D grid indexing, bilinear interpolation, boundary clamping
//
// Compile: nvcc -o warp_kernel warp_kernel.cu
// Launch:
//   dim3 block(16, 16);
//   dim3 grid((width+15)/16, (height+15)/16);
//   warp_kernel<<<grid, block>>>(src, flow, out, W, H, 0.5f);

#include <cuda_runtime.h>
#include <math.h>

__global__ void warp_kernel(
    const unsigned char* src_frame,  // source image (H, W, 3), BGR row-major
    const float*         flow,       // optical flow (H, W, 2), [dx, dy] interleaved
    unsigned char*       output,     // warped result (H, W, 3)
    int width, int height,
    float t                          // interpolation factor (0.5 for midpoint warp)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Read optical flow at this pixel
    int flow_idx = (y * width + x) * 2;
    float dx = flow[flow_idx + 0];
    float dy = flow[flow_idx + 1];

    // Inverse warp: find where to sample in the source frame
    float src_x = (float)x - t * dx;
    float src_y = (float)y - t * dy;

    // Bilinear interpolation corners
    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = src_x - (float)x0;
    float fy = src_y - (float)y0;

    // Clamp to image bounds (BORDER_REPLICATE)
    x0 = max(0, min(x0, width  - 1));
    x1 = max(0, min(x1, width  - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    // Sample 4 neighbors and bilinear blend for each BGR channel
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
