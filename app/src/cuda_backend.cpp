#include "cuda_backend.h"
#include <iostream>

#if HAS_CUDA
#include <cuda_runtime.h>
#include "kernels.h"
#endif

namespace minidlss {

void CudaBackend::init() {
#if HAS_CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        available_ = false;
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    gpuName_ = prop.name;
    vramMB_ = static_cast<int>(prop.totalGlobalMem / (1024 * 1024));
    available_ = true;

    std::cout << "CUDA GPU: " << gpuName_ << " (" << vramMB_ << " MB)" << std::endl;
#else
    available_ = false;
#endif
}

void CudaBackend::warpFrame(const cv::Mat& src, const cv::Mat& flow, float t,
                            cv::Mat& output) {
#if HAS_CUDA
    output.create(src.rows, src.cols, CV_8UC3);

    // Ensure contiguous memory
    cv::Mat srcCont = src.isContinuous() ? src : src.clone();
    cv::Mat flowCont = flow.isContinuous() ? flow : flow.clone();

    warp_frame_cuda(
        srcCont.data,
        reinterpret_cast<const float*>(flowCont.data),
        output.data,
        src.cols, src.rows, t);
#else
    (void)src; (void)flow; (void)t; (void)output;
#endif
}

void CudaBackend::blendFrames(const cv::Mat& warpedA, const cv::Mat& warpedC,
                              const cv::Mat& mask, cv::Mat& output) {
#if HAS_CUDA
    output.create(warpedA.rows, warpedA.cols, CV_8UC3);

    cv::Mat aCont = warpedA.isContinuous() ? warpedA : warpedA.clone();
    cv::Mat cCont = warpedC.isContinuous() ? warpedC : warpedC.clone();
    cv::Mat mCont = mask.isContinuous() ? mask : mask.clone();

    blend_frames_cuda(
        aCont.data,
        cCont.data,
        reinterpret_cast<const float*>(mCont.data),
        output.data,
        warpedA.cols, warpedA.rows);
#else
    (void)warpedA; (void)warpedC; (void)mask; (void)output;
#endif
}

void CudaBackend::computeOcclusionMask(const cv::Mat& flowAC, const cv::Mat& flowCA,
                                       float threshold, cv::Mat& mask) {
#if HAS_CUDA
    mask.create(flowAC.rows, flowAC.cols, CV_32F);

    cv::Mat acCont = flowAC.isContinuous() ? flowAC : flowAC.clone();
    cv::Mat caCont = flowCA.isContinuous() ? flowCA : flowCA.clone();

    occlusion_mask_cuda(
        reinterpret_cast<const float*>(acCont.data),
        reinterpret_cast<const float*>(caCont.data),
        reinterpret_cast<float*>(mask.data),
        flowAC.cols, flowAC.rows, threshold);
#else
    (void)flowAC; (void)flowCA; (void)threshold; (void)mask;
#endif
}

} // namespace minidlss
