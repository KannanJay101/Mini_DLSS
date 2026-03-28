#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace minidlss {

class CudaBackend {
public:
    void init();
    bool isAvailable() const { return available_; }
    std::string gpuName() const { return gpuName_; }
    int vramMB() const { return vramMB_; }

    // GPU-accelerated operations (only callable when isAvailable() == true)
    void warpFrame(const cv::Mat& src, const cv::Mat& flow, float t,
                   cv::Mat& output);
    void blendFrames(const cv::Mat& warpedA, const cv::Mat& warpedC,
                     const cv::Mat& mask, cv::Mat& output);
    void computeOcclusionMask(const cv::Mat& flowAC, const cv::Mat& flowCA,
                              float threshold, cv::Mat& mask);

private:
    bool available_ = false;
    std::string gpuName_ = "None";
    int vramMB_ = 0;
};

} // namespace minidlss
