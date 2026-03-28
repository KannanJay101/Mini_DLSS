#pragma once

#include <opencv2/core.hpp>

namespace minidlss {

// Forward-backward flow consistency check to detect occluded regions.
// Input: flowAC (H,W) CV_32FC2 (A→C), flowCA (H,W) CV_32FC2 (C→A).
// Output: mask (H,W) CV_32F, range [0,1].
//   1.0 = visible in both frames (trust both warps equally)
//   0.0 = occluded (fall back to warped_c only)
cv::Mat computeOcclusionMask(const cv::Mat& flowAC, const cv::Mat& flowCA,
                             float threshold = 1.0f);

// Occlusion-aware blend of two warped frames.
// result = mask * warpedA + (1 - mask) * warpedC
// Input: warpedA, warpedC (H,W,3) uint8 BGR; mask (H,W) CV_32F [0,1].
// Output: blended frame (H,W,3) uint8 BGR.
cv::Mat blendWithMask(const cv::Mat& warpedA, const cv::Mat& warpedC,
                      const cv::Mat& mask);

} // namespace minidlss
