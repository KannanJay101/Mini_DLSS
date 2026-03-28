#pragma once

#include <opencv2/core.hpp>

namespace minidlss {

// Warp a frame using optical flow, moving pixels by t * flow.
// For midpoint interpolation:
//   warpFrame(frameA, flowAC, +0.5)  — warp A forward to midpoint
//   warpFrame(frameC, flowCA, +0.5)  — warp C backward to midpoint
// Input: frame (H,W,3) uint8 BGR, flow (H,W) CV_32FC2.
// Output: warped frame (H,W,3) uint8 BGR.
cv::Mat warpFrame(const cv::Mat& frame, const cv::Mat& flow, float t);

} // namespace minidlss
