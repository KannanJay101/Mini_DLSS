#pragma once

#include <opencv2/core.hpp>

namespace minidlss {

// Compute dense optical flow from frameA to frameC using Farneback's algorithm.
// Input: BGR uint8 images (same size).
// Output: CV_32FC2 (H, W) — channel 0 = dx, channel 1 = dy.
cv::Mat computeOpticalFlow(const cv::Mat& frameA, const cv::Mat& frameC);

} // namespace minidlss
