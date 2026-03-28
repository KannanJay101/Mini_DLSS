#include "blend.h"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace minidlss {

cv::Mat computeOcclusionMask(const cv::Mat& flowAC, const cv::Mat& flowCA,
                             float threshold) {
    int h = flowAC.rows;
    int w = flowAC.cols;

    // Split forward flow into dx, dy
    cv::Mat acChannels[2];
    cv::split(flowAC, acChannels);

    // Build remap coordinates: where forward flow points to in frame C
    cv::Mat xCoords(h, w, CV_32F);
    cv::Mat yCoords(h, w, CV_32F);
    for (int y = 0; y < h; y++) {
        float* xRow = xCoords.ptr<float>(y);
        float* yRow = yCoords.ptr<float>(y);
        for (int x = 0; x < w; x++) {
            xRow[x] = static_cast<float>(x);
            yRow[x] = static_cast<float>(y);
        }
    }

    cv::Mat mapX = xCoords + acChannels[0];
    cv::Mat mapY = yCoords + acChannels[1];

    // Warp backward flow by forward flow
    cv::Mat warpedBackFlow;
    cv::remap(flowCA, warpedBackFlow, mapX, mapY,
              cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // Consistency check: forward + warped_backward should be ~zero
    cv::Mat consistency = flowAC + warpedBackFlow;

    cv::Mat consistChannels[2];
    cv::split(consistency, consistChannels);

    // Error magnitude per pixel
    cv::Mat error;
    cv::magnitude(consistChannels[0], consistChannels[1], error);

    // Soft mask: exp(-error / threshold)
    cv::Mat mask;
    cv::exp(-error / threshold, mask);

    return mask;
}

cv::Mat blendWithMask(const cv::Mat& warpedA, const cv::Mat& warpedC,
                      const cv::Mat& mask) {
    // Convert to float for blending
    cv::Mat aFloat, cFloat;
    warpedA.convertTo(aFloat, CV_32F);
    warpedC.convertTo(cFloat, CV_32F);

    // Expand mask to 3 channels
    cv::Mat mask3ch;
    cv::Mat channels[3] = {mask, mask, mask};
    cv::merge(channels, 3, mask3ch);

    // result = mask * warpedA + (1 - mask) * warpedC
    cv::Mat result = mask3ch.mul(aFloat) + (1.0f - mask3ch).mul(cFloat);

    cv::Mat output;
    result.convertTo(output, CV_8U);
    return output;
}

} // namespace minidlss
