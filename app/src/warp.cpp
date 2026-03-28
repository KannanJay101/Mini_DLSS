#include "warp.h"
#include <opencv2/imgproc.hpp>

namespace minidlss {

cv::Mat warpFrame(const cv::Mat& frame, const cv::Mat& flow, float t) {
    int h = frame.rows;
    int w = frame.cols;

    // Split flow into dx and dy channels
    cv::Mat channels[2];
    cv::split(flow, channels);

    // Build coordinate grids
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

    // Inverse warp: map_x = x - t * dx, map_y = y - t * dy
    cv::Mat mapX = xCoords - t * channels[0];
    cv::Mat mapY = yCoords - t * channels[1];

    cv::Mat warped;
    cv::remap(frame, warped, mapX, mapY,
              cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    return warped;
}

} // namespace minidlss
