#include "flow.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

namespace minidlss {

cv::Mat computeOpticalFlow(const cv::Mat& frameA, const cv::Mat& frameC) {
    cv::Mat grayA, grayC;
    cv::cvtColor(frameA, grayA, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frameC, grayC, cv::COLOR_BGR2GRAY);

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(
        grayA, grayC, flow,
        /*pyr_scale=*/0.5,
        /*levels=*/3,
        /*winsize=*/15,
        /*iterations=*/3,
        /*poly_n=*/5,
        /*poly_sigma=*/1.2,
        /*flags=*/0);

    return flow;
}

} // namespace minidlss
