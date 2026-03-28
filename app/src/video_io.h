#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>

namespace minidlss {

struct VideoInfo {
    int width = 0;
    int height = 0;
    double fps = 0.0;
    int frameCount = 0;
};

class VideoReader {
public:
    bool open(const std::string& path);
    bool readFrame(cv::Mat& frame);
    VideoInfo info() const { return info_; }
    void release();

private:
    cv::VideoCapture cap_;
    VideoInfo info_;
};

class VideoWriter {
public:
    // Opens writer with codec fallback: H.264 -> mp4v -> MJPG
    bool open(const std::string& path, double fps, int width, int height);
    bool writeFrame(const cv::Mat& frame);
    void release();

private:
    cv::VideoWriter writer_;
};

} // namespace minidlss
