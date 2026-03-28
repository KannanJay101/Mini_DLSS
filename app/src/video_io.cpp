#include "video_io.h"
#include <iostream>

namespace minidlss {

bool VideoReader::open(const std::string& path) {
    if (!cap_.open(path)) {
        std::cerr << "Failed to open video: " << path << std::endl;
        return false;
    }

    info_.width      = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    info_.height     = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    info_.fps        = cap_.get(cv::CAP_PROP_FPS);
    info_.frameCount = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Opened video: " << path << std::endl;
    std::cout << "  Resolution: " << info_.width << "x" << info_.height << std::endl;
    std::cout << "  FPS: " << info_.fps << std::endl;
    std::cout << "  Frames: " << info_.frameCount << std::endl;

    return true;
}

bool VideoReader::readFrame(cv::Mat& frame) {
    return cap_.read(frame);
}

void VideoReader::release() {
    cap_.release();
}

// Try to open a VideoWriter with the given fourcc. Returns true on success.
static bool tryOpenWriter(cv::VideoWriter& writer, const std::string& path,
                          int fourcc, double fps, cv::Size size) {
    writer.open(path, fourcc, fps, size);
    return writer.isOpened();
}

bool VideoWriter::open(const std::string& path, double fps, int width, int height) {
    cv::Size size(width, height);

    // Try codecs in order of preference
    struct CodecOption {
        int fourcc;
        const char* name;
    };

    CodecOption codecs[] = {
        {cv::VideoWriter::fourcc('a','v','c','1'), "H.264 (avc1)"},
        {cv::VideoWriter::fourcc('m','p','4','v'), "MPEG-4 (mp4v)"},
        {cv::VideoWriter::fourcc('X','V','I','D'), "XVID"},
        {cv::VideoWriter::fourcc('M','J','P','G'), "MJPEG"},
    };

    for (auto& codec : codecs) {
        if (tryOpenWriter(writer_, path, codec.fourcc, fps, size)) {
            std::cout << "Video writer opened with codec: " << codec.name << std::endl;
            return true;
        }
    }

    std::cerr << "Failed to open video writer with any codec" << std::endl;
    return false;
}

bool VideoWriter::writeFrame(const cv::Mat& frame) {
    if (!writer_.isOpened()) return false;
    writer_.write(frame);
    return true;
}

void VideoWriter::release() {
    writer_.release();
}

} // namespace minidlss
