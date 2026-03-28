#pragma once

#include <string>
#include <atomic>
#include <functional>

namespace minidlss {

struct PipelineConfig {
    std::string inputPath;
    std::string outputPath;
    bool useCuda = false;
};

struct PipelineStatus {
    int currentFrame = 0;
    int totalFrames = 0;
    float fps = 0.0f;       // processing speed (frames/sec)
    std::string message;
};

// Callback type for progress updates
using ProgressCallback = std::function<void(const PipelineStatus&)>;

class Pipeline {
public:
    // Run the full video-to-video interpolation pipeline.
    // Blocks until complete or cancelled. Call from a worker thread.
    // Returns true on success, false on error/cancellation.
    bool run(const PipelineConfig& config,
             std::atomic<bool>& cancelRequested,
             ProgressCallback progressCb = nullptr);

    std::string lastError() const { return lastError_; }

private:
    std::string lastError_;
};

} // namespace minidlss
