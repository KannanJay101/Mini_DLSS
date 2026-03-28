#include "pipeline.h"
#include "flow.h"
#include "warp.h"
#include "blend.h"
#include "video_io.h"
#include "cuda_backend.h"
#include <iostream>
#include <chrono>

namespace minidlss {

// Interpolate a single pair of consecutive frames to produce the midpoint.
static cv::Mat interpolatePair(const cv::Mat& frameA, const cv::Mat& frameC,
                               CudaBackend* cuda) {
    // Compute bidirectional optical flow (always on CPU — Farneback)
    cv::Mat flowAC = computeOpticalFlow(frameA, frameC);
    cv::Mat flowCA = computeOpticalFlow(frameC, frameA);

    cv::Mat mask, warpedA, warpedC, result;

    if (cuda && cuda->isAvailable()) {
        cuda->warpFrame(frameA, flowAC, 0.5f, warpedA);
        cuda->warpFrame(frameC, flowCA, 0.5f, warpedC);
        cuda->computeOcclusionMask(flowAC, flowCA, 1.0f, mask);
        cuda->blendFrames(warpedA, warpedC, mask, result);
    } else {
        // CPU path
        warpedA = warpFrame(frameA, flowAC, 0.5f);
        warpedC = warpFrame(frameC, flowCA, 0.5f);
        mask = computeOcclusionMask(flowAC, flowCA, 1.0f);
        result = blendWithMask(warpedA, warpedC, mask);
    }

    return result;
}

bool Pipeline::run(const PipelineConfig& config,
                   std::atomic<bool>& cancelRequested,
                   ProgressCallback progressCb) {
    lastError_.clear();

    // Open input video
    VideoReader reader;
    if (!reader.open(config.inputPath)) {
        lastError_ = "Failed to open input video: " + config.inputPath;
        return false;
    }

    VideoInfo vi = reader.info();
    if (vi.frameCount < 2) {
        lastError_ = "Video must have at least 2 frames";
        reader.release();
        return false;
    }

    // Open output video at 2x FPS
    double outputFps = vi.fps * 2.0;
    VideoWriter writer;
    if (!writer.open(config.outputPath, outputFps, vi.width, vi.height)) {
        lastError_ = "Failed to open output video: " + config.outputPath;
        reader.release();
        return false;
    }

    // Initialize CUDA backend if requested
    CudaBackend* cuda = nullptr;
    CudaBackend cudaBackend;
    if (config.useCuda) {
        cudaBackend.init();
        if (cudaBackend.isAvailable()) {
            cuda = &cudaBackend;
            std::cout << "Using CUDA backend: " << cudaBackend.gpuName() << std::endl;
        } else {
            std::cout << "CUDA not available, falling back to CPU" << std::endl;
        }
    }

    // Read first frame
    cv::Mat prevFrame;
    if (!reader.readFrame(prevFrame)) {
        lastError_ = "Failed to read first frame";
        reader.release();
        writer.release();
        return false;
    }

    // Write first frame
    writer.writeFrame(prevFrame);

    int totalPairs = vi.frameCount - 1;
    auto startTime = std::chrono::steady_clock::now();

    // Process consecutive pairs
    for (int i = 0; i < totalPairs; i++) {
        if (cancelRequested.load()) {
            lastError_ = "Cancelled by user";
            reader.release();
            writer.release();
            return false;
        }

        cv::Mat currFrame;
        if (!reader.readFrame(currFrame)) {
            // Some videos report more frames than they have
            break;
        }

        // Generate interpolated midpoint
        cv::Mat interp = interpolatePair(prevFrame, currFrame, cuda);

        // Write: interpolated frame, then the real frame
        writer.writeFrame(interp);
        writer.writeFrame(currFrame);

        prevFrame = currFrame;

        // Report progress
        if (progressCb) {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            float fps = (elapsed > 0.0f) ? (i + 1) / elapsed : 0.0f;

            PipelineStatus status;
            status.currentFrame = i + 1;
            status.totalFrames = totalPairs;
            status.fps = fps;
            status.message = "Processing pair " + std::to_string(i + 1) +
                             "/" + std::to_string(totalPairs);
            progressCb(status);
        }
    }

    reader.release();
    writer.release();

    auto endTime = std::chrono::steady_clock::now();
    float totalSec = std::chrono::duration<float>(endTime - startTime).count();
    std::cout << "Done! Processed " << totalPairs << " pairs in "
              << totalSec << "s (" << totalPairs / totalSec << " pairs/sec)"
              << std::endl;

    return true;
}

} // namespace minidlss
