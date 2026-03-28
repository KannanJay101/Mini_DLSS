#pragma once

#include "pipeline.h"
#include "cuda_backend.h"
#include <string>
#include <atomic>
#include <mutex>
#include <thread>

namespace minidlss {

enum class AppState {
    Idle,
    Processing,
    Complete,
    Error
};

class App {
public:
    App();
    ~App();

    void init();

    // Called every frame from the main loop
    void update();

    // --- State queries (thread-safe) ---
    bool isProcessing() const { return state_ == AppState::Processing; }
    bool isComplete() const { return state_ == AppState::Complete || state_ == AppState::Error; }
    PipelineStatus status() const;
    std::string lastError() const;

    // --- CUDA info ---
    bool cudaAvailable() const { return cuda_.isAvailable(); }
    std::string gpuName() const { return cuda_.gpuName(); }
    bool useCuda() const { return useCuda_; }
    void setUseCuda(bool v) { useCuda_ = v; }

    // --- File paths ---
    void setInputPath(const std::string& p) { inputPath_ = p; }
    void setOutputPath(const std::string& p) { outputPath_ = p; }
    std::string inputPath() const { return inputPath_; }
    std::string outputPath() const { return outputPath_; }

    // --- Processing control ---
    void startProcessing();
    void cancelProcessing();

private:
    void workerThread();

    AppState state_ = AppState::Idle;
    CudaBackend cuda_;
    bool useCuda_ = false;

    std::string inputPath_;
    std::string outputPath_;

    // Worker thread communication
    std::thread worker_;
    std::atomic<bool> cancelRequested_{false};
    mutable std::mutex statusMutex_;
    PipelineStatus currentStatus_;
    std::string errorMsg_;
};

} // namespace minidlss
