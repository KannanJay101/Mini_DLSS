#include "app.h"
#include <iostream>

namespace minidlss {

App::App() = default;

App::~App() {
    if (worker_.joinable()) {
        cancelRequested_.store(true);
        worker_.join();
    }
}

void App::init() {
    cuda_.init();
    if (cuda_.isAvailable()) {
        useCuda_ = true;
    }
}

void App::update() {
    // Nothing to poll — status is updated via callback from worker thread.
    // GUI reads status via status() which is mutex-protected.
}

PipelineStatus App::status() const {
    std::lock_guard<std::mutex> lock(statusMutex_);
    return currentStatus_;
}

std::string App::lastError() const {
    std::lock_guard<std::mutex> lock(statusMutex_);
    return errorMsg_;
}

void App::startProcessing() {
    if (state_ == AppState::Processing) return;

    state_ = AppState::Processing;
    cancelRequested_.store(false);
    {
        std::lock_guard<std::mutex> lock(statusMutex_);
        currentStatus_ = PipelineStatus{};
        errorMsg_.clear();
    }

    // Join previous worker if any
    if (worker_.joinable()) {
        worker_.join();
    }

    worker_ = std::thread(&App::workerThread, this);
}

void App::cancelProcessing() {
    cancelRequested_.store(true);
}

void App::workerThread() {
    Pipeline pipeline;
    PipelineConfig config;
    config.inputPath = inputPath_;
    config.outputPath = outputPath_;
    config.useCuda = useCuda_ && cuda_.isAvailable();

    auto progressCb = [this](const PipelineStatus& s) {
        std::lock_guard<std::mutex> lock(statusMutex_);
        currentStatus_ = s;
    };

    bool ok = pipeline.run(config, cancelRequested_, progressCb);

    {
        std::lock_guard<std::mutex> lock(statusMutex_);
        if (!ok) {
            errorMsg_ = pipeline.lastError();
        }
    }

    state_ = ok ? AppState::Complete : AppState::Error;
}

} // namespace minidlss
