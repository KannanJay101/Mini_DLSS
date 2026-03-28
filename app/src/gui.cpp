#if HAS_GUI

#include "gui.h"
#include "app.h"
#include "imgui.h"
#include "tinyfiledialogs.h"
#include <cstring>
#include <algorithm>

namespace minidlss {

void Gui::init(App* app) {
    app_ = app;
}

void Gui::draw() {
    // Full-window ImGui panel
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoCollapse;

    ImGui::Begin("MiniDLSS", nullptr, flags);

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f),
                       "Mini-DLSS Frame Interpolation");
    ImGui::Separator();
    ImGui::Spacing();

    drawBackendInfo();
    ImGui::Spacing();
    drawFileSelection();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    drawControls();
    ImGui::Spacing();
    drawProgress();
    ImGui::Spacing();
    drawStatus();

    ImGui::End();
}

void Gui::drawFileSelection() {
    bool processing = app_->isProcessing();

    ImGui::BeginDisabled(processing);

    // Input file
    ImGui::Text("Input Video:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
    ImGui::InputText("##input", inputPath_, sizeof(inputPath_),
                     ImGuiInputTextFlags_ReadOnly);
    ImGui::SameLine();
    if (ImGui::Button("Browse##in")) {
        const char* filters[] = {"*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv"};
        const char* result = tinyfd_openFileDialog(
            "Select Input Video", "", 5, filters, "Video Files", 0);
        if (result) {
            strncpy(inputPath_, result, sizeof(inputPath_) - 1);
            // Auto-generate output path
            std::string inp(result);
            auto dot = inp.rfind('.');
            if (dot != std::string::npos) {
                std::string out = inp.substr(0, dot) + "_2x" + inp.substr(dot);
                strncpy(outputPath_, out.c_str(), sizeof(outputPath_) - 1);
            }
            app_->setInputPath(inputPath_);
            app_->setOutputPath(outputPath_);
        }
    }

    // Output file
    ImGui::Text("Output Video:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
    ImGui::InputText("##output", outputPath_, sizeof(outputPath_),
                     ImGuiInputTextFlags_ReadOnly);
    ImGui::SameLine();
    if (ImGui::Button("Browse##out")) {
        const char* filters[] = {"*.mp4"};
        const char* result = tinyfd_saveFileDialog(
            "Save Output Video", outputPath_, 1, filters, "MP4 Video");
        if (result) {
            strncpy(outputPath_, result, sizeof(outputPath_) - 1);
            app_->setOutputPath(outputPath_);
        }
    }

    ImGui::EndDisabled();
}

void Gui::drawBackendInfo() {
    if (app_->cudaAvailable()) {
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "GPU: %s",
                           app_->gpuName().c_str());
        bool useCuda = app_->useCuda();
        if (ImGui::Checkbox("Use CUDA acceleration", &useCuda)) {
            app_->setUseCuda(useCuda);
        }
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                           "No CUDA GPU detected - CPU mode");
    }
}

void Gui::drawControls() {
    bool processing = app_->isProcessing();
    bool hasInput = strlen(inputPath_) > 0;
    bool hasOutput = strlen(outputPath_) > 0;

    ImGui::BeginDisabled(processing || !hasInput || !hasOutput);
    if (ImGui::Button("Start Processing", ImVec2(200, 40))) {
        app_->startProcessing();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();

    ImGui::BeginDisabled(!processing);
    if (ImGui::Button("Cancel", ImVec2(100, 40))) {
        app_->cancelProcessing();
    }
    ImGui::EndDisabled();
}

void Gui::drawProgress() {
    if (!app_->isProcessing() && !app_->isComplete()) return;

    auto status = app_->status();
    float progress = (status.totalFrames > 0)
        ? static_cast<float>(status.currentFrame) / status.totalFrames
        : 0.0f;

    ImGui::ProgressBar(progress, ImVec2(-1, 0));

    if (status.totalFrames > 0) {
        ImGui::Text("Frame %d / %d  |  %.1f pairs/sec",
                    status.currentFrame, status.totalFrames, status.fps);

        if (status.fps > 0.0f && status.currentFrame < status.totalFrames) {
            float eta = (status.totalFrames - status.currentFrame) / status.fps;
            if (eta < 60.0f)
                ImGui::SameLine(), ImGui::Text("  |  ETA: %.0fs", eta);
            else
                ImGui::SameLine(), ImGui::Text("  |  ETA: %.1fmin", eta / 60.0f);
        }
    }
}

void Gui::drawStatus() {
    auto status = app_->status();

    if (app_->isComplete()) {
        if (app_->lastError().empty()) {
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f),
                               "Complete! Output saved.");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                               "Error: %s", app_->lastError().c_str());
        }
    } else if (app_->isProcessing()) {
        ImGui::Text("%s", status.message.c_str());
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Ready");
    }
}

} // namespace minidlss

#endif // HAS_GUI
