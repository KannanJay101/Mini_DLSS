#include "app.h"
#include "pipeline.h"
#include <iostream>
#include <string>
#include <atomic>

#if HAS_GUI
#include "gui.h"
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#endif

#if HAS_GUI

static void glfwErrorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

static int runGui(int argc, char* argv[]) {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }

    // OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(800, 500,
        "Mini-DLSS Frame Interpolation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // Setup Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    // Style tweaks for a cleaner look
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 6);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Create app and GUI
    minidlss::App app;
    app.init();

    minidlss::Gui gui;
    gui.init(&app);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        app.update();
        gui.draw();

        ImGui::Render();
        int displayW, displayH;
        glfwGetFramebufferSize(window, &displayW, &displayH);
        glViewport(0, 0, displayW, displayH);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

#endif // HAS_GUI

static int runCli(const std::string& inputPath, const std::string& outputPath,
                  bool useCuda) {
    std::cout << "Mini-DLSS Frame Interpolation (CLI Mode)" << std::endl;
    std::cout << "Input:  " << inputPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;

    minidlss::PipelineConfig config;
    config.inputPath = inputPath;
    config.outputPath = outputPath;
    config.useCuda = useCuda;

    std::atomic<bool> cancel{false};
    minidlss::Pipeline pipeline;

    auto progressCb = [](const minidlss::PipelineStatus& s) {
        std::cout << "\r" << s.message
                  << " (" << static_cast<int>(100.0f * s.currentFrame / s.totalFrames)
                  << "%, " << s.fps << " pairs/sec)" << std::flush;
    };

    bool ok = pipeline.run(config, cancel, progressCb);
    std::cout << std::endl;

    if (!ok) {
        std::cerr << "Error: " << pipeline.lastError() << std::endl;
        return 1;
    }

    std::cout << "Output saved to: " << outputPath << std::endl;
    return 0;
}

#if defined(_WIN32) && HAS_GUI
#include <windows.h>
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR lpCmdLine, int) {
    // If command-line args provided, run in CLI mode
    int argc;
    LPWSTR* argvW = CommandLineToArgvW(GetCommandLineW(), &argc);

    if (argc >= 3) {
        // Convert wide strings to narrow
        char inputPath[512], outputPath[512];
        wcstombs(inputPath, argvW[1], sizeof(inputPath));
        wcstombs(outputPath, argvW[2], sizeof(outputPath));
        bool useCuda = (argc >= 4);
        LocalFree(argvW);
        return runCli(inputPath, outputPath, useCuda);
    }
    LocalFree(argvW);
    return runGui(argc, nullptr);
}
#else
int main(int argc, char* argv[]) {
    if (argc >= 3) {
        bool useCuda = false;
        for (int i = 3; i < argc; i++) {
            if (std::string(argv[i]) == "--cuda") useCuda = true;
        }
        return runCli(argv[1], argv[2], useCuda);
    }

#if HAS_GUI
    return runGui(argc, argv);
#else
    std::cerr << "Usage: MiniDLSS <input.mp4> <output.mp4> [--cuda]" << std::endl;
    return 1;
#endif
}
#endif
