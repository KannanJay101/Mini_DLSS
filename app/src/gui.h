#pragma once

#if HAS_GUI

#include <string>

namespace minidlss {

class App; // forward declaration

class Gui {
public:
    void init(App* app);
    void draw();

private:
    App* app_ = nullptr;
    char inputPath_[512] = "";
    char outputPath_[512] = "";

    void drawFileSelection();
    void drawBackendInfo();
    void drawControls();
    void drawProgress();
    void drawStatus();
};

} // namespace minidlss

#endif // HAS_GUI
