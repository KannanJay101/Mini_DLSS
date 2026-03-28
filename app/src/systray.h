#pragma once

#ifdef _WIN32

#include <windows.h>
#include <string>

namespace minidlss {

class SysTray {
public:
    void init(HWND hwnd);
    void show();
    void hide();
    void showBalloon(const wchar_t* title, const wchar_t* message);
    void destroy();
    bool handleMessage(UINT msg, WPARAM wParam, LPARAM lParam);

private:
    NOTIFYICONDATAW nid_ = {};
    HWND hwnd_ = nullptr;
    bool visible_ = false;
};

} // namespace minidlss

#endif // _WIN32
