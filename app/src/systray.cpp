#ifdef _WIN32

#include "systray.h"
#include <shellapi.h>

#define WM_TRAYICON (WM_APP + 1)

namespace minidlss {

void SysTray::init(HWND hwnd) {
    hwnd_ = hwnd;

    memset(&nid_, 0, sizeof(nid_));
    nid_.cbSize = sizeof(NOTIFYICONDATAW);
    nid_.hWnd = hwnd;
    nid_.uID = 1;
    nid_.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    nid_.uCallbackMessage = WM_TRAYICON;
    nid_.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wcscpy_s(nid_.szTip, L"Mini-DLSS Frame Interpolation");
}

void SysTray::show() {
    if (!visible_) {
        Shell_NotifyIconW(NIM_ADD, &nid_);
        visible_ = true;
    }
}

void SysTray::hide() {
    if (visible_) {
        Shell_NotifyIconW(NIM_DELETE, &nid_);
        visible_ = false;
    }
}

void SysTray::showBalloon(const wchar_t* title, const wchar_t* message) {
    nid_.uFlags |= NIF_INFO;
    wcscpy_s(nid_.szInfoTitle, title);
    wcscpy_s(nid_.szInfo, message);
    nid_.dwInfoFlags = NIIF_INFO;
    Shell_NotifyIconW(NIM_MODIFY, &nid_);
}

void SysTray::destroy() {
    hide();
}

bool SysTray::handleMessage(UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_TRAYICON) {
        if (LOWORD(lParam) == WM_LBUTTONDBLCLK) {
            // Restore window on double-click
            ShowWindow(hwnd_, SW_RESTORE);
            SetForegroundWindow(hwnd_);
            hide();
            return true;
        }
    }
    return false;
}

} // namespace minidlss

#endif // _WIN32
