#ifndef VIRA_GUI_H
#define VIRA_GUI_H

#include "vira_rt.h"

#ifdef _WIN32
#include <windows.h>
#include <windowsx.h>
#include <time.h>

typedef struct {
    HWND        hwnd;
    HDC         hdc_screen;
    HDC         hdc_mem;
    HBITMAP     hbm_mem;
    HBITMAP     hbm_old;
    HFONT       hfont_default;
    int32_t     width;
    int32_t     height;
    bool        should_close;
    int32_t     target_fps;
    LARGE_INTEGER perf_freq;
    LARGE_INTEGER frame_start;
    float       delta_time;

    // Mouse state
    int32_t     mouse_x;
    int32_t     mouse_y;
    bool        mouse_down[3];
    bool        mouse_clicked[3];
    bool        mouse_prev_down[3];

    // Key state
    bool        keys_down[256];
    bool        keys_pressed[256];
    bool        keys_prev[256];
} vira_gui_ctx_t;

static vira_gui_ctx_t _vira_gui = {0};

static inline COLORREF _vira_color_to_colorref(int32_t hex) {
    BYTE r = (BYTE)((hex >> 16) & 0xFF);
    BYTE g = (BYTE)((hex >> 8) & 0xFF);
    BYTE b = (BYTE)(hex & 0xFF);
    return RGB(r, g, b);
}

static LRESULT CALLBACK _vira_gui_wndproc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CLOSE:
        case WM_DESTROY:
            _vira_gui.should_close = true;
            PostQuitMessage(0);
            return 0;

        case WM_MOUSEMOVE:
            _vira_gui.mouse_x = GET_X_LPARAM(lParam);
            _vira_gui.mouse_y = GET_Y_LPARAM(lParam);
            return 0;

        case WM_LBUTTONDOWN:
            _vira_gui.mouse_down[0] = true;
            return 0;
        case WM_LBUTTONUP:
            _vira_gui.mouse_down[0] = false;
            return 0;

        case WM_RBUTTONDOWN:
            _vira_gui.mouse_down[1] = true;
            return 0;
        case WM_RBUTTONUP:
            _vira_gui.mouse_down[1] = false;
            return 0;

        case WM_KEYDOWN:
            if (wParam < 256) _vira_gui.keys_down[wParam] = true;
            return 0;
        case WM_KEYUP:
            if (wParam < 256) _vira_gui.keys_down[wParam] = false;
            return 0;

        case WM_ERASEBKGND:
            return 1; // Prevent flicker
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

static inline void vira_gui_init(int32_t width, int32_t height, vira_string title) {
    if (_vira_gui.hwnd != NULL) return;

    HINSTANCE hInst = GetModuleHandle(NULL);
    WNDCLASSEXA wc = {0};
    wc.cbSize = sizeof(WNDCLASSEXA);
    wc.lpfnWndProc = _vira_gui_wndproc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "VexNativeGUIClass";
    RegisterClassExA(&wc);

    char* title_cstr = (char*)malloc(title.len + 1);
    memcpy(title_cstr, title.data, title.len);
    title_cstr[title.len] = '\0';

    RECT rc = {0, 0, width, height};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    int win_w = rc.right - rc.left;
    int win_h = rc.bottom - rc.top;

    _vira_gui.width = width;
    _vira_gui.height = height;
    _vira_gui.target_fps = 60;
    _vira_gui.should_close = false;
    _vira_gui.delta_time = 0.016f;

    _vira_gui.hwnd = CreateWindowExA(
        0, "VexNativeGUIClass", title_cstr,
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, win_w, win_h,
        NULL, NULL, hInst, NULL
    );
    free(title_cstr);

    if (!_vira_gui.hwnd) {
        fprintf(stderr, "vex gui error: failed to create Win32 window\n");
        exit(1);
    }

    _vira_gui.hdc_screen = GetDC(_vira_gui.hwnd);
    _vira_gui.hdc_mem = CreateCompatibleDC(_vira_gui.hdc_screen);
    _vira_gui.hbm_mem = CreateCompatibleBitmap(_vira_gui.hdc_screen, width, height);
    _vira_gui.hbm_old = (HBITMAP)SelectObject(_vira_gui.hdc_mem, _vira_gui.hbm_mem);

    _vira_gui.hfont_default = CreateFontA(
        16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI"
    );
    SelectObject(_vira_gui.hdc_mem, _vira_gui.hfont_default);
    SetBkMode(_vira_gui.hdc_mem, TRANSPARENT);

    QueryPerformanceFrequency(&_vira_gui.perf_freq);
    QueryPerformanceCounter(&_vira_gui.frame_start);
}

static inline bool vira_gui_should_close(void) {
    return _vira_gui.should_close;
}

static inline void vira_gui_set_target_fps(int32_t fps) {
    if (fps > 0) _vira_gui.target_fps = fps;
}

static inline void vira_gui_begin(void) {
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }

    // Update click transitions
    for (int i = 0; i < 3; i++) {
        _vira_gui.mouse_clicked[i] = (!_vira_gui.mouse_prev_down[i] && _vira_gui.mouse_down[i]);
        _vira_gui.mouse_prev_down[i] = _vira_gui.mouse_down[i];
    }
    for (int i = 0; i < 256; i++) {
        _vira_gui.keys_pressed[i] = (!_vira_gui.keys_prev[i] && _vira_gui.keys_down[i]);
        _vira_gui.keys_prev[i] = _vira_gui.keys_down[i];
    }
}

static inline void vira_gui_end(void) {
    if (!_vira_gui.hdc_screen || !_vira_gui.hdc_mem) return;

    // Blit from backbuffer to screen
    BitBlt(_vira_gui.hdc_screen, 0, 0, _vira_gui.width, _vira_gui.height,
           _vira_gui.hdc_mem, 0, 0, SRCCOPY);

    // Frame rate limiter
    LARGE_INTEGER frame_end;
    QueryPerformanceCounter(&frame_end);
    double elapsed_sec = (double)(frame_end.QuadPart - _vira_gui.frame_start.QuadPart) / (double)_vira_gui.perf_freq.QuadPart;
    double target_sec = 1.0 / (double)_vira_gui.target_fps;

    if (elapsed_sec < target_sec) {
        DWORD sleep_ms = (DWORD)((target_sec - elapsed_sec) * 1000.0);
        if (sleep_ms > 0) Sleep(sleep_ms);
        QueryPerformanceCounter(&frame_end);
        elapsed_sec = (double)(frame_end.QuadPart - _vira_gui.frame_start.QuadPart) / (double)_vira_gui.perf_freq.QuadPart;
    }

    _vira_gui.delta_time = (float)elapsed_sec;
    _vira_gui.frame_start = frame_end;
}

static inline void vira_gui_close(void) {
    if (_vira_gui.hdc_mem) {
        SelectObject(_vira_gui.hdc_mem, _vira_gui.hbm_old);
        DeleteObject(_vira_gui.hbm_mem);
        DeleteDC(_vira_gui.hdc_mem);
        _vira_gui.hdc_mem = NULL;
    }
    if (_vira_gui.hfont_default) {
        DeleteObject(_vira_gui.hfont_default);
        _vira_gui.hfont_default = NULL;
    }
    if (_vira_gui.hdc_screen && _vira_gui.hwnd) {
        ReleaseDC(_vira_gui.hwnd, _vira_gui.hdc_screen);
        _vira_gui.hdc_screen = NULL;
    }
    if (_vira_gui.hwnd) {
        DestroyWindow(_vira_gui.hwnd);
        _vira_gui.hwnd = NULL;
    }
}

static inline void vira_gui_clear(int32_t hex_color) {
    if (!_vira_gui.hdc_mem) return;
    COLORREF cr = _vira_color_to_colorref(hex_color);
    HBRUSH br = CreateSolidBrush(cr);
    RECT rc = {0, 0, _vira_gui.width, _vira_gui.height};
    FillRect(_vira_gui.hdc_mem, &rc, br);
    DeleteObject(br);
}

static inline void vira_gui_draw_text(vira_string text, int32_t x, int32_t y, int32_t size, int32_t hex_color) {
    if (!_vira_gui.hdc_mem || text.len <= 0) return;
    COLORREF cr = _vira_color_to_colorref(hex_color);

    HFONT font = _vira_gui.hfont_default;
    HFONT oldFont = NULL;
    if (size != 16 && size > 4) {
        font = CreateFontA(
            size, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI"
        );
        oldFont = (HFONT)SelectObject(_vira_gui.hdc_mem, font);
    }

    SetTextColor(_vira_gui.hdc_mem, cr);
    TextOutA(_vira_gui.hdc_mem, x, y, (const char*)text.data, text.len);

    if (oldFont) {
        SelectObject(_vira_gui.hdc_mem, oldFont);
        DeleteObject(font);
    }
}

static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) {
    if (!_vira_gui.hdc_mem) return;
    COLORREF cr = _vira_color_to_colorref(hex_color);
    HBRUSH br = CreateSolidBrush(cr);
    RECT rc = {x, y, x + width, y + height};
    FillRect(_vira_gui.hdc_mem, &rc, br);
    DeleteObject(br);
}

static inline void vira_gui_draw_circle(int32_t cx, int32_t cy, float radius, int32_t hex_color) {
    if (!_vira_gui.hdc_mem) return;
    COLORREF cr = _vira_color_to_colorref(hex_color);
    HBRUSH br = CreateSolidBrush(cr);
    HPEN pen = CreatePen(PS_SOLID, 1, cr);
    HGDIOBJ oldBr = SelectObject(_vira_gui.hdc_mem, br);
    HGDIOBJ oldPen = SelectObject(_vira_gui.hdc_mem, pen);

    int r = (int)radius;
    Ellipse(_vira_gui.hdc_mem, cx - r, cy - r, cx + r, cy + r);

    SelectObject(_vira_gui.hdc_mem, oldBr);
    SelectObject(_vira_gui.hdc_mem, oldPen);
    DeleteObject(br);
    DeleteObject(pen);
}

static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t hex_color) {
    if (!_vira_gui.hdc_mem) return;
    COLORREF cr = _vira_color_to_colorref(hex_color);
    HPEN pen = CreatePen(PS_SOLID, 2, cr);
    HGDIOBJ oldPen = SelectObject(_vira_gui.hdc_mem, pen);

    MoveToEx(_vira_gui.hdc_mem, x1, y1, NULL);
    LineTo(_vira_gui.hdc_mem, x2, y2);

    SelectObject(_vira_gui.hdc_mem, oldPen);
    DeleteObject(pen);
}

// --- IMMEDIATE-MODE INTERACTIVE CONTROLS ---

static inline bool vira_gui_button(int32_t x, int32_t y, int32_t width, int32_t height, vira_string label) {
    if (!_vira_gui.hdc_mem) return false;

    int mx = _vira_gui.mouse_x;
    int my = _vira_gui.mouse_y;
    bool is_hovered = (mx >= x && mx <= x + width && my >= y && my <= y + height);
    bool is_down = is_hovered && _vira_gui.mouse_down[0];
    bool is_clicked = is_hovered && _vira_gui.mouse_clicked[0];

    // Button background
    int32_t bg_color = is_down ? 0x313244 : (is_hovered ? 0x45475A : 0x1E1E2E);
    int32_t border_color = is_hovered ? 0x89B4FA : 0x585B70;
    int32_t text_color = is_hovered ? 0xFFFFFF : 0xCDD6F4;

    vira_gui_draw_rect(x, y, width, height, bg_color);

    // Border
    COLORREF border_cr = _vira_color_to_colorref(border_color);
    HPEN pen = CreatePen(PS_SOLID, 1, border_cr);
    HGDIOBJ oldPen = SelectObject(_vira_gui.hdc_mem, pen);
    HGDIOBJ oldBr = SelectObject(_vira_gui.hdc_mem, GetStockObject(NULL_BRUSH));
    Rectangle(_vira_gui.hdc_mem, x, y, x + width, y + height);
    SelectObject(_vira_gui.hdc_mem, oldPen);
    SelectObject(_vira_gui.hdc_mem, oldBr);
    DeleteObject(pen);

    // Centered label text
    if (label.len > 0) {
        SIZE txt_size;
        GetTextExtentPoint32A(_vira_gui.hdc_mem, (const char*)label.data, label.len, &txt_size);
        int tx = x + (width - txt_size.cx) / 2;
        int ty = y + (height - txt_size.cy) / 2;
        SetTextColor(_vira_gui.hdc_mem, _vira_color_to_colorref(text_color));
        TextOutA(_vira_gui.hdc_mem, tx, ty, (const char*)label.data, label.len);
    }

    return is_clicked;
}

static inline float vira_gui_slider(int32_t x, int32_t y, int32_t width, int32_t height,
                                  vira_string text_left, vira_string text_right,
                                  float val, float min_val, float max_val) {
    if (!_vira_gui.hdc_mem || width <= 0 || max_val <= min_val) return val;

    int mx = _vira_gui.mouse_x;
    int my = _vira_gui.mouse_y;
    bool is_hovered = (mx >= x && mx <= x + width && my >= y && my <= y + height);

    if (is_hovered && _vira_gui.mouse_down[0]) {
        float ratio = (float)(mx - x) / (float)width;
        if (ratio < 0.0f) ratio = 0.0f;
        if (ratio > 1.0f) ratio = 1.0f;
        val = min_val + ratio * (max_val - min_val);
    }

    // Track bar
    int track_y = y + height / 2 - 2;
    vira_gui_draw_rect(x, track_y, width, 4, 0x313244);

    // Active fill
    float active_ratio = (val - min_val) / (max_val - min_val);
    if (active_ratio < 0.0f) active_ratio = 0.0f;
    if (active_ratio > 1.0f) active_ratio = 1.0f;
    int fill_w = (int)(active_ratio * (float)width);
    if (fill_w > 0) {
        vira_gui_draw_rect(x, track_y, fill_w, 4, 0x89B4FA);
    }

    // Slider Knob
    int knob_x = x + fill_w;
    int knob_y = y + height / 2;
    vira_gui_draw_circle(knob_x, knob_y, 7.0f, is_hovered ? 0xB4BEFE : 0x89B4FA);

    // Left and Right label text
    if (text_left.len > 0) {
        vira_gui_draw_text(text_left, x, y - 18, 14, 0xA6ADC8);
    }
    if (text_right.len > 0) {
        SIZE txt_size;
        GetTextExtentPoint32A(_vira_gui.hdc_mem, (const char*)text_right.data, text_right.len, &txt_size);
        vira_gui_draw_text(text_right, x + width - txt_size.cx, y - 18, 14, 0xA6ADC8);
    }

    return val;
}

static inline bool vira_gui_checkbox(int32_t x, int32_t y, int32_t width, int32_t height,
                                   vira_string label, bool checked) {
    if (!_vira_gui.hdc_mem) return checked;

    int box_size = 18;
    int mx = _vira_gui.mouse_x;
    int my = _vira_gui.mouse_y;
    bool is_hovered = (mx >= x && mx <= x + width && my >= y && my <= y + height);

    if (is_hovered && _vira_gui.mouse_clicked[0]) {
        checked = !checked;
    }

    // Box outer
    vira_gui_draw_rect(x, y + (height - box_size) / 2, box_size, box_size, 0x313244);
    if (checked) {
        vira_gui_draw_rect(x + 3, y + (height - box_size) / 2 + 3, box_size - 6, box_size - 6, 0xA6E3A1);
    }

    // Label
    if (label.len > 0) {
        vira_gui_draw_text(label, x + box_size + 8, y + (height - 16) / 2, 15, is_hovered ? 0xFFFFFF : 0xCDD6F4);
    }

    return checked;
}

static inline void vira_gui_label(int32_t x, int32_t y, int32_t width, int32_t height, vira_string text) {
    (void)width;
    (void)height;
    vira_gui_draw_text(text, x, y, 16, 0xCDD6F4);
}

static inline int32_t vira_gui_mouse_x(void) {
    return _vira_gui.mouse_x;
}

static inline int32_t vira_gui_mouse_y(void) {
    return _vira_gui.mouse_y;
}

static inline bool vira_gui_is_mouse_down(int32_t button) {
    if (button >= 0 && button < 3) return _vira_gui.mouse_down[button];
    return false;
}

static inline bool vira_gui_is_key_pressed(int32_t key) {
    if (key >= 0 && key < 256) return _vira_gui.keys_pressed[key];
    return false;
}

static inline float vira_gui_frame_time(void) {
    return _vira_gui.delta_time;
}

#else
// Fallback for non-Windows platforms (POSIX/Linux/macOS)
static inline void vira_gui_init(int32_t width, int32_t height, vira_string title) { (void)width; (void)height; (void)title; }
static inline bool vira_gui_should_close(void) { return true; }
static inline void vira_gui_set_target_fps(int32_t fps) { (void)fps; }
static inline void vira_gui_begin(void) {}
static inline void vira_gui_end(void) {}
static inline void vira_gui_close(void) {}
static inline void vira_gui_clear(int32_t hex_color) { (void)hex_color; }
static inline void vira_gui_draw_text(vira_string text, int32_t x, int32_t y, int32_t size, int32_t hex_color) { (void)text; (void)x; (void)y; (void)size; (void)hex_color; }
static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) { (void)x; (void)y; (void)width; (void)height; (void)hex_color; }
static inline void vira_gui_draw_circle(int32_t cx, int32_t cy, float radius, int32_t hex_color) { (void)cx; (void)cy; (void)radius; (void)hex_color; }
static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t hex_color) { (void)x1; (void)y1; (void)x2; (void)y2; (void)hex_color; }
static inline bool vira_gui_button(int32_t x, int32_t y, int32_t width, int32_t height, vira_string label) { (void)x; (void)y; (void)width; (void)height; (void)label; return false; }
static inline float vira_gui_slider(int32_t x, int32_t y, int32_t width, int32_t height, vira_string text_left, vira_string text_right, float val, float min_val, float max_val) { (void)x; (void)y; (void)width; (void)height; (void)text_left; (void)text_right; (void)min_val; (void)max_val; return val; }
static inline bool vira_gui_checkbox(int32_t x, int32_t y, int32_t width, int32_t height, vira_string label, bool checked) { (void)x; (void)y; (void)width; (void)height; (void)label; return checked; }
static inline void vira_gui_label(int32_t x, int32_t y, int32_t width, int32_t height, vira_string text) { (void)x; (void)y; (void)width; (void)height; (void)text; }
static inline int32_t vira_gui_mouse_x(void) { return 0; }
static inline int32_t vira_gui_mouse_y(void) { return 0; }
static inline bool vira_gui_is_mouse_down(int32_t button) { (void)button; return false; }
static inline bool vira_gui_is_key_pressed(int32_t key) { (void)key; return false; }
static inline float vira_gui_frame_time(void) { return 0.016f; }
#endif

#endif // VIRA_GUI_H
