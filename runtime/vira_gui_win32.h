#ifndef VIRA_GUI_WIN32_H
#define VIRA_GUI_WIN32_H

// ------------------------------------------------------------------------------
// NATIVE WIN32 GDI BACKEND (Zero-Dependency Native Windows)
// ------------------------------------------------------------------------------
#include <windows.h>
#include <windowsx.h>
#include <time.h>

typedef struct {
    HWND          hwnd;
    HDC           hdc_screen;
    HDC           hdc_mem;
    HBITMAP       hbm_mem;
    HBITMAP       hbm_old;
    HFONT         hfont_default;
    int32_t       width;
    int32_t       height;
    bool          should_close;
    int32_t       target_fps;
    LARGE_INTEGER perf_freq;
    LARGE_INTEGER frame_start;
    float         delta_time;

    int32_t       mouse_x;
    int32_t       mouse_y;
    bool          mouse_down[3];
    bool          mouse_clicked[3];
    bool          mouse_prev_down[3];

    bool          keys_down[256];
    bool          keys_pressed[256];
    bool          keys_prev[256];
} vira_gui_ctx_t;

static vira_gui_ctx_t _vira_gui = {0};

/**
 * Converts a standard 24-bit RGB hex color (0xRRGGBB) to a native Win32 GDI COLORREF (0x00BBGGRR).
 * @param hex 24-bit RGB hex color value.
 * @return Win32 COLORREF format color.
 * Usage: Used internally whenever passing colors to Win32 GDI drawing APIs like CreateSolidBrush, CreatePen, and SetTextColor.
 */
static inline COLORREF _vira_color_to_colorref(int32_t hex) {
    BYTE r = (BYTE)((hex >> 16) & 0xFF);
    BYTE g = (BYTE)((hex >> 8) & 0xFF);
    BYTE b = (BYTE)(hex & 0xFF);
    return RGB(r, g, b);
}

/**
 * Native Win32 window message procedure callback.
 * Processes OS window messages: mouse movement, clicks, wheel scrolling, character typing, key down/up, and window close requests.
 * @param hwnd Window handle.
 * @param msg Window message identifier.
 * @param wParam Message parameter (keys, mouse buttons, etc.).
 * @param lParam Message parameter (cursor coordinates, etc.).
 * @return Message result code.
 * Usage: Registered with WNDCLASSEXA and called by Windows OS event dispatch loop.
 */
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
        case WM_MOUSEWHEEL:
            _vira_mouse_wheel_delta += (float)GET_WHEEL_DELTA_WPARAM(wParam) / (float)WHEEL_DELTA;
            return 0;
        case WM_LBUTTONDOWN:
            _vira_gui.mouse_x = GET_X_LPARAM(lParam);
            _vira_gui.mouse_y = GET_Y_LPARAM(lParam);
            _vira_gui.mouse_down[0] = true;
            _vira_gui.mouse_clicked[0] = true;
            return 0;
        case WM_LBUTTONUP:
            _vira_gui.mouse_x = GET_X_LPARAM(lParam);
            _vira_gui.mouse_y = GET_Y_LPARAM(lParam);
            _vira_gui.mouse_down[0] = false;
            return 0;
        case WM_RBUTTONDOWN:
            _vira_gui.mouse_x = GET_X_LPARAM(lParam);
            _vira_gui.mouse_y = GET_Y_LPARAM(lParam);
            _vira_gui.mouse_down[1] = true;
            _vira_gui.mouse_clicked[1] = true;
            return 0;
        case WM_RBUTTONUP:
            _vira_gui.mouse_x = GET_X_LPARAM(lParam);
            _vira_gui.mouse_y = GET_Y_LPARAM(lParam);
            _vira_gui.mouse_down[1] = false;
            return 0;
        case WM_CHAR:
            if (wParam == 8) {
                _vira_backspace_pressed = true;
            } else if (wParam >= 32 && wParam <= 126) {
                if (_vira_char_queue_count < VIRA_CHAR_QUEUE_SIZE) {
                    _vira_char_queue[_vira_char_queue_count++] = (char)wParam;
                }
            }
            return 0;
        case WM_KEYDOWN:
            if (wParam == VK_TAB) {
                _vira_tab_pressed = true;
                _vira_shift_down = ((GetKeyState(VK_SHIFT) & 0x8000) != 0);
                return 0;
            }
            if (wParam == VK_RETURN || wParam == VK_ESCAPE) {
                _vira_active_input_id = -1;
            }
            if (wParam < 256) _vira_gui.keys_down[wParam] = true;
            return 0;
        case WM_KEYUP:
            if (wParam < 256) _vira_gui.keys_down[wParam] = false;
            return 0;
        case WM_ERASEBKGND:
            return 1;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

/**
 * Initializes the Win32 window, message class, double-buffered GDI device contexts, and timers.
 * @param width Window client width in pixels.
 * @param height Window client height in pixels.
 * @param title Window title text string.
 * Usage: Called by `gui_init(w, h, title)` to create and display the native Win32 window.
 */
static inline void vira_gui_init(int32_t width, int32_t height, vira_string title) {
    if (_vira_gui.hwnd != NULL) return;

    HINSTANCE hInst = GetModuleHandle(NULL);
    WNDCLASSEXA wc = {0};
    wc.cbSize = sizeof(WNDCLASSEXA);
    wc.lpfnWndProc = _vira_gui_wndproc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "ViraNativeGUIClass";
    RegisterClassExA(&wc);

    char stack_title[256];
    int32_t tlen = title.len < 255 ? title.len : 255;
    if (tlen > 0 && title.data) memcpy(stack_title, title.data, (size_t)tlen);
    stack_title[tlen] = '\0';

    RECT rc = {0, 0, width, height};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    _vira_gui.width = width;
    _vira_gui.height = height;
    _vira_gui.target_fps = 60;
    _vira_gui.should_close = false;
    _vira_gui.delta_time = 0.016f;

    _vira_gui.hwnd = CreateWindowExA(
        0, "ViraNativeGUIClass", tlen > 0 ? stack_title : "Vira GUI",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top,
        NULL, NULL, hInst, NULL
    );

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
    _vira_layout_depth = 0;
}

/**
 * Checks whether the user has closed the window or sent WM_CLOSE.
 * @return True if window close was requested; false otherwise.
 * Usage: Called as the condition in `for !gui_should_close() { ... }`.
 */
static inline bool vira_gui_should_close(void) { return _vira_gui.should_close; }

/**
 * Cancels a pending close request, allowing the application to prompt for confirmation.
 * Usage: Called by `gui_cancel_close()` when intercepting close events.
 */
static inline void vira_gui_cancel_close(void) { _vira_gui.should_close = false; }

/**
 * Configures the target frame rate for frame pacing.
 * @param fps Target FPS (e.g. 60 or 120).
 * Usage: Called by `gui_set_target_fps(fps)`.
 */
static inline void vira_gui_set_target_fps(int32_t fps) { if (fps > 0) _vira_gui.target_fps = fps; }

/**
 * Returns the current window client area width in pixels.
 * @return Window client width.
 * Usage: Called by `gui_window_width()` to compute responsive layout coordinates.
 */
static inline int32_t vira_gui_window_width(void) { return _vira_gui.width; }

/**
 * Returns the current window client area height in pixels.
 * @return Window client height.
 * Usage: Called by `gui_window_height()` to compute responsive layout coordinates.
 */
static inline int32_t vira_gui_window_height(void) { return _vira_gui.height; }

/**
 * Begins a new frame by pumping the Win32 message queue and updating mouse/keyboard transitions.
 * Usage: Invoked internally by `gui_begin()` at the start of each frame.
 */
static inline void _vira_backend_gui_begin(void) {
    _vira_char_queue_count = 0;
    _vira_backspace_pressed = false;
    _vira_cursor_blink_counter++;
    _vira_mouse_wheel_delta = 0.0f;
    for (int i = 0; i < 3; i++) {
        _vira_gui.mouse_clicked[i] = false;
    }
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    for (int i = 0; i < 3; i++) {
        if (!_vira_gui.mouse_prev_down[i] && _vira_gui.mouse_down[i]) {
            _vira_gui.mouse_clicked[i] = true;
        }
        _vira_gui.mouse_prev_down[i] = _vira_gui.mouse_down[i];
    }
    for (int i = 0; i < 256; i++) {
        _vira_gui.keys_pressed[i] = (!_vira_gui.keys_prev[i] && _vira_gui.keys_down[i]);
        _vira_gui.keys_prev[i] = _vira_gui.keys_down[i];
    }
}

/**
 * Concludes the current frame: performs an atomic BitBlt from backbuffer to screen and sleeps remaining time to pace target FPS.
 * Usage: Invoked internally by `gui_end()` at the end of each frame.
 */
static inline void _vira_backend_gui_end(void) {
    if (!_vira_gui.hdc_screen || !_vira_gui.hdc_mem) return;
    BitBlt(_vira_gui.hdc_screen, 0, 0, _vira_gui.width, _vira_gui.height, _vira_gui.hdc_mem, 0, 0, SRCCOPY);

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

/**
 * Destroys the native Win32 window, releases device contexts, frees fonts, and frees cached images.
 * Usage: Called by `gui_close()` at application teardown.
 */
static inline void vira_gui_close(void) {
    for (int i = 0; i < VIRA_MAX_GUI_IMAGES; i++) {
        if (_vira_gui_images[i].is_valid) {
            if (_vira_gui_images[i].pixels) {
                free(_vira_gui_images[i].pixels);
                _vira_gui_images[i].pixels = NULL;
            }
            _vira_gui_images[i].is_valid = false;
            _vira_gui_images[i].width = 0;
            _vira_gui_images[i].height = 0;
        }
    }
    if (_vira_gui.hdc_mem) {
        SelectObject(_vira_gui.hdc_mem, _vira_gui.hbm_old);
        DeleteObject(_vira_gui.hbm_mem);
        DeleteDC(_vira_gui.hdc_mem);
        _vira_gui.hdc_mem = NULL;
    }
    if (_vira_gui.hfont_default) DeleteObject(_vira_gui.hfont_default);
    if (_vira_gui.hdc_screen && _vira_gui.hwnd) ReleaseDC(_vira_gui.hwnd, _vira_gui.hdc_screen);
    if (_vira_gui.hwnd) DestroyWindow(_vira_gui.hwnd);
}

/**
 * Draws a solid filled rectangle onto the offscreen GDI backbuffer.
 * @param x Top-left X coordinate.
 * @param y Top-left Y coordinate.
 * @param width Rectangle width in pixels.
 * @param height Rectangle height in pixels.
 * @param hex_color 24-bit RGB fill color.
 * Usage: Called by `gui_draw_rect(x, y, w, h, color)`.
 */
static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) {
    if (!_vira_gui.hdc_mem || width <= 0 || height <= 0) return;
    HBRUSH br = CreateSolidBrush(_vira_color_to_colorref(hex_color));
    RECT rc = {x, y, x + width, y + height};
    FillRect(_vira_gui.hdc_mem, &rc, br);
    DeleteObject(br);
}

/**
 * Draws a rounded rectangle with background fill and border outline onto the GDI backbuffer.
 * @param x Top-left X coordinate.
 * @param y Top-left Y coordinate.
 * @param width Rectangle width in pixels.
 * @param height Rectangle height in pixels.
 * @param radius Corner radius in pixels.
 * @param bg_hex 24-bit RGB background color (-1 to omit fill).
 * @param border_hex 24-bit RGB border outline color (-1 to omit border).
 * Usage: Called by `gui_draw_rounded_rect(...)` for styling buttons, cards, and input controls.
 */
static inline void vira_gui_draw_rounded_rect(int32_t x, int32_t y, int32_t width, int32_t height, float radius, int32_t bg_hex, int32_t border_hex) {
    if (!_vira_gui.hdc_mem || width <= 0 || height <= 0) return;
    HBRUSH br = (bg_hex >= 0) ? CreateSolidBrush(_vira_color_to_colorref(bg_hex)) : (HBRUSH)GetStockObject(NULL_BRUSH);
    HPEN pen = (border_hex >= 0) ? CreatePen(PS_SOLID, 1, _vira_color_to_colorref(border_hex)) : (HPEN)GetStockObject(NULL_PEN);
    HGDIOBJ oldBr = SelectObject(_vira_gui.hdc_mem, br);
    HGDIOBJ oldPen = SelectObject(_vira_gui.hdc_mem, pen);
    int d = (int)(radius * 2);
    if (d > width) d = width;
    if (d > height) d = height;
    RoundRect(_vira_gui.hdc_mem, x, y, x + width, y + height, d, d);
    SelectObject(_vira_gui.hdc_mem, oldBr);
    SelectObject(_vira_gui.hdc_mem, oldPen);
    if (bg_hex >= 0) DeleteObject(br);
    if (border_hex >= 0) DeleteObject(pen);
}

/**
 * Clears the full window backbuffer to a solid hex color.
 * @param hex_color 24-bit RGB background clear color.
 * Usage: Called by `gui_clear(color)` at the beginning of each frame.
 */
static inline void vira_gui_clear(int32_t hex_color) {
    vira_gui_draw_rect(0, 0, _vira_gui.width, _vira_gui.height, hex_color);
}

/**
 * Draws antialiased ClearType text onto the GDI backbuffer.
 * @param text Vira string to render.
 * @param x X origin coordinate.
 * @param y Y origin coordinate.
 * @param size Font size hint in points.
 * @param hex_color 24-bit RGB text color.
 * Usage: Called by `gui_draw_text(text, x, y, size, color)`.
 */
static inline void vira_gui_draw_text(vira_string text, int32_t x, int32_t y, int32_t size, int32_t hex_color) {
    (void)size;
    if (!_vira_gui.hdc_mem || text.len <= 0) return;
    SetTextColor(_vira_gui.hdc_mem, _vira_color_to_colorref(hex_color));
    TextOutA(_vira_gui.hdc_mem, x, y, (const char*)text.data, text.len);
}

/**
 * Draws a filled anti-aliased circle onto the GDI backbuffer.
 * @param cx Center X coordinate.
 * @param cy Center Y coordinate.
 * @param radius Circle radius in pixels.
 * @param hex_color 24-bit RGB circle color.
 * Usage: Called by `gui_draw_circle(cx, cy, radius, color)`.
 */
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

/**
 * Draws a 2-pixel vector line between two points on the GDI backbuffer.
 * @param x1 Start X coordinate.
 * @param y1 Start Y coordinate.
 * @param x2 End X coordinate.
 * @param y2 End Y coordinate.
 * @param hex_color 24-bit RGB line color.
 * Usage: Called by `gui_draw_line(x1, y1, x2, y2, color)`.
 */
static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t hex_color) {
    if (!_vira_gui.hdc_mem) return;
    HPEN pen = CreatePen(PS_SOLID, 2, _vira_color_to_colorref(hex_color));
    HGDIOBJ oldPen = SelectObject(_vira_gui.hdc_mem, pen);
    MoveToEx(_vira_gui.hdc_mem, x1, y1, NULL);
    LineTo(_vira_gui.hdc_mem, x2, y2);
    SelectObject(_vira_gui.hdc_mem, oldPen);
    DeleteObject(pen);
}

/**
 * Computes exact text advance width using native GDI GetTextExtentPoint32A.
 * @param text Text string to measure.
 * @return Width in pixels.
 * Usage: Used by auto-layout and widgets to size buttons, labels, and table cells precisely.
 */
static inline int32_t _vira_measure_text_w(vira_string text) {
    if (!_vira_gui.hdc_mem || text.len <= 0) return 0;
    SIZE sz;
    GetTextExtentPoint32A(_vira_gui.hdc_mem, (const char*)text.data, text.len, &sz);
    return (int32_t)sz.cx;
}

/**
 * Returns the current mouse cursor X coordinate in window client space.
 * @return Mouse cursor X coordinate.
 * Usage: Called by `gui_mouse_x()`.
 */
static inline int32_t vira_gui_mouse_x(void) { return _vira_gui.mouse_x; }

/**
 * Returns the current mouse cursor Y coordinate in window client space.
 * @return Mouse cursor Y coordinate.
 * Usage: Called by `gui_mouse_y()`.
 */
static inline int32_t vira_gui_mouse_y(void) { return _vira_gui.mouse_y; }

/**
 * Checks whether a mouse button is held down (0 = Left, 1 = Right, 2 = Middle).
 * @param b Mouse button index.
 * @return True if held; false otherwise.
 * Usage: Called by `gui_is_mouse_down(button)`.
 */
static inline bool vira_gui_is_mouse_down(int32_t b) { return (b >= 0 && b < 3) ? _vira_gui.mouse_down[b] : false; }

/**
 * Checks whether the left mouse button transitioned from up to down on this frame.
 * @return True if clicked this frame; false otherwise.
 * Usage: Used internally by immediate-mode widgets for single-click triggers.
 */
static inline bool _vira_backend_mouse_clicked_left(void) { return _vira_gui.mouse_clicked[0]; }

/**
 * Checks whether a specific key was pressed down on this frame.
 * @param k Key code.
 * @return True if key was pressed; false otherwise.
 * Usage: Called by `gui_is_key_pressed(key)`.
 */
static inline bool vira_gui_is_key_pressed(int32_t k) { return (k >= 0 && k < 256) ? _vira_gui.keys_pressed[k] : false; }

/**
 * Returns the elapsed delta time of the previous frame in seconds.
 * @return Delta time in seconds.
 * Usage: Called by `gui_frame_time()`.
 */
static inline float vira_gui_frame_time(void) { return _vira_gui.delta_time; }

/**
 * Loads a BMP image file into a free image handle slot for Win32 GDI rendering.
 * @param filepath Path to the image file.
 * @return Image slot index, or -1 on failure.
 * Usage: Called by `gui_load_image(path)`.
 */
static inline int32_t _vira_win32_load_image(const char* filepath) {
    int32_t w = 0, h = 0;
    uint32_t* pixels = NULL;
    if (!_vira_load_bmp_file(filepath, &w, &h, &pixels)) return -1;
    for (int i = 0; i < VIRA_MAX_GUI_IMAGES; i++) {
        if (!_vira_gui_images[i].is_valid) {
            _vira_gui_images[i].width = w;
            _vira_gui_images[i].height = h;
            _vira_gui_images[i].channels = 4;
            _vira_gui_images[i].pixels = pixels;
            _vira_gui_images[i].native_handle = NULL;
            _vira_gui_images[i].is_valid = true;
            return i;
        }
    }
    free(pixels);
    return -1;
}

/**
 * Renders a loaded image onto the GDI backbuffer with scaling via StretchDIBits.
 * @param id Image handle index.
 * @param x Top-left X destination coordinate.
 * @param y Top-left Y destination coordinate.
 * @param w Destination width.
 * @param h Destination height.
 * Usage: Called by `gui_draw_image(id, x, y, w, h)`.
 */
static inline void _vira_win32_draw_image(int32_t id, int32_t x, int32_t y, int32_t w, int32_t h) {
    if (id < 0 || id >= VIRA_MAX_GUI_IMAGES || !_vira_gui_images[id].is_valid) return;
    if (!_vira_gui.hdc_mem || !_vira_gui_images[id].pixels) return;

    BITMAPINFO bmi;
    memset(&bmi, 0, sizeof(bmi));
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = _vira_gui_images[id].width;
    bmi.bmiHeader.biHeight = -_vira_gui_images[id].height; // top-down DIB
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    StretchDIBits(_vira_gui.hdc_mem, x, y, w, h,
                  0, 0, _vira_gui_images[id].width, _vira_gui_images[id].height,
                  _vira_gui_images[id].pixels, &bmi, DIB_RGB_COLORS, SRCCOPY);
}

/**
 * Frees pixel memory associated with a Win32 image handle slot.
 * @param id Image handle index.
 * Usage: Called by `gui_free_image(id)`.
 */
static inline void _vira_win32_free_image(int32_t id) {
    if (id < 0 || id >= VIRA_MAX_GUI_IMAGES || !_vira_gui_images[id].is_valid) return;
    if (_vira_gui_images[id].pixels) {
        free(_vira_gui_images[id].pixels);
        _vira_gui_images[id].pixels = NULL;
    }
    _vira_gui_images[id].is_valid = false;
    _vira_gui_images[id].width = 0;
    _vira_gui_images[id].height = 0;
}

/**
 * Sets a rectangular GDI scissor clipping region on the memory DC.
 * @param x Left coordinate of clipping box.
 * @param y Top coordinate of clipping box.
 * @param w Width of clipping box.
 * @param h Height of clipping box.
 * Usage: Called by scroll views and modals to clip child widgets.
 */
static inline void _vira_win32_set_clip(int32_t x, int32_t y, int32_t w, int32_t h) {
    if (!_vira_gui.hdc_mem) return;
    HRGN rgn = CreateRectRgn(x, y, x + (w > 0 ? w : 0), y + (h > 0 ? h : 0));
    SelectClipRgn(_vira_gui.hdc_mem, rgn);
    DeleteObject(rgn);
}

/**
 * Clears the GDI scissor clipping region, restoring drawing across the entire window viewport.
 * Usage: Called at the end of scroll containers and modals.
 */
static inline void _vira_win32_reset_clip(void) {
    if (_vira_gui.hdc_mem) SelectClipRgn(_vira_gui.hdc_mem, NULL);
}

#endif // VIRA_GUI_WIN32_H
