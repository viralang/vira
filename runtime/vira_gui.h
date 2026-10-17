#ifndef VIRA_GUI_H
#define VIRA_GUI_H

#include "vira_rt.h"

// ==============================================================================
// 1. COMMON DECLARATIVE AUTO-LAYOUT DEFINITIONS
// ==============================================================================

typedef enum {
    VIRA_LAYOUT_NONE = 0,
    VIRA_LAYOUT_VERTICAL,
    VIRA_LAYOUT_HORIZONTAL
} vira_layout_type_t;

typedef struct {
    vira_layout_type_t type;
    int32_t x;
    int32_t y;
    int32_t cur_x;
    int32_t cur_y;
    int32_t width;
    int32_t height;
    int32_t spacing;
    int32_t start_y;
    int32_t is_card;
} vira_layout_node_t;

#define VIRA_MAX_LAYOUT_DEPTH 16

static vira_layout_node_t _vira_layout_stack[VIRA_MAX_LAYOUT_DEPTH];
static int32_t            _vira_layout_depth = 0;

static inline vira_layout_node_t* _vira_gui_cur_layout(void) {
    if (_vira_layout_depth > 0) {
        return &_vira_layout_stack[_vira_layout_depth - 1];
    }
    return NULL;
}

#define VIRA_MAX_TEXT_INPUTS 32
#define VIRA_TEXT_INPUT_MAX_LEN 256
#define VIRA_CHAR_QUEUE_SIZE 32

typedef struct {
    char    buffer[VIRA_TEXT_INPUT_MAX_LEN];
    int32_t len;
    bool    initialized;
} vira_text_input_state_t;

static vira_text_input_state_t _vira_text_inputs[VIRA_MAX_TEXT_INPUTS] = {0};
static int32_t                 _vira_active_input_id = -1;
static int32_t                 _vira_input_counter = 0;
static char                    _vira_char_queue[VIRA_CHAR_QUEUE_SIZE] = {0};
static int32_t                 _vira_char_queue_count = 0;
static bool                    _vira_backspace_pressed = false;
static uint32_t                _vira_cursor_blink_counter = 0;

static int32_t                 _vira_card_heights[16] = {0};
static int32_t                 _vira_card_counter = 0;
static int32_t                 _vira_card_end_counter = 0;

// ==============================================================================
// 2. NATIVE OS BACKEND IMPLEMENTATIONS (NO RAYLIB REQUIRED)
// ==============================================================================

#if defined(_WIN32)

// ------------------------------------------------------------------------------
// 2A. NATIVE WIN32 GDI BACKEND (Zero-Dependency Native Windows)
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

static inline bool vira_gui_should_close(void) { return _vira_gui.should_close; }
static inline void vira_gui_cancel_close(void) { _vira_gui.should_close = false; }
static inline void vira_gui_set_target_fps(int32_t fps) { if (fps > 0) _vira_gui.target_fps = fps; }
static inline int32_t vira_gui_window_width(void) { return _vira_gui.width; }
static inline int32_t vira_gui_window_height(void) { return _vira_gui.height; }

static inline void vira_gui_begin(void) {
    _vira_layout_depth = 0;
    _vira_input_counter = 0;
    _vira_card_counter = 0;
    _vira_card_end_counter = 0;
    _vira_char_queue_count = 0;
    _vira_backspace_pressed = false;
    _vira_cursor_blink_counter++;
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
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

static inline void vira_gui_close(void) {
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

static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) {
    if (!_vira_gui.hdc_mem) return;
    HBRUSH br = CreateSolidBrush(_vira_color_to_colorref(hex_color));
    RECT rc = {x, y, x + width, y + height};
    FillRect(_vira_gui.hdc_mem, &rc, br);
    DeleteObject(br);
}

static inline void vira_gui_clear(int32_t hex_color) {
    vira_gui_draw_rect(0, 0, _vira_gui.width, _vira_gui.height, hex_color);
}

static inline void vira_gui_draw_text(vira_string text, int32_t x, int32_t y, int32_t size, int32_t hex_color) {
    (void)size;
    if (!_vira_gui.hdc_mem || text.len <= 0) return;
    SetTextColor(_vira_gui.hdc_mem, _vira_color_to_colorref(hex_color));
    TextOutA(_vira_gui.hdc_mem, x, y, (const char*)text.data, text.len);
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
    HPEN pen = CreatePen(PS_SOLID, 2, _vira_color_to_colorref(hex_color));
    HGDIOBJ oldPen = SelectObject(_vira_gui.hdc_mem, pen);
    MoveToEx(_vira_gui.hdc_mem, x1, y1, NULL);
    LineTo(_vira_gui.hdc_mem, x2, y2);
    SelectObject(_vira_gui.hdc_mem, oldPen);
    DeleteObject(pen);
}

static inline int32_t _vira_measure_text_w(vira_string text) {
    if (!_vira_gui.hdc_mem || text.len <= 0) return 0;
    SIZE sz;
    GetTextExtentPoint32A(_vira_gui.hdc_mem, (const char*)text.data, text.len, &sz);
    return (int32_t)sz.cx;
}

static inline int32_t vira_gui_mouse_x(void) { return _vira_gui.mouse_x; }
static inline int32_t vira_gui_mouse_y(void) { return _vira_gui.mouse_y; }
static inline bool vira_gui_is_mouse_down(int32_t b) { return (b >= 0 && b < 3) ? _vira_gui.mouse_down[b] : false; }
static inline bool _vira_gui_mouse_clicked_left(void) { return _vira_gui.mouse_clicked[0]; }
static inline bool vira_gui_is_key_pressed(int32_t k) { return (k >= 0 && k < 256) ? _vira_gui.keys_pressed[k] : false; }
static inline float vira_gui_frame_time(void) { return _vira_gui.delta_time; }

#elif (defined(__linux__) || defined(__unix__)) && !defined(VIRA_NO_X11)

// ------------------------------------------------------------------------------
// 2B. NATIVE LINUX X11 BACKEND (Zero-Dependency Linux Windowing via -lX11)
// ------------------------------------------------------------------------------
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <time.h>

typedef struct {
    Display*        dpy;
    int             screen;
    Window          win;
    Pixmap          backbuf;
    GC              gc;
    XFontStruct*    font;
    Atom            wm_delete;
    int32_t         width;
    int32_t         height;
    int32_t         depth;
    bool            should_close;
    int32_t         target_fps;
    struct timespec frame_start;
    float           delta_time;

    int32_t         mouse_x;
    int32_t         mouse_y;
    bool            mouse_down[3];
    bool            mouse_clicked[3];
    bool            mouse_prev_down[3];
    bool            keys_down[256];
    bool            keys_pressed[256];
    bool            keys_prev[256];
} vira_x11_t;

static vira_x11_t _vira_x11 = {0};

static inline void vira_gui_init(int32_t width, int32_t height, vira_string title) {
    if (_vira_x11.dpy != NULL) return;
    _vira_x11.dpy = XOpenDisplay(NULL);
    if (!_vira_x11.dpy) {
        fprintf(stderr, "vira gui error: cannot open X11 display\n");
        exit(1);
    }
    _vira_x11.screen = DefaultScreen(_vira_x11.dpy);
    _vira_x11.depth  = DefaultDepth(_vira_x11.dpy, _vira_x11.screen);
    _vira_x11.width  = width;
    _vira_x11.height = height;
    _vira_x11.target_fps = 60;
    _vira_x11.delta_time = 0.016f;

    _vira_x11.win = XCreateSimpleWindow(
        _vira_x11.dpy, RootWindow(_vira_x11.dpy, _vira_x11.screen),
        100, 100, (unsigned int)width, (unsigned int)height, 0, 0x11111B, 0x11111B
    );

    char stack_title[256];
    int32_t tlen = title.len < 255 ? title.len : 255;
    if (tlen > 0 && title.data) memcpy(stack_title, title.data, (size_t)tlen);
    stack_title[tlen] = '\0';
    XStoreName(_vira_x11.dpy, _vira_x11.win, tlen > 0 ? stack_title : "Vira GUI");

    XSelectInput(_vira_x11.dpy, _vira_x11.win,
                 ExposureMask | KeyPressMask | KeyReleaseMask |
                 ButtonPressMask | ButtonReleaseMask | PointerMotionMask);
    _vira_x11.wm_delete = XInternAtom(_vira_x11.dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(_vira_x11.dpy, _vira_x11.win, &_vira_x11.wm_delete, 1);

    // Lock fixed window size hints for tiling window managers
    XSizeHints hints = {0};
    hints.flags = PSize | PMinSize | PMaxSize;
    hints.min_width = hints.max_width = width;
    hints.min_height = hints.max_height = height;
    XSetWMNormalHints(_vira_x11.dpy, _vira_x11.win, &hints);

    _vira_x11.backbuf = XCreatePixmap(_vira_x11.dpy, _vira_x11.win, (unsigned int)width, (unsigned int)height, (unsigned int)_vira_x11.depth);
    _vira_x11.gc      = XCreateGC(_vira_x11.dpy, _vira_x11.backbuf, 0, NULL);
    _vira_x11.font    = XLoadQueryFont(_vira_x11.dpy, "9x15");
    if (!_vira_x11.font) _vira_x11.font = XLoadQueryFont(_vira_x11.dpy, "fixed");
    if (_vira_x11.font) XSetFont(_vira_x11.dpy, _vira_x11.gc, _vira_x11.font->fid);

    XMapWindow(_vira_x11.dpy, _vira_x11.win);
    XFlush(_vira_x11.dpy);
    clock_gettime(CLOCK_MONOTONIC, &_vira_x11.frame_start);
    _vira_layout_depth = 0;
}

static inline bool vira_gui_should_close(void) { return _vira_x11.should_close; }
static inline void vira_gui_cancel_close(void) { _vira_x11.should_close = false; }
static inline void vira_gui_set_target_fps(int32_t fps) { if (fps > 0) _vira_x11.target_fps = fps; }
static inline int32_t vira_gui_window_width(void) { return _vira_x11.width; }
static inline int32_t vira_gui_window_height(void) { return _vira_x11.height; }

static inline void vira_gui_begin(void) {
    _vira_layout_depth = 0;
    _vira_input_counter = 0;
    _vira_card_counter = 0;
    _vira_card_end_counter = 0;
    _vira_char_queue_count = 0;
    _vira_backspace_pressed = false;
    _vira_cursor_blink_counter++;
    for (int i = 0; i < 3; i++) {
        _vira_x11.mouse_clicked[i] = false;
    }
    for (int i = 0; i < 256; i++) {
        _vira_x11.keys_pressed[i] = false;
    }
    while (_vira_x11.dpy && XPending(_vira_x11.dpy) > 0) {
        XEvent ev;
        XNextEvent(_vira_x11.dpy, &ev);
        if (ev.type == ClientMessage && (Atom)ev.xclient.data.l[0] == _vira_x11.wm_delete) {
            _vira_x11.should_close = true;
        } else if (ev.type == MotionNotify) {
            _vira_x11.mouse_x = ev.xmotion.x;
            _vira_x11.mouse_y = ev.xmotion.y;
        } else if (ev.type == ButtonPress) {
            int b = (ev.xbutton.button == Button1) ? 0 : ((ev.xbutton.button == Button3) ? 1 : 2);
            _vira_x11.mouse_down[b] = true;
            _vira_x11.mouse_clicked[b] = true;
        } else if (ev.type == ButtonRelease) {
            int b = (ev.xbutton.button == Button1) ? 0 : ((ev.xbutton.button == Button3) ? 1 : 2);
            _vira_x11.mouse_down[b] = false;
        } else if (ev.type == KeyPress) {
            char char_buf[32];
            KeySym sym = 0;
            int count = XLookupString(&ev.xkey, char_buf, sizeof(char_buf), &sym, NULL);
            if (sym == XK_BackSpace) {
                _vira_backspace_pressed = true;
            } else if (sym == XK_Return || sym == XK_Escape) {
                _vira_active_input_id = -1;
            } else if (count > 0) {
                for (int c = 0; c < count; c++) {
                    if ((unsigned char)char_buf[c] >= 32 && (unsigned char)char_buf[c] <= 126) {
                        if (_vira_char_queue_count < VIRA_CHAR_QUEUE_SIZE) {
                            _vira_char_queue[_vira_char_queue_count++] = char_buf[c];
                        }
                    }
                }
            }
            if (sym < 256) {
                if (!_vira_x11.keys_down[sym]) _vira_x11.keys_pressed[sym] = true;
                _vira_x11.keys_down[sym] = true;
            }
        } else if (ev.type == KeyRelease) {
            KeySym sym = XLookupKeysym(&ev.xkey, 0);
            if (sym < 256) {
                _vira_x11.keys_down[sym] = false;
            }
        }
    }
}

static inline void vira_gui_end(void) {
    if (!_vira_x11.dpy) return;
    XCopyArea(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.win, _vira_x11.gc,
              0, 0, (unsigned int)_vira_x11.width, (unsigned int)_vira_x11.height, 0, 0);
    XFlush(_vira_x11.dpy);

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (double)(now.tv_sec - _vira_x11.frame_start.tv_sec) +
                     (double)(now.tv_nsec - _vira_x11.frame_start.tv_nsec) / 1e9;
    double target = 1.0 / (double)_vira_x11.target_fps;
    if (elapsed < target) {
        double rem = target - elapsed;
        struct timespec req = { (time_t)rem, (long)((rem - (time_t)rem) * 1e9) };
        nanosleep(&req, NULL);
        clock_gettime(CLOCK_MONOTONIC, &now);
    }
    _vira_x11.frame_start = now;
}

static inline void vira_gui_close(void) {
    if (!_vira_x11.dpy) return;
    if (_vira_x11.backbuf) XFreePixmap(_vira_x11.dpy, _vira_x11.backbuf);
    if (_vira_x11.gc) XFreeGC(_vira_x11.dpy, _vira_x11.gc);
    if (_vira_x11.win) XDestroyWindow(_vira_x11.dpy, _vira_x11.win);
    XCloseDisplay(_vira_x11.dpy);
    _vira_x11.dpy = NULL;
}

static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) {
    if (!_vira_x11.dpy || width <= 0 || height <= 0) return;
    XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(hex_color & 0xFFFFFF));
    XFillRectangle(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, x, y, (unsigned int)width, (unsigned int)height);
}

static inline void vira_gui_clear(int32_t hex_color) {
    vira_gui_draw_rect(0, 0, _vira_x11.width, _vira_x11.height, hex_color);
}

static inline void vira_gui_draw_text(vira_string text, int32_t x, int32_t y, int32_t size, int32_t hex_color) {
    (void)size;
    if (!_vira_x11.dpy || text.len <= 0) return;
    XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(hex_color & 0xFFFFFF));
    XDrawString(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, x, y + 13, (const char*)text.data, text.len);
}

static inline void vira_gui_draw_circle(int32_t cx, int32_t cy, float radius, int32_t hex_color) {
    if (!_vira_x11.dpy || radius <= 0.0f) return;
    int r = (int)radius;
    XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(hex_color & 0xFFFFFF));
    XFillArc(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, cx - r, cy - r, (unsigned int)(r * 2), (unsigned int)(r * 2), 0, 360 * 64);
}

static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t hex_color) {
    if (!_vira_x11.dpy) return;
    XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(hex_color & 0xFFFFFF));
    XDrawLine(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, x1, y1, x2, y2);
}

static inline int32_t _vira_measure_text_w(vira_string text) {
    if (_vira_x11.font && text.len > 0) return XTextWidth(_vira_x11.font, (const char*)text.data, text.len);
    return text.len * 8;
}

static inline int32_t vira_gui_mouse_x(void) { return _vira_x11.mouse_x; }
static inline int32_t vira_gui_mouse_y(void) { return _vira_x11.mouse_y; }
static inline bool vira_gui_is_mouse_down(int32_t b) { return (b >= 0 && b < 3) ? _vira_x11.mouse_down[b] : false; }
static inline bool _vira_gui_mouse_clicked_left(void) { return _vira_x11.mouse_clicked[0]; }
static inline bool vira_gui_is_key_pressed(int32_t k) { return (k >= 0 && k < 256) ? _vira_x11.keys_pressed[k] : false; }
static inline float vira_gui_frame_time(void) { return _vira_x11.delta_time; }

#else

// 2C. HEADLESS STUB
static inline void vira_gui_init(int32_t w, int32_t h, vira_string t) { (void)w; (void)h; (void)t; }
static inline bool vira_gui_should_close(void) { return true; }
static inline void vira_gui_cancel_close(void) {}
static inline void vira_gui_set_target_fps(int32_t fps) { (void)fps; }
static inline int32_t vira_gui_window_width(void) { return 800; }
static inline int32_t vira_gui_window_height(void) { return 600; }
static inline void vira_gui_begin(void) {}
static inline void vira_gui_end(void) {}
static inline void vira_gui_close(void) {}
static inline void vira_gui_clear(int32_t c) { (void)c; }
static inline void vira_gui_draw_text(vira_string t, int32_t x, int32_t y, int32_t s, int32_t c) { (void)t; (void)x; (void)y; (void)s; (void)c; }
static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t w, int32_t h, int32_t c) { (void)x; (void)y; (void)w; (void)h; (void)c; }
static inline void vira_gui_draw_circle(int32_t cx, int32_t cy, float r, int32_t c) { (void)cx; (void)cy; (void)r; (void)c; }
static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t c) { (void)x1; (void)y1; (void)x2; (void)y2; (void)c; }
static inline int32_t _vira_measure_text_w(vira_string t) { return t.len * 8; }
static inline int32_t vira_gui_mouse_x(void) { return 0; }
static inline int32_t vira_gui_mouse_y(void) { return 0; }
static inline bool vira_gui_is_mouse_down(int32_t b) { (void)b; return false; }
static inline bool _vira_gui_mouse_clicked_left(void) { return false; }
static inline bool vira_gui_is_key_pressed(int32_t k) { (void)k; return false; }
static inline float vira_gui_frame_time(void) { return 0.016f; }
static inline vira_string vira_gui_text_input(int32_t x, int32_t y, int32_t w, int32_t h, vira_string p, vira_string t) { (void)x; (void)y; (void)w; (void)h; (void)p; return t; }
static inline void vira_gui_progress_bar(int32_t x, int32_t y, int32_t w, int32_t h, float pct) { (void)x; (void)y; (void)w; (void)h; (void)pct; }
static inline bool vira_gui_radio_button(int32_t x, int32_t y, vira_string l, bool s) { (void)x; (void)y; (void)l; return s; }
static inline void vira_gui_draw_badge(int32_t x, int32_t y, vira_string t, int32_t bg, int32_t fg) { (void)x; (void)y; (void)t; (void)bg; (void)fg; }
static inline vira_string vira_gui_input(vira_string p, vira_string t) { (void)p; return t; }
static inline void vira_gui_progress(float pct) { (void)pct; }
static inline bool vira_gui_radio(vira_string l, bool s) { (void)l; return s; }
static inline void vira_gui_separator(void) {}
static inline void vira_gui_badge(vira_string t, int32_t bg, int32_t fg) { (void)t; (void)bg; (void)fg; }
static inline bool vira_gui_tab_item(vira_string l, bool a) { (void)l; return a; }
static inline void vira_gui_modal_begin(vira_string t, int32_t w, int32_t h) { (void)t; (void)w; (void)h; }
static inline void vira_gui_modal_end(void) {}
static inline void vira_gui_lbl_temp(vira_string t) { vira_str_free(&t); }
static inline void vira_gui_draw_text_temp(vira_string t, int32_t x, int32_t y, int32_t s, int32_t c) { (void)x; (void)y; (void)s; (void)c; vira_str_free(&t); }
static inline void vira_gui_card_begin_temp(vira_string t, int32_t w) { (void)w; vira_str_free(&t); }
static inline void vira_gui_badge_temp(vira_string t, int32_t bg, int32_t fg) { (void)bg; (void)fg; vira_str_free(&t); }
static inline bool vira_gui_btn_temp(vira_string l) { vira_str_free(&l); return false; }
static inline bool vira_gui_tab_item_temp(vira_string l, bool a) { vira_str_free(&l); return a; }
static inline void vira_gui_modal_begin_temp(vira_string t, int32_t w, int32_t h) { (void)w; (void)h; vira_str_free(&t); }

#endif

// ==============================================================================
// 3. SHARED WIDGETS & DECLARATIVE AUTO-LAYOUT ENGINE
// ==============================================================================

static inline bool vira_gui_button(int32_t x, int32_t y, int32_t width, int32_t height, vira_string label) {
    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();
    bool is_hovered = (mx >= x && mx <= x + width && my >= y && my <= y + height);
    bool is_down    = is_hovered && vira_gui_is_mouse_down(0);
    bool is_clicked = is_hovered && _vira_gui_mouse_clicked_left();

    int32_t bg_color     = is_down ? 0x313244 : (is_hovered ? 0x45475A : 0x1E1E2E);
    int32_t border_color = is_hovered ? 0x89B4FA : 0x585B70;
    int32_t text_color   = is_hovered ? 0xFFFFFF : 0xCDD6F4;

    vira_gui_draw_rect(x, y, width, height, bg_color);
    // 1px border outline using lines (works identically on Win32 & X11)
    vira_gui_draw_line(x, y, x + width, y, border_color);
    vira_gui_draw_line(x, y + height, x + width, y + height, border_color);
    vira_gui_draw_line(x, y, x, y + height, border_color);
    vira_gui_draw_line(x + width, y, x + width, y + height, border_color);

    if (label.len > 0) {
        int32_t tw = _vira_measure_text_w(label);
        vira_gui_draw_text(label, x + (width - tw) / 2, y + (height - 16) / 2, 16, text_color);
    }
    return is_clicked;
}

static inline float vira_gui_slider(int32_t x, int32_t y, int32_t width, int32_t height,
                                    vira_string text_left, vira_string text_right,
                                    float val, float min_val, float max_val) {
    if (width <= 0 || max_val <= min_val) return val;

    bool has_labels = (text_left.len > 0 || text_right.len > 0);
    int32_t track_y = has_labels ? (y + 22) : (y + height / 2 - 2);

    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();
    // Generous vertical hitbox so dragging doesn't drop easily
    bool is_hovered = (mx >= x - 6 && mx <= x + width + 6 && my >= y && my <= y + height + 8);

    if (is_hovered && vira_gui_is_mouse_down(0)) {
        float ratio = (float)(mx - x) / (float)width;
        if (ratio < 0.0f) ratio = 0.0f;
        if (ratio > 1.0f) ratio = 1.0f;
        val = min_val + ratio * (max_val - min_val);
    }

    if (text_left.len > 0) {
        vira_gui_draw_text(text_left, x, y, 14, 0xA6ADC8);
    }
    if (text_right.len > 0) {
        int32_t tw = _vira_measure_text_w(text_right);
        vira_gui_draw_text(text_right, x + width - tw, y, 14, 0x89B4FA);
    }

    vira_gui_draw_rect(x, track_y, width, 4, 0x313244);

    float active_ratio = (val - min_val) / (max_val - min_val);
    if (active_ratio < 0.0f) active_ratio = 0.0f;
    if (active_ratio > 1.0f) active_ratio = 1.0f;
    int fill_w = (int)(active_ratio * (float)width);
    if (fill_w > 0) {
        vira_gui_draw_rect(x, track_y, fill_w, 4, 0x89B4FA);
    }

    vira_gui_draw_circle(x + fill_w, track_y + 2, 7.0f, is_hovered ? 0xB4BEFE : 0x89B4FA);
    return val;
}

static inline bool vira_gui_checkbox(int32_t x, int32_t y, int32_t width, int32_t height,
                                     vira_string label, bool checked) {
    int box_size = 18;
    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();
    bool is_hovered = (mx >= x && mx <= x + width && my >= y && my <= y + height);

    if (is_hovered && _vira_gui_mouse_clicked_left()) {
        checked = !checked;
    }

    int box_y = y + (height - box_size) / 2;
    vira_gui_draw_rect(x, box_y, box_size, box_size, 0x313244);
    if (checked) {
        vira_gui_draw_rect(x + 3, box_y + 3, box_size - 6, box_size - 6, 0xA6E3A1);
    }
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

static inline vira_string vira_gui_text_input(int32_t x, int32_t y, int32_t width, int32_t height,
                                              vira_string placeholder, vira_string text) {
    if (_vira_input_counter >= VIRA_MAX_TEXT_INPUTS) {
        return text;
    }
    int32_t id = _vira_input_counter++;
    vira_text_input_state_t* state = &_vira_text_inputs[id];

    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();
    bool is_hovered = (mx >= x && mx <= x + width && my >= y && my <= y + height);

    if (_vira_gui_mouse_clicked_left()) {
        if (is_hovered) {
            _vira_active_input_id = id;
        } else if (_vira_active_input_id == id) {
            _vira_active_input_id = -1;
        }
    }

    // Sync from external text if not initialized or if external change occurred while not focused
    if (!state->initialized) {
        state->len = 0;
        if (text.len > 0 && text.data) {
            int32_t copy_len = text.len < (VIRA_TEXT_INPUT_MAX_LEN - 1) ? text.len : (VIRA_TEXT_INPUT_MAX_LEN - 1);
            memcpy(state->buffer, text.data, (size_t)copy_len);
            state->len = copy_len;
        }
        state->buffer[state->len] = '\0';
        state->initialized = true;
    } else if (_vira_active_input_id != id) {
        if (text.len != state->len || (text.len > 0 && text.data && memcmp(state->buffer, text.data, (size_t)text.len) != 0)) {
            state->len = 0;
            if (text.len > 0 && text.data) {
                int32_t copy_len = text.len < (VIRA_TEXT_INPUT_MAX_LEN - 1) ? text.len : (VIRA_TEXT_INPUT_MAX_LEN - 1);
                memcpy(state->buffer, text.data, (size_t)copy_len);
                state->len = copy_len;
            }
            state->buffer[state->len] = '\0';
        }
    }

    bool is_focused = (_vira_active_input_id == id);

    // Keystroke handling
    if (is_focused) {
        if (_vira_backspace_pressed && state->len > 0) {
            state->len--;
            state->buffer[state->len] = '\0';
        }
        for (int i = 0; i < _vira_char_queue_count; i++) {
            if (state->len < VIRA_TEXT_INPUT_MAX_LEN - 1) {
                state->buffer[state->len++] = _vira_char_queue[i];
                state->buffer[state->len] = '\0';
            }
        }
    }

    // Modern Catppuccin styling
    int32_t bg_color     = is_focused ? 0x181825 : (is_hovered ? 0x1E1E2E : 0x11111B);
    int32_t border_color = is_focused ? 0x89B4FA : (is_hovered ? 0x585B70 : 0x313244);

    vira_gui_draw_rect(x, y, width, height, bg_color);
    vira_gui_draw_line(x, y, x + width, y, border_color);
    vira_gui_draw_line(x, y + height, x + width, y + height, border_color);
    vira_gui_draw_line(x, y, x, y + height, border_color);
    vira_gui_draw_line(x + width, y, x + width, y + height, border_color);

    int text_y = y + (height - 16) / 2;
    int pad_x = 10;

    if (state->len > 0) {
        vira_string s_val = (vira_string){ .data = (const uint8_t*)state->buffer, .len = state->len, .is_lit = true };
        vira_gui_draw_text(s_val, x + pad_x, text_y, 15, 0xCDD6F4);

        if (is_focused && ((_vira_cursor_blink_counter / 30) % 2 == 0)) {
            int tw = _vira_measure_text_w(s_val);
            vira_gui_draw_line(x + pad_x + tw + 2, text_y, x + pad_x + tw + 2, text_y + 16, 0x89B4FA);
        }
    } else {
        if (placeholder.len > 0) {
            vira_gui_draw_text(placeholder, x + pad_x, text_y, 15, 0x6C7086);
        }
        if (is_focused && ((_vira_cursor_blink_counter / 30) % 2 == 0)) {
            vira_gui_draw_line(x + pad_x, text_y, x + pad_x, text_y + 16, 0x89B4FA);
        }
    }

    return (vira_string){ .data = (const uint8_t*)state->buffer, .len = state->len, .is_lit = true };
}

static inline void vira_gui_progress_bar(int32_t x, int32_t y, int32_t width, int32_t height, float percent) {
    if (width <= 0 || height <= 0) return;
    if (percent > 1.0f) percent = percent / 100.0f;
    if (percent < 0.0f) percent = 0.0f;
    if (percent > 1.0f) percent = 1.0f;

    vira_gui_draw_rect(x, y, width, height, 0x11111B);
    vira_gui_draw_line(x, y, x + width, y, 0x313244);
    vira_gui_draw_line(x, y + height, x + width, y + height, 0x313244);
    vira_gui_draw_line(x, y, x, y + height, 0x313244);
    vira_gui_draw_line(x + width, y, x + width, y + height, 0x313244);

    int fill_w = (int)(percent * (float)(width - 4));
    if (fill_w > 0) {
        vira_gui_draw_rect(x + 2, y + 2, fill_w, height - 4, 0xA6E3A1);
    }

    if (height >= 18) {
        char buf[16];
        int pct_int = (int)(percent * 100.0f + 0.5f);
        int len = snprintf(buf, sizeof(buf), "%d%%", pct_int);
        vira_string s = (vira_string){ .data = (const uint8_t*)buf, .len = len, .is_lit = true };
        int tw = _vira_measure_text_w(s);
        int tx = x + (width - tw) / 2;
        int ty = y + (height - 14) / 2;
        vira_gui_draw_text(s, tx, ty, 13, 0xFFFFFF);
    }
}

static inline bool vira_gui_radio_button(int32_t x, int32_t y, vira_string label, bool is_selected) {
    int radius = 9;
    int cy = y + 12;
    int cx = x + radius;

    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();
    int total_w = 2 * radius + 8 + _vira_measure_text_w(label);
    bool is_hovered = (mx >= x && mx <= x + total_w && my >= y && my <= y + 24);

    if (is_hovered && _vira_gui_mouse_clicked_left()) {
        is_selected = true;
    }

    int32_t ring_color = is_selected ? 0x89B4FA : (is_hovered ? 0xB4BEFE : 0x585B70);
    vira_gui_draw_circle(cx, cy, (float)radius, ring_color);
    vira_gui_draw_circle(cx, cy, (float)(radius - 2), 0x1E1E2E);
    if (is_selected) {
        vira_gui_draw_circle(cx, cy, 4.0f, 0x89B4FA);
    }

    if (label.len > 0) {
        int32_t text_color = is_selected ? 0xFFFFFF : (is_hovered ? 0xCDD6F4 : 0xA6ADC8);
        vira_gui_draw_text(label, x + 2 * radius + 8, y + 4, 15, text_color);
    }

    return is_selected;
}

static inline void vira_gui_draw_badge(int32_t x, int32_t y, vira_string text, int32_t hex_bg, int32_t hex_fg) {
    int32_t tw = _vira_measure_text_w(text);
    int32_t bw = tw + 16;
    int32_t bh = 22;

    vira_gui_draw_rect(x, y, bw, bh, hex_bg);
    int32_t border = ((hex_bg >> 1) & 0x7F7F7F);
    vira_gui_draw_line(x, y, x + bw, y, border);
    vira_gui_draw_line(x, y + bh, x + bw, y + bh, border);
    vira_gui_draw_line(x, y, x, y + bh, border);
    vira_gui_draw_line(x + bw, y, x + bw, y + bh, border);

    vira_gui_draw_text(text, x + 8, y + 3, 14, hex_fg);
}

static inline void vira_gui_vstack_begin(int32_t x, int32_t y, int32_t width, int32_t spacing) {
    if (_vira_layout_depth >= VIRA_MAX_LAYOUT_DEPTH) return;
    vira_layout_node_t* parent = _vira_gui_cur_layout();
    if (parent != NULL) {
        if (x <= 0) x = parent->cur_x;
        if (y <= 0) y = parent->cur_y;
        if (width <= 0) width = parent->width;
    }
    vira_layout_node_t* node = &_vira_layout_stack[_vira_layout_depth++];
    node->type = VIRA_LAYOUT_VERTICAL;
    node->x = x;
    node->y = y;
    node->cur_x = x;
    node->cur_y = y;
    node->width = width > 0 ? width : 300;
    node->height = 0;
    node->spacing = spacing >= 0 ? spacing : 10;
    node->start_y = y;
    node->is_card = 0;
}

static inline void vira_gui_hstack_begin(int32_t height, int32_t spacing) {
    if (_vira_layout_depth >= VIRA_MAX_LAYOUT_DEPTH) return;
    vira_layout_node_t* parent = _vira_gui_cur_layout();
    int32_t x = parent ? parent->cur_x : 20;
    int32_t y = parent ? parent->cur_y : 20;
    int32_t w = parent ? parent->width : 300;

    vira_layout_node_t* node = &_vira_layout_stack[_vira_layout_depth++];
    node->type = VIRA_LAYOUT_HORIZONTAL;
    node->x = x;
    node->y = y;
    node->cur_x = x;
    node->cur_y = y;
    node->width = w;
    node->height = height > 0 ? height : 36;
    node->spacing = spacing >= 0 ? spacing : 10;
    node->start_y = y;
    node->is_card = 0;
}

static inline void vira_gui_stack_end(void) {
    if (_vira_layout_depth <= 0) return;
    vira_layout_node_t closed = _vira_layout_stack[--_vira_layout_depth];
    vira_layout_node_t* parent = _vira_gui_cur_layout();
    if (parent != NULL) {
        if (parent->type == VIRA_LAYOUT_VERTICAL) {
            int32_t consumed_h;
            if (closed.is_card) {
                int32_t content_h = (closed.cur_y - closed.start_y) + 42 + 15;
                consumed_h = content_h > 180 ? content_h : 180;
            } else if (closed.type == VIRA_LAYOUT_VERTICAL) {
                consumed_h = closed.cur_y - closed.start_y;
            } else {
                consumed_h = closed.height;
            }
            parent->cur_y += consumed_h + parent->spacing;
        } else if (parent->type == VIRA_LAYOUT_HORIZONTAL) {
            int32_t consumed_w = closed.is_card ? (closed.width + 30) : (closed.cur_x - closed.x);
            parent->cur_x += consumed_w + parent->spacing;
        }
    }
}

static inline void vira_gui_spacing(int32_t px) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return;
    if (node->type == VIRA_LAYOUT_VERTICAL) {
        node->cur_y += px;
    } else {
        node->cur_x += px;
    }
}

static inline void vira_gui_card_begin(vira_string title, int32_t width) {
    vira_layout_node_t* parent = _vira_gui_cur_layout();
    int32_t x = parent ? parent->cur_x : 40;
    int32_t y = parent ? parent->cur_y : 90;
    int32_t w = width > 0 ? width : (parent ? parent->width : 340);

    int32_t c_idx = _vira_card_counter++;
    int32_t card_h = (c_idx < 16 && _vira_card_heights[c_idx] > 0) ? _vira_card_heights[c_idx] : 220;

    vira_gui_draw_rect(x, y, w, card_h, 0x1E1E2E);
    vira_gui_draw_line(x, y, x + w, y, 0x313244);
    vira_gui_draw_line(x, y + card_h, x + w, y + card_h, 0x313244);
    vira_gui_draw_line(x, y, x, y + card_h, 0x313244);
    vira_gui_draw_line(x + w, y, x + w, y + card_h, 0x313244);

    if (title.len > 0) {
        vira_gui_draw_text(title, x + 15, y + 12, 16, 0x89B4FA);
    }

    vira_gui_vstack_begin(x + 15, y + 42, w - 30, 10);
    _vira_layout_stack[_vira_layout_depth - 1].is_card = 1;
}

static inline void vira_gui_card_end(void) {
    if (_vira_layout_depth <= 0) return;
    vira_layout_node_t closed = _vira_layout_stack[_vira_layout_depth - 1];
    int32_t c_idx = _vira_card_end_counter++;
    int32_t h = (closed.cur_y - closed.start_y) + 42 + 15;
    if (c_idx < 16) {
        _vira_card_heights[c_idx] = (h > 180) ? h : 180;
    }
    vira_gui_stack_end();
}

static inline bool vira_gui_btn(vira_string label) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return false;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        int32_t w = 140;
        int32_t h = node->height;
        bool clicked = vira_gui_button(x, y, w, h, label);
        node->cur_x += w + node->spacing;
        return clicked;
    } else {
        int32_t w = node->width;
        int32_t h = 36;
        bool clicked = vira_gui_button(x, y, w, h, label);
        node->cur_y += h + node->spacing;
        return clicked;
    }
}

static inline bool vira_gui_check(vira_string label, bool checked) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return checked;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t w = node->width;
    int32_t h = (node->type == VIRA_LAYOUT_HORIZONTAL) ? node->height : 26;
    bool result = vira_gui_checkbox(x, y, w, h, label, checked);
    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        node->cur_x += w + node->spacing;
    } else {
        node->cur_y += h + node->spacing;
    }
    return result;
}

static inline float vira_gui_slide(vira_string text_left, vira_string text_right, float val, float min_val, float max_val) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return val;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t w = node->width;
    bool has_labels = (text_left.len > 0 || text_right.len > 0);
    int32_t h = (node->type == VIRA_LAYOUT_HORIZONTAL) ? node->height : (has_labels ? 36 : 25);
    float result = vira_gui_slider(x, y, w, h, text_left, text_right, val, min_val, max_val);
    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        node->cur_x += w + node->spacing;
    } else {
        node->cur_y += h + node->spacing;
    }
    return result;
}

static inline void vira_gui_lbl(vira_string text) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t h = 22;
    vira_gui_draw_text(text, x, y, 16, 0xCDD6F4);
    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        node->cur_x += _vira_measure_text_w(text) + node->spacing;
    } else {
        node->cur_y += h + node->spacing;
    }
}

static inline vira_string vira_gui_input(vira_string placeholder, vira_string text) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return text;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t h = (node->type == VIRA_LAYOUT_HORIZONTAL) ? node->height : 34;
    int32_t w = (node->type == VIRA_LAYOUT_HORIZONTAL) ? 200 : node->width;

    vira_string result = vira_gui_text_input(x, y, w, h, placeholder, text);

    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        node->cur_x += w + node->spacing;
    } else {
        node->cur_y += h + node->spacing;
    }
    return result;
}

static inline void vira_gui_progress(float percent) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t h = (node->type == VIRA_LAYOUT_HORIZONTAL) ? node->height : 22;
    int32_t w = (node->type == VIRA_LAYOUT_HORIZONTAL) ? 160 : node->width;

    vira_gui_progress_bar(x, y, w, h, percent);

    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        node->cur_x += w + node->spacing;
    } else {
        node->cur_y += h + node->spacing;
    }
}

static inline bool vira_gui_radio(vira_string label, bool is_selected) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return is_selected;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t h = (node->type == VIRA_LAYOUT_HORIZONTAL) ? node->height : 26;

    bool result = vira_gui_radio_button(x, y, label, is_selected);

    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        int32_t item_w = 18 + 8 + _vira_measure_text_w(label);
        node->cur_x += item_w + node->spacing;
    } else {
        node->cur_y += h + node->spacing;
    }
    return result;
}

static inline void vira_gui_separator(void) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return;
    if (node->type == VIRA_LAYOUT_VERTICAL) {
        int32_t y = node->cur_y + 4;
        vira_gui_draw_line(node->cur_x, y, node->cur_x + node->width, y, 0x313244);
        node->cur_y += 10 + node->spacing;
    } else {
        int32_t x = node->cur_x + 4;
        vira_gui_draw_line(x, node->cur_y, x, node->cur_y + node->height, 0x313244);
        node->cur_x += 10 + node->spacing;
    }
}

static inline void vira_gui_badge(vira_string text, int32_t hex_bg, int32_t hex_fg) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t tw = _vira_measure_text_w(text);
    int32_t bw = tw + 16;

    vira_gui_draw_badge(x, y, text, hex_bg, hex_fg);

    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        node->cur_x += bw + node->spacing;
    } else {
        node->cur_y += 24 + node->spacing;
    }
}

static inline bool vira_gui_tab_item(vira_string label, bool is_active) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return is_active;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t tw = _vira_measure_text_w(label);
    int32_t w = tw + 28;
    int32_t h = (node->type == VIRA_LAYOUT_HORIZONTAL) ? node->height : 36;

    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();
    bool is_hovered = (mx >= x && mx <= x + w && my >= y && my <= y + h);
    bool clicked = is_hovered && _vira_gui_mouse_clicked_left();

    int32_t bg_color   = is_active ? 0x313244 : (is_hovered ? 0x242537 : 0x181825);
    int32_t text_color = is_active ? 0xFFFFFF : (is_hovered ? 0xCDD6F4 : 0x6C7086);

    vira_gui_draw_rect(x, y, w, h, bg_color);
    if (is_active) {
        vira_gui_draw_rect(x, y + h - 3, w, 3, 0x89B4FA);
    }
    vira_gui_draw_text(label, x + 14, y + (h - 16) / 2, 15, text_color);

    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        node->cur_x += w + node->spacing;
    } else {
        node->cur_y += h + node->spacing;
    }
    return clicked;
}

static inline void vira_gui_modal_begin(vira_string title, int32_t width, int32_t height) {
    int32_t win_w = vira_gui_window_width();
    int32_t win_h = vira_gui_window_height();

    // Dim the entire background with dark backdrop
    vira_gui_draw_rect(0, 0, win_w, win_h, 0x0C0C14);

    int32_t mx = (win_w - width) / 2;
    int32_t my = (win_h - height) / 2;
    if (mx < 20) mx = 20;
    if (my < 20) my = 20;

    vira_gui_draw_rect(mx, my, width, height, 0x1E1E2E);
    vira_gui_draw_line(mx, my, mx + width, my, 0x89B4FA);
    vira_gui_draw_line(mx, my + height, mx + width, my + height, 0x585B70);
    vira_gui_draw_line(mx, my, mx, my + height, 0x89B4FA);
    vira_gui_draw_line(mx + width, my, mx + width, my + height, 0x585B70);

    vira_gui_draw_rect(mx, my, width, 38, 0x181825);
    vira_gui_draw_line(mx, my + 38, mx + width, my + 38, 0x313244);
    if (title.len > 0) {
        vira_gui_draw_text(title, mx + 16, my + 10, 16, 0xCDD6F4);
    }

    vira_gui_vstack_begin(mx + 16, my + 50, width - 32, 10);
    _vira_layout_stack[_vira_layout_depth - 1].is_card = 0;
}

static inline void vira_gui_modal_end(void) {
    vira_gui_stack_end();
}

// ==============================================================================
// 4. AUTOMATIC TEMPORARY STRING FREE WRAPPERS (ZERO LEAK GUARANTEE)
// ==============================================================================

static inline void vira_gui_lbl_temp(vira_string text) {
    vira_gui_lbl(text);
    vira_str_free(&text);
}

static inline void vira_gui_draw_text_temp(vira_string text, int32_t x, int32_t y, int32_t size, int32_t hex_color) {
    vira_gui_draw_text(text, x, y, size, hex_color);
    vira_str_free(&text);
}

static inline void vira_gui_card_begin_temp(vira_string title, int32_t width) {
    vira_gui_card_begin(title, width);
    vira_str_free(&title);
}

static inline void vira_gui_badge_temp(vira_string text, int32_t hex_bg, int32_t hex_fg) {
    vira_gui_badge(text, hex_bg, hex_fg);
    vira_str_free(&text);
}

static inline bool vira_gui_btn_temp(vira_string label) {
    bool res = vira_gui_btn(label);
    vira_str_free(&label);
    return res;
}

static inline bool vira_gui_tab_item_temp(vira_string label, bool is_active) {
    bool res = vira_gui_tab_item(label, is_active);
    vira_str_free(&label);
    return res;
}

static inline void vira_gui_modal_begin_temp(vira_string title, int32_t width, int32_t height) {
    vira_gui_modal_begin(title, width, height);
    vira_str_free(&title);
}

#endif // VIRA_GUI_H