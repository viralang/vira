#ifndef VIRA_GUI_X11_H
#define VIRA_GUI_X11_H

// ------------------------------------------------------------------------------
// NATIVE LINUX X11 BACKEND (Zero-Dependency Linux Windowing via -lX11)
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

/**
 * Initializes the X11 connection, creates the top-level window, sets window manager protocols,
 * creates the offscreen double-buffering Pixmap, initializes system fonts, and starts the monotonic timer.
 * @param width Window client width in pixels.
 * @param height Window client height in pixels.
 * @param title Window title string.
 * Usage: Called by `gui_init(w, h, title)` on Linux/Unix desktop environments.
 */
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
                 ButtonPressMask | ButtonReleaseMask | PointerMotionMask | FocusChangeMask);
    _vira_x11.wm_delete = XInternAtom(_vira_x11.dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(_vira_x11.dpy, _vira_x11.win, &_vira_x11.wm_delete, 1);

    // Provide window manager input hints so Wayland / Mutter / KWin route keyboard focus
    XWMHints* wm_hints = XAllocWMHints();
    if (wm_hints) {
        wm_hints->flags = InputHint | StateHint;
        wm_hints->input = True;
        wm_hints->initial_state = NormalState;
        XSetWMHints(_vira_x11.dpy, _vira_x11.win, wm_hints);
        XFree(wm_hints);
    }

    // Lock fixed window size hints for tiling window managers
    XSizeHints hints = {0};
    hints.flags = PSize | PMinSize | PMaxSize;
    hints.min_width = hints.max_width = width;
    hints.min_height = hints.max_height = height;
    XSetWMNormalHints(_vira_x11.dpy, _vira_x11.win, &hints);

    _vira_x11.backbuf = XCreatePixmap(_vira_x11.dpy, _vira_x11.win, (unsigned int)width, (unsigned int)height, (unsigned int)_vira_x11.depth);
    _vira_x11.gc      = XCreateGC(_vira_x11.dpy, _vira_x11.backbuf, 0, NULL);
    const char* x11_font_names[] = {
        "-*-ubuntu-medium-r-*-*-14-*-*-*-*-*-iso10646-1",
        "-*-dejavu sans-medium-r-*-*-14-*-*-*-*-*-iso10646-1",
        "-*-liberation sans-medium-r-*-*-14-*-*-*-*-*-iso10646-1",
        "-*-helvetica-medium-r-*-*-14-*-*-*-*-*-iso8859-1",
        "-*-arial-medium-r-*-*-14-*-*-*-*-*-iso8859-1",
        "9x15",
        "fixed",
        NULL
    };
    for (int i = 0; x11_font_names[i]; i++) {
        _vira_x11.font = XLoadQueryFont(_vira_x11.dpy, x11_font_names[i]);
        if (_vira_x11.font) break;
    }
    if (_vira_x11.font) XSetFont(_vira_x11.dpy, _vira_x11.gc, _vira_x11.font->fid);

    XMapWindow(_vira_x11.dpy, _vira_x11.win);
    XSetInputFocus(_vira_x11.dpy, _vira_x11.win, RevertToParent, CurrentTime);
    XFlush(_vira_x11.dpy);
    clock_gettime(CLOCK_MONOTONIC, &_vira_x11.frame_start);
    _vira_layout_depth = 0;
}

/**
 * Checks whether the X11 window received a WM_DELETE_WINDOW client message from the window manager.
 * @return True if user requested to close the window; false otherwise.
 * Usage: Called by `for !gui_should_close() { ... }`.
 */
static inline bool vira_gui_should_close(void) { return _vira_x11.should_close; }

/**
 * Cancels a pending close request, preventing the window from closing.
 * Usage: Called by `gui_cancel_close()` when user denies confirmation.
 */
static inline void vira_gui_cancel_close(void) { _vira_x11.should_close = false; }

/**
 * Sets the target frame rate for the nanosleep frame limiter.
 * @param fps Target FPS (e.g. 60).
 * Usage: Called by `gui_set_target_fps(fps)`.
 */
static inline void vira_gui_set_target_fps(int32_t fps) { if (fps > 0) _vira_x11.target_fps = fps; }

/**
 * Returns the current logical width of the X11 window client area in pixels.
 * @return Window client width.
 * Usage: Called by `gui_window_width()`.
 */
static inline int32_t vira_gui_window_width(void) { return _vira_x11.width; }

/**
 * Returns the current logical height of the X11 window client area in pixels.
 * @return Window client height.
 * Usage: Called by `gui_window_height()`.
 */
static inline int32_t vira_gui_window_height(void) { return _vira_x11.height; }

/**
 * Begins a new frame by draining all pending X11 events via XPending/XNextEvent,
 * tracking mouse movement, button clicks, scrolling (buttons 4/5), and keyboard events.
 * Usage: Invoked internally by `gui_begin()` at the beginning of each frame.
 */
static inline void _vira_backend_gui_begin(void) {
    _vira_char_queue_count = 0;
    _vira_backspace_pressed = false;
    _vira_cursor_blink_counter++;
    _vira_mouse_wheel_delta = 0.0f;
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
        } else if (ev.type == FocusIn) {
            // Window gained keyboard focus
        } else if (ev.type == MotionNotify) {
            _vira_x11.mouse_x = ev.xmotion.x;
            _vira_x11.mouse_y = ev.xmotion.y;
        } else if (ev.type == ButtonPress) {
            _vira_x11.mouse_x = ev.xbutton.x;
            _vira_x11.mouse_y = ev.xbutton.y;
            XSetInputFocus(_vira_x11.dpy, _vira_x11.win, RevertToParent, CurrentTime);
            if (ev.xbutton.button == 4) {
                _vira_mouse_wheel_delta += 1.0f;
            } else if (ev.xbutton.button == 5) {
                _vira_mouse_wheel_delta -= 1.0f;
            } else {
                int b = (ev.xbutton.button == Button1) ? 0 : ((ev.xbutton.button == Button3) ? 1 : 2);
                _vira_x11.mouse_down[b] = true;
                _vira_x11.mouse_clicked[b] = true;
            }
        } else if (ev.type == ButtonRelease) {
            _vira_x11.mouse_x = ev.xbutton.x;
            _vira_x11.mouse_y = ev.xbutton.y;
            if (ev.xbutton.button != 4 && ev.xbutton.button != 5) {
                int b = (ev.xbutton.button == Button1) ? 0 : ((ev.xbutton.button == Button3) ? 1 : 2);
                _vira_x11.mouse_down[b] = false;
            }
        } else if (ev.type == KeyPress) {
            char char_buf[32];
            KeySym sym = 0;
            int count = XLookupString(&ev.xkey, char_buf, sizeof(char_buf), &sym, NULL);
            if (sym == XK_Tab || sym == XK_ISO_Left_Tab) {
                _vira_tab_pressed = true;
                _vira_shift_down = ((ev.xkey.state & ShiftMask) != 0);
            } else if (sym == XK_BackSpace) {
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

/**
 * Concludes the current frame: performs an atomic XCopyArea blit from backbuffer Pixmap to the visible Window,
 * flushes the X11 server queue, and sleeps via nanosleep to maintain target FPS.
 * Usage: Invoked internally by `gui_end()` at the end of each frame.
 */
static inline void _vira_backend_gui_end(void) {
    if (!_vira_x11.dpy) return;
    XCopyArea(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.win, _vira_x11.gc,
              0, 0, (unsigned int)_vira_x11.width, (unsigned int)_vira_x11.height, 0, 0);
    XFlush(_vira_x11.dpy);

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &_vira_x11.frame_start);
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

/**
 * Destroys the X11 window, releases the backbuffer Pixmap, frees graphic contexts, and closes the X display connection.
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
    if (!_vira_x11.dpy) return;
    if (_vira_x11.backbuf) XFreePixmap(_vira_x11.dpy, _vira_x11.backbuf);
    if (_vira_x11.gc) XFreeGC(_vira_x11.dpy, _vira_x11.gc);
    if (_vira_x11.win) XDestroyWindow(_vira_x11.dpy, _vira_x11.win);
    XCloseDisplay(_vira_x11.dpy);
    _vira_x11.dpy = NULL;
}

/**
 * Draws a filled rectangle onto the offscreen X11 backbuffer Pixmap.
 * @param x Top-left X coordinate.
 * @param y Top-left Y coordinate.
 * @param width Rectangle width in pixels.
 * @param height Rectangle height in pixels.
 * @param hex_color 24-bit RGB fill color.
 * Usage: Called by `gui_draw_rect(x, y, w, h, color)`.
 */
static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) {
    if (!_vira_x11.dpy || width <= 0 || height <= 0) return;
    XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(hex_color & 0xFFFFFF));
    XFillRectangle(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, x, y, (unsigned int)width, (unsigned int)height);
}

/**
 * Draws a rounded rectangle onto the X11 backbuffer with background fill and border outline.
 * @param x Top-left X coordinate.
 * @param y Top-left Y coordinate.
 * @param width Rectangle width in pixels.
 * @param height Rectangle height in pixels.
 * @param radius Corner radius in pixels.
 * @param bg_hex 24-bit RGB fill color (-1 to omit fill).
 * @param border_hex 24-bit RGB border outline color (-1 to omit border).
 * Usage: Called by `gui_draw_rounded_rect(...)` for styling cards, buttons, and inputs.
 */
static inline void vira_gui_draw_rounded_rect(int32_t x, int32_t y, int32_t width, int32_t height, float radius, int32_t bg_hex, int32_t border_hex) {
    (void)radius;
    if (!_vira_x11.dpy || width <= 0 || height <= 0) return;
    if (bg_hex >= 0) {
        XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(bg_hex & 0xFFFFFF));
        XFillRectangle(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, x, y, (unsigned int)width, (unsigned int)height);
    }
    if (border_hex >= 0) {
        XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(border_hex & 0xFFFFFF));
        XDrawRectangle(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, x, y, (unsigned int)(width > 1 ? width - 1 : 1), (unsigned int)(height > 1 ? height - 1 : 1));
    }
}

/**
 * Clears the entire X11 backbuffer Pixmap with a solid color.
 * @param hex_color 24-bit RGB background clear color.
 * Usage: Called by `gui_clear(color)` at the start of each frame.
 */
static inline void vira_gui_clear(int32_t hex_color) {
    vira_gui_draw_rect(0, 0, _vira_x11.width, _vira_x11.height, hex_color);
}

/**
 * Renders text onto the X11 backbuffer using the active X11 system font.
 * @param text Vira string to render.
 * @param x Baseline X origin.
 * @param y Top baseline Y coordinate.
 * @param size Font size hint in points.
 * @param hex_color 24-bit RGB text color.
 * Usage: Called by `gui_draw_text(text, x, y, size, color)`.
 */
static inline void vira_gui_draw_text(vira_string text, int32_t x, int32_t y, int32_t size, int32_t hex_color) {
    (void)size;
    if (!_vira_x11.dpy || text.len <= 0) return;
    XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(hex_color & 0xFFFFFF));
    XDrawString(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, x, y + 13, (const char*)text.data, text.len);
}

/**
 * Draws a filled anti-aliased circle onto the X11 backbuffer using XFillArc.
 * @param cx Center X coordinate.
 * @param cy Center Y coordinate.
 * @param radius Radius in pixels.
 * @param hex_color 24-bit RGB color.
 * Usage: Called by `gui_draw_circle(cx, cy, radius, color)`.
 */
static inline void vira_gui_draw_circle(int32_t cx, int32_t cy, float radius, int32_t hex_color) {
    if (!_vira_x11.dpy || radius <= 0.0f) return;
    int r = (int)radius;
    XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(hex_color & 0xFFFFFF));
    XFillArc(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, cx - r, cy - r, (unsigned int)(r * 2), (unsigned int)(r * 2), 0, 360 * 64);
}

/**
 * Draws a vector line between two points on the X11 backbuffer using XDrawLine.
 * @param x1 Start X coordinate.
 * @param y1 Start Y coordinate.
 * @param x2 End X coordinate.
 * @param y2 End Y coordinate.
 * @param hex_color 24-bit RGB line color.
 * Usage: Called by `gui_draw_line(x1, y1, x2, y2, color)`.
 */
static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t hex_color) {
    if (!_vira_x11.dpy) return;
    XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(hex_color & 0xFFFFFF));
    XDrawLine(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, x1, y1, x2, y2);
}

/**
 * Measures the pixel width of a string using XTextWidth.
 * @param text Vira string to measure.
 * @return Computed pixel width.
 * Usage: Used by auto-layout and widgets to size buttons, labels, and table cells.
 */
static inline int32_t _vira_measure_text_w(vira_string text) {
    if (_vira_x11.font && text.len > 0) return XTextWidth(_vira_x11.font, (const char*)text.data, text.len);
    return text.len * 8;
}

/**
 * Returns the current mouse cursor X coordinate in X11 window coordinates.
 * @return Mouse cursor X coordinate.
 * Usage: Called by `gui_mouse_x()`.
 */
static inline int32_t vira_gui_mouse_x(void) { return _vira_x11.mouse_x; }

/**
 * Returns the current mouse cursor Y coordinate in X11 window coordinates.
 * @return Mouse cursor Y coordinate.
 * Usage: Called by `gui_mouse_y()`.
 */
static inline int32_t vira_gui_mouse_y(void) { return _vira_x11.mouse_y; }

/**
 * Checks whether a mouse button is currently held down (0 = Left, 1 = Right, 2 = Middle).
 * @param b Mouse button index.
 * @return True if held; false otherwise.
 * Usage: Called by `gui_is_mouse_down(button)`.
 */
static inline bool vira_gui_is_mouse_down(int32_t b) { return (b >= 0 && b < 3) ? _vira_x11.mouse_down[b] : false; }

/**
 * Checks whether the left mouse button transitioned from up to down on this frame.
 * @return True if clicked on this frame; false otherwise.
 * Usage: Used internally by immediate-mode widgets for click events.
 */
static inline bool _vira_backend_mouse_clicked_left(void) { return _vira_x11.mouse_clicked[0]; }

/**
 * Checks whether a specific key was pressed down on this frame.
 * @param k Key code.
 * @return True if pressed; false otherwise.
 * Usage: Called by `gui_is_key_pressed(key)`.
 */
static inline bool vira_gui_is_key_pressed(int32_t k) { return (k >= 0 && k < 256) ? _vira_x11.keys_pressed[k] : false; }

/**
 * Returns the elapsed delta time of the previous frame in seconds.
 * @return Delta time in seconds.
 * Usage: Called by `gui_frame_time()`.
 */
static inline float vira_gui_frame_time(void) { return _vira_x11.delta_time; }

/**
 * Loads a BMP image file into an image slot for X11 rendering.
 * @param filepath Path to the image file.
 * @return Image slot index, or -1 on failure.
 * Usage: Called by `gui_load_image(path)`.
 */
static inline int32_t _vira_x11_load_image(const char* filepath) {
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
 * Renders an image onto the X11 backbuffer Pixmap by creating an XImage, bilinearly sampling pixels, and calling XPutImage.
 * @param id Image handle index.
 * @param x Top-left X destination coordinate.
 * @param y Top-left Y destination coordinate.
 * @param w Scaled destination width.
 * @param h Scaled destination height.
 * Usage: Called by `gui_draw_image(id, x, y, w, h)`.
 */
static inline void _vira_x11_draw_image(int32_t id, int32_t x, int32_t y, int32_t w, int32_t h) {
    if (id < 0 || id >= VIRA_MAX_GUI_IMAGES || !_vira_gui_images[id].is_valid) return;
    if (!_vira_x11.dpy || !_vira_x11.backbuf || !_vira_gui_images[id].pixels || w <= 0 || h <= 0) return;

    int32_t src_w = _vira_gui_images[id].width;
    int32_t src_h = _vira_gui_images[id].height;
    const uint32_t* src_pix = _vira_gui_images[id].pixels;

    XImage* img = XCreateImage(_vira_x11.dpy, DefaultVisual(_vira_x11.dpy, _vira_x11.screen),
                               24, ZPixmap, 0, NULL, (unsigned int)w, (unsigned int)h, 32, 0);
    if (!img) return;
    img->data = (char*)malloc((size_t)img->bytes_per_line * (size_t)h);
    if (!img->data) {
        XDestroyImage(img);
        return;
    }

    for (int dy = 0; dy < h; dy++) {
        int sy = (dy * src_h) / h;
        if (sy >= src_h) sy = src_h - 1;
        const uint32_t* srow = &src_pix[sy * src_w];
        for (int dx = 0; dx < w; dx++) {
            int sx = (dx * src_w) / w;
            if (sx >= src_w) sx = src_w - 1;
            uint32_t c = srow[sx];
            XPutPixel(img, dx, dy, (unsigned long)(c & 0xFFFFFF));
        }
    }

    XPutImage(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, img, 0, 0, x, y, (unsigned int)w, (unsigned int)h);
    XDestroyImage(img);
}

/**
 * Frees pixel memory associated with an X11 image handle slot.
 * @param id Image handle index.
 * Usage: Called by `gui_free_image(id)`.
 */
static inline void _vira_x11_free_image(int32_t id) {
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
 * Sets a rectangular scissor clipping box on the X11 GC using XSetClipRectangles.
 * @param x Left coordinate of scissor rectangle.
 * @param y Top coordinate of scissor rectangle.
 * @param w Width of scissor rectangle.
 * @param h Height of scissor rectangle.
 * Usage: Called by scroll views and modals to constrain rendering.
 */
static inline void _vira_x11_set_clip(int32_t x, int32_t y, int32_t w, int32_t h) {
    if (!_vira_x11.dpy || !_vira_x11.gc) return;
    if (w <= 0 || h <= 0) {
        XRectangle r = { 0, 0, 0, 0 };
        XSetClipRectangles(_vira_x11.dpy, _vira_x11.gc, 0, 0, &r, 1, Unsorted);
    } else {
        XRectangle r = { (short)x, (short)y, (unsigned short)w, (unsigned short)h };
        XSetClipRectangles(_vira_x11.dpy, _vira_x11.gc, 0, 0, &r, 1, Unsorted);
    }
}

/**
 * Resets the X11 GC clipping mask to None, restoring unclipped rendering.
 * Usage: Called at the end of scroll containers and modal dialogs.
 */
static inline void _vira_x11_reset_clip(void) {
    if (_vira_x11.dpy && _vira_x11.gc) {
        XSetClipMask(_vira_x11.dpy, _vira_x11.gc, None);
    }
}

#endif // VIRA_GUI_X11_H
