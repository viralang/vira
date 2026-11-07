#ifndef VIRA_GUI_RAYLIB_H
#define VIRA_GUI_RAYLIB_H

// ------------------------------------------------------------------------------
// HARDWARE ACCELERATED RAYLIB BACKEND (macOS, Windows & Linux GPU)
// ------------------------------------------------------------------------------
#include <raylib.h>
#include <math.h>

static bool    _vira_rl_should_close = false;
static int32_t _vira_rl_win_w = 800;
static int32_t _vira_rl_win_h = 600;
static Font    _vira_rl_font = {0};
static bool    _vira_rl_font_loaded = false;

/**
 * Converts a 24-bit RGB hex color to a Raylib Color struct with alpha = 255.
 * @param hex 24-bit RGB hex value (e.g., 0x89B4FA).
 * @return Raylib Color structure.
 * Usage: Used internally for Raylib drawing primitives.
 */
static inline Color _vira_rl_color(int32_t hex) {
    return (Color){
        .r = (unsigned char)((hex >> 16) & 0xFF),
        .g = (unsigned char)((hex >> 8) & 0xFF),
        .b = (unsigned char)(hex & 0xFF),
        .a = 255
    };
}

/**
 * Probes the operating system's standard font directories (Ubuntu, DejaVu, Liberation, Segoe UI, SFNS)
 * and loads a high-resolution 36px font atlas with bilinear filtering for sharp vector-quality text rendering.
 * Usage: Invoked automatically during `vira_gui_init` under the Raylib backend.
 */
static inline void _vira_rl_load_system_font(void) {
    if (_vira_rl_font_loaded) return;
    const char* cand_paths[] = {
        // Linux desktop system fonts
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        // Project local fallback fonts
        "assets/font.ttf",
        "fonts/font.ttf",
        // Windows fonts
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\tahoma.ttf",
        // macOS fonts
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        NULL
    };
    for (int i = 0; cand_paths[i]; i++) {
        if (FileExists(cand_paths[i])) {
            // High-resolution 36px font atlas for razor-sharp rendering at all UI sizes
            _vira_rl_font = LoadFontEx(cand_paths[i], 36, NULL, 0);
            if (_vira_rl_font.texture.id > 0) {
                SetTextureFilter(_vira_rl_font.texture, TEXTURE_FILTER_BILINEAR);
                _vira_rl_font_loaded = true;
                return;
            }
        }
    }
}

/**
 * Initializes the Raylib window, sets MSAA 4X anti-aliasing and HighDPI hints, sets target FPS, and loads fonts.
 * @param width Window client width in pixels.
 * @param height Window client height in pixels.
 * @param title Window title text.
 * Usage: Called by `gui_init(w, h, title)` when compiling with VIRA_USE_RAYLIB.
 */
static inline void vira_gui_init(int32_t width, int32_t height, vira_string title) {
    if (IsWindowReady()) return;
    _vira_rl_win_w = width;
    _vira_rl_win_h = height;
    _vira_rl_should_close = false;

    char stack_title[256];
    int32_t tlen = title.len < 255 ? title.len : 255;
    if (tlen > 0 && title.data) memcpy(stack_title, title.data, (size_t)tlen);
    stack_title[tlen] = '\0';

    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIGHDPI | FLAG_VSYNC_HINT);
    InitWindow(width, height, stack_title);
    SetTargetFPS(60);
    _vira_rl_load_system_font();
}

/**
 * Checks whether the Raylib window has received a close request from the OS or user.
 * @return True if window should close; false otherwise.
 * Usage: Main event loop condition in `for !gui_should_close() { ... }`.
 */
static inline bool vira_gui_should_close(void) {
    return _vira_rl_should_close || WindowShouldClose();
}

/**
 * Cancels a pending window close request.
 * Usage: Called by `gui_cancel_close()` when intercepting close confirmation modals.
 */
static inline void vira_gui_cancel_close(void) {
    _vira_rl_should_close = false;
}

/**
 * Sets the target frame rate in Raylib.
 * @param fps Target FPS (e.g., 60).
 * Usage: Called by `gui_set_target_fps(fps)`.
 */
static inline void vira_gui_set_target_fps(int32_t fps) {
    if (fps > 0) SetTargetFPS(fps);
}

/**
 * Returns the current width of the Raylib window.
 * @return Width in pixels.
 * Usage: Called by `gui_window_width()`.
 */
static inline int32_t vira_gui_window_width(void) {
    return IsWindowReady() ? GetScreenWidth() : _vira_rl_win_w;
}

/**
 * Returns the current height of the Raylib window.
 * @return Height in pixels.
 * Usage: Called by `gui_window_height()`.
 */
static inline int32_t vira_gui_window_height(void) {
    return IsWindowReady() ? GetScreenHeight() : _vira_rl_win_h;
}

/**
 * Begins a new frame: collects mouse wheel delta, polls character input queue, and calls BeginDrawing().
 * Usage: Invoked internally at the start of `gui_begin()`.
 */
static inline void _vira_backend_gui_begin(void) {
    _vira_char_queue_count = 0;
    _vira_backspace_pressed = false;
    _vira_cursor_blink_counter++;
    _vira_mouse_wheel_delta = GetMouseWheelMove();

    if (IsKeyPressed(KEY_BACKSPACE)) {
        _vira_backspace_pressed = true;
    }
    if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_ESCAPE)) {
        _vira_active_input_id = -1;
    }
    int key = GetCharPressed();
    while (key > 0) {
        if (key >= 32 && key <= 126 && _vira_char_queue_count < VIRA_CHAR_QUEUE_SIZE) {
            _vira_char_queue[_vira_char_queue_count++] = (char)key;
        }
        key = GetCharPressed();
    }

    BeginDrawing();
}

/**
 * Concludes the current frame: calls Raylib EndDrawing(), swapping double-buffers and maintaining frame timing.
 * Usage: Invoked internally at the end of `gui_end()`.
 */
static inline void _vira_backend_gui_end(void) {
    EndDrawing();
}

/**
 * Cleans up Raylib textures, unloads font atlases, and closes the window.
 * Usage: Called by `gui_close()` at application exit.
 */
static inline void vira_gui_close(void) {
    if (_vira_rl_font_loaded && _vira_rl_font.texture.id > 0) {
        UnloadFont(_vira_rl_font);
        _vira_rl_font_loaded = false;
        _vira_rl_font = (Font){0};
    }
    for (int i = 0; i < VIRA_MAX_GUI_IMAGES; i++) {
        if (_vira_gui_images[i].is_valid && _vira_gui_images[i].native_handle) {
            Texture2D tex = {
                .id = (unsigned int)(intptr_t)_vira_gui_images[i].native_handle,
                .width = _vira_gui_images[i].width,
                .height = _vira_gui_images[i].height,
                .mipmaps = 1,
                .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
            };
            UnloadTexture(tex);
            _vira_gui_images[i].native_handle = NULL;
            _vira_gui_images[i].is_valid = false;
        }
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
}

/**
 * Clears the background canvas to a solid hex color.
 * @param hex_color 24-bit RGB hex clear color.
 * Usage: Called by `gui_clear(color)`.
 */
static inline void vira_gui_clear(int32_t hex_color) {
    ClearBackground(_vira_rl_color(hex_color));
}

/**
 * Draws a solid filled rectangle using DrawRectangle.
 * @param x Top-left X coordinate.
 * @param y Top-left Y coordinate.
 * @param width Rectangle width in pixels.
 * @param height Rectangle height in pixels.
 * @param hex_color 24-bit RGB fill color.
 * Usage: Called by `gui_draw_rect(x, y, w, h, color)`.
 */
static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) {
    if (width <= 0 || height <= 0) return;
    DrawRectangle(x, y, width, height, _vira_rl_color(hex_color));
}

/**
 * Draws a rounded rectangle with background fill and border outline using DrawRectangleRounded.
 * @param x Top-left X coordinate.
 * @param y Top-left Y coordinate.
 * @param width Rectangle width in pixels.
 * @param height Rectangle height in pixels.
 * @param radius Corner radius in pixels.
 * @param bg_hex 24-bit RGB fill color (-1 to omit fill).
 * @param border_hex 24-bit RGB border outline color (-1 to omit border).
 * Usage: Called by `gui_draw_rounded_rect(...)` for styling buttons, cards, and text inputs.
 */
static inline void vira_gui_draw_rounded_rect(int32_t x, int32_t y, int32_t width, int32_t height, float radius, int32_t bg_hex, int32_t border_hex) {
    if (width <= 0 || height <= 0) return;
    int32_t min_dim = width < height ? width : height;
    float roundness = (float)radius / ((float)min_dim * 0.5f);
    if (roundness > 1.0f) roundness = 1.0f;
    if (roundness < 0.0f) roundness = 0.0f;
    Rectangle rect = { (float)x, (float)y, (float)width, (float)height };
    if (bg_hex >= 0) {
        DrawRectangleRounded(rect, roundness, 8, _vira_rl_color(bg_hex));
    }
    if (border_hex >= 0) {
        DrawRectangleRoundedLines(rect, roundness, 8, 1.0f, _vira_rl_color(border_hex));
    }
}

/**
 * Draws an anti-aliased circle using DrawCircle.
 * @param cx Center X coordinate.
 * @param cy Center Y coordinate.
 * @param radius Radius in pixels.
 * @param hex_color 24-bit RGB fill color.
 * Usage: Called by `gui_draw_circle(cx, cy, radius, color)`.
 */
static inline void vira_gui_draw_circle(int32_t cx, int32_t cy, float radius, int32_t hex_color) {
    if (radius <= 0.0f) return;
    DrawCircle(cx, cy, radius, _vira_rl_color(hex_color));
}

/**
 * Draws a 1-pixel vector line between two points using DrawLine.
 * @param x1 Start X coordinate.
 * @param y1 Start Y coordinate.
 * @param x2 End X coordinate.
 * @param y2 End Y coordinate.
 * @param hex_color 24-bit RGB line color.
 * Usage: Called by `gui_draw_line(x1, y1, x2, y2, color)`.
 */
static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t hex_color) {
    DrawLine(x1, y1, x2, y2, _vira_rl_color(hex_color));
}

/**
 * Renders antialiased text using the loaded system font atlas via DrawTextEx.
 * @param text Vira string to render.
 * @param x X origin coordinate.
 * @param y Y origin coordinate.
 * @param size Font size in points.
 * @param hex_color 24-bit RGB text color.
 * Usage: Called by `gui_draw_text(text, x, y, size, color)`.
 */
static inline void vira_gui_draw_text(vira_string text, int32_t x, int32_t y, int32_t size, int32_t hex_color) {
    if (text.len <= 0 || !text.data) return;
    char stack_buf[512];
    char *buf = stack_buf;
    if (text.len >= 512) {
        buf = (char *)malloc((size_t)text.len + 1);
        if (!buf) return;
    }
    memcpy(buf, text.data, (size_t)text.len);
    buf[text.len] = '\0';
    float fsize = (float)(size > 0 ? size : 15);
    if (_vira_rl_font_loaded && _vira_rl_font.texture.id > 0) {
        DrawTextEx(_vira_rl_font, buf, (Vector2){ (float)x, (float)y }, fsize, 1.0f, _vira_rl_color(hex_color));
    } else {
        DrawText(buf, x, y, (int)fsize, _vira_rl_color(hex_color));
    }
    if (buf != stack_buf) free(buf);
}

/**
 * Measures the rendered pixel width of a string using MeasureTextEx.
 * @param text Vira string to measure.
 * @return Width in pixels.
 * Usage: Used by auto-layout and widgets for element sizing.
 */
static inline int32_t _vira_measure_text_w(vira_string text) {
    if (text.len <= 0 || !text.data) return 0;
    char stack_buf[256];
    char *buf = stack_buf;
    if (text.len >= 256) {
        buf = (char *)malloc((size_t)text.len + 1);
        if (!buf) return text.len * 8;
    }
    memcpy(buf, text.data, (size_t)text.len);
    buf[text.len] = '\0';
    int32_t w = 0;
    if (_vira_rl_font_loaded && _vira_rl_font.texture.id > 0) {
        Vector2 sz = MeasureTextEx(_vira_rl_font, buf, 15.0f, 1.0f);
        w = (int32_t)ceilf(sz.x);
    } else {
        w = MeasureText(buf, 15);
    }
    if (buf != stack_buf) free(buf);
    return w;
}

/**
 * Returns the current mouse cursor X coordinate in window space via GetMouseX().
 * @return Mouse cursor X coordinate.
 * Usage: Called by `gui_mouse_x()`.
 */
static inline int32_t vira_gui_mouse_x(void) {
    return GetMouseX();
}

/**
 * Returns the current mouse cursor Y coordinate in window space via GetMouseY().
 * @return Mouse cursor Y coordinate.
 * Usage: Called by `gui_mouse_y()`.
 */
static inline int32_t vira_gui_mouse_y(void) {
    return GetMouseY();
}

/**
 * Checks whether a mouse button is held down via IsMouseButtonDown().
 * @param button Mouse button index (0 = Left, 1 = Right, 2 = Middle).
 * @return True if button is down; false otherwise.
 * Usage: Called by `gui_is_mouse_down(button)`.
 */
static inline bool vira_gui_is_mouse_down(int32_t button) {
    if (button >= 0 && button < 3) return IsMouseButtonDown(button);
    return false;
}

/**
 * Checks whether the left mouse button transitioned from up to down on this frame via IsMouseButtonPressed().
 * @return True if pressed this frame; false otherwise.
 * Usage: Used internally by immediate-mode widgets.
 */
static inline bool _vira_backend_mouse_clicked_left(void) {
    return IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

/**
 * Checks whether a key was pressed down on this frame via IsKeyPressed().
 * @param key Raylib keycode.
 * @return True if key was pressed; false otherwise.
 * Usage: Called by `gui_is_key_pressed(key)`.
 */
static inline bool vira_gui_is_key_pressed(int32_t key) {
    return IsKeyPressed(key);
}

/**
 * Returns the elapsed frame delta time in seconds via GetFrameTime().
 * @return Delta time in seconds.
 * Usage: Called by `gui_frame_time()`.
 */
static inline float vira_gui_frame_time(void) {
    return GetFrameTime();
}

/**
 * Loads an image file from disk directly into a GPU texture via Raylib LoadTexture().
 * @param filepath Path to the image file.
 * @return Image slot index, or -1 on failure.
 * Usage: Called by `gui_load_image(path)`.
 */
static inline int32_t _vira_raylib_load_image(const char* filepath) {
    if (!filepath || !FileExists(filepath)) return -1;
    Texture2D tex = LoadTexture(filepath);
    if (tex.id == 0) return -1;
    for (int i = 0; i < VIRA_MAX_GUI_IMAGES; i++) {
        if (!_vira_gui_images[i].is_valid) {
            _vira_gui_images[i].width = tex.width;
            _vira_gui_images[i].height = tex.height;
            _vira_gui_images[i].channels = 4;
            _vira_gui_images[i].pixels = NULL;
            _vira_gui_images[i].native_handle = (void*)(intptr_t)tex.id;
            _vira_gui_images[i].is_valid = true;
            return i;
        }
    }
    UnloadTexture(tex);
    return -1;
}

/**
 * Renders a loaded GPU texture scaled into the destination rectangle via DrawTexturePro().
 * @param id Image handle index.
 * @param x Top-left X destination coordinate.
 * @param y Top-left Y destination coordinate.
 * @param w Destination width.
 * @param h Destination height.
 * Usage: Called by `gui_draw_image(id, x, y, w, h)`.
 */
static inline void _vira_raylib_draw_image(int32_t id, int32_t x, int32_t y, int32_t w, int32_t h) {
    if (id < 0 || id >= VIRA_MAX_GUI_IMAGES || !_vira_gui_images[id].is_valid) return;
    Texture2D tex = {
        .id = (unsigned int)(intptr_t)_vira_gui_images[id].native_handle,
        .width = _vira_gui_images[id].width,
        .height = _vira_gui_images[id].height,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    };
    Rectangle src = { 0.0f, 0.0f, (float)tex.width, (float)tex.height };
    Rectangle dst = { (float)x, (float)y, (float)w, (float)h };
    Vector2 origin = { 0.0f, 0.0f };
    DrawTexturePro(tex, src, dst, origin, 0.0f, WHITE);
}

/**
 * Unloads a GPU texture and frees the image slot.
 * @param id Image handle index.
 * Usage: Called by `gui_free_image(id)`.
 */
static inline void _vira_raylib_free_image(int32_t id) {
    if (id < 0 || id >= VIRA_MAX_GUI_IMAGES || !_vira_gui_images[id].is_valid) return;
    Texture2D tex = {
        .id = (unsigned int)(intptr_t)_vira_gui_images[id].native_handle,
        .width = _vira_gui_images[id].width,
        .height = _vira_gui_images[id].height,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    };
    UnloadTexture(tex);
    _vira_gui_images[id].native_handle = NULL;
    _vira_gui_images[id].is_valid = false;
    _vira_gui_images[id].width = 0;
    _vira_gui_images[id].height = 0;
}

/**
 * Activates scissor clipping via BeginScissorMode().
 * @param x Left coordinate of scissor box.
 * @param y Top coordinate of scissor box.
 * @param w Width of scissor box.
 * @param h Height of scissor box.
 * Usage: Called by scroll views and modals.
 */
static inline void _vira_raylib_set_clip(int32_t x, int32_t y, int32_t w, int32_t h) {
    if (w > 0 && h > 0) BeginScissorMode(x, y, w, h);
}

/**
 * Ends scissor clipping via EndScissorMode().
 * Usage: Restores full viewport rendering after scroll views and modals.
 */
static inline void _vira_raylib_reset_clip(void) {
    EndScissorMode();
}

#endif // VIRA_GUI_RAYLIB_H
