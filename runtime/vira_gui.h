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

static bool                    _vira_tab_pressed = false;
static bool                    _vira_shift_down = false;
static int32_t                 _vira_total_inputs = 0;
static bool                    _vira_modal_active = false;
static bool                    _vira_modal_open_this_frame = false;
static bool                    _vira_modal_open_prev_frame = false;
static bool                    _vira_modal_just_opened = false;
static int32_t                 _vira_modal_first_input = -1;
static int32_t                 _vira_modal_last_input = -1;
static int32_t                 _vira_prev_modal_first_input = -1;
static int32_t                 _vira_prev_modal_last_input = -1;

static int32_t                 _vira_card_heights[16] = {0};
static int32_t                 _vira_card_counter = 0;
static int32_t                 _vira_card_end_counter = 0;

static bool                    _vira_click_consumed = false;

// Dropdown State
typedef struct {
    bool              active;
    int32_t           x;
    int32_t           y;
    int32_t           w;
    int32_t           h;
    vira_slice_string items;
    int32_t           selected_idx;
} vira_dropdown_popup_t;

static vira_dropdown_popup_t   _vira_current_dropdown_popup = {0};
static int32_t                 _vira_active_dropdown_id = -1;
static int32_t                 _vira_dropdown_counter = 0;

#define VIRA_MAX_TABLES 16
typedef struct {
    int32_t sort_col;
    bool    sort_asc;
    int32_t page;
    int32_t selected_row;
    int32_t last_search_len;
    bool    initialized;
} vira_table_state_t;

static vira_table_state_t      _vira_table_states[VIRA_MAX_TABLES] = {0};
static int32_t                 _vira_table_counter = 0;
static int32_t                 _vira_last_table_sort_col = 0;
static bool                    _vira_last_table_sort_asc = true;

// ------------------------------------------------------------------------------
// Asset & Image Subsystem Definitions
// ------------------------------------------------------------------------------
#define VIRA_MAX_GUI_IMAGES 64

typedef struct {
    int32_t   width;
    int32_t   height;
    int32_t   channels;
    uint32_t* pixels;
    void*     native_handle;
    bool      is_valid;
} vira_gui_image_t;

static vira_gui_image_t _vira_gui_images[VIRA_MAX_GUI_IMAGES] = {0};

static inline bool _vira_load_bmp_file(const char* filepath, int32_t* out_w, int32_t* out_h, uint32_t** out_pixels) {
    if (!filepath || !out_w || !out_h || !out_pixels) return false;
    FILE* f = fopen(filepath, "rb");
    if (!f) return false;

    uint8_t header[54];
    if (fread(header, 1, 54, f) < 54) { fclose(f); return false; }
    if (header[0] != 'B' || header[1] != 'M') { fclose(f); return false; }

    uint32_t data_offset = *(uint32_t*)&header[10];
    int32_t width = *(int32_t*)&header[18];
    int32_t height = *(int32_t*)&header[22];
    uint16_t bpp = *(uint16_t*)&header[28];
    uint32_t compression = *(uint32_t*)&header[30];

    if (width <= 0 || height == 0 || (bpp != 24 && bpp != 32) || (compression != 0 && compression != 3)) {
        fclose(f);
        return false;
    }

    bool flip_y = (height > 0);
    int32_t abs_h = height < 0 ? -height : height;
    size_t pixel_count = (size_t)width * (size_t)abs_h;
    uint32_t* pixels = (uint32_t*)calloc(pixel_count, sizeof(uint32_t));
    if (!pixels) { fclose(f); return false; }

    fseek(f, (long)data_offset, SEEK_SET);
    int row_stride = ((width * (bpp / 8) + 3) / 4) * 4;
    uint8_t* row_buf = (uint8_t*)malloc((size_t)row_stride);
    if (!row_buf) { free(pixels); fclose(f); return false; }

    for (int32_t row = 0; row < abs_h; row++) {
        int32_t target_row = flip_y ? (abs_h - 1 - row) : row;
        if (fread(row_buf, 1, (size_t)row_stride, f) < (size_t)row_stride) break;
        uint32_t* dst = &pixels[target_row * width];
        if (bpp == 32) {
            for (int32_t col = 0; col < width; col++) {
                uint8_t b = row_buf[col * 4 + 0];
                uint8_t g = row_buf[col * 4 + 1];
                uint8_t r = row_buf[col * 4 + 2];
                uint8_t a = row_buf[col * 4 + 3];
                dst[col] = ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
            }
        } else if (bpp == 24) {
            for (int32_t col = 0; col < width; col++) {
                uint8_t b = row_buf[col * 3 + 0];
                uint8_t g = row_buf[col * 3 + 1];
                uint8_t r = row_buf[col * 3 + 2];
                dst[col] = (0xFF << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
            }
        }
    }

    free(row_buf);
    fclose(f);
    *out_w = width;
    *out_h = abs_h;
    *out_pixels = pixels;
    return true;
}

// ==============================================================================
// 2. NATIVE OS & HARDWARE ACCELERATED BACKENDS
// ==============================================================================

#if defined(VIRA_USE_RAYLIB)

// ------------------------------------------------------------------------------
// 2A. HARDWARE ACCELERATED RAYLIB BACKEND (macOS, Windows & Linux GPU)
// ------------------------------------------------------------------------------
#include <raylib.h>
#include <math.h>

static bool    _vira_rl_should_close = false;
static int32_t _vira_rl_win_w = 800;
static int32_t _vira_rl_win_h = 600;
static Font    _vira_rl_font = {0};
static bool    _vira_rl_font_loaded = false;

static inline Color _vira_rl_color(int32_t hex) {
    return (Color){
        .r = (unsigned char)((hex >> 16) & 0xFF),
        .g = (unsigned char)((hex >> 8) & 0xFF),
        .b = (unsigned char)(hex & 0xFF),
        .a = 255
    };
}

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

static inline bool vira_gui_should_close(void) {
    return _vira_rl_should_close || WindowShouldClose();
}

static inline void vira_gui_cancel_close(void) {
    _vira_rl_should_close = false;
}

static inline void vira_gui_set_target_fps(int32_t fps) {
    if (fps > 0) SetTargetFPS(fps);
}

static inline int32_t vira_gui_window_width(void) {
    return IsWindowReady() ? GetScreenWidth() : _vira_rl_win_w;
}

static inline int32_t vira_gui_window_height(void) {
    return IsWindowReady() ? GetScreenHeight() : _vira_rl_win_h;
}

static inline void _vira_backend_gui_begin(void) {
    _vira_char_queue_count = 0;
    _vira_backspace_pressed = false;
    _vira_cursor_blink_counter++;

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

static inline void _vira_backend_gui_end(void) {
    EndDrawing();
}

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

static inline void vira_gui_clear(int32_t hex_color) {
    ClearBackground(_vira_rl_color(hex_color));
}

static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) {
    if (width <= 0 || height <= 0) return;
    DrawRectangle(x, y, width, height, _vira_rl_color(hex_color));
}

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

static inline void vira_gui_draw_circle(int32_t cx, int32_t cy, float radius, int32_t hex_color) {
    if (radius <= 0.0f) return;
    DrawCircle(cx, cy, radius, _vira_rl_color(hex_color));
}

static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t hex_color) {
    DrawLine(x1, y1, x2, y2, _vira_rl_color(hex_color));
}

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

static inline int32_t vira_gui_mouse_x(void) {
    return GetMouseX();
}

static inline int32_t vira_gui_mouse_y(void) {
    return GetMouseY();
}

static inline bool vira_gui_is_mouse_down(int32_t button) {
    if (button >= 0 && button < 3) return IsMouseButtonDown(button);
    return false;
}

static inline bool _vira_backend_mouse_clicked_left(void) {
    return IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

static inline bool vira_gui_is_key_pressed(int32_t key) {
    return IsKeyPressed(key);
}

static inline float vira_gui_frame_time(void) {
    return GetFrameTime();
}

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

#elif defined(_WIN32)

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
            _vira_gui.mouse_clicked[0] = true;
            return 0;
        case WM_LBUTTONUP:
            _vira_gui.mouse_down[0] = false;
            return 0;
        case WM_RBUTTONDOWN:
            _vira_gui.mouse_down[1] = true;
            _vira_gui.mouse_clicked[1] = true;
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

static inline void _vira_backend_gui_begin(void) {
    _vira_char_queue_count = 0;
    _vira_backspace_pressed = false;
    _vira_cursor_blink_counter++;
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

static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) {
    if (!_vira_gui.hdc_mem || width <= 0 || height <= 0) return;
    HBRUSH br = CreateSolidBrush(_vira_color_to_colorref(hex_color));
    RECT rc = {x, y, x + width, y + height};
    FillRect(_vira_gui.hdc_mem, &rc, br);
    DeleteObject(br);
}

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
static inline bool _vira_backend_mouse_clicked_left(void) { return _vira_gui.mouse_clicked[0]; }
static inline bool vira_gui_is_key_pressed(int32_t k) { return (k >= 0 && k < 256) ? _vira_gui.keys_pressed[k] : false; }
static inline float vira_gui_frame_time(void) { return _vira_gui.delta_time; }

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
    XFlush(_vira_x11.dpy);
    clock_gettime(CLOCK_MONOTONIC, &_vira_x11.frame_start);
    _vira_layout_depth = 0;
}

static inline bool vira_gui_should_close(void) { return _vira_x11.should_close; }
static inline void vira_gui_cancel_close(void) { _vira_x11.should_close = false; }
static inline void vira_gui_set_target_fps(int32_t fps) { if (fps > 0) _vira_x11.target_fps = fps; }
static inline int32_t vira_gui_window_width(void) { return _vira_x11.width; }
static inline int32_t vira_gui_window_height(void) { return _vira_x11.height; }

static inline void _vira_backend_gui_begin(void) {
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

static inline void _vira_backend_gui_end(void) {
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

static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) {
    if (!_vira_x11.dpy || width <= 0 || height <= 0) return;
    XSetForeground(_vira_x11.dpy, _vira_x11.gc, (unsigned long)(hex_color & 0xFFFFFF));
    XFillRectangle(_vira_x11.dpy, _vira_x11.backbuf, _vira_x11.gc, x, y, (unsigned int)width, (unsigned int)height);
}

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
static inline bool _vira_backend_mouse_clicked_left(void) { return _vira_x11.mouse_clicked[0]; }
static inline bool vira_gui_is_key_pressed(int32_t k) { return (k >= 0 && k < 256) ? _vira_x11.keys_pressed[k] : false; }
static inline float vira_gui_frame_time(void) { return _vira_x11.delta_time; }

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

#elif defined(__APPLE__) && defined(__OBJC__)

// ------------------------------------------------------------------------------
// 2C. NATIVE MACOS COCOA & COREGRAPHICS BACKEND (Zero-Dependency Native Mac)
// ------------------------------------------------------------------------------
#import <Cocoa/Cocoa.h>
#import <CoreGraphics/CoreGraphics.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

@class ViraCocoaView;

typedef struct {
    NSWindow*       window;
    ViraCocoaView*  view;
    uint32_t*       pixel_buf;
    CGContextRef    bitmap_ctx;
    CGColorSpaceRef color_space;
    int32_t         width;
    int32_t         height;
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
} vira_cocoa_t;

static vira_cocoa_t _vira_cocoa = {0};

@interface ViraCocoaView : NSView <NSWindowDelegate>
@end

@implementation ViraCocoaView
- (BOOL)isFlipped { return YES; }
- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)canBecomeKeyView { return YES; }
- (BOOL)needsPanelToBecomeKey { return YES; }

- (BOOL)performKeyEquivalent:(NSEvent *)event {
    if ([event keyCode] == 48) { // Tab key
        _vira_tab_pressed = true;
        NSEventModifierFlags flags = [event modifierFlags];
        _vira_shift_down = ((flags & NSEventModifierFlagShift) != 0);
        return YES;
    }
    return [super performKeyEquivalent:event];
}

- (void)updateMouse:(NSEvent *)event {
    NSPoint loc = [self convertPoint:[event locationInWindow] fromView:nil];
    _vira_cocoa.mouse_x = (int32_t)loc.x;
    _vira_cocoa.mouse_y = (int32_t)loc.y;
}
- (void)mouseMoved:(NSEvent *)event { [self updateMouse:event]; }
- (void)mouseDragged:(NSEvent *)event { [self updateMouse:event]; }
- (void)rightMouseDragged:(NSEvent *)event { [self updateMouse:event]; }
- (void)otherMouseDragged:(NSEvent *)event { [self updateMouse:event]; }

- (void)mouseDown:(NSEvent *)event {
    _vira_cocoa.mouse_down[0] = true;
    _vira_cocoa.mouse_clicked[0] = true;
    [self updateMouse:event];
}
- (void)mouseUp:(NSEvent *)event {
    _vira_cocoa.mouse_down[0] = false;
    [self updateMouse:event];
}
- (void)rightMouseDown:(NSEvent *)event {
    _vira_cocoa.mouse_down[1] = true;
    _vira_cocoa.mouse_clicked[1] = true;
    [self updateMouse:event];
}
- (void)rightMouseUp:(NSEvent *)event {
    _vira_cocoa.mouse_down[1] = false;
    [self updateMouse:event];
}

- (void)keyDown:(NSEvent *)event {
    NSEventModifierFlags flags = [event modifierFlags];
    _vira_shift_down = ((flags & NSEventModifierFlagShift) != 0);

    NSString *chars = [event characters];
    for (NSUInteger i = 0; i < [chars length]; i++) {
        unichar c = [chars characterAtIndex:i];
        if (c == 0x09 || c == 0x19) {
            _vira_tab_pressed = true;
        } else if (c == NSBackspaceCharacter || c == 0x7f) {
            _vira_backspace_pressed = true;
        } else if (c == NSEnterCharacter || c == NSCarriageReturnCharacter || c == 0x1b) {
            _vira_active_input_id = -1;
        } else if (c >= 32 && c <= 126 && _vira_char_queue_count < VIRA_CHAR_QUEUE_SIZE) {
            _vira_char_queue[_vira_char_queue_count++] = (char)c;
        }
    }
    unsigned short code = [event keyCode];
    if (code == 48) {
        _vira_tab_pressed = true;
    }
    if (code < 256) _vira_cocoa.keys_down[code] = true;
}

- (void)keyUp:(NSEvent *)event {
    unsigned short code = [event keyCode];
    if (code < 256) _vira_cocoa.keys_down[code] = false;
}

- (BOOL)windowShouldClose:(NSWindow *)sender {
    _vira_cocoa.should_close = true;
    return NO;
}

- (void)drawRect:(NSRect)dirtyRect {
    if (!_vira_cocoa.bitmap_ctx) return;
    CGImageRef img = CGBitmapContextCreateImage(_vira_cocoa.bitmap_ctx);
    if (!img) return;
    CGContextRef screenCtx = [[NSGraphicsContext currentContext] CGContext];
    CGContextDrawImage(screenCtx, CGRectMake(0, 0, _vira_cocoa.width, _vira_cocoa.height), img);
    CGImageRelease(img);
}
@end

static inline void vira_gui_init(int32_t width, int32_t height, vira_string title) {
    if (_vira_cocoa.window) return;
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        _vira_cocoa.width = width;
        _vira_cocoa.height = height;
        _vira_cocoa.target_fps = 60;
        _vira_cocoa.should_close = false;
        clock_gettime(CLOCK_MONOTONIC, &_vira_cocoa.frame_start);

        _vira_cocoa.pixel_buf = (uint32_t*)calloc((size_t)width * height, sizeof(uint32_t));
        _vira_cocoa.color_space = CGColorSpaceCreateDeviceRGB();
        _vira_cocoa.bitmap_ctx = CGBitmapContextCreate(
            _vira_cocoa.pixel_buf,
            width, height, 8, width * 4,
            _vira_cocoa.color_space,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
        );

        char stack_title[256];
        int32_t tlen = title.len < 255 ? title.len : 255;
        if (tlen > 0 && title.data) memcpy(stack_title, title.data, (size_t)tlen);
        stack_title[tlen] = '\0';

        NSRect screenFrame = [[NSScreen mainScreen] visibleFrame];
        CGFloat x = screenFrame.origin.x + (screenFrame.size.width - width) / 2.0;
        CGFloat y = screenFrame.origin.y + (screenFrame.size.height - height) / 2.0;
        NSRect winFrame = NSMakeRect(x, y, width, height);

        NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable;
        NSWindow *win = [[NSWindow alloc] initWithContentRect:winFrame styleMask:style backing:NSBackingStoreBuffered defer:NO];
        [win setTitle:[NSString stringWithUTF8String:stack_title]];

        ViraCocoaView *view = [[ViraCocoaView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
        [win setContentView:view];
        [win setDelegate:view];

        NSTrackingArea *ta = [[NSTrackingArea alloc] initWithRect:NSMakeRect(0, 0, width, height)
            options:(NSTrackingMouseMoved | NSTrackingActiveAlways | NSTrackingInVisibleRect)
            owner:view userInfo:nil];
        [view addTrackingArea:ta];
        [win setAcceptsMouseMovedEvents:YES];
        [win makeKeyAndOrderFront:nil];
        [win makeFirstResponder:view];
        [NSApp activateIgnoringOtherApps:YES];

        _vira_cocoa.window = win;
        _vira_cocoa.view = view;
    }
}

static inline bool vira_gui_should_close(void) {
    return _vira_cocoa.should_close;
}

static inline void vira_gui_cancel_close(void) {
    _vira_cocoa.should_close = false;
}

static inline void vira_gui_set_target_fps(int32_t fps) {
    if (fps > 0) _vira_cocoa.target_fps = fps;
}

static inline int32_t vira_gui_window_width(void) {
    return _vira_cocoa.width;
}

static inline int32_t vira_gui_window_height(void) {
    return _vira_cocoa.height;
}

static inline void _vira_backend_gui_begin(void) {
    if (!_vira_cocoa.window) return;
    _vira_char_queue_count = 0;
    _vira_backspace_pressed = false;
    _vira_cursor_blink_counter++;

    for (int i = 0; i < 3; i++) {
        _vira_cocoa.mouse_clicked[i] = false;
    }

    @autoreleasepool {
        NSEvent *event;
        while ((event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                           untilDate:[NSDate distantPast]
                                              inMode:NSDefaultRunLoopMode
                                             dequeue:YES])) {
            [NSApp sendEvent:event];
        }
    }

    for (int i = 0; i < 3; i++) {
        if (!_vira_cocoa.mouse_prev_down[i] && _vira_cocoa.mouse_down[i]) {
            _vira_cocoa.mouse_clicked[i] = true;
        }
        _vira_cocoa.mouse_prev_down[i] = _vira_cocoa.mouse_down[i];
    }
    for (int i = 0; i < 256; i++) {
        _vira_cocoa.keys_pressed[i] = (!_vira_cocoa.keys_prev[i] && _vira_cocoa.keys_down[i]);
        _vira_cocoa.keys_prev[i] = _vira_cocoa.keys_down[i];
    }
}

static inline void _vira_backend_gui_end(void) {
    if (!_vira_cocoa.window || !_vira_cocoa.view) return;
    [_vira_cocoa.view setNeedsDisplay:YES];
    [_vira_cocoa.view displayIfNeeded];

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (now.tv_sec - _vira_cocoa.frame_start.tv_sec) +
                     (now.tv_nsec - _vira_cocoa.frame_start.tv_nsec) * 1e-9;
    double target = 1.0 / (double)_vira_cocoa.target_fps;
    if (elapsed < target) {
        uint32_t us = (uint32_t)((target - elapsed) * 1000000.0);
        if (us > 0) usleep(us);
        clock_gettime(CLOCK_MONOTONIC, &now);
        elapsed = (now.tv_sec - _vira_cocoa.frame_start.tv_sec) +
                  (now.tv_nsec - _vira_cocoa.frame_start.tv_nsec) * 1e-9;
    }
    _vira_cocoa.delta_time = (float)elapsed;
    _vira_cocoa.frame_start = now;
}

static inline void _vira_cocoa_free_image(int32_t slot);

static inline void vira_gui_close(void) {
    for (int i = 0; i < VIRA_MAX_GUI_IMAGES; i++) {
        if (_vira_gui_images[i].is_valid) {
            _vira_cocoa_free_image(i);
        }
    }
    if (_vira_cocoa.bitmap_ctx) {
        CGContextRelease(_vira_cocoa.bitmap_ctx);
        _vira_cocoa.bitmap_ctx = NULL;
    }
    if (_vira_cocoa.color_space) {
        CGColorSpaceRelease(_vira_cocoa.color_space);
        _vira_cocoa.color_space = NULL;
    }
    if (_vira_cocoa.pixel_buf) {
        free(_vira_cocoa.pixel_buf);
        _vira_cocoa.pixel_buf = NULL;
    }
    if (_vira_cocoa.window) {
        [_vira_cocoa.window close];
        _vira_cocoa.window = nil;
        _vira_cocoa.view = nil;
    }
}

static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t width, int32_t height, int32_t hex_color) {
    if (!_vira_cocoa.bitmap_ctx || width <= 0 || height <= 0) return;
    float r = ((hex_color >> 16) & 0xFF) / 255.0f;
    float g = ((hex_color >> 8) & 0xFF) / 255.0f;
    float b = (hex_color & 0xFF) / 255.0f;
    CGContextSetRGBFillColor(_vira_cocoa.bitmap_ctx, r, g, b, 1.0f);
    CGContextFillRect(_vira_cocoa.bitmap_ctx, CGRectMake(x, y, width, height));
}

static inline void vira_gui_draw_rounded_rect(int32_t x, int32_t y, int32_t width, int32_t height, float radius, int32_t bg_hex, int32_t border_hex) {
    if (!_vira_cocoa.bitmap_ctx || width <= 0 || height <= 0) return;
    CGRect rect = CGRectMake((CGFloat)x, (CGFloat)y, (CGFloat)width, (CGFloat)height);
    CGPathRef path = CGPathCreateWithRoundedRect(rect, (CGFloat)radius, (CGFloat)radius, NULL);
    if (bg_hex >= 0) {
        float r = ((bg_hex >> 16) & 0xFF) / 255.0f;
        float g = ((bg_hex >> 8) & 0xFF) / 255.0f;
        float b = (bg_hex & 0xFF) / 255.0f;
        CGContextSetRGBFillColor(_vira_cocoa.bitmap_ctx, r, g, b, 1.0f);
        CGContextAddPath(_vira_cocoa.bitmap_ctx, path);
        CGContextFillPath(_vira_cocoa.bitmap_ctx);
    }
    if (border_hex >= 0) {
        float r = ((border_hex >> 16) & 0xFF) / 255.0f;
        float g = ((border_hex >> 8) & 0xFF) / 255.0f;
        float b = (border_hex & 0xFF) / 255.0f;
        CGContextSetRGBStrokeColor(_vira_cocoa.bitmap_ctx, r, g, b, 1.0f);
        CGContextSetLineWidth(_vira_cocoa.bitmap_ctx, 1.0f);
        CGContextAddPath(_vira_cocoa.bitmap_ctx, path);
        CGContextStrokePath(_vira_cocoa.bitmap_ctx);
    }
    CGPathRelease(path);
}

static inline void vira_gui_clear(int32_t hex_color) {
    vira_gui_draw_rect(0, 0, _vira_cocoa.width, _vira_cocoa.height, hex_color);
}

static inline void vira_gui_draw_circle(int32_t cx, int32_t cy, float radius, int32_t hex_color) {
    if (!_vira_cocoa.bitmap_ctx || radius <= 0.0f) return;
    float r = ((hex_color >> 16) & 0xFF) / 255.0f;
    float g = ((hex_color >> 8) & 0xFF) / 255.0f;
    float b = (hex_color & 0xFF) / 255.0f;
    CGContextSetRGBFillColor(_vira_cocoa.bitmap_ctx, r, g, b, 1.0f);
    CGContextFillEllipseInRect(_vira_cocoa.bitmap_ctx, CGRectMake(cx - radius, cy - radius, radius * 2, radius * 2));
}

static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t hex_color) {
    if (!_vira_cocoa.bitmap_ctx) return;
    float r = ((hex_color >> 16) & 0xFF) / 255.0f;
    float g = ((hex_color >> 8) & 0xFF) / 255.0f;
    float b = (hex_color & 0xFF) / 255.0f;
    CGContextSetRGBStrokeColor(_vira_cocoa.bitmap_ctx, r, g, b, 1.0f);
    CGContextSetLineWidth(_vira_cocoa.bitmap_ctx, 1.0f);
    CGContextBeginPath(_vira_cocoa.bitmap_ctx);
    CGContextMoveToPoint(_vira_cocoa.bitmap_ctx, x1, y1);
    CGContextAddLineToPoint(_vira_cocoa.bitmap_ctx, x2, y2);
    CGContextStrokePath(_vira_cocoa.bitmap_ctx);
}

static inline void vira_gui_draw_text(vira_string text, int32_t x, int32_t y, int32_t size, int32_t hex_color) {
    if (!_vira_cocoa.bitmap_ctx || text.len <= 0 || !text.data) return;
    @autoreleasepool {
        NSString *str = [[NSString alloc] initWithBytes:text.data length:text.len encoding:NSUTF8StringEncoding];
        if (!str) return;
        float r = ((hex_color >> 16) & 0xFF) / 255.0f;
        float g = ((hex_color >> 8) & 0xFF) / 255.0f;
        float b = (hex_color & 0xFF) / 255.0f;
        NSDictionary *attrs = @{
            NSFontAttributeName: [NSFont systemFontOfSize:(size > 0 ? size : 14)],
            NSForegroundColorAttributeName: [NSColor colorWithRed:r green:g blue:b alpha:1.0f]
        };
        [NSGraphicsContext saveGraphicsState];
        NSGraphicsContext *nsCtx = [NSGraphicsContext graphicsContextWithCGContext:_vira_cocoa.bitmap_ctx flipped:YES];
        [NSGraphicsContext setCurrentContext:nsCtx];
        [str drawAtPoint:NSMakePoint(x, y) withAttributes:attrs];
        [NSGraphicsContext restoreGraphicsState];
    }
}

static inline int32_t _vira_measure_text_w(vira_string text) {
    if (text.len <= 0 || !text.data) return 0;
    @autoreleasepool {
        NSString *str = [[NSString alloc] initWithBytes:text.data length:text.len encoding:NSUTF8StringEncoding];
        if (!str) return text.len * 8;
        NSDictionary *attrs = @{ NSFontAttributeName: [NSFont systemFontOfSize:14] };
        NSSize sz = [str sizeWithAttributes:attrs];
        return (int32_t)ceil(sz.width);
    }
}

static inline int32_t vira_gui_mouse_x(void) { return _vira_cocoa.mouse_x; }
static inline int32_t vira_gui_mouse_y(void) { return _vira_cocoa.mouse_y; }
static inline bool vira_gui_is_mouse_down(int32_t button) {
    if (button >= 0 && button < 3) return _vira_cocoa.mouse_down[button];
    return false;
}
static inline bool _vira_backend_mouse_clicked_left(void) {
    return _vira_cocoa.mouse_clicked[0];
}
static inline bool vira_gui_is_key_pressed(int32_t key) {
    if (key >= 0 && key < 256) return _vira_cocoa.keys_pressed[key];
    return false;
}
static inline float vira_gui_frame_time(void) {
    return _vira_cocoa.delta_time;
}

static inline int32_t _vira_cocoa_load_image(const char* cpath) {
    if (!cpath || cpath[0] == '\0') return -1;
    @autoreleasepool {
        NSString *nsPath = [NSString stringWithUTF8String:cpath];
        if (!nsPath) return -1;
        if (![[NSFileManager defaultManager] fileExistsAtPath:nsPath]) {
            NSString *appDir = [[NSBundle mainBundle] bundlePath];
            NSString *altPath = [appDir stringByAppendingPathComponent:nsPath];
            if ([[NSFileManager defaultManager] fileExistsAtPath:altPath]) {
                nsPath = altPath;
            } else {
                return -1;
            }
        }
        NSImage *img = [[NSImage alloc] initWithContentsOfFile:nsPath];
        if (!img) return -1;

        NSSize sz = [img size];
        int32_t w = (int32_t)ceil(sz.width);
        int32_t h = (int32_t)ceil(sz.height);
        if (w <= 0 || h <= 0) return -1;

        int32_t slot = -1;
        for (int i = 0; i < VIRA_MAX_GUI_IMAGES; i++) {
            if (!_vira_gui_images[i].is_valid) {
                slot = i;
                break;
            }
        }
        if (slot < 0) return -1;

        _vira_gui_images[slot].width = w;
        _vira_gui_images[slot].height = h;
        _vira_gui_images[slot].channels = 4;
        _vira_gui_images[slot].pixels = NULL;
        _vira_gui_images[slot].native_handle = (void*)img;
        _vira_gui_images[slot].is_valid = true;
        return slot;
    }
}

static inline void _vira_cocoa_draw_image(int32_t slot, int32_t x, int32_t y, int32_t w, int32_t h) {
    if (slot < 0 || slot >= VIRA_MAX_GUI_IMAGES || !_vira_gui_images[slot].is_valid) return;
    if (!_vira_cocoa.bitmap_ctx) return;

    @autoreleasepool {
        NSImage *img = (NSImage*)_vira_gui_images[slot].native_handle;
        if (!img) return;

        int32_t target_w = (w > 0) ? w : _vira_gui_images[slot].width;
        int32_t target_h = (h > 0) ? h : _vira_gui_images[slot].height;

        [NSGraphicsContext saveGraphicsState];
        NSGraphicsContext *nsCtx = [NSGraphicsContext graphicsContextWithCGContext:_vira_cocoa.bitmap_ctx flipped:YES];
        [NSGraphicsContext setCurrentContext:nsCtx];
        [img drawInRect:NSMakeRect(x, y, target_w, target_h)
               fromRect:NSZeroRect
              operation:NSCompositingOperationSourceOver
               fraction:1.0f
         respectFlipped:YES
                  hints:nil];
        [NSGraphicsContext restoreGraphicsState];
    }
}

static inline void _vira_cocoa_free_image(int32_t slot) {
    if (slot < 0 || slot >= VIRA_MAX_GUI_IMAGES || !_vira_gui_images[slot].is_valid) return;
    if (_vira_gui_images[slot].native_handle) {
        [(NSImage*)_vira_gui_images[slot].native_handle release];
        _vira_gui_images[slot].native_handle = NULL;
    }
    if (_vira_gui_images[slot].pixels) {
        free(_vira_gui_images[slot].pixels);
        _vira_gui_images[slot].pixels = NULL;
    }
    _vira_gui_images[slot].is_valid = false;
    _vira_gui_images[slot].width = 0;
    _vira_gui_images[slot].height = 0;
}

#else

// 2C. HEADLESS STUB
static inline void vira_gui_init(int32_t w, int32_t h, vira_string t) { (void)w; (void)h; (void)t; }
static inline bool vira_gui_should_close(void) { return true; }
static inline void vira_gui_cancel_close(void) {}
static inline void vira_gui_set_target_fps(int32_t fps) { (void)fps; }
static inline int32_t vira_gui_window_width(void) { return 800; }
static inline int32_t vira_gui_window_height(void) { return 600; }
static inline void _vira_backend_gui_begin(void) {}
static inline void _vira_backend_gui_end(void) {}
static inline void vira_gui_close(void) {}
static inline void vira_gui_clear(int32_t c) { (void)c; }
static inline void vira_gui_draw_text(vira_string t, int32_t x, int32_t y, int32_t s, int32_t c) { (void)t; (void)x; (void)y; (void)s; (void)c; }
static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t w, int32_t h, int32_t c) { (void)x; (void)y; (void)w; (void)h; (void)c; }
static inline void vira_gui_draw_rounded_rect(int32_t x, int32_t y, int32_t w, int32_t h, float r, int32_t bg, int32_t border) { (void)x; (void)y; (void)w; (void)h; (void)r; (void)bg; (void)border; }
static inline void vira_gui_draw_circle(int32_t cx, int32_t cy, float r, int32_t c) { (void)cx; (void)cy; (void)r; (void)c; }
static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t c) { (void)x1; (void)y1; (void)x2; (void)y2; (void)c; }
static inline int32_t _vira_measure_text_w(vira_string t) { return t.len * 8; }
static inline int32_t vira_gui_mouse_x(void) { return 0; }
static inline int32_t vira_gui_mouse_y(void) { return 0; }
static inline bool vira_gui_is_mouse_down(int32_t b) { (void)b; return false; }
static inline bool _vira_backend_mouse_clicked_left(void) { return false; }
static inline bool vira_gui_is_key_pressed(int32_t k) { (void)k; return false; }
static inline float vira_gui_frame_time(void) { return 0.016f; }
static inline int32_t _vira_headless_load_image(const char* filepath) { (void)filepath; return -1; }
static inline void _vira_headless_draw_image(int32_t id, int32_t x, int32_t y, int32_t w, int32_t h) { (void)id; (void)x; (void)y; (void)w; (void)h; }
static inline void _vira_headless_free_image(int32_t id) { (void)id; }

#endif

// ------------------------------------------------------------------------------
// Unified Cross-Platform Image Dispatch
// ------------------------------------------------------------------------------
static inline int32_t vira_gui_load_image(vira_string path) {
    if (path.len <= 0 || !path.data) return -1;
    char stack_path[512];
    int32_t plen = path.len < 511 ? path.len : 511;
    memcpy(stack_path, path.data, (size_t)plen);
    stack_path[plen] = '\0';

#if defined(VIRA_USE_RAYLIB)
    return _vira_raylib_load_image(stack_path);
#elif defined(_WIN32)
    return _vira_win32_load_image(stack_path);
#elif (defined(__linux__) || defined(__unix__)) && !defined(VIRA_NO_X11)
    return _vira_x11_load_image(stack_path);
#elif defined(__APPLE__) && defined(__OBJC__)
    return _vira_cocoa_load_image(stack_path);
#else
    return _vira_headless_load_image(stack_path);
#endif
}

static inline int32_t vira_gui_image_width(int32_t id) {
    if (id >= 0 && id < VIRA_MAX_GUI_IMAGES && _vira_gui_images[id].is_valid) {
        return _vira_gui_images[id].width;
    }
    return 0;
}

static inline int32_t vira_gui_image_height(int32_t id) {
    if (id >= 0 && id < VIRA_MAX_GUI_IMAGES && _vira_gui_images[id].is_valid) {
        return _vira_gui_images[id].height;
    }
    return 0;
}

static inline void vira_gui_draw_image(int32_t id, int32_t x, int32_t y, int32_t w, int32_t h) {
    if (id < 0 || id >= VIRA_MAX_GUI_IMAGES || !_vira_gui_images[id].is_valid) return;
    int32_t draw_w = (w > 0) ? w : _vira_gui_images[id].width;
    int32_t draw_h = (h > 0) ? h : _vira_gui_images[id].height;

#if defined(VIRA_USE_RAYLIB)
    _vira_raylib_draw_image(id, x, y, draw_w, draw_h);
#elif defined(_WIN32)
    _vira_win32_draw_image(id, x, y, draw_w, draw_h);
#elif (defined(__linux__) || defined(__unix__)) && !defined(VIRA_NO_X11)
    _vira_x11_draw_image(id, x, y, draw_w, draw_h);
#elif defined(__APPLE__) && defined(__OBJC__)
    _vira_cocoa_draw_image(id, x, y, draw_w, draw_h);
#else
    _vira_headless_draw_image(id, x, y, draw_w, draw_h);
#endif
}

static inline void vira_gui_free_image(int32_t id) {
    if (id < 0 || id >= VIRA_MAX_GUI_IMAGES || !_vira_gui_images[id].is_valid) return;
#if defined(VIRA_USE_RAYLIB)
    _vira_raylib_free_image(id);
#elif defined(_WIN32)
    _vira_win32_free_image(id);
#elif (defined(__linux__) || defined(__unix__)) && !defined(VIRA_NO_X11)
    _vira_x11_free_image(id);
#elif defined(__APPLE__) && defined(__OBJC__)
    _vira_cocoa_free_image(id);
#else
    _vira_headless_free_image(id);
#endif
}

// ==============================================================================
// 3. SHARED WIDGETS & DECLARATIVE AUTO-LAYOUT ENGINE
// ==============================================================================

static inline bool _vira_gui_mouse_clicked_left(void) {
    if (_vira_click_consumed) return false;
    if (_vira_modal_active) return false;
    return _vira_backend_mouse_clicked_left();
}

static inline void _vira_gui_draw_deferred_popups(void);

static inline void vira_gui_begin(void) {
    _vira_layout_depth = 0;
    _vira_input_counter = 0;
    _vira_card_counter = 0;
    _vira_card_end_counter = 0;
    _vira_dropdown_counter = 0;
    _vira_table_counter = 0;
    _vira_click_consumed = false;
    _vira_modal_active = _vira_modal_open_prev_frame;
    _vira_modal_first_input = -1;
    _vira_modal_last_input = -1;

    _vira_backend_gui_begin();

    // Keyboard Tab Focus Navigation
    if (_vira_tab_pressed && _vira_total_inputs > 0) {
        int32_t min_id = 0;
        int32_t max_id = _vira_total_inputs - 1;
        if (_vira_modal_open_prev_frame && _vira_prev_modal_first_input >= 0 && _vira_prev_modal_last_input >= _vira_prev_modal_first_input) {
            min_id = _vira_prev_modal_first_input;
            max_id = _vira_prev_modal_last_input;
        }
        int32_t span = max_id - min_id + 1;
        if (span > 0) {
            if (_vira_active_input_id < min_id || _vira_active_input_id > max_id) {
                _vira_active_input_id = _vira_shift_down ? max_id : min_id;
            } else {
                int32_t cur_offset = _vira_active_input_id - min_id;
                if (_vira_shift_down) {
                    cur_offset = (cur_offset - 1 + span) % span;
                } else {
                    cur_offset = (cur_offset + 1) % span;
                }
                _vira_active_input_id = min_id + cur_offset;
            }
        }
        _vira_tab_pressed = false;
    }
}

static inline void vira_gui_end(void) {
    _vira_gui_draw_deferred_popups();
    _vira_total_inputs = _vira_input_counter;
    _vira_prev_modal_first_input = _vira_modal_first_input;
    _vira_prev_modal_last_input = _vira_modal_last_input;
    _vira_modal_open_prev_frame = _vira_modal_open_this_frame;
    _vira_modal_open_this_frame = false;
    _vira_backend_gui_end();
}

static inline bool vira_gui_button(int32_t x, int32_t y, int32_t width, int32_t height, vira_string label) {
    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();
    bool is_hovered = (mx >= x && mx <= x + width && my >= y && my <= y + height);
    bool is_down    = is_hovered && vira_gui_is_mouse_down(0);
    bool is_clicked = is_hovered && _vira_gui_mouse_clicked_left();

    int32_t bg_color     = is_down ? 0x313244 : (is_hovered ? 0x45475A : 0x1E1E2E);
    int32_t border_color = is_hovered ? 0x89B4FA : 0x585B70;
    int32_t text_color   = is_hovered ? 0xFFFFFF : 0xCDD6F4;

    vira_gui_draw_rounded_rect(x, y, width, height, 6.0f, bg_color, border_color);

    if (label.len > 0) {
        int32_t tw = _vira_measure_text_w(label);
        vira_gui_draw_text(label, x + (width - tw) / 2, y + (height - 16) / 2, 15, text_color);
    }
    return is_clicked;
}

static inline void _vira_gui_draw_trash_icon(int32_t x, int32_t y, int32_t color) {
    // Lid handle
    vira_gui_draw_line(x + 4, y + 1, x + 8, y + 1, color);
    // Lid
    vira_gui_draw_line(x + 1, y + 3, x + 11, y + 3, color);
    // Can body
    vira_gui_draw_line(x + 2, y + 4, x + 3, y + 13, color);
    vira_gui_draw_line(x + 10, y + 4, x + 9, y + 13, color);
    vira_gui_draw_line(x + 3, y + 13, x + 9, y + 13, color);
    // Vertical ribs
    vira_gui_draw_line(x + 5, y + 5, x + 5, y + 11, color);
    vira_gui_draw_line(x + 7, y + 5, x + 7, y + 11, color);
}

static inline bool vira_gui_button_danger(int32_t x, int32_t y, int32_t width, int32_t height, vira_string label) {
    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();
    bool is_hovered = (mx >= x && mx <= x + width && my >= y && my <= y + height);
    bool is_down    = is_hovered && vira_gui_is_mouse_down(0);
    bool is_clicked = is_hovered && _vira_gui_mouse_clicked_left();

    int32_t bg_color     = is_down ? 0x4A1E29 : (is_hovered ? 0x3D202A : 0x24171E);
    int32_t border_color = is_hovered ? 0xF38BA8 : 0xA63A50;
    int32_t text_color   = is_hovered ? 0xFFFFFF : 0xF38BA8;

    vira_gui_draw_rounded_rect(x, y, width, height, 6.0f, bg_color, border_color);

    int32_t tw = _vira_measure_text_w(label);
    int32_t total_w = 14 + 6 + tw;
    int32_t start_x = x + (width - total_w) / 2;
    int32_t mid_y = y + (height - 16) / 2;

    _vira_gui_draw_trash_icon(start_x, mid_y, text_color);
    if (label.len > 0) {
        vira_gui_draw_text(label, start_x + 18, mid_y, 15, text_color);
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
    vira_gui_draw_rounded_rect(x, box_y, box_size, box_size, 4.0f, 0x313244, is_hovered ? 0x89B4FA : 0x585B70);
    if (checked) {
        vira_gui_draw_rounded_rect(x + 3, box_y + 3, box_size - 6, box_size - 6, 2.0f, 0xA6E3A1, -1);
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

    if (_vira_modal_open_this_frame) {
        if (_vira_modal_first_input < 0) _vira_modal_first_input = id;
        _vira_modal_last_input = id;
        if (_vira_modal_just_opened && _vira_active_input_id < 0) {
            _vira_active_input_id = id;
            _vira_modal_just_opened = false;
        }
    }

    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();
    bool is_hovered = (mx >= x && mx <= x + width && my >= y && my <= y + height);

    if (_vira_gui_mouse_clicked_left()) {
        if (is_hovered) {
            _vira_active_input_id = id;
            _vira_click_consumed = true;
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

    vira_gui_draw_rounded_rect(x, y, width, height, 6.0f, bg_color, border_color);

    if (is_focused) {
        vira_gui_draw_rounded_rect(x - 1, y - 1, width + 2, height + 2, 7.0f, -1, 0x89B4FA);
    }

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

    vira_gui_draw_rounded_rect(x, y, width, height, 4.0f, 0x11111B, 0x313244);

    int fill_w = (int)(percent * (float)(width - 4));
    if (fill_w > 0) {
        vira_gui_draw_rounded_rect(x + 2, y + 2, fill_w, height - 4, 3.0f, 0xA6E3A1, -1);
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
    int32_t border = ((hex_bg >> 1) & 0x7F7F7F);

    vira_gui_draw_rounded_rect(x, y, bw, bh, 6.0f, hex_bg, border);
    vira_gui_draw_text(text, x + 8, y + 3, 13, hex_fg);
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

    vira_gui_draw_rounded_rect(x, y, w, card_h, 8.0f, 0x1E1E2E, 0x313244);

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
        int32_t w = _vira_measure_text_w(label) + 28;
        if (w < 120) w = 120;
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

static inline bool vira_gui_btn_danger(vira_string label) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return false;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        int32_t w = _vira_measure_text_w(label) + 40;
        if (w < 130) w = 130;
        int32_t h = node->height;
        bool clicked = vira_gui_button_danger(x, y, w, h, label);
        node->cur_x += w + node->spacing;
        return clicked;
    } else {
        int32_t w = node->width;
        int32_t h = 36;
        bool clicked = vira_gui_button_danger(x, y, w, h, label);
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

static inline void vira_gui_image(int32_t id, int32_t w, int32_t h) {
    if (id < 0 || id >= VIRA_MAX_GUI_IMAGES || !_vira_gui_images[id].is_valid) return;
    int32_t draw_w = (w > 0) ? w : _vira_gui_images[id].width;
    int32_t draw_h = (h > 0) ? h : _vira_gui_images[id].height;

    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (node) {
        int32_t x = node->cur_x;
        int32_t y = node->cur_y;
        vira_gui_draw_image(id, x, y, draw_w, draw_h);
        if (node->type == VIRA_LAYOUT_HORIZONTAL) {
            node->cur_x += draw_w + node->spacing;
        } else {
            node->cur_y += draw_h + node->spacing;
        }
    } else {
        vira_gui_draw_image(id, 0, 0, draw_w, draw_h);
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

    vira_gui_draw_rounded_rect(x, y, w, h, 6.0f, bg_color, -1);
    if (is_active) {
        vira_gui_draw_rounded_rect(x + 2, y + h - 3, w - 4, 3, 1.5f, 0x89B4FA, -1);
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
    if (!_vira_modal_open_prev_frame) {
        _vira_modal_just_opened = true;
        _vira_active_input_id = -1;
    }
    _vira_modal_open_this_frame = true;
    _vira_modal_active = false; // Enable clicks inside modal dialog

    int32_t win_w = vira_gui_window_width();
    int32_t win_h = vira_gui_window_height();

    // Dim the entire background with dark backdrop
    vira_gui_draw_rect(0, 0, win_w, win_h, 0x0C0C14);

    int32_t mx = (win_w - width) / 2;
    int32_t my = (win_h - height) / 2;
    if (mx < 20) mx = 20;
    if (my < 20) my = 20;

    vira_gui_draw_rounded_rect(mx, my, width, height, 10.0f, 0x1E1E2E, 0x585B70);
    vira_gui_draw_rounded_rect(mx, my, width, 38, 10.0f, 0x181825, -1);
    vira_gui_draw_line(mx, my + 38, mx + width, my + 38, 0x313244);
    if (title.len > 0) {
        vira_gui_draw_text(title, mx + 16, my + 10, 16, 0xCDD6F4);
    }

    vira_gui_vstack_begin(mx + 16, my + 50, width - 32, 10);
    _vira_layout_stack[_vira_layout_depth - 1].is_card = 0;
}

static inline void vira_gui_modal_end(void) {
    vira_gui_stack_end();
    _vira_click_consumed = true;
}

// ------------------------------------------------------------------------------
// 3B. DROPDOWN SELECT & INTERACTIVE DATA TABLE COMPONENTS
// ------------------------------------------------------------------------------

static inline void _vira_gui_draw_deferred_popups(void) {
    if (!_vira_current_dropdown_popup.active) return;

    int32_t px = _vira_current_dropdown_popup.x;
    int32_t py = _vira_current_dropdown_popup.y;
    int32_t pw = _vira_current_dropdown_popup.w;
    int32_t ph = _vira_current_dropdown_popup.h;
    vira_slice_string items = _vira_current_dropdown_popup.items;
    int32_t sel = _vira_current_dropdown_popup.selected_idx;
    int32_t item_h = 28;

    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();

    // Solid dark background for popup list
    vira_gui_draw_rounded_rect(px, py, pw, ph, 6.0f, 0x181825, 0x585B70);

    for (int i = 0; i < items.len; i++) {
        int iy = py + i * item_h;
        if (iy + item_h > py + ph) break;
        bool is_hover = (mx >= px && mx <= px + pw && my >= iy && my < iy + item_h);
        bool is_sel = (i == sel);

        int32_t bg = is_sel ? 0x313244 : (is_hover ? 0x45475A : 0x181825);
        int32_t fg = is_sel ? 0x89B4FA : (is_hover ? 0xFFFFFF : 0xCDD6F4);

        if (is_sel || is_hover) {
            vira_gui_draw_rounded_rect(px + 4, iy + 2, pw - 8, item_h - 4, 4.0f, bg, -1);
        }
        if (is_sel) {
            vira_gui_draw_rounded_rect(px + 2, iy + 4, 3, item_h - 8, 1.5f, 0x89B4FA, -1);
        }
        vira_gui_draw_text(items.data[i], px + 12, iy + (item_h - 15) / 2, 14, fg);
    }

    _vira_current_dropdown_popup.active = false;
}

static inline int32_t vira_gui_dropdown(int32_t x, int32_t y, int32_t width, int32_t height,
                                       vira_slice_string items, int32_t selected_idx) {
    int32_t id = _vira_dropdown_counter++;
    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();

    bool is_hovered = (mx >= x && mx <= x + width && my >= y && my <= y + height);
    bool is_open = (_vira_active_dropdown_id == id);

    int item_h = 28;
    int popup_h = items.len * item_h;
    if (popup_h > 240) popup_h = 240;

    int popup_x = x;
    int popup_y = y + height;
    int win_h = vira_gui_window_height();
    if (win_h > 0 && popup_y + popup_h > win_h - 10 && y - popup_h > 10) {
        popup_y = y - popup_h;
    }

    if (_vira_gui_mouse_clicked_left()) {
        if (is_open) {
            if (mx >= popup_x && mx <= popup_x + width && my >= popup_y && my <= popup_y + popup_h) {
                int clicked_item = (my - popup_y) / item_h;
                if (clicked_item >= 0 && clicked_item < items.len) {
                    selected_idx = clicked_item;
                }
            }
            _vira_active_dropdown_id = -1;
            _vira_click_consumed = true;
            is_open = false;
        } else if (is_hovered) {
            _vira_active_dropdown_id = id;
            _vira_click_consumed = true;
            is_open = true;
        }
    }

    int32_t bg_color     = is_open ? 0x181825 : (is_hovered ? 0x242537 : 0x1E1E2E);
    int32_t border_color = is_open ? 0x89B4FA : (is_hovered ? 0x585B70 : 0x313244);

    vira_gui_draw_rounded_rect(x, y, width, height, 6.0f, bg_color, border_color);

    if (is_open) {
        vira_gui_draw_rounded_rect(x - 1, y - 1, width + 2, height + 2, 7.0f, -1, 0x89B4FA);
    }

    int text_y = y + (height - 16) / 2;
    if (selected_idx >= 0 && selected_idx < items.len) {
        vira_gui_draw_text(items.data[selected_idx], x + 10, text_y, 15, 0xCDD6F4);
    } else {
        vira_string ph = (vira_string){ .data = (const uint8_t*)"Select...", .len = 9, .is_lit = true };
        vira_gui_draw_text(ph, x + 10, text_y, 15, 0x6C7086);
    }

    vira_string arrow = is_open
        ? (vira_string){ .data = (const uint8_t*)"^", .len = 1, .is_lit = true }
        : (vira_string){ .data = (const uint8_t*)"v", .len = 1, .is_lit = true };
    vira_gui_draw_text(arrow, x + width - 20, text_y, 13, border_color);

    if (is_open) {
        _vira_current_dropdown_popup.active = true;
        _vira_current_dropdown_popup.x = popup_x;
        _vira_current_dropdown_popup.y = popup_y;
        _vira_current_dropdown_popup.w = width;
        _vira_current_dropdown_popup.h = popup_h;
        _vira_current_dropdown_popup.items = items;
        _vira_current_dropdown_popup.selected_idx = selected_idx;
    }

    return selected_idx;
}

static inline int32_t vira_gui_select(vira_slice_string items, int32_t selected_idx) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return selected_idx;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t h = (node->type == VIRA_LAYOUT_HORIZONTAL) ? node->height : 34;
    int32_t w = (node->type == VIRA_LAYOUT_HORIZONTAL) ? 180 : node->width;

    int32_t res = vira_gui_dropdown(x, y, w, h, items, selected_idx);

    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        node->cur_x += w + node->spacing;
    } else {
        node->cur_y += h + node->spacing;
    }
    return res;
}

static inline bool _vira_table_str_contains_ci(vira_string haystack, vira_string needle) {
    if (needle.len == 0 || !needle.data) return true;
    if (haystack.len < needle.len || !haystack.data) return false;
    int max_i = haystack.len - needle.len;
    for (int i = 0; i <= max_i; i++) {
        bool match = true;
        for (int j = 0; j < needle.len; j++) {
            if (tolower((unsigned char)haystack.data[i + j]) != tolower((unsigned char)needle.data[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

static inline bool _vira_is_numeric_str(vira_string s, double* out_val) {
    if (s.len <= 0 || !s.data) return false;
    int i = 0;
    while (i < s.len && (s.data[i] == ' ' || s.data[i] == '$' || s.data[i] == '\t')) i++;
    if (i >= s.len) return false;
    char buf[64];
    int len = s.len - i;
    if (len >= (int)sizeof(buf)) return false;
    memcpy(buf, s.data + i, (size_t)len);
    buf[len] = '\0';
    char* endptr = NULL;
    double v = strtod(buf, &endptr);
    if (endptr != buf && (*endptr == '\0' || *endptr == ' ' || *endptr == '%')) {
        *out_val = v;
        return true;
    }
    return false;
}

static inline int _vira_table_compare_cells(vira_string a, vira_string b) {
    double na = 0.0, nb = 0.0;
    if (_vira_is_numeric_str(a, &na) && _vira_is_numeric_str(b, &nb)) {
        if (na < nb) return -1;
        if (na > nb) return 1;
        return 0;
    }
    int min_len = a.len < b.len ? a.len : b.len;
    for (int i = 0; i < min_len; i++) {
        int ca = tolower((unsigned char)a.data[i]);
        int cb = tolower((unsigned char)b.data[i]);
        if (ca != cb) return ca - cb;
    }
    return a.len - b.len;
}

static inline void _vira_gui_draw_clipped_text(vira_string text, int32_t x, int32_t y, int32_t size, int32_t color, int32_t max_w) {
    if (text.len <= 0 || !text.data || max_w <= 10) return;
    int tw = _vira_measure_text_w(text);
    if (tw <= max_w) {
        vira_gui_draw_text(text, x, y, size, color);
        return;
    }
    char buf[128];
    int copy_len = text.len < 120 ? text.len : 120;
    while (copy_len > 0) {
        memcpy(buf, text.data, (size_t)copy_len);
        buf[copy_len] = '.';
        buf[copy_len + 1] = '.';
        buf[copy_len + 2] = '\0';
        vira_string s = (vira_string){ .data = (const uint8_t*)buf, .len = copy_len + 2, .is_lit = true };
        if (_vira_measure_text_w(s) <= max_w) {
            vira_gui_draw_text(s, x, y, size, color);
            return;
        }
        copy_len--;
    }
}

static inline int32_t vira_gui_table(int32_t x, int32_t y, int32_t width, int32_t height,
                                    vira_slice_string headers, vira_slice_vira_slice_string rows,
                                    vira_string search_query, int32_t selected_row) {
    if (headers.len <= 0) return selected_row;

    int32_t id = _vira_table_counter++;
    if (id >= VIRA_MAX_TABLES) id = VIRA_MAX_TABLES - 1;
    vira_table_state_t* state = &_vira_table_states[id];

    if (!state->initialized) {
        state->sort_col = 0;
        state->sort_asc = true;
        state->page = 0;
        state->selected_row = selected_row;
        state->last_search_len = search_query.len;
        state->initialized = true;
    }

    if (selected_row >= 0 && selected_row != state->selected_row) {
        state->selected_row = selected_row;
    }

    if (search_query.len != state->last_search_len) {
        state->page = 0;
        state->last_search_len = search_query.len;
    }

    // Filter matching rows
    int32_t filtered[2048];
    int32_t filtered_count = 0;
    for (int r = 0; r < rows.len && filtered_count < 2048; r++) {
        if (search_query.len == 0) {
            filtered[filtered_count++] = r;
        } else {
            bool matched = false;
            for (int c = 0; c < rows.data[r].len; c++) {
                if (_vira_table_str_contains_ci(rows.data[r].data[c], search_query)) {
                    matched = true;
                    break;
                }
            }
            if (matched) {
                filtered[filtered_count++] = r;
            }
        }
    }

    // Sort matching rows
    if (state->sort_col >= 0 && state->sort_col < headers.len && filtered_count > 1) {
        for (int i = 0; i < filtered_count - 1; i++) {
            int best = i;
            for (int j = i + 1; j < filtered_count; j++) {
                int r_a = filtered[j];
                int r_b = filtered[best];
                vira_string str_a = (state->sort_col < rows.data[r_a].len) ? rows.data[r_a].data[state->sort_col] : (vira_string){0};
                vira_string str_b = (state->sort_col < rows.data[r_b].len) ? rows.data[r_b].data[state->sort_col] : (vira_string){0};
                int cmp = _vira_table_compare_cells(str_a, str_b);
                if (!state->sort_asc) cmp = -cmp;
                if (cmp < 0) {
                    best = j;
                }
            }
            if (best != i) {
                int tmp = filtered[i];
                filtered[i] = filtered[best];
                filtered[best] = tmp;
            }
        }
    }

    int header_h = 32;
    int footer_h = 28;
    int row_h = 28;
    int avail_h = height - header_h - footer_h;
    int visible_rows = avail_h / row_h;
    if (visible_rows < 1) visible_rows = 1;

    int total_pages = (filtered_count + visible_rows - 1) / visible_rows;
    if (total_pages < 1) total_pages = 1;
    if (state->page >= total_pages) state->page = total_pages - 1;
    if (state->page < 0) state->page = 0;

    int start_row = state->page * visible_rows;
    int end_row = start_row + visible_rows;
    if (end_row > filtered_count) end_row = filtered_count;

    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();
    int col_w = (width - 2) / headers.len;
    if (col_w < 50) col_w = 50;

    // Table Container Background
    vira_gui_draw_rounded_rect(x, y, width, height, 6.0f, 0x181825, 0x313244);

    // Header Row
    for (int c = 0; c < headers.len; c++) {
        int cx = x + 1 + c * col_w;
        int cw = (c == headers.len - 1) ? (width - 2 - c * col_w) : col_w;
        bool is_col_hover = (mx >= cx && mx < cx + cw && my >= y && my < y + header_h);

        if (is_col_hover && _vira_gui_mouse_clicked_left()) {
            if (state->sort_col == c) {
                state->sort_asc = !state->sort_asc;
            } else {
                state->sort_col = c;
                state->sort_asc = true;
            }
            _vira_click_consumed = true;
        }

        int32_t header_bg = is_col_hover ? 0x313244 : 0x181825;
        vira_gui_draw_rect(cx, y + 1, cw, header_h - 1, header_bg);

        // Header text + sort indicator
        char hbuf[128];
        if (state->sort_col == c) {
            snprintf(hbuf, sizeof(hbuf), "%.*s %s", headers.data[c].len, (const char*)headers.data[c].data,
                     state->sort_asc ? "^" : "v");
        } else {
            snprintf(hbuf, sizeof(hbuf), "%.*s", headers.data[c].len, (const char*)headers.data[c].data);
        }
        vira_string hs = (vira_string){ .data = (const uint8_t*)hbuf, .len = (int32_t)strlen(hbuf), .is_lit = true };
        int32_t hcolor = (state->sort_col == c) ? 0x89B4FA : (is_col_hover ? 0xFFFFFF : 0xCDD6F4);
        _vira_gui_draw_clipped_text(hs, cx + 10, y + (header_h - 16) / 2, 14, hcolor, cw - 14);

        if (c > 0) {
            vira_gui_draw_line(cx, y, cx, y + header_h, 0x313244);
        }
    }
    vira_gui_draw_line(x, y + header_h, x + width, y + header_h, 0x313244);

    // Data Rows
    for (int vi = start_row; vi < end_row; vi++) {
        int orig_r = filtered[vi];
        int ry = y + header_h + (vi - start_row) * row_h;
        bool is_hover = (mx >= x && mx < x + width && my >= ry && my < ry + row_h);
        bool is_sel = (state->selected_row == orig_r);

        if (is_hover && _vira_gui_mouse_clicked_left()) {
            state->selected_row = orig_r;
            selected_row = orig_r;
            is_sel = true;
            _vira_click_consumed = true;
        }

        int32_t row_bg = is_sel ? 0x313244 : (is_hover ? 0x242537 : ((vi % 2 == 1) ? 0x181825 : 0x1E1E2E));
        vira_gui_draw_rect(x + 1, ry, width - 2, row_h, row_bg);

        if (is_sel) {
            vira_gui_draw_rect(x + 1, ry, 3, row_h, 0x89B4FA);
        }

        for (int c = 0; c < headers.len; c++) {
            int cx = x + 1 + c * col_w;
            int cw = (c == headers.len - 1) ? (width - 2 - c * col_w) : col_w;
            if (c < rows.data[orig_r].len) {
                vira_string val = rows.data[orig_r].data[c];
                int32_t tcolor = is_sel ? 0xFFFFFF : (is_hover ? 0xCDD6F4 : 0xBAC2DE);
                _vira_gui_draw_clipped_text(val, cx + 10, ry + (row_h - 15) / 2, 14, tcolor, cw - 14);
            }
            if (c > 0) {
                vira_gui_draw_line(cx, ry, cx, ry + row_h, 0x222436);
            }
        }
    }

    // Status Bar & Pagination Footer
    int fy = y + height - footer_h;
    vira_gui_draw_line(x, fy, x + width, fy, 0x313244);
    vira_gui_draw_rect(x + 1, fy + 1, width - 2, footer_h - 2, 0x181825);

    char sbuf[128];
    if (filtered_count == 0) {
        snprintf(sbuf, sizeof(sbuf), "No records found");
    } else {
        snprintf(sbuf, sizeof(sbuf), "Showing %d-%d of %d%s", start_row + 1, end_row, filtered_count,
                 (filtered_count != rows.len) ? " (filtered)" : "");
    }
    vira_string ss = (vira_string){ .data = (const uint8_t*)sbuf, .len = (int32_t)strlen(sbuf), .is_lit = true };
    vira_gui_draw_text(ss, x + 12, fy + 6, 13, 0x6C7086);

    // Pagination controls: [ < ] [Page X / Y] [ > ]
    int btn_w = 24;
    int btn_h = 20;
    int by = fy + (footer_h - btn_h) / 2;
    int next_x = x + width - 34;
    int page_x = next_x - 64;
    int prev_x = page_x - btn_w - 4;

    bool prev_avail = (state->page > 0);
    bool prev_hover = (mx >= prev_x && mx <= prev_x + btn_w && my >= by && my <= by + btn_h);
    if (prev_avail && prev_hover && _vira_gui_mouse_clicked_left()) {
        state->page--;
        _vira_click_consumed = true;
    }
    int32_t prev_bg = !prev_avail ? 0x181825 : (prev_hover ? 0x45475A : 0x1E1E2E);
    int32_t prev_fg = !prev_avail ? 0x45475A : (prev_hover ? 0xFFFFFF : 0xCDD6F4);
    vira_gui_draw_rounded_rect(prev_x, by, btn_w, btn_h, 4.0f, prev_bg, 0x313244);
    vira_string prev_str = (vira_string){ .data = (const uint8_t*)"<", .len = 1, .is_lit = true };
    vira_gui_draw_text(prev_str, prev_x + 8, by + 2, 13, prev_fg);

    char pbuf[32];
    snprintf(pbuf, sizeof(pbuf), "%d / %d", state->page + 1, total_pages);
    vira_string ps = (vira_string){ .data = (const uint8_t*)pbuf, .len = (int32_t)strlen(pbuf), .is_lit = true };
    int pw = _vira_measure_text_w(ps);
    vira_gui_draw_text(ps, page_x + (60 - pw) / 2, fy + 6, 13, 0xA6ADC8);

    bool next_avail = (state->page < total_pages - 1);
    bool next_hover = (mx >= next_x && mx <= next_x + btn_w && my >= by && my <= by + btn_h);
    if (next_avail && next_hover && _vira_gui_mouse_clicked_left()) {
        state->page++;
        _vira_click_consumed = true;
    }
    int32_t next_bg = !next_avail ? 0x181825 : (next_hover ? 0x45475A : 0x1E1E2E);
    int32_t next_fg = !next_avail ? 0x45475A : (next_hover ? 0xFFFFFF : 0xCDD6F4);
    vira_gui_draw_rounded_rect(next_x, by, btn_w, btn_h, 4.0f, next_bg, 0x313244);
    vira_string next_str = (vira_string){ .data = (const uint8_t*)">", .len = 1, .is_lit = true };
    vira_gui_draw_text(next_str, next_x + 8, by + 2, 13, next_fg);

    // Outer Table Border
    vira_gui_draw_rounded_rect(x, y, width, height, 6.0f, -1, 0x313244);

    _vira_last_table_sort_col = state->sort_col;
    _vira_last_table_sort_asc = state->sort_asc;

    return state->selected_row;
}

static inline int32_t vira_gui_tbl(vira_slice_string headers, vira_slice_vira_slice_string rows,
                                   vira_string search_query, int32_t selected_row) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return selected_row;
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t w = node->width;
    int32_t h = (node->type == VIRA_LAYOUT_HORIZONTAL) ? node->height : 260;
    if (h < 120) h = 120;

    int32_t res = vira_gui_table(x, y, w, h, headers, rows, search_query, selected_row);

    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        node->cur_x += w + node->spacing;
    } else {
        node->cur_y += h + node->spacing;
    }
    return res;
}

static inline int32_t vira_gui_table_sort_col(void) {
    return _vira_last_table_sort_col;
}

static inline bool vira_gui_table_sort_asc(void) {
    return _vira_last_table_sort_asc;
}

static inline void vira_gui_table_set_sort(int32_t col, bool asc) {
    _vira_last_table_sort_col = col;
    _vira_last_table_sort_asc = asc;
    for (int i = 0; i < VIRA_MAX_TABLES; i++) {
        _vira_table_states[i].sort_col = col;
        _vira_table_states[i].sort_asc = asc;
    }
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

static inline bool vira_gui_btn_danger_temp(vira_string label) {
    bool res = vira_gui_btn_danger(label);
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

static inline int32_t vira_gui_table_temp(int32_t x, int32_t y, int32_t width, int32_t height,
                                         vira_slice_string headers, vira_slice_vira_slice_string rows,
                                         vira_string search_query, int32_t selected_row) {
    int32_t res = vira_gui_table(x, y, width, height, headers, rows, search_query, selected_row);
    vira_str_free(&search_query);
    return res;
}

static inline int32_t vira_gui_tbl_temp(vira_slice_string headers, vira_slice_vira_slice_string rows,
                                       vira_string search_query, int32_t selected_row) {
    int32_t res = vira_gui_tbl(headers, rows, search_query, selected_row);
    vira_str_free(&search_query);
    return res;
}

static inline int32_t vira_gui_load_image_temp(vira_string path) {
    int32_t res = vira_gui_load_image(path);
    vira_str_free(&path);
    return res;
}

#endif // VIRA_GUI_H