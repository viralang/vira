#ifndef VIRA_GUI_TYPES_H
#define VIRA_GUI_TYPES_H

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

#endif // VIRA_GUI_TYPES_H
