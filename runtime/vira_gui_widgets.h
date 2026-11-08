#ifndef VIRA_GUI_WIDGETS_H
#define VIRA_GUI_WIDGETS_H

// ------------------------------------------------------------------------------
// Unified Cross-Platform Image Dispatch
// ------------------------------------------------------------------------------
/**
 * Loads an image file (PNG, JPG, BMP) from disk into the GUI image cache and backend texture system.
 * @param path Vira string holding the filesystem path to the image file.
 * @return Non-negative integer image ID handle on success, or -1 on failure.
 * Usage: Called by Vira GUI applications to load textures for display via `gui::image()` or `gui::draw_image()`.
 */
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

/**
 * Retrieves the pixel width of a previously loaded image handle.
 * @param id Non-negative image ID returned by `vira_gui_load_image`.
 * @return Width in pixels, or 0 if the image ID is invalid or not loaded.
 * Usage: Used to compute aspect ratios or layout geometry when rendering images.
 */
static inline int32_t vira_gui_image_width(int32_t id) {
    if (id >= 0 && id < VIRA_MAX_GUI_IMAGES && _vira_gui_images[id].is_valid) {
        return _vira_gui_images[id].width;
    }
    return 0;
}

/**
 * Retrieves the pixel height of a previously loaded image handle.
 * @param id Non-negative image ID returned by `vira_gui_load_image`.
 * @return Height in pixels, or 0 if the image ID is invalid or not loaded.
 * Usage: Used to compute aspect ratios or layout geometry when rendering images.
 */
static inline int32_t vira_gui_image_height(int32_t id) {
    if (id >= 0 && id < VIRA_MAX_GUI_IMAGES && _vira_gui_images[id].is_valid) {
        return _vira_gui_images[id].height;
    }
    return 0;
}

/**
 * Draws an image at the specified pixel coordinates with optional width and height scaling.
 * @param id Valid image handle ID.
 * @param x Screen X coordinate in pixels.
 * @param y Screen Y coordinate in pixels.
 * @param w Destination width (or <= 0 to use native image width).
 * @param h Destination height (or <= 0 to use native image height).
 * Usage: Direct manual rendering of textures/images in immediate-mode GUI loops.
 */
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

/**
 * Frees an image from memory, releasing underlying backend textures and slot cache.
 * @param id Valid image handle ID to release.
 * Usage: Called when an image is no longer needed to reclaim CPU/GPU texture memory.
 */
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

// ------------------------------------------------------------------------------
// Unified Cross-Platform Scissor / Clipping Dispatch
// ------------------------------------------------------------------------------
/**
 * Restricts subsequent rendering operations to a rectangular scissor/clipping region.
 * @param x Left coordinate of the clipping rectangle.
 * @param y Top coordinate of the clipping rectangle.
 * @param w Width of the clipping rectangle in pixels.
 * @param h Height of the clipping rectangle in pixels.
 * Usage: Used internally by scroll containers and modals to prevent child widgets from overflowing their bounds.
 */
static inline void vira_gui_set_clip(int32_t x, int32_t y, int32_t w, int32_t h) {
#if defined(VIRA_USE_RAYLIB)
    _vira_raylib_set_clip(x, y, w, h);
#elif defined(_WIN32)
    _vira_win32_set_clip(x, y, w, h);
#elif (defined(__linux__) || defined(__unix__)) && !defined(VIRA_NO_X11)
    _vira_x11_set_clip(x, y, w, h);
#elif defined(__APPLE__) && defined(__OBJC__)
    _vira_cocoa_set_clip(x, y, w, h);
#else
    _vira_headless_set_clip(x, y, w, h);
#endif
}

/**
 * Removes any active scissor/clipping rectangle, restoring rendering to the entire window surface.
 * Usage: Called after completing rendering inside a scroll container or clipped view.
 */
static inline void vira_gui_reset_clip(void) {
#if defined(VIRA_USE_RAYLIB)
    _vira_raylib_reset_clip();
#elif defined(_WIN32)
    _vira_win32_reset_clip();
#elif (defined(__linux__) || defined(__unix__)) && !defined(VIRA_NO_X11)
    _vira_x11_reset_clip();
#elif defined(__APPLE__) && defined(__OBJC__)
    _vira_cocoa_reset_clip();
#else
    _vira_headless_reset_clip();
#endif
}

/**
 * Returns the vertical mouse wheel scroll delta accumulated during the current frame.
 * @return Vertical scroll wheel delta (positive = scroll up, negative = scroll down).
 * Usage: Used by scrollable containers and custom widgets to support mouse wheel scrolling.
 */
static inline float vira_gui_mouse_wheel(void) {
    return _vira_mouse_wheel_delta;
}

// ==============================================================================
// 3. SHARED WIDGETS & DECLARATIVE AUTO-LAYOUT ENGINE
// ==============================================================================

/**
 * Internal helper to check if the left mouse button was clicked without being consumed by an active modal or overlay.
 * @return True if left button was clicked on unconsumed interactive elements.
 * Usage: Internal event gate for button, dropdown, and checkbox click detection.
 */
static inline bool _vira_gui_mouse_clicked_left(void) {
    if (_vira_click_consumed) return false;
    if (_vira_modal_active) return false;
    return _vira_backend_mouse_clicked_left();
}

static inline void _vira_gui_draw_deferred_popups(void);

/**
 * Initializes the frame layout stack, input focus navigation, and backend render targets.
 * Usage: Must be called at the beginning of every frame loop before any GUI widgets or layout declarations.
 */
static inline void vira_gui_begin(void) {
    _vira_layout_depth = 0;
    _vira_input_counter = 0;
    _vira_card_counter = 0;
    _vira_card_end_counter = 0;
    _vira_dropdown_counter = 0;
    _vira_table_counter = 0;
    _vira_scroll_counter = 0;
    _vira_click_consumed = false;
    _vira_modal_active = _vira_modal_open_prev_frame;
    _vira_modal_first_input = -1;
    _vira_modal_last_input = -1;

    _vira_backend_gui_begin();

    // Keyboard Tab Focus Navigation
    int32_t total = _vira_total_inputs > 0 ? _vira_total_inputs : _vira_input_counter;
    if (_vira_tab_pressed && total > 0) {
        int32_t min_id = 0;
        int32_t max_id = total - 1;
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

/**
 * Finalizes the frame, draws deferred popups/modals, updates input focus states, and presents the backbuffer.
 * Usage: Must be called at the end of every frame loop after all GUI drawing commands.
 */
static inline void vira_gui_end(void) {
    _vira_gui_draw_deferred_popups();
    _vira_total_inputs = _vira_input_counter;
    _vira_prev_modal_first_input = _vira_modal_first_input;
    _vira_prev_modal_last_input = _vira_modal_last_input;
    _vira_modal_open_prev_frame = _vira_modal_open_this_frame;
    _vira_modal_open_this_frame = false;
    _vira_backend_gui_end();
}

/**
 * Renders an immediate-mode interactive push button with hover/active animations at explicit coordinates.
 * @param x Screen X coordinate in pixels.
 * @param y Screen Y coordinate in pixels.
 * @param width Button width in pixels.
 * @param height Button height in pixels.
 * @param label Text string displayed on the button.
 * @return True if the button was clicked during the current frame, false otherwise.
 * Usage: Used for explicit manual positioning of clickable push buttons.
 */
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

/**
 * Internal helper to draw a mini vector trash can icon for danger buttons.
 * @param x Left coordinate in pixels.
 * @param y Top coordinate in pixels.
 * @param color 24-bit RGB hex color for icon strokes.
 * Usage: Renders danger/delete button icons without requiring external image assets.
 */
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

/**
 * Renders an immediate-mode danger/delete button featuring a trash icon and warning color palette.
 * @param x Screen X coordinate in pixels.
 * @param y Screen Y coordinate in pixels.
 * @param width Button width in pixels.
 * @param height Button height in pixels.
 * @param label Text label displayed beside the icon.
 * @return True if clicked during the current frame.
 * Usage: Used for destructive or irreversible actions (delete, clear, discard).
 */
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

/**
 * Renders an immediate-mode horizontal slider control with value dragging and optional labels.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param width Total width of the slider track.
 * @param height Slider control height.
 * @param text_left Optional label text displayed on the left.
 * @param text_right Optional label or value text displayed on the right.
 * @param val Current float value of the slider.
 * @param min_val Minimum bound of the slider range.
 * @param max_val Maximum bound of the slider range.
 * @return Updated float value after mouse interaction.
 * Usage: Used for numeric range input, volume, temperature, thresholds, etc.
 */
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

/**
 * Renders an interactive checkbox with toggle state and label at explicit coordinates.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param width Bounding box width.
 * @param height Bounding box height.
 * @param label Text label displayed next to the box.
 * @param is_checked Current boolean state.
 * @return New boolean state toggled on click.
 * Usage: Used for toggling boolean flags or settings in manual layouts.
 */
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

static inline bool _vira_str_has_newline(vira_string s) {
    if (!s.data || s.len <= 0) return false;
    for (int32_t i = 0; i < s.len; i++) {
        if (s.data[i] == '\n' || s.data[i] == '\r') return true;
    }
    return false;
}

static inline int32_t _vira_measure_text_size_w(vira_string text, int32_t size) {
    if (size <= 0) size = 15;
    if (text.len <= 0 || !text.data) return 0;
#if defined(VIRA_USE_RAYLIB)
    char stack_buf[256];
    char *buf = stack_buf;
    if (text.len >= 256) {
        buf = (char *)malloc((size_t)text.len + 1);
        if (!buf) return text.len * (size * 8 / 15);
    }
    memcpy(buf, text.data, (size_t)text.len);
    buf[text.len] = '\0';
    int32_t w = 0;
    if (_vira_rl_font_loaded && _vira_rl_font.texture.id > 0) {
        Vector2 sz = MeasureTextEx(_vira_rl_font, buf, (float)size, 1.0f);
        w = (int32_t)ceilf(sz.x);
    } else {
        w = MeasureText(buf, size);
    }
    if (buf != stack_buf) free(buf);
    return w;
#else
    int32_t base_w = _vira_measure_text_w(text);
    if (size == 15) return base_w;
    return (base_w * size + 7) / 15;
#endif
}

/**
 * Internal helper that truncates and ellipses text ("...") if it exceeds available pixel width.
 * @param text String to draw.
 * @param x Left coordinate in pixels.
 * @param y Top coordinate in pixels.
 * @param size Font size in points.
 * @param color Text RGB hex color.
 * @param max_w Maximum allowed pixel width.
 * Usage: Prevents titles, table cells, or narrow labels from overflowing column boundaries.
 */
static inline void _vira_gui_draw_clipped_text(vira_string text, int32_t x, int32_t y, int32_t size, int32_t color, int32_t max_w) {
    if (text.len <= 0 || !text.data || max_w <= 10) return;
    int tw = _vira_measure_text_size_w(text, size);
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
        if (_vira_measure_text_size_w(s, size) <= max_w) {
            vira_gui_draw_text(s, x, y, size, color);
            return;
        }
        copy_len--;
    }
}

/**
 * Internal word-wrapping and multi-line text layout engine.
 * Breaks text into lines respecting existing newlines (\n, \r\n), wrapping long lines at word
 * boundaries or character boundaries if words exceed max_w, and enforcing window boundaries.
 * @param text String to render.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param max_w Available width before wrapping.
 * @param size Font size in points (e.g. 15 or 16).
 * @param hex_color Text color in 24-bit RGB.
 * @return Total height in pixels consumed by all rendered lines.
 */
static inline int32_t _vira_gui_draw_text_wrapped(vira_string text, int32_t x, int32_t y, int32_t max_w, int32_t size, int32_t hex_color) {
    if (text.len <= 0 || !text.data) return 0;

    int32_t fsize = size > 0 ? size : 15;
    int32_t line_h = fsize + 6;
    if (max_w <= 30) {
        vira_gui_draw_text(text, x, y, fsize, hex_color);
        return line_h;
    }

    // Fast path: single-line without newline that fits within max_w
    if (!_vira_str_has_newline(text) && _vira_measure_text_size_w(text, fsize) <= max_w) {
        vira_gui_draw_text(text, x, y, fsize, hex_color);
        return line_h;
    }

    int32_t cur_y = y;
    int32_t len = text.len;
    const uint8_t* data = text.data;
    int32_t pos = 0;

    while (pos < len) {
        // Find paragraph boundary (newline)
        int32_t para_end = pos;
        while (para_end < len && data[para_end] != '\n' && data[para_end] != '\r') {
            para_end++;
        }

        if (para_end == pos) {
            // Empty line (\n\n)
            cur_y += line_h;
        } else {
            int32_t p = pos;
            while (p < para_end) {
                // Skip leading whitespace at line start
                while (p < para_end && (data[p] == ' ' || data[p] == '\t')) {
                    p++;
                }
                if (p >= para_end) break;

                int32_t line_start = p;
                int32_t last_fit_end = p;
                int32_t scan = p;

                while (scan < para_end) {
                    // Find next word end
                    int32_t word_start = scan;
                    while (scan < para_end && data[scan] != ' ' && data[scan] != '\t') {
                        scan++;
                    }
                    int32_t word_end = scan;
                    while (scan < para_end && (data[scan] == ' ' || data[scan] == '\t')) {
                        scan++;
                    }

                    // Candidate line from line_start to word_end
                    vira_string cand = (vira_string){
                        .data = &data[line_start],
                        .len = word_end - line_start,
                        .is_lit = true
                    };
                    int32_t cw = _vira_measure_text_size_w(cand, fsize);

                    if (cw <= max_w) {
                        last_fit_end = word_end;
                        p = scan;
                    } else {
                        // Word makes line exceed max_w
                        if (last_fit_end > line_start) {
                            // Break at last fitting word
                            break;
                        } else {
                            // First word on this line exceeds max_w on its own! Break character-by-character
                            int32_t char_end = word_start + 1;
                            while (char_end <= word_end) {
                                vira_string sub = (vira_string){
                                    .data = &data[line_start],
                                    .len = char_end - line_start,
                                    .is_lit = true
                                };
                                if (_vira_measure_text_size_w(sub, fsize) > max_w) {
                                    break;
                                }
                                char_end++;
                            }
                            char_end--;
                            if (char_end <= line_start) char_end = line_start + 1;
                            last_fit_end = char_end;
                            p = char_end;
                            break;
                        }
                    }
                }

                if (last_fit_end > line_start) {
                    vira_string emit_line = (vira_string){
                        .data = &data[line_start],
                        .len = last_fit_end - line_start,
                        .is_lit = true
                    };
                    vira_gui_draw_text(emit_line, x, cur_y, fsize, hex_color);
                    cur_y += line_h;
                } else {
                    // Safety guard: ensure forward progress
                    p++;
                }
            }
        }

        // Advance past newline characters
        pos = para_end;
        if (pos < len && data[pos] == '\r') pos++;
        if (pos < len && data[pos] == '\n') pos++;
    }

    int32_t total_h = cur_y - y;
    return total_h > 0 ? total_h : line_h;
}

/**
 * Draws wrapped multi-line text with an explicit maximum bounding width, font size, and color.
 * Automatically constrains rendering within window and container boundaries.
 * @param text Text string to display.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param max_width Maximum bounding width before wrapping (or <= 0 for auto window bounds).
 * @param size Font size in points.
 * @param hex_color 24-bit RGB text color.
 * Usage: Called by `gui_draw_text_wrapped(text, x, y, max_width, size, color)`.
 */
static inline void vira_gui_draw_text_wrapped(vira_string text, int32_t x, int32_t y, int32_t max_width, int32_t size, int32_t hex_color) {
    int32_t win_w = vira_gui_window_width();
    int32_t max_w = max_width > 0 ? max_width : (win_w > x + 30 ? win_w - x - 20 : 400);
    if (win_w > 0 && x + max_w > win_w - 15) {
        max_w = win_w - x - 15;
    }
    if (max_w < 50) max_w = 50;
    _vira_gui_draw_text_wrapped(text, x, y, max_w, size > 0 ? size : 16, hex_color);
}

/**
 * Renders static text at explicit coordinates with automatic wrapping within bounding width.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param width Bounding width (or <= 0 for window bounds).
 * @param height Bounding height (or <= 0 for unconstrained height).
 * @param text Text string to display.
 * Usage: Used for displaying informational text, paragraphs, or captions with guaranteed wrapping.
 */
static inline void vira_gui_label(int32_t x, int32_t y, int32_t width, int32_t height, vira_string text) {
    int32_t win_w = vira_gui_window_width();
    int32_t max_w = width > 0 ? width : (win_w > x + 30 ? win_w - x - 20 : 400);
    if (win_w > 0 && x + max_w > win_w - 15) {
        max_w = win_w - x - 15;
    }
    if (max_w < 50) max_w = 50;

    if (height > 0) {
        vira_gui_set_clip(x, y, max_w, height);
    }
    _vira_gui_draw_text_wrapped(text, x, y, max_w, 16, 0xCDD6F4);
    if (height > 0) {
        vira_gui_reset_clip();
    }
}

/**
 * Renders an immediate-mode single-line text input field supporting focus, caret, typing, backspace, selection, and clipboard.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param width Field width in pixels.
 * @param height Field height in pixels.
 * @param placeholder Hint text displayed when empty and unfocused.
 * @param text Current text content of the input field.
 * @return Updated `vira_string` containing the edited text.
 * Usage: Used for single-line user input fields, search bars, and forms.
 */
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
    int max_text_w = width - pad_x - 10;
    if (max_text_w < 10) max_text_w = 10;

    vira_gui_set_clip(x + 4, y + 2, width - 8, height - 4);

    if (state->len > 0) {
        vira_string s_val = (vira_string){ .data = (const uint8_t*)state->buffer, .len = state->len, .is_lit = true };
        int tw = _vira_measure_text_w(s_val);
        int offset_x = (is_focused && tw > max_text_w) ? (max_text_w - tw) : 0;

        vira_gui_draw_text(s_val, x + pad_x + offset_x, text_y, 15, 0xCDD6F4);

        if (is_focused && ((_vira_cursor_blink_counter / 30) % 2 == 0)) {
            int cx = x + pad_x + offset_x + tw + 2;
            vira_gui_draw_line(cx, text_y, cx, text_y + 16, 0x89B4FA);
        }
    } else {
        if (placeholder.len > 0) {
            vira_gui_draw_text(placeholder, x + pad_x, text_y, 15, 0x6C7086);
        }
        if (is_focused && ((_vira_cursor_blink_counter / 30) % 2 == 0)) {
            vira_gui_draw_line(x + pad_x, text_y, x + pad_x, text_y + 16, 0x89B4FA);
        }
    }

    vira_gui_reset_clip();

    return (vira_string){ .data = (const uint8_t*)state->buffer, .len = state->len, .is_lit = true };
}

/**
 * Renders a horizontal progress bar showing completion percentage.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param width Progress bar width.
 * @param height Progress bar height.
 * @param percent Completion fraction between 0.0f and 1.0f.
 * Usage: Displays loading states, generation progress, or download status.
 */
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

/**
 * Renders an interactive radio button option with circular indicator.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param label Option text label.
 * @param is_selected Current selection state.
 * @return True if the radio button was selected by user click.
 * Usage: Used for mutually exclusive option groups.
 */
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

/**
 * Renders a styled rounded badge/pill with custom background and foreground colors.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param text Badge label text.
 * @param hex_bg 24-bit RGB background color.
 * @param hex_fg 24-bit RGB text color.
 * Usage: Used for status indicators, tag labels, counts, or categorization tags.
 */
static inline void vira_gui_draw_badge(int32_t x, int32_t y, vira_string text, int32_t hex_bg, int32_t hex_fg) {
    int32_t tw = _vira_measure_text_w(text);
    int32_t bw = tw + 16;
    int32_t bh = 22;
    int32_t border = ((hex_bg >> 1) & 0x7F7F7F);

    vira_gui_draw_rounded_rect(x, y, bw, bh, 6.0f, hex_bg, border);
    vira_gui_draw_text(text, x + 8, y + 3, 13, hex_fg);
}

/**
 * Begins a vertical declarative layout stack container (VStack).
 * @param x Origin X coordinate.
 * @param y Origin Y coordinate.
 * @param width Total container width.
 * @param spacing Vertical gap in pixels between child widgets.
 * Usage: Encloses vertical auto-layout widget sequences; closed with `vira_gui_stack_end()`.
 */
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

/**
 * Begins a horizontal declarative layout stack container (HStack) within the current parent layout.
 * @param height Uniform height of widgets inside the horizontal row.
 * @param spacing Horizontal gap in pixels between child widgets.
 * Usage: Encloses horizontal auto-layout widget rows; closed with `vira_gui_stack_end()`.
 */
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

/**
 * Closes the most recently opened declarative layout stack (VStack, HStack, or Scroll container).
 * Usage: Must match every preceding `_begin` call in the declarative layout hierarchy.
 */
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

/**
 * Begins a scrollable container with automatic clipping, mouse-wheel scrolling, and vertical scrollbar.
 * @param width Visible viewport width in pixels.
 * @param height Visible viewport height in pixels.
 * Usage: Wraps scrollable lists, logs, or chat histories; closed with `vira_gui_scroll_end()`.
 */
static inline void vira_gui_scroll_begin(int32_t width, int32_t height) {
    vira_layout_node_t* parent = _vira_gui_cur_layout();
    int32_t x = parent ? parent->cur_x : 0;
    int32_t y = parent ? parent->cur_y : 0;
    int32_t w = width > 0 ? width : (parent ? parent->width : (vira_gui_window_width() - x));
    int32_t h = height > 0 ? height : (parent ? (vira_gui_window_height() - y) : vira_gui_window_height());
    if (h < 40) h = 40;

    int32_t sid = _vira_scroll_counter++;
    if (sid >= VIRA_MAX_SCROLL_VIEWS) sid = VIRA_MAX_SCROLL_VIEWS - 1;
    vira_scroll_view_state_t* sv = &_vira_scroll_views[sid];

    int mx = vira_gui_mouse_x();
    int my = vira_gui_mouse_y();

    // Mouse wheel scrolling
    if (mx >= x && mx <= x + w && my >= y && my <= y + h) {
        if (_vira_mouse_wheel_delta != 0.0f) {
            sv->scroll_y -= (int32_t)(_vira_mouse_wheel_delta * 45.0f);
        }
    }

    int32_t max_scroll = sv->content_h > h ? (sv->content_h - h) : 0;
    int32_t track_x = x + w - 8;
    int32_t track_w = 6;
    int32_t track_h = h;

    if (max_scroll > 0) {
        int32_t thumb_h = (track_h * track_h) / sv->content_h;
        if (thumb_h < 24) thumb_h = 24;
        if (thumb_h > track_h) thumb_h = track_h;
        int32_t thumb_y = y + (sv->scroll_y * (track_h - thumb_h)) / max_scroll;

        if (sv->is_dragging) {
            if (vira_gui_is_mouse_down(0)) {
                int32_t dy = my - sv->drag_start_y;
                int32_t scrollable_track = track_h - thumb_h;
                if (scrollable_track > 0) {
                    sv->scroll_y = sv->drag_start_scroll + (dy * max_scroll) / scrollable_track;
                }
            } else {
                sv->is_dragging = false;
            }
        } else if (_vira_gui_mouse_clicked_left()) {
            if (mx >= track_x - 6 && mx <= track_x + track_w + 6 && my >= y && my <= y + h) {
                if (my >= thumb_y && my <= thumb_y + thumb_h) {
                    sv->is_dragging = true;
                    sv->drag_start_y = my;
                    sv->drag_start_scroll = sv->scroll_y;
                    _vira_click_consumed = true;
                } else {
                    if (my < thumb_y) {
                        sv->scroll_y -= h / 2;
                    } else {
                        sv->scroll_y += h / 2;
                    }
                    _vira_click_consumed = true;
                }
            }
        }
    } else {
        sv->is_dragging = false;
        sv->scroll_y = 0;
    }

    if (sv->scroll_y > max_scroll) sv->scroll_y = max_scroll;
    if (sv->scroll_y < 0) sv->scroll_y = 0;

    // Apply hardware scissor / clip
    vira_gui_set_clip(x, y, w, h);

    if (_vira_layout_depth < VIRA_MAX_LAYOUT_DEPTH) {
        vira_layout_node_t* node = &_vira_layout_stack[_vira_layout_depth++];
        node->type = VIRA_LAYOUT_VERTICAL;
        node->x = x;
        node->y = y;
        node->cur_x = x;
        node->cur_y = y - sv->scroll_y;
        node->width = (max_scroll > 0) ? (w - 12) : w;
        node->height = h;
        node->spacing = 10;
        node->start_y = y - sv->scroll_y;
        node->is_card = 0;
        node->is_scroll = 1;
        node->scroll_id = sid;
        node->scroll_offset_y = sv->scroll_y;
    }
}

/**
 * Ends a scrollable container, restores clipping, and draws the interactive scrollbar track and thumb.
 * Usage: Must be called to conclude a scroll container opened with `vira_gui_scroll_begin()`.
 */
static inline void vira_gui_scroll_end(void) {
    if (_vira_layout_depth <= 0) return;
    vira_layout_node_t closed = _vira_layout_stack[--_vira_layout_depth];
    int32_t sid = closed.scroll_id;
    if (sid < 0 || sid >= VIRA_MAX_SCROLL_VIEWS) return;
    vira_scroll_view_state_t* sv = &_vira_scroll_views[sid];

    int32_t content_h = closed.cur_y - closed.start_y;
    sv->content_h = content_h;

    // Release hardware scissor / clip
    vira_gui_reset_clip();

    int32_t max_scroll = sv->content_h > closed.height ? (sv->content_h - closed.height) : 0;
    if (max_scroll > 0) {
        int32_t track_x = closed.x + closed.width + 4;
        int32_t track_y = closed.y;
        int32_t track_w = 6;
        int32_t track_h = closed.height;

        vira_gui_draw_rounded_rect(track_x, track_y, track_w, track_h, 3.0f, 0x181825, 0x313244);

        int32_t thumb_h = (track_h * track_h) / sv->content_h;
        if (thumb_h < 24) thumb_h = 24;
        if (thumb_h > track_h) thumb_h = track_h;
        int32_t thumb_y = track_y + (sv->scroll_y * (track_h - thumb_h)) / max_scroll;

        int mx = vira_gui_mouse_x();
        int my = vira_gui_mouse_y();
        bool is_hovered = (mx >= track_x - 6 && mx <= track_x + track_w + 6 && my >= thumb_y && my <= thumb_y + thumb_h);
        int32_t thumb_color = (sv->is_dragging || is_hovered) ? 0x89B4FA : 0x585B70;
        vira_gui_draw_rounded_rect(track_x, thumb_y, track_w, thumb_h, 3.0f, thumb_color, -1);
    }

    vira_layout_node_t* parent = _vira_gui_cur_layout();
    if (parent != NULL) {
        if (parent->type == VIRA_LAYOUT_VERTICAL) {
            parent->cur_y += closed.height + parent->spacing;
        } else if (parent->type == VIRA_LAYOUT_HORIZONTAL) {
            parent->cur_x += closed.width + parent->spacing;
        }
    }
}

/**
 * Inserts blank spacing along the active layout axis (vertical or horizontal).
 * @param px Number of pixels to advance the layout cursor.
 * Usage: Adds margins, separators, or whitespace between widgets in auto-layout stacks.
 */
static inline void vira_gui_spacing(int32_t px) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) return;
    if (node->type == VIRA_LAYOUT_VERTICAL) {
        node->cur_y += px;
    } else {
        node->cur_x += px;
    }
}

/**
 * Begins a visually distinct styled card container with an optional header title.
 * @param title Card header text.
 * @param width Total card width in pixels.
 * Usage: Groups related widgets into an elevated surface; closed with `vira_gui_card_end()`.
 */
static inline void vira_gui_card_begin(vira_string title, int32_t width) {
    vira_layout_node_t* parent = _vira_gui_cur_layout();
    int32_t x = parent ? parent->cur_x : 40;
    int32_t y = parent ? parent->cur_y : 90;
    int32_t w = width > 0 ? width : (parent ? parent->width : 340);

    int32_t c_idx = _vira_card_counter++;
    int32_t card_h = (c_idx < VIRA_MAX_CARDS && _vira_card_heights[c_idx] > 0) ? _vira_card_heights[c_idx] : 140;

    vira_gui_draw_rounded_rect(x, y, w, card_h, 8.0f, 0x1E1E2E, 0x313244);

    if (title.len > 0) {
        _vira_gui_draw_clipped_text(title, x + 15, y + 12, 16, 0x89B4FA, w - 30);
    }

    vira_gui_vstack_begin(x + 15, y + 42, w - 30, 10);
    _vira_layout_stack[_vira_layout_depth - 1].is_card = 1;
}

/**
 * Concludes a card container, drawing its background panel, border, and shadows.
 * Usage: Must be called after children of `vira_gui_card_begin()` have been declared.
 */
static inline void vira_gui_card_end(void) {
    if (_vira_layout_depth <= 0) return;
    vira_layout_node_t closed = _vira_layout_stack[_vira_layout_depth - 1];
    int32_t c_idx = _vira_card_end_counter++;
    int32_t h = (closed.cur_y - closed.start_y) + 42 + 15;
    if (c_idx < VIRA_MAX_CARDS) {
        _vira_card_heights[c_idx] = (h > 60) ? h : 60;
    }
    vira_gui_stack_end();
}

/**
 * Auto-layout button widget placed automatically at the current cursor within a VStack or HStack.
 * @param label Text string displayed on the button.
 * @return True if clicked during the current frame.
 * Usage: Primary concise button API in declarative Vira GUI code (`gui::btn("OK")`).
 */
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

/**
 * Auto-layout danger button widget placed at the current cursor within a VStack or HStack.
 * @param label Text string displayed on the danger button.
 * @return True if clicked during the current frame.
 * Usage: Primary concise danger button API in declarative Vira GUI code (`gui::btn_danger("Delete")`).
 */
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

/**
 * Auto-layout checkbox widget placed at the current layout cursor.
 * @param label Checkbox text label.
 * @param checked Current boolean checked state.
 * @return New boolean checked state.
 * Usage: Primary concise checkbox API (`gui::check("Dark Mode", is_dark)`).
 */
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

/**
 * Auto-layout horizontal slider widget placed at the current layout cursor.
 * @param text_left Label displayed on the left side of the slider.
 * @param text_right Label or formatted value displayed on the right side.
 * @param val Current float value.
 * @param min_val Minimum range bound.
 * @param max_val Maximum range bound.
 * @return Updated float value after interaction.
 * Usage: Primary concise slider API (`gui::slide("Temp", "0.7", temp, 0.0, 1.0)`).
 */
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

/**
 * Auto-layout text label widget placed at the current layout cursor.
 * Automatically wraps to container width (VStack / Card / Column) and respects window boundaries.
 * @param text String content to render.
 * Usage: Primary concise label API (`gui_lbl("Hello, world!")`).
 */
static inline void vira_gui_lbl(vira_string text) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) {
        int32_t win_w = vira_gui_window_width();
        int32_t max_w = (win_w > 60) ? (win_w - 40) : 400;
        _vira_gui_draw_text_wrapped(text, 20, 20, max_w, 16, 0xCDD6F4);
        return;
    }
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t win_w = vira_gui_window_width();

    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        int32_t max_w = (win_w > x + 30) ? (win_w - x - 20) : 400;
        if (_vira_measure_text_size_w(text, 16) <= max_w && !_vira_str_has_newline(text)) {
            int32_t tw = _vira_measure_text_size_w(text, 16);
            vira_gui_draw_text(text, x, y, 16, 0xCDD6F4);
            node->cur_x += tw + node->spacing;
        } else {
            int32_t drawn_h = _vira_gui_draw_text_wrapped(text, x, y, max_w, 16, 0xCDD6F4);
            node->cur_x += max_w + node->spacing;
            if (drawn_h > node->height) node->height = drawn_h;
        }
    } else {
        int32_t max_w = node->width > 0 ? node->width : 340;
        if (win_w > 0 && x + max_w > win_w - 15) {
            max_w = win_w - x - 15;
        }
        if (max_w < 50) max_w = 50;

        int32_t drawn_h = _vira_gui_draw_text_wrapped(text, x, y, max_w, 16, 0xCDD6F4);
        node->cur_y += drawn_h + node->spacing;
    }
}

/**
 * Auto-layout text label widget with an explicit maximum wrapping width.
 * @param text String content to render.
 * @param max_width Maximum allowed width before wrapping to subsequent lines.
 * Usage: Primary API for bounded auto-wrapping labels (`gui_lbl_wrap(text, 400)`).
 */
static inline void vira_gui_lbl_wrap(vira_string text, int32_t max_width) {
    vira_layout_node_t* node = _vira_gui_cur_layout();
    if (!node) {
        int32_t mw = max_width > 0 ? max_width : 400;
        _vira_gui_draw_text_wrapped(text, 20, 20, mw, 16, 0xCDD6F4);
        return;
    }
    int32_t x = node->cur_x;
    int32_t y = node->cur_y;
    int32_t win_w = vira_gui_window_width();
    int32_t max_w = max_width > 0 ? max_width : (node->width > 0 ? node->width : 340);
    if (win_w > 0 && x + max_w > win_w - 15) {
        max_w = win_w - x - 15;
    }
    if (max_w < 50) max_w = 50;

    int32_t drawn_h = _vira_gui_draw_text_wrapped(text, x, y, max_w, 16, 0xCDD6F4);
    if (node->type == VIRA_LAYOUT_HORIZONTAL) {
        node->cur_x += max_w + node->spacing;
        if (drawn_h > node->height) node->height = drawn_h;
    } else {
        node->cur_y += drawn_h + node->spacing;
    }
}

/**
 * Auto-layout image display widget placed at the current layout cursor.
 * @param id Valid image handle ID returned from `vira_gui_load_image`.
 * @param w Desired display width (0 to use native width).
 * @param h Desired display height (0 to use native height).
 * Usage: Primary concise image display API (`gui::image(img_id, 200, 150)`).
 */
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

/**
 * Auto-layout single-line text input field placed at the current layout cursor.
 * @param placeholder Hint text shown when input is empty.
 * @param text Current string value.
 * @return Updated string value.
 * Usage: Primary concise text input API (`text = gui::input("Prompt...", text)`).
 */
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

/**
 * Auto-layout progress bar widget placed at the current layout cursor.
 * @param percent Fraction between 0.0f and 1.0f.
 * Usage: Primary concise progress bar API (`gui::progress(pct)`).
 */
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

/**
 * Auto-layout radio button widget placed at the current layout cursor.
 * @param label Radio option text label.
 * @param is_selected True if currently selected.
 * @return True if clicked and selected this frame.
 * Usage: Primary concise radio button API (`gui::radio("Option A", sel == 0)`).
 */
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

/**
 * Auto-layout subtle horizontal divider line spanning the width of the parent container.
 * Usage: Visual grouping and division between sections in a layout (`gui::separator()`).
 */
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

/**
 * Auto-layout pill badge widget placed at the current layout cursor.
 * @param text Badge label text.
 * @param hex_bg Background hex color.
 * @param hex_fg Foreground text hex color.
 * Usage: Primary concise badge API (`gui::badge("Ready", 0x223322, 0x44EE44)`).
 */
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

/**
 * Auto-layout tab item header button for tabbed navigation bars.
 * @param label Tab display name.
 * @param is_active True if this tab is currently the active selected tab.
 * @return True if user clicked to select this tab.
 * Usage: Used in horizontal stacks to build tab switchers (`gui::tab_item("Chat", active_tab == 0)`).
 */
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

/**
 * Begins an overlay modal dialog centered on the screen, dimming background content and trapping input focus.
 * @param title Modal window title text.
 * @param width Modal dialog width in pixels.
 * @param height Modal dialog height in pixels.
 * Usage: Opens modal confirmation or settings dialogs; closed with `vira_gui_modal_end()`.
 */
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

/**
 * Ends a modal dialog, restoring layout state and consuming background clicks.
 * Usage: Concludes a modal block opened with `vira_gui_modal_begin()`.
 */
static inline void vira_gui_modal_end(void) {
    vira_gui_stack_end();
    _vira_click_consumed = true;
}

// ------------------------------------------------------------------------------
// 3B. DROPDOWN SELECT & INTERACTIVE DATA TABLE COMPONENTS
// ------------------------------------------------------------------------------

/**
 * Internal helper to render dropdown popup lists on top of all other widgets at the end of the frame.
 * Usage: Invoked by `vira_gui_end()` to ensure dropdown menus render without z-order clipping.
 */
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

/**
 * Renders an interactive dropdown selection box with popup menu at explicit coordinates.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param width Box width in pixels.
 * @param height Box height in pixels.
 * @param items Slice of string items to choose from.
 * @param selected_idx Index of currently selected item (-1 for none).
 * @return Updated selected index after user selection.
 * Usage: Used for dropdown selection in manual layouts.
 */
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

/**
 * Auto-layout dropdown selection box placed at the current layout cursor.
 * @param items Slice of string items to choose from.
 * @param selected_idx Currently selected item index.
 * @return Updated selected item index.
 * Usage: Primary concise dropdown select API (`choice = gui::select(models, choice)`).
 */
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

/**
 * Internal case-insensitive substring search helper for table row filtering.
 * @param haystack String to search within.
 * @param needle Substring to look for.
 * @return True if needle is found inside haystack (ignoring case).
 * Usage: Powers table search filtering.
 */
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

/**
 * Internal helper to test if a string represents a numeric value (e.g. integer, float, currency).
 * @param s String to inspect.
 * @param out_val Pointer to receive parsed double value if numeric.
 * @return True if string is numeric, false otherwise.
 * Usage: Enables automatic numeric sorting of table columns.
 */
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

/**
 * Internal cell comparison function that performs natural numeric or case-insensitive alphabetical comparison.
 * @param a Left string cell value.
 * @param b Right string cell value.
 * @return Negative if a < b, positive if a > b, 0 if equal.
 * Usage: Used during table column sorting.
 */
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


/**
 * Renders a full-featured data table with searchable filter, sortable columns, pagination, and row selection at explicit coordinates.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param width Table width in pixels.
 * @param height Table height in pixels.
 * @param headers Slice of column header strings.
 * @param rows 2D slice of row and cell strings.
 * @param search_query Filter string to filter visible rows.
 * @param selected_row Index of currently selected row (-1 for none).
 * @return Index of selected row after user interaction.
 * Usage: Used for manual placement of interactive data tables.
 */
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

/**
 * Auto-layout interactive data table placed at the current layout cursor.
 * @param headers Column header strings.
 * @param rows 2D slice of cell data strings.
 * @param search_query Real-time search query for row filtering.
 * @param selected_row Currently selected row index.
 * @return Updated selected row index.
 * Usage: Primary concise table API (`row = gui::tbl(headers, rows, query, row)`).
 */
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

/**
 * Retrieves the zero-based column index currently used for sorting the most recent table.
 * @return Column index.
 * Usage: Query active table sort column state.
 */
static inline int32_t vira_gui_table_sort_col(void) {
    return _vira_last_table_sort_col;
}

/**
 * Retrieves the sort direction of the most recent table.
 * @return True for ascending sort, false for descending sort.
 * Usage: Query active table sort direction.
 */
static inline bool vira_gui_table_sort_asc(void) {
    return _vira_last_table_sort_asc;
}

/**
 * Programmatically overrides the active sort column and direction for data tables.
 * @param col Column index to sort by.
 * @param asc True for ascending order, false for descending.
 * Usage: Controls table sorting from external buttons or commands.
 */
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

/**
 * Temporary string variant of `vira_gui_lbl` that automatically frees dynamic string allocations.
 * @param text Allocated string to display and free.
 * Usage: Used by compiler code generation for string expressions with guaranteed zero leaks.
 */
static inline void vira_gui_lbl_temp(vira_string text) {
    vira_gui_lbl(text);
    vira_str_free(&text);
}

/**
 * Temporary string variant of `vira_gui_lbl_wrap` that automatically frees dynamic string allocations.
 * @param text Allocated string to display and free.
 * @param max_width Maximum allowed width before wrapping.
 * Usage: Used by compiler code generation for string expressions with guaranteed zero leaks.
 */
static inline void vira_gui_lbl_wrap_temp(vira_string text, int32_t max_width) {
    vira_gui_lbl_wrap(text, max_width);
    vira_str_free(&text);
}

/**
 * Temporary string variant of `vira_gui_draw_text` that automatically frees dynamic string allocations.
 * @param text Allocated string to display and free.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param size Font size.
 * @param hex_color Text color.
 * Usage: Used by compiler code generation for temporary formatted strings.
 */
static inline void vira_gui_draw_text_temp(vira_string text, int32_t x, int32_t y, int32_t size, int32_t hex_color) {
    vira_gui_draw_text(text, x, y, size, hex_color);
    vira_str_free(&text);
}

/**
 * Temporary string variant of `vira_gui_draw_text_wrapped` that automatically frees dynamic string allocations.
 * @param text Allocated string to display and free.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param max_width Maximum bounding width before wrapping.
 * @param size Font size.
 * @param hex_color Text color.
 * Usage: Used by compiler code generation for dynamically formatted wrapped strings.
 */
static inline void vira_gui_draw_text_wrapped_temp(vira_string text, int32_t x, int32_t y, int32_t max_width, int32_t size, int32_t hex_color) {
    vira_gui_draw_text_wrapped(text, x, y, max_width, size, hex_color);
    vira_str_free(&text);
}

/**
 * Temporary string variant of `vira_gui_card_begin` that automatically frees dynamic string allocations.
 * @param title Allocated string card title to display and free.
 * @param width Card width in pixels.
 * Usage: Used by compiler code generation for dynamically formatted card titles.
 */
static inline void vira_gui_card_begin_temp(vira_string title, int32_t width) {
    vira_gui_card_begin(title, width);
    vira_str_free(&title);
}

/**
 * Temporary string variant of `vira_gui_badge` that automatically frees dynamic string allocations.
 * @param text Allocated string badge text to display and free.
 * @param hex_bg Background hex color.
 * @param hex_fg Foreground hex color.
 * Usage: Used by compiler code generation for dynamic badge text.
 */
static inline void vira_gui_badge_temp(vira_string text, int32_t hex_bg, int32_t hex_fg) {
    vira_gui_badge(text, hex_bg, hex_fg);
    vira_str_free(&text);
}

/**
 * Temporary string variant of `vira_gui_btn` that automatically frees dynamic string allocations.
 * @param label Allocated string button label to display and free.
 * @return True if clicked.
 * Usage: Used by compiler code generation for dynamically formatted button labels.
 */
static inline bool vira_gui_btn_temp(vira_string label) {
    bool res = vira_gui_btn(label);
    vira_str_free(&label);
    return res;
}

/**
 * Temporary string variant of `vira_gui_btn_danger` that automatically frees dynamic string allocations.
 * @param label Allocated string label to display and free.
 * @return True if clicked.
 * Usage: Used by compiler code generation for dynamic danger button labels.
 */
static inline bool vira_gui_btn_danger_temp(vira_string label) {
    bool res = vira_gui_btn_danger(label);
    vira_str_free(&label);
    return res;
}

/**
 * Temporary string variant of `vira_gui_tab_item` that automatically frees dynamic string allocations.
 * @param label Allocated string tab name to display and free.
 * @param is_active Whether active.
 * @return True if clicked.
 * Usage: Used by compiler code generation for dynamically named tabs.
 */
static inline bool vira_gui_tab_item_temp(vira_string label, bool is_active) {
    bool res = vira_gui_tab_item(label, is_active);
    vira_str_free(&label);
    return res;
}

/**
 * Temporary string variant of `vira_gui_modal_begin` that automatically frees dynamic string allocations.
 * @param title Allocated string modal title to display and free.
 * @param width Modal width.
 * @param height Modal height.
 * Usage: Used by compiler code generation for dynamically formatted modal titles.
 */
static inline void vira_gui_modal_begin_temp(vira_string title, int32_t width, int32_t height) {
    vira_gui_modal_begin(title, width, height);
    vira_str_free(&title);
}

/**
 * Temporary string variant of `vira_gui_table` that automatically frees dynamic search query strings.
 * @param x Screen X coordinate.
 * @param y Screen Y coordinate.
 * @param width Table width.
 * @param height Table height.
 * @param headers Column header strings.
 * @param rows 2D row/cell strings.
 * @param search_query Filter string to free after rendering.
 * @param selected_row Selected row index.
 * @return Selected row index.
 * Usage: Used by compiler code generation for dynamic search queries.
 */
static inline int32_t vira_gui_table_temp(int32_t x, int32_t y, int32_t width, int32_t height,
                                         vira_slice_string headers, vira_slice_vira_slice_string rows,
                                         vira_string search_query, int32_t selected_row) {
    int32_t res = vira_gui_table(x, y, width, height, headers, rows, search_query, selected_row);
    vira_str_free(&search_query);
    return res;
}

/**
 * Temporary string variant of `vira_gui_tbl` that automatically frees dynamic search query strings.
 * @param headers Column header strings.
 * @param rows 2D row/cell strings.
 * @param search_query Filter string to free after rendering.
 * @param selected_row Selected row index.
 * @return Selected row index.
 * Usage: Used by compiler code generation for auto-layout tables with dynamic queries.
 */
static inline int32_t vira_gui_tbl_temp(vira_slice_string headers, vira_slice_vira_slice_string rows,
                                       vira_string search_query, int32_t selected_row) {
    int32_t res = vira_gui_tbl(headers, rows, search_query, selected_row);
    vira_str_free(&search_query);
    return res;
}

/**
 * Temporary string variant of `vira_gui_load_image` that automatically frees dynamic file path strings.
 * @param path File path string to load and free.
 * @return Image handle ID, or -1 on error.
 * Usage: Used by compiler code generation when image paths are computed dynamically.
 */
static inline int32_t vira_gui_load_image_temp(vira_string path) {
    int32_t res = vira_gui_load_image(path);
    vira_str_free(&path);
    return res;
}


#endif // VIRA_GUI_WIDGETS_H
