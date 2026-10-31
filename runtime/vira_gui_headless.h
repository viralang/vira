#ifndef VIRA_GUI_HEADLESS_H
#define VIRA_GUI_HEADLESS_H

// ------------------------------------------------------------------------------
// HEADLESS STUB BACKEND (For CI, Tests, and Headless Servers)
// ------------------------------------------------------------------------------

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

#endif // VIRA_GUI_HEADLESS_H
