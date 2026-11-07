#ifndef VIRA_GUI_HEADLESS_H
#define VIRA_GUI_HEADLESS_H

// ------------------------------------------------------------------------------
// HEADLESS STUB BACKEND (For CI, Tests, and Headless Servers)
// ------------------------------------------------------------------------------

/**
 * Initializes the headless GUI backend stub.
 * @param w Desired initial window width in pixels.
 * @param h Desired initial window height in pixels.
 * @param t Window title string.
 * Usage: Called by `gui_init(w, h, title)` when compiling in headless mode (CI or servers without display).
 */
static inline void vira_gui_init(int32_t w, int32_t h, vira_string t) { (void)w; (void)h; (void)t; }

/**
 * Checks whether the GUI window has requested to close.
 * @return Always returns true in headless mode so event loops exit cleanly after a single pass or immediately.
 * Usage: Called in `for !gui_should_close() { ... }` event loops.
 */
static inline bool vira_gui_should_close(void) { return true; }

/**
 * Cancels a pending close request, keeping the window active.
 * Usage: Called by `gui_cancel_close()` when intercepting close confirmation dialogs.
 */
static inline void vira_gui_cancel_close(void) {}

/**
 * Sets the target frame rate for frame pacing.
 * @param fps Target frames per second (e.g., 60).
 * Usage: Called by `gui_set_target_fps(60)`.
 */
static inline void vira_gui_set_target_fps(int32_t fps) { (void)fps; }

/**
 * Retrieves the current logical width of the headless window viewport.
 * @return Fixed width in pixels (800).
 * Usage: Called by `gui_window_width()` to compute responsive layouts.
 */
static inline int32_t vira_gui_window_width(void) { return 800; }

/**
 * Retrieves the current logical height of the headless window viewport.
 * @return Fixed height in pixels (600).
 * Usage: Called by `gui_window_height()` to compute responsive layouts.
 */
static inline int32_t vira_gui_window_height(void) { return 600; }

/**
 * Starts a new headless frame by clearing input events and resetting frame timers.
 * Usage: Invoked internally at the beginning of each `gui_begin()` call.
 */
static inline void _vira_backend_gui_begin(void) {}

/**
 * Concludes the current headless frame, flushes buffers, and paces target frame time.
 * Usage: Invoked internally at the end of each `gui_end()` call.
 */
static inline void _vira_backend_gui_end(void) {}

/**
 * Destroys the headless GUI context and releases mock resources.
 * Usage: Called by `gui_close()` at application exit.
 */
static inline void vira_gui_close(void) {}

/**
 * Clears the virtual viewport buffer with a solid background color.
 * @param c 24-bit hex RGB color (e.g., 0x1E1E2E).
 * Usage: Called by `gui_clear(color)` at the start of rendering.
 */
static inline void vira_gui_clear(int32_t c) { (void)c; }

/**
 * Draws text onto the virtual headless canvas.
 * @param t Text string to render.
 * @param x X coordinate of top-left text origin.
 * @param y Y coordinate of top-left text origin.
 * @param s Font size in points.
 * @param c 24-bit hex RGB text color.
 * Usage: Called by `gui_draw_text(text, x, y, size, color)`.
 */
static inline void vira_gui_draw_text(vira_string t, int32_t x, int32_t y, int32_t s, int32_t c) { (void)t; (void)x; (void)y; (void)s; (void)c; }

/**
 * Draws a solid filled rectangle onto the virtual headless canvas.
 * @param x Top-left X coordinate.
 * @param y Top-left Y coordinate.
 * @param w Rectangle width in pixels.
 * @param h Rectangle height in pixels.
 * @param c 24-bit hex RGB fill color.
 * Usage: Called by `gui_draw_rect(x, y, w, h, color)`.
 */
static inline void vira_gui_draw_rect(int32_t x, int32_t y, int32_t w, int32_t h, int32_t c) { (void)x; (void)y; (void)w; (void)h; (void)c; }

/**
 * Draws a rounded rectangle with background fill and border outline.
 * @param x Top-left X coordinate.
 * @param y Top-left Y coordinate.
 * @param w Width in pixels.
 * @param h Height in pixels.
 * @param r Corner corner radius in pixels.
 * @param bg 24-bit hex RGB background fill color.
 * @param border 24-bit hex RGB border outline color.
 * Usage: Called by `gui_draw_rounded_rect(...)` for cards, buttons, and inputs.
 */
static inline void vira_gui_draw_rounded_rect(int32_t x, int32_t y, int32_t w, int32_t h, float r, int32_t bg, int32_t border) { (void)x; (void)y; (void)w; (void)h; (void)r; (void)bg; (void)border; }

/**
 * Draws an anti-aliased circle on the virtual headless canvas.
 * @param cx Center X coordinate.
 * @param cy Center Y coordinate.
 * @param r Circle radius in pixels.
 * @param c 24-bit hex RGB color.
 * Usage: Called by `gui_draw_circle(cx, cy, r, color)`.
 */
static inline void vira_gui_draw_circle(int32_t cx, int32_t cy, float r, int32_t c) { (void)cx; (void)cy; (void)r; (void)c; }

/**
 * Draws a 1-pixel vector line between two coordinates.
 * @param x1 Start X coordinate.
 * @param y1 Start Y coordinate.
 * @param x2 End X coordinate.
 * @param y2 End Y coordinate.
 * @param c 24-bit hex RGB line color.
 * Usage: Called by `gui_draw_line(x1, y1, x2, y2, color)` for dividers and graphs.
 */
static inline void vira_gui_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t c) { (void)x1; (void)y1; (void)x2; (void)y2; (void)c; }

/**
 * Estimates text advance width in pixels using a fixed-pitch heuristic (8px per char).
 * @param t Text string to measure.
 * @return Approximate width in pixels.
 * Usage: Used by auto-layout and widgets to size buttons and labels in headless environments.
 */
static inline int32_t _vira_measure_text_w(vira_string t) { return t.len * 8; }

/**
 * Queries the current mouse cursor X coordinate in window space.
 * @return Always returns 0 in headless mode.
 * Usage: Called by `gui_mouse_x()`.
 */
static inline int32_t vira_gui_mouse_x(void) { return 0; }

/**
 * Queries the current mouse cursor Y coordinate in window space.
 * @return Always returns 0 in headless mode.
 * Usage: Called by `gui_mouse_y()`.
 */
static inline int32_t vira_gui_mouse_y(void) { return 0; }

/**
 * Checks whether a given mouse button is currently held down.
 * @param b Button index: 0 = Left, 1 = Right, 2 = Middle.
 * @return False in headless mode.
 * Usage: Called by `gui_is_mouse_down(button)`.
 */
static inline bool vira_gui_is_mouse_down(int32_t b) { (void)b; return false; }

/**
 * Checks if the primary (left) mouse button transitioned from up to down on this frame.
 * @return False in headless mode.
 * Usage: Used internally by immediate-mode widgets to trigger click actions.
 */
static inline bool _vira_backend_mouse_clicked_left(void) { return false; }

/**
 * Checks if a specific key was pressed down on this frame.
 * @param k Keycode (ASCII or virtual key).
 * @return False in headless mode.
 * Usage: Called by `gui_is_key_pressed(key)`.
 */
static inline bool vira_gui_is_key_pressed(int32_t k) { (void)k; return false; }

/**
 * Returns elapsed delta time for the last completed frame in seconds.
 * @return Fixed mock frame time (0.016f = ~60 FPS).
 * Usage: Called by `gui_frame_time()` for smooth physics and animations.
 */
static inline float vira_gui_frame_time(void) { return 0.016f; }

/**
 * Mock image loading function for headless environments.
 * @param filepath Path to the image file.
 * @return -1 indicating no native handle created in headless mode.
 * Usage: Called by `gui_load_image(path)`.
 */
static inline int32_t _vira_headless_load_image(const char* filepath) { (void)filepath; return -1; }

/**
 * Mock image drawing stub.
 * @param id Image handle index.
 * @param x Top-left X destination.
 * @param y Top-left Y destination.
 * @param w Destination width.
 * @param h Destination height.
 * Usage: Called by `gui_draw_image(id, x, y, w, h)`.
 */
static inline void _vira_headless_draw_image(int32_t id, int32_t x, int32_t y, int32_t w, int32_t h) { (void)id; (void)x; (void)y; (void)w; (void)h; }

/**
 * Frees a mock image handle.
 * @param id Image handle index.
 * Usage: Called by `gui_free_image(id)`.
 */
static inline void _vira_headless_free_image(int32_t id) { (void)id; }

/**
 * Sets a rectangular scissor clipping region for rendering.
 * @param x Left coordinate of scissor box.
 * @param y Top coordinate of scissor box.
 * @param w Width of scissor box.
 * @param h Height of scissor box.
 * Usage: Called by scroll views and modals to constrain drawing.
 */
static inline void _vira_headless_set_clip(int32_t x, int32_t y, int32_t w, int32_t h) { (void)x; (void)y; (void)w; (void)h; }

/**
 * Resets the scissor clipping region back to the full window viewport.
 * Usage: Called at the end of scroll views and modals to restore unclipped rendering.
 */
static inline void _vira_headless_reset_clip(void) {}

#endif // VIRA_GUI_HEADLESS_H
