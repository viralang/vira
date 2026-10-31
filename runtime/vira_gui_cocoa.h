#ifndef VIRA_GUI_COCOA_H
#define VIRA_GUI_COCOA_H

// ------------------------------------------------------------------------------
// NATIVE MACOS COCOA & COREGRAPHICS BACKEND (Zero-Dependency Native Mac)
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

#endif // VIRA_GUI_COCOA_H
