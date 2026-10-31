# `widget` — Flutter-Inspired Declarative UI Kit for Vira

The `widget` package provides high-level, Flutter-inspired ergonomics, responsive layout containers, Catppuccin design tokens, and a complete UI component library on top of Vira's hardware-accelerated Immediate-Mode GUI (IMGUI) engine.

---

## Quick Start Example

```v
import widget as w

fn main() {
    mut app := w.App.new('Studio Designer', 800, 600)
    app.init()
    defer app.cleanup()

    mut theme := w.Theme.mocha()
    mut counter := 0
    mut progress := 35

    for !app.should_close() {
        app.begin_frame(theme.bg_color)
        w.Scaffold.begin(theme)

        // 1. Responsive Margin Container
        w.Padding.begin(w.EdgeInsets.all(24), w.window_width(), 14)

        // 2. Framed Elevation Card
        w.Card.begin('Flutter-Style Control Panel', w.content_width(24))

        w.Heading('System Telemetry')
        w.Text('Build responsive desktop software with zero external dependencies.')
        w.Spacer(8)

        w.StatusBadge('Service Online', true)
        w.Divider()

        // 3. Horizontal Action Row
        w.Row.begin(38, 12)
        if w.ElevatedButton('Increment Counter') {
            counter += 1
            if progress < 100 {
                progress += 5
            }
        }
        if w.DangerButton('Reset') {
            counter = 0
            progress = 0
        }
        w.Row.end()

        w.Spacer(10)
        w.Text('Clicks: ${counter}')
        w.ProgressBar(progress, 100)

        w.Card.end()
        w.Padding.end()

        w.Scaffold.end()
        app.end_frame()
    }
}
```

---

## Component Catalog

### 1. Layout & Structure
- **`w.Column.begin(width, spacing)` / `w.Column.end()`**: Vertical auto-layout stack.
- **`w.Row.begin(height, spacing)` / `w.Row.end()`**: Horizontal auto-layout row.
- **`w.Card.begin(title, width)` / `w.Card.end()`**: Framed elevation container.
- **`w.Padding.begin(insets, width, spacing)` / `w.Padding.end()`**: Inset padding wrapper.
- **`w.Center.begin(width, spacing)` / `w.Center.end()`**: Horizontally centered column.
- **`w.Spacer(px)`**: Explicit layout gap.
- **`w.Divider()`**: Subtle 1px horizontal line.

### 2. Responsive UI Kit (`MediaQuery`)
- **`w.window_width()` & `w.window_height()`**: Current viewport dimensions in pixels.
- **`w.pct_w(50.0)`**: Calculates 50% of the current window width.
- **`w.pct_h(25.0)`**: Calculates 25% of the current window height.
- **`w.content_width(24)`**: Computes remaining width after subtracting 24px padding on both sides.
- **`w.is_mobile()`**: True when `window_width < 640`.
- **`w.is_tablet()`**: True when `640 <= window_width <= 1024`.
- **`w.is_desktop()`**: True when `window_width > 1024`.
- **`w.screen()` / `w.MediaQuery.of()`**: Returns a `Screen` struct with width, height, breakpoints, and orientation.

### 3. Interactive Controls
- **`w.Button(label)` / `w.ElevatedButton(label)`**: Primary action button.
- **`w.DangerButton(label)`**: Red warning/destructive button.
- **`w.SecondaryButton(label)`**: Subtle secondary button.
- **`w.TextField(placeholder, text)`**: Interactive text input.
- **`w.Checkbox(label, checked)`**: Labeled toggle checkbox.
- **`w.Radio(label, is_selected)`**: Single-choice radio item.
- **`w.Slider(label, val, min, max)`**: Smooth horizontal range slider.
- **`w.Dropdown(items, selected_idx)` / `w.Select(items, selected_idx)`**: Expandable choice selector with top z-index popup.

### 4. Data Display & Typography
- **`w.Text(content)`**: Standard body text.
- **`w.Heading(content)`**: Section header.
- **`w.Subheading(content)`**: Secondary section title.
- **`w.Caption(content)`**: Small metadata label.
- **`w.Badge(label, bg_hex, fg_hex)`**: Custom colored badge.
- **`w.StatusBadge(label, is_ok)`**: Pre-styled semantic badge (Green/Red).
- **`w.DataTable(headers, rows, search, selected_row)`**: Sortable, searchable data grid.
- **`w.ProgressBar(current, max)`**: Percentage-filled progress indicator.
- **`w.Image(image_id, width, height)`**: Hardware-accelerated GPU image.

### 5. Application Lifecycle & Overlays
- **`w.App.new(title, width, height)`**: Application descriptor with target FPS and native event loop.
- **`w.Scaffold.begin(theme)` / `w.Scaffold.end()`**: Themed page background.
- **`w.Tab(label, is_active)`**: Tab strip navigation item.
- **`w.Modal.begin(title, width, height)` / `w.Modal.end()`**: Centered overlay modal with backdrop dimming.
