# Chapter 20: Native Desktop GUI & Flutter-Style Declarative Widgets

Vira brings the same zero-GC, C99-compiled determinism of its backend runtime to native desktop user interfaces. Instead of embedding a 150 MB Chromium/Electron shell, Vira renders double-buffered **60 FPS immediate-mode** frames directly against platform windowing APIs (`X11` + OpenGL on Linux, `Win32` + GDI/OpenGL on Windows, `Cocoa` on macOS, or `Raylib` when enabled), layered with an auto-layout stack engine and a Flutter-inspired declarative `import widget` package.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch20_declarative_gui.vir`](../code/ch20_declarative_gui.vir)

---

## 20.1. Immediate-Mode 60 FPS Native Windowing (X11, Win32, Cocoa, Raylib)

At the lowest level, Vira's built-in GUI runtime exposes a clean immediate-mode frame loop:

```v
gui_init(900, 720, 'Vira Window')
gui_set_target_fps(60)

for !gui_should_close() {
	gui_begin()
	gui_clear(0x181825)

	gui_draw_rect(0, 0, 900, 64, 0x1E1E2E)
	gui_draw_text('Hello Native GUI', 24, 18, 20, 0xCDD6F4)
	gui_draw_line(0, 64, 900, 64, 0x89B4FA)

	gui_end()
}

gui_close()
```

### Core Frame Lifecycle Builtins

| Builtin | Role |
| :--- | :--- |
| **`gui_init(w, h, title)`** | Opens the native window (X11/OpenGL on Linux, Win32 on Windows, Cocoa on macOS) and initializes the double-buffered framebuffer. |
| **`gui_set_target_fps(fps)`** | Configures hardware-timed frame pacing (default `60`). |
| **`gui_should_close()`** | Polls OS window close/quit events; returns `true` when the user closes the window. |
| **`gui_begin()`** / **`gui_end()`** | Starts a new frame, processes mouse/keyboard/scroll input, and swaps buffers cleanly. |
| **`gui_clear(rgb_hex)`** | Clears the background using a 24-bit RGB hex color (e.g. `0x1E1E2E`). |
| **`gui_frame_time()`** | Returns elapsed frame delta time in seconds (`f32`) for animations and telemetry. |

Because UI state lives in normal Vira variables (`mut count := 0`, `mut slider_val := 45.0f32`), there is no retained widget tree to synchronize or garbage-collect between frames.

---

## 20.2. Auto-Layout Stacks (`VStack`, `HStack`, `Card`, `ScrollView`) and Controls

Manually computing `(x, y)` coordinates for every button and label quickly becomes tedious. Vira's 16-level immediate-mode **auto-layout cursor stack** tracks vertical and horizontal placement automatically:

- **`gui_vstack_begin(x, y, width, spacing)` ... `gui_stack_end()`:** Stacks child controls vertically from `(x, y)` with `spacing` pixels between items.
- **`gui_hstack_begin(height, spacing)` ... `gui_stack_end()`:** Divides the parent container width into equal columns for buttons, badges, or form rows.
- **`gui_card_begin(title, width)` ... `gui_card_end()`:** Draws a titled elevation card whose height grows dynamically to fit its children.
- **`gui_scroll_begin(width, height)` ... `gui_scroll_end()`:** Clips overflowing content into a mouse-wheel scrollable viewport with an interactive scrollbar.

### Interactive Controls Without Pixel Math

Inside any stack or card, controls render at the current layout cursor and immediately return updated user state:

```v
gui_vstack_begin(20, 20, 420, 10)
gui_card_begin('Control Panel', 420)
	gui_hstack_begin(36, 12)
		if gui_btn('Increment (+1)') {
			count += 1
		}
		if gui_btn('Reset (0)') {
			count = 0
		}
	gui_stack_end()

	is_active = gui_check('Enable Telemetry', is_active)
	slider_val = gui_slide('Scale: ', 'px', slider_val, 10.0f32, 100.0f32)
	node_name = gui_input('Node name...', node_name)
	gui_progress(slider_val / 100.0f32)
gui_card_end()
gui_stack_end()
```

---

## 20.3. Declarative UI with `import widget` (`w.App`, `w.Scaffold`, `w.DataTable`, Themes, Async Images)

The standard library `widget` package (`packages/widget/src/`) wraps Vira's immediate-mode primitives in ergonomic, Flutter-style composable APIs:

```v
import widget as w
```

### Key `widget` Types & Components

1. **Application Shell & Responsive Breakpoints (`w.App`, `w.Scaffold`, `w.MediaQuery`):**
   - `mut app := w.App.new('Vira Studio Hub', 960, 640)` configures the window (`target_fps: 60`), `app.init()` opens it, and `defer app.cleanup()` guarantees deterministic teardown.
   - `w.MediaQuery.of()` returns a `w.Screen` with `width`, `height`, `is_mobile` (`< 640px`), `is_tablet` (`640..1024px`), `is_desktop` (`> 1024px`), and `is_landscape`.
2. **Design Tokens & Themes (`w.Theme`, `w.EdgeInsets`, `w.TextStyle`):**
   - Built-in palettes: `w.Theme.mocha()`, `w.Theme.latte()`, and `w.Theme.midnight()`.
   - Padding helpers: `w.EdgeInsets.all(16)`, `w.EdgeInsets.symmetric(20, 14)`, `w.EdgeInsets.only(t, r, b, l)`, and `w.EdgeInsets.zero()`.
3. **Layout & Controls (`w.Padding`, `w.Row`, `w.Column`, `w.Card`, `w.SingleChildScrollView`, `w.Modal`):**
   - Composable containers (`w.Column.begin_at`, `w.Row.begin`, `w.Card.begin`, `w.SingleChildScrollView.begin`, `w.Modal.begin`) paired with `w.ElevatedButton`, `w.SecondaryButton`, `w.DangerButton`, `w.TextField`, `w.Checkbox`, `w.Radio`, `w.Slider`, `w.Dropdown`, `w.Tab`, `w.StatusBadge`, and `w.ProgressBar`.
4. **Sortable Data Tables (`w.DataTable`):**
   - `selected_row = w.DataTable(headers, rows, search_filter, selected_row)` renders a searchable, sortable, clickable multi-column table in a single call.
5. **Async Images & Textures (`w.LoadImage`, `w.Image`, `w.FreeImage`):**
   - Load PNG/JPEG assets onto GPU textures with `tex := w.LoadImage(path)`, draw them with `w.Image(tex, width, height)`, and release them cleanly with `w.FreeImage(tex)`. For remote images (as shown in [`docs/examples/39_gui_realworld_cloud_ai_hub.vir`](../../../examples/39_gui_realworld_cloud_ai_hub.vir)), fetch assets asynchronously or in a non-blocking state machine into a local cache path and reload the GPU texture once the download completes so the 60 FPS render loop never stalls.

---

## 20.4. Cross-Platform App Scaffolding & Native Packaging (`vira init-gui`, `vira bundle`)

Vira ships two dedicated CLI commands (`cmd/vira/init_gui.go` and `cmd/vira/bundle.go`) to scaffold and package standalone desktop applications:

### 1. Scaffolding a GUI Project (`vira init-gui`)

```bash
./vira init-gui my_studio_app
```

This generates a complete, ready-to-run project layout:
- `my_studio_app/vira.mod` — App manifest with `app_name`, `app_id`, `icon`, and `assets` directory configuration.
- `my_studio_app/assets/icon.png` — Generated default PNG application icon.
- `my_studio_app/src/config/theme.vir` — Theme module using `import widget as w`.
- `my_studio_app/src/main.vir` — Responsive interactive dashboard starter.
- `my_studio_app/tests/` — Directory for widget and unit tests.

### 2. Packaging Native Bundles (`vira bundle`)

When you are ready to ship to end users, `vira bundle` compiles your application in production mode and packages platform-native installers and directories:

```bash
./vira bundle -name "MyStudio" -id "com.example.mystudio" -icon assets/icon.png src/main.vir
```

Depending on the target OS, `vira bundle` produces:
- **Linux:** An AppDir-style directory with an executable binary, bundled `assets/`, icon, and `.desktop` launcher entry.
- **macOS:** A standard `.app` bundle (`Contents/MacOS/`, `Contents/Resources/`, `Info.plist`, and `.icns`/icon assets).
- **Windows:** A GUI-subsystem `.exe` (no console flash) alongside bundled resources.

---

## 20.5. Running the Complete Chapter 20 Example

By default, [`docs/handbook/chapters/code/ch20_declarative_gui.vir`](../code/ch20_declarative_gui.vir) runs deterministic headless widget, theme, and layout checks so it executes cleanly in terminals and CI pipelines without blocking on a display server. Pass `--window` when running on a desktop session to open the live 60 FPS interactive window:

```bash
# Headless / CI verification mode (default)
./vira run docs/handbook/chapters/code/ch20_declarative_gui.vir

# Interactive 60 FPS native window mode (requires X11 / Win32 / macOS display)
./vira run docs/handbook/chapters/code/ch20_declarative_gui.vir -- --window
```

Output (headless verification mode):
```text
[1] Themes & Tokens: mocha=0x1E1E2E, insets=(20,14), heading=22px, fps=60
[2] Auto-Layout Stack: VStack/HStack/Card/ProgressBar frame verified
[3] Declarative Widgets: table_cols=3, table_rows=3, wheel_delta=0.00
[4] Packaging: scaffold with `vira init-gui` and bundle with `vira bundle` (pass `--window` for live UI)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=9
```

---

## Summary & Exercises

1. **Responsive Breakpoint Inspection:** Add a helper check in `run_headless_checks()` inside `ch20_declarative_gui.vir` that inspects `w.MediaQuery.of()` and verifies `is_mobile || is_tablet || is_desktop` evaluates cleanly with `active_leaks=0`.
2. **Custom Design Theme:** Define a `pub fn cyber_theme() w.Theme` helper in `ch20_declarative_gui.vir` with custom dark slate background (`0x0B1020`) and cyan accent (`0x22D3EE`), and pass it into `w.Scaffold.begin(...)` inside `run_live_window()`.
3. **Interactive GUI Scaffolding:** Run `./vira init-gui /tmp/demo_gui_app && ./vira check /tmp/demo_gui_app/src/main.vir` to inspect the generated project structure and Autofree report, then remove `/tmp/demo_gui_app`.
