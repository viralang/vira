package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/viralang/vira/internal/mod"
)

func initGuiCmd(args []string) {
	name := "my_gui_app"
	if len(args) > 0 {
		for _, a := range args {
			if a != "--gui" && a != "-gui" {
				name = a
				break
			}
		}
	}

	targetDir := name
	if name == "." {
		targetDir = "."
		curr, _ := os.Getwd()
		name = filepath.Base(curr)
	}

	modName := filepath.Base(targetDir)
	if modName == "." || modName == "/" || modName == "\\" {
		curr, _ := os.Getwd()
		modName = filepath.Base(curr)
	}
	modName = strings.ReplaceAll(modName, "-", "_")
	modName = strings.ReplaceAll(modName, " ", "_")
	if len(modName) > 0 && modName[0] >= '0' && modName[0] <= '9' {
		modName = "_" + modName
	}

	appTitle := toTitleCase(name)

	// 1. Create directory structure
	_ = os.MkdirAll(filepath.Join(targetDir, "assets"), 0755)
	_ = os.MkdirAll(filepath.Join(targetDir, "src", "config"), 0755)
	_ = os.MkdirAll(filepath.Join(targetDir, "tests"), 0755)

	// 2. Generate default modern icon in assets/icon.png
	iconPath := filepath.Join(targetDir, "assets", "icon.png")
	if err := generateDefaultAppIcon(iconPath); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not generate default icon: %v\n", err)
	}

	// 3. Generate vira.mod
	m := &mod.Manifest{
		Module:      modName,
		Version:     "0.1.0",
		License:     "MIT",
		Description: fmt.Sprintf("Modern Native GUI Application for %s", appTitle),
		AppName:     appTitle,
		AppID:       fmt.Sprintf("com.viralang.%s", strings.ToLower(modName)),
		Icon:        "assets/icon.png",
		Assets:      "assets",
	}
	_ = os.WriteFile(filepath.Join(targetDir, "vira.mod"), []byte(m.Format()), 0644)

	// 4. Generate src/config/theme.vir
	themeVir := fmt.Sprintf(`module %s

import widget as w

pub struct AppTheme {
pub:
	app_title string
	theme w.Theme
}

pub fn AppTheme.new() AppTheme {
	return AppTheme{
		app_title: '%s',
		theme: w.Theme.mocha(),
	}
}
`, modName, appTitle)
	_ = os.WriteFile(filepath.Join(targetDir, "src", "config", "theme.vir"), []byte(themeVir), 0644)

	// 5. Generate src/main.vir (Rich, interactive native GUI dashboard using Flutter-style widget package)
	mainVir := fmt.Sprintf(`module %s

import widget as w

fn main() {
	mut app := w.App.new('%s', 860, 620)
	app.init()
	defer app.cleanup()

	theme := w.Theme.mocha()
	mut counter := 0
	mut progress := 35
	mut status_msg := 'System Online'

	for !app.should_close() {
		app.begin_frame(theme.bg_color)
		w.Scaffold.begin(theme)

		w.Padding.begin(w.EdgeInsets.all(24), w.window_width(), 16)
		w.Card.begin('%s', w.content_width(24))

		w.StatusBadge('ARCHITECTURE: FLUTTER-STYLE NATIVE IMGUI', true)
		w.Spacer(6)
		w.Heading('Welcome to %s')
		w.Text('Status: ${status_msg}')
		w.Text('Metric Clicks: ${counter}')
		w.Divider()

		w.Row.begin(38, 12)
		if w.ElevatedButton('Increment (+1)') {
			counter = counter + 1
			if progress < 100 {
				progress = progress + 5
			}
			status_msg = 'Counter incremented'
		}
		if w.SecondaryButton('Quick Action (+5)') {
			counter = counter + 5
			if progress < 100 {
				progress = progress + 15
			}
			status_msg = 'Added +5 to counter'
		}
		if w.DangerButton('Reset') {
			counter = 0
			progress = 0
			status_msg = 'Counter reset to zero'
		}
		w.Row.end()

		w.Spacer(10)
		w.ProgressBar(progress, 100)

		w.Card.end()
		w.Padding.end()

		w.Scaffold.end()
		app.end_frame()
	}
}
`, modName, appTitle, appTitle, appTitle)
	_ = os.WriteFile(filepath.Join(targetDir, "src", "main.vir"), []byte(mainVir), 0644)

	// 6. Generate tests/app_test.vir
	testVir := fmt.Sprintf(`module %s_test

import widget as w

fn test_theme_defaults() {
	theme := w.Theme.mocha()
	if theme.bg_color != 0x1E1E2E {
		panic("expected mocha bg_color == 0x1E1E2E")
	}
	if theme.accent_color != 0x89B4FA {
		panic("expected mocha accent_color == 0x89B4FA")
	}
}
`, modName)
	_ = os.WriteFile(filepath.Join(targetDir, "tests", "app_test.vir"), []byte(testVir), 0644)

	// 7. Generate README.md
	readmeContent := fmt.Sprintf(`# %s

A cross-platform native GUI application built in [Vira](https://github.com/viralang/vira).

## Development

Run the application:
`+"```bash"+`
vira run src/main.vir
`+"```"+`

Run tests:
`+"```bash"+`
vira test
`+"```"+`

## Distribution & Packaging

Bundle this application into a native desktop package (.app on macOS, .desktop on Linux, .exe on Windows):
`+"```bash"+`
vira bundle
`+"```"+`
Output packages are generated in the `+"`dist/`"+` directory.
`, appTitle)
	_ = os.WriteFile(filepath.Join(targetDir, "README.md"), []byte(readmeContent), 0644)

	// Print feedback
	fmt.Printf("✓ Initialized dedicated native GUI application %q in ./%s\n\n", appTitle, targetDir)
	fmt.Println("Project Structure:")
	fmt.Printf("  ├── %s/vira.mod            (App manifest, bundle ID & icon settings)\n", targetDir)
	fmt.Printf("  ├── %s/assets/icon.png     (Application icon)\n", targetDir)
	fmt.Printf("  ├── %s/src/main.vir        (Main GUI application loop & widgets)\n", targetDir)
	fmt.Printf("  ├── %s/src/config/theme.vir (App theme, dimensions & colors)\n", targetDir)
	fmt.Printf("  └── %s/tests/app_test.vir  (Unit tests)\n\n", targetDir)
	fmt.Println("Next steps:")
	if targetDir != "." {
		fmt.Printf("  cd %s\n", targetDir)
	}
	fmt.Println("  vira run src/main.vir     # Run live desktop app")
	fmt.Println("  vira bundle               # Package into .app / .desktop / .exe")
}

func toTitleCase(s string) string {
	s = strings.ReplaceAll(s, "_", " ")
	s = strings.ReplaceAll(s, "-", " ")
	words := strings.Fields(s)
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + strings.ToLower(w[1:])
		}
	}
	res := strings.Join(words, "")
	if res == "" {
		return "ViraApp"
	}
	return res
}

// generateDefaultAppIcon creates a stylish 128x128 rounded gradient icon with a geometric V logo
func generateDefaultAppIcon(destPath string) error {
	const size = 128
	img := image.NewRGBA(image.Rect(0, 0, size, size))

	cx := float64(size) / 2.0
	cy := float64(size) / 2.0
	radius := float64(size)*0.46 - 2.0

	// Draw rounded squircle / circular gradient background
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dx := float64(x) - cx
			dy := float64(y) - cy
			dist := math.Sqrt(dx*dx + dy*dy)

			if dist <= radius {
				// Gradient from Sapphire (#74C7EC) to Lavender (#B4BEFE) to Dark (#1E1E2E)
				ratio := (float64(x) + float64(y)) / (float64(size) * 2.0)
				r := uint8(116 + ratio*(180-116))
				g := uint8(199 + ratio*(190-199))
				b := uint8(236 + ratio*(254-236))
				img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
			} else if dist <= radius+1.5 {
				// Anti-aliased border
				alpha := uint8((radius + 1.5 - dist) / 1.5 * 255.0)
				img.SetRGBA(x, y, color.RGBA{R: 116, G: 199, B: 236, A: alpha})
			} else {
				img.SetRGBA(x, y, color.RGBA{0, 0, 0, 0})
			}
		}
	}

	// Draw central V geometric glyph
	vColor := color.RGBA{R: 24, G: 24, B: 37, A: 255} // #181825 dark Catppuccin
	stroke := 10.0

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			fx := float64(x)
			fy := float64(y)

			// Left arm of V: from (36, 38) to (64, 94)
			// Right arm of V: from (92, 38) to (64, 94)
			dLeft := distToSegment(fx, fy, 38, 42, 64, 92)
			dRight := distToSegment(fx, fy, 90, 42, 64, 92)

			if dLeft <= stroke/2.0 || dRight <= stroke/2.0 {
				img.SetRGBA(x, y, vColor)
			}
		}
	}

	f, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer f.Close()

	return png.Encode(f, img)
}

func distToSegment(px, py, x1, y1, x2, y2 float64) float64 {
	dx := x2 - x1
	dy := y2 - y1
	if dx == 0 && dy == 0 {
		return math.Hypot(px-x1, py-y1)
	}
	t := ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)
	if t < 0.0 {
		t = 0.0
	} else if t > 1.0 {
		t = 1.0
	}
	projX := x1 + t*dx
	projY := y1 + t*dy
	return math.Hypot(px-projX, py-projY)
}
