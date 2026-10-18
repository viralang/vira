package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	goRuntime "runtime"
	"strings"
	"time"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/autofree"
	"github.com/viralang/vira/internal/cgen"
	"github.com/viralang/vira/internal/checker"
	"github.com/viralang/vira/internal/formatter"
	"github.com/viralang/vira/internal/lsp"
	"github.com/viralang/vira/internal/mod"
	"github.com/viralang/vira/internal/parser"
	"github.com/viralang/vira/internal/pkg"
	"github.com/viralang/vira/internal/source"
	virart "github.com/viralang/vira/runtime"
)

const usageBanner = `Vira Programming Language Compiler (Stage-0 Toolchain)

Usage:
  vira <command> [flags] [target]

Commands:
  run     Compile and execute a .vir file or directory immediately
  build   Compile a .vir file or directory into a standalone native binary
  cgen    Emit lowered C99 code (SIMD + Work-Stealing Actors)
  check   Run syntax, 4-quadrant mutability, and autofree escape analysis
  test    Discover and run test functions/files with leak checks
  fmt     Format .vir source files according to official style rules
  lsp     Start the Vira Language Server (LSP) over stdin/stdout
  init    Initialize a new Vira project scaffold with vira.mod and src/
  add     Add and download a package dependency (local or git)
  install Install dependencies declared in vira.mod and vira.lock
  remove  Remove a package dependency
  vendor  Vendor all dependencies into ./vendor for offline/air-gapped builds
  demo    Generate and run the built-in OOP + SIMD Tensor + Actor demo
  version Display compiler version and target platform

Flags for 'vira build':
  -prod       Enable production optimizations (-O3 -march=native -flto)
  -sanitize   Enable AddressSanitizer & UndefinedBehaviorSanitizer
  -o <path>   Output binary path (default: <filename>)
  -cc <bin>   Specify C compiler backend (cc, clang, gcc, tcc)
  -keep-c     Keep the intermediate .c file in build/
`

type compileOptions struct {
	prod         bool
	sanitize     bool
	cc           string
	outPath      string
	keepC        bool
	showAutofree bool
}

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usageBanner)
		os.Exit(1)
	}

	subcmd := os.Args[1]
	switch subcmd {
	case "run":
		runCmd(os.Args[2:])
	case "build":
		buildCmd(os.Args[2:])
	case "cgen":
		cgenCmd(os.Args[2:])
	case "check":
		checkCmd(os.Args[2:])
	case "test":
		testCmd(os.Args[2:])
	case "fmt":
		fmtCmd(os.Args[2:])
	case "lsp":
		lspCmd()
	case "init":
		initCmd(os.Args[2:])
	case "add":
		addCmd(os.Args[2:])
	case "install", "get":
		installCmd(os.Args[2:])
	case "remove":
		removeCmd(os.Args[2:])
	case "vendor":
		vendorCmd(os.Args[2:])
	case "demo":
		demoCmd()
	case "version", "-v", "--version":
		versionCmd()
	case "-h", "--help", "help":
		fmt.Print(usageBanner)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", subcmd, usageBanner)
		os.Exit(1)
	}
}

func versionCmd() {
	fmt.Printf("Vira v0.5.8 (%s/%s)\n", goRuntime.GOOS, goRuntime.GOARCH)
}

// --- SUBCOMMAND 1: `vira run` ---

func runCmd(args []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	ccFlag := fs.String("cc", "", "C compiler to use (auto-detects tcc, cc, gcc, or clang)")
	prodFlag := fs.Bool("prod", false, "Run with -O3 -march=native optimizations")
	_ = fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: vira run [-prod] [-cc compiler] <file.vir | directory>")
		os.Exit(1)
	}
	vexPath := fs.Arg(0)

	cCode, _, _ := compileVexToC99(vexPath, false)

	tmpDir, err := os.MkdirTemp("", "vira_run_*")
	if err != nil {
		fatalIO(err)
	}
	defer os.RemoveAll(tmpDir)

	cFile := filepath.Join(tmpDir, "out.c")
	binExt := ""
	if os.PathSeparator == '\\' {
		binExt = ".exe"
	}
	binFile := filepath.Join(tmpDir, "app"+binExt)
	if err := os.WriteFile(cFile, []byte(cCode), 0644); err != nil {
		fatalIO(err)
	}

	opts := compileOptions{
		prod: *prodFlag,
		cc:   *ccFlag,
	}
	compiler, _ := invokeCBackend(cFile, binFile, opts)

	cmd := exec.Command(binFile, fs.Args()[1:]...)
	compilerDir := filepath.Dir(compiler)
	cmd.Env = append(os.Environ(), "PATH="+compilerDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		os.Exit(1)
	}
}

// --- SUBCOMMAND 2: `vira build` ---

func buildCmd(args []string) {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	prod := fs.Bool("prod", false, "Compile with -O3 -march=native -flto")
	sanitize := fs.Bool("sanitize", false, "Enable ASan and UBSan")
	outPath := fs.String("o", "", "Output binary path")
	ccFlag := fs.String("cc", "", "C compiler (cc, clang, gcc, tcc)")
	keepC := fs.Bool("keep-c", false, "Keep generated C99 file in build/")
	_ = fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: vira build [-prod] [-sanitize] [-o out] <file.vir | directory>")
		os.Exit(1)
	}
	vexPath := fs.Arg(0)
	start := time.Now()

	cCode, _, _ := compileVexToC99(vexPath, false)

	baseName := strings.TrimSuffix(filepath.Base(vexPath), filepath.Ext(vexPath))
	targetBin := *outPath
	if targetBin == "" {
		targetBin = baseName
		if os.PathSeparator == '\\' && !strings.HasSuffix(strings.ToLower(targetBin), ".exe") {
			targetBin += ".exe"
		}
	}

	_ = os.MkdirAll("build", 0755)
	cFile := filepath.Join("build", baseName+".c")
	if err := os.WriteFile(cFile, []byte(cCode), 0644); err != nil {
		fatalIO(err)
	}

	opts := compileOptions{
		prod:     *prod,
		sanitize: *sanitize,
		cc:       *ccFlag,
		outPath:  targetBin,
		keepC:    *keepC,
	}

	compiler, modeDesc := invokeCBackend(cFile, targetBin, opts)
	elapsed := time.Since(start)

	if !*keepC {
		_ = os.Remove(cFile)
	}

	fmt.Printf("✓ Built standalone binary %s in %s (backend: %s [%s])\n",
		targetBin, elapsed.Round(time.Millisecond), compiler, modeDesc)
}

// --- SUBCOMMAND 3: `vira cgen` ---

func cgenCmd(args []string) {
	fs := flag.NewFlagSet("cgen", flag.ExitOnError)
	outPath := fs.String("o", "", "Write generated C99 to file instead of stdout")
	_ = fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: vira cgen [-o out.c] <file.vir | directory>")
		os.Exit(1)
	}
	vexPath := fs.Arg(0)
	cCode, _, _ := compileVexToC99(vexPath, false)

	if *outPath != "" {
		if err := os.WriteFile(*outPath, []byte(cCode), 0644); err != nil {
			fatalIO(err)
		}
		fmt.Printf("✓ Wrote C99 code (%d bytes) -> %s\n", len(cCode), *outPath)
		return
	}
	fmt.Print(cCode)
}

// --- SUBCOMMAND 4: `vira check` ---

func checkCmd(args []string) {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	showAutofree := fs.Bool("autofree", true, "Print static escape analysis and autofree report")
	_ = fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: vira check [-autofree] <file.vir | directory>")
		os.Exit(1)
	}
	vexPath := fs.Arg(0)
	_, chk, af := compileVexToC99(vexPath, false)

	fmt.Printf("✓ %s passed syntax, 4-quadrant mutability, and type checks (0 errors)\n", vexPath)
	if *showAutofree {
		fmt.Println("\n" + af.FormatReport())
	}
	_ = chk
}

// --- SUBCOMMAND 5: `vira test` ---

func testCmd(args []string) {
	fs := flag.NewFlagSet("test", flag.ExitOnError)
	_ = fs.Parse(args)

	target := "."
	if fs.NArg() > 0 {
		target = fs.Arg(0)
	}

	var testFiles []string
	fi, err := os.Stat(target)
	if err != nil {
		fatalIO(err)
	}

	if fi.IsDir() {
		_ = filepath.Walk(target, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() && strings.HasSuffix(path, ".vir") {
				if strings.HasSuffix(path, "_test.vir") {
					testFiles = append(testFiles, path)
				} else if raw, rErr := os.ReadFile(path); rErr == nil && strings.Contains(string(raw), "fn test_") {
					testFiles = append(testFiles, path)
				}
			}
			return nil
		})
	} else {
		testFiles = append(testFiles, target)
	}

	if len(testFiles) == 0 {
		fmt.Println("No tests found (searching for *_test.vir or files containing 'fn test_')")
		return
	}

	passCount := 0
	failCount := 0
	startTotal := time.Now()

	for _, tFile := range testFiles {
		fmt.Printf("=== RUN   %s\n", tFile)
		start := time.Now()
		cCode, _, _ := compileVexToC99(tFile, true)

		tmpDir, err := os.MkdirTemp("", "vira_test_*")
		if err != nil {
			fatalIO(err)
		}
		cFile := filepath.Join(tmpDir, "out.c")
		binExt := ""
		if os.PathSeparator == '\\' {
			binExt = ".exe"
		}
		binFile := filepath.Join(tmpDir, "test_bin"+binExt)
		_ = os.WriteFile(cFile, []byte(cCode), 0644)

		opts := compileOptions{prod: false}
		compiler, _ := invokeCBackend(cFile, binFile, opts)

		cmd := exec.Command(binFile)
		compilerDir := filepath.Dir(compiler)
		cmd.Env = append(os.Environ(), "PATH="+compilerDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		out, runErr := cmd.CombinedOutput()
		dur := time.Since(start)

		if runErr != nil {
			fmt.Printf("--- FAIL: %s (%.2fs)\n%s\n", tFile, dur.Seconds(), string(out))
			failCount++
		} else {
			fmt.Printf("--- PASS: %s (%.2fs)\n", tFile, dur.Seconds())
			if strings.Contains(string(out), "active_leaks=0") {
				fmt.Printf("    active_leaks=0 [Zero Memory Leaks Verified]\n")
			}
			passCount++
		}
		_ = os.RemoveAll(tmpDir)
	}

	fmt.Printf("\nTest Summary: %d passed, %d failed in %.2fs\n", passCount, failCount, time.Since(startTotal).Seconds())
	if failCount > 0 {
		os.Exit(1)
	}
}

// --- SUBCOMMAND 6: `vira fmt` ---

func fmtCmd(args []string) {
	fs := flag.NewFlagSet("fmt", flag.ExitOnError)
	writeBack := fs.Bool("w", false, "Write result back to file instead of stdout")
	_ = fs.Parse(args)

	targets := fs.Args()
	if len(targets) == 0 {
		targets = []string{"."}
	}

	for _, target := range targets {
		fi, err := os.Stat(target)
		if err != nil {
			fmt.Fprintf(os.Stderr, "vira fmt error: %v\n", err)
			continue
		}
		if fi.IsDir() {
			_ = filepath.Walk(target, func(path string, info os.FileInfo, err error) error {
				if err == nil && !info.IsDir() && strings.HasSuffix(path, ".vir") {
					formatSingleFile(path, *writeBack)
				}
				return nil
			})
		} else {
			formatSingleFile(target, *writeBack)
		}
	}
}

func formatSingleFile(path string, writeBack bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "vira fmt read error (%s): %v\n", path, err)
		return
	}
	src := string(raw)
	formatted := formatter.Format(src)
	if writeBack {
		if src != formatted {
			if err := os.WriteFile(path, []byte(formatted), 0644); err != nil {
				fmt.Fprintf(os.Stderr, "vira fmt write error (%s): %v\n", path, err)
				return
			}
			fmt.Printf("reformatted %s\n", path)
		}
	} else {
		fmt.Print(formatted)
	}
}

// --- SUBCOMMAND 7: `vira lsp` ---

func lspCmd() {
	server := lsp.NewServer(os.Stdin, os.Stdout)
	if err := server.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "vira lsp error: %v\n", err)
		os.Exit(1)
	}
}

// --- SUBCOMMAND 8: `vira init` ---

func initCmd(args []string) {
	name := "my_vira_app"
	if len(args) > 0 {
		name = args[0]
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

	// 1. Create modern directory layout
	_ = os.MkdirAll(filepath.Join(targetDir, "src", "config"), 0755)
	_ = os.MkdirAll(filepath.Join(targetDir, "src", "models"), 0755)
	_ = os.MkdirAll(filepath.Join(targetDir, "src", "services"), 0755)
	_ = os.MkdirAll(filepath.Join(targetDir, "tests"), 0755)

	// 2. vira.mod
	modContent := fmt.Sprintf(`module %s
version 0.1.0
license MIT
description Modern scalable Vira application
`, modName)
	_ = os.WriteFile(filepath.Join(targetDir, "vira.mod"), []byte(modContent), 0644)

	// 3. src/config/settings.vir
	configContent := fmt.Sprintf(`module %s

pub struct AppConfig {
pub:
	app_name string
	version string
	debug bool
}

pub fn AppConfig.default() AppConfig {
	return AppConfig{
		app_name: '%s',
		version: '0.1.0',
		debug: true
	}
}
`, modName, name)
	_ = os.WriteFile(filepath.Join(targetDir, "src", "config", "settings.vir"), []byte(configContent), 0644)

	// 4. src/models/data.vir
	modelContent := fmt.Sprintf(`module %s

pub struct WorkItem {
pub:
	id int
	title string
mut:
	completed bool
}

pub fn WorkItem.new(id int, title string) WorkItem {
	return WorkItem{
		id: id,
		title: title,
		completed: false
	}
}

pub fn (mut w WorkItem) mark_done() {
	w.completed = true
}
`, modName)
	_ = os.WriteFile(filepath.Join(targetDir, "src", "models", "data.vir"), []byte(modelContent), 0644)

	// 5. src/services/compute.vir
	serviceContent := fmt.Sprintf(`module %s

pub struct ComputeService {
pub:
	worker_id int
}

pub fn ComputeService.new(worker_id int) ComputeService {
	return ComputeService{ worker_id: worker_id }
}

pub fn (s ComputeService) process(item WorkItem) string {
	return 'Worker #${s.worker_id} processed: ${item.title}'
}
`, modName)
	_ = os.WriteFile(filepath.Join(targetDir, "src", "services", "compute.vir"), []byte(serviceContent), 0644)

	// 6. src/main.vir
	mainContent := fmt.Sprintf(`module %s

fn main() {
	cfg := AppConfig.default()
	println('=== ${cfg.app_name} (v${cfg.version}) ===')

	mut task := WorkItem.new(101, 'Data Processing Batch')
	service := ComputeService.new(1)

	status_msg := service.process(task)
	println(status_msg)

	task.mark_done()
	println('Task #${task.id} [${task.title}] completed: ${task.completed}')
}
`, modName)
	_ = os.WriteFile(filepath.Join(targetDir, "src", "main.vir"), []byte(mainContent), 0644)

	// 7. tests/app_test.vir
	testContent := fmt.Sprintf(`module %s

fn test_basic_pipeline() {
	t := tensor_fill(16, 16, 1.0)
	if tensor_sum(t) != 256.0 {
		panic('tensor sum mismatch')
	}
	println('✓ test_basic_pipeline passed')
}
`, modName)
	_ = os.WriteFile(filepath.Join(targetDir, "tests", "app_test.vir"), []byte(testContent), 0644)

	// 8. README.md
	readmeContent := fmt.Sprintf(`# %s

A modern, scalable application built with the **Vira Programming Language**.

## Project Architecture

`+"```"+`
%s/
├── vira.mod               # Project manifest
├── .gitignore             # Git ignore patterns
├── README.md              # Documentation
├── src/
│   ├── main.vir           # Application entry point
│   ├── config/
│   │   └── settings.vir   # Environment & app configuration
│   ├── models/
│   │   └── data.vir       # 4-quadrant encapsulated domain models
│   └── services/
│       └── compute.vir    # Business logic & services
└── tests/
    └── app_test.vir       # Unit & integration test suite
`+"```"+`

## Running the Application

`+"```bash"+`
# Run directly
vira run .

# Run test suite
vira test tests/

# Build standalone production binary
vira build -prod -o bin/%s .
./bin/%s
`+"```"+`
`, name, name, name, name)
	_ = os.WriteFile(filepath.Join(targetDir, "README.md"), []byte(readmeContent), 0644)

	// 9. .gitignore
	gitIgnore := `build/
bin/
*.exe
*.o
out.c
.vir/
.vira/
`
	_ = os.WriteFile(filepath.Join(targetDir, ".gitignore"), []byte(gitIgnore), 0644)

	fmt.Printf("✓ Initialized modern Vira project %q in %s/\n", name, targetDir)
	fmt.Printf("  ├── vira.mod (package manifest)\n")
	fmt.Printf("  ├── .gitignore\n")
	fmt.Printf("  ├── README.md\n")
	fmt.Printf("  ├── src/\n")
	fmt.Printf("  │   ├── main.vir (entry point)\n")
	fmt.Printf("  │   ├── config/settings.vir\n")
	fmt.Printf("  │   ├── models/data.vir\n")
	fmt.Printf("  │   └── services/compute.vir\n")
	fmt.Printf("  └── tests/\n")
	fmt.Printf("      └── app_test.vir\n\n")
	fmt.Printf("Next steps:\n")
	if targetDir != "." {
		fmt.Printf("  cd %s\n", targetDir)
	}
	fmt.Printf("  vira run .        # Run project\n")
	fmt.Printf("  vira test tests/  # Run tests\n")
}

// --- SUBCOMMAND 9: `vira demo` ---

func demoCmd() {
	_ = os.MkdirAll("examples", 0755)
	demoFile := filepath.Join("examples", "parallel_ai_actors.vir")
	if err := os.WriteFile(demoFile, []byte(defaultDemoSource), 0644); err != nil {
		fatalIO(err)
	}
	fmt.Printf("✓ Created sample program: %s\n", demoFile)
	fmt.Println("✓ Running `vira run -prod examples/parallel_ai_actors.vir`...")
	fmt.Println()
	runCmd([]string{"-prod", demoFile})
}

// --- SUBCOMMAND 10: `vira add` ---

func addCmd(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: vira add <package[@version]> [version]")
		os.Exit(1)
	}

	raw := args[0]
	pkgPath := raw
	version := "latest"
	if idx := strings.Index(raw, "@"); idx != -1 {
		pkgPath = raw[:idx]
		version = raw[idx+1:]
	} else if len(args) >= 2 {
		version = args[1]
	}

	cwd, err := os.Getwd()
	if err != nil {
		fatalIO(err)
	}
	root, _, err := mod.FindProjectRoot(cwd)
	if err != nil {
		root = cwd
	}

	mgr := pkg.NewManager()
	fmt.Printf("Fetching package %s (%s)...\n", pkgPath, version)
	locked, err := mgr.Add(root, pkgPath, version)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✓ Added dependency %s (%s)\n", locked.Path, locked.Version)
	fmt.Printf("  checksum: %s\n", locked.Checksum)
}

// --- SUBCOMMAND 11: `vira install` ---

func installCmd(args []string) {
	cwd, err := os.Getwd()
	if err != nil {
		fatalIO(err)
	}
	root, _, err := mod.FindProjectRoot(cwd)
	if err != nil {
		root = cwd
	}

	mgr := pkg.NewManager()
	count, err := mgr.Install(root)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error installing dependencies: %v\n", err)
		os.Exit(1)
	}
	if count == 0 {
		fmt.Println("All dependencies are up to date.")
	} else {
		fmt.Printf("✓ Successfully installed %d package(s)\n", count)
	}
}

// --- SUBCOMMAND 12: `vira remove` ---

func removeCmd(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: vira remove <package>")
		os.Exit(1)
	}
	pkgPath := args[0]
	cwd, err := os.Getwd()
	if err != nil {
		fatalIO(err)
	}
	root, _, err := mod.FindProjectRoot(cwd)
	if err != nil {
		root = cwd
	}

	mgr := pkg.NewManager()
	if err := mgr.Remove(root, pkgPath); err != nil {
		fmt.Fprintf(os.Stderr, "error removing dependency: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("✓ Removed dependency %s\n", pkgPath)
}

// --- SUBCOMMAND 13: `vira vendor` ---

func vendorCmd(args []string) {
	cwd, err := os.Getwd()
	if err != nil {
		fatalIO(err)
	}
	root, _, err := mod.FindProjectRoot(cwd)
	if err != nil {
		root = cwd
	}

	mgr := pkg.NewManager()
	count, err := mgr.Vendor(root)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error vendoring dependencies: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("✓ Vendored %d package(s) into ./vendor\n", count)
}

// --- CORE FRONTEND + MIDDLEEND PIPELINE WITH MULTI-FILE SUPPORT ---

func compileVexToC99(targetPath string, isTestMode bool) (string, *checker.Checker, *autofree.Analyzer) {
	fi, err := os.Stat(targetPath)
	if err != nil {
		fatalIO(err)
	}

	startDir := targetPath
	if !fi.IsDir() {
		startDir = filepath.Dir(targetPath)
	}
	pRoot, manifest, _ := mod.FindProjectRoot(startDir)

	var fileQueue []string
	visited := make(map[string]bool)

	addFile := func(path string) bool {
		abs, err := filepath.Abs(path)
		if err != nil || visited[abs] {
			return false
		}
		visited[abs] = true
		fileQueue = append(fileQueue, abs)
		return true
	}

	if fi.IsDir() {
		_ = filepath.Walk(targetPath, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			if info.IsDir() {
				base := filepath.Base(path)
				if base == ".git" || base == "build" || base == "bin" || base == "dist" || base == "node_modules" || base == ".vir" || base == ".vira" {
					return filepath.SkipDir
				}
				if !isTestMode && base == "tests" {
					return filepath.SkipDir
				}
				if base == "examples" && !strings.HasSuffix(filepath.Clean(targetPath), "examples") {
					return filepath.SkipDir
				}
				return nil
			}
			isSource := (strings.HasSuffix(info.Name(), ".vir") || strings.HasSuffix(info.Name(), ".vex")) &&
				!strings.HasSuffix(info.Name(), "_test.vir") && !strings.HasSuffix(info.Name(), "_test.vex")
			if isSource {
				addFile(path)
			}
			return nil
		})
		if len(fileQueue) == 0 {
			fmt.Fprintf(os.Stderr, "no .vir source files found in %s\n", targetPath)
			os.Exit(1)
		}
	} else {
		absTarget, _ := filepath.Abs(targetPath)
		if pRoot != "" {
			_ = filepath.Walk(pRoot, func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return nil
				}
				if info.IsDir() {
					base := filepath.Base(path)
					if base == ".git" || base == "build" || base == "bin" || base == "dist" || base == "node_modules" || base == ".vir" || base == ".vira" || base == "tests" || base == "vendor" || (base == "examples" && !strings.Contains(absTarget, string(filepath.Separator)+"examples"+string(filepath.Separator))) {
						return filepath.SkipDir
					}
					return nil
				}
				absPath, _ := filepath.Abs(path)
				if absPath == absTarget {
					return nil
				}
				isSource := (strings.HasSuffix(info.Name(), ".vir") || strings.HasSuffix(info.Name(), ".vex")) &&
					!strings.HasSuffix(info.Name(), "_test.vir") && !strings.HasSuffix(info.Name(), "_test.vex")
				if isSource {
					addFile(path)
				}
				return nil
			})
		}
		addFile(targetPath)
	}

	var mergedDecls []ast.Decl
	var firstMod ast.Ident
	var srcFiles []*source.File
	var testFnNames []string
	hasMain := false

	for i := 0; i < len(fileQueue); i++ {
		vFile := fileQueue[i]
		rawBytes, err := os.ReadFile(vFile)
		if err != nil {
			fatalIO(err)
		}
		srcText := string(rawBytes)
		srcFile := source.New(vFile, srcText)
		srcFiles = append(srcFiles, srcFile)

		p := parser.New(srcText)
		astFile := p.ParseFile()
		if parseErrs := p.Errors(); len(parseErrs) > 0 {
			for _, pe := range parseErrs {
				fmt.Fprintln(os.Stderr, srcFile.FormatDiagnostic("SYNTAX", pe.Msg, pe.Span))
			}
			fmt.Fprintf(os.Stderr, "compilation aborted due to %d syntax error(s) in %s\n", len(parseErrs), vFile)
			os.Exit(1)
		}

		if firstMod.Name == "" && astFile.Module.Name != "" {
			firstMod = astFile.Module
		}

		// Resolve explicit import declarations dynamically
		vDir := filepath.Dir(vFile)
		for _, imp := range astFile.Imports {
			impTarget := imp.Name
			candidates := pkg.ResolveImportCandidates(vDir, pRoot, impTarget)

			found := false
			for _, cand := range candidates {
				cStat, cErr := os.Stat(cand)
				if cErr == nil {
					if cStat.IsDir() {
						_ = filepath.Walk(cand, func(subPath string, subInfo os.FileInfo, err error) error {
							if err != nil {
								return nil
							}
							if subInfo.IsDir() {
								if subInfo.Name() == "tests" || subInfo.Name() == ".git" {
									return filepath.SkipDir
								}
								return nil
							}
							if (strings.HasSuffix(subInfo.Name(), ".vir") || strings.HasSuffix(subInfo.Name(), ".vex")) &&
								!strings.HasSuffix(subInfo.Name(), "_test.vir") && !strings.HasSuffix(subInfo.Name(), "_test.vex") {
								addFile(subPath)
								found = true
							}
							return nil
						})
						if found {
							break
						}
					} else {
						if strings.HasSuffix(cand, ".vir") || strings.HasSuffix(cand, ".vex") {
							addFile(cand)
							found = true
							break
						}
					}
				}
			}
			if !found {
				fmt.Fprintln(os.Stderr, srcFile.FormatDiagnostic("IMPORT", fmt.Sprintf("cannot find imported file or package '%s'", impTarget), imp.NamePos))
				os.Exit(1)
			}
		}

		for _, decl := range astFile.Decls {
			if fn, ok := decl.(*ast.FnDecl); ok {
				if fn.Name.Name == "main" {
					if isTestMode && vFile != targetPath {
						// Skip application entrypoint during tests so test runner main() is synthesized
						continue
					}
					hasMain = true
				} else if strings.HasPrefix(fn.Name.Name, "test_") {
					testFnNames = append(testFnNames, fn.Name.Name)
				}
			}
			mergedDecls = append(mergedDecls, decl)
		}
	}

	if manifest != nil && manifest.Module != "" {
		firstMod.Name = manifest.Module
	}
	if firstMod.Name == "" {
		firstMod.Name = "main"
	}

	// In test mode: synthesize a test runner main() if none is defined
	if isTestMode && !hasMain && len(testFnNames) > 0 {
		var runnerSrc strings.Builder
		runnerSrc.WriteString("fn main() {\n")
		for _, tName := range testFnNames {
			runnerSrc.WriteString(fmt.Sprintf("\t%s()\n", tName))
		}
		runnerSrc.WriteString("}\n")

		pRunner := parser.New(runnerSrc.String())
		runnerAst := pRunner.ParseFile()
		mergedDecls = append(mergedDecls, runnerAst.Decls...)
	}

	mergedAst := &ast.File{
		Module: firstMod,
		Decls:  mergedDecls,
	}

	// Phase 2: 3-Pass Type, Mutability, Tensor & Actor Checker
	chk := checker.New()
	chk.CheckFile(mergedAst)
	if semErrs := chk.Errors(); len(semErrs) > 0 {
		for _, se := range semErrs {
			sFile := srcFiles[0]
			fmt.Fprintln(os.Stderr, sFile.FormatDiagnostic("SEMANTIC", se.Msg, se.Span))
		}
		fmt.Fprintf(os.Stderr, "compilation aborted due to %d semantic error(s)\n", len(semErrs))
		os.Exit(1)
	}

	// Phase 3: Static Escape Analysis & Deterministic LIFO Destructor Injection
	af := autofree.New(chk)
	af.AnalyzeFile(mergedAst)

	// Phase 4: C99 Code Emission
	gen := cgen.New(chk, af)
	cCode := gen.EmitFile(mergedAst)

	return cCode, chk, af
}

// --- C99 COMPILER DRIVER ---

func invokeCBackend(cFile, binFile string, opts compileOptions) (string, string) {
	runtimeDir := resolveRuntimeDir()
	compiler := opts.cc
	var cFlags []string
	var modeDesc string

	if compiler == "" {
		for _, cand := range []string{"gcc", "clang", "cc", "tcc"} {
			if _, err := exec.LookPath(cand); err == nil {
				compiler = cand
				break
			}
		}
		if compiler == "" {
			if home, err := os.UserHomeDir(); err == nil {
				cand := filepath.Join(home, "w64devkit", "bin", "gcc.exe")
				if info, err := os.Stat(cand); err == nil && !info.IsDir() {
					compiler = cand
				}
			}
		}
		if compiler == "" {
			for _, cand := range []string{`C:\w64devkit\bin\gcc.exe`, `C:\msys64\ucrt64\bin\gcc.exe`, `C:\msys64\mingw64\bin\gcc.exe`} {
				if info, err := os.Stat(cand); err == nil && !info.IsDir() {
					compiler = cand
					break
				}
			}
		}
		if compiler == "" {
			fmt.Fprintln(os.Stderr, "error: no C99 backend compiler found in PATH (searched: gcc, clang, cc, tcc)")
			fmt.Fprintln(os.Stderr, "  = help: Install GCC (MinGW-w64 or w64devkit) or Clang (LLVM) and add it to your PATH.")
			fmt.Fprintln(os.Stderr, "  = note: You can inspect and emit C99 code directly using: vira cgen <file.vir>")
			os.Exit(1)
		}
	}

	archFlag := "-march=native"

	incFlags, libFlags, hasRaylib, staticRaylibPath := resolveExternalLibPaths()

	extraLibs := []string{}
	staticFlags := []string{}

	if goRuntime.GOOS == "windows" {
		staticFlags = append(staticFlags, "-static")
		if hasRaylib {
			if staticRaylibPath != "" {
				extraLibs = append(extraLibs, staticRaylibPath)
			} else {
				extraLibs = append(extraLibs, "-lraylib")
			}
		}
		extraLibs = append(extraLibs, "-lopengl32", "-lgdi32", "-luser32", "-lwinmm", "-lws2_32")
	} else if goRuntime.GOOS == "linux" {
		extraLibs = append(extraLibs, "-lX11")
		if hasRaylib {
			if staticRaylibPath != "" {
				extraLibs = append(extraLibs, staticRaylibPath, "-lGL", "-ldl", "-lrt")
			} else {
				extraLibs = append(extraLibs, "-lraylib", "-lGL", "-ldl", "-lrt")
			}
		}
		extraLibs = append(extraLibs, "-lm")
	} else if goRuntime.GOOS == "darwin" && hasRaylib {
		if staticRaylibPath != "" {
			extraLibs = append(extraLibs, staticRaylibPath, "-framework", "OpenGL", "-framework", "Cocoa", "-framework", "IOKit", "-framework", "CoreVideo")
		} else {
			extraLibs = append(extraLibs, "-lraylib", "-framework", "OpenGL", "-framework", "Cocoa", "-framework", "IOKit", "-framework", "CoreVideo")
		}
	}

	baseIncludes := append([]string{"-I" + runtimeDir}, incFlags...)
	baseIncludes = append(baseIncludes, libFlags...)
	if hasRaylib {
		baseIncludes = append(baseIncludes, "-DVIRA_USE_RAYLIB")
	}

	if opts.prod {
		modeDesc = "prod (-O3 " + archFlag + ")"
		cFlags = append([]string{"-std=c99", "-O3", archFlag, "-mstackrealign", "-pthread"}, staticFlags...)
		cFlags = append(cFlags, baseIncludes...)
		cFlags = append(cFlags, cFile, "-o", binFile)
		cFlags = append(cFlags, extraLibs...)
	} else if opts.sanitize {
		modeDesc = "sanitize (ASan+UBSan)"
		cFlags = append([]string{"-std=c99", "-O1", "-g", "-mstackrealign", "-fsanitize=address,undefined", "-pthread"}, baseIncludes...)
		cFlags = append(cFlags, cFile, "-o", binFile)
		cFlags = append(cFlags, extraLibs...)
	} else if compiler == "tcc" {
		modeDesc = "dev (TinyCC instant build)"
		cFlags = append([]string{"-pthread"}, baseIncludes...)
		cFlags = append(cFlags, cFile, "-o", binFile)
		cFlags = append(cFlags, extraLibs...)
	} else {
		modeDesc = "dev (-O2 " + archFlag + ")"
		cFlags = append([]string{"-std=c99", "-O2", archFlag, "-mstackrealign", "-pthread"}, staticFlags...)
		cFlags = append(cFlags, baseIncludes...)
		cFlags = append(cFlags, cFile, "-o", binFile)
		cFlags = append(cFlags, extraLibs...)
	}

	cmd := exec.Command(compiler, cFlags...)
	compilerDir := filepath.Dir(compiler)
	cmd.Env = append(os.Environ(), "PATH="+compilerDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "C backend (%s) failed: %v\n", compiler, err)
		os.Exit(1)
	}
	return compiler, modeDesc
}

func resolveExternalLibPaths() ([]string, []string, bool, string) {
	var incPaths []string
	var libPaths []string
	hasRaylib := false
	staticRaylibPath := ""

	home, _ := os.UserHomeDir()
	exe, _ := os.Executable()
	exeDir := filepath.Dir(exe)

	searchRoots := []string{
		"vendor",
		exeDir,
		filepath.Join(exeDir, "vendor"),
	}
	if home != "" {
		searchRoots = append(searchRoots,
			filepath.Join(home, ".vira"),
			filepath.Join(home, ".vir"),
			filepath.Join(home, ".vex"),
		)
	}
	if goRuntime.GOOS != "windows" {
		searchRoots = append(searchRoots,
			"/usr/local",
			"/usr",
			"/opt/homebrew",
		)
	}

	seenInc := make(map[string]bool)
	seenLib := make(map[string]bool)

	addInc := func(p string) {
		if !seenInc[p] {
			seenInc[p] = true
			incPaths = append(incPaths, "-I"+p)
		}
	}
	addLib := func(p string) {
		if !seenLib[p] {
			seenLib[p] = true
			libPaths = append(libPaths, "-L"+p)
			if goRuntime.GOOS == "linux" {
				libPaths = append(libPaths, "-Wl,-rpath,"+p)
			}
		}
	}

	for _, root := range searchRoots {
		// Generic include & lib directories
		inc := filepath.Join(root, "include")
		lib := filepath.Join(root, "lib")
		lib64 := filepath.Join(root, "lib64")
		libArch := filepath.Join(root, "lib", "x86_64-linux-gnu")

		if info, err := os.Stat(inc); err == nil && info.IsDir() {
			addInc(inc)
		}
		if info, err := os.Stat(lib); err == nil && info.IsDir() {
			addLib(lib)
		}
		if info, err := os.Stat(lib64); err == nil && info.IsDir() {
			addLib(lib64)
		}
		if info, err := os.Stat(libArch); err == nil && info.IsDir() {
			addLib(libArch)
		}

		// Specific C libraries (Raylib, Raygui, Nuklear, SDL2)
		for _, libName := range []string{"raylib", "raygui", "nuklear", "sdl2"} {
			libInc := filepath.Join(root, libName, "include")
			libDir := filepath.Join(root, libName, "lib")
			vendorInc := filepath.Join(root, "vendor", libName, "include")
			vendorLib := filepath.Join(root, "vendor", libName, "lib")
			if info, err := os.Stat(libInc); err == nil && info.IsDir() {
				addInc(libInc)
			}
			if info, err := os.Stat(libDir); err == nil && info.IsDir() {
				addLib(libDir)
			}
			if info, err := os.Stat(vendorInc); err == nil && info.IsDir() {
				addInc(vendorInc)
			}
			if info, err := os.Stat(vendorLib); err == nil && info.IsDir() {
				addLib(vendorLib)
			}
		}

		// Check if raylib static or shared library is present
		for _, cand := range []string{
			filepath.Join(root, "libraylib.a"),
			filepath.Join(root, "raylib.lib"),
			filepath.Join(root, "lib", "libraylib.a"),
			filepath.Join(root, "lib", "libraylib.so"),
			filepath.Join(root, "lib", "libraylib.dylib"),
			filepath.Join(root, "lib64", "libraylib.a"),
			filepath.Join(root, "lib", "x86_64-linux-gnu", "libraylib.a"),
			filepath.Join(root, "lib", "x86_64-linux-gnu", "libraylib.so"),
			filepath.Join(root, "raylib", "lib", "libraylib.a"),
			filepath.Join(root, "vendor", "raylib", "lib", "libraylib.a"),
		} {
			if fi, err := os.Stat(cand); err == nil && !fi.IsDir() {
				hasRaylib = true
				if strings.HasSuffix(cand, ".a") || strings.HasSuffix(cand, ".lib") {
					if staticRaylibPath == "" {
						staticRaylibPath = cand
					}
				}
				break
			}
		}
	}

	return incPaths, libPaths, hasRaylib, staticRaylibPath
}

func resolveRuntimeDir() string {
	if info, err := os.Stat("runtime/vira_rt.h"); err == nil && !info.IsDir() {
		return "runtime"
	}
	if exe, err := os.Executable(); err == nil {
		cand := filepath.Join(filepath.Dir(exe), "runtime")
		if info, err := os.Stat(filepath.Join(cand, "vira_rt.h")); err == nil && !info.IsDir() {
			return cand
		}
	}

	// Portable fallback: Extract embedded headers from virart.RuntimeFS to ~/.vir/runtime/
	home, err := os.UserHomeDir()
	if err == nil {
		cacheDir := filepath.Join(home, ".vira", "runtime")
		_ = os.MkdirAll(cacheDir, 0755)

		entries, err := virart.RuntimeFS.ReadDir(".")
		if err == nil && len(entries) > 0 {
			for _, e := range entries {
				if !e.IsDir() && strings.HasSuffix(e.Name(), ".h") {
					data, err := virart.RuntimeFS.ReadFile(e.Name())
					if err == nil {
						_ = os.WriteFile(filepath.Join(cacheDir, e.Name()), data, 0644)
					}
				}
			}
			return cacheDir
		}
	}
	return "runtime"
}

func fatalIO(err error) {
	fmt.Fprintf(os.Stderr, "vex io error: %v\n", err)
	os.Exit(1)
}

const defaultDemoSource = `module billing

pub struct AuditMeta {
pub:
	created_at i64
pub mut:
	updated_at i64
}

pub fn (mut a AuditMeta) touch() {
	a.updated_at = 1760000000
}

pub struct Invoice {
	AuditMeta
	secret_token string
mut:
	retry_count int = 0
pub:
	id string
	amount_cents i64
pub mut:
	is_paid bool
}

pub interface Payable {
	id string
	summary() string
mut:
	mark_paid() !string
}

pub fn Invoice.new(id string, amount_cents i64) Invoice {
	res := Invoice{
		AuditMeta: AuditMeta{
			created_at: 1750000000,
			updated_at: 1750000000
		},
		secret_token: 'sec_live_01',
		id: id,
		amount_cents: amount_cents,
		is_paid: false
	}
	return res
}

pub fn (inv Invoice) summary() string {
	return 'Invoice #${inv.id} [updated_at=${inv.updated_at}]'
}

pub fn (mut inv Invoice) mark_paid() !string {
	if inv.amount_cents <= 0 {
		return error('invalid invoice amount')
	}
	inv.is_paid = true
	inv.retry_count += 1
	inv.touch()
	return 'Invoice #${inv.id} paid (${inv.amount_cents} cents)'
}

pub actor AIWorker {
pub:
	id int
mut:
	batches_run int = 0
}

pub fn (mut w AIWorker) predict(scale f32, simulate_fault bool) f32 {
	w.batches_run += 1
	weights := tensor_fill(64, 64, 0.25)
	input := tensor_fill(64, 64, scale)
	out := weights @ input

	if simulate_fault {
		panic('CUDA/NaN anomaly on parallel batch')
	}
	return tensor_sum(out)
}

fn main() {
	println('=== 1. OOP + STRUCT EMBEDDING + FAT-POINTER INTERFACE ===')
	mut inv := Invoice.new('inv_9001', 15000)
	msg := inv.mark_paid() or {
		eprintln(err)
		return
	}
	println(msg)
	p := Payable(inv)
	_ := p

	println('=== 2. PARALLEL FUTURE[f32] FAN-OUT ACROSS WORK-STEALING POOL ===')
	mut w1 := AIWorker{ id: 1, batches_run: 0 }
	mut w2 := AIWorker{ id: 2, batches_run: 0 }
	mut w3 := AIWorker{ id: 3, batches_run: 0 }

	fut1 := w1 <- predict(2.0, false)
	fut2 := w2 <- predict(3.0, true)
	fut3 := w3 <- predict(4.0, false)

	s1 := fut1.await() or { 0.0 }
	s2 := fut2.await() or {
		println('Future #2 caught parallel Worker #2 fault: ${err}')
		0.0
	}
	s3 := fut3.await() or { 0.0 }

	println('Parallel Future results -> w1=${s1}, w2(faulted)=${s2}, w3=${s3}')

	fut2_retry := w2 <- predict(3.0, false)
	s2_recovered := fut2_retry.await() or { 0.0 }
	println('Worker #2 recovered Future result -> w2=${s2_recovered}')
}
`
