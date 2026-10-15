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

	"github.com/vexlang/vex/internal/ast"
	"github.com/vexlang/vex/internal/autofree"
	"github.com/vexlang/vex/internal/cgen"
	"github.com/vexlang/vex/internal/checker"
	"github.com/vexlang/vex/internal/formatter"
	"github.com/vexlang/vex/internal/lsp"
	"github.com/vexlang/vex/internal/mod"
	"github.com/vexlang/vex/internal/parser"
	"github.com/vexlang/vex/internal/source"
	vexrt "github.com/vexlang/vex/runtime"
)

const usageBanner = `Vex Programming Language Compiler (Stage-0 Toolchain)

Usage:
  vex <command> [flags] [target]

Commands:
  run     Compile and execute a .vex file or directory immediately
  build   Compile a .vex file or directory into a standalone native binary
  cgen    Emit lowered C99 code (SIMD + Work-Stealing Actors)
  check   Run syntax, 4-quadrant mutability, and autofree escape analysis
  test    Discover and run test functions/files with leak checks
  fmt     Format .vex source files according to official style rules
  lsp     Start the Vex Language Server (LSP) over stdin/stdout
  init    Initialize a new Vex project scaffold with vex.mod and src/
  demo    Generate and run the built-in OOP + SIMD Tensor + Actor demo

Flags for 'vex build':
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
	case "demo":
		demoCmd()
	case "-h", "--help", "help":
		fmt.Print(usageBanner)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", subcmd, usageBanner)
		os.Exit(1)
	}
}

// --- SUBCOMMAND 1: `vex run` ---

func runCmd(args []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	ccFlag := fs.String("cc", "", "C compiler to use (auto-detects tcc, cc, gcc, or clang)")
	prodFlag := fs.Bool("prod", false, "Run with -O3 -march=native optimizations")
	_ = fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: vex run [-prod] [-cc compiler] <file.vex | directory>")
		os.Exit(1)
	}
	vexPath := fs.Arg(0)

	cCode, _, _ := compileVexToC99(vexPath, false)

	tmpDir, err := os.MkdirTemp("", "vex_run_*")
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

// --- SUBCOMMAND 2: `vex build` ---

func buildCmd(args []string) {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	prod := fs.Bool("prod", false, "Compile with -O3 -march=native -flto")
	sanitize := fs.Bool("sanitize", false, "Enable ASan and UBSan")
	outPath := fs.String("o", "", "Output binary path")
	ccFlag := fs.String("cc", "", "C compiler (cc, clang, gcc, tcc)")
	keepC := fs.Bool("keep-c", false, "Keep generated C99 file in build/")
	_ = fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: vex build [-prod] [-sanitize] [-o out] <file.vex | directory>")
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

// --- SUBCOMMAND 3: `vex cgen` ---

func cgenCmd(args []string) {
	fs := flag.NewFlagSet("cgen", flag.ExitOnError)
	outPath := fs.String("o", "", "Write generated C99 to file instead of stdout")
	_ = fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: vex cgen [-o out.c] <file.vex | directory>")
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

// --- SUBCOMMAND 4: `vex check` ---

func checkCmd(args []string) {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	showAutofree := fs.Bool("autofree", true, "Print static escape analysis and autofree report")
	_ = fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: vex check [-autofree] <file.vex | directory>")
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

// --- SUBCOMMAND 5: `vex test` ---

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
			if err == nil && !info.IsDir() && strings.HasSuffix(path, ".vex") {
				if strings.HasSuffix(path, "_test.vex") {
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
		fmt.Println("No tests found (searching for *_test.vex or files containing 'fn test_')")
		return
	}

	passCount := 0
	failCount := 0
	startTotal := time.Now()

	for _, tFile := range testFiles {
		fmt.Printf("=== RUN   %s\n", tFile)
		start := time.Now()
		cCode, _, _ := compileVexToC99(tFile, true)

		tmpDir, err := os.MkdirTemp("", "vex_test_*")
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

// --- SUBCOMMAND 6: `vex fmt` ---

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
			fmt.Fprintf(os.Stderr, "vex fmt error: %v\n", err)
			continue
		}
		if fi.IsDir() {
			_ = filepath.Walk(target, func(path string, info os.FileInfo, err error) error {
				if err == nil && !info.IsDir() && strings.HasSuffix(path, ".vex") {
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
		fmt.Fprintf(os.Stderr, "vex fmt read error (%s): %v\n", path, err)
		return
	}
	src := string(raw)
	formatted := formatter.Format(src)
	if writeBack {
		if src != formatted {
			if err := os.WriteFile(path, []byte(formatted), 0644); err != nil {
				fmt.Fprintf(os.Stderr, "vex fmt write error (%s): %v\n", path, err)
				return
			}
			fmt.Printf("reformatted %s\n", path)
		}
	} else {
		fmt.Print(formatted)
	}
}

// --- SUBCOMMAND 7: `vex lsp` ---

func lspCmd() {
	server := lsp.NewServer(os.Stdin, os.Stdout)
	if err := server.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "vex lsp error: %v\n", err)
		os.Exit(1)
	}
}

// --- SUBCOMMAND 8: `vex init` ---

func initCmd(args []string) {
	name := "my_vex_app"
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

	_ = os.MkdirAll(filepath.Join(targetDir, "src"), 0755)
	_ = os.MkdirAll(filepath.Join(targetDir, "tests"), 0755)

	modContent := fmt.Sprintf(`module %s
version 0.1.0
license MIT
description High-performance Vex application
`, modName)
	_ = os.WriteFile(filepath.Join(targetDir, "vex.mod"), []byte(modContent), 0644)

	mainContent := fmt.Sprintf(`module main

fn main() {
	println('Welcome to %s powered by Vex!')
}
`, modName)
	_ = os.WriteFile(filepath.Join(targetDir, "src", "main.vex"), []byte(mainContent), 0644)

	gitIgnore := `build/
bin/
*.exe
*.o
out.c
`
	_ = os.WriteFile(filepath.Join(targetDir, ".gitignore"), []byte(gitIgnore), 0644)

	fmt.Printf("✓ Initialized new Vex project %q in %s/\n", name, targetDir)
	fmt.Printf("  • vex.mod (package manifest)\n")
	fmt.Printf("  • src/main.vex (entry point)\n")
	fmt.Printf("  • .gitignore\n\n")
	fmt.Printf("Next steps:\n")
	if targetDir != "." {
		fmt.Printf("  cd %s\n", targetDir)
	}
	fmt.Printf("  vex run src/main.vex\n")
}

// --- SUBCOMMAND 9: `vex demo` ---

func demoCmd() {
	_ = os.MkdirAll("examples", 0755)
	demoFile := filepath.Join("examples", "parallel_ai_actors.vex")
	if err := os.WriteFile(demoFile, []byte(defaultDemoSource), 0644); err != nil {
		fatalIO(err)
	}
	fmt.Printf("✓ Created sample program: %s\n", demoFile)
	fmt.Println("✓ Running `vex run -prod examples/parallel_ai_actors.vex`...")
	fmt.Println()
	runCmd([]string{"-prod", demoFile})
}

// --- CORE FRONTEND + MIDDLEEND PIPELINE WITH MULTI-FILE SUPPORT ---

func compileVexToC99(targetPath string, isTestMode bool) (string, *checker.Checker, *autofree.Analyzer) {
	fi, err := os.Stat(targetPath)
	if err != nil {
		fatalIO(err)
	}

	var vexFiles []string
	if fi.IsDir() {
		entries, err := os.ReadDir(targetPath)
		if err != nil {
			fatalIO(err)
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".vex") && !strings.HasSuffix(e.Name(), "_test.vex") {
				vexFiles = append(vexFiles, filepath.Join(targetPath, e.Name()))
			}
		}
		if len(vexFiles) == 0 {
			srcDir := filepath.Join(targetPath, "src")
			if sfi, sErr := os.Stat(srcDir); sErr == nil && sfi.IsDir() {
				sEntries, _ := os.ReadDir(srcDir)
				for _, se := range sEntries {
					if !se.IsDir() && strings.HasSuffix(se.Name(), ".vex") {
						vexFiles = append(vexFiles, filepath.Join(srcDir, se.Name()))
					}
				}
			}
		}
		if len(vexFiles) == 0 {
			fmt.Fprintf(os.Stderr, "no .vex source files found in %s\n", targetPath)
			os.Exit(1)
		}
	} else {
		vexFiles = []string{targetPath}
	}

	var mergedDecls []ast.Decl
	var firstMod ast.Ident
	var srcFiles []*source.File
	var testFnNames []string
	hasMain := false

	for _, vFile := range vexFiles {
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

		for _, decl := range astFile.Decls {
			if fn, ok := decl.(*ast.FnDecl); ok {
				if fn.Name.Name == "main" {
					hasMain = true
				} else if strings.HasPrefix(fn.Name.Name, "test_") {
					testFnNames = append(testFnNames, fn.Name.Name)
				}
			}
			mergedDecls = append(mergedDecls, decl)
		}
	}

	if root, manifest, err := mod.FindProjectRoot(targetPath); err == nil && manifest != nil {
		_ = root
		if manifest.Module != "" {
			firstMod.Name = manifest.Module
		}
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
			fmt.Fprintln(os.Stderr, "  = note: You can inspect and emit C99 code directly using: vex cgen <file.vex>")
			os.Exit(1)
		}
	}

	archFlag := "-march=native"
	if goRuntime.GOARCH == "amd64" {
		archFlag = "-march=x86-64-v3"
	}

	incFlags, libFlags, hasRaylib := resolveExternalLibPaths()

	extraLibs := []string{}
	staticFlags := []string{}

	if goRuntime.GOOS == "windows" {
		staticFlags = append(staticFlags, "-static")
		extraLibs = append(extraLibs, "-lws2_32", "-lgdi32", "-luser32", "-lwinmm", "-lopengl32")
		if hasRaylib {
			extraLibs = append(extraLibs, "-lraylib")
		}
	} else if goRuntime.GOOS == "linux" {
		extraLibs = append(extraLibs, "-lm")
		if hasRaylib {
			extraLibs = append(extraLibs, "-lraylib", "-lGL", "-lX11", "-ldl", "-lrt")
		}
	} else if goRuntime.GOOS == "darwin" && hasRaylib {
		extraLibs = append(extraLibs, "-lraylib", "-framework", "OpenGL", "-framework", "Cocoa", "-framework", "IOKit", "-framework", "CoreVideo")
	}

	baseIncludes := append([]string{"-I" + runtimeDir}, incFlags...)
	baseIncludes = append(baseIncludes, libFlags...)

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

func resolveExternalLibPaths() ([]string, []string, bool) {
	var incPaths []string
	var libPaths []string
	hasRaylib := false

	home, _ := os.UserHomeDir()
	exe, _ := os.Executable()
	exeDir := filepath.Dir(exe)

	searchRoots := []string{
		"vendor",
		"lib",
		filepath.Join(exeDir, "vendor"),
		filepath.Join(exeDir, "lib"),
	}
	if home != "" {
		searchRoots = append(searchRoots,
			filepath.Join(home, ".vex", "lib"),
			filepath.Join(home, ".vex", "vendor"),
		)
	}

	for _, root := range searchRoots {
		// Generic lib/include layout
		inc := filepath.Join(root, "include")
		lib := filepath.Join(root, "lib")
		if info, err := os.Stat(inc); err == nil && info.IsDir() {
			incPaths = append(incPaths, "-I"+inc)
		}
		if info, err := os.Stat(lib); err == nil && info.IsDir() {
			libPaths = append(libPaths, "-L"+lib)
		}

		// Specific C libraries (Raylib, Raygui, Nuklear, SDL2)
		for _, libName := range []string{"raylib", "raygui", "nuklear", "sdl2"} {
			libInc := filepath.Join(root, libName, "include")
			libDir := filepath.Join(root, libName, "lib")
			if info, err := os.Stat(libInc); err == nil && info.IsDir() {
				incPaths = append(incPaths, "-I"+libInc)
			}
			if info, err := os.Stat(libDir); err == nil && info.IsDir() {
				libPaths = append(libPaths, "-L"+libDir)
			}
		}

		// Check if raylib static library is present
		for _, cand := range []string{
			filepath.Join(root, "libraylib.a"),
			filepath.Join(root, "raylib.lib"),
			filepath.Join(root, "lib", "libraylib.a"),
			filepath.Join(root, "raylib", "lib", "libraylib.a"),
		} {
			if fi, err := os.Stat(cand); err == nil && !fi.IsDir() {
				hasRaylib = true
				break
			}
		}
	}

	return incPaths, libPaths, hasRaylib
}

func resolveRuntimeDir() string {
	if info, err := os.Stat("runtime/vex_rt.h"); err == nil && !info.IsDir() {
		return "runtime"
	}
	if exe, err := os.Executable(); err == nil {
		cand := filepath.Join(filepath.Dir(exe), "runtime")
		if info, err := os.Stat(filepath.Join(cand, "vex_rt.h")); err == nil && !info.IsDir() {
			return cand
		}
	}

	// Portable fallback: Extract embedded headers from vexrt.RuntimeFS to ~/.vex/runtime/
	home, err := os.UserHomeDir()
	if err == nil {
		cacheDir := filepath.Join(home, ".vex", "runtime")
		_ = os.MkdirAll(cacheDir, 0755)

		entries, err := vexrt.RuntimeFS.ReadDir(".")
		if err == nil && len(entries) > 0 {
			for _, e := range entries {
				if !e.IsDir() && strings.HasSuffix(e.Name(), ".h") {
					data, err := vexrt.RuntimeFS.ReadFile(e.Name())
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
