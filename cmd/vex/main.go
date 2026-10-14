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

	"github.com/vexlang/vex/internal/autofree"
	"github.com/vexlang/vex/internal/cgen"
	"github.com/vexlang/vex/internal/checker"
	"github.com/vexlang/vex/internal/parser"
	"github.com/vexlang/vex/internal/source"
)

const usageBanner = `Vex Programming Language Compiler (Stage-0 Toolchain)

Usage:
  vex <command> [flags] <file.vex>

Commands:
  run     Compile and execute a .vex source file immediately
  build   Compile a .vex file into a native standalone binary
  cgen    Emit the lowered C99 code (SIMD + Work-Stealing Actors)
  check   Run syntax, 4-quadrant mutability, and autofree escape analysis
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
		fmt.Fprintln(os.Stderr, "usage: vex run [-prod] [-cc compiler] <file.vex>")
		os.Exit(1)
	}
	vexPath := fs.Arg(0)

	cCode, _, _ := compileVexToC99(vexPath)

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
		fmt.Fprintln(os.Stderr, "usage: vex build [-prod] [-sanitize] [-o out] <file.vex>")
		os.Exit(1)
	}
	vexPath := fs.Arg(0)
	start := time.Now()

	cCode, _, _ := compileVexToC99(vexPath)

	baseName := strings.TrimSuffix(filepath.Base(vexPath), filepath.Ext(vexPath))
	targetBin := *outPath
	if targetBin == "" {
		targetBin = baseName
		if os.PathSeparator == '\\' && !strings.HasSuffix(targetBin, ".exe") {
			targetBin += ".exe"
		}
	}

	var cFile string
	if *keepC {
		_ = os.MkdirAll("build", 0755)
		cFile = filepath.Join("build", baseName+".c")
	} else {
		tmpDir, err := os.MkdirTemp("", "vex_build_*")
		if err != nil {
			fatalIO(err)
		}
		defer os.RemoveAll(tmpDir)
		cFile = filepath.Join(tmpDir, baseName+".c")
	}

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
	usedCC, modeDesc := invokeCBackend(cFile, targetBin, opts)

	elapsed := time.Since(start).Milliseconds()
	fmt.Printf("✓ Built '%s' [%s via %s] in %d ms\n", targetBin, modeDesc, usedCC, elapsed)
	if *keepC {
		fmt.Printf("  └── Kept C99 translation unit: %s\n", cFile)
	}
}

// --- SUBCOMMAND 3: `vex cgen` ---

func cgenCmd(args []string) {
	fs := flag.NewFlagSet("cgen", flag.ExitOnError)
	outPath := fs.String("o", "", "Write generated C99 to file instead of stdout")
	_ = fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: vex cgen [-o out.c] <file.vex>")
		os.Exit(1)
	}
	vexPath := fs.Arg(0)
	cCode, _, _ := compileVexToC99(vexPath)

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
		fmt.Fprintln(os.Stderr, "usage: vex check [-autofree] <file.vex>")
		os.Exit(1)
	}
	vexPath := fs.Arg(0)
	_, chk, af := compileVexToC99(vexPath)

	fmt.Printf("✓ %s passed syntax, 4-quadrant mutability, and type checks (0 errors)\n", vexPath)
	if *showAutofree {
		fmt.Println("\n" + af.FormatReport())
	}
	_ = chk
}

// --- SUBCOMMAND 5: `vex demo` ---

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

// --- CORE FRONTEND + MIDDLEEND PIPELINE WITH SOURCE-SPAN DIAGNOSTICS ---

func compileVexToC99(vexPath string) (string, *checker.Checker, *autofree.Analyzer) {
	rawBytes, err := os.ReadFile(vexPath)
	if err != nil {
		fatalIO(err)
	}
	srcText := string(rawBytes)
	srcFile := source.New(vexPath, srcText)

	// Phase 1: Parse into AST
	p := parser.New(srcText)
	astFile := p.ParseFile()
	if parseErrs := p.Errors(); len(parseErrs) > 0 {
		for _, pe := range parseErrs {
			fmt.Fprintln(os.Stderr, srcFile.FormatDiagnostic("SYNTAX", pe.Msg, pe.Span))
		}
		fmt.Fprintf(os.Stderr, "compilation aborted due to %d syntax error(s)\n", len(parseErrs))
		os.Exit(1)
	}

	// Phase 2: 3-Pass Type, Mutability, Tensor & Actor Checker
	chk := checker.New()
	chk.CheckFile(astFile)
	if semErrs := chk.Errors(); len(semErrs) > 0 {
		for _, se := range semErrs {
			fmt.Fprintln(os.Stderr, srcFile.FormatDiagnostic("SEMANTIC", se.Msg, se.Span))
		}
		fmt.Fprintf(os.Stderr, "compilation aborted due to %d semantic error(s)\n", len(semErrs))
		os.Exit(1)
	}

	// Phase 3: Static Escape Analysis & Deterministic LIFO Destructor Injection
	af := autofree.New(chk)
	af.AnalyzeFile(astFile)

	// Phase 4: C99 Code Emission
	gen := cgen.New(chk, af)
	cCode := gen.EmitFile(astFile)

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

	if opts.prod {
		modeDesc = "prod (-O3 " + archFlag + ")"
		cFlags = []string{"-std=c99", "-O3", archFlag, "-mstackrealign", "-static", "-pthread", "-I" + runtimeDir, cFile, "-o", binFile}
	} else if opts.sanitize {
		modeDesc = "sanitize (ASan+UBSan)"
		cFlags = []string{"-std=c99", "-O1", "-g", "-mstackrealign", "-fsanitize=address,undefined", "-pthread", "-I" + runtimeDir, cFile, "-o", binFile}
	} else if compiler == "tcc" {
		modeDesc = "dev (TinyCC instant build)"
		cFlags = []string{"-pthread", "-I" + runtimeDir, cFile, "-o", binFile}
	} else {
		modeDesc = "dev (-O2 " + archFlag + ")"
		cFlags = []string{"-std=c99", "-O2", archFlag, "-mstackrealign", "-static", "-pthread", "-I" + runtimeDir, cFile, "-o", binFile}
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
