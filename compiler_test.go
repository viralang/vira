package main_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/autofree"
	"github.com/viralang/vira/internal/cgen"
	"github.com/viralang/vira/internal/checker"
	"github.com/viralang/vira/internal/lexer"
	"github.com/viralang/vira/internal/parser"
	"github.com/viralang/vira/internal/source"
	"github.com/viralang/vira/internal/token"
)

// --- 1. ZERO-ALLOCATION LEXER VERIFICATION & BENCHMARK ---

const benchSource = `
module billing

pub struct Invoice {
	AuditMeta
	secret_token string
pub:
	id string
	amount_cents i64
pub mut:
	is_paid bool
}

pub fn (mut inv Invoice) mark_paid(scale f32) !string {
	w := tensor_fill(64, 64, 0.25)
	x := tensor_fill(64, 64, scale)
	y := w @ x
	inv.is_paid = true
	return 'Invoice #${inv.id} paid: ${inv.amount_cents} cents (sum=${tensor_sum(y)})'
}
`

func TestZeroAllocLexer(t *testing.T) {
	allocs := testing.AllocsPerRun(100, func() {
		l := lexer.New(benchSource)
		for {
			tok := l.Next()
			if tok.Kind == token.TokEOF || tok.Kind == token.TokIllegal {
				break
			}
		}
	})
	if allocs > 0 {
		t.Fatalf("expected 0 heap allocations in lexer, got %.2f allocs/run", allocs)
	}
}

func BenchmarkLexerZeroAlloc(b *testing.B) {
	b.ReportAllocs()
	b.SetBytes(int64(len(benchSource)))
	for i := 0; i < b.N; i++ {
		l := lexer.New(benchSource)
		for {
			tok := l.Next()
			if tok.Kind == token.TokEOF || tok.Kind == token.TokIllegal {
				break
			}
		}
	}
}

// --- 2. NEGATIVE SEMANTIC DIAGNOSTIC TESTS (`testdata/checker/*.vir`) ---

func TestNegativeSemanticDiagnostics(t *testing.T) {
	ensureTestdataSeeded(t)

	files, err := filepath.Glob(filepath.Join("testdata", "checker", "*.vir"))
	if err != nil || len(files) == 0 {
		t.Fatalf("failed to discover checker test files: %v", err)
	}

	for _, file := range files {
		file := file
		t.Run(filepath.Base(file), func(t *testing.T) {
			rawBytes, err := os.ReadFile(file)
			if err != nil {
				t.Fatalf("read error: %v", err)
			}
			content := string(rawBytes)
			srcFile := source.New(file, content)

			expectedByLine := parseExpectedErrorComments(content)
			if len(expectedByLine) == 0 {
				t.Fatalf("%s has no '// ERROR: <msg>' annotations", file)
			}

			p := parser.New(content)
			astFile := p.ParseFile()
			if len(p.Errors()) > 0 {
				t.Fatalf("unexpected syntax errors in semantic test %s: %v", file, p.Errors())
			}

			chk := checker.New()
			chk.CheckFile(astFile)
			actualErrs := chk.Errors()

			matched := make(map[int]bool)

			for _, se := range actualErrs {
				line, _ := srcFile.Location(se.Span.Start)
				wantSubstr, exists := expectedByLine[line]
				if !exists {
					t.Errorf("unexpected semantic error at line %d:\n%s",
						line, srcFile.FormatDiagnostic("SEMANTIC", se.Msg, se.Span))
					continue
				}
				if !strings.Contains(se.Msg, wantSubstr) {
					t.Errorf("line %d error mismatch:\n  got:  %s\n  want substring: %s",
						line, se.Msg, wantSubstr)
				}
				matched[line] = true
			}

			for line, wantSubstr := range expectedByLine {
				if !matched[line] {
					t.Errorf("missing expected semantic error on line %d (wanted %q)", line, wantSubstr)
				}
			}
		})
	}
}

// --- 3. STATIC ESCAPE ANALYSIS & LIFO AUTOFREE INVARIANTS ---

func TestAutofreeEscapeAnalysis(t *testing.T) {
	src := `
module memtest

pub struct Packet {
pub:
	payload string
}

pub fn Packet.new(p string) Packet {
	pkt := Packet{ payload: p }
	return pkt
}

pub fn might_fail(flag bool) !string {
	if flag {
		return error('failed')
	}
	return 'ok'
}

fn main() {
	first := 'first_alloc'
	second := Packet.new('second_alloc')
	res := might_fail(true) or {
		return
	}
	println(first)
	println(res)
	_ := second
}
`
	p := parser.New(src)
	astFile := p.ParseFile()
	if len(p.Errors()) > 0 {
		t.Fatalf("parse errors: %v", p.Errors())
	}

	chk := checker.New()
	chk.CheckFile(astFile)
	if len(chk.Errors()) > 0 {
		t.Fatalf("semantic errors: %v", chk.Errors())
	}

	af := autofree.New(chk)
	af.AnalyzeFile(astFile)

	if !af.StructNeedsFree["Packet"] {
		t.Fatalf("expected struct 'Packet' (owning string) to require an _autofree destructor")
	}

	report := af.FormatReport()
	// 1. Verify return escape in `Packet.new`
	if !strings.Contains(report, "ESCAPES_VIA_RETURN") {
		t.Errorf("expected 'pkt' in Packet.new to be marked ESCAPES_VIA_RETURN, got:\n%s", report)
	}
	// 2. Verify LIFO order: `second` must be freed before `first` inside `or { return }`
	idxSecond := strings.Index(report, "struct_autofree(&second)")
	idxFirst := strings.Index(report, "vira_str_free(&first)")
	if idxSecond == -1 || idxFirst == -1 || idxSecond > idxFirst {
		t.Errorf("expected LIFO cleanup order (second freed before first), got report:\n%s", report)
	}
}

func TestZeroLeakTemporaryStringAutofree(t *testing.T) {
	src := `
module temp_str_test

pub struct Logger {
pub:
	prefix string
}

pub fn (l Logger) log(msg string) {
	println('[${l.prefix}] ${msg}')
}

fn emit_event(source string, payload string) {
	println('event: ${source} -> ${payload}')
}

fn main() {
	l := Logger{ prefix: 'Telemetry' }
	
	id := 42
	temp := 98.6
	emit_event('sensor_${id}', 'temperature_${temp}')

	status := 'ONLINE'
	l.log('Service status is ${status}')

	println('Done testing temporary strings: ${id}')
}
`
	p := parser.New(src)
	astFile := p.ParseFile()
	if len(p.Errors()) > 0 {
		t.Fatalf("parse errors: %v", p.Errors())
	}

	chk := checker.New()
	chk.CheckFile(astFile)
	if len(chk.Errors()) > 0 {
		t.Fatalf("semantic errors: %v", chk.Errors())
	}

	af := autofree.New(chk)
	af.AnalyzeFile(astFile)

	gen := cgen.New(chk, af)
	cCode := gen.EmitFile(astFile)

	// Verify generated C contains injected frees for temporary strings
	if !strings.Contains(cCode, "vira_str_free(&_vira_tmp_str_") {
		t.Fatalf("expected generated C code to contain vira_str_free(&_vira_tmp_str_...), got:\n%s", cCode)
	}

	// Discover host C compiler
	var cCompiler string
	for _, cand := range []string{"cc", "gcc", "clang", "tcc"} {
		if _, err := exec.LookPath(cand); err == nil {
			cCompiler = cand
			break
		}
	}
	if cCompiler == "" {
		if home, err := os.UserHomeDir(); err == nil {
			cand := filepath.Join(home, "w64devkit", "bin", "gcc.exe")
			if info, err := os.Stat(cand); err == nil && !info.IsDir() {
				cCompiler = cand
			}
		}
	}
	if cCompiler == "" {
		t.Skip("C compiler not available, skipping native run")
		return
	}

	runtimeDir, err := filepath.Abs("runtime")
	if err != nil {
		t.Fatalf("failed to resolve runtime dir: %v", err)
	}

	tmpDir := t.TempDir()
	cFile := filepath.Join(tmpDir, "out.c")
	binFile := filepath.Join(tmpDir, "app")
	if os.PathSeparator == '\\' {
		binFile += ".exe"
	}
	if err := os.WriteFile(cFile, []byte(cCode), 0644); err != nil {
		t.Fatalf("write C file: %v", err)
	}

	cFlags := []string{"-std=c99", "-O2", "-march=native", "-pthread", "-I" + runtimeDir, cFile, "-o", binFile}
	buildCmd := exec.Command(cCompiler, cFlags...)
	buildCmd.Env = append(os.Environ(), "PATH="+filepath.Dir(cCompiler)+string(os.PathListSeparator)+os.Getenv("PATH"))
	if buildOut, err := buildCmd.CombinedOutput(); err != nil {
		t.Fatalf("C99 compilation failed (%v):\n%s\nGenerated C99:\n%s", err, string(buildOut), cCode)
	}

	runCmd := exec.Command(binFile)
	runCmd.Env = append(os.Environ(), "ASAN_OPTIONS=detect_leaks=0")
	rawOut, err := runCmd.CombinedOutput()
	outStr := string(rawOut)
	if err != nil {
		t.Fatalf("binary execution failed (%v):\n%s", err, outStr)
	}

	if !strings.Contains(outStr, "[vira_rt] active_leaks=0") {
		t.Fatalf("expected 0 active leaks, got output:\n%s", outStr)
	}
}

// --- 4. END-TO-END GOLDEN & ASAN/UBSAN EXECUTION TESTS (`testdata/run/*.vir`) ---

func TestEndToEndGoldenAndSanitizers(t *testing.T) {
	ensureTestdataSeeded(t)

	// Discover host C compiler
	var cCompiler string
	for _, cand := range []string{"cc", "gcc", "clang", "tcc"} {
		if _, err := exec.LookPath(cand); err == nil {
			cCompiler = cand
			break
		}
	}
	if cCompiler == "" {
		if home, err := os.UserHomeDir(); err == nil {
			cand := filepath.Join(home, "w64devkit", "bin", "gcc.exe")
			if info, err := os.Stat(cand); err == nil && !info.IsDir() {
				cCompiler = cand
			}
		}
	}
	if cCompiler == "" {
		t.Skip("Host C compiler (cc/gcc/clang/tcc) not found in PATH; skipping native binary execution test")
		return
	}

	useSanitizer := probeSanitizerSupport(cCompiler)
	if useSanitizer {
		t.Logf("✓ Host C compiler (%s) supports -fsanitize=address,undefined (ASan + UBSan enabled)", cCompiler)
	} else {
		t.Logf("ℹ Host C compiler (%s) running with standard C99 + _vira_active_heap_allocs leak check", cCompiler)
	}

	files, err := filepath.Glob(filepath.Join("testdata", "run", "*.vir"))
	if err != nil || len(files) == 0 {
		t.Fatalf("failed to discover golden run tests: %v", err)
	}

	runtimeDir, err := filepath.Abs("runtime")
	if err != nil {
		t.Fatalf("failed to resolve runtime dir: %v", err)
	}

	for _, file := range files {
		file := file
		t.Run(filepath.Base(file), func(t *testing.T) {
			t.Parallel()

			rawBytes, err := os.ReadFile(file)
			if err != nil {
				t.Fatalf("read error: %v", err)
			}
			content := string(rawBytes)
			expectedOut := parseExpectedOutputComments(content)

			// 1. Frontend & Middleend
			p := parser.New(content)
			astFile := p.ParseFile()
			if len(p.Errors()) > 0 {
				t.Fatalf("parse errors in %s: %v", file, p.Errors())
			}

			chk := checker.New()
			chk.CheckFile(astFile)
			if len(chk.Errors()) > 0 {
				t.Fatalf("semantic errors in %s: %v", file, chk.Errors())
			}

			af := autofree.New(chk)
			af.AnalyzeFile(astFile)

			// 2. C99 Code Generation
			gen := cgen.New(chk, af)
			cCode := gen.EmitFile(astFile)

			tmpDir := t.TempDir()
			cFile := filepath.Join(tmpDir, "out.c")
			binFile := filepath.Join(tmpDir, "app")
			if os.PathSeparator == '\\' {
				binFile += ".exe"
			}
			if err := os.WriteFile(cFile, []byte(cCode), 0644); err != nil {
				t.Fatalf("write C file: %v", err)
			}

			// 3. Compile with C99 + pthreads (+ ASan/UBSan if supported)
			cFlags := []string{"-std=c99", "-O2", "-march=native", "-pthread", "-I" + runtimeDir, cFile, "-o", binFile}
			if useSanitizer {
				cFlags = []string{
					"-std=c99", "-O1", "-g", "-fno-omit-frame-pointer",
					"-fsanitize=address,undefined", "-pthread",
					"-I" + runtimeDir, cFile, "-o", binFile,
				}
			}

			buildCmd := exec.Command(cCompiler, cFlags...)
			buildCmd.Env = append(os.Environ(), "PATH="+filepath.Dir(cCompiler)+string(os.PathListSeparator)+os.Getenv("PATH"))
			if buildOut, err := buildCmd.CombinedOutput(); err != nil {
				t.Fatalf("C99 compilation failed (%v):\n%s\nGenerated C99:\n%s", err, string(buildOut), cCode)
			}

			// 4. Execute Binary & Verify Golden Output + 0 Leaked Bytes
			runCmd := exec.Command(binFile)
			runCmd.Env = append(os.Environ(), "ASAN_OPTIONS=detect_leaks=0")
			rawOut, err := runCmd.CombinedOutput()
			outStr := string(rawOut)
			if err != nil {
				t.Fatalf("binary execution failed (%v):\n%s", err, outStr)
			}

			// Verify zero runtime memory leaks via Vira telemetry counter
			if !strings.Contains(outStr, "[vira_rt] active_leaks=0") {
				t.Errorf("memory leak detected in %s!\nFull output:\n%s", file, outStr)
			}

			// Verify every `// OUT:` line appears
			for _, wantLine := range expectedOut {
				if !strings.Contains(outStr, wantLine) {
					t.Errorf("missing expected output line in %s:\n  want: %q\n  got:\n%s",
						file, wantLine, outStr)
				}
			}
		})
	}
}

// --- TEST DIRECTIVE PARSERS & SEEDING HELPERS ---

var errDirectiveRe = regexp.MustCompile(`//\s*ERROR:\s*(.+)$`)
var outDirectiveRe = regexp.MustCompile(`//\s*OUT:\s*(.+)$`)

func parseExpectedErrorComments(src string) map[int]string {
	res := make(map[int]string)
	lines := strings.Split(src, "\n")
	for i, line := range lines {
		if m := errDirectiveRe.FindStringSubmatch(line); len(m) == 2 {
			res[i+1] = strings.TrimSpace(m[1])
		}
	}
	return res
}

func parseExpectedOutputComments(src string) []string {
	var out []string
	lines := strings.Split(src, "\n")
	for _, line := range lines {
		if m := outDirectiveRe.FindStringSubmatch(line); len(m) == 2 {
			out = append(out, strings.TrimSpace(m[1]))
		}
	}
	return out
}

func probeSanitizerSupport(compiler string) bool {
	tmpDir, err := os.MkdirTemp("", "vira_asan_probe_*")
	if err != nil {
		return false
	}
	defer os.RemoveAll(tmpDir)

	probeC := filepath.Join(tmpDir, "probe.c")
	probeBin := filepath.Join(tmpDir, "probe")
	_ = os.WriteFile(probeC, []byte("int main(void) { return 0; }\n"), 0644)

	cmd := exec.Command(compiler, "-fsanitize=address,undefined", probeC, "-o", probeBin)
	cmd.Env = append(os.Environ(), "PATH="+filepath.Dir(compiler)+string(os.PathListSeparator)+os.Getenv("PATH"))
	if err := cmd.Run(); err != nil {
		return false
	}
	return exec.Command(probeBin).Run() == nil
}

func ensureTestdataSeeded(t *testing.T) {
	t.Helper()
	chkDir := filepath.Join("testdata", "checker")
	runDir := filepath.Join("testdata", "run")
	_ = os.MkdirAll(chkDir, 0755)
	_ = os.MkdirAll(runDir, 0755)

	// 1. Seed Negative Checker Test: Immutability, 4-Quadrant Access, and Unhandled !T
	negFile := filepath.Join(chkDir, "01_mutability_and_results.vir")
	if _, err := os.Stat(negFile); os.IsNotExist(err) {
		_ = os.WriteFile(negFile, []byte(`module billing

pub struct Account {
	secret_pin int
pub:
	id string
pub mut:
	balance i64
}

pub fn (mut a Account) deposit(amount i64) !i64 {
	a.balance += amount
	return a.balance
}

fn main() {
	x := 10
	x = 20 // ERROR: cannot reassign immutable variable 'x'

	imm_acc := Account{ secret_pin: 1234, id: 'acc_1', balance: 100 }
	_ := imm_acc.deposit(50) or { 0 } // ERROR: cannot call mutable receiver method 'deposit()' on immutable binding 'imm_acc'

	mut mut_acc := Account{ secret_pin: 1234, id: 'acc_2', balance: 200 }
	mut_acc.id = 'hacked' // ERROR: field 'Account.id' is 'pub:' (read-only) and cannot be mutated
	mut_acc.secret_pin = 9999 // ERROR: field 'Account.secret_pin' is private and immutable

	bad_res := mut_acc.deposit(500) // ERROR: unhandled Result '!i64' from 'Account.deposit()'
	_ := bad_res
}
`), 0644)
	}

	// 2. Seed Negative Checker Test: Structural Interface Satisfaction
	ifaceFile := filepath.Join(chkDir, "02_interface_satisfaction.vir")
	if _, err := os.Stat(ifaceFile); os.IsNotExist(err) {
		_ = os.WriteFile(ifaceFile, []byte(`module main

pub interface Drawable {
	id string
	render() string
}

pub struct BrokenWidget {
pub:
	id string
}

fn main() {
	w := BrokenWidget{ id: 'w1' }
	d := Drawable(w) // ERROR: struct 'BrokenWidget' does not implement interface 'Drawable': missing method 'render'
	_ := d
}
`), 0644)
	}

	// 3. Seed End-to-End Golden Test 1: OOP Struct Embedding, Interfaces & Autofree
	oopRun := filepath.Join(runDir, "01_oop_and_autofree.vir")
	if _, err := os.Stat(oopRun); os.IsNotExist(err) {
		_ = os.WriteFile(oopRun, []byte(`module billing

// OUT: Invoice #inv_9001 paid (15000 cents)
// OUT: Invoice #inv_9001 [updated_at=1760000000]
// OUT: [vira_rt] active_leaks=0

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
		return error('invalid amount')
	}
	inv.is_paid = true
	inv.retry_count += 1
	inv.touch()
	return 'Invoice #${inv.id} paid (${inv.amount_cents} cents)'
}

fn main() {
	mut inv := Invoice.new('inv_9001', 15000)
	msg := inv.mark_paid() or {
		return
	}
	println(msg)
	println(inv.summary())
	p := Payable(inv)
	_ := p
}
`), 0644)
	}

	// 4. Seed End-to-End Golden Test 2: SIMD Tensors (@) + Parallel Future[T] Actors
	aiActorRun := filepath.Join(runDir, "02_simd_tensors_and_actors.vir")
	if _, err := os.Stat(aiActorRun); os.IsNotExist(err) {
		_ = os.WriteFile(aiActorRun, []byte(`module ai

// OUT: Future #2 caught fault: CUDA/NaN anomaly on batch
// OUT: Parallel Future results -> w1=131072.00, w2(faulted)=0.00, w3=262144.00
// OUT: Worker #2 recovered Future result -> w2=196608.00
// OUT: [vira_rt] active_leaks=0

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
		panic('CUDA/NaN anomaly on batch')
	}
	return tensor_sum(out)
}

fn main() {
	mut w1 := AIWorker{ id: 1, batches_run: 0 }
	mut w2 := AIWorker{ id: 2, batches_run: 0 }
	mut w3 := AIWorker{ id: 3, batches_run: 0 }

	fut1 := w1 <- predict(2.0, false)
	fut2 := w2 <- predict(3.0, true)
	fut3 := w3 <- predict(4.0, false)

	s1 := fut1.await() or { 0.0 }
	s2 := fut2.await() or {
		println('Future #2 caught fault: ${err}')
		0.0
	}
	s3 := fut3.await() or { 0.0 }

	println('Parallel Future results -> w1=${s1}, w2(faulted)=${s2}, w3=${s3}')

	fut2_retry := w2 <- predict(3.0, false)
	s2_recovered := fut2_retry.await() or { 0.0 }
	println('Worker #2 recovered Future result -> w2=${s2_recovered}')
}
`), 0644)
	}
}

func TestMultiFileCompilation(t *testing.T) {
	file1Src := `module main

pub struct Vector3 {
pub:
	x f32
	y f32
	z f32
}

pub fn Vector3.new(x f32, y f32, z f32) Vector3 {
	return Vector3{ x: x, y: y, z: z }
}

pub fn (v Vector3) dot(other Vector3) f32 {
	return v.x * other.x + v.y * other.y + v.z * other.z
}
`
	file2Src := `module main

fn main() {
	v1 := Vector3.new(1.0, 2.0, 3.0)
	v2 := Vector3.new(4.0, 5.0, 6.0)
	d := v1.dot(v2)
	println('dot=${d}')
}
`
	p1 := parser.New(file1Src)
	ast1 := p1.ParseFile()
	if len(p1.Errors()) > 0 {
		t.Fatalf("p1 errors: %v", p1.Errors())
	}

	p2 := parser.New(file2Src)
	ast2 := p2.ParseFile()
	if len(p2.Errors()) > 0 {
		t.Fatalf("p2 errors: %v", p2.Errors())
	}

	merged := &ast.File{
		Module: ast2.Module,
		Decls:  append(ast1.Decls, ast2.Decls...),
	}

	chk := checker.New()
	chk.CheckFile(merged)
	if len(chk.Errors()) > 0 {
		t.Fatalf("checker errors: %v", chk.Errors())
	}

	af := autofree.New(chk)
	af.AnalyzeFile(merged)

	gen := cgen.New(chk, af)
	cCode := gen.EmitFile(merged)

	if !strings.Contains(cCode, "Vector3_dot") {
		t.Errorf("expected Vector3_dot in emitted C code")
	}
	if !strings.Contains(cCode, "int main(") {
		t.Errorf("expected int main( in emitted C code")
	}
}

func TestNativeHardwareGUI(t *testing.T) {
	guiSrc := `
module main

fn main() {
	gui_init(640, 480, 'Test GUI')
	gui_set_target_fps(60)

	mut val := 10.0f32
	mut active := true

	if !gui_should_close() {
		gui_begin()
		gui_clear(0x181825)
		gui_draw_text('Hello Vira GUI', 20, 20, 18, 0xFFFFFF)
		gui_draw_rect(10, 10, 100, 50, 0x334455)
		gui_draw_circle(200, 200, 25.0f32, 0x8899AA)
		gui_draw_line(0, 0, 100, 100, 0xFF0000)

		_ := gui_button(20, 100, 120, 30, 'Click Me')
		val = gui_slider(20, 150, 200, 20, 'Min', 'Max', val, 0.0f32, 100.0f32)
		active = gui_checkbox(20, 190, 150, 25, 'Option', active)

		mx := gui_mouse_x()
		my := gui_mouse_y()
		_ := mx + my

		gui_end()
	}

	gui_close()
}
`
	p := parser.New(guiSrc)
	astFile := p.ParseFile()
	if len(p.Errors()) > 0 {
		t.Fatalf("parser errors in GUI test: %v", p.Errors())
	}

	chk := checker.New()
	chk.CheckFile(astFile)
	if len(chk.Errors()) > 0 {
		t.Fatalf("checker errors in GUI test: %v", chk.Errors())
	}

	af := autofree.New(chk)
	af.AnalyzeFile(astFile)

	gen := cgen.New(chk, af)
	cCode := gen.EmitFile(astFile)

	for _, expected := range []string{
		"vira_gui_init",
		"vira_gui_set_target_fps",
		"vira_gui_should_close",
		"vira_gui_begin",
		"vira_gui_clear",
		"vira_gui_draw_text",
		"vira_gui_draw_rect",
		"vira_gui_draw_circle",
		"vira_gui_draw_line",
		"vira_gui_button",
		"vira_gui_slider",
		"vira_gui_checkbox",
		"vira_gui_mouse_x",
		"vira_gui_mouse_y",
		"vira_gui_end",
		"vira_gui_close",
	} {
		if !strings.Contains(cCode, expected) {
			t.Errorf("expected %s in emitted C code", expected)
		}
	}
}

// --- 11. IMPORT PARSING & DYNAMIC FILE RESOLUTION TESTS ---

func TestImportParsing(t *testing.T) {
	src := `
module test_imports

import (
	math_util
	"services/compute.vir"
	'config/settings.vir'
)

import helpers

fn main() {
	println('ok')
}
`
	p := parser.New(src)
	f := p.ParseFile()
	if len(p.Errors()) > 0 {
		t.Fatalf("unexpected parser errors: %v", p.Errors())
	}

	expectedImports := []string{
		"math_util",
		"services/compute.vir",
		"config/settings.vir",
		"helpers",
	}

	if len(f.Imports) != len(expectedImports) {
		t.Fatalf("expected %d imports, got %d", len(expectedImports), len(f.Imports))
	}

	for i, want := range expectedImports {
		if f.Imports[i].Name != want {
			t.Errorf("import[%d] mismatch: got %q, want %q", i, f.Imports[i].Name, want)
		}
	}
}

func TestNamespaceQualifiedTypeAndImportAliasParsing(t *testing.T) {
	src := `
module test_app

import net as mynet
import http
import (
	io as vira_io
)

pub struct ServerConfig {
pub:
	stream mynet.TcpStream
	headers []http.Header
}

pub fn handle(mut s mynet.TcpStream) !http.Response {
	hdr := http.Header{ name: 'X-Vira', value: '1.0' }
	return http.Response.ok()
}
`
	p := parser.New(src)
	f := p.ParseFile()
	if len(p.Errors()) > 0 {
		t.Fatalf("unexpected parser errors: %v", p.Errors())
	}

	// Verify imports
	if len(f.Imports) != 3 {
		t.Fatalf("expected 3 imports, got %d", len(f.Imports))
	}
	if f.Imports[0].Path != "net" || f.Imports[0].Alias != "mynet" || f.Imports[0].Name != "mynet" {
		t.Errorf("import[0] alias mismatch: %+v", f.Imports[0])
	}
	if f.Imports[1].Path != "http" || f.Imports[1].Alias != "" || f.Imports[1].Name != "http" {
		t.Errorf("import[1] mismatch: %+v", f.Imports[1])
	}
	if f.Imports[2].Path != "io" || f.Imports[2].Alias != "vira_io" || f.Imports[2].Name != "vira_io" {
		t.Errorf("import[2] alias mismatch: %+v", f.Imports[2])
	}

	// Verify struct field qualified types
	stDecl, ok := f.Decls[0].(*ast.StructDecl)
	if !ok {
		t.Fatalf("expected StructDecl, got %T", f.Decls[0])
	}
	selType, ok := stDecl.Fields[0].Type.(*ast.SelectorType)
	if !ok || selType.Pkg.Name != "mynet" || selType.Name.Name != "TcpStream" {
		t.Errorf("expected SelectorType mynet.TcpStream, got %#v", stDecl.Fields[0].Type)
	}

	arrType, ok := stDecl.Fields[1].Type.(*ast.ArrayType)
	if !ok {
		t.Fatalf("expected ArrayType, got %#v", stDecl.Fields[1].Type)
	}
	elemSelType, ok := arrType.Elem.(*ast.SelectorType)
	if !ok || elemSelType.Pkg.Name != "http" || elemSelType.Name.Name != "Header" {
		t.Errorf("expected slice elem http.Header, got %#v", arrType.Elem)
	}

	// Verify function qualified parameter and return types
	fnDecl, ok := f.Decls[1].(*ast.FnDecl)
	if !ok {
		t.Fatalf("expected FnDecl, got %T", f.Decls[1])
	}
	paramSel, ok := fnDecl.Params[0].Type.(*ast.SelectorType)
	if !ok || paramSel.Pkg.Name != "mynet" || paramSel.Name.Name != "TcpStream" {
		t.Errorf("expected param type mynet.TcpStream, got %#v", fnDecl.Params[0].Type)
	}
	resType, ok := fnDecl.ReturnType.(*ast.ResultType)
	if !ok {
		t.Fatalf("expected ResultType, got %#v", fnDecl.ReturnType)
	}
	retSel, ok := resType.Inner.(*ast.SelectorType)
	if !ok || retSel.Pkg.Name != "http" || retSel.Name.Name != "Response" {
		t.Errorf("expected return type !http.Response, got %#v", resType.Inner)
	}

	// Verify qualified struct literal initialization: http.Header{ ... }
	varDecl, ok := fnDecl.Body.Stmts[0].(*ast.VarDeclStmt)
	if !ok {
		t.Fatalf("expected VarDeclStmt, got %T", fnDecl.Body.Stmts[0])
	}
	structInit, ok := varDecl.Value.(*ast.StructInitExpr)
	if !ok {
		t.Fatalf("expected StructInitExpr, got %T", varDecl.Value)
	}
	initType, ok := structInit.Type.(*ast.SelectorType)
	if !ok || initType.Pkg.Name != "http" || initType.Name.Name != "Header" {
		t.Errorf("expected struct init type http.Header, got %#v", structInit.Type)
	}
}

func TestDynamicImportAndCircularResolution(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "vira_import_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	subDir := filepath.Join(tmpDir, "utils")
	if err := os.MkdirAll(subDir, 0755); err != nil {
		t.Fatalf("failed to create sub dir: %v", err)
	}

	// Helper file that also imports main.vir (circular import verification)
	mathFile := filepath.Join(subDir, "math.vir")
	mathSrc := `
module calc_app

import "../main.vir"

pub fn calc_sum(a int, b int) int {
	return a + b
}
`
	if err := os.WriteFile(mathFile, []byte(mathSrc), 0644); err != nil {
		t.Fatalf("failed to write math.vir: %v", err)
	}

	// Main file importing utils/math.vir
	mainFile := filepath.Join(tmpDir, "main.vir")
	mainSrc := `
module calc_app

import "utils/math.vir"

fn main() {
	ans := calc_sum(20, 22)
	println('Computed Sum: ${ans}')
}
`
	if err := os.WriteFile(mainFile, []byte(mainSrc), 0644); err != nil {
		t.Fatalf("failed to write main.vir: %v", err)
	}

	binName := "vira"
	if runtime.GOOS == "windows" {
		binName = "vira.exe"
	}
	viraBin, err := filepath.Abs(binName)
	if err != nil {
		t.Fatalf("failed to get absolute path to vira: %v", err)
	}

	cmd := exec.Command(viraBin, "run", mainFile)
	cmd.Dir = tmpDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run dynamically imported program: %v\nOutput: %s", err, string(out))
	}

	outStr := string(out)
	if !strings.Contains(outStr, "Computed Sum: 42") {
		t.Errorf("expected 'Computed Sum: 42' in output, got:\n%s", outStr)
	}
}

func TestCheckerNamespacesAndDisambiguation(t *testing.T) {
	netSrc := `
module net

pub struct TcpStream {
pub:
	fd int
}

pub fn TcpStream.connect(addr string) !TcpStream {
	return TcpStream{ fd: 10 }
}

struct SecretInternals {
	key string
}
`
	httpSrc := `
module http

import net

pub struct Request {
pub:
	method string
	path   string
	stream net.TcpStream
}

pub fn Request.new(m string, p string, s net.TcpStream) Request {
	return Request{ method: m, path: p, stream: s }
}
`
	rpcSrc := `
module rpc

pub struct Request {
pub:
	id int
	service string
}
`
	appSrc := `
module app

import (
	net
	http
	rpc as myrpc
)

fn main() {
	stream := net.TcpStream.connect('127.0.0.1:8080') or { panic(err) }
	hReq := http.Request.new('GET', '/index', stream)
	rReq := myrpc.Request{ id: 1, service: 'auth' }
	println(hReq.method)
	println(rReq.service)
}
`

	pNet := parser.New(netSrc)
	fNet := pNet.ParseFile()
	if len(pNet.Errors()) > 0 {
		t.Fatalf("parser errors in net: %v", pNet.Errors())
	}

	pHttp := parser.New(httpSrc)
	fHttp := pHttp.ParseFile()
	if len(pHttp.Errors()) > 0 {
		t.Fatalf("parser errors in http: %v", pHttp.Errors())
	}

	pRpc := parser.New(rpcSrc)
	fRpc := pRpc.ParseFile()
	if len(pRpc.Errors()) > 0 {
		t.Fatalf("parser errors in rpc: %v", pRpc.Errors())
	}

	pApp := parser.New(appSrc)
	fApp := pApp.ParseFile()
	if len(pApp.Errors()) > 0 {
		t.Fatalf("parser errors in app: %v", pApp.Errors())
	}

	chk := checker.New()
	chk.CheckFiles([]*ast.File{fNet, fHttp, fRpc, fApp})

	if len(chk.Errors()) > 0 {
		for _, err := range chk.Errors() {
			t.Errorf("unexpected semantic error: %s", err.Msg)
		}
		t.FailNow()
	}

	// Verify both http.Request and rpc.Request exist without collision
	httpReq := chk.LookupStructInModule("http", "Request")
	rpcReq := chk.LookupStructInModule("rpc", "Request")
	if httpReq == nil || rpcReq == nil {
		t.Fatalf("expected both http.Request and rpc.Request to exist")
	}
	if httpReq == rpcReq {
		t.Fatalf("http.Request and rpc.Request should be distinct structs")
	}
	if len(httpReq.Fields) != 3 || len(rpcReq.Fields) != 2 {
		t.Errorf("struct field mismatch: http=%d, rpc=%d", len(httpReq.Fields), len(rpcReq.Fields))
	}

	// Verify private type access is rejected across modules
	badSrc := `
module bad_app
import net

fn test() {
	s := net.SecretInternals{ key: 'secret' }
}
`
	pBad := parser.New(badSrc)
	fBad := pBad.ParseFile()
	chkBad := checker.New()
	chkBad.CheckFiles([]*ast.File{fNet, fBad})
	if len(chkBad.Errors()) == 0 {
		t.Fatalf("expected semantic error for private struct access, got none")
	}
	foundPrivErr := false
	for _, e := range chkBad.Errors() {
		if strings.Contains(e.Msg, "private to module") {
			foundPrivErr = true
			break
		}
	}
	if !foundPrivErr {
		t.Errorf("expected 'private to module' error, got: %v", chkBad.Errors())
	}

	// Verify unqualified external access suggests qualification
	unqualSrc := `
module unqual_app
import net

fn test() {
	s := TcpStream{ fd: 1 }
}
`
	pUnqual := parser.New(unqualSrc)
	fUnqual := pUnqual.ParseFile()
	chkUnqual := checker.New()
	chkUnqual.CheckFiles([]*ast.File{fNet, fUnqual})
	if len(chkUnqual.Errors()) == 0 {
		t.Fatalf("expected semantic error for unqualified type, got none")
	}
	foundSuggestErr := false
	for _, e := range chkUnqual.Errors() {
		if strings.Contains(e.Msg, "did you mean 'net.TcpStream'?") {
			foundSuggestErr = true
			break
		}
	}
	if !foundSuggestErr {
		t.Errorf("expected 'did you mean net.TcpStream' error, got: %v", chkUnqual.Errors())
	}
}

func TestSwitchStatement(t *testing.T) {
	src := `
module switch_test

fn test_sw(x int, s string) int {
	mut res := 0
	switch x {
	case 1:
		res = 10
	case 2, 3:
		res = 20
	default:
		res = 99
	}

	switch s {
	case "GET", "HEAD":
		res += 1
	default:
		res += 2
	}

	switch {
	case res > 10:
		res += 100
	default:
		res += 50
	}

	return res
}
`
	p := parser.New(src)
	f := p.ParseFile()
	if len(p.Errors()) > 0 {
		t.Fatalf("parser errors in switch test: %v", p.Errors())
	}

	chk := checker.New()
	chk.CheckFiles([]*ast.File{f})
	if len(chk.Errors()) > 0 {
		t.Fatalf("checker errors in switch test: %v", chk.Errors())
	}

	af := autofree.New(chk)
	af.AnalyzeFile(f)

	gen := cgen.New(chk, af)
	cCode := gen.EmitFile(f)
	if !strings.Contains(cCode, "switch (x)") {
		t.Errorf("expected native switch (x) in C code, got:\n%s", cCode)
	}
	if !strings.Contains(cCode, "vira_str_eq") {
		t.Errorf("expected vira_str_eq in C code for string switch, got:\n%s", cCode)
	}
}


