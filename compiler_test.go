package main_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/vexlang/vex/internal/autofree"
	"github.com/vexlang/vex/internal/cgen"
	"github.com/vexlang/vex/internal/checker"
	"github.com/vexlang/vex/internal/lexer"
	"github.com/vexlang/vex/internal/parser"
	"github.com/vexlang/vex/internal/source"
	"github.com/vexlang/vex/internal/token"
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

// --- 2. NEGATIVE SEMANTIC DIAGNOSTIC TESTS (`testdata/checker/*.vex`) ---

func TestNegativeSemanticDiagnostics(t *testing.T) {
	ensureTestdataSeeded(t)

	files, err := filepath.Glob(filepath.Join("testdata", "checker", "*.vex"))
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
	idxFirst := strings.Index(report, "vex_str_free(&first)")
	if idxSecond == -1 || idxFirst == -1 || idxSecond > idxFirst {
		t.Errorf("expected LIFO cleanup order (second freed before first), got report:\n%s", report)
	}
}

// --- 4. END-TO-END GOLDEN & ASAN/UBSAN EXECUTION TESTS (`testdata/run/*.vex`) ---

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
		t.Logf("ℹ Host C compiler (%s) running with standard C99 + _vex_active_heap_allocs leak check", cCompiler)
	}

	files, err := filepath.Glob(filepath.Join("testdata", "run", "*.vex"))
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

			// Verify zero runtime memory leaks via Vex telemetry counter
			if !strings.Contains(outStr, "[vex_rt] active_leaks=0") {
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
	tmpDir, err := os.MkdirTemp("", "vex_asan_probe_*")
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
	negFile := filepath.Join(chkDir, "01_mutability_and_results.vex")
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
	ifaceFile := filepath.Join(chkDir, "02_interface_satisfaction.vex")
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
	oopRun := filepath.Join(runDir, "01_oop_and_autofree.vex")
	if _, err := os.Stat(oopRun); os.IsNotExist(err) {
		_ = os.WriteFile(oopRun, []byte(`module billing

// OUT: Invoice #inv_9001 paid (15000 cents)
// OUT: Invoice #inv_9001 [updated_at=1760000000]
// OUT: [vex_rt] active_leaks=0

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
	aiActorRun := filepath.Join(runDir, "02_simd_tensors_and_actors.vex")
	if _, err := os.Stat(aiActorRun); os.IsNotExist(err) {
		_ = os.WriteFile(aiActorRun, []byte(`module ai

// OUT: Future #2 caught fault: CUDA/NaN anomaly on batch
// OUT: Parallel Future results -> w1=131072.00, w2(faulted)=0.00, w3=262144.00
// OUT: Worker #2 recovered Future result -> w2=196608.00
// OUT: [vex_rt] active_leaks=0

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
