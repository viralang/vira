# Chapter 12: Automated Testing, Diagnostics & Tooling

A systems toolchain is only as trustworthy as its verification story. Vira pairs deterministic memory management with first-class CLI tooling for unit testing (`vira test`), compile-time escape audits (`vira check`), hardware-backed sanitizer builds (`-sanitize`), inspectable C99 lowering (`vira cgen`, `-keep-c`), and an integrated Language Server (`vira lsp`).

> **Companion Runnable Code:**
> - Diagnostics target: [`docs/handbook/chapters/code/ch12_testing_and_diagnostics.vir`](../code/ch12_testing_and_diagnostics.vir)
> - Unit test suite: [`docs/handbook/chapters/code/ch12_math_test.vir`](../code/ch12_math_test.vir)

---

## 12.1. Writing Unit Tests (`*_test.vir`) and Running `vira test`

In Vira, unit tests live in `*_test.vir` files (commonly inside a project's `tests/` directory) or in any `.vir` file containing functions prefixed with `test_`:

```v
module main

import math

fn test_trig_and_hypot() {
	if math.PI < 3.14 || math.PI > 3.15 {
		panic('invalid PI constant')
	}
	hyp := math.hypot(3.0, 4.0)
	if hyp < 4.9999 || hyp > 5.0001 {
		panic('hypot(3, 4) must be 5')
	}
	println('✓ test_trig_and_hypot passed')
}
```

When you run `vira test <file_or_dir>`:
1. **Discovery:** If you pass a directory, `vira test` walks it for `*_test.vir` files and any `.vir` file containing `fn test_`.
2. **Test Runner Synthesis:** If a test file does not define its own `fn main()`, the compiler automatically synthesizes a `main()` runner that invokes every `test_*()` function in sequence. When testing inside a `vira.mod` project, any application `fn main()` in `src/` is skipped so your test suite runs cleanly without duplicate `main` symbols.
3. **Zero-Leak Enforcement (`active_leaks=0`):** Each test binary is compiled and executed. A test fails automatically if any `test_*` function panics (non-zero exit code) **or** if the runtime reports `active_leaks != 0` at exit.
4. **Diagnostic Override (`-allow-leaks`):** Passing `-allow-leaks` (`vira test -allow-leaks <target>`) reports non-zero leak counts as warnings instead of failing the run while you diagnose a leak.

---

## 12.2. Memory Sanitizer Verification (`-sanitize`) and Zero-Leak Assertions

Before relying on a binary in production, Vira gives you two complementary layers of memory verification:

### 1. Static Escape & Autofree Audit (`vira check`)

Running `vira check` validates syntax, 4-quadrant mutability, type safety, and prints the Autofree escape-analysis report showing every heap variable's lifetime classification and injected destructor:

```bash
./vira check docs/handbook/chapters/code/ch12_testing_and_diagnostics.vir
```

Output excerpt:
```text
✓ docs/handbook/chapters/code/ch12_testing_and_diagnostics.vir passed syntax, 4-quadrant mutability, and type checks (0 errors)

1. Structs Requiring Generated `_autofree` Destructors:
   ├── struct main.SampleBatch -> requires `main.SampleBatch_autofree(&self)` (owns heap fields)
   ├── struct SampleBatch  -> requires `SampleBatch_autofree(&self)` (owns heap fields)
...
   [fn main]
     ├── var b            (SampleBatch) : SCOPED_LOCAL (autofree injected)
     ├── var report       (string  ) : SCOPED_LOCAL (autofree injected)
     ├── var msg          (string  ) : ARENA_MANAGED (O(1) slab free)
     ├── var _vira_tmp_str_5 (string  ) : MOVED_TO_CONTAINER (ownership -> struct/iface)
     └── INJECTED FREE: vira_arena_destroy(&_vira_arena_1) [vira_arena] (arena block exit (O(1) slab deallocation))
     └── INJECTED FREE: vira_str_free(&_vira_tmp_str_5) [string] (zero-leak temporary string operand cleanup)
     └── INJECTED FREE: vira_str_free(&report) [string] (function scope exit)
     └── INJECTED FREE: struct_autofree(&b) [main.SampleBatch] (function scope exit)
```

### 2. AddressSanitizer & UndefinedBehaviorSanitizer (`vira build -sanitize`)

Passing `-sanitize` to `vira build` compiles the emitted C99 with `-std=c99 -O1 -g -fwrapv -fsanitize=address,undefined` (using GCC or Clang). Running the resulting binary verifies at the machine-instruction level that there are no buffer overflows, use-after-free bugs, misaligned accesses, or memory leaks:

```bash
./vira build -sanitize -o /tmp/ch12_san docs/handbook/chapters/code/ch12_testing_and_diagnostics.vir
/tmp/ch12_san
```

Output:
```text
✓ Built standalone binary /tmp/ch12_san in 1.245s (backend: gcc [sanitize (ASan+UBSan)])
latency_ms: n=3, mean=4.00, rms=4.08
arena_verified=true
[defer] finished diagnostics run
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=6
```

> **Suppressing Runtime Telemetry in Production:** Both `vira run` and `vira build` accept `-quiet-rt` (`-DVIRA_QUIET_RT=1`) when you want clean CLI stdout without the trailing `[vira_rt]` statistics lines.

---

## 12.3. Inspecting Generated C99 (`vira cgen`, `-keep-c`) and IDE Integration (`vira lsp`)

Because Vira lowers directly to readable, standard C99, you can audit every struct layout, SIMD loop, and destructor call without guessing what the compiler emitted.

### Emitting C99 Directly (`vira cgen` & `-keep-c`)

- **Print to stdout or write to a file:**
```bash
./vira cgen docs/handbook/chapters/code/ch12_testing_and_diagnostics.vir -o /tmp/ch12_out.c
```
- **Keep intermediate C99 alongside a native build:**
```bash
./vira build -keep-c -o /tmp/ch12_app docs/handbook/chapters/code/ch12_testing_and_diagnostics.vir
# Keeps build/ch12_testing_and_diagnostics.c on disk
```

Inspecting `main__summarize_batch` in the generated C99 confirms that Vira passes borrowed parameters cleanly and lowers method calls to direct static C function calls:

```c
vira_string main__summarize_batch(main__SampleBatch batch) {
    const double avg = main__SampleBatch_mean(&batch);
    const double rms_val = main__SampleBatch_rms(&batch);
```

### Official Style Formatting (`vira fmt`) and Language Server (`vira lsp`)

- **Canonical Formatting:** Run `vira fmt -w <file_or_dir>` to format Vira sources in place according to official tab-indentation and spacing rules (or omit `-w` to preview formatted code on stdout).
- **IDE Integration (`vira lsp`):** Starting `vira lsp` launches Vira's built-in JSON-RPC 2.0 Language Server over `stdin`/`stdout` (supporting full document sync, live `textDocument/didOpen`/`didChange` diagnostics, `textDocument/hover`, `textDocument/definition`, `textDocument/completion` triggered on `.` and `:`, `textDocument/documentSymbol`, and `textDocument/formatting`).

---

## 12.4. Running the Complete Chapter 12 Example

Run the diagnostics program [`docs/handbook/chapters/code/ch12_testing_and_diagnostics.vir`](../code/ch12_testing_and_diagnostics.vir):

```bash
./vira run docs/handbook/chapters/code/ch12_testing_and_diagnostics.vir
```

Output:
```text
latency_ms: n=3, mean=4.00, rms=4.08
arena_verified=true
[defer] finished diagnostics run
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=6
```

Run the unit test suite [`docs/handbook/chapters/code/ch12_math_test.vir`](../code/ch12_math_test.vir):

```bash
./vira test docs/handbook/chapters/code/ch12_math_test.vir
```

Output:
```text
=== RUN   docs/handbook/chapters/code/ch12_math_test.vir
--- PASS: docs/handbook/chapters/code/ch12_math_test.vir (0.77s)
    active_leaks=0 [Zero Memory Leaks Verified]

Test Summary: 1 passed, 0 failed in 0.77s
```

---

## Summary & Exercises

1. **Test Failure & Leak Detection:** Add a `test_intentional_fail()` function in a copy of `ch12_math_test.vir` that calls `panic('expected failure')`, run `vira test`, and observe how the runner formats the `--- FAIL:` summary and exits with status `1`.
2. **Production Optimization Diff:** Compare the flags and runtime speed of `vira build -prod -o /tmp/ch12_prod docs/handbook/chapters/code/ch12_testing_and_diagnostics.vir` against `vira build -sanitize -o /tmp/ch12_san ...`, then clean up the temporary binaries in `/tmp/`.
3. **C99 Audit:** Run `vira cgen docs/handbook/chapters/code/ch12_testing_and_diagnostics.vir` and locate `main__SampleBatch_autofree` and `vira_arena_destroy` in the emitted C99 `main()` function.
