# Tutorial 5: Deterministic Autofree & Bump Arenas

> **Difficulty:** Advanced  
> **Matching Code Example:** [`docs/examples/05_arenas_and_autofree.vir`](../examples/05_arenas_and_autofree.vir)

---

## 1. Overview

One of Vira's core breakthroughs is **deterministic automatic memory management without a garbage collector**.

Traditional approaches force developers to choose between:
- **Manual Management (C/C++):** Maximum speed, but high risk of use-after-free bugs, double frees, and memory leaks.
- **Traced Garbage Collection (Go/Java):** Safe, but introduces unpredictable stop-the-world pauses, heavy cache pollution, and high memory footprint.
- **Borrow Checking (Rust):** Zero-cost safety, but complex lifetime annotations (`'a`, `'b`) and steep cognitive friction.

Vira eliminates this dilemma with **Compiler-Driven Static Autofree & Bump Arenas**.

---

## 2. How Autofree Works

During compilation (Phase 3), the Vira **Static Escape Analyzer** evaluates the control-flow graph of every function:

1. **Ownership Detection:** Identifies any local variable that allocates heap memory (strings, tensors, boxed interfaces, or structs owning heap fields).
2. **Escape Analysis:** Analyzes if the variable escapes the current stack frame (e.g. through a `return` statement or by being stored inside another container).
3. **LIFO Destructor Injection:** If the variable does not escape, the compiler injects deterministic cleanup calls right before block and function exits in strict **Last-In, First-Out (LIFO)** order.

```v
fn simulate_standard_autofree() {
	r1 := ReportRecord{ tag: 'batch_alpha', value: 100 }
	r2 := ReportRecord{ tag: 'batch_beta', value: 200 }
	println('Created records: ${r1.tag}, ${r2.tag}')

	// The compiler automatically injects at scope exit:
	// vira_str_free(&r2.tag); // LIFO #1
	// vira_str_free(&r1.tag); // LIFO #2
}
```

---

## 3. Escape Analysis Classification States

Every tracked variable is placed into one of four states:

| State | Lifecycle Description | Cleanup Action |
| :--- | :--- | :--- |
| `StateScopedLocal` | Variable does not escape its lexical block | Injected with deterministic `free` at scope exit |
| `StateEscapesViaReturn` | Variable is returned to the calling function | Caller inherits ownership; current scope skips cleanup |
| `StateMovedToContainer` | Variable is boxed into an interface or struct | Outer container inherits ownership; skipped locally |
| `StateManagedByArena` | Allocated inside an `arena { ... }` block | Managed by the arena slab; individual frees skipped |

---

## 4. Auditing Autofree with `vira check`

You can inspect the compiler's escape decisions directly by running:

```powershell
.\vira.exe check docs/examples/05_arenas_and_autofree.vir
```

Sample audit report:
```text
✓ docs/examples/05_arenas_and_autofree.vir passed syntax and type checks (0 errors)

1. Structs Requiring Generated `_autofree` Destructors:
   └── struct ReportRecord -> requires `ReportRecord_autofree(&self)`

2. Function Escape Analysis & Injected LIFO Destructors:
   [fn simulate_standard_autofree]
     ├── var r1 (ReportRecord) : SCOPED_LOCAL (autofree injected)
     ├── var r2 (ReportRecord) : SCOPED_LOCAL (autofree injected)
     └── INJECTED FREE: ReportRecord_autofree(&r2) [ReportRecord] (function scope exit)
     └── INJECTED FREE: ReportRecord_autofree(&r1) [ReportRecord] (function scope exit)
```

---

## 5. Bump Arenas (`arena { ... }`)

For high-throughput loops, neural net forward passes, or temporary batch computations, individual heap allocations introduce fragmentation and lock contention.

Vira provides native **bump arenas**:

```v
arena {
	t1 := tensor_fill(32, 32, 1.5)
	t2 := tensor_fill(32, 32, 2.0)
	t3 := t1 @ t2
	println('Arena MatMul computed sum: ${tensor_sum(t3)}')
}
```

### Arena Characteristics

- **64-Byte Cache Alignment:** Every allocation inside the arena is aligned to CPU cache-line boundaries (AVX2/AVX-512 vector friendly).
- **O(1) Allocation:** Allocating memory simply increments an offset pointer.
- **O(1) Bulk Teardown:** Upon block exit, individual object destructors are bypassed, and the entire memory slab is reset in a single operation.

---

## 6. Running the Code

```powershell
.\vira.exe run docs/examples/05_arenas_and_autofree.vir
```

Expected output:
```text
=== 1. DETERMINISTIC LIFO AUTOFREE (NO GC) ===
--- Standard LIFO Scope-Exit Autofree ---
Created records: batch_alpha=100, batch_beta=200
=== 2. HIGH-PERFORMANCE BUMP ARENA ALLOCATION ===
--- 64-Byte Cache-Aligned Bump Arena ---
Arena MatMul computed sum: 98304.00
Both memory domains completed with zero memory leaks
[vira_rt] active_leaks=0 | total_freed=2
```

---

## Next Steps

Proceed to [**Tutorial 6: First-Class AI Tensors & SIMD Acceleration**](./06_simd_tensors_and_ai_math.md) to explore Vira's native matrix math capabilities.
