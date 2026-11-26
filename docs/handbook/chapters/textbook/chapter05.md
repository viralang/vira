# Chapter 5: Understanding Ownership, Autofree & Bump Arenas

Ownership is Vira's defining systems guarantee: every heap-allocated value—strings, slices, maps, tensors, closures, and structs containing them—is released deterministically when its owner leaves scope, with **no stop-the-world garbage collector** and **no explicit lifetime annotations**. In this chapter, we examine how Vira enforces single-owner value semantics, turns last-use stores into zero-copy moves, borrows arguments across calls, classifies lifetimes in static escape analysis, and accelerates batch workloads with `O(1)` bump arenas.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch05_ownership_and_arenas.vir`](../code/ch05_ownership_and_arenas.vir)

---

## 5.1. Memory Safety Without a Garbage Collector or Borrow-Checker Annotations

Systems languages typically force a trade-off between three memory models:
- **Manual `malloc`/`free` (C/C++):** Maximum control, but vulnerable to use-after-free, double-free, and leaked error paths.
- **Tracing Garbage Collection (Go/Java):** Safe, but requires runtime heap scanning, GC write barriers, and higher memory headroom.
- **Lifetime-Annotated Borrow Checking (Rust):** Zero-cost at runtime, but requires explicit `'a` lifetime parameters and fighting reference aliasing rules.

Vira takes a fourth path: **Single-Owner Value Semantics + Compile-Time Autofree + Optional Bump Arenas**.

1. **Every heap value has exactly one owner:** A local variable, a struct field, a slice/map element, or an actor.
2. **No shared mutable aliasing:** Two normal variables never share ownership of the same mutable heap buffer.
3. **Compile-time destructor injection:** Phase 3 of the Vira compiler (`autofree`) traces every lexical scope—including early `return`, `break`, and `continue` jumps—and emits C99 cleanup calls in strict **Last-In, First-Out (LIFO)** order.

---

## 5.2. The Single-Owner Rule: Automatic Deep Copies vs. Last-Use Moves

When you store a value into a new owner—binding a new variable (`b := a`), reassigning (`x = y`), initializing a struct field (`ReportRecord{ tag: name }`), or pushing into a slice (`batch.push(item)`)—the compiler automatically chooses between a **deep copy** and a **last-use move**:

- **Copy:** If any subsequent execution path can still read the source variable, Vira emits an independent deep copy so mutating one owner never affects the other.
- **Move:** If the compiler proves no path reads the source variable again, ownership transfers directly to the destination with **zero heap allocation and zero data copying**.

```v
orig := make_record('alpha', 101)

// `orig` is read again below, so `copy` gets an independent deep copy.
mut copy := orig
copy.tag = 'beta'
append_tag(mut copy, 'canary')

println('    orig: ${format_record(orig)}')
println('    copy: ${format_record(copy)} (first=${copy.tags[0]}, second=${copy.tags[1]})')
```

Because `orig` is passed to `format_record(orig)` later in the function, `copy := orig` clones `orig` and its nested `tags` slice. Modifying `copy.tag` and pushing `'canary'` onto `copy.tags` leaves `orig` untouched (`tag=alpha, tags=1`).

By contrast, when a local variable reaches its final use—such as `item` inside a loop or `staged` inside a branch—the compiler moves ownership directly into `batch`:

```v
mut batch := []string{}
for mut i := 0; i < 3; i += 1 {
	item := 'job-${i}'
	batch.push(item) // `item` is never read again: moved into `batch`
}

// Branch move: `staged` is declared outside the `if` and moved on the taken branch.
staged := 'priority-task'
take_staged := true
if take_staged {
	batch.push(staged)
}
```

### Move vs. Copy Decision Rules

| Situation | Outcome | Why |
| :--- | :--- | :--- |
| Source not read again in the same block | **Move** | Local destructor is skipped; destination becomes the sole owner. |
| Source declared in an enclosing block, stored inside `if` / `switch`, never read again | **Move** | Source is zeroed after the move so its outer scope destructor is a safe no-op. |
| Source read again on any later line | **Copy** | Both bindings remain valid, independent owners. |
| Source declared outside a loop, stored inside the loop body | **Copy** | The next loop iteration may read the outer variable again. |
| Source is a parameter, receiver, `for ... in` variable, field, or slice/map element | **Copy** | The caller or container retains ownership of the original. |

---

## 5.3. Borrowed Function Arguments and Returning Ownership

Function parameters in Vira are **borrowed by default**—passing a `string`, `[]T`, `map[K]V`, or `ReportRecord` to a function does **not** copy its heap buffers. The caller keeps ownership across the call:

```v
// Borrowed parameter: `r` stays owned by the caller; zero extra copy.
fn format_record(r ReportRecord) string {
	return 'Record#${r.id}[tag=${r.tag}, tags=${r.tags.len}]'
}

// Mutating a caller-owned struct in place via an explicit `mut` parameter.
fn append_tag(mut r ReportRecord, extra string) {
	r.tags.push(extra)
}
```

If a callee stores a borrowed parameter into a struct field, slice, or return value, Vira clones the stored value so the caller's original remains safe.

Conversely, when a function constructs a local value and returns it, ownership **moves** to the caller with zero copy:

```v
// Returning ownership: `res` is classified ESCAPES_VIA_RETURN and moves to caller.
fn make_record(name string, id i64) ReportRecord {
	mut tags := []string{}
	primary := 'env-' + name
	tags.push(primary) // Last use of `primary`: moved into `tags` without copy
	res := ReportRecord{
		id: id
		tag: name
		tags: tags
	}
	return res
}
```

Inside `make_record`, `primary` moves into `tags`, `tags` moves into `res`, and `res` moves out to the caller—all without cloning intermediate containers.

---

## 5.4. Static Escape Analysis (`StateScopedLocal`, `StateEscapesViaReturn`, `StateMovedToContainer`)

During compilation, the `autofree` pass assigns each heap-owning local variable an escape state from `internal/autofree/autofree.go`:

| Compiler State | Audit Report Label | Lifetime & Cleanup Behavior |
| :--- | :--- | :--- |
| `StateScopedLocal` | `SCOPED_LOCAL (autofree injected)` | Never leaves its declaring block; compiler injects LIFO destructor at scope exit (and before `return` / `break` / `continue`). |
| `StateEscapesViaReturn` | `ESCAPES_VIA_RETURN (ownership -> caller)` | Returned to the caller; local cleanup is skipped because the caller now owns it. |
| `StateMovedToContainer` | `MOVED_TO_CONTAINER (ownership -> struct/iface)` | Ownership moved into a struct, slice, map, or interface box; the container frees it when destroyed. |
| `StateManagedByArena` | `ARENA_MANAGED (O(1) slab free)` | Allocated inside `arena { ... }`; individual frees are skipped and the slab resets in `O(1)` at block exit. |

Consider `demo_escape_states()` in the companion file:

```v
fn demo_escape_states() {
	println('[2] Static Escape Analysis:')
	scoped := 'ephemeral-banner'
	moved_label := 'wrapped-item'
	holder := ReportRecord{
		id: 202
		tag: moved_label
		tags: ['core']
	}
	returned := make_record('gamma', 303)
	println('    scoped=${scoped}')
	println('    moved_holder=${format_record(holder)}')
	println('    returned=${format_record(returned)}')
}
```

Here, `scoped`, `holder`, and `returned` are `SCOPED_LOCAL` inside `demo_escape_states` (freed in LIFO order: `returned`, `holder`, `scoped`), while `moved_label` is `MOVED_TO_CONTAINER` because its last use moved it directly into `holder.tag`.

---

## 5.5. High-Throughput Batch Memory with Bump Arenas (`arena { ... }`)

When a hot loop, parser pass, or tensor forward pass creates many short-lived allocations, calling `malloc` and `free` per object adds allocator overhead. Vira provides lexically scoped **bump arenas**:

```v
fn demo_arena_batch() {
	println('[3] Bump Arena (O(1) Slab Allocation & Wipe):')
	mut persisted := 'initial'
	arena {
		t1 := tensor_fill(16, 16, 1.5)
		t2 := tensor_fill(16, 16, 2.0)
		prod := t1 @ t2
		total := tensor_sum(prod)
		scratch := 'arena-sum=${total}'
		// Assigning to an outer variable allocates on the heap so it survives slab reset.
		persisted = 'kept(' + scratch + ')'
	}
	println('    persisted=${persisted}')
}
```

Key properties of `arena { ... }`:
1. **64-Byte Cache Alignment:** Arena allocations are aligned for SIMD vector operations (`tensor[f32]` `@` matrix multiply).
2. **`O(1)` Bump Allocation & Bulk Teardown:** Individual object destructors inside the arena are skipped; when the `arena { ... }` block exits (including via early `return`, `break`, or `continue`), `vira_arena_destroy` wipes the entire slab at once.
3. **Safe Escape Out of an Arena:** Assigning from inside an arena to a variable declared **outside** the arena (such as `persisted = 'kept(' + scratch + ')'`) allocates the escaped result on the regular heap so it remains valid after the arena slab is destroyed.

---

## 5.6. Auditing Lifetimes and Leak Telemetry (`vira check` and `-sanitize`)

You can inspect every generated struct destructor, variable escape state, and injected cleanup call using `vira check`:

```bash
./vira check docs/handbook/chapters/code/ch05_ownership_and_arenas.vir
```

Output excerpt:
```text
✓ docs/handbook/chapters/code/ch05_ownership_and_arenas.vir passed syntax, 4-quadrant mutability, and type checks (0 errors)

1. Structs Requiring Generated `_autofree` Destructors:
   ├── struct main.ReportRecord -> requires `main.ReportRecord_autofree(&self)` (owns heap fields)
   ├── struct ReportRecord -> requires `ReportRecord_autofree(&self)` (owns heap fields)

2. Function Escape Analysis & Injected LIFO Destructors:
...
   [fn make_record]
     ├── var tags         ([]string) : MOVED_TO_CONTAINER (ownership -> struct/iface)
     ├── var primary      (string  ) : MOVED_TO_CONTAINER (ownership -> struct/iface)
     ├── var res          (ReportRecord) : ESCAPES_VIA_RETURN (ownership -> caller)
...
   [fn demo_arena_batch]
     ├── var persisted    (string  ) : SCOPED_LOCAL (autofree injected)
     ├── var t1           (tensor[f32]) : ARENA_MANAGED (O(1) slab free)
     ├── var t2           (tensor[f32]) : ARENA_MANAGED (O(1) slab free)
     ├── var prod         (tensor[f32]) : ARENA_MANAGED (O(1) slab free)
     ├── var scratch      (string  ) : ARENA_MANAGED (O(1) slab free)
     ├── var _vira_tmp_str_12 (string  ) : MOVED_TO_CONTAINER (ownership -> struct/iface)
     └── INJECTED FREE: vira_arena_destroy(&_vira_arena_1) [vira_arena] (arena block exit (O(1) slab deallocation))
     └── INJECTED FREE: vira_str_free(&_vira_tmp_str_12) [string] (zero-leak temporary string operand cleanup)
     └── INJECTED FREE: vira_str_free(&persisted) [string] (function scope exit)
```

For hardware-level verification of the generated C99 code, build with `-sanitize` to enable **AddressSanitizer (ASan)** and **UndefinedBehaviorSanitizer (UBSan)**, then run the binary and verify both zero sanitizer findings and `active_leaks=0`:

```bash
./vira build -sanitize -o /tmp/ch05_san docs/handbook/chapters/code/ch05_ownership_and_arenas.vir && /tmp/ch05_san
```

---

## 5.7. Running the Complete Chapter 5 Example

Run [`docs/handbook/chapters/code/ch05_ownership_and_arenas.vir`](../code/ch05_ownership_and_arenas.vir):

```bash
./vira run docs/handbook/chapters/code/ch05_ownership_and_arenas.vir
```

Output:
```text
[1] Copies vs. Last-Use Moves:
    orig: Record#101[tag=alpha, tags=1]
    copy: Record#101[tag=beta, tags=2] (first=env-alpha, second=canary)
    batch: len=4, items=[job-0, job-1, job-2, priority-task]
[2] Static Escape Analysis:
    scoped=ephemeral-banner
    moved_holder=Record#202[tag=wrapped-item, tags=1]
    returned=Record#303[tag=gamma, tags=1]
[3] Bump Arena (O(1) Slab Allocation & Wipe):
    persisted=kept(arena-sum=12288.00)
[vira_rt] simd=x86_64 SSSE3 + SSE2 (128-bit Integer/Float SIMD, Multi-Threaded)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=33
```

---

## Summary & Exercises

1. **Observe Copy vs. Move Telemetry:** In `demo_copies_and_moves()`, remove the `println('    orig: ...')` line after `mut copy := orig` (or move it before the copy) and compare `total_freed` and the `./vira check` report.
2. **Early Return Inside an Arena:** Add an early `if total > 1000.0 { return }` or return a `ReportRecord` directly from inside `arena { ... }`, then verify with `./vira build -sanitize` that `vira_arena_destroy` still runs cleanly with `active_leaks=0`.
3. **Loop-Carried vs. Loop-Local Moves:** Move `item := 'job-${i}'` outside the `for` loop in `demo_copies_and_moves()`, pass it to `batch.push(item)` on every iteration, and check how `./vira check` and `total_freed` reflect that loop-external values are copied rather than moved.
