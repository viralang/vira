# Chapter 4: Functions, Control Flow & Pattern Switching

Once you know how Vira binds immutable values and enforces strict primitive types, the next step is structuring program logic into reusable functions and deterministic control flow. Vira keeps control flow intentionally uniform: functions borrow parameters cleanly by default, `if` requires strict `bool` conditions, `switch` breaks automatically without fallthrough traps, a single `for` keyword handles every kind of iteration, and `defer` binds cleanup directly to lexical block exits.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch04_functions_and_control_flow.vir`](../code/ch04_functions_and_control_flow.vir)

---

## 4.1. Defining Functions, Parameters, and Return Values

Functions in Vira are declared with the `fn` keyword. Like module constants, functions are private to their module by default and exported with `pub fn`:

```v
pub fn clamp(val int, lo int, hi int) int {
	if val < lo {
		return lo
	}
	if val > hi {
		return hi
	}
	return val
}
```

### Borrowed Parameters vs. Explicit `mut` Parameters

Vira passes parameters as **read-only borrowed values by default**:
- For scalar primitives (`int`, `f64`, `bool`), parameters behave like local read-only copies.
- For heap-backed values (`string`, `[]T`, `map[K]V`, structs), the caller keeps ownership while the callee borrows the value with zero extra allocations.

When a function genuinely needs to mutate a caller's variable in place, you must mark the parameter `mut` in **both** the function signature and every call site:

```v
fn bump_counter(mut count int, delta int) {
	count += delta
}

fn main() {
	mut jobs := 10
	bump_counter(mut jobs, 5) // jobs is now 15
}
```

> **Requiring `mut` at Call Sites:** Writing `bump_counter(mut jobs, 5)` ensures a reader never has to open `bump_counter`'s definition to know whether `jobs` might change on that line. Omitting `mut` at the call site is a compile-time error.

### Multi-Line Formatting and the Newline Expression Boundary

Because Vira omits semicolons, the parser uses newlines to separate statements cleanly. Across a newline, expressions do **not** implicitly continue onto a leading unary `+`, `-`, matrix `@`, index `[`, or call `(` token on the next line. If you want a binary expression to wrap across lines, keep the binary operator at the end of the line before the break:

```v
total := alpha +
	beta +
	gamma
```

---

## 4.2. Conditional Branching (`if`, `else if`, `else`)

Conditional statements use `if`, `else if`, and `else`. Conditions do **not** need parentheses, and curly braces `{ ... }` are always mandatory—even for single-line branches:

```v
fn classify_load(cpu_pct int) string {
	if cpu_pct >= 90 {
		return 'critical'
	} else if cpu_pct >= 65 {
		return 'elevated'
	} else {
		return 'nominal'
	}
}
```

### Strict `bool` Conditions

Vira has **no truthy or falsy implicit conversions**:
- Integers (`if count { ... }`), strings (`if name { ... }`), and pointers cannot be used directly as branch conditions.
- You must write an explicit boolean comparison (`if count > 0 { ... }`, `if name.len > 0 { ... }`).

---

## 4.3. Multi-Branch `switch` Statements

When a branch depends on comparing one value against several alternatives, or choosing among ordered ranges, `switch` is cleaner and safer than a long `else if` chain.

Vira's `switch` follows four core rules:
1. **Automatic `break`:** Matching a `case` executes that branch and exits the `switch` automatically. There is no accidental C-style fallthrough.
2. **Comma-Separated Case Lists:** Match multiple values in a single branch with commas (`case 201, 202:`).
3. **Strings and Enums:** Compare integer codes, enums, or UTF-8 `string` values directly (string cases lower cleanly to fast byte-length and content checks via `vira_str_eq`).
4. **Tagless `switch { ... }`:** Omit the target expression to evaluate arbitrary boolean conditions top-to-bottom.

```v
fn status_group(code int) string {
	switch code {
	case 200:
		return 'ok'
	case 201, 202:
		return 'accepted'
	case 400, 404:
		return 'client-error'
	case 500, 502, 503:
		return 'server-error'
	default:
		return 'other'
	}
}

fn dispatch_cmd(cmd string) string {
	mut action := ''
	switch cmd {
	case 'start', 'resume':
		action = 'worker-active'
	case 'pause':
		action = 'worker-paused'
	case 'stop':
		action = 'worker-stopped'
	default:
		action = 'unknown-command'
	}
	return action
}

fn latency_tier(latency_us int) string {
	switch {
	case latency_us < 50:
		return 'l1-fast'
	case latency_us < 500:
		return 'in-memory'
	default:
		return 'io-bound'
	}
}
```

Just like ordinary blocks, any temporary strings or containers allocated inside a `case` arm are tracked by Autofree and released before leaving that branch.

---

## 4.4. Unified `for` Loops

Vira unifies `while`, C-style loops, infinite event loops, slice traversal, and map iteration under a **single keyword: `for`**.

| Form | Syntax | Common Use Case |
| :--- | :--- | :--- |
| **Condition-Only (`while`)** | `for cond { ... }` | Repeat until a state condition becomes false. |
| **3-Part C-Style** | `for mut i := 0; i < n; i += 1 { ... }` | Explicit index counters with custom step size. |
| **Infinite Loop** | `for { ... }` | Server accept loops or state machines exited via `break` / `return`. |
| **Sequence Iteration** | `for item in items { ... }` | Zero-allocation indexed iteration over `[N]T` arrays and `[]T` slices. |
| **Map Iteration** | `for k, v in table { ... }` | Insertion-ordered key/value traversal over `map[K]V`. |

All five forms support `continue` (skip to the next iteration) and `break` (exit the innermost loop immediately):

```v
// 1. Condition-only (`while` style)
mut n := 8
mut steps := 0
for n > 1 {
	n = n / 2
	steps += 1
}

// 2. 3-part C-style with `continue` and `break`
mut odd_sum := 0
for mut i := 0; i < 8; i += 1 {
	if i % 2 == 0 {
		continue
	}
	if i > 5 {
		break
	}
	odd_sum += i
}

// 3. Infinite loop with `break`
mut ticks := 0
for {
	ticks += 1
	if ticks == 3 {
		break
	}
}

// 4. Slice iteration and 5. Insertion-ordered map iteration
weights := [2, 4, 6]
mut total_w := 0
for w in weights {
	total_w += w
}

quotas := map[string]int{
	'gpu': 4,
	'cpu': 16,
}
for k, v in quotas {
	println('${k}=${v}')
}
```

---

## 4.5. Deterministic Scope Cleanup with `defer`

When a function acquires a handle, changes state temporarily, or logs trace markers across multiple early `return` paths, duplicating cleanup code at every exit is error-prone. Vira's `defer` statement schedules cleanup deterministically on exit—accepting either a single expression (`defer println(...)`) or a multi-statement block (`defer { ... }`).

### Lexical Scope & LIFO Execution Order

Unlike runtime-allocated defer lists in garbage-collected languages, Vira lowers `defer` **purely at compile time** in the Autofree pass with zero heap allocation per `defer`:
1. **LIFO Order (Last-In, First-Out):** Multiple `defer` statements in the same scope execute in reverse order of registration—just like unwinding a stack.
2. **Lexical Block Scoped:** A `defer` runs when its **enclosing lexical block** exits (normal fallthrough, early `return`, `break`, or `continue`), **before** Autofree emits destructors for variables declared in that same block. Putting a `defer` inside a loop body runs it at the end of each iteration rather than leaking work until the end of the function.
3. **Runs on Early Returns (and Actor Panics):** Returning early from any branch triggers every active `defer` along the unwinding path.

Look closely at `run_stage` in the companion example:

```v
fn run_stage(name string, fail_early bool) string {
	mut trace := '${name}:'
	defer {
		println('    [defer-block] cleanup ${name}')
	}
	defer println('    [defer-stmt] close ${name}')

	if fail_early {
		trace += 'aborted'
		return trace
	}

	if !fail_early {
		defer println('    [defer-inner] branch-scope-exit')
		trace += 'ok'
	}
	return trace
}
```

Notice what happens in both call paths:
- When `fail_early` is `false`, the inner `if` branch finishes first and triggers `[defer-inner]`, and then `return trace` triggers the outer two defers in LIFO order: first `[defer-stmt]`, then `[defer-block]`.
- When `fail_early` is `true`, the early `return trace` still triggers `[defer-stmt]` followed by `[defer-block]` before returning `'beta:aborted'`.

---

## 4.6. Running the Complete Chapter 4 Example

Run [`docs/handbook/chapters/code/ch04_functions_and_control_flow.vir`](../code/ch04_functions_and_control_flow.vir):

```bash
./vira run docs/handbook/chapters/code/ch04_functions_and_control_flow.vir
```

Output:
```text
[1] Functions: clamp(145)=100, badge=[scheduler:lvl=3], jobs=15
[2] Branches: load(42)=nominal, load(75)=elevated, load(95)=critical
[3] Switch: 201->accepted, 503->server-error, resume->worker-active, 120us->in-memory
[4] Loops: while_steps=3, odd_sum=9, inf_ticks=3, weights=12, quotas=[gpu=4,cpu=16]
[5] Defer:
    [defer-inner] branch-scope-exit
    [defer-stmt] close alpha
    [defer-block] cleanup alpha
    result=alpha:ok
    [defer-stmt] close beta
    [defer-block] cleanup beta
    result=beta:aborted
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=29
```

---

## Summary & Exercises

1. **`mut` Call-Site Enforcement:** Remove `mut` from `bump_counter(mut jobs, 5)` in `ch04_functions_and_control_flow.vir` and run `./vira check` to inspect the semantic error diagnostic.
2. **Tagless Range Classifier:** Extend `latency_tier` with another threshold case (`< 2000` for `'disk-spill'`) and a boundary test case in `main()`—verify `./vira run` passes with `active_leaks=0`.
3. **Loop-Scoped `defer`:** Add a `for mut step := 0; step < 3; step += 1` loop containing a `defer println('end step ${step}')` and observe that it runs at the end of every iteration (and even if an iteration hits `continue`), not just when `main()` exits.
