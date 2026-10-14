# Tutorial 11: Control Flow, Loops, and Operators

> **Difficulty:** Beginner to Intermediate  
> **Matching Code Example:** [`docs/examples/11_control_flow_loops_operators.vex`](../examples/11_control_flow_loops_operators.vex)

---

## 1. Overview

Vex combines expressive, clean syntax with zero-overhead C99 emission. Unlike languages with sprawling operator overloading or ambiguous ternary expressions, Vex emphasizes:
- **Explicit Branching:** Predictable `if` / `else if` / `else` chains.
- **Unified Loop Keyword (`for`):** One keyword for all looping needs: C-style 3-part iterations, condition-only (while) loops, and infinite event loops.
- **Fine-Grained Loop Control:** Native `break` and `continue`.
- **Complete Operator Set:** Arithmetic, bitwise, logical, relational, compound assignments, and first-class hardware operators like `@` (SIMD matrix multiplication).

---

## 2. Conditional Branching (`if`, `else if`, `else`)

Conditions in Vex do not require parentheses around the test expression, but braces `{ ... }` around the body are mandatory:

```vex
fn classify_temperature(temp f32) string {
	if temp >= 95.0 {
		return 'CRITICAL_OVERHEAT'
	} else if temp >= 80.0 {
		return 'WARNING_HOT'
	} else if temp >= 50.0 {
		return 'OPTIMAL'
	} else if temp >= 20.0 {
		return 'COOL'
	} else {
		return 'SUB_ZERO'
	}
}
```

### Key Rules
1. **Boolean Conditions:** The condition expression must evaluate to `bool`. Truthy/falsy integer casting is rejected at compile time.
2. **Deterministic Scoping:** Variables declared within an `if` or `else` block are scoped strictly to that block and automatically freed upon exit.

---

## 3. The Unified `for` Loop

Vex unifies all iteration patterns under the `for` keyword.

### Form 1: 3-Part C-Style Loop
Used for counted iterations over intervals:

```vex
mut sum_squares := 0
for mut i := 1; i <= 6; i += 1 {
	sq := i * i
	sum_squares += sq
	println('Iter i=${i} -> square=${sq}, running_sum=${sum_squares}')
}
```

- The loop variable is initialized in the first clause (e.g. `mut i := 1`).
- The condition (`i <= 6`) is checked before each iteration.
- The post-statement (`i += 1`) executes after each block completion.

### Form 2: Condition-Only Loop (While-Style)
Used when iteration continues until a dynamic condition becomes false:

```vex
fn compute_collatz_steps(start_n int) int {
	mut n := start_n
	mut steps := 0
	for n > 1 {
		steps += 1
		if (n % 2) == 0 {
			n /= 2
		} else {
			n = (3 * n) + 1
		}
	}
	return steps
}
```

### Form 3: Infinite Loop with `break` and `continue`
Used for stream processors, event loops, or retry mechanisms:

```vex
mut stream_tick := 0
mut accepted_packets := 0
for {
	stream_tick += 1

	// Termination boundary
	if stream_tick > 12 {
		break
	}

	// Skip odd ticks using modulo operator (%)
	if (stream_tick % 2) != 0 {
		continue
	}

	accepted_packets += 1
	println('Processed even tick #${stream_tick}')
}
```

- `break`: Immediately exits the nearest enclosing loop.
- `continue`: Skips the remainder of the current iteration and jumps to the next loop cycle.

---

## 4. Operator Catalog

### Arithmetic Operators
Operate on `int`, `f32`, and `f64`:

| Operator | Description | Example |
| :---: | :--- | :--- |
| `+` | Addition | `a + b` |
| `-` | Subtraction or Unary Negation | `a - b`, `-x` |
| `*` | Multiplication | `a * b` |
| `/` | Division | `a / b` |
| `%` | Modulo (Remainder) | `a % b` |

### Compound Assignment Operators
Modify a mutable variable in-place:

```vex
mut accumulator := 100
accumulator += 50   // 150
accumulator -= 25   // 125
accumulator *= 3    // 375
accumulator /= 5    // 75
accumulator %= 32   // 11
```

### Relational & Comparison Operators
Produce a `bool` result:

| Operator | Meaning | Example |
| :---: | :--- | :--- |
| `==` | Equal | `x == y` |
| `!=` | Not equal | `x != y` |
| `<` | Less than | `x < y` |
| `<=` | Less than or equal | `x <= y` |
| `>` | Greater than | `x > y` |
| `>=` | Greater than or equal | `x >= y` |

### Logical Operators
Short-circuiting boolean operations:

| Operator | Meaning | Example |
| :---: | :--- | :--- |
| `&&` | Logical AND | `is_online && is_ready` |
| `\|\|` | Logical OR | `is_warning \|\| is_critical` |
| `!` | Logical NOT (unary prefix) | `!is_critical` |

```vex
is_online := node.has_flag(flag_online)
is_critical := node.has_flag(flag_critical)
is_stable := is_online && (!is_critical)
```

### Bitwise & Shift Operators
Perform bit-level manipulation on integer values:

| Operator | Meaning | Example |
| :---: | :--- | :--- |
| `&` | Bitwise AND | `flags & MASK` |
| `\|` | Bitwise OR | `flags \| MASK` |
| `^` | Bitwise XOR | `flags ^ MASK` |
| `<<` | Bitwise Shift Left | `1 << 3` (evaluates to `8`) |
| `>>` | Bitwise Shift Right | `32 >> 2` (evaluates to `8`) |

#### Practical Bitmask Example:
```vex
pub fn (mut node TelemetryNode) update_flag(flag int, enable bool) {
	if enable {
		node.status_flags = node.status_flags | flag
	} else {
		if (node.status_flags & flag) != 0 {
			node.status_flags = node.status_flags ^ flag
		}
	}
}

pub fn (node TelemetryNode) has_flag(flag int) bool {
	return (node.status_flags & flag) != 0
}
```

### Vex Special Operators

| Operator | Name | Purpose | Example |
| :---: | :--- | :--- | :--- |
| `@` | SIMD MatMul | Native hardware matrix multiplication | `c := a @ b` |
| `<-` | Actor Dispatch | Non-blocking message send returning `Future[T]` | `fut := worker <- process(batch)` |
| `!` | Result Type | Indicates a fallible function return | `fn read() !string` |
| `or { ... }` | Unwrap Fallback | Mandatory error handling block with implicit `err` | `val := compute() or { 0 }` |

---

## 5. Operator Precedence Hierarchy

Vex utilizes a Pratt precedence-climbing parser with unambiguous precedence tiers:

| Tier | Precedence Level | Operators | Associativity |
| :---: | :---: | :--- | :---: |
| **Highest** | **8** | `()`, `[]`, `.`, `<-`, Prefix `!`, Prefix `-` | Left-to-Right |
| | **7** | `*`, `/`, `%`, `@` (MatMul) | Left-to-Right |
| | **6** | `+`, `-` | Left-to-Right |
| | **5** | `<<`, `>>` | Left-to-Right |
| | **4** | `\|`, `^`, `&` | Left-to-Right |
| | **3** | `==`, `!=`, `<`, `<=`, `>`, `>=` | Left-to-Right |
| | **2** | `&&` | Left-to-Right |
| **Lowest** | **1** | `\|\|` | Left-to-Right |

---

## 6. Running the Example Program

Execute the verified example:

```powershell
.\vex.exe run docs/examples/11_control_flow_loops_operators.vex
```

### Execution Output
```
=================================================================
   VEX CONTROL FLOW, LOOPS & OPERATORS DEMONSTRATION             
=================================================================
--- 1. BRANCHING & MULTI-WAY IF / ELSE-IF / ELSE ---
Thermal reading: 82.50C -> Classification: WARNING_HOT
Thermal reading: -15.00C -> Classification: SUB_ZERO
--- 2. LOOPS: 3-PART C-STYLE FOR LOOP ---
  Iter i=1 -> square=1, running_sum=1
  Iter i=2 -> square=4, running_sum=5
  Iter i=3 -> square=9, running_sum=14
  Iter i=4 -> square=16, running_sum=30
  Iter i=5 -> square=25, running_sum=55
  Iter i=6 -> square=36, running_sum=91
Sum of squares 1..6: 91
--- 3. LOOPS: CONDITION-ONLY (WHILE) CONVERGENCE ---
Collatz conjecture steps for 27 to reach 1: 111
--- 4. LOOPS: INFINITE LOOP WITH BREAK & CONTINUE ---
  Processed even tick #2 (Total accepted: 1)
  Processed even tick #4 (Total accepted: 2)
  Processed even tick #6 (Total accepted: 3)
  Processed even tick #8 (Total accepted: 4)
  Processed even tick #10 (Total accepted: 5)
  Processed even tick #12 (Total accepted: 6)
--- 5. ARITHMETIC & COMPOUND ASSIGNMENT OPERATORS ---
Initial accumulator: 100
After (+= 50, -= 25, *= 3, /= 5, %= 32): 11
--- 6. BITWISE & SHIFT OPERATORS (&, |, ^, <<, >>) ---
Node status flags: 3
  has_flag(flag_online):  true
  has_flag(flag_warning): true
  has_flag(flag_critical):false
Bitwise shifts: (1 << 3) = 8, (8 >> 2) = 2
--- 7. LOGICAL BOOLEAN OPERATORS (&&, ||, !) ---
System Stability Check: PASSED (Node is online and not critical)
--- 8. SIMD TENSOR & SPECIAL OPERATORS (@) ---
SIMD Matmul (@) total sum of 16x16: 4096.00
=================================================================
   END OF FLOW, LOOPS & OPERATORS DEMO                           
=================================================================
[vex_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
[vex_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vex_rt] active_leaks=0 | total_freed=27
```
