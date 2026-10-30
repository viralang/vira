# Tutorial 1: Getting Started, Variables & Control Flow

> **Difficulty:** Beginner  
> **Matching Code Example:** [`docs/examples/01_basics.vir`](../examples/01_basics.vir)

---

## 1. Overview & Setup

Welcome to the Vira programming language! In this tutorial, you will learn the foundational syntax and core language features:
- How to declare and structure Vira source modules
- Variables, immutability by default, and the `mut` keyword
- Top-level compile-time constants (`const` and `pub const`)
- Primitive data types, octal/binary literals, and explicit numeric casts
- Zero-cost `${...}` string interpolation and standard console output (`print`, `println`, `eprint`, `eprintln`)
- Slices and sub-slicing (`.slice(start, end)`)
- Deterministic cleanup with `defer`
- Conditional expressions (`if`/`else`)
- The blank identifier `_`
- Importing standard library packages from [`packages/`](file:///mnt/flutter_drive/vira/packages)

---

## 2. Module Declarations

Every Vira source file begins with a `module` declaration defining the namespace boundary:

```v
module basics
```

In Vira, symbols marked with `pub` are visible outside the module; any symbol without `pub` remains strictly module-private.

---

## 3. Variables, Mutability & Constants

### Immutable & Mutable Variables
In Vira, all variable bindings are **immutable by default**. This eliminates accidental reassignments and data races in concurrent code:

```v
// Immutable binding (value cannot change)
company_name := 'VexCorp'
established := 2026

// Attempting to reassign will cause a compiler error:
// company_name = 'OtherCorp' // ERROR: cannot reassign immutable variable 'company_name'
```

To allow a variable to be reassigned, declare it explicitly with the `mut` keyword:

```v
// Mutable binding
mut current_headcount := 42
current_headcount = current_headcount + 5 // OK!
```

### Top-Level Constants (`const` and `pub const`)
Constants are immutable values known at compile time. Declare them at the file or module level:

```v
const MAX_RETRIES = 5
const BUFFER_SIZE = 4096
pub const DEFAULT_PORT = 8080
pub const APP_NAME = 'ViraService'
```

Constants can be exported across packages using `pub const` and referenced with qualified access (e.g., `syscall.CLONE_NEWPID` or `math.PI`).

---

## 4. Primitive Types, Numeric Literals & Type Casting

Vira provides explicit, hardware-sized scalar types:

| Type | Bits | Description | Example |
| :--- | :--- | :--- | :--- |
| `u8` | 8 | Unsigned 8-bit byte | `b := u8(65)` |
| `int` | 32 | Default signed integer | `id := 1001` |
| `i64` | 64 | 64-bit integer (timestamps, large offsets) | `salary := 12500000i64` |
| `f32` | 32 | Single-precision float | `ratio := 0.25f32` |
| `f64` | 64 | Double-precision float | `pi := 3.1415926535` |
| `bf16`| 16 | Brain Floating Point for AI tensors | `weight := 0.125bf16` |
| `bool`| 8 | Boolean (`true` or `false`) | `is_active := true` |
| `string` | 128 | UTF-8 slice (pointer + length) | `title := 'Vira Language'` |

### Octal & Binary Literals
In systems and low-level development, permissions and bitmasks are represented natively:
- **Octal literals** begin with `0o` (e.g. `0o755` for directory permissions, `0o644` for files).
- **Binary literals** begin with `0b` (e.g. `0b1010` for bitflags).
- **Hexadecimal literals** begin with `0x` (e.g. `0xFF`).

```v
mode := 0o755        // POSIX read/write/execute permissions
flags := 0b10100100   // Bitmask flag
```

### Explicit Numeric Casting
Vira enforces strict type safety without implicit narrowing or widening. Convert between primitive types using explicit functional cast syntax:

```v
count := 42
as_i64 := i64(count)     // Convert int to i64
as_f64 := f64(count)     // Convert int to f64
byte_val := u8(count)    // Convert int to u8 byte
as_int := int(as_f64)    // Convert float back to int
```

---

## 5. String Interpolation & Console Output

Vira provides built-in console output functions:
- `println(msg)`: Prints `msg` to standard output followed by a newline.
- `print(msg)`: Prints `msg` to standard output without an automatic newline (ideal for prompts and progress bars).
- `eprintln(msg)`: Prints `msg` to standard error followed by a newline (for diagnostics, warnings, and errors).
- `eprint(msg)`: Prints `msg` to standard error without a newline.

String interpolation uses `${...}` syntax without heap overhead:

```v
user := 'Alice'
score := 98
println('User @${user} scored ${score} points!')
eprintln('Warning: low memory threshold reached')
```

Built-in strings also provide rich utility methods:
```v
s := '  Hello, Vira!  '
trimmed := s.trim()                      // 'Hello, Vira!'
upper := trimmed.to_upper()              // 'HELLO, VIRA!'
sub := trimmed.replace('Vira', 'World')  // 'Hello, World!'
has_sub := trimmed.contains('Vira')      // true
```

---

## 6. Slices & Sub-Slicing

Dynamic collections in Vira are represented as slices (`[]T`). You can create slices, push items, and slice sub-ranges using `.slice(start, end)`:

```v
mut items := ['alpha', 'beta', 'gamma', 'delta']
items.push('epsilon')

// Extract sub-slice from index 1 to 4: ['beta', 'gamma', 'delta']
sub := items.slice(1, 4)
println('Sub-slice length: ${sub.len}')
```

All slice elements are managed by Vira's compile-time Autofree engine, guaranteeing zero leaks upon scope exit.

---

## 7. Deterministic Resource Cleanup with `defer`

The `defer` statement schedules an expression or block to execute when the enclosing function or scope returns, in Last-In-First-Out (**LIFO**) order:

```v
fn process_resource() {
    println('1. Acquired resource')
    defer println('3. Released resource (via defer)')
    
    println('2. Working with resource...')
} // Defer executes here automatically before function exit
```

`defer` ensures that files, network handles, locks, and temporary directories are reliably cleaned up on every exit branch—including early returns and error unwraps.

---

## 8. Control Flow: `if` / `else`

Conditions in `if` statements must evaluate strictly to a `bool`. Parentheses around the condition are optional:

```v
fn calculate_bonus(base i64, bonus_cents i64, is_top_performer bool) i64 {
	if is_top_performer {
		return base + bonus_cents
	} else {
		return base
	}
}
```

---

## 9. The Blank Identifier (`_`)

The blank identifier `_` is used to discard values without compiler warnings or symbol table pollution:

```v
_ := calculate_bonus(50000, 10000, false)
_ := 'discarded string'
```

You can assign to `_` multiple times in the same scope without redeclaration errors.

---

## 10. Using Standard Library Packages (`packages/*`)

Vira includes a rich, high-performance standard library directly under [`packages/`](file:///mnt/flutter_drive/vira/packages). You can import any standard library package by name without extra configuration:

```v
import os
import math
import strings
import time

fn main() {
    // Current executable arguments via os.args()
    args := os.args()
    println('Executing: ${args[0]}')

    // High-performance math
    hyp := math.hypot(3.0, 4.0)
    println('Hypotenuse: ${hyp}')

    // String utilities
    joined := strings.join(['vira', 'is', 'fast'], '-')
    println('Joined: ${joined}')

    // High-resolution clock
    now := time.now_unix()
    println('Unix epoch: ${now}')
}
```

---

## 11. Running the Code

Execute the matching example using the Vira CLI:

```bash
vira run docs/examples/01_basics.vir
```

Expected output:
```text
=== 1. IMMUTABILITY & MUTABILITY ===
Welcome to VexCorp (Est. 2026)
Initial headcount: 42
Updated headcount after hiring: 47
=== 2. NUMERIC & PRIMITIVE TYPES ===
Employee #1001: senior=true, rating=4.85
=== 3. CONTROL FLOW & CONDITIONAL LOGIC ===
Base salary: 12500000 cents
Final compensation with bonus: 15000000 cents
=== 4. BLANK IDENTIFIER (_) ===
Blank identifier discards values cleanly without warnings
```

---

## Next Steps

- Proceed to [**Tutorial 2: OOP, 4-Quadrant Structs & Composition**](./02_oop_and_4_quadrant_structs.md) to learn how to model real-world domain logic with Vira's struct encapsulation and composition system.
- Explore the comprehensive [**Standard Library Guide**](../STANDARD_LIBRARY.md) and [`packages/`](../../packages) reference for in-depth coverage of all 14 standard packages.

