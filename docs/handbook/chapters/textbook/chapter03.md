# Chapter 3: Variables, Data Types & Operators

Every systems programming language is built on a foundation of how values are bound to names, how primitive types are represented in memory, and what operations are valid on those types. In this chapter, we explore Vira's immutability-first variable bindings, compile-time constants, scalar data types, numeric literals, explicit type conversions, and strict operator rules.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch03_variables_and_types.vir`](../code/ch03_variables_and_types.vir)

---

## 3.1. Variables and Immutability by Default

In Vira, variables are declared and initialized with the `:=` short declaration operator. By default, **every variable binding is immutable**.

```v
port := 8080
// port = 9090  <-- Compile-time error!
```

If you attempt to reassign `port`, the Vira type checker stops compilation with an actionable diagnostic:

```text
error[SEMANTIC]: cannot reassign immutable variable 'port'
  --> main.vir:5:2
   |
 5 |     port = 9090
   |     ^^^^^^^^^^^
   = help: declare with 'mut port := ...'
```

### Opting Into Mutability with `mut`

When a variable's value needs to change over time, prefix its declaration with `mut`:

```v
mut active_connections := 0
active_connections += 1
active_connections = active_connections + 2
```

Notice the distinction between **declaration** (`:=`) and **reassignment** (`=` or `+=`, `-=`, `*=`, `/=`, `%=`):
- `:=` creates a **new** variable in the current lexical scope. Re-declaring the same variable name in the same scope is a compile-time error (no accidental variable shadowing).
- `=` mutates an **existing** variable that was declared with `mut`.

### The Blank Identifier (`_`)

Sometimes a function returns a value that you intentionally want to discard. Assigning to the blank identifier `_` evaluates the right-hand side (and runs any automatic cleanup if the value owns heap memory) without binding a variable name in scope:

```v
_ := compute_Work()
_ := 'discarded string'
```

Unlike normal variables, `_` can be assigned multiple times in the same block without triggering a redeclaration error.

---

## 3.2. Compile-Time Constants (`const` and `pub const`)

Constants are values bound to a name at compile time using the `const` keyword at the module top level:

```v
pub const MAX_RETRIES = 3
const PI_APPROX = 3.14159265
const SERVICE_TAG = 'vira-core'
```

Key rules for constants:
1. **Declared at Module Scope:** Constants are declared outside functions using `=` (not `:=`).
2. **Visibility:** A constant is private to its module by default; prefix it with `pub const` to export it to other modules.
3. **Always Immutable:** Constants can never be marked `mut` or reassigned.

---

## 3.3. Scalar Primitive Types

Vira is a statically typed language: every expression has a concrete type known at compile time. When you write `x := 42`, the compiler infers `int`; when you write `y := 3.14`, it infers `f64`.

| Type | C99 Representation | Size | Description & Typical Use |
| :--- | :--- | :--- | :--- |
| **`u8`** / **`byte`** | `uint8_t` | 1 byte (8-bit) | Unsigned byte; raw binary buffers, network packets, ASCII bytes. |
| **`int`** | `int32_t` | 4 bytes (32-bit) | Default signed integer; loop indices, counts, general arithmetic. |
| **`i64`** | `int64_t` | 8 bytes (64-bit) | Signed 64-bit integer; timestamps, file sizes, financial cents, large counters. |
| **`f32`** | `float` | 4 bytes (32-bit) | Single-precision IEEE 754 float; neural network weights, SIMD tensors, GUI coordinates. |
| **`f64`** | `double` | 8 bytes (64-bit) | Default double-precision IEEE 754 float; scientific math, high-precision ratios. |
| **`bf16`** | `uint16_t` | 2 bytes (16-bit) | Brain Floating Point; compact AI and Transformer weights. |
| **`bool`** | `bool` | 1 byte (8-bit) | Boolean truth value (`true` or `false`). |
| **`string`** | `vira_string` | 16 bytes | Immutable UTF-8 string header (`{ data*, len, is_lit, is_borrowed }`). |

> **FFI Scalar Types:** For C-ABI interoperability (covered in Chapter 17), Vira also provides exact C width aliases such as `c_char`, `c_short`, `c_int`, `c_long`, `c_size`, `c_float`, `c_double`, and `voidptr`.

---

## 3.4. Numeric Literals and Explicit Type Casts

### Base Prefixes and Digit Separators

Integer literals can be written in decimal, hexadecimal (`0x`), octal (`0o`), or binary (`0b`). You can insert underscores (`_`) anywhere between digits to make large numbers or bit masks easy to read:

```v
hex_mask := 0xFF
oct_perm := 0o755
bin_flags := 0b1010_0101
million := 1_000_000
```

### Explicit Type Casts (`T(expr)`)

Vira does not silently convert between integers and floating-point numbers, nor does it implicitly narrow wider types. To convert a value from one scalar type to another, call the target type name like a function:

```v
b_val := u8(255)
i_val := 42
big_val := i64(9000000000)
f32_val := f32(0.125)

ratio := f64(i_val) / PI_APPROX
truncated := int(ratio) // Truncates toward zero (13.37 -> 13)
```

---

## 3.5. Operators, Precedence, Overflow Wrapping, and Strict Type Rules

Vira enforces strict operator typing: **both operands of a binary operator must have compatible types, and the operator must be explicitly defined for that type.**

### Operator Compatibility Matrix

| Category | Operators | Defined For | Notes |
| :--- | :--- | :--- | :--- |
| **Addition** | `+`, `+=` | Numbers, `string`, `tensor[f32]`, pointers | On `string`, `+` concatenates (a chain `a + b + c` allocates only once). |
| **Subtraction** | `-`, `-=` | Numbers, `tensor[f32]`, pointers | Rejects `string` and `bool` at compile time. |
| **Multiply / Divide** | `*`, `/`, `*=`, `/=` | Numbers | Integer `/` by zero panics cleanly instead of triggering hardware UB. |
| **Remainder** | `%`, `%=` | Integers (`u8`, `int`, `i64`) | Not defined on `f32`/`f64`; `% 0` panics cleanly. |
| **Matrix Multiply** | `@` | `tensor[f32]` | 2D GEMM, 3D/4D batched attention, and broadcast projections. |
| **Bitwise** | `&`, `\|`, `^` | Integers, `bool` | Bitwise AND, OR, XOR. |
| **Bit Shifts** | `<<`, `>>` | Integers | Left shift and arithmetic/logical right shift. |
| **Ordering** | `<`, `<=`, `>`, `>=` | Numbers, `string`, pointers | Strings are compared lexicographically by UTF-8 byte content. |
| **Equality** | `==`, `!=` | Scalar primitives, `string`, `enum`, pointers | Strings compare by content. Structs, slices, maps, and actors do not define `==`. |
| **Logical** | `&&`, `\|\|`, `!` | `bool` only | Short-circuiting boolean logic (`1 && 0` is a compile error). |

### Defined Arithmetic Semantics (No C Undefined Behavior)

In standard C, signed integer overflow and division by zero are Undefined Behavior (UB) that optimizing compilers can exploit in surprising ways. Vira eliminates both hazards:

1. **Deterministic Two's-Complement Overflow Wrapping:** Vira compiles with defined two's-complement wrap-around semantics (`-fwrapv`). Adding `1` to the maximum 32-bit signed integer (`2147483647`) deterministically wraps to `-2147483648`.
2. **Checked Integer Division & Modulo:** Every integer `/` and `%` operation checks for a zero divisor via `_vira_div` / `_vira_mod` in the runtime. Dividing by zero prints `vira panic: integer division by zero` (and is safely caught and restarted if inside an actor).
3. **Left-to-Right Evaluation Order:** Function arguments and chained expressions like string `+` are evaluated strictly left to right.

---

## 3.6. Running the Complete Chapter 3 Example

Run [`docs/handbook/chapters/code/ch03_variables_and_types.vir`](../code/ch03_variables_and_types.vir):

```bash
./vira run docs/handbook/chapters/code/ch03_variables_and_types.vir
```

Output:
```text
[1] Bindings: port=8080, active_connections=3, max_retries=3
[2] Scalars: u8=255, int=42, i64=9000000000, f32=0.12, f64=2.72, bool=true
[3] Bases: hex=0xFF(255), oct=0o755(493), bin=0b10100101(165)
    Casts: f64(42) / 3.14 = 13.37 -> int(13)
[4] Operators: sum=42, rem=2, shifted=19, masked=3, toggled=12
    Overflow wrap: 2147483647 + 1 = -2147483648
[5] Strings: "[vira-core] status=online" (match=true)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=10
```

---

## Summary & Exercises

1. **Operator Type Safety:** Try adding `'hello' - 'world'` or `1 && 2` inside `ch03_variables_and_types.vir` and run `./vira check` to see how the type checker rejects invalid operator combinations.
2. **Bit Manipulation:** Write an expression using `<<`, `|`, and `&` that packs four `u8` color channels `(r, g, b, a)` into a single 32-bit `int` and extracts the green channel back out.
3. **Division by Zero Trap:** Try computing `10 / 0` in `main()` and run the program to observe Vira's runtime division-by-zero panic message.

