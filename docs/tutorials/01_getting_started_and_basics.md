# Tutorial 1: Getting Started, Variables & Control Flow

> **Difficulty:** Beginner  
> **Matching Code Example:** [`docs/examples/01_basics.vir`](../examples/01_basics.vir)

---

## 1. Overview & Setup

Welcome to the Vira programming language! In this tutorial, you will learn the foundational syntax and concepts of Vira:
- How to declare and structure Vira source modules
- Variables, immutability by default, and the `mut` keyword
- Primitive data types
- Zero-cost `${...}` string interpolation
- Conditional expressions (`if`/`else`)
- The blank identifier `_`

---

## 2. Module Declarations

Every Vira source file begins with a `module` declaration defining the namespace boundary:

```v
module basics
```

In Vira, symbols marked with `pub` are visible outside the module; any symbol without `pub` remains strictly module-private.

---

## 3. Variables & Immutability

In Vira, all variable bindings are **immutable by default**. This eliminates accidental reassignments and data races in concurrent code.

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

---

## 4. Primitive Types

Vira provides explicit, hardware-sized scalar types:

| Type | Bits | Description | Example |
| :--- | :--- | :--- | :--- |
| `int` | 32 | Default signed integer | `id := 1001` |
| `i64` | 64 | 64-bit integer (timestamps, cents) | `salary := 12500000` |
| `f32` | 32 | Single-precision float | `ratio := 0.25` |
| `f64` | 64 | Double-precision float | `pi := 3.14159265` |
| `bf16`| 16 | Brain Floating Point for AI | `weight := 0.125bf16` |
| `bool`| 8 | Boolean | `is_active := true` |
| `string` | 128 | UTF-8 slice (pointer + length) | `title := 'Vira Language'` |

---

## 5. String Interpolation

Vira supports string interpolation without runtime string concatenation libraries or heap overhead for literals:

```v
user := 'Alice'
score := 98
println('User @${user} scored ${score} points!')
```

---

## 6. Control Flow: `if` / `else`

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

## 7. The Blank Identifier (`_`)

The blank identifier `_` is used to discard values without compiler warnings or symbol table pollution:

```v
_ := calculate_bonus(50000, 10000, false)
_ := 'discarded string'
```

You can assign to `_` multiple times in the same scope without redeclaration errors.

---

## 8. Running the Code

Execute the matching example using the Vira CLI:

```powershell
.\vira.exe run docs/examples/01_basics.vir
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

Proceed to [**Tutorial 2: OOP, 4-Quadrant Structs & Composition**](./02_oop_and_4_quadrant_structs.md) to learn how to model real-world domain logic with Vira's struct encapsulation and composition system.
