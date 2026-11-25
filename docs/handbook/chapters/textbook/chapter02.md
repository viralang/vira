# Chapter 2: Tutorial — A Complete CLI Mini-Project

Just as *The Rust Programming Language* introduces the core feel of the language through a hands-on guessing game in Chapter 2, let's build a complete command-line number-guessing program in Vira. Along the way, we will see how Vira handles:

1. Project scaffolding (`vira init`) and standard library imports (`import os`, `import io`, `import strings`)
2. Compile-time constants (`const`) and mutable variables (`mut`)
3. Reading CLI arguments (`os.args()`) and interactive console input (`io.prompt`)
4. Parsing text into numbers and handling recoverable errors with `!T` and `or { ... }`
5. Looping (`for ... in`), branching (`if`/`else if`), and verifying **zero memory leaks** (`active_leaks=0`) across early `continue` and `break` jumps

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch02_guessing_game.vir`](../code/ch02_guessing_game.vir)

---

## 2.1. Setting Up a New Project (`vira init`)

When starting a standalone Vira project outside this repository, `vira init` scaffolds a clean project directory with a `vira.mod` manifest, `src/main.vir`, and `tests/`:

```bash
vira init guessing_game
cd guessing_game
```

This creates:
```text
guessing_game/
├── vira.mod          # Module manifest (name, version, dependencies)
├── src/
│   └── main.vir      # Application entry point
└── tests/
    └── main_test.vir # Automated unit test suite
```

For our handbook, you can also run any single `.vir` file directly without creating a directory. Open [`docs/handbook/chapters/code/ch02_guessing_game.vir`](../code/ch02_guessing_game.vir).

### Importing Standard Library Packages

At the top of `ch02_guessing_game.vir`, we declare our module and import three built-in standard library packages:

```v
module main

import os
import io
import strings

const SECRET_NUMBER = 42
```

- **`import os`**: Provides operating system helpers, including `os.args()` to read command-line arguments passed to the binary.
- **`import io`**: Provides stream and console I/O, including `io.prompt('...')` to print a prompt and read a line from `stdin`.
- **`import strings`**: Provides string utilities and integer base parsing (`strings.parse_i64_base`).
- **`const SECRET_NUMBER = 42`**: Declares a compile-time constant. Unlike variables (which live on the stack or heap at runtime), constants are evaluated at compile time and cannot be marked `mut`.

---

## 2.2. Reading Input & Command-Line Arguments (`io.prompt`, `os.args`)

To make our program easy to test both interactively and from automated scripts, let's check whether the user passed guesses on the command line (or `--interactive` for live terminal input), and fall back to a built-in sequence of test guesses when run with no arguments:

```v
	// Collect guesses from CLI arguments, or use a simulated sequence if none passed
	args := os.args()
	mut inputs := []string{}
	if args.len > 1 {
		for mut i := 1; i < args.len; i += 1 {
			inputs.push(args[i])
		}
	} else {
		inputs = ['18', 'not_a_num', '75', '42']
	}
```

### What's Happening Here:
1. **`args := os.args()`**: Returns a dynamic slice of strings (`[]string`) containing the command-line arguments. Just like in C's `argv`, `args[0]` is the program executable path, and `args[1]` onward are the user arguments.
2. **`mut inputs := []string{}`**: Initializes an empty growable slice of strings. Because we are going to modify `inputs` (either via `.push(...)` or reassignment), we declare it with `mut`.
3. **`for mut i := 1; i < args.len; i += 1`**: A 3-part C-style `for` loop. Notice that the loop counter `i` must be declared `mut i := 1` because `i += 1` mutates it on every iteration.

---

## 2.3. Handling Potential Failure with `!T` and `or { ... }`

User input from the command line or keyboard is untrusted: it might be empty, contain letters like `'not_a_num'`, or be outside our valid `1..100` range.

In Vira, functions that can fail do **not** throw exceptions or return error codes that you might accidentally ignore. Instead, they return a **Result type** written `!T` (pronounced *"Result of T"*):

```v
fn parse_guess(raw string) !int {
	trimmed := raw.trim()
	if trimmed.len == 0 {
		return error('empty input')
	}
	val := strings.parse_i64_base(trimmed, 10) or {
		return error('not a valid integer: "${trimmed}"')
	}
	if val < 1 || val > 100 {
		return error('guess ${val} is out of range (1..100)')
	}
	return int(val)
}
```

### Key Details in `parse_guess`:
- **Return Signature (`!int`)**: Tells the compiler and caller that `parse_guess` either succeeds with an `int` or fails with an error message.
- **`raw.trim()`**: Strips leading and trailing whitespace (`\n`, `\r`, spaces, tabs) and returns a zero-copy string view into `raw`.
- **`return error('...')`**: Constructs and returns an error from any `!T` function.
- **Unwrapping with `or { ... }`**: `strings.parse_i64_base(trimmed, 10)` returns `!i64`. Vira enforces at compile time that you handle the failure case with `or { ... }`. Inside the `or` block, you can return a fallback value, `return` from the function, or `break` / `continue` a loop.
- **Explicit Type Cast (`int(val)`)**: Vira never performs silent narrowing conversions between integer sizes. Because `parse_i64_base` returns a 64-bit `i64` and our function returns a 32-bit `int`, we convert it explicitly with `int(val)`.

---

## 2.4. Looping, Branching, and Zero-Leak Verification

Next, we write a pure helper function `check_guess` to compare the guess against `SECRET_NUMBER`, and our main loop in `fn main()` to process each input:

```v
fn check_guess(guess int, secret int) string {
	if guess < secret {
		return 'Too small!'
	} else if guess > secret {
		return 'Too big!'
	}
	return 'Exact match!'
}
```

And in `main()`:

```v
	mut attempts := 0
	for raw in inputs {
		println('\n> Input: "${raw}"')
		if raw == '--interactive' {
			line := io.prompt('Enter your guess (1-100): ') or {
				eprintln('End of input: ${err}')
				break
			}
			guess := parse_guess(line) or {
				println('  [Skip] ${err}')
				continue
			}
			attempts += 1
			verdict := check_guess(guess, SECRET_NUMBER)
			println('  Attempt #${attempts}: ${guess} -> ${verdict}')
			continue
		}

		guess := parse_guess(raw) or {
			println('  [Skip] ${err}')
			continue
		}

		attempts += 1
		verdict := check_guess(guess, SECRET_NUMBER)
		println('  Attempt #${attempts}: ${guess} -> ${verdict}')

		if guess == SECRET_NUMBER {
			println('\nYou won in ${attempts} valid attempts!')
			break
		}
	}
```

### Running the Mini-Project

#### 1. Default Demo Sequence
```bash
./vira run docs/handbook/chapters/code/ch02_guessing_game.vir
```

Output:
```text
=== Vira Number Guessing Tutorial ===
Target range: 1 to 100

> Input: "18"
  Attempt #1: 18 -> Too small!

> Input: "not_a_num"
  [Skip] not a valid integer: "not_a_num"

> Input: "75"
  Attempt #2: 75 -> Too big!

> Input: "42"
  Attempt #3: 42 -> Exact match!

You won in 3 valid attempts!
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=20
```

#### 2. Passing Custom Guesses via CLI Arguments
```bash
./vira run docs/handbook/chapters/code/ch02_guessing_game.vir 50 25 42
```

Output:
```text
=== Vira Number Guessing Tutorial ===
Target range: 1 to 100

> Input: "50"
  Attempt #1: 50 -> Too big!

> Input: "25"
  Attempt #2: 25 -> Too small!

> Input: "42"
  Attempt #3: 42 -> Exact match!

You won in 3 valid attempts!
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=22
```

#### 3. Interactive Prompt Mode
```bash
./vira run docs/handbook/chapters/code/ch02_guessing_game.vir --interactive
```

### How Autofree Handles `return`, `break`, and `continue`

Look closely at the `vira check` escape analysis output for `main` in this program:

```bash
./vira check docs/handbook/chapters/code/ch02_guessing_game.vir
```

```text
   [fn main]
     ├── var args         ([]string) : SCOPED_LOCAL (autofree injected)
     ├── var inputs       ([]string) : SCOPED_LOCAL (autofree injected)
     ├── var line         (string  ) : SCOPED_LOCAL (autofree injected)
     ├── var verdict      (string  ) : SCOPED_LOCAL (autofree injected)
     ...
     └── INJECTED FREE: vira_str_free(&line) [string] (before continue)
     └── INJECTED FREE: vira_str_free(&verdict) [string] (before continue)
     └── INJECTED FREE: vira_str_free(&verdict) [string] (before break)
     └── INJECTED FREE: vira_str_free(&verdict) [string] (block scope exit)
     └── INJECTED FREE: slice_autofree(&inputs) [[]string] (function scope exit)
     └── INJECTED FREE: slice_autofree(&args) [[]string] (function scope exit)
```

Notice that when a loop iteration hits `continue` inside an `or { ... }` block or `break` upon winning, Vira's escape analyzer automatically injects `vira_str_free` for variables declared earlier in that loop body (`line`, `verdict`) **right before the jump**, and frees `inputs` and `args` (including every string element inside those slices) when `main` exits.

---

## Summary & Exercises

In this chapter, you built a complete command-line program combining standard library packages (`os`, `io`, `strings`), mutable slices (`[]string`), result error handling (`!int` and `or { ... }`), and deterministic memory management across loop control flow (`break` and `continue`).

1. **Custom Range Validation:** Modify `parse_guess` so that the minimum and maximum bounds are passed as parameters (`fn parse_guess(raw string, min_val int, max_val int) !int`).
2. **Attempt Limit:** Add a `const MAX_ATTEMPTS = 5` constant and terminate the game with a message if the player exceeds the maximum number of valid guesses.
3. **Inspect Loop Cleanup:** Run `./vira cgen docs/handbook/chapters/code/ch02_guessing_game.vir` and examine how `vira_str_free` is emitted before each `continue` and `break` statement.

