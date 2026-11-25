# Chapter 1: Getting Started

Welcome to your journey with **Vira**. In this first chapter, we will set up the Vira compiler toolchain, write and run our first program, and inspect how Vira compiles your code into a fast, zero-leak native executable—without a Garbage Collector or manual `free()` calls.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch01_hello.vir`](../code/ch01_hello.vir)

---

## 1.1. Installation & C99 Backend Setup

Vira is an **Ahead-Of-Time (AOT)** compiled language. Its compiler front-end performs lexical scanning, type checking, and static escape analysis (`autofree`), and then lowers your program into portable **C99** that is compiled into a standalone native machine binary by your system C compiler (`gcc`, `clang`, or `x86_64-w64-mingw32-gcc`).

### Prerequisites

To build and run Vira programs, you need:
1. **A C99 Compiler:** `gcc` or `clang` on Linux/macOS, or MinGW-w64 / LLVM on Windows.
2. **Go 1.22+ (Only if building the Vira compiler from source).**

#### Installing the C Backend on Linux (Ubuntu / Debian)
```bash
sudo apt-get update
sudo apt-get install -y build-essential libx11-dev libgl1-mesa-dev
```
*(Optional: add `gcc-mingw-w64-x86-64` if you want to cross-compile Windows `.exe` binaries directly from Linux.)*

#### Installing the C Backend on macOS
```bash
xcode-select --install
```

#### Installing the C Backend on Windows
Install [w64devkit](https://github.com/skeeto/w64devkit/releases) or [MSYS2 MinGW-w64](https://www.msys2.org/) so that `gcc.exe` is available in your `PATH`.

---

### Installing or Building `vira`

#### Option A: One-Line Installer
On **Linux & macOS**:
```bash
curl -fsSL https://raw.githubusercontent.com/viralang/vira/main/scripts/install.sh | bash
```

On **Windows (PowerShell)**:
```powershell
irm https://raw.githubusercontent.com/viralang/vira/main/scripts/install.ps1 | iex
```

#### Option B: Building from Source (Maintainer Workflow)
From the root of the Vira repository:
```bash
go build -o vira ./cmd/vira
./vira version
```

---

## 1.2. Hello, World!

Let's write our first Vira program. Open [`docs/handbook/chapters/code/ch01_hello.vir`](../code/ch01_hello.vir):

```v
module main

fn main() {
	language := 'Vira'
	version := '0.7.9'
	println('Hello from ${language} v${version}!')
}
```

### Anatomy of a Vira Program

Let's break down every line of `ch01_hello.vir`:

1. **`module main`**  
   Every Vira file belongs to a module namespace declared at the top of the file. Executable programs always start in `module main`. (If you omit `module main` in a single-file script, the compiler defaults to `module main` automatically.)
2. **`fn main() { ... }`**  
   The `fn` keyword declares a function. `main()` is the entry point where execution begins. Notice there are no semicolons at the end of lines.
3. **`language := 'Vira'`**  
   The `:=` operator declares and initializes a new variable with automatic type inference (`string`). In Vira, **all variables are immutable by default**—once `language` is bound, it cannot be reassigned unless declared with `mut`.
4. **String Literals & Interpolation (`'...'` and `${...}`)**  
   Strings can use single quotes (`'...'`) or double quotes (`"..."`). Inside any string literal, `${expr}` evaluates the expression once and splices its formatted value into a newly allocated string.
5. **`println(...)`**  
   A built-in function that writes a line of UTF-8 text to standard output (`stdout`).

### Running the Program

Use `vira run` to compile and execute the file in one step:

```bash
./vira run docs/handbook/chapters/code/ch01_hello.vir
```

Output:
```text
Hello from Vira v0.7.9!
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=1
```

### Understanding the Runtime Telemetry Footer

When you run a Vira program in development mode (without the `-prod` flag), the runtime prints a two-line diagnostic footer upon exit:
- **`pool_workers=0 | actor_tasks_run=0 | work_steals=0`**: Shows activity on Vira's lock-free actor thread pool. Because this program did not spawn any concurrent actors, the thread pool stayed dormant (`0` workers spun up).
- **`active_leaks=0 | total_freed=1`**: The string interpolation `'Hello from ${language} v${version}!'` allocated **1** heap buffer at runtime (`total_freed=1`). The compiler's `autofree` engine automatically freed that temporary buffer immediately after `println` finished, leaving **`active_leaks=0`**.

---

## 1.3. The Vira CLI Toolchain

As you work through this book, you will use five core CLI commands to inspect, test, and build your code.

### 1. `vira check` — Static Type & Autofree Escape Analysis
Before even compiling to machine code, you can ask Vira to type-check your file and display the **Autofree Escape Analysis Report**:

```bash
./vira check docs/handbook/chapters/code/ch01_hello.vir
```

Output:
```text
✓ docs/handbook/chapters/code/ch01_hello.vir passed syntax, 4-quadrant mutability, and type checks (0 errors)

1. Structs Requiring Generated `_autofree` Destructors:

2. Function Escape Analysis & Injected LIFO Destructors:

   [fn main]
     ├── var language     (string  ) : SCOPED_LOCAL (autofree injected)
     ├── var version      (string  ) : SCOPED_LOCAL (autofree injected)
     ├── var _vira_tmp_str_1 (string  ) : MOVED_TO_CONTAINER (ownership -> struct/iface)
     └── INJECTED FREE: vira_str_free(&_vira_tmp_str_1) [string] (zero-leak temporary string operand cleanup)
     └── INJECTED FREE: vira_str_free(&version) [string] (function scope exit)
     └── INJECTED FREE: vira_str_free(&language) [string] (function scope exit)
```

Notice how transparent Vira's memory model is: you can see the exact temporary variable (`_vira_tmp_str_1`) hoisted for the interpolated string and the exact **Last-In, First-Out (LIFO)** order in which `vira_str_free` is injected.

### 2. `vira cgen` — Inspecting the Generated C99 Code
Nothing in Vira is hidden behind a black-box virtual machine. You can inspect the exact C99 code generated for any `.vir` file:

```bash
./vira cgen docs/handbook/chapters/code/ch01_hello.vir
```

Inside the emitted `main` function, you will see clean, readable C99:
```c
int main(int argc, char** argv) {
    _vira_rt_init_args(argc, argv);
    vira_string language = vira_str_lit("Vira", 4);
    vira_string version = vira_str_lit("0.7.9", 5);
    vira_string _vira_tmp_str_1 = ({
        vira_string _ip_1[5];
        _ip_1[0] = vira_str_lit("Hello from ", 11);
        _ip_1[1] = language;
        _ip_1[2] = vira_str_lit(" v", 2);
        _ip_1[3] = version;
        _ip_1[4] = vira_str_lit("!", 1);
        vira_string _ipr_2 = vira_str_concat_n(_ip_1, 5);
        _ipr_2;
    });
    vira_println(_vira_tmp_str_1);
    vira_str_free(&_vira_tmp_str_1); // autofree: zero-leak temporary string operand cleanup
    vira_str_free(&version);         // autofree: function scope exit
    vira_str_free(&language);        // autofree: function scope exit
    _vira_rt_free_args();
    vira_report_mem_stats();
    return 0;
}
```
> **Note:** Even though `vira_str_free(&language)` is called on a string literal at scope exit, `vira_str_lit` sets `is_lit = true`, so `vira_str_free` is a fast no-op on literals while still freeing if the variable was later reassigned to a heap string.

### 3. `vira build` — Building Standalone Binaries (Dev vs. Production)
To compile an executable binary to disk:

```bash
# Development build (includes runtime leak telemetry)
./vira build -o hello docs/handbook/chapters/code/ch01_hello.vir

# Production build (-O3 -march=native -flto, silent telemetry footer)
./vira build -prod -o hello_prod docs/handbook/chapters/code/ch01_hello.vir

# Sanitizer build (enables AddressSanitizer + UndefinedBehaviorSanitizer)
./vira build -sanitize -o hello_san docs/handbook/chapters/code/ch01_hello.vir
```

When you run `./hello_prod`, the `[vira_rt]` debug telemetry footer is omitted and only your program's output is printed:
```text
Hello from Vira v0.7.9!
```

### 4. `vira fmt` — Canonical Code Formatting
Just like `gofmt` or `rustfmt`, Vira includes a built-in formatter that formats `.vir` files in place to canonical style:

```bash
./vira fmt docs/handbook/chapters/code/ch01_hello.vir
```

---

## Summary & Exercises

Try these hands-on exercises using [`docs/handbook/chapters/code/ch01_hello.vir`](../code/ch01_hello.vir):

1. **Immutability Diagnostics:** Add `language = 'Rust'` on the line after `language := 'Vira'` and run `./vira check docs/handbook/chapters/code/ch01_hello.vir`. Observe the compiler's caret diagnostic and the suggestion `= help: declare with 'mut language := ...'`.
2. **Expression Interpolation:** Interpolate an arithmetic expression directly inside a string literal, such as `'2 + 2 = ${2 + 2}'`, and inspect how `vira cgen` lowers it to C99.
3. **Production vs. Debug Binaries:** Compile `ch01_hello.vir` with `./vira build -prod -o hello_prod` and compare its output and binary size against a standard debug build.

