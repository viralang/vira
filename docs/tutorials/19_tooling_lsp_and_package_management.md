# Tutorial 19: Developer Tooling, LSP, VS Code Extension & Package Management

A modern language needs a world-class developer experience. The Vira toolchain comes out of the box with zero-dependency built-in developer tools:
1. **Package Management & Project Scaffolding:** `vira init`, `vex.mod`, and multi-file compilation.
2. **Deterministic Code Formatter:** `vira fmt`.
3. **Built-in Unit Testing Framework:** `vira test`.
4. **Language Server Protocol (LSP):** `vira lsp`.
5. **VS Code & Cursor Extension:** Syntax highlighting, snippets, and editor integration.

---

## 1. Project Initialization & `vex.mod`

To scaffold a new Vira project:
```bash
vira init my_service
```

This creates the canonical Vira project structure:
```
my_service/
├── vex.mod          # Package manifest (module, version, license)
├── .gitignore       # Pre-configured ignores for build/ and binaries
├── src/
│   └── main.vir     # Application entry point
└── tests/           # Unit and integration test suite
```

### The `vira.mod` Manifest
The `vira.mod` file defines project metadata and dependencies:
```ini
module my_service
version 0.1.0
license MIT
description High-performance Vira application
```

> [!TIP]
> For a comprehensive guide on building, publishing, versioning, lockfiles (`vira.lock`), and air-gapped vendoring (`vira vendor`), see **[Tutorial 25: Native Package Management, Publishing & Air-Gapped Vendoring](./25_package_management_publishing_and_vendoring.md)**.

---

## 2. Multi-File Compilation & Project Builds

Vira supports compiling entire directories without manual header files or forward declarations:

```bash
# Compile and run immediately
vira run my_service/

# Build standalone optimized production binary
vira build -prod -o bin/my_service my_service/
```

During compilation:
1. Vira scans the directory and parses all `.vir` files into ASTs.
2. The module name is validated against `vex.mod`.
3. All structs, interfaces, functions, and actors are registered in a single unified symbol table.
4. Static escape analysis (Autofree) injects deterministic LIFO destructors across all files.
5. Emits optimized C99 and links into a native standalone executable.

---

## 3. Official Code Formatter: `vira fmt`

Vira enforces clean, consistent style rules using the built-in formatter:

```bash
# Preview formatted output to stdout
vira fmt src/main.vir

# Format file in-place
vira fmt -w src/main.vir

# Format entire project recursively
vira fmt -w .
```

### Formatting Rules
- **Tabs for indentation:** Clean 1-tab indentation for blocks, functions, and control flow.
- **Quadrant alignment:** Struct quadrant labels (`pub:`, `mut:`, `pub mut:`) are formatted with 0 indent, while member fields are indented by 1 tab.
- **Operator spacing:** Ensures consistent spacing around operators (`:=`, `@`, `<-`, `+`, `-`).

---

## 4. Built-in Testing Framework: `vira test`

Vira includes a built-in test runner that discovers and executes tests with automated memory leak detection:

```bash
# Run all tests in a directory
vira test tests/

# Run a specific test file
vira test tests/math_test.vir
```

### Writing Tests in Vex
Any function prefixed with `test_` is treated as a test case:

```vex
module main

fn test_tensor_addition() {
	a := tensor_fill(16, 16, 1.0)
	b := tensor_fill(16, 16, 2.0)
	c := a + b
	if tensor_sum(c) != 768.0 {
		panic('tensor addition mismatch')
	}
	println('test_tensor_addition: ok')
}
```

When `vira test` runs:
1. It automatically synthesizes an isolated test harness.
2. Runs each test function sequentially.
3. Automatically queries the runtime memory counters to verify `active_leaks=0`.
4. Reports pass/fail status and execution time:

```text
=== RUN   tests\math_test.vir
--- PASS: tests\math_test.vir (0.87s)
    active_leaks=0 [Zero Memory Leaks Verified]

Test Summary: 1 passed, 0 failed in 0.87s
```

---

## 5. Language Server Protocol (`vira lsp`)

Vira ships with a native JSON-RPC 2.0 Language Server that communicates over standard input/output (`stdin`/`stdout`):

```bash
vira lsp
```

### Supported LSP Capabilities
- **Real-Time Diagnostics (`textDocument/publishDiagnostics`):** Syntactic errors and semantic type checking are reported as you type.
- **Hover Documentation (`textDocument/hover`):** Hovering over keywords (`actor`, `arena`, `tensor`, `@`, `<-`) displays detailed language semantics and architecture explanations.
- **Document Formatting (`textDocument/formatting`):** Powers "Format on Save" in any modern editor.

---

## 6. VS Code & Cursor Extension

The official Vira VS Code extension is located in [editors/vscode](file:///c:/Users/hp/vex/editors/vscode):

### Files
- [package.json](file:///c:/Users/hp/vex/editors/vscode/package.json): Extension manifest declaring language `vex` and command bindings.
- [language-configuration.json](file:///c:/Users/hp/vex/editors/vscode/language-configuration.json): Bracket matching, comment toggling (`//`), and auto-closing pairs.
- [syntaxes/vex.tmLanguage.json](file:///c:/Users/hp/vex/editors/vscode/syntaxes/vex.tmLanguage.json): TextMate syntax grammar highlighting keywords (`actor`, `arena`, `tensor`), quadrants (`pub:`, `mut:`), operators (`@`, `<-`), and built-ins.

### Installing in VS Code
Copy or symlink the folder to your VS Code extensions directory:
```bash
# Windows
cp -r editors/vscode ~/.vscode/extensions/viralang.vir-0.1.0

# Linux / macOS
cp -r editors/vscode ~/.vscode/extensions/viralang.vir-0.1.0
```
Restart VS Code or Cursor, and all `.vir` files will have full syntax highlighting and formatting support!
