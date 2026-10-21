# Tutorial 25: Native Package Management, Publishing & Air-Gapped Vendoring

> **Difficulty:** Intermediate to Advanced  
> **Matching Code Example:** [`docs/examples/24_package_management_consumer.vir`](../examples/24_package_management_consumer.vir)  
> **Reference Package Template:** [`C:\Users\hp\vira_test`](file:///C:/Users/hp/vira_test)

---

## 1. Overview & Architecture

Modern software engineering relies on modularity, code reuse, and reliable dependency management. Vira provides a **native, decentralized, Git-first package management system** built directly into the compiler toolchain without requiring external tools, Python scripts, or centralized package registries.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           VIRA PACKAGE ECOSYSTEM                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   DEVELOPER PACKAGE                  VIRA TOOLCHAIN               CONSUMER APP   │
│  ┌─────────────────┐               ┌────────────────┐          ┌──────────────┐  │
│  │ vira.mod        │               │ vira add       │          │ vira.mod     │  │
│  │ src/            │──► Git Push ──│ vira install   │──► Fetch ─│ vira.lock    │  │
│  │   vec2.vir      │    (GitHub/   │ vira remove    │    &     │ .vira/pkg/   │  │
│  │   logger.vir    │     GitLab)   │ vira vendor    │   Verify │ vendor/      │  │
│  │ tests/          │               └────────────────┘          │ src/main.vir │  │
│  └─────────────────┘                        │                  └──────────────┘  │
│                                             ▼                                    │
│                              Deterministic SHA-256 Checksums                     │
│                              Zero-Drift Offline Reproducibility                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles
1. **Decentralized & Git-Native:** Packages are standard Git repositories (GitHub, GitLab, Gitea, or local folders). No centralized gatekeeper or proprietary registry required.
2. **Cryptographic Integrity (`vira.lock`):** Every dependency's directory tree is hashed into a deterministic SHA-256 tree hash (`sha256:<hex>`). Any upstream tampering or man-in-the-middle alteration triggers an immediate compile-time abort.
3. **Air-Gapped Offline Vendoring (`vira vendor`):** Packages can be frozen directly into `<project>/vendor/` for high-security, air-gapped, or offline production deployments.
4. **Hierarchical Import Resolution:** The compiler and Language Server Protocol (LSP) resolve imports using a deterministic 6-tier search order across local files, project sources, vendored packages, project cache, and global user cache.

---

## 2. Anatomy of a Vira Package

A Vira package is simply a directory containing a `vira.mod` manifest and one or more `.vir` source files.

### Standard Package Directory Layout

```text
my_package/
├── vira.mod                 # Package manifest metadata & dependencies
├── .gitignore               # Ignored build artifacts & temp files
├── README.md                # Documentation & code examples
├── src/
│   ├── vec2.vir             # Domain structs & mathematical operations
│   ├── mathx.vir            # Pure utility functions
│   └── logger.vir           # Tagged console logging utility
└── tests/
    └── pkg_test.vir         # Automated unit & integration tests
```

### The `vira.mod` Manifest

The `vira.mod` manifest declares the module's identity, version, and optional dependencies:

```ini
module vira_test
version 0.1.0
license MIT
description Useful 2D vector geometry, clamp utilities, and structured logging for Vira
author Mohamed Alaaeldin

# Optional dependencies required by this package
deps
	github.com/viralang/http v0.2.0
	github.com/viralang/json v0.1.0
```

#### Manifest Fields:
- `module`: The canonical package identifier. Must be unique, alphanumeric (with underscores), and match the `module <name>` declarations in your source files.
- `version`: Semantic version (`MAJOR.MINOR.PATCH`).
- `license`: Software license (e.g. `MIT`, `Apache-2.0`, `BSD-3-Clause`).
- `author`: Package maintainer or organization.
- `description`: A concise summary of package capabilities.
- `deps`: Multi-line block declaring external dependencies.

---

## 3. Writing Exportable Code & 4-Quadrant Encapsulation

Symbols in a package are private to the module by default. To make structs, methods, and functions available to consumers, declare them with `pub`.

### Exporting Structs & Methods (`src/vec2.vir`)

```vira
module vira_test

// Vec2 represents a 2-dimensional floating-point vector.
// Fields in `pub:` are readable by external packages.
pub struct Vec2 {
pub:
	x f64
	y f64
}

// Static factory constructor
pub fn Vec2.new(x f64, y f64) Vec2 {
	return Vec2{
		x: x,
		y: y
	}
}

// Vector addition method
pub fn (v Vec2) add(other Vec2) Vec2 {
	return Vec2{
		x: v.x + other.x,
		y: v.y + other.y
	}
}

// Vector subtraction method
pub fn (v Vec2) sub(other Vec2) Vec2 {
	return Vec2{
		x: v.x - other.x,
		y: v.y - other.y
	}
}

// Scalar scaling method
pub fn (v Vec2) scale(factor f64) Vec2 {
	return Vec2{
		x: v.x * factor,
		y: v.y * factor
	}
}

// Scalar dot product
pub fn (v Vec2) dot(other Vec2) f64 {
	return (v.x * other.x) + (v.y * other.y)
}
```

### Exporting Functions (`src/mathx.vir`)

```vira
module vira_test

// clamp restricts val to the [min_val, max_val] interval.
pub fn clamp(val int, min_val int, max_val int) int {
	if val < min_val {
		return min_val
	}
	if val > max_val {
		return max_val
	}
	return val
}

// min_int returns the smaller of two integers.
pub fn min_int(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

// max_int returns the larger of two integers.
pub fn max_int(a int, b int) int {
	if a > b {
		return a
	}
	return b
}
```

### Exporting State with Encapsulation (`src/logger.vir`)

```vira
module vira_test

// Logger provides structured prefix-tagged logging.
// `tag` is public read-only; `counter` is public mutable.
pub struct Logger {
pub:
	tag string
pub mut:
	counter int
}

pub fn Logger.new(tag string) Logger {
	return Logger{
		tag: tag,
		counter: 0
	}
}

pub fn (mut l Logger) info(msg string) {
	l.counter = l.counter + 1
	println('[INFO] [${l.tag}] #${l.counter}: ${msg}')
}

pub fn (mut l Logger) ok(msg string) {
	l.counter = l.counter + 1
	println('[ OK ] [${l.tag}] #${l.counter}: ${msg}')
}

pub fn (mut l Logger) warn(msg string) {
	l.counter = l.counter + 1
	println('[WARN] [${l.tag}] #${l.counter}: ${msg}')
}

pub fn (mut l Logger) error(msg string) {
	l.counter = l.counter + 1
	println('[ERR ] [${l.tag}] #${l.counter}: ${msg}')
}
```

---

## 4. Testing Your Package

Every production package should provide automated unit tests in the `tests/` directory.

### Writing Tests (`tests/pkg_test.vir`)

Any function starting with `test_` is automatically discovered and executed by `vira test`:

```vira
module vira_test

fn test_vec2_operations() {
	v1 := Vec2.new(3.0, 4.0)
	v2 := Vec2.new(1.0, 2.0)

	sum := v1.add(v2)
	if sum.x != 4.0 || sum.y != 6.0 {
		panic('Vec2.add failed')
	}

	diff := v1.sub(v2)
	if diff.x != 2.0 || diff.y != 2.0 {
		panic('Vec2.sub failed')
	}

	scaled := v2.scale(3.0)
	if scaled.x != 3.0 || scaled.y != 6.0 {
		panic('Vec2.scale failed')
	}

	dp := v1.dot(v2)
	if dp != 11.0 {
		panic('Vec2.dot failed')
	}

	println('✓ test_vec2_operations passed')
}

fn test_mathx_helpers() {
	if clamp(5, 0, 10) != 5 {
		panic('clamp inside failed')
	}
	if clamp(-3, 0, 10) != 0 {
		panic('clamp below failed')
	}
	if clamp(15, 0, 10) != 10 {
		panic('clamp above failed')
	}

	if min_int(10, 20) != 10 {
		panic('min_int failed')
	}
	if max_int(10, 20) != 20 {
		panic('max_int failed')
	}

	println('✓ test_mathx_helpers passed')
}

fn test_logger() {
	mut log := Logger.new('TestTag')
	log.info('Running pkg tests')
	log.ok('All checks operational')
	if log.counter != 2 {
		panic('logger counter mismatch')
	}
	println('✓ test_logger passed')
}
```

### Running Tests with Memory Leak Detection

Execute the test suite from your package root:

```bash
vira test tests/
```

**Output:**
```text
=== RUN   tests\pkg_test.vir
[INFO] [TestTag] #1: Running pkg tests
[ OK ] [TestTag] #2: All checks operational
✓ test_vec2_operations passed
✓ test_mathx_helpers passed
✓ test_logger passed
--- PASS: tests\pkg_test.vir (1.44s)
    active_leaks=0 [Zero Memory Leaks Verified]

Test Summary: 1 passed, 0 failed in 1.44s
```

> [!TIP]
> Notice the `active_leaks=0` line. The Vira test runner automatically instruments the binary with allocation tracking to guarantee your package has zero memory leaks before publishing.

---

## 5. Publishing Packages to Git & GitHub

Publishing a package in Vira does **not** require uploading to a centralized registry. Simply publish your repository to Git.

### Step 1: Initialize Git & Commit
Inside your package directory (`C:\Users\hp\vira_test`):
```bash
cd C:\Users\hp\vira_test
git init
git add .
git commit -m "feat: initial release of vira_test package"
```

### Step 2: Push to GitHub / GitLab
Create a repository (e.g. `vira_test`) on GitHub and push:
```bash
git branch -M main
git remote add origin https://github.com/<your-username>/vira_test.git
git push -u origin main
```

### Step 3: Semantic Version Tagging
To release versioned snapshots, tag your repository using Git tags:
```bash
# Tag a release
git tag v0.1.0
git push origin v0.1.0
```

Consumers can now install exact versions using `@v0.1.0`!

---

## 6. Consuming Packages with the Vira CLI

### 1. `vira add`: Adding Dependencies

You can add dependencies either from **GitHub** or from a **local directory**:

```bash
# From a GitHub repository (latest default branch)
vira add github.com/<username>/vira_test

# From a specific GitHub release tag
vira add github.com/<username>/vira_test@v0.1.0

# From a local directory path (ideal for monorepos or offline development)
vira add C:/Users/hp/vira_test
```

**What `vira add` does automatically:**
1. Clones or copies the package into `.vira/pkg/<target>/`.
2. Computes the cryptographic SHA-256 tree hash.
3. Appends the dependency to `vira.mod`.
4. Updates `vira.lock` with the exact commit hash and tree checksum.

```text
Fetching package C:\Users\hp\vira_test (latest)...
✓ Added dependency C:\Users\hp\vira_test (latest)
  checksum: sha256:db77227fd38302e0564541098036456188b0a1e40217890477e34bb91850b97d
```

### 2. Using Imported Packages in Code

Once added, use the standard `import` statement in any `.vir` file:

```vira
module my_app

import "vira_test"

fn main() {
    // 1. Qualified package namespace instantiation
    mut log := vira_test.Logger.new('MyApp')
    log.info('Application started successfully with package namespacing!')

    // 2. Qualified struct static factory
    v1 := vira_test.Vec2.new(10.0, 20.0)
    v2 := vira_test.Vec2.new(5.0, 15.0)
    sum := v1.add(v2)
    scaled := sum.scale(2.0)

    log.ok('Vector calculation: scaled=(${scaled.x}, ${scaled.y})')

    // 3. Qualified top-level utility function
    speed := vira_test.clamp(150, 0, 100)
    log.info('Bounded speed: ${speed}')
}
```

Run immediately:
```bash
vira run .
```

**Output:**
```text
[INFO] [MyApp] #1: Application started successfully with package namespacing!
[ OK ] [MyApp] #2: Vector calculation: scaled=(30.00, 70.00)
[INFO] [MyApp] #3: Bounded speed: 100
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
```

---

## 7. Deterministic Builds & `vira.lock`

Every project with dependencies generates a `vira.lock` file:

```json
{
  "version": 1,
  "packages": [
    {
      "path": "github.com/viralang/vira_test",
      "version": "v0.1.0",
      "commit": "05483a31c586b3ecf124...",
      "checksum": "sha256:db77227fd38302e0564541098036456188b0a1e40217890477e34bb91850b97d"
    }
  ]
}
```

> [!IMPORTANT]
> **Commit `vira.lock` to version control!**  
> Committing `vira.lock` guarantees that every member of your team and your CI/CD pipelines will compile against the **exact same code bytes**, preventing unexpected upstream changes or dependency poisoning.

### Restoring Dependencies: `vira install`

When cloning a repository that already contains `vira.mod` and `vira.lock`:

```bash
# Clean clone setup
git clone https://github.com/myorg/my_app.git
cd my_app

# Download and verify all locked dependencies
vira install
```

`vira install` (or `vira get`) reads `vira.lock`, downloads any missing packages, and verifies that the directory tree matches the stored SHA-256 checksum.

### Removing Dependencies: `vira remove`

To cleanly remove a dependency from `vira.mod`, `vira.lock`, and the `.vira/pkg` cache:

```bash
vira remove github.com/viralang/vira_test
```

---

## 8. Air-Gapped & Offline Vendoring (`vira vendor`)

In mission-critical enterprise environments, financial systems, or defense systems, production builds must run **completely disconnected from the internet (air-gapped)**.

Vira natively supports zero-dependency offline vendoring:

```bash
vira vendor
```

**Output:**
```text
✓ Vendored 1 package(s) into ./vendor
```

### How Vendoring Works:
- Copies all dependencies directly into `<projectRoot>/vendor/`.
- Even if `.vira/` is completely deleted and internet access is disabled, `vira run .` and `vira build -prod .` will compile directly from `./vendor`.

```bash
# Verify offline execution without .vira/ cache
Remove-Item -Recurse -Force .vira

# Compiles directly from ./vendor with zero network requests
vira run .
```

---

## 9. Import Candidate Search Precedence

When you write `import "my_package"`, the compiler and LSP search the filesystem using this exact priority order:

| Precedence | Search Location | Description |
| :-: | :--- | :--- |
| **1** | `<current_file_dir>/<target>` | Companion files in the same directory |
| **2** | `<project_root>/src/<target>` | Internal application subpackages |
| **3** | `<project_root>/<target>` | Project root modules |
| **4** | `<project_root>/vendor/<target>` | Vendored air-gapped dependencies |
| **5** | `<project_root>/.vira/pkg/<target>` | Project-local cached dependencies |
| **6** | `~/.vir/pkg/<target>` | Global toolchain user cache |

This ensures that local code always takes precedence over cached packages, and vendored packages take precedence over remote network caches.

---

## 10. Summary Cheat Sheet

| Task | Command |
| :--- | :--- |
| **Initialize new package** | `vira init my_lib` |
| **Add Git dependency** | `vira add github.com/<user>/<repo>[@tag]` |
| **Add local dependency** | `vira add /path/to/local/pkg` |
| **Install/verify all deps** | `vira install` (or `vira get`) |
| **Remove a dependency** | `vira remove <package>` |
| **Vendor all dependencies** | `vira vendor` |
| **Run package unit tests** | `vira test tests/` |
| **Build production binary** | `vira build -prod -o bin/app .` |
