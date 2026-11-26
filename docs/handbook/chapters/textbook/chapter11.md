# Chapter 11: Modules, Packages & Dependency Management

As a Vira codebase grows beyond a single file, clean module boundaries and reproducible dependency management keep builds predictable and fast. Vira organizes code using explicit `module` declarations, `pub` visibility controls, scoped `import` namespaces, a 6-tier import resolver, and decentralized Git/local package management backed by SHA-256 lockfiles (`vira.lock`) and offline vendoring (`vira vendor`).

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch11_modules_and_packages.vir`](../code/ch11_modules_and_packages.vir)

---

## 11.1. Modules (`module`) and Cross-Module Visibility (`pub`)

Every `.vir` source file begins with a `module` declaration (defaulting to `module main` for standalone entry points):

```v
module main
```

By default, **all declarations in a module are private to that module**. Constants, functions, structs, enums, interfaces, and actors are only visible to importing packages when explicitly marked `pub`:

```v
pub const DEFAULT_TIMEOUT_MS = 250   // Visible to importers as <mod>.DEFAULT_TIMEOUT_MS
const INTERNAL_SCALE = 2             // Private to this module

pub struct ServiceEndpoint {
	secret_token string              // Private immutable field (only this module)
pub:
	name string                      // Public read-only field
	port int
pub mut:
	timeout time.Duration            // Public read/write field
}

pub fn ServiceEndpoint.new(name string, port int, timeout_ms i64) ServiceEndpoint {
	...
}

fn format_metric(tag string, val f64) string {
	...                              // Module-private helper
}
```

When a project has multiple `.vir` files sharing the same `module` name (such as `src/main.vir`, `src/config/settings.vir`, and `src/models/data.vir`), Vira compiles them together into one cohesive module namespace while keeping `tests/` excluded from normal `run`/`build` commands.

---

## 11.2. Importing Packages (`import`, `import ... as ...`) and 6-Tier Import Resolution

External modules and standard library packages are loaded using `import` at the top of a file (after `module`). Every imported symbol must be accessed through its module qualifier—preventing silent name collisions across packages:

```v
import math
import strings as str
import time
```

- **Standard Import (`import math`):** Qualifies symbols as `math.PI`, `math.hypot(...)`, or `time.Duration.from_millis(...)`.
- **Aliased Import (`import strings as str`):** Rebinds the local namespace qualifier so you write `str.join(...)` and `str.to_upper(...)` without changing the underlying package.
- **Quoted & Sub-Package Paths (`import "mypkg"` / `import mypkg/sub`):** Imports local directories, nested sub-packages, or external dependencies whose names contain paths.

### The 6-Tier Import Resolution Order

When the compiler encounters `import <target>`, `ResolveImportCandidates` searches paths in strict priority order:

| Tier | Search Location | Purpose |
| :--- | :--- | :--- |
| **1** | **Current File Directory** (`<file_dir>/<target>`) | Co-located helper files or local sub-modules next to the importing file. |
| **2** | **Project `src/` Directory** (`<root>/src/<target>`) | Internal packages inside a `vira.mod` project. |
| **3** | **Project Root Directory** (`<root>/<target>`) | Root-level package folders or `.vir` files. |
| **4** | **Vendored Dependencies** (`./vendor/<pkg>`) | Air-gapped, self-contained copies created by `vira vendor`. |
| **5** | **Local Project Cache** (`./.vira/pkg/<pkg>`) | Dependencies fetched by `vira add` / `vira install` (matching declared `module` names in `vira.mod` / `vira.lock` first, then path/URL basenames, including sub-packages). |
| **6** | **Standard Library & Global Cache** (`packages/` and `~/.vir/pkg/`) | Built-in standard library modules (`math`, `strings`, `time`, `io`, `os`, `net`, `http`, `json`, `flag`, `log`, `exec`, `syscall`, `ctype`, `eventbus`, `viraserve`) followed by the global user cache. |

Because `./vendor/` is checked before `./.vira/pkg/`, vendored dependencies automatically override network caches for deterministic offline builds.

---

## 11.3. Multi-Package Project Layout (`vira.mod`, `src/`, `tests/`)

Running `vira init my_service` scaffolds a standard multi-directory Vira project:

```text
my_service/
├── vira.mod               # Project manifest
├── .gitignore             # Ignores build/, bin/, .vira/, out.c
├── README.md
├── src/
│   ├── main.vir           # Application entry point
│   ├── config/
│   │   └── settings.vir   # Environment & app configuration
│   ├── models/
│   │   └── data.vir       # 4-quadrant encapsulated domain models
│   └── services/
│       └── compute.vir    # Business logic & services
└── tests/
    └── app_test.vir       # Unit & integration test suite
```

### The `vira.mod` Manifest Format

`vira.mod` uses a clean, human-readable key/value format with an optional indented `deps` block:

```text
module my_service
version 0.1.0
license MIT
description Modern scalable Vira application

deps
	github.com/viralang/odbc v1.2.0
	./libs/shared_utils
```

Running `vira run .` or `vira build -prod .` from inside the project root discovers `vira.mod`, compiles all `.vir` source files in the project (automatically skipping `tests/`, `vendor/`, `.vira/`, `build/`, and nested sub-modules that carry their own `vira.mod`), and links imported packages cleanly.

---

## 11.4. Decentralized Dependencies, Lockfiles (`vira.lock`), and Air-Gapped Vendoring (`vira vendor`)

Vira has no central package registry single point of failure. Dependencies resolve directly from **Git repositories** (with optional tag/branch `@version` and monorepo subpath extraction like `github.com/org/repo/subpkg`) or **local filesystem directories** (`./libs/shared_utils`, `../common`).

| Command | Behavior |
| :--- | :--- |
| **`vira add <pkg[@ver]>`** | Fetches the package (and its transitive dependencies), detects version conflicts, updates `vira.mod`, and records the exact commit and SHA-256 checksum in `vira.lock`. |
| **`vira install`** (alias `get`) | Reads `vira.mod` and `vira.lock`, restores any missing dependencies into `./.vira/pkg/`, fetches the exact locked Git commit, and verifies the directory checksum against `vira.lock`. |
| **`vira remove <pkg>`** | Removes the dependency from `vira.mod`, `vira.lock`, and `./.vira/pkg/`. |
| **`vira vendor`** | Ensures all direct and transitive dependencies are installed and verified, then copies them into `./vendor/` for offline and air-gapped CI/CD builds. |

### Exact Reproducibility with `vira.lock`

When a dependency is added or installed, Vira computes a deterministic SHA-256 digest (`ComputeDirChecksum`) across every relative path and file byte inside the package directory and writes it to `vira.lock`:

```json
{
  "version": 1,
  "packages": [
    {
      "path": "github.com/viralang/odbc",
      "version": "v1.2.0",
      "commit": "a1b2c3d4e5f6...",
      "checksum": "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
      "module": "odbc"
    }
  ]
}
```

If a remote tag is force-pushed or tampered with so its content checksum no longer matches `vira.lock`, `vira install` rejects the download and removes the corrupted directory rather than silently building unverified code.

---

## 11.5. Running the Complete Chapter 11 Example

Run [`docs/handbook/chapters/code/ch11_modules_and_packages.vir`](../code/ch11_modules_and_packages.vir):

```bash
./vira run docs/handbook/chapters/code/ch11_modules_and_packages.vir
```

Output:
```text
[1] Stdlib: joined=vira/pkg/layout, hypot(3,4)=5.00, PI=3.14
[2] Endpoint: GATEWAY:8443 (500ms)
[3] Updated: GATEWAY:8443 (2000ms), metric=norm=005
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=17
```

---

## Summary & Exercises

1. **Module Aliasing:** Add `import json as j` to `ch11_modules_and_packages.vir`, call `j.is_valid('{"ok":true}')`, and verify the output with `./vira run`.
2. **Scaffold a Multi-Package Project:** In a temporary directory (such as `/tmp/demo_pkg`), run `vira init demo_pkg`, inspect the generated `vira.mod`, `src/`, and `tests/` files, and run `vira run .` followed by `vira test tests/`.
3. **Local Dependency & Vendoring:** Create a second local library package in `/tmp/demo_lib` with `pub fn greet(who string) string`, add it to `/tmp/demo_pkg` via `vira add ../demo_lib`, inspect the generated `vira.lock`, and run `vira vendor` to populate `./vendor/demo_lib`.
