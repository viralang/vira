# Tutorial 16: Compiling, Building & Deploying Vex Applications

> **Difficulty:** Beginner to Advanced  
> **Matching Code Example:** [`docs/examples/16_compilation_and_build_modes.vex`](../examples/16_compilation_and_build_modes.vex)

---

## 1. The Vex Compilation Architecture

Vex employs a **multi-stage Ahead-Of-Time (AOT) compilation pipeline**. Rather than running an interpreted virtual machine or a stop-the-world garbage-collected runtime, Vex translates high-level code with strict type guarantees into **pure, standards-compliant C99**, which is then compiled down to native machine code by industrial C backends (GCC, Clang, or TinyCC).

```
+───────────────────────────────────────────────────────────────────────────────+
│                         VEX COMPILATION PIPELINE                              │
│                                                                               │
│  [1] Source Code (.vex)                                                       │
│          │                                                                    │
│          ▼                                                                    │
│  [2] Lexer & Recursive-Descent Parser                                         │
│          │                                                                    │
│          ▼                                                                    │
│  [3] Abstract Syntax Tree (AST)                                               │
│          │                                                                    │
│          ▼                                                                    │
│  [4] 4-Quadrant Mutability & Static Type Checker                              │
│          │                                                                    │
│          ▼                                                                    │
│  [5] Autofree Static Escape Analysis (LIFO Destructors Injected)              │
│          │                                                                    │
│          ▼                                                                    │
│  [6] C99 Generator (runtime/vex_rt.h: SIMD, Actors, Arenas)                   │
│          │                                                                    │
│          ▼                                                                    │
│  [7] Native C Backend Compiler (GCC / Clang / TinyCC)                         │
│          │                                                                    │
│          ▼                                                                    │
│  [8] Standalone Native Executable Binary (.exe / ELF)                         │
│      ├── Zero Runtime VM / Bytecode                                           │
│      ├── Zero Garbage Collector                                               │
│      └── Statically Linked (-static -pthread)                                 │
+───────────────────────────────────────────────────────────────────────────────+
```

---

## 2. The 4 Primary CLI Commands

The `vex` command-line tool provides four specialized subcommands for development, building, auditing, and inspecting code:

```
vex <command> [flags] <file.vex>
```

### 1. `vex run`: Rapid Development & Execution
Compiles the application into a temporary sandbox and executes it immediately, forwarding all command-line arguments to the application:

```powershell
# Run a file in development mode
vex run docs/examples/16_compilation_and_build_modes.vex

# Run in production mode with -O3 optimizations
vex run -prod docs/examples/16_compilation_and_build_modes.vex
```

### 2. `vex build`: Ahead-Of-Time Standalone Compilation
Produces a permanent, self-contained native executable binary:

```powershell
# Build development binary (defaults to <name>.exe)
vex build docs/examples/16_compilation_and_build_modes.vex

# Build with custom output binary name
vex build -o server.exe docs/examples/16_compilation_and_build_modes.vex

# Build production binary with -O3 and vectorization
vex build -prod -o server_prod.exe docs/examples/16_compilation_and_build_modes.vex
```

### 3. `vex check`: Sub-Millisecond Type & Memory Audit
Executes semantic validation and static escape analysis without invoking the C compiler. Ideal for IDE language servers, pre-commit hooks, and CI linting:

```powershell
vex check docs/examples/16_compilation_and_build_modes.vex
```

**Sample Output:**
```
✓ docs/examples/16_compilation_and_build_modes.vex passed syntax, 4-quadrant mutability, and type checks (0 errors)

1. Structs Requiring Generated `_autofree` Destructors:
   ├── struct ServiceConfig -> requires `ServiceConfig_autofree(&self)` (owns heap fields)

2. Function Escape Analysis & Injected LIFO Destructors:
   [fn compute_workload]
     ├── var inputs       (tensor[f32]) : ARENA_MANAGED (O(1) slab free)
     ├── var weights      (tensor[f32]) : ARENA_MANAGED (O(1) slab free)
     ├── var res          (tensor[f32]) : ARENA_MANAGED (O(1) slab free)
     └── INJECTED FREE: vex_arena_destroy(&_vex_arena_1) [vex_arena] (arena block exit)
```

### 4. `vex cgen`: C99 Code Inspection
Emits the generated C99 translation unit to stdout or a file. Useful for auditing generated code, verifying SIMD intrinsics, or embedding Vex modules into C/C++ projects:

```powershell
# Inspect C99 code on stdout
vex cgen docs/examples/16_compilation_and_build_modes.vex

# Save C99 code directly to a file
vex cgen -o build/service.c docs/examples/16_compilation_and_build_modes.vex
```

---

## 3. Build Flags & Optimization Profiles

| Flag | Purpose | Backend Flags Injected |
| :--- | :--- | :--- |
| *(default)* | Development build | `-std=c99 -O2 -march=x86-64-v3 -mstackrealign -static -pthread` |
| `-prod` | Production build | `-std=c99 -O3 -march=x86-64-v3 -mstackrealign -static -pthread` |
| `-sanitize` | Memory leak & UB debugging | `-std=c99 -O1 -g -fsanitize=address,undefined -mstackrealign -pthread` |
| `-keep-c` | Keep generated C99 unit | Saves intermediate C translation unit into `build/<name>.c` |
| `-cc <name>` | Custom C compiler | Overrides compiler selection (`gcc`, `clang`, `tcc`, `cc`) |
| `-o <path>` | Custom binary destination | Specifies target executable path (e.g., `-o dist/app.exe`) |

### Development Mode vs. Production Mode
- **Development (`default`):** Compiles in under 1 second with `-O2`. Enables function-level symbols for fast debugging while still compiling hardware AVX2/NEON SIMD math.
- **Production (`-prod`):** Applies `-O3` aggressive loop unrolling, vector register pipelining, and maximum compiler optimization for high-throughput AI services.

### Sanitizer Mode (`-sanitize`)
When hunting memory corruption, buffer overflows, or undefined behavior, the `-sanitize` flag compiles the program with LLVM/GCC **AddressSanitizer (ASan)** and **UndefinedBehaviorSanitizer (UBSan)**:

```powershell
vex build -sanitize -o app_sanitized.exe docs/examples/16_compilation_and_build_modes.vex
.\app_sanitized.exe
```

---

## 4. Selecting Backend Compilers (`-cc`)

Vex automatically detects any available C compiler in your `PATH` or standard installation locations:

```
Priority Search Order:
  1. User flag: -cc <compiler>
  2. gcc       (MinGW-w64 / w64devkit)
  3. clang     (LLVM)
  4. cc        (Standard system compiler)
  5. tcc       (TinyCC instant build)
```

### 1. MinGW GCC (Default on Windows)
- Toolchain: `w64devkit` or MSYS2.
- Produces self-contained, statically linked `.exe` files with `-static -pthread`.
- Full 256-bit AVX2 + FMA hardware vectorization.

### 2. Clang / LLVM (`-cc clang`)
- Excellent for cross-compiling or integrating with LLVM toolchains:
  ```powershell
  vex build -cc clang -o app.exe main.vex
  ```

### 3. TinyCC (`-cc tcc`)
- For instant compilation speed (compiles in **< 15 milliseconds**):
  ```powershell
  vex build -cc tcc -o app.exe main.vex
  ```

---

## 5. Static Linking & Self-Contained Binaries

Unlike Python (requires Python runtime + NumPy DLLs) or Go (large runtime GC threads), Vex binaries are **completely standalone**:

1. **No External DLL Dependencies:** Compiled with `-static`, bundling pthread and standard C libraries directly into the executable.
2. **No Runtime Overhead:** Execution starts directly at OS `main()`. There is no JIT compilation, no class loading, and no bytecode interpretation.
3. **Deterministic Memory Cleanup:** The binary contains no garbage collector thread. All memory cleanup happens deterministically via injected static LIFO frees.

---

## 6. Containerization & Production Docker Deployment

Because Vex binaries are statically linked, they can run inside a **distroless** or minimal **Scratch / Alpine** Docker image with an image size under **15 MB**.

### Multi-Stage Production Dockerfile

```dockerfile
# Stage 1: Build Environment
FROM golang:1.22-alpine AS builder

# Install GCC, Make, and Git
RUN apk add --no-cache gcc musl-dev git

WORKDIR /app
COPY . .

# Build Vex compiler
RUN go build -o /usr/local/bin/vex ./cmd/vex

# Compile Vex application in production mode
RUN vex build -prod -o /app/server docs/examples/16_compilation_and_build_modes.vex

# Stage 2: Minimal Distroless Runtime Image
FROM alpine:3.19

# Copy the standalone statically linked binary
COPY --from=builder /app/server /usr/local/bin/server

# Expose port (if web server)
EXPOSE 8080

# Run standalone binary directly
ENTRYPOINT ["/usr/local/bin/server"]
```

---

## 7. Step-by-Step Hands-on Walkthrough

Let's compile and run the companion example [`docs/examples/16_compilation_and_build_modes.vex`](file:///c:/Users/hp/vex/docs/examples/16_compilation_and_build_modes.vex):

### Step 1: Type & Memory Audit
```powershell
.\vex.exe check docs/examples/16_compilation_and_build_modes.vex
```

### Step 2: Compile Production Binary
```powershell
.\vex.exe build -prod -o prod_service.exe docs/examples/16_compilation_and_build_modes.vex
```

### Step 3: Execute Production Binary
```powershell
.\prod_service.exe
```

### Output:
```
=================================================================
   VEX COMPILATION, BUILD MODES & DEPLOYMENT TELEMETRY           
=================================================================
Service Identity:
  Name            : VexProductionService
  Environment     : production
  Worker Threads  : 4
  Batch Capacity  : 1024
--- 1. EXECUTING BENCHMARK WORKLOAD ---
Running 50 consecutive tensor activations...
Workload Output Score : 204800.00
--- 2. RUNTIME DEPLOYMENT PROFILE ---
Binary Features:
  * Native Machine Code (No Virtual Machine, No Bytecode)
  * Zero Garbage Collector Pauses (Deterministic Autofree)
  * Self-Contained Statically Linked Binary
  * Hardware SIMD Vector Acceleration (AVX2 / ARM NEON)
=================================================================
   BUILD COMPLETE                                                
=================================================================
[vex_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
[vex_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vex_rt] active_leaks=0 | total_freed=6
```

---

## Summary Checklist for Production Builds

- [x] Run `vex check <file.vex>` to ensure zero type errors and verify escape analysis.
- [x] Compile with `vex build -prod -o <binary> <file.vex>` for `-O3` optimizations.
- [x] Validate zero memory leaks in telemetry (`active_leaks=0`).
- [x] Package into minimal Docker scratch/alpine containers without runtime dependencies.
