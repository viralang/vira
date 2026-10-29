# The Vira Programming Language: Complete Language Guide & Reference

> **Version:** Production Toolchain (v0.5.13)  
> **Status:** Production-Ready Toolchain & Runtime  
> **Target:** Native Executables via Optimized C99 Backend (GCC / Clang / MinGW-w64)

---

## Table of Contents

1. [Introduction & Core Philosophy](#1-introduction--core-philosophy)
2. [Toolchain & CLI Reference](#2-toolchain--cli-reference)
3. [Language Basics](#3-language-basics)
   - [Modules](#modules)
   - [Variables & Immutability](#variables--immutability)
   - [Blank Identifier (`_`)](#blank-identifier-_)
   - [Primitive Types](#primitive-types)
   - [String Interpolation](#string-interpolation)
4. [4-Quadrant Struct Encapsulation & OOP](#4-4-quadrant-struct-encapsulation--oop)
   - [Access Quadrants](#access-quadrants)
   - [Struct Embedding & Composition](#struct-embedding--composition)
   - [Methods (Value vs Mutable Receivers)](#methods-value-vs-mutable-receivers)
   - [Static Factory Methods](#static-factory-methods)
5. [Structural Interfaces & Dynamic Dispatch](#5-structural-interfaces--dynamic-dispatch)
   - [Interface Definitions](#interface-definitions)
   - [Implicit Satisfaction](#implicit-satisfaction)
   - [Fat-Pointer Boxing & Dispatch](#fat-pointer-boxing--dispatch)
6. [Error Handling: `!T` Result & `?T` Option](#6-error-handling-t-result--t-option)
   - [Result Types (`!T`)](#result-types-t)
   - [The `or { ... }` Unwrap Block](#the-or----block)
   - [Compiler Enforcement](#compiler-enforcement)
7. [Memory Management: Deterministic Autofree & Arenas](#7-memory-management-deterministic-autofree--arenas)
   - [How Autofree Works (No GC)](#how-autofree-works-no-gc)
   - [Static Escape Analysis Classification](#static-escape-analysis-classification)
   - [Bump Arenas (`arena { ... }`)](#bump-arenas-arena----)
8. [First-Class AI Tensors & SIMD Acceleration](#8-first-class-ai-tensors--simd-acceleration)
   - [Tensor Types & Construction](#tensor-types--construction)
   - [Matrix Multiplication (`@`)](#matrix-multiplication-)
   - [SIMD Hardware Backends (AVX2 / NEON)](#simd-hardware-backends-avx2--neon)
9. [Actor Concurrency & Fault Supervision](#9-actor-concurrency--fault-supervision)
   - [Actor Definition](#actor-definition)
   - [Message Passing (`<-`) & Futures](#message-passing----futures)
   - [Lock-Free Work-Stealing Pool](#lock-free-work-stealing-pool)
   - [Erlang-Style Fault Boundaries (`panic` & `OneForOne`)](#erlang-style-fault-boundaries-panic--oneforone)
10. [Native File I/O & Buffer Operations](#10-native-file-io--buffer-operations)
11. [Diagnostic Engine](#11-diagnostic-engine)
12. [Complete End-to-End Program Walkthrough](#12-complete-end-to-end-program-walkthrough)

---

## 1. Introduction & Core Philosophy

**Vira** is a modern systems and AI programming language designed to unite the performance of C and Rust with the ergonomics and compilation speed of Go and V.

### Guiding Principles

- **Zero-Garbage Collector Overhead:** Memory management is static, deterministic, and invisible via an escape-analyzing LIFO destructor injector (`autofree`).
- **Zero-Allocation Frontend:** The compiler lexer operates at **0 B/op and 0 allocs/op**, tokenizing millions of lines per second without touching the heap.
- **Strict 4-Quadrant Mutability:** Variables are immutable by default. Struct fields can be granularly isolated into private/public and immutable/mutable quadrants.
- **Mandatory Result Handling:** Functions returning `!T` cannot be silently ignored; failure to unwrap via `or { ... }` or propagate via `!` is a compile-time error.
- **First-Class AI & N-D Tensor Arithmetic:** 1D–4D strided `tensor[f32]` primitives with batched & broadcast matrix multiplication (`@`), executed by a zero-dependency **multi-threaded CPU SIMD engine** (`AVX2+FMA`, `SSE2`, `ARM NEON`) parallelized across the work-stealing thread pool.
- **Supervised Actor Concurrency:** Concurrency is achieved through isolated actors communicating via non-blocking message sends (`<-`), executed across a lock-free Chase-Lev work-stealing thread pool with Erlang-style `setjmp`/`longjmp` fault containment.

---

## 2. Toolchain & CLI Reference

The Vira compiler CLI binary is `vira` (or `vira.exe` on Windows).

```
vira <command> [flags] <file.vir>
```

### Available Commands

| Command | Purpose | Example |
| :--- | :--- | :--- |
| `run` | Compile and immediately execute a Vira program | `vira run main.vir` |
| `build` | Compile into a native standalone executable (`.exe` / ELF) | `vira build -prod -o app main.vir` |
| `test` | Discover and execute `*_test.vir` test suites | `vira test ./...` |
| `fmt` | Format `.vir` source files to canonical style | `vira fmt main.vir` |
| `doc` | Generate documentation from `.vir` module declarations | `vira doc packages/http` |
| `pkg` | Manage `vira.mod` dependencies and cryptographic `vira.lock` files | `vira pkg lock` |
| `lsp` | Start the JSON-RPC 2.0 Language Server (`virals`) over stdio | `vira lsp` |
| `cgen` | Lower the Vira AST into human-readable C99 code | `vira cgen main.vir` |
| `check` | Run syntax, 4-quadrant mutability, and print the autofree escape analysis report | `vira check main.vir` |
| `demo` | Generate and execute the built-in parallel AI actor demo | `vira demo` |

### Compiler Flags

- `-prod`: Enables maximum optimization flags (`-O3 -march=native -flto`).
- `-sanitize`: Compiles with AddressSanitizer (`ASan`) and UndefinedBehaviorSanitizer (`UBSan`).
- `-o <path>`: Specifies output binary filename.
- `-cc <bin>`: Overrides backend C compiler (`gcc`, `clang`, `cc`, `tcc`).
- `-keep-c`: Preserves intermediate generated `.c` files in `build/`.

---

## 3. Language Basics

### Modules & Package Namespacing

Every Vira source file begins with a `module` declaration defining its namespace boundary:

```v
module http
```

If omitted, standalone files default to `module main`.

#### Visibility & Encapsulation
- Symbols prefixed with `pub` (`pub struct`, `pub fn`, `pub mut:`) are exported and accessible across modules.
- Unprefixed symbols are private to the owning module.
- Within the same module, symbols are directly accessible unqualified.
- Across modules, external symbols must be qualified with their package or alias name:

```v
import net
import http
import io as vira_io

// Qualified types
pub fn handle_connection(mut stream net.TcpStream) !http.Response {
    // Qualified static method calls
    mut srv := http.HttpServer.bind('127.0.0.1:8080')!
    return http.Response.json('{"status":"ok"}')
}
```

#### Reproducible Dependencies (`vira.mod` & `vira.lock`)
Packages declare their module identity and dependencies in `vira.mod`:
```text
module http
version 0.1.0
deps
    net 0.1.0
    io 0.1.0
```
Cryptographic SHA-256 lockfiles (`vira.lock`) ensure 100% reproducible builds across machines and CI pipelines.

---

### Variables & Immutability

Bindings in Vira are **immutable by default**. To reassign a variable, declare it with `mut`:

```v
// Immutable binding (cannot be reassigned)
x := 10
// x = 20  <-- Compile-time error: cannot reassign immutable variable 'x'

// Mutable binding
mut y := 10
y = 20 // OK
```

---

### Blank Identifier (`_`)

The blank identifier `_` discards values, unused bindings, or Result returns without polluting the scope:

```v
_ := compute_heavy_task()
```

Multiple assignments to `_` in the same scope are valid and do not cause variable redeclaration errors.

---

### Primitive Types

| Type | Description | Size |
| :--- | :--- | :--- |
| `u8` / `byte` | Unsigned 8-bit integer (raw byte buffers, I/O, network packets) | 8-bit |
| `int` | Default signed integer | 32-bit |
| `i64` | Signed 64-bit integer (timestamps, currency, file offsets) | 64-bit |
| `f32` | Single-precision IEEE 754 float | 32-bit |
| `f64` | Double-precision IEEE 754 float | 64-bit |
| `bf16`| Brain Floating Point (AI weights) | 16-bit |
| `bool`| Boolean (`true` or `false`) | 8-bit |
| `string`| Immutable UTF-8 string slice (16 bytes: `data`, `len`, `is_lit`) | 16 bytes |

---

### Numeric Literals & Type Casts

Vira supports decimal, hexadecimal (`0x`), octal (`0o`), and binary (`0b`) integer literals:
```v
mask := 0b10110010
permissions := 0o755
hex_code := 0xFF
```

Convert between primitive types using explicit casts:
```v
val_int := 42
val_i64 := i64(val_int)
val_f64 := f64(val_int)
val_u8  := u8(val_int)
```

---

### Constants (`const` / `pub const`)

Compile-time immutable typed constants:
```v
pub const MAX_CONNECTIONS = 1000
const DEFAULT_TIMEOUT_MS = 5000
```

---

### Cleanup with `defer`

The `defer` statement schedules a cleanup action to execute in **LIFO** (Last-In, First-Out) order when the enclosing function or block scope exits:
```v
mut f := io.File.open('records.csv', 'rb') or { return }
defer f.close()

// Normal processing...
// f.close() runs deterministically upon return, error unwrap, or panic!
```

---

### String Interpolation

Strings support zero-cost `${...}` expression interpolation:

```v
name := 'Alice'
score := 98.5
println('Player ${name} scored ${score}%!')
```

---

### Native Collections: Fixed Arrays (`[N]T`) & Slices (`[]T`)

Vira provides two first-class collection types designed with mechanical sympathy:

#### Fixed-Size Stack Arrays (`[N]T`)
Allocated entirely on the stack as value-type structs (`typedef struct { T data[N]; } vira_array_N_T;`), ensuring zero heap allocations and full value semantics:
```v
mut scores := [4]int{95, 88, 72, 91}
scores[1] = 99
println("scores.len = ${scores.len}") // 4
println("scores[1] = ${scores[1]}")   // 99

// Safe access with Result Monad fallback
val := scores.get(10) or { -1 } // -1

// Queries
has := scores.contains(88) // true
```

#### Dynamic Growable Slices (`[]T`)
Dynamic, contiguous heap arrays backed by `(data, len, cap, is_arena)` with amortized $2\times$ expansion and automatic deterministic deallocation via Autofree:
```v
mut items := [1, 2, 3]
items.push(40)
items.push(50)
last := items.pop() // 50

// Safe element lookup returning Result !T
first := items.get(0) or { 0 }

// Deep duplication & clear
cloned := items.clone()
items.clear()
println("is_empty = ${items.is_empty()}") // true
```

#### Iteration (`for ... in`)
Iterate cleanly through any array, slice, or string:
```v
for item in scores {
    println("Score: ${item}")
}
```

#### Multi-Branch Control Flow (`switch`)
Vira provides a native, zero-cost `switch` statement with automatic break, support for primitives, strings, and tagless boolean conditions:
```v
// Integer / Enum switch with multiple comma-separated cases
switch status_code {
case 200:
    println('OK')
case 201, 202:
    println('Accepted')
case 404:
    println('Not Found')
default:
    println('Unknown status')
}

// Native string matching for protocols and routes
switch req.method {
case 'GET', 'HEAD':
    handle_read(req)
case 'POST', 'PUT':
    handle_write(req)
case 'DELETE':
    handle_delete(req)
default:
    handle_other(req)
}

// Tagless boolean condition switch
switch {
case score >= 90:
    grade = 'A'
case score >= 80:
    grade = 'B'
default:
    grade = 'C'
}
```

---

## 4. 4-Quadrant Struct Encapsulation & OOP

Vira provides granular, 4-quadrant field encapsulation within a single struct definition:

```v
pub struct Account {
    // 1. Private & Immutable (default): only accessible within this module; read-only
    secret_pin int

mut:
    // 2. Private & Mutable: only accessible within this module; can be modified
    retry_count int = 0

pub:
    // 3. Public & Read-Only: visible everywhere; cannot be mutated outside module
    id string
    created_at i64

pub mut:
    // 4. Public & Mutable: visible and mutable everywhere
    balance i64
}
```

### Mutability Rules Enforced by Checker

1. A field in `pub:` can **never** be reassigned outside its defining module.
2. A field in `mut:` or `pub mut:` can only be mutated if the enclosing struct instance is bound as `mut`:
   ```vira
   acc := Account{ ... }
   acc.balance += 50 // ERROR: cannot mutate field on immutable binding 'acc'

   mut mut_acc := Account{ ... }
   mut_acc.balance += 50 // OK!
   ```

---

### Struct Embedding & Composition

Vira replaces class inheritance with **struct embedding**. Fields and methods of the embedded struct are automatically promoted:

```v
pub struct AuditMeta {
pub:
    created_at i64
pub mut:
    updated_at i64
}

pub fn (mut a AuditMeta) touch() {
    a.updated_at = 1760000000
}

pub struct Invoice {
    AuditMeta // Transparently embedded!
pub:
    id string
    amount i64
}

fn main() {
    mut inv := Invoice{ ... }
    inv.touch()              // Promoted method dispatch!
    println('${inv.updated_at}') // Promoted field access!
}
```

---

### Methods (Value vs Mutable Receivers)

Methods are declared with an explicit receiver before the function name:

```v
// Immutable receiver (read-only; can be called on immutable or mutable bindings)
pub fn (inv Invoice) summary() string {
    return 'Invoice #${inv.id}: ${inv.amount}'
}

// Mutable receiver (can only be called on 'mut' bindings)
pub fn (mut inv Invoice) apply_discount(percent f32) {
    inv.amount = i64(f32(inv.amount) * (1.0 - percent))
}
```

---

### Static Factory Methods

Type-associated static functions use the `TypeName.method_name` syntax:

```v
pub fn Invoice.new(id string, amount i64) Invoice {
    return Invoice{
        AuditMeta: AuditMeta{ created_at: 1750000000, updated_at: 1750000000 },
        id: id,
        amount: amount,
    }
}

// Called via:
inv := Invoice.new('inv_101', 5000)
```

---

## 5. Structural Interfaces & Dynamic Dispatch

### Interface Definitions

Interfaces define a set of methods that a type must implement. Methods requiring mutation use `mut:`:

```v
pub interface Payable {
    summary() string
mut:
    mark_paid() !string
}
```

### Implicit Satisfaction

Types do **not** declare `implements Payable`. If a struct provides the required methods with matching signatures, it automatically satisfies the interface.

### Fat-Pointer Boxing & Dynamic Dispatch

Cast any satisfying struct to an interface to create a 16-byte fat pointer (`._object`, `._vtable`):

```v
inv := Invoice.new('inv_9001', 15000)
p := Payable(inv) // Boxes into dynamic interface fat-pointer

println(p.summary()) // Dispatches through vtable
```

When `p` goes out of scope, the runtime invokes `p._vtable->_autofree(p._object)` to release both the internal struct heap fields and the boxed container.

---

## 6. Error Handling: `!T` Result & `?T` Option

Vira eschews unhandled exceptions in favor of explicit, zero-cost `!T` Result types.

### Result Types (`!T`)

A function returning a Result indicates potential failure:

```v
pub fn (mut inv Invoice) mark_paid() !string {
    if inv.amount <= 0 {
        return error('invalid invoice amount')
    }
    inv.is_paid = true
    return 'Invoice #${inv.id} paid successfully'
}
```

### The `or { ... }` Unwrap Block

Result values must be unwrapped at the call site. The `or { ... }` block executes only on error, exposing the error message via the implicit variable `err`:

```v
msg := inv.mark_paid() or {
    eprintln('Failed to pay: ${err}')
    return
}
println(msg)
```

Default fallback values can also be supplied directly:

```v
val := risky_calc() or { 0.0 }
```

### Compiler Enforcement

If you call a function returning `!T` without `or { ... }` or `!`, compilation fails:

```
error[SEMANTIC]: unhandled Result '!string' from 'Invoice.mark_paid()': handle it with 'or { ... }' or propagate with '!'
```

---

## 7. Memory Management: Deterministic Autofree & Arenas

Vira delivers automatic memory safety **without a garbage collector** and without manual calls to `free()`.

### How Autofree Works

1. During Phase 3 compilation, the **Static Escape Analyzer** examines variable lifecycles and control flow across all function blocks.
2. If a variable owns heap memory (strings, tensors, boxed interfaces, structs with heap fields), the analyzer tracks its ownership state.
3. At block and function exits, the analyzer injects deterministic **LIFO (Last-In, First-Out)** destructor calls (`vira_str_free`, `vira_tensor_f32_free`, `_autofree`).

### Static Escape Classification

| Classification | Meaning | Action Taken |
| :--- | :--- | :--- |
| `StateScopedLocal` | Variable does not outlive its enclosing block | Destructor injected at scope exit |
| `StateEscapesViaReturn` | Variable is returned to the caller | Ownership transferred; cleanup skipped in current scope |
| `StateMovedToContainer` | Variable is moved into a struct or interface box | Container takes ownership; cleanup transferred |
| `StateManagedByArena` | Allocated within an `arena { ... }` block | Lifetime tied to arena slab; individual frees skipped |

Run `vira check <file.vir>` to inspect the complete autofree escape analysis audit log.

---

### Bump Arenas (`arena { ... }`)

For high-throughput workloads, temporary batch allocations can be placed in a fast, 64-byte cache-aligned bump arena:

```v
arena {
    t1 := tensor_fill(128, 128, 1.0)
    t2 := tensor_fill(128, 128, 2.0)
    t3 := t1 @ t2
    println('Sum: ${tensor_sum(t3)}')
} // Entire arena slab wiped in O(1) time upon block exit!
```

---

## 8. First-Class AI Tensors & Multi-Threaded CPU SIMD Acceleration

### 1D–4D Strided Tensor Types & Construction

Tensors (`tensor[f32]`) are first-class 1D–4D strided primitives in Vira (`dims[4]`, `strides[4]`, `n_dims`, plus synchronized `rows`/`cols` views for 2D compatibility):

```v
// 2D matrix (M x N)
w := tensor_fill(64, 64, 0.25)

// 3D sequence batch (Batch x Seq x Hidden)
hidden := tensor_fill_3d(2, 16, 64, 0.5)

// 4D multi-head attention tensor (Batch x Heads x Seq x HeadDim)
q := tensor_fill_4d(2, 8, 16, 64, 0.125)
k := q.transpose() // Swaps last two axes -> (2, 8, 64, 16)

println('ndim=${q.ndim}, numel=${q.numel()}, head_dim=${q.dim(3)}')
```

Inside `arena { ... }` blocks, `.reshape_2d(...)`, `.reshape_3d(...)`, and `.reshape_4d(...)` return $O(1)$ zero-copy views (`is_view = true`); outside arenas they safely allocate independent autofreed buffers.

### Batched & Broadcast Matrix Multiplication (`@`) & Bias Broadcasting

The `@` operator supports:
- **2D GEMM:** `(M, K) @ (K, N) -> (M, N)`
- **Batched 3D / 4D Attention MatMul:** `(B, H, M, K) @ (B, H, K, N) -> (B, H, M, N)`
- **N-D × 2D Linear Projection Broadcast:** `(B, S, K) @ (K, N) -> (B, S, N)`

Element-wise `+`, `-`, and `tensor_elemwise_mul` also support trailing-dimension bias broadcasting (e.g., `(B, S, D) + (1, D)` or `(M, D) + (D)`).

```v
scores := q @ k           // (2, 8, 16, 16) multi-head attention scores
proj   := hidden @ w      // (2, 16, 64) broadcasted linear projection
total  := tensor_sum(proj)
```

### Multi-Threaded CPU SIMD Hardware Backends (Zero External GPU Drivers)

Vira's built-in tensor runtime is an honest, **100% zero-dependency CPU SIMD engine** (requiring no CUDA, ROCm, or OpenCL installation):

- **x86_64 AVX2 + FMA:** 256-bit vector registers (8 single-precision floats per FMA instruction)
- **AArch64 NEON:** 128-bit vector registers (4 single-precision floats per cycle)
- **x86_64 SSE2:** 128-bit vector registers (4 single-precision floats per cycle)
- **L2 Cache Blocking:** Hierarchical 64×64 cache-tiled matrix multiplication (`@`) keeping intermediate submatrices resident within L1/L2 data cache.
- **Work-Stealing Multi-Core GEMM:** Matrix multiplications exceeding $128^3$ operations automatically partition output row tiles across the Chase-Lev worker thread pool using zero-allocation stack job descriptors.
- **Native C99 FFI Interop:** External C/GPU libraries can be bound cleanly via `@[intrinsic: '...']` and NUL-terminated string marshaling helpers (`vira_str_cstr` / `vira_str_to_cstr_buf` in `vira_rt.h`).

---

## 9. Actor Concurrency & Fault Supervision

Vira features Erlang-inspired actor concurrency built for multicore machine learning workloads and microservices.

### Actor Definition

An actor is declared with the `actor` keyword:

```v
pub actor AIWorker {
pub:
    id int
mut:
    batches_run int = 0
}

pub fn (mut w AIWorker) predict(scale f32, simulate_fault bool) f32 {
    w.batches_run += 1
    weights := tensor_fill(64, 64, 0.25)
    input := tensor_fill(64, 64, scale)
    out := weights @ input

    if simulate_fault {
        panic('CUDA/NaN anomaly on parallel batch')
    }
    return tensor_sum(out)
}
```

---

### Message Passing Modes (`<-`)

Vira supports three execution modes for message passing:

#### 1. Asynchronous with Futures (`fut := actor <- method()`)
Dispatches a message onto the Chase-Lev work-stealing pool immediately returning a `Future[T]` handle. The calling thread continues execution without blocking:

```v
fut1 := w1 <- predict(2.0, false)
fut2 := w2 <- predict(3.0, true) // Fault injected

// Non-blocking work-stealing await with cooperative work helping
res1 := fut1.await() or { 0.0 }
res2 := fut2.await() or {
    println('Worker 2 faulted: ${err}')
    0.0
}
```

#### 2. Detached Fire-and-Forget Sends (`actor <- method()`)
When the return value is not needed (e.g. background logging, metrics emission, fire-and-forget ingestion), dispatch the call as a standalone statement or discard the handle with `_ :=`:

```v
w1 <- predict(1.0, false)       // Standalone statement send
_ := w2 <- predict(1.5, false)  // Discarded binding
```

The compiler routes both forms to `_send_detached_` (`done = -1`). The background worker thread automatically frees the task envelope upon completion without creating or tracking a `Future[T]`, delivering maximum throughput with zero memory leaks (`active_leaks = 0`).

#### 3. Zero-Alloc Synchronous Calls (`res := actor <- method() or { ... }`)
When the caller immediately unwraps an actor call with `or { ... }`, the compiler optimizes the dispatch into a zero-allocation synchronous call. The envelope is allocated on the caller's stack frame, eliminating heap `malloc`/`free` overhead while maintaining supervisor fault containment.

```v
sync_res := w1 <- predict(2.5, false) or { 0.0 }
```

---

### Lock-Free Work-Stealing Pool & Cache Optimization

- Tasks are submitted to a multicore pool of worker threads sized to `min(CPU cores, 16)`.
- **Chase-Lev Deques with Cacheline Separation:** Worker deques place `top` and `bottom` pointers on separate 64-byte cachelines (`_Alignas(64)`), eliminating false sharing under high concurrency.
- **Round-Robin Task Distribution:** External submissions utilize an isolated cacheline round-robin cursor to balance queue pressure across all workers.
- **Monotonic Clocks:** Condition variables synchronize using `CLOCK_MONOTONIC` to ensure deterministic parking and wake-ups immune to system clock shifts.
- **Work-Helping Await:** Calling `fut.await()` causes the awaiting thread to steal and execute pending tasks from victim deques instead of sleeping.

---

### Erlang-Style Fault Boundaries & Actor Lifecycle

Actors follow an Erlang-inspired supervision lifecycle:

```
[VIRA_ACTOR_RUNNING] ──(panic)──► [VIRA_ACTOR_FAULTED]
         ▲                               │
         │ (restarts < max_restarts)     │ (restarts >= max_restarts)
         └─── [OneForOne Restart] ◄──────┴────────► [VIRA_ACTOR_STOPPED]
                                                           │
                                                           ▼
                                                [Dead-Letter Rejection]
```

1. **Panic Interception & Stack Defer Unwinding:** Per-thread TLS fault boundaries (`setjmp`/`longjmp`) intercept panics before they can crash the worker thread or host process. Before `longjmp`, the runtime unwinds the actor's LIFO stack-frame `defer_stack`, deterministically freeing any live local heap strings, slices, tensors, structs, or nested arenas (`active_leaks = 0`).
2. **O(1) Arena Wipe & Deep Envelope Isolation:** The actor's private scratch arena slab is cleared in $O(1)$ time (`isolated_heap.offset = 0`), and heap-owning message arguments/unconsumed results on async (`<-`) or detached sends are deep-cloned and cleaned up automatically.
3. **OneForOne Restart:** If consecutive restarts are below `max_restarts` (default: 3), the supervisor increments the restart counter, executes `_supervisor_reset(&self)`, and restores the actor to `VIRA_ACTOR_RUNNING`.
4. **Consecutive Fault Reset:** A successful message execution resets consecutive `restarts = 0`, ensuring intermittent faults across long-running services do not accumulate toward the limit.
5. **Dead-Letter Containment (`VIRA_ACTOR_STOPPED`):** If an actor suffers consecutive faults exceeding `max_restarts`, it transitions to `VIRA_ACTOR_STOPPED`. Subsequent message attempts are immediately rejected with error `"Actor has stopped due to unhandled faults"`, protecting downstream queues from cascade failures.

```text
[vira_supervisor | worker_thread=1] Actor 'AIWorker' (#2) trapped fault ('CUDA/NaN anomaly on parallel batch') -> O(1) arena wipe & OneForOne restart [1/3]
Future #2 caught parallel Worker #2 fault: CUDA/NaN anomaly on parallel batch
```

---

## 10. Native File I/O & Buffer Operations

Vira provides native, hardware-level file operations returning zero-cost Result containers (`!string`, `!bool`):

| Builtin Signature | Description |
| :--- | :--- |
| `file_exists(path string) bool` | Returns `true` if the target path exists. |
| `read_file(path string) !string` | Reads the entire file into a heap-allocated UTF-8 string buffer managed by Autofree. |
| `write_file(path string, content string) !bool` | Writes content to file atomically, creating or overwriting it. |
| `append_file(path string, content string) !bool` | Appends content to the end of a file. |
| `remove_file(path string) !bool` | Deletes the file from the filesystem. |

### Example Usage

```v
// Atomic write
_ := write_file('log.txt', 'Pipeline started\n') or {
    eprintln('Write failed: ${err}')
    return
}

// Read with automatic buffer cleanup
data := read_file('log.txt') or {
    eprintln('Read failed: ${err}')
    return
}
println(data) // Buffer freed automatically at scope exit!
```

### Standard Library Packages (`packages/`)

Vira's compiler resolves built-in pure-Vira standard library packages automatically under `packages/`:
- `import io` — Unified `Reader`/`Writer`/`Closer` streams and `io.File` handles.
- `import net` — Cross-platform TCP/UDP sockets, DNS resolution, and non-blocking pollers.
- `import http` — HTTP/1.1 request parser, response builder, header maps, and SSE streaming.
- `import json` — 100% pure-Vira RFC 8259 JSON validator (`json.is_valid`), depth-aware parser (`json.get_str`, `json.get_int`, `json.get_i64`, `json.get_bool`, `json.array_get_str`), and $O(N)$ `json.JsonBuilder`.
- `import viraserve` — Work-stealing actor HTTP microframework.

---

## 11. Diagnostic Engine

The Vira compiler provides Rust/V-style caret diagnostics with exact column spans and actionable `= help:` suggestions.

```
error[SEMANTIC]: cannot reassign immutable variable 'x'
  --> examples/bad_account.vir:18:2
   |
18 |     x = 20
   |     ^^^^^^
   = help: declare with 'mut x := ...'

error[SEMANTIC]: cannot call mutable receiver method 'deposit()' on immutable binding 'imm_acc'
  --> examples/bad_account.vir:21:7
   |
21 |     _ := imm_acc.deposit(50) or { 0 }
   |          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   = help: declare 'mut imm_acc := ...'

error[SEMANTIC]: field 'Account.id' is 'pub:' (read-only) and cannot be mutated
  --> examples/bad_account.vir:24:2
   |
24 |     mut_acc.id = 'hacked_id'
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
```

---

## 12. Complete End-to-End Program Walkthrough

The following program combines all core language features: OOP composition, fat-pointer interfaces, SIMD tensors, work-stealing actor concurrency, and fault tolerance:

```v
module billing

// --- 1. OOP STRUCTS & COMPOSITION ---

pub struct AuditMeta {
pub:
	created_at i64
pub mut:
	updated_at i64
}

pub fn (mut a AuditMeta) touch() {
	a.updated_at = 1760000000
}

pub struct Invoice {
	AuditMeta
	secret_token string
mut:
	retry_count int = 0
pub:
	id string
	amount_cents i64
pub mut:
	is_paid bool
}

pub interface Payable {
	summary() string
mut:
	mark_paid() !string
}

pub fn Invoice.new(id string, amount_cents i64) Invoice {
	return Invoice{
		AuditMeta: AuditMeta{ created_at: 1750000000, updated_at: 1750000000 },
		secret_token: 'sec_live_01',
		id: id,
		amount_cents: amount_cents,
		is_paid: false,
	}
}

pub fn (inv Invoice) summary() string {
	return 'Invoice #${inv.id} [updated_at=${inv.updated_at}]'
}

pub fn (mut inv Invoice) mark_paid() !string {
	if inv.amount_cents <= 0 {
		return error('invalid invoice amount')
	}
	inv.is_paid = true
	inv.retry_count += 1
	inv.touch()
	return 'Invoice #${inv.id} paid (${inv.amount_cents} cents)'
}

// --- 2. MULTI-THREADED AI WORKER ACTOR ---

pub actor AIWorker {
pub:
	id int
mut:
	batches_run int = 0
}

pub fn (mut w AIWorker) predict(scale f32, simulate_fault bool) f32 {
	w.batches_run += 1
	weights := tensor_fill(64, 64, 0.25)
	input := tensor_fill(64, 64, scale)
	out := weights @ input

	if simulate_fault {
		panic('CUDA/NaN anomaly on parallel batch')
	}
	return tensor_sum(out)
}

// --- 3. APPLICATION ENTRYPOINT ---

fn main() {
	println('=== 1. OOP + STRUCT EMBEDDING + FAT-POINTER INTERFACE ===')
	mut inv := Invoice.new('inv_9001', 15000)
	msg := inv.mark_paid() or {
		eprintln(err)
		return
	}
	println(msg)
	p := Payable(inv)
	_ := p

	println('=== 2. PARALLEL FUTURE[f32] FAN-OUT ACROSS WORK-STEALING POOL ===')
	mut w1 := AIWorker{ id: 1, batches_run: 0 }
	mut w2 := AIWorker{ id: 2, batches_run: 0 }
	mut w3 := AIWorker{ id: 3, batches_run: 0 }

	// Dispatch 3 concurrent actor tasks across the thread pool
	fut1 := w1 <- predict(2.0, false)
	fut2 := w2 <- predict(3.0, true) // Injected panic
	fut3 := w3 <- predict(4.0, false)

	s1 := fut1.await() or { 0.0 }
	s2 := fut2.await() or {
		println('Future #2 caught parallel Worker #2 fault: ${err}')
		0.0
	}
	s3 := fut3.await() or { 0.0 }

	println('Parallel Future results -> w1=${s1}, w2(faulted)=${s2}, w3=${s3}')

	// Supervised restart recovery verification
	fut2_retry := w2 <- predict(3.0, false)
	s2_recovered := fut2_retry.await() or { 0.0 }
	println('Worker #2 recovered Future result -> w2=${s2_recovered}')
}
```

### Compiling and Running

```powershell
# Fast build with native production optimizations
.\vira.exe build -prod -o app.exe examples/parallel_ai_actors.vir

# Execute
.\app.exe
```

### Execution Telemetry Output

```text
=== 1. OOP + STRUCT EMBEDDING + FAT-POINTER INTERFACE ===
Invoice #inv_9001 paid (15000 cents)
=== 2. PARALLEL FUTURE[f32] FAN-OUT ACROSS WORK-STEALING POOL ===
[vira_supervisor | worker_thread=1] Actor 'AIWorker' (#2) trapped fault ('CUDA/NaN anomaly on parallel batch') -> O(1) arena wipe & OneForOne restart [1/3]
Future #2 caught parallel Worker #2 fault: CUDA/NaN anomaly on parallel batch
Parallel Future results -> w1=131072.00, w2(faulted)=0.00, w3=262144.00
Worker #2 recovered Future result -> w2=196608.00
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
[vira_rt] pool_workers=4 | actor_tasks_run=4 | work_steals=3
[vira_rt] active_leaks=0 | total_freed=12
```
