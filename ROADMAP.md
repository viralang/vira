# Vira Language Roadmap & Strategic Frontiers

Welcome to the official roadmap for the **Vira Programming Language**! This document tracks our completed milestones, current architectural foundation, and upcoming strategic frontiers on the path to **v1.0**.

---

## 🎯 Vision & Core Philosophy

Vira is an Ahead-Of-Time (AOT) compiled systems programming language engineered specifically for **native AI systems, deterministic performance, and actor concurrency**.

* **Deterministic Resource Management:** Zero garbage collector overhead, zero hidden pauses, and zero memory leaks (`active_leaks=0`) via static scope escape analysis (Autofree) and 64-byte aligned bump arenas.
* **First-Class AI Tensors:** Contiguous, cache-friendly tensor representations with built-in hardware SIMD vectorization (AVX2 + FMA, ARM NEON), native Hugging Face `.safetensors`, and GGUF v2/v3 local LLM inference with zero Python or C++ runtime dependencies.
* **Erlang-Inspired Fault Tolerance:** Lightweight actor concurrency with Chase-Lev work-stealing and `OneForOne` supervisor containment that isolates failures without crashing the entire service.
* **Modern Developer Tooling:** First-party CLI commands (`fmt`, `test`, `lsp`, `init`), package manifests (`vira.mod`), and native VS Code integration.

---

## 🗺️ High-Level Milestone Overview

```
 [v0.1 - v0.3] Core Language & Compiler         ✅ COMPLETED
       │
 [v0.4] Multi-Core Actors & Fault Tolerance     ✅ COMPLETED
       │
 [v0.5] AI Tensors, SafeTensors & GGUF LLM      ✅ COMPLETED
       │
 [v0.5.5] Tooling: fmt, test, lsp & vira.mod     ✅ COMPLETED
       │
 [v0.5.6] Enterprise Scaffolding & GUI Engine   ✅ COMPLETED
       │
 [v0.5.7] Package Manager, Lockfiles & Vendoring✅ COMPLETED
       │
 [v0.5.8] Zero-Leak Autofree & Actor Benchmarks ✅ COMPLETED
       │
 ═════════════════════════════════════════════════════════════ CURRENT STATUS
       │
 [v0.6] Phase 1: VexServe Web & LLM Streaming  ⏳ NEXT MILESTONE
       │
 [v0.7] Phase 2: C & Python FFI / Shared Libs  📅 PLANNED
       │
 [v0.8] Phase 3: Cross-Platform GPU Compute    📅 PLANNED
       │
 [v0.9] Phase 4: Generics & Pattern Matching    📅 PLANNED
       │
 [v1.0] Phase 5: Production Registry & Freeze   📅 PLANNED
```

---

## ✅ Completed Milestones

### 1. Core Language & Compiler (v0.1 – v0.3)
- [x] **Lexer & Parser:** Full recursive descent parser supporting immutable-by-default variables, 4-quadrant struct encapsulation (`default`, `mut:`, `pub:`, `pub mut:`), struct composition/embedding, and structural interfaces with dynamic vtables.
- [x] **Zero-Cost Error Handling:** Explicit Result types (`!T`), `return error('...')`, and mandatory unwrap blocks (`val := call() or { ... }`).
- [x] **Static Autofree Engine:** Compiler-driven escape analysis tracking variable lifecycles (`ScopedLocal`, `EscapesViaReturn`, `MovedToContainer`) with automatic LIFO destructor insertion and 0 runtime GC overhead.
- [x] **Bump Arenas:** 64-byte aligned linear bump allocators (`arena { ... }`) offering O(1) bulk slab teardown.
- [x] **AOT C99 Transpiler:** High-speed code generator emitting clean, warning-free C99 code compatible with GCC, Clang, and MSVC.

### 2. Multi-Core Concurrency & Fault Tolerance (v0.4)
- [x] **Chase-Lev Work-Stealing Pool:** Multi-threaded runtime scheduling actor tasks across CPU hardware threads with lock-free double-ended queues.
- [x] **Actor Concurrency:** Non-blocking message dispatch (`actor <- msg`) returning typed asynchronous futures (`Future[T].await()`).
- [x] **Panic Containment & Supervision:** Actor thread crash isolation via `setjmp`/`longjmp`, O(1) actor arena wiping, and Erlang-style `OneForOne` supervisor restarts.
- [x] **File I/O:** POSIX and Win32 file manipulation API (`file_exists`, `read_file`, `write_file`, `append_file`, `remove_file`).

### 3. First-Class AI Tensors & Native LLM Inference (v0.5)
- [x] **SIMD Vectorization:** Contiguous 2D numerical tensors (`tensor[f32]`) with hardware-accelerated matrix multiplication (`@`) targeting AVX2+FMA (8 float32s/cycle) and ARM NEON.
- [x] **Native SafeTensors Parser:** Zero-copy ingestion of Hugging Face `.safetensors` model weights without Python or external libraries.
- [x] **Autograd Engine:** Tensor gradient backward operators (`tensor_matmul_grad_a`, `tensor_matmul_grad_b`, `tensor_relu_backward`) and in-place SGD optimization (`tensor_sgd_step`).
- [x] **Native GGUF v2/v3 LLM Inference Engine:**
  - Binary GGUF header, metadata, and tensor info parser.
  - Q4_0 and Q8_0 SIMD block dequantization kernels.
  - Full Transformer architecture: RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU activations, and Causal Multi-Head Attention.
  - Temperature-scaled token sampling generating autoregressive text.

### 4. Developer Tooling & Package Ecosystem (v0.5.5)
- [x] **Code Formatter (`vira fmt`):** AST-aware formatter with `-w` in-place formatting and automated multi-file directory discovery.
- [x] **Automated Test Runner (`vira test`):** Synthesizes isolated test harnesses, runs `*_test.vir` files, and automatically validates zero memory leaks (`active_leaks=0`).
- [x] **Language Server Protocol (`vira lsp`):** Standard JSON-RPC 2.0 LSP server providing diagnostics, hover documentation, and on-save formatting.
- [x] **VS Code Extension (`editors/vscode/`):** Full extension package with syntax grammar, bracket matching, code snippets, and LSP client integration.
- [x] **Package Manifest & Multi-File Projects (`vira init` & `vira.mod`):** Scaffolding and multi-file directory compilation into single unified binaries.
- [x] **Curriculum & Documentation:** 19 runnable standalone examples and 20 step-by-step tutorials with verified 0 leaks.

### 5. Enterprise Multi-Package Scaffolding & Declarative GUI Engine (v0.5.6)
- [x] **Enterprise Multi-Package Scaffolding (`vira init`):** Modern project layout generation (`src/config/`, `src/models/`, `src/services/`, `tests/app_test.vir`, `vira.mod`, `README.md`, `.gitignore`).
- [x] **Recursive Package Traversal:** Deep multi-directory compiler walking supporting arbitrary package hierarchies.
- [x] **C Identifier Sanitization:** Robust module name mangling for numeric directory prefixes.
- [x] **Integrated Test Harness:** Automatic project source resolution for unit test files with application `main()` isolation.
- [x] **Declarative IMGUI Auto-Layout Engine:** Vertical stacks (`gui_vstack_begin`), horizontal stacks (`gui_hstack_begin`), auto-sized cards, text inputs, multi-tab routing, modal dialogs, and close interception.
- [x] **Curriculum Expansion:** 23 runnable examples and 24 in-depth tutorials.

### 6. Native Package Management, Deterministic Lockfiles & Vendoring (v0.5.7)
- [x] **Git-Native Package Management (`vira add`):** Direct Git cloning (`github.com/user/pkg[@version]`) and local directory dependency linking with path traversal protection.
- [x] **Cryptographic Deterministic Lockfile (`vira.lock`):** Full directory tree SHA-256 tree hashing (`ComputeDirChecksum`) guaranteeing supply chain integrity and reproducible bit-exact builds.
- [x] **Dependency Restoration (`vira install` / `vira get`):** Verifies checksum integrity against `vira.lock` and restores missing dependencies.
- [x] **Clean Dependency Removal (`vira remove`):** Cleanly unlinks packages from `vira.mod`, `vira.lock`, and the `.vira/pkg` cache.
- [x] **Air-Gapped Offline Vendoring (`vira vendor`):** Deep copying of dependencies into `./vendor` for 100% offline, zero-internet compilations.
- [x] **Unified 6-Tier Import Resolution:** Identical import lookup order across both the compiler (`cmd/vira/main.go`) and Language Server (`internal/lsp/server.go`) (file dir → `src/` → root → `vendor/` → `.vira/pkg/` → `~/.vir/pkg/`).
- [x] **Curriculum & Verification:** Added [Tutorial 25: Native Package Management, Publishing & Air-Gapped Vendoring](docs/tutorials/25_package_management_publishing_and_vendoring.md) and runnable example [24_package_management_consumer.vir](docs/examples/24_package_management_consumer.vir) with verified 0 leaks.

### 7. Zero-Leak Temporary String Autofree & High-Concurrency Actor Stress Testing (v0.5.8)
- [x] **Zero-Leak Temporary String Autofree:** Static AST hoist pass in `internal/autofree/autofree.go` (`hoistTemporaryStrings`, `walkAndHoistCallArgs`) capturing unassigned interpolated strings and call returns passed to functions/methods into synthetic variables (`_vira_tmp_str_N`) with immediate post-statement `DeferFreeStmt` cleanup. Eliminates intermediate string leaks (`active_leaks=0`).
- [x] **Lazy Bump Arena for Actors:** Optimized actor memory initialization in `runtime/vira_rt.h` (`vira_actor_ctx_init`), allocating isolated heaps lazily on demand. Drops idle actor memory footprint from 256 KB to **~64 bytes per actor**, enabling 100,000+ lightweight actors without RAM exhaustion.
- [x] **Chase-Lev Work-Stealing Capacity Expansion:** Scaled deque ring buffer capacity `VIRA_WS_CAP` to 65,536 tasks across up to 32 worker threads, eliminating thread contention and lock spins during massive message bursts.
- [x] **High-Precision Monotonic Telemetry Timers:** Builtin `time_now_us()` and `time_now_ms()` delivering sub-microsecond timing accuracy via Win32 `QueryPerformanceCounter` and POSIX `clock_gettime(CLOCK_MONOTONIC)`.
- [x] **Comprehensive Actor Example Suite:**
  - [25_actors_banking_fault_tolerance.vir](docs/examples/25_actors_banking_fault_tolerance.vir): State isolation, `setjmp`/`longjmp` panic isolation, and `OneForOne` supervisor state rollback.
  - [26_actors_worker_pool_fanout.vir](docs/examples/26_actors_worker_pool_fanout.vir): Parallel SIMD matrix worker pool fanning out across Chase-Lev threads with work-helping `.await()`.
  - [27_actors_ring_pipeline.vir](docs/examples/27_actors_ring_pipeline.vir): Pipelined token passing across chained stage actors with concurrent interleaved streams.
- [x] **High-Concurrency Stress Benchmarks & Chaos Testing:**
  - Standalone benchmark [examples/actor_stress_benchmark.vir](examples/actor_stress_benchmark.vir) executing 10,000 actors spawned in 6 µs (10,000,000 spawns/sec), 4,000 async messages processed in 3.3 ms (1,000,000 msgs/sec), 15% random panic chaos injection with 100% recovery rate in 476 µs, and 10,000 ring hops in 5.1 ms (1,666,666 hops/sec) with verified `active_leaks=0`.
  - Added CI golden stress test [testdata/run/03_actor_high_concurrency_stress.vir](testdata/run/03_actor_high_concurrency_stress.vir).
  - Published comprehensive architectural study: [Tutorial 26: Actor System Architecture, Trade-Offs & High-Concurrency Benchmarks](docs/tutorials/26_actor_system_tradeoffs_and_benchmarks.md).

---

## 🚀 Upcoming Strategic Frontiers

### 🌐 Phase 1: VexServe — Asynchronous Web & LLM Streaming Framework (v0.6)
*Target: High-throughput AI microservices, REST APIs, and real-time streaming.*

- [ ] **HTTP Request Parser & Router:**
  - Fast, zero-copy HTTP/1.1 request line and header parser.
  - Path pattern matching with dynamic URL parameters (e.g., `GET /v1/models`, `POST /v1/chat/completions`).
  - Extensible middleware chain (CORS headers, bearer token authentication, request timing).
- [ ] **Actor Connection Pool:**
  - Dedicated connection-acceptor actors fanning out incoming client sockets across worker actors via work-stealing queues.
  - Non-blocking socket polling (`select`/`epoll`/`kqueue`/`WSAPoll`).
- [ ] **Real-Time Token Streaming (SSE):**
  - Native Server-Sent Events (`text/event-stream`) chunked response writer.
  - Hook the pure Vira GGUF inference engine directly into the streaming writer, yielding tokens to web clients token-by-token with sub-millisecond dispatch.
- [ ] **Standardized JSON Schema Serialization:**
  - High-level struct-to-JSON serialization (`json_encode(struct)`) matching OpenAI-compatible completion specifications.

---

### 🔌 Phase 2: C & Python FFI & Shared Library Export (v0.7)
*Target: Seamless interoperability with the broader software ecosystem.*

- [ ] **Native C Interoperability (`extern "C"`):**
  - Header declaration syntax for linking external C functions and structs directly:
    ```vira
    extern "C" {
        fn sqlite3_open(filename: string, db: &void) -> int
        fn sqlite3_exec(db: void, sql: string, callback: void, arg: void, errmsg: &string) -> int
    }
    ```
  - Linking flags (`-l`, `-L`) integrated into `vira.mod` manifests.
- [ ] **Shared Dynamic Library Compilation (`vira build -shared`):**
  - Produce `.dll` (Windows), `.so` (Linux), and `.dylib` (macOS) shared libraries adhering to standard C-ABI calling conventions.
  - Automatic C header file (`.h`) generation matching exported public Vira functions.
- [ ] **Python Interoperability / C-Extension Generator:**
  - Tooling command `vira pybind` generating Python C-extension bindings with zero-copy NumPy / PyTorch tensor buffer interchange.
  - Enables Python data science stacks to invoke Vira SIMD tensors and local GGUF models directly.

---

### ⚡ Phase 3: Cross-Platform GPU Compute Acceleration (v0.8)
*Target: Accelerating heavy model inference beyond CPU SIMD limits.*

- [ ] **Compute Shader Pipeline (Vulkan / Direct3D 12 / Metal):**
  - Portable GPU compute kernels (SPIR-V / HLSL / MSL) for matrix multiplication (`@`), RMSNorm, and Softmax.
  - Headless GPU device discovery and command queue creation.
- [ ] **Unified DMA & Device Allocators:**
  - 64-byte aligned host-visible device memory allocations.
  - Zero-copy tensor handoff over PCIe bus and Apple Silicon unified memory.
- [ ] **Actor Fault-Tolerant GPU Supervision:**
  - Asynchronous GPU compute fence tracking inside actor tasks.
  - Automatic fallback to CPU SIMD kernels if GPU device memory encounters Out-of-Memory (OOM) or hardware faults.

---

### 🧩 Phase 4: Advanced Language Features & Expressiveness (v0.9)
*Target: Modern type system ergonomic enhancements while preserving mechanical sympathy.*

- [ ] **Parametric Polymorphism (Generics):**
  - Type-parameterized structs and functions:
    ```vira
    struct RingBuffer[T] {
        items: []T
        head: int
        tail: int
    }
    
    fn map[T, U](list: []T, transform: fn(T) -> U) -> []U
    ```
  - Monomorphized ahead-of-time code generation ensuring zero dynamic dispatch overhead for generic types.
- [ ] **Exhaustive Pattern Matching (`match` expressions):**
  - Multi-branch pattern matching on primitive values, strings, Result unwrap, and discriminated unions:
    ```vira
    match status {
        200 => print("OK")
        404 => print("Not Found")
        _   => print("Unknown")
    }
    ```
- [ ] **Zero-Copy Slice Range Expressions:**
  - Native range slicing syntax for strings and tensors: `sub_slice := buffer[0..128]`.

---

### 📦 Phase 5: Production Registry & Ecosystem Maturity (v1.0)
*Target: Production readiness and distributed developer tooling.*

- [x] **Package Management CLI & Lockfiles (Completed in v0.5.7):**
  - Dependency fetching via Git repositories and local paths (`vira add github.com/user/pkg`).
  - Deterministic lockfile generation (`vira.lock`) with cryptographic SHA-256 tree hash verification.
  - Air-gapped offline vendoring (`vira vendor`).
  - See [Tutorial 25: Native Package Management, Publishing & Air-Gapped Vendoring](docs/tutorials/25_package_management_publishing_and_vendoring.md).
- [ ] **Documentation Generator (`vira doc`):**
  - Extract docstrings and type signatures from `.vir` source code and generate browsable, search-indexed static HTML documentation sites.
- [ ] **Standard Library Expansion:**
  - Cryptography (`crypto/sha256`, `crypto/aes`).
  - Compression (`compress/gzip`, `compress/zstd`).
  - High-precision timing and benchmarking harness (`vira bench`).
- [ ] **v1.0 API Stability Guarantee:**
  - Syntax and ABI freeze with guaranteed backwards compatibility for all 1.x releases.

---

## 📊 Quality & Performance Standards

Every feature added to the Vira codebase must adhere to the following non-negotiable principles:

| Metric | Target Standard | Verification Method |
| :--- | :--- | :--- |
| **Memory Leaks** | Exactly `0` (`active_leaks=0`) | Runtime leak counter and GCC `-fsanitize=address` |
| **Test Coverage** | 100% green test suite | `go test ./...` and `vira test` |
| **Documentation** | Tutorial + Example pair | Every major feature requires a runnable `.vir` example and a markdown tutorial in `docs/` |
| **Cross-Platform** | Zero Windows/Linux/macOS friction | Tested across Win32/MinGW, Linux (glibc/musl), and macOS (Apple Clang) |
| **SIMD Alignment** | 64-byte aligned buffers | Tested for AVX2 (256-bit) and ARM NEON (128-bit) alignment |

---

## 🤝 Contributing & Community

* **Source Code:** [github.com/viralang/vira](https://github.com/viralang/vira)
* **Language Tutorials:** [docs/tutorials/](docs/tutorials/)
* **Runnable Examples:** [docs/examples/](docs/examples/)
* **Language Specification & Architecture:** [LANGUAGE_GUIDE.md](LANGUAGE_GUIDE.md)
