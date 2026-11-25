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
 [v0.5.9] Native Collections: Arrays & Slices   ✅ COMPLETED
       │
 [v0.6] CLI, Containers & Complete GUI Suite    ✅ COMPLETED
       │
 [v0.7] Native C & Python FFI / Shared Libs     ✅ COMPLETED
       │
 [v0.7.1] Pure-Vira Hugging Face & Qwen2.5 LLM  ✅ COMPLETED
       │
 [v0.8] Cross-Platform GPU Compute Acceleration ✅ COMPLETED
       │
 [v0.7.8–v0.7.9] CUDA Graphs, FlashAttention-2, MoE/MLA & Hardening ✅ COMPLETED
       │
 [v0.9] Enums, Maps, Closures, Generics, Ref[T] & EventBus ✅ COMPLETED (unreleased)
       │
 ═════════════════════════════════════════════════════════════ CURRENT STATUS
       │
 [v0.9.x] Pattern Matching, Ranges & Generic Constraints  ⏳ NEXT MILESTONE
       │
 [v1.0] Production Registry, Stability & Freeze 📅 PLANNED
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
  - Q4_0 and Q8_0 SIMD block dequantization kernels (extended in milestone 15 to every llama.cpp K-quant, BF16, MXFP4 and I32).
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
- [x] **Chase-Lev Work-Stealing Capacity Expansion:** Per-worker Chase-Lev deques (`VIRA_WS_CAP` = 4,096 tasks, with a shared overflow queue beyond that) across up to 32 worker threads, eliminating thread contention and lock spins during massive message bursts.
- [x] **High-Precision Monotonic Telemetry Timers:** Builtin `time_now_us()` and `time_now_ms()` delivering sub-microsecond timing accuracy via Win32 `QueryPerformanceCounter` and POSIX `clock_gettime(CLOCK_MONOTONIC)`.
- [x] **Comprehensive Actor Example Suite:**
  - [25_actors_banking_fault_tolerance.vir](docs/examples/25_actors_banking_fault_tolerance.vir): State isolation, `setjmp`/`longjmp` panic isolation, and `OneForOne` supervisor state rollback.
  - [26_actors_worker_pool_fanout.vir](docs/examples/26_actors_worker_pool_fanout.vir): Parallel SIMD matrix worker pool fanning out across Chase-Lev threads with work-helping `.await()`.
  - [27_actors_ring_pipeline.vir](docs/examples/27_actors_ring_pipeline.vir): Pipelined token passing across chained stage actors with concurrent interleaved streams.
- [x] **High-Concurrency Stress Benchmarks & Chaos Testing:**
  - Standalone benchmark [examples/actor_stress_benchmark.vir](examples/actor_stress_benchmark.vir) executing 10,000 actors spawned in 6 µs (10,000,000 spawns/sec), 4,000 async messages processed in 3.3 ms (1,000,000 msgs/sec), 15% random panic chaos injection with 100% recovery rate in 476 µs, and 10,000 ring hops in 5.1 ms (1,666,666 hops/sec) with verified `active_leaks=0`.
  - Added CI golden stress test [testdata/run/03_actor_high_concurrency_stress.vir](testdata/run/03_actor_high_concurrency_stress.vir).
  - Published comprehensive architectural study: [Tutorial 26: Actor System Architecture, Trade-Offs & High-Concurrency Benchmarks](docs/tutorials/26_actor_system_tradeoffs_and_benchmarks.md).

### 8. Native Collections: Fixed Arrays, Dynamic Slices & LSP Tooling (v0.5.9)
- [x] **Native Fixed Arrays (`[N]T`):** Stack-allocated, zero-overhead fixed-size contiguous arrays with compile-time size checking, zero-cost value semantics, and bounds validation.
- [x] **Native Dynamic Slices (`[]T`):** Heap-backed, geometrically growing dynamically sized collections with automatic growth (`.push()`), shrinkage (`.pop()`), fast indexed access, and bounds protection.
- [x] **Idiomatic Collection Iteration (`for item in collection`):** First-class for-in statement syntax compiling directly into zero-cost index-driven C loops without iterator heap allocations.
- [x] **LIFO Zero-Leak Slice Autofree:** Compiler-driven `CleanupSlice` injection ensuring deterministic, leak-free deallocation of dynamic slice buffers when going out of scope (`active_leaks=0`).
- [x] **Comprehensive Builtin Method Suite:** Native methods including `.len`, `.cap`, `.get(idx)` (monadic `!T`), `.set(idx, val)`, `.push(val)`, `.pop()`, `.clear()`, and `.copy()`.
- [x] **Full Toolchain & Editor Integration:** VS Code TextMate grammar, snippets, LSP type formatting, member auto-completion, and hover documentation.
- [x] **Tutorial & Examples:** [Tutorial 27: Native Collections — Arrays and Slices](docs/tutorials/27_native_collections_arrays_and_slices.md) and runnable example [28_native_collections.vir](docs/examples/28_native_collections.vir).

### 9. Standard Library Packages, Switch Statement & First-Class Namespacing (v0.5.9-dev)
- [x] **Standard Library Packages (`packages/`):**
  - **`packages/io`:** Unified `Reader`, `Writer`, and `Closer` structural interfaces, `File` descriptor handles, and zero-allocation `BufReader`.
  - **`packages/net`:** Non-blocking `TcpListener`, `TcpStream`, `UdpSocket`, `Datagram`, `SocketAddr`, and OS-level `Poller` (`epoll`/`kqueue`/`WSAPoll`).
  - **`packages/http`:** Transport-agnostic HTTP/1.1 engine with case-insensitive `HeaderMap`, zero-copy `Request.parse`, `Response` builder, `SseWriter`, and `HttpServer.bind`.
  - **`packages/viraserve`:** Microframework rewrite with live loopback socket dispatch, Chase-Lev actor concurrency, and sub-millisecond response latency.
- [x] **Multi-Branch Control Flow (`switch`):**
  - Native compile-time branch evaluation with automatic break semantics.
  - Multi-match comma-separated cases (`case 200, 201:`).
  - String matching via optimized C99 `vira_str_eq` branches.
  - Tagless boolean condition switching (`switch { case x >= 90: ... }`).
  - Scoped Autofree LIFO cleanup for temporary variables in case blocks.
- [x] **First-Class Package Namespacing & Module Scoping:**
  - Distinct package boundaries via `module <name>` declarations.
  - Package aliasing (`import net as mynet`) and qualified type signatures (`net.TcpStream`, `http.Response`).
  - C99 symbol mangling (`net__TcpStream`, `http__Response_json`) guaranteeing zero link collisions.
  - Cryptographic dependency management with canonical `vira.mod` manifests and deterministic SHA-256 `vira.lock` lockfiles.
- [x] **Tutorials & Runnable Examples:**
  - [Tutorial 28: Standard Library Networking, HTTP & ViraServe](docs/tutorials/28_standard_library_net_http_and_viraserve.md) & [29_stdlib_net_http_server.vir](docs/examples/29_stdlib_net_http_server.vir).
  - [Tutorial 29: Control Flow & Switch Statement Patterns](docs/tutorials/29_control_flow_and_switch_patterns.md) & [30_switch_statement.vir](docs/examples/30_switch_statement.vir).
  - [Tutorial 30: Package Namespacing, Modules & Dependency Management](docs/tutorials/30_package_namespacing_and_dependency_management.md) & [31_package_namespacing_and_modules.vir](docs/examples/31_package_namespacing_and_modules.vir).

### 10. CLI Systems Engineering, Linux Containers & Declarative GUI Suite (v0.6)
- [x] **Complete Standard Library Suite (`packages/`):**
  - Expanded stdlib with 15 production-grade packages: `os`, `exec`, `syscall`, `flag`, `log`, `strings`, `math`, `time`, `ctype`, `io`, `net`, `http`, `json`, `viraserve`, and `widget`.
  - Process management, argument parsing, leveled logging, environment variables, and monotonic timers.
- [x] **Pure-Vira Linux Container Runtime Reference:**
  - Full Linux container engine: private namespaces (`CLONE_NEWPID`, `CLONE_NEWUTS`, `CLONE_NEWNS`, `CLONE_NEWNET`, `CLONE_NEWIPC`), cgroups v2 resource capping, virtual ethernet plumbing, `/dev` character devices via `syscall.mknod`, and pipe-based IPC.
- [x] **Hardware-Accelerated IMGUI & Declarative Widget Kit (`packages/widget`):**
  - Flutter-inspired declarative UI kit (`w.App`, `w.Scaffold`, `w.Column`, `w.Row`, `w.Card`, `w.Padding`, `w.MediaQuery`).
  - Interactive controls: buttons, inputs, dropdown select, sliders, checkboxes, radio groups, progress bars, and sortable data tables.
  - Non-blocking remote URL asset loading over HTTP/TLS, disk caching, and GPU image texture rendering (`w.LoadImage`, `w.Image`, `w.FreeImage`).
  - Cross-platform packaging: Windows Win32 subsystem (`-mwindows`), macOS `.app` bundle with Retina High-DPI support, and Linux FreeDesktop bundles.
- [x] **Tutorials & Examples:**
  - Tutorials 31 through 37 (CLI apps, flags, Linux containers, data tables, GUI app bundling, Flutter-style widgets, real-world Cloud AI Hub).
  - Examples 34 through 39 with verified zero memory leaks (`active_leaks=0`).

### 11. Native C & Python FFI, Shared Library Export & ctypes (v0.7)
- [x] **Zero-Overhead Native C Interoperability (`extern "C"`):**
  - Direct C-ABI symbol calling convention without reflection or wrapper trampolines.
  - Declarations for external functions using bodiless function prototypes in `extern "C" { ... }` blocks.
  - `#include` directives for native headers and `#flag` directives for compiler/linker options.
- [x] **Deterministic Safety Boundary (`unsafe { ... }` & `@[unsafe]`):**
  - Compiler-enforced boundary preventing memory hazards from leaking into safe Vira code.
  - Calling external C functions or dereferencing pointers outside an `unsafe` block triggers a compile error.
- [x] **C-ABI Scalar Primitives & Raw Pointer Operations:**
  - Primitive types matching exact C99 types: `c_char`, `c_uchar`, `c_short`, `c_ushort`, `c_int`, `c_uint`, `c_long`, `c_ulong`, `c_longlong`, `c_ulonglong`, `c_size`, `c_ssize`, `c_float`, `c_double`, and `voidptr`.
  - Address-of (`&x`, `&mut x`) and pointer dereference (`*ptr`) operators.
  - Automatic argument marshaling: Vira `string` auto-marshals to `(const char*)`, Vira `tensor` and `slice` auto-marshal to `.data`.
- [x] **C-Compatible Struct Layouts & Opaque Handles:**
  - `@[c_layout]` struct attribute enforcing standard C alignment and padding.
  - `@[opaque]` attribute for type-safe foreign pointer handles (e.g., `sqlite3*`).
- [x] **Shared Dynamic Library Compilation (`vira build -shared`):**
  - Direct generation of `.dll` (Windows), `.so` (Linux), and `.dylib` (macOS) shared libraries.
  - Exporting C-linkage symbols with `@[export]` and custom export names with `@[export: "c_name"]`.
- [x] **Zero-Boilerplate Python `ctypes` Interop:**
  - Native loading of compiled Vira shared libraries via Python's standard `ctypes.CDLL`.
  - High-performance numeric acceleration and C-struct passing from Python.
- [x] **Tutorials & Examples:**
  - [Tutorial 38: Native C & Python FFI: Zero-Overhead Interop, Shared Libraries & ctypes](docs/tutorials/38_native_c_and_python_ffi.md).
  - Runnable example [40_native_c_and_python_ffi.vir](docs/examples/40_native_c_and_python_ffi.vir).
  - Python companion script [40_python_ffi_caller.py](docs/examples/40_python_ffi_caller.py) with verified 0 leaks (`active_leaks=0`).

### 12. Pure-Vira CLI LLM Chat Engine, Hugging Face GGUF v2/v3 & Qwen 2.5 (v0.7.1)
- [x] **Multi-Threaded Linear Projection (`.linear(w)`):** Parallel Chase-Lev work-stealing + AVX2+FMA / NEON $x W^T$ projection operating directly on GGUF row-major weights without transposing.
- [x] **Qwen 2.5 & Universal GGUF Architecture Support:**
  - Per-head NeoX split-half RoPE (`1,000,000.0` base) and interleaved LLaMA RoPE (`10,000.0` / `500,000.0`).
  - Grouped-Query Attention (7:1 GQA), additive QKV bias broadcasting (`+ bq`, `+ bk`, `+ bv`), and tied/untied `output.weight` fallback.
  - 65,536-bucket FNV-1a hash-indexed ChatML BPE tokenizer (`151,936` tokens indexed in ~120 ms) and sliding-window KV-cache context management (`--ctx`, `shift_kv_cache_window`).
- [x] **Tutorials & Examples:**
  - [Tutorial 39: Pure-Vira CLI LLM Chat Engine & Hugging Face GGUF Inference](docs/tutorials/39_pure_vira_cli_llm_chat_engine.md).
  - Runnable examples [41_cli_llm_chat_engine.vir](docs/examples/41_cli_llm_chat_engine.vir), [42_huggingface_gguf_llm_inference.vir](docs/examples/42_huggingface_gguf_llm_inference.vir), and [43_qwen2_5_gguf_chat_inference.vir](docs/examples/43_qwen2_5_gguf_chat_inference.vir).

### 13. Cross-Platform GPU Compute Acceleration (v0.8)
- [x] **Zero-SDK Dynamic GPU Driver Pipeline (`runtime/vira_gpu.h` & `packages/gpu`):**
  - Runtime dynamic discovery (`LoadLibraryA` / `dlopen`) of native OS GPU drivers (OpenCL / Vulkan) with zero compile-time SDK dependencies.
  - JIT-compiled GPU compute kernels for `.gpu_linear(w)` ($x W^T$) and `.gpu_matmul(b)` ($A B$), plus `gpu.rmsnorm`, `gpu.rope`, `gpu.softmax`, and `gpu.silu`.
- [x] **Unified 64-Byte DMA & Persistent GPU Weight Cache:**
  - Zero-copy host pointer mapping (`CL_MEM_USE_HOST_PTR`) on unified-memory GPUs and 512-slot persistent GPU weight caching for GGUF LLM weights with automatic eviction in `vira_tensor_f32_free()`.
- [x] **Actor Fault-Tolerant GPU Supervision:**
  - Automatic VRAM Out-of-Memory (`OOM`) cache eviction and transparent fallback to Vira's multi-threaded AVX2+FMA / NEON CPU pool (`gpu.oom_fallbacks()`) with `active_leaks=0`.
- [x] **Tutorials & Examples:**
  - [Tutorial 40: Cross-Platform GPU Compute Acceleration & Actor OOM Supervision](docs/tutorials/40_cross_platform_gpu_compute_acceleration.md).
  - Runnable example [44_gpu_compute_acceleration.vir](docs/examples/44_gpu_compute_acceleration.vir) and `--gpu` / `/gpu` offload controls in [43_qwen2_5_gguf_chat_inference.vir](docs/examples/43_qwen2_5_gguf_chat_inference.vir).

### 14. Runtime & Compiler Hardening (after the v0.7.7 review)
- [x] **Scheduler correctness:** fixed a mailbox race that wedged actors under concurrent senders, a teardown race, park timeouts that never fired and a lost wake-up; the worker pool now follows physical core count with platform-tuned spin iterations (`VIRA_WORKERS` override).
- [x] **Supervised faults:** out-of-bounds indexing, `pop()` on an empty slice, integer division by zero, and tensor/LLM runtime validation errors (`_vira_rt_fault`) are caught by the actor supervisor instead of killing the process; restarts reset only mutable fields.
- [x] **Defined C semantics & overflow guards:** signed overflow wraps (`-fwrapv`), arguments are evaluated left to right, each interpolated expression is evaluated once, `_vira_slice_grow` / `str.repeat` / `str.replace` guard `INT32_MAX` overflow, and `to_int` / `to_i64` enforce range and embedded-NUL checks.
- [x] **Memory safety:** fixed the autofree use-after-free, double-free and leak cases from the review; temporaries inside struct literals, array literals, and chained method receivers are freed; discarded `slice.pop()` heap results and `.clone()` temporaries are released; `vira test` enforces `active_leaks=0` by default unless `-allow-leaks` is passed.
- [x] **Single-owner ownership with moves:** storing a value copies it unless the source is never read again (including from inside `if`/`switch` branches and into slices), then it moves; arguments are borrowed; non-actor `mut` params/receivers are dereferenced cleanly on value passes.
- [x] **Strings, lexer & parser:** `+` and `+=` on strings (one allocation per chain); strict operator checks; zero-alloc scientific float exponents (`1e-12`); unary `+` on numbers; newline expression boundaries before `@`, `+`, `-`, `[`, `(`.
- [x] **Actors, `Ref[T]`, `fut.ready()` & EventBus:** actors live on the heap and never move; non-owning sendable `Ref[T]` handles (`actor.ref()`, `r.alive()`), non-blocking `Future[T].ready()`, and `packages/eventbus` (`testdata/run/26_actor_refs.vir` .. `28_future_ready.vir`).
- [x] **Deadlock avoidance:** an actor awaiting itself, or actors awaiting each other in a cycle, get an error instead of hanging.
- [x] **Runtime distribution & CLI ergonomics:** version-stamped atomic `~/.vira/runtime/<version>` header extraction (`T-115`) and order-independent flags (`vira build`/`cgen`/`check`/`test`/`fmt`).
- [x] **Tests & tooling:** ASan/UBSan golden test suite (`testdata/run/01`–`34`) and checker tests, C scheduler harness, 18/18 stdlib packages passing with `active_leaks=0`, and `scripts/bench_actors.sh`.
- [x] **Tutorials & Examples:** [Tutorial 42: Ownership, Strings & Actor Handles](docs/tutorials/42_ownership_strings_and_actor_handles.md) and [47_ownership_strings_and_actor_handles.vir](docs/examples/47_ownership_strings_and_actor_handles.vir).
- [ ] **Still open:** `return error(err)` still copies the error string; cross-platform Windows hardware re-verification since recent runtime updates.

### 15. GGUF Weight Formats & Zero-SDK CUDA Engine (v0.7.9)
- [x] **Quantized weights & fast lookup:** `gguf_load_tensor` keeps raw blocks in `_vira_rt_quant_table` next to lazy FP32 tensors (`VIRA_LAZY_HOST_FP32=1`), backed by a lock-free 256-entry TLS quant cache (`_vira_tls_qcache`).
- [x] **64-bit `numel()` & hardened GGUF loader:** widened `numel()` to `i64` (`T-011`); added file-extent bounds checks, single open file handle, hashed directory lookup, a 4-slot thread-safe LRU directory cache (`T-012`, `T-109`), and software `fp16` subnormal decoding (`T-003`/`T-004`), verified by a 700-mutant ASan+UBSan corpus and concurrent actor stress tests.
- [x] **Weight & index formats:** BF16 (`bf16 x f32` dot), Q2_K and Q3_K (`q8_K` dot), MXFP4 (E8M0-scaled FP4, `q8_0` dot; gpt-oss and DeepSeek-V4 experts) and I32 index tables (widened to `f32`; DeepSeek-V4 hash routing). GPU kernels cover F16, Q4_0–Q8_0 and Q4_K–Q6_K; the other types use CPU SIMD kernels.
- [x] **CUDA Tensor Cores, FlashAttention-2, Graphs, Mamba-2, MoE & MLA:** added Int8 Tensor Core prompt kernels (`mmqt`/`mmqb`, fused `mmqg` SwiGLU — `588 tok/s` `pp512` on Tesla T4), FlashAttention-2 (`mma.m16n8k8`) over FP16 KV cache, zero-toolkit CUDA Graphs (`m == 1` dense decode — `45 tok/s` Qwen3-32B on A100), Mamba-2 SSM kernels (`110 tok/s` Nemotron-Nano-9B-v2 on A100), zero-sync GPU MoE (`T-127`) with ungated `relu²`, and GPU MLA attention (`T-128`).
- [x] **Verified end to end** by ViraLama's reference parity suite: 70 model cases across 31 architectures against an independent NumPy forward pass, prefill and decode.
- [ ] **Still open:** native GPU kernels for `BF16`, `Q2_K`, `Q3_K`, and `MXFP4` (`T-051`); `IQ*` quantization formats (`T-030`, `T-031`); tighter GPU reference parity tolerances (`T-125`).

### 16. Enums, Maps, Closures & Generics (v0.9, unreleased, Nov 2026)
- [x] **Enums:** `enum Color { red green blue }` with optional explicit values; `Color.red` / `pkg.Color.red`, `==` / `!=`, `switch`, struct fields, slices, map keys, `int()`, and the variant name in string interpolation. Lowered to `int32_t`.
- [x] **Maps:** `map[K]V` (keys: strings, integer types, `bool`, enums) with literals, `m[k]` (zero value when missing), `m[k] = v`, `m[k] op= v`, `.len`, `has` / `delete` / `keys` / `values` / `clear`, and insertion-ordered `for k, v in m`. Value semantics like slices (independent copies, autofree), including as struct and actor fields. One untyped runtime hash map (`runtime/rt/map/map.h`) plus a generated ops table per value type.
- [x] **Closures & function types:** `fn (A, B) R` types and anonymous functions that capture by value, read-only; named functions (and `pkg.fn`) as values; closures in variables, fields, slices, maps and actor messages. Lifted to top-level functions; at runtime a function pointer plus an atomically reference-counted environment (`runtime/rt/closure/closure.h`).
- [x] **Generics:** generic functions with type arguments inferred from the arguments (through slices, maps, function types and generic structs) or explicit (`f[int](x)`), and generic structs with receiver and static methods (`Box[T]`, `fn Box[T].new()`, `pkg.Stack[string].new()`). Monomorphised in the checker: each distinct use is a concrete copy (`first__int`), so there is no runtime dispatch.
- [x] **Tests & docs:** 6 run tests under ASan/UBSan and 4 checker tests (`testdata/run/20_*` to `25_*`, `testdata/checker/07_*` to `10_*`); [Tutorial 43](docs/tutorials/43_enums_maps_closures_generics.md), [example 48](docs/examples/48_enums_maps_closures_generics.vir), Language Guide §14. All example programs check identically before and after.
- [x] **Fixed on the way:** a fresh slice or map read only for a field (`xs.clone().len`, `m.keys().len`) leaked; chunked streaming fallback in `os.read_file` supports `/proc` files that report size `0`.
- [ ] **Still open:** generic constraints (an unsupported operation is reported inside the instance, `help: in max_of[P]`); generic interfaces and enums; type parameters on methods; explicit type arguments other than plain names (`f[[]int]` must be inferred); a map element cannot be changed in place (copy out, store back); changing a map while iterating it is not detected; no `match` / `Option` integration for map lookups (`m[k]` returns the zero value; use `has`).

---

## 📈 Actor Performance vs Erlang/OTP and Akka

Measured with `scripts/bench_actor_compare.sh` (`scripts/bench/actor_compare.vir`, `actor_memory.vir`): 9 runs, `VIRA_WORKERS=4`, a 4-core Linux container, `-prod` (gcc -O3), Nov 2026.

| Metric | Vira (median, min–max) | What it measures |
| :-- | :-- | :-- |
| One-way messages, 1 sender → 1 actor | **2.54 M msg/s** (2.35–3.01 M) | 1,000,000 fire-and-forget sends, then one awaited reply |
| One-way messages, 1 sender → 64 actors | **1.78 M msg/s** (1.61–1.90 M) | same, spread round-robin |
| Ring of 1,000 actors | **1.08 M hops/s** (1.06–1.38 M) | 1,000 laps; each hop is an awaited request/reply driven by the ring's owner |
| Spawn | **0.69 µs** per actor (0.45–0.76) | create, send one message, drop; 100,000 actors |
| Memory per idle actor | **408 bytes** | VmRSS difference for 1,000,000 actors that each handled one message |

Published figures for comparison. They were **not measured here** (Erlang and Akka could not be installed or downloaded in this environment), they come from other hardware, and the vendor pages could not be re-fetched from here, so check them against the current documentation before citing them:

| Runtime | Published figure | Source |
| :-- | :-- | :-- |
| Akka | "up to 50 million msg/sec on a single machine" | akka.io; measured on a large multi-core server with many senders |
| Akka | "~2.5 million actors per GB of heap" (≈ 430 bytes per actor) | akka.io |
| Erlang/OTP | a newly spawned process uses 327 words (≈ 2.6 KB on 64-bit), including its initial 233-word heap | Erlang Efficiency Guide, "Processes" |

What this does and does not show:
- **Memory:** a Vira actor (≈ 408 bytes resident) is in the same range as Akka's published ≈ 430 bytes and about 6× smaller than a fresh Erlang process. Erlang's figure includes a private heap and stack that Vira and Akka do not pre-allocate.
- **Throughput:** Akka's 50 M msg/s is an aggregate over many cores and many concurrent senders; Vira's figures above have **one** sending thread, which is the bottleneck. They are not comparable per core, and no conclusion should be drawn from the ratio. A multi-sender benchmark is the next step (below).
- **Ring:** the classic Erlang/Akka ring has each actor forward the token to its neighbour. Vira actors cannot hold references to one another yet (single owner), so here the owner relays every hop as a request/reply, which costs two messages and a wake-up per hop.
- Run-to-run spread on this container was 1.2–1.7× (min to max); compare medians from the same machine only.

Follow-up work:
- [x] **Actor references** (`Ref[T]`, `.ref()`, `.alive()`) and `packages/eventbus` so actors can message each other directly without single-owner cycles (`testdata/run/26_actor_refs.vir`, `27_eventbus.vir`).
- [ ] **True actor-to-actor ring & multi-sender throughput benchmark** (using `Ref[T]` and one sender per worker), the shape behind Akka's and Erlang's published numbers.
- [ ] **Side-by-side runs** of the same three benchmarks in Erlang/OTP and Akka on one machine, scripted next to `bench_actor_compare.sh`.

---

## 🚀 Upcoming Strategic Frontiers

### 🧩 Phase 1: Advanced Language Features & Expressiveness (v0.9 / v0.9.x) ⏳ IN PROGRESS
*Target: Modern type system ergonomic enhancements while preserving mechanical sympathy.*

- [x] **Parametric Polymorphism (Generics):** done in v0.9 (milestone 16), with enums, maps and closures:
    ```vira
    struct RingBuffer[T] {
        items []T
        head  int
        tail  int
    }

    fn map_to[T, U](list []T, transform fn (T) U) []U
    ```
  - Monomorphised ahead-of-time: zero dynamic dispatch. Constraints, generic interfaces and method-level type parameters remain open.
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

### 📦 Phase 3: Production Registry & Ecosystem Maturity (v1.0) 📅 PLANNED
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
