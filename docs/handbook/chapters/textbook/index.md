# The Vira Programming Language: Official Textbook & Cookbook

Welcome to **The Vira Programming Language**, an end-to-end guide and cookbook modeled after *The Rust Programming Language* and *The Go Programming Language*, tailored to Vira's deterministic autofree memory model, 4-quadrant encapsulation, supervised actor concurrency, and built-in AI/GUI runtimes.

Each chapter pairs core language concepts in `docs/handbook/chapters/textbook/` with runnable, hand-tested `.vir` recipes in `docs/handbook/chapters/code/`.

---

## Part I: Getting Started & Common Programming Concepts

### [Chapter 1: Getting Started](chapter01.md)
- **1.1.** Installation & C99 Backend Setup (`gcc`, `clang`, `mingw-w64`)
- **1.2.** Hello, World! (`module main`, `fn main()`, `println`)
- **1.3.** The Vira CLI Toolchain (`vira run`, `vira build`, `vira check`, `vira cgen`, `vira fmt`)

### [Chapter 2: Tutorial — A Complete CLI Mini-Project](chapter02.md)
- **2.1.** Setting Up a New Project (`vira init`)
- **2.2.** Reading Input & Command-Line Arguments (`io.prompt`, `os.args`)
- **2.3.** Handling Potential Failure with `!T` and `or { ... }`
- **2.4.** Looping, Branching, and Zero-Leak Verification (`active_leaks=0`)

### [Chapter 3: Variables, Data Types & Operators](chapter03.md)
- **3.1.** Variables and Immutability by Default (`:=`, `mut`, and the Blank Identifier `_`)
- **3.2.** Compile-Time Constants (`const` and `pub const`)
- **3.3.** Scalar Primitive Types (`int`, `i64`, `u8`/`byte`, `f32`, `f64`, `bf16`, `bool`)
- **3.4.** Numeric Literals (`0x`, `0o`, `0b`) and Explicit Type Casts
- **3.5.** Operators, Precedence, Overflow Wrapping, and Strict Type Rules

### [Chapter 4: Functions, Control Flow & Pattern Switching](chapter04.md)
- **4.1.** Defining Functions, Parameters, and Return Values
- **4.2.** Conditional Branching (`if`, `else if`, `else`)
- **4.3.** Multi-Branch `switch` Statements (Values, Comma Cases, String Matching, Tagless Conditions)
- **4.4.** Unified `for` Loops (While-Style, 3-Part C-Style, Infinite, `for ... in`, `break`, `continue`)
- **4.5.** Deterministic Scope Cleanup with `defer`

---

## Part II: Ownership, Deterministic Memory & Collections

### [Chapter 5: Understanding Ownership, Autofree & Bump Arenas](chapter05.md)
- **5.1.** Memory Safety Without a Garbage Collector or Borrow-Checker Annotations
- **5.2.** The Single-Owner Rule: Automatic Deep Copies vs. Last-Use Moves
- **5.3.** Borrowed Function Arguments and Returning Ownership
- **5.4.** Static Escape Analysis (`StateScopedLocal`, `StateEscapesViaReturn`, `StateMovedToContainer`)
- **5.5.** High-Throughput Batch Memory with Bump Arenas (`arena { ... }`)
- **5.6.** Auditing Lifetimes and Leak Telemetry (`vira check` and `-sanitize`)

### [Chapter 6: Strings, Arrays, Slices & Maps](chapter06.md)
- **6.1.** UTF-8 Strings (`string`), Interpolation (`${...}`), Concatenation (`+`, `+=`), and Zero-Copy Views
- **6.2.** Fixed-Size Stack Arrays (`[N]T`) and Value Semantics
- **6.3.** Dynamic Growable Slices (`[]T`), `push`, `pop`, `clone`, and Safe Indexing (`.get(i) or { ... }`)
- **6.4.** Insertion-Ordered Hash Maps (`map[K]V`), Lookups, Mutation, and Key/Value Iteration

---

## Part III: Structs, Enums, Interfaces & Error Handling

### [Chapter 7: Structs, 4-Quadrant Encapsulation & Composition](chapter07.md)
- **7.1.** Defining and Instantiating Structs
- **7.2.** The 4 Access Quadrants (`default` Private Immutable, `mut:`, `pub:`, `pub mut:`)
- **7.3.** Value Receivers vs. Mutable Receivers (`fn (s T)` vs. `fn (mut s T)`)
- **7.4.** Associated Static Factory Methods (`fn Type.new(...)`)
- **7.5.** Struct Embedding, Composition, and Promoted Fields/Methods

### [Chapter 8: Enums & Structural Interfaces](chapter08.md)
- **8.1.** Defining Enums, Explicit Integer Discriminants, and String Formatting
- **8.2.** Structural Interfaces and Implicit Satisfaction (`interface`, `mut:` methods)
- **8.3.** 16-Byte Fat-Pointer Boxing, Dynamic vtable Dispatch, and Automatic Box Cleanup

### [Chapter 9: Error Handling (`!T` Results & Fault Boundaries)](chapter09.md)
- **9.1.** Recoverable Errors with Result Types (`!T`) and `return error('...')`
- **9.2.** Mandatory Unwrapping at Call Sites (`or { ... }`, Fallback Values, and the Implicit `err`)
- **9.3.** Propagating Errors Up the Call Stack (`!`)
- **9.4.** Unrecoverable Faults (`panic`) and When to Use Result vs. Panic

---

## Part IV: Generics, Closures, Modules & Testing

### [Chapter 10: Generics & Closures](chapter10.md)
- **10.1.** Generic Functions (`fn name[T, U](...)`) and Type Argument Inference
- **10.2.** Generic Structs (`struct Box[T]`) and Generic Methods
- **10.3.** How Zero-Cost Monomorphization Works
- **10.4.** Anonymous Functions, First-Class Function Types (`fn(A) R`), and Value-Captured Closures

### [Chapter 11: Modules, Packages & Dependency Management](chapter11.md)
- **11.1.** Modules (`module`) and Cross-Module Visibility (`pub`)
- **11.2.** Importing Packages (`import`, `import ... as ...`) and 6-Tier Import Resolution
- **11.3.** Multi-Package Project Layout (`vira.mod`, `src/`, `tests/`)
- **11.4.** Decentralized Dependencies, Lockfiles (`vira.lock`), and Air-Gapped Vendoring (`vira vendor`)

### [Chapter 12: Automated Testing, Diagnostics & Tooling](chapter12.md)
- **12.1.** Writing Unit Tests (`*_test.vir`) and Running `vira test`
- **12.2.** Memory Sanitizer Verification (`-sanitize`) and Zero-Leak Assertions
- **12.3.** Inspecting Generated C99 (`vira cgen`, `-keep-c`) and IDE Integration (`vira lsp`)

---

## Part V: Concurrency — Actors, Supervision & Event Bus

### [Chapter 13: Fearless Actor Concurrency](chapter13.md)
- **13.1.** Defining Actors (`actor`) and Single-Owner Handle Semantics
- **13.2.** Compile-Time Actor State Isolation
- **13.3.** Message Passing Modes (`<-`): Async `Future[T]` (`.await()`, `.ready()`), Fire-and-Forget, and Synchronous Calls
- **13.4.** Inside the Lock-Free Chase-Lev Work-Stealing Scheduler

### [Chapter 14: Fault Supervision, Actor References (`Ref[T]`) & Event Bus](chapter14.md)
- **14.1.** Erlang-Style Fault Containment (`setjmp`/`longjmp`, LIFO `defer` Unwinding, $O(1)$ Arena Wipe)
- **14.2.** `OneForOne` Supervisor Restarts and Dead-Letter Stopping (`VIRA_ACTOR_STOPPED`)
- **14.3.** Wait-For Graph Cycle Detection and Automatic Deadlock Avoidance
- **14.4.** Non-Owning Actor References (`Ref[T]`, `.ref()`, `.alive()`) for Direct Actor-to-Actor Topologies
- **14.5.** Decoupled Pub/Sub and Request/Reply with `import eventbus`

---

## Part VI: Systems Programming, Networking & FFI

### [Chapter 15: I/O, Filesystem, CLI Applications & Subprocesses](chapter15.md)
- **15.1.** File I/O Builtins and Buffered Streams (`import io`)
- **15.2.** Environment, Arguments, and Paths (`import os`)
- **15.3.** Declarative Command-Line Flag Parsing (`import flag`) and Structured Logging (`import log`)
- **15.4.** Spawning Subprocesses (`import exec`) and Raw Kernel Syscalls (`import syscall`)

### [Chapter 16: Networking, HTTP Servers & Pure-Vira JSON](chapter16.md)
- **16.1.** TCP Listeners, Streams, UDP Sockets, and Non-Blocking Polling (`import net`)
- **16.2.** Fast RFC 8259 JSON Parsing and $O(N)$ `JsonBuilder` (`import json`)
- **16.3.** HTTP/1.1 Clients, Servers, and Server-Sent Events (`import http`)
- **16.4.** Building Actor-Powered Web Services (`import viraserve`)

### [Chapter 17: Unsafe Vira & Native C / Python FFI](chapter17.md)
- **17.1.** The `unsafe { ... }` Boundary and Raw Pointers (`&x`, `&mut x`, `*ptr`, `voidptr`)
- **17.2.** C-ABI Primitive Types, `@[c_layout]` Structs, and `@[opaque]` Handles
- **17.3.** Calling C Libraries with `#include`, `#flag`, and `extern "C"`
- **17.4.** Exporting Shared Libraries (`@[export]`, `vira build -shared`) for Python `ctypes`, C, and Rust

---

## Part VII: AI Tensors, GPU Compute & Native Desktop GUI

### [Chapter 18: First-Class AI Tensors, SIMD & Autograd](chapter18.md)
- **18.1.** 1D–4D Strided Tensors (`tensor[f32]`), Views, and Reshaping
- **18.2.** Batched & Broadcast Matrix Multiplication (`@`) and Multi-Threaded AVX2 / NEON SIMD
- **18.3.** Reverse-Mode Autograd and Training Neural Networks in Pure Vira
- **18.4.** Loading Pretrained Weights (Binary, Hugging Face `.safetensors`, and Quantized GGUF)

### [Chapter 19: Zero-SDK GPU Acceleration & Local LLM Inference](chapter19.md)
- **19.1.** Zero-SDK GPU Architecture (`import gpu`, Dynamic CUDA Driver PTX JIT & OpenCL Fallback)
- **19.2.** VRAM Weight Caching, Tensor Core (`WMMA`) Dispatch, and Supervised OOM Fallbacks
- **19.3.** Building a Quantized GGUF Local LLM Chat Engine (RoPE, RMSNorm, SwiGLU, GQA, KV Cache)

### [Chapter 20: Native Desktop GUI & Flutter-Style Declarative Widgets](chapter20.md)
- **20.1.** Immediate-Mode 60 FPS Native Windowing (X11, Win32, Cocoa, Raylib)
- **20.2.** Auto-Layout Stacks (`VStack`, `HStack`, `Card`, `ScrollView`) and Controls
- **20.3.** Declarative UI with `import widget` (`w.App`, `w.Scaffold`, `w.DataTable`, Themes, Async Images)
- **20.4.** Cross-Platform App Scaffolding & Native Packaging (`vira init-gui`, `vira bundle`)

---

## Appendices

- **Appendix A:** Keywords, Operators & Tokens Reference
- **Appendix B:** Built-in Functions & Standard Library Quick Reference
- **Appendix C:** C99 Lowering & Runtime Memory Layout Guide
- **Appendix D:** Syntax Evolution Log (Notes & Language Improvements Made While Writing This Book)

