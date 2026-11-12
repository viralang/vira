# Vira Runnable Examples

This directory contains standalone, self-contained, runnable `.vir` example programs covering every feature and pattern in the language.

All examples can be executed directly using the `vira` compiler CLI:

```powershell
# Run directly
vira run docs/examples/<filename>.vir

# Or build to an optimized native binary
.\vira.exe build -prod -o app.exe docs/examples/<filename>.vir
.\app.exe
```

---

## Catalog of Examples

| File | Topic & Focus Area | Features Demonstrated |
| :--- | :--- | :--- |
| **[`01_basics.vir`](./01_basics.vir)** | Syntax, Variables & Control Flow | Immutability by default, `mut`, primitives (`int`, `i64`, `f32`, `bool`, `string`), string interpolation `${...}`, `if`/`else`, and blank identifier `_`. |
| **[`02_oop_composition.vir`](./02_oop_composition.vir)** / **[`02_timestamp.vir`](./02_timestamp.vir)** | OOP, Encapsulation & Modular Composition | 4-quadrant struct access levels (`default`, `mut:`, `pub:`, `pub mut:`), modular multi-file composition via `import`, struct embedding with promoted methods/fields, and value vs mutable pointer receivers. |
| **[`03_interfaces.vir`](./03_interfaces.vir)** | Structural Interfaces & VTables | Implicit interface satisfaction (no `implements` keyword), 16-byte fat-pointer boxing (`._object`, `._vtable`), dynamic polymorphic dispatch, and automatic destructor chaining. |
| **[`04_result_handling.vir`](./04_result_handling.vir)** | Zero-Cost Error Handling | Result types (`!T`), `return error('...')`, mandatory unwrap blocks `or { ... }`, default fallbacks, and compile-time unhandled error prevention. |
| **[`05_arenas_and_autofree.vir`](./05_arenas_and_autofree.vir)** | Deterministic Memory Management | Static LIFO scope-exit autofree without a GC, and high-performance 64-byte aligned bump arenas (`arena { ... }`) with O(1) bulk slab teardown. |
| **[`06_simd_tensors.vir`](./06_simd_tensors.vir)** | First-Class AI Tensors & SIMD | 2D tensor allocations (`tensor[f32]`), matrix multiplication with the `@` operator, hardware AVX2/NEON vectorization, and multi-layer forward passes. |
| **[`07_actors_and_supervisors.vir`](./07_actors_and_supervisors.vir)** | Actor Concurrency & Fault Tolerance | Multi-threaded actors (`actor`), non-blocking asynchronous message dispatch (`<-`), `Future[T]` `.await()`, Chase-Lev work-stealing, and `setjmp`/`longjmp` fault trapping with `OneForOne` supervisor restarts. |
| **[`08_production_pipeline.vir`](./08_production_pipeline.vir)** | End-to-End AI Production Pipeline | Integrated real-world pipeline combining OOP telemetry, dynamic interfaces, parallel actor inference, error propagation, and 0 memory leaks. |
| **[`09_file_io.vir`](./09_file_io.vir)** | Native File I/O & Buffers | File existence checking, atomic writes, appending records, reading into autofreed buffers, and file deletion with Result error handling. |
| **[`10_threads_actors_fault_tolerance.vir`](./10_threads_actors_fault_tolerance.vir)** | Threads, Actors, Synchronization & Fault Tolerance | Multicore Chase-Lev work-stealing pool, actor isolation, non-blocking dispatch (`<-`), `Future[T]` `.await()`, panic containment with `setjmp`/`longjmp`, O(1) arena wipes, and `OneForOne` supervisor restarts. |
| **[`11_control_flow_loops_operators.vir`](./11_control_flow_loops_operators.vir)** | Control Flow, Loops & Operators | Multi-way branching (`if`/`else if`/`else`), unified `for` loops (3-part C-style, while-style, infinite), `break`/`continue`, arithmetic, bitwise shifts (`<<`, `>>`), compound assignments (`+=`, `-=`, `*=`, `/=`, `%=`), logical (`&&`, `\|\|`, `!`), and `@` MatMul. |
| **[`12_deep_dive_tensors.vir`](./12_deep_dive_tensors.vir)** | Deep Dive AI Tensors & SIMD | Multi-layer MLP forward pass, 64-byte aligned tensors, hardware AVX2+FMA vectorization, zero-allocation bump arena inference, and zero-GIL multicore actor serving. |
| **[`13_hardware_acceleration_gpu.vir`](./13_hardware_acceleration_gpu.vir)** | Hardware Acceleration & GPU Queues | Discrete GPUs (CUDA/ROCm), Apple Metal Unified DMA, 64-byte PCIe cache alignment, zero-copy C-ABI handoff, and Erlang-style supervisor containment for GPU OOM faults. |
| **[`14_neural_network_cpu_gpu.vir`](./14_neural_network_cpu_gpu.vir)** | Neural Network CPU SIMD vs. GPU Streams | 3-layer MLP classifier forward pass (Dense + Bias + ReLU + Softmax Scaling), AVX2/NEON SIMD execution, GPU device stream actors, numerical parity verification, and CPU fallback on GPU OOM. |
| **[`15_pretrained_weights_loader.vir`](./15_pretrained_weights_loader.vir)** | Loading Pretrained PyTorch & SafeTensors Weights | Zero-copy binary tensor ingestion (`tensor_load_bin`), bit-exact forward pass matching PyTorch reference output, asynchronous actor serving, and bidirectional export (`tensor_save_bin`). |
| **[`16_compilation_and_build_modes.vir`](./16_compilation_and_build_modes.vir)** | Compilation, Build Modes & Production Deployment | Demonstrates build modes (dev `-O2`, prod `-O3`, sanitizer `-sanitize`), static linking, standalone binary deployment, and compiler telemetry. |
| **[`17_http_web_server_and_json.vir`](./17_http_web_server_and_json.vir)** | High-Performance HTTP 1.1 Web Engine & JSON Microservices | Fast JSON field parsing (`json_get_str`, `json_get_int`, `json_get_f32`), HTTP 1.1 response generation (`http_response`), actor-based request processing, and zero memory leaks. |
| **[`18_autograd_and_safetensors.vir`](./18_autograd_and_safetensors.vir)** | Pure Vira Autograd Engine & Native SafeTensors Loader | Native zero-copy loading of Hugging Face `.safetensors` files (`safetensors_load`), SIMD Autograd backpropagation (`tensor_matmul_grad_b`, `tensor_relu_backward`), and hardware in-place SGD optimization (`tensor_sgd_step`). |
| **[`19_local_llm_inference.vir`](./19_local_llm_inference.vir)** | Pure Vira Local LLM Inference Engine & GGUF Parser | Native GGUF v2/v3 binary parser, Q4_0/Q8_0 SIMD dequantization, RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU activations, Causal Self-Attention, and temperature-based token sampling. |
| **[`20_native_gui.vir`](./20_native_gui.vir)** | Hardware-Accelerated Native GUI & Actor Dashboard | Immediate-Mode GUI (IMGUI), double-buffered 60 FPS Win32/X11 rendering, interactive widgets (buttons, sliders, checkboxes), and real-time concurrent background actor metrics. |
| **[`21_declarative_gui.vir`](./21_declarative_gui.vir)** | Declarative GUI Auto-Layout Engine | Vertical Stacks (`gui_vstack_begin`), horizontal rows (`gui_hstack_begin`), auto-sized cards, text inputs (`gui_input`), progress bars (`gui_progress`), radio buttons (`gui_radio`), separators, badges, and zero-math widgets (`gui_btn`, `gui_slide`, `gui_check`, `gui_lbl`). |
| **[`22_gui_navigation_lifecycle.vir`](./22_gui_navigation_lifecycle.vir)** | Window Navigation, Tabs, Modals & Lifecycle Control | Multi-tab bar routing (`gui_tab_item`), backdrop modal dialogs (`gui_modal_begin`/`gui_modal_end`), close interception (`gui_cancel_close`), viewport metrics (`gui_window_width`/`gui_window_height`), and zero-leak 60 FPS string interpolation. |
| **[`23_multi_package_architecture.vir`](./23_multi_package_architecture.vir)** | Multi-Package Architecture & Scopes | Enterprise 5-tier lexical scoping, cross-package 4-quadrant encapsulation, actor pipelining, and zero-leak SIMD tensor computing. |
| **[`24_package_management_consumer.vir`](./24_package_management_consumer.vir)** | Native Package Management & Vendoring | Consuming external packages (`import "vira_test"`), Vec2 arithmetic, structured logging, offline vendoring, and lockfile reproducibility. |
| **[`25_actors_banking_fault_tolerance.vir`](./25_actors_banking_fault_tolerance.vir)** | Resilient Banking & Fault-Tolerant Actors | Erlang-style fault containment, OneForOne supervisor state reset, panic isolation (`setjmp`/`longjmp`), and transaction retry with 0 memory leaks. |
| **[`26_actors_worker_pool_fanout.vir`](./26_actors_worker_pool_fanout.vir)** | Matrix Compute Worker Pool & Fan-Out/Fan-In | Parallel tensor fan-out (`<-`), Chase-Lev multicore work-stealing, work-helping `.await()`, and zero-leak O(1) private actor scratch arenas. |
| **[`27_actors_ring_pipeline.vir`](./27_actors_ring_pipeline.vir)** | Chained Actor Pipeline & Token Passing | High-throughput sequential and interleaved token passing across independent pipeline stages, mailbox serialization, and 0 memory leaks. |
| **[`28_native_collections.vir`](./28_native_collections.vir)** | Native Collections: Fixed Arrays (`[N]T`) & Slices (`[]T`) | Stack-allocated fixed arrays, dynamic slices with geometric growth, bounds check panics, safe retrieval via Result Monad (`.get() or {}`), for-in iteration, and zero-leak Autofree validation. |
| **[`29_stdlib_net_http_server.vir`](./29_stdlib_net_http_server.vir)** | Standard Library Networking & HTTP Server | `packages/net` (TCP streams, loopback listener), `packages/http` (HttpServer, request parsing, response formatting), and switch statement routing with 0 leaks. |
| **[`30_switch_statement.vir`](./30_switch_statement.vir)** | Switch Statements & Pattern Branching | Multi-branch switching on integer codes, string routes, comma-separated multiple matches, tagless boolean condition switches, and deterministic autofree safety. |
| **[`31_package_namespacing_and_modules.vir`](./31_package_namespacing_and_modules.vir)** | First-Class Package Namespacing & Modules | Module declarations (`module http`), qualified type signatures (`net.TcpStream`), qualified static factory calls, and cryptographic `vira.lock` dependency locking. |
| **[`32_actor_detached_and_lifecycle.vir`](./32_actor_detached_and_lifecycle.vir)** | Actor Detached Sends & Lifecycle Management | Detached fire-and-forget messaging (`w <- method()`), zero-alloc synchronous execution (`w <- method() or { ... }`), supervisor restart count reset on healthy run, `VIRA_ACTOR_STOPPED` dead-letter rejection, and 0 memory leaks. |
| **[`33_nd_tensors_and_stdlib_json.vir`](./33_nd_tensors_and_stdlib_json.vir)** | 1D-4D Strided Tensors & Pure-Vira JSON | 4D batched attention `@`, zero-copy arena reshape, trailing-dimension bias broadcast, pure-Vira JSON serialization, and 0 memory leaks. |
| **[`34_basic_cli_app.vir`](./34_basic_cli_app.vir)** | Basic CLI Application Development | Command-line argument inspection via `os.args()`, environment variables (`os.getenv`/`os.setenv`), path helpers, file I/O with deterministic cleanup via `defer`, and 0 memory leaks. |
| **[`35_advanced_cli_flags_syscall_exec.vir`](./35_advanced_cli_flags_syscall_exec.vir)** | Advanced CLI Flags, Logging, Syscall & Exec | Production flag parsing with `flag.FlagSet`, leveled logging with `log.Logger`, child process execution via `exec.Command`, raw UNIX pipe IPC via `syscall.pipe`, and 0 memory leaks. |
| **[`36_linux_container_runtime_reference.vir`](./36_linux_container_runtime_reference.vir)** | Linux Container Runtime Reference Implementation | Full Linux container engine: private namespaces (`CLONE_NEWPID/NEWUTS/NEWNS/NEWNET/NEWIPC`), cgroups v2 resource capping, virtual ethernet plumbing, `/dev` character devices via `syscall.mknod`, and 0 memory leaks. |
| **[`37_gui_image_and_asset_loading.vir`](./37_gui_image_and_asset_loading.vir)** | Native GUI Asset & Image Subsystem | Loading PNG/JPEG/BMP assets (`gui_load_image`), intrinsic resolution queries (`gui_image_width`/`height`), auto-layout flow (`gui_image`), absolute canvas drawing (`gui_draw_image`), alpha transparency blending, interactive zoom slider, preset scaling, and zero memory leaks. |
| **[`38_gui_remote_url_image_loading.vir`](./38_gui_remote_url_image_loading.vir)** | Remote URL Image Loading & HTTP Header Inspection | Non-blocking remote asset download, custom HTTP request headers (`Authorization: Bearer`, `User-Agent`, `Accept`), 60 FPS unblocked GUI, disk-backed caching, and alpha-blended canvas rendering. |
| **[`39_gui_realworld_cloud_ai_hub.vir`](./39_gui_realworld_cloud_ai_hub.vir)** | Real-World GUI Apps: HTTP, Images & Widget Suite | Production Cloud AI Hub desktop app: live HTTP REST API querying over TLS, zero-alloc JSON parsing, GPU image texture caching & rendering, complete widget catalog, and 0 leaks. |
| **[`40_native_c_and_python_ffi.vir`](./40_native_c_and_python_ffi.vir)** / **[`40_python_ffi_caller.py`](./40_python_ffi_caller.py)** | Native C & Python FFI, Shared Libraries & ctypes | Direct C-ABI interop, `#include`, `extern "C"`, mandatory `unsafe { ... }`, C scalar types (`c_int`, `c_double`, `voidptr`), raw pointers (`&`, `&mut`, `*`), `@[c_layout]` structs, exporting shared libraries (`@[export]` & `vira build -shared`), and Python `ctypes` interop with 0 leaks. |
| **[`41_cli_llm_chat_engine.vir`](./41_cli_llm_chat_engine.vir)** | Pure-Vira CLI LLM Chat Engine (Zero FFI) | Pure-Vira GGUF v3 header & `tokenizer.ggml.tokens` vocabulary parser (`os.read_bytes`), word/subword tokenizer, stateful KV-cache attention (`k_cache.copy_row`, `k_cache.slice_rows`), RoPE, RMSNorm, SwiGLU inside `arena {}`, repetition penalty (`logits.get`/`set`), CLI flags (`flag.FlagSet`), and interactive REPL (`io.prompt`) with 0 leaks. |
| **[`42_huggingface_gguf_llm_inference.vir`](./42_huggingface_gguf_llm_inference.vir)** | Pretrained Hugging Face 5-Layer LLaMA GGUF Runner (Zero FFI) | Auto-downloads `ggml-org/test-model-stories260K` (`models/stories260K.gguf`), parses 512 SentencePiece BPE tokens & IEEE-754 merge scores, runs 5-layer 8Q/4KV Grouped-Query Attention (GQA) + interleaved RoPE + SwiGLU at >3,000 tok/s with 0 leaks. |
| **[`43_qwen2_5_gguf_chat_inference.vir`](./43_qwen2_5_gguf_chat_inference.vir)** | Qwen 2.5 (`qwen2`) GGUF ChatML Inference Engine (Zero FFI) | Loads `models/qwen2_5_instruct.gguf` (or `--download-hf` for `Qwen/Qwen2.5-0.5B-Instruct-GGUF`), parses `qwen2` metadata & GPT-2 ChatML tokens (`<\|im_start\|>`/`<\|im_end\|>`), runs QKV bias addition (`attn_q/k/v.bias`), 1M NeoX half-split RoPE, 8Q/2KV GQA, and `Q4_0`/`Q8_0` SIMD dequantization with 0 leaks. |
| **[`44_gpu_compute_acceleration.vir`](./44_gpu_compute_acceleration.vir)** | Cross-Platform GPU Compute Acceleration & OOM Supervision | Dynamic GPU backend discovery (`gpu.info()`, `gpu.vram_mb()`, `gpu.allocated_kb()`), persistent GPU weight caching (`x.gpu_linear(w)`), GPU transformer ops (`rmsnorm`, `rope`, `softmax`, `silu`), and actor-supervised VRAM OOM fallback (`gpu.simulate_oom`) to CPU SIMD. |
| **[`45_gui_qwen2_5_gguf_chat_studio.vir`](./45_gui_qwen2_5_gguf_chat_studio.vir)** | Native GUI Qwen 2.5 GGUF Chat Studio | Non-blocking 1-layer-per-frame GGUF streaming loader (`Q4_0`/`Q8_0`/`F32` + QKV biases), live token-by-token 60 FPS GUI ChatML streaming with `tok/s` telemetry, interactive temperature/max-tokens sliders, live CPU/GPU toggle, and KV cache reset. |
| **[`46_qwen2_5_t4_large_ctx_benchmark.vir`](./46_qwen2_5_t4_large_ctx_benchmark.vir)** | Qwen 2.5 (3B / 7B) Large-Context NVIDIA Tesla T4 GPU Benchmark | Resumable Hugging Face GGUF download (`.part` + `io.rename`), `7B`/`3B`/`1.5B` (`Q4_K_M`/`Q4_0`/`Q8_0`) presets, chunked batched prefill scaling (`pp` up to `4096+`, WMMA TFLOPS), long-context incremental KV decode sweeps (`ctx` `8192`–`32768`, GDDR6 GB/s), `--verify` ChatML check, CUDA/OpenCL/CPU flags, and 0 leaks. |
| **[`47_ownership_strings_and_actor_handles.vir`](./47_ownership_strings_and_actor_handles.vir)** | Ownership, Strings & Actor Handles | String `+` / `+=`, copies vs moves, borrowed arguments, an actor returned from a function with a message still queued, a growing slice of 100 actors, and a two-actor wait cycle broken with `deadlock avoided: actors are waiting on each other`. Companion to [Tutorial 42](../tutorials/42_ownership_strings_and_actor_handles.md). |

---

## Verifying All Examples

You can run each example in sequence from your shell:

```powershell
vira run docs/examples/01_basics.vir
vira run docs/examples/02_oop_composition.vir
vira run docs/examples/03_interfaces.vir
vira run docs/examples/04_result_handling.vir
vira run docs/examples/05_arenas_and_autofree.vir
vira run docs/examples/06_simd_tensors.vir
vira run docs/examples/07_actors_and_supervisors.vir
vira run docs/examples/08_production_pipeline.vir
vira run docs/examples/09_file_io.vir
vira run docs/examples/10_threads_actors_fault_tolerance.vir
vira run docs/examples/11_control_flow_loops_operators.vir
vira run docs/examples/12_deep_dive_tensors.vir
vira run docs/examples/13_hardware_acceleration_gpu.vir
vira run docs/examples/14_neural_network_cpu_gpu.vir
vira run docs/examples/15_pretrained_weights_loader.vir
vira run docs/examples/16_compilation_and_build_modes.vir
vira run docs/examples/17_http_web_server_and_json.vir
vira run docs/examples/18_autograd_and_safetensors.vir
vira run docs/examples/19_local_llm_inference.vir
vira run docs/examples/20_native_gui.vir
vira run docs/examples/21_declarative_gui.vir
vira run docs/examples/22_gui_navigation_lifecycle.vir
vira run docs/examples/23_multi_package_architecture.vir
vira run docs/examples/24_package_management_consumer.vir
vira run docs/examples/25_actors_banking_fault_tolerance.vir
vira run docs/examples/26_actors_worker_pool_fanout.vir
vira run docs/examples/27_actors_ring_pipeline.vir
vira run docs/examples/28_native_collections.vir
vira run docs/examples/29_stdlib_net_http_server.vir
vira run docs/examples/30_switch_statement.vir
vira run docs/examples/31_package_namespacing_and_modules.vir
vira run docs/examples/32_actor_detached_and_lifecycle.vir
vira run docs/examples/33_nd_tensors_and_stdlib_json.vir
vira run docs/examples/34_basic_cli_app.vir
vira run docs/examples/35_advanced_cli_flags_syscall_exec.vir
vira run docs/examples/36_linux_container_runtime_reference.vir
vira run docs/examples/37_gui_image_and_asset_loading.vir
vira run docs/examples/38_gui_remote_url_image_loading.vir
vira run docs/examples/39_gui_realworld_cloud_ai_hub.vir
vira run docs/examples/40_native_c_and_python_ffi.vir
python docs/examples/40_python_ffi_caller.py
vira run docs/examples/41_cli_llm_chat_engine.vir
vira run docs/examples/42_huggingface_gguf_llm_inference.vir
vira run docs/examples/43_qwen2_5_gguf_chat_inference.vir
vira run docs/examples/44_gpu_compute_acceleration.vir
vira run docs/examples/45_gui_qwen2_5_gguf_chat_studio.vir
.\vira.exe build -prod -o bench46.exe docs/examples/46_qwen2_5_t4_large_ctx_benchmark.vir
.\bench46.exe
```

