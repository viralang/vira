# Vira Tutorials Curriculum & Guide Index

Welcome to the official tutorials for the **Vira Programming Language**! This structured curriculum will guide you from the basics of Vira syntax to building high-performance, concurrent, SIMD-accelerated AI pipelines.

Every tutorial includes a corresponding runnable code example in the [`docs/examples/`](../examples/) directory.

---

## Curriculum Roadmap

```
Beginner                           Intermediate                               Advanced
   │                                     │                                        │
   ├── [1. Basics & Control Flow]        ├── [3. Interfaces & Polymorphism]       ├── [5. Autofree & Bump Arenas]
   │                                     │                                        │
   └── [2. 4-Quadrant Structs & OOP]     ├── [4. Error Handling & Results]        ├── [6. SIMD Tensors & Math]
                                         │                                        │
                                         └── [25. Package Management & Vendoring] ├── [7. Actors & Supervision]
                                                                                  │
                                                                                  └── [8. Production Systems]
```

---

## Tutorial Catalog

| # | Tutorial Title | Level | Matching Code Example | Core Topics Covered |
| :-: | :--- | :-: | :--- | :--- |
| **01** | [**Getting Started, Variables & Control Flow**](./01_getting_started_and_basics.md) | Beginner | [`01_basics.vir`](../examples/01_basics.vir) | Immutability by default, `mut`, primitives (`int`, `i64`, `f32`, `bool`, `string`), string interpolation `${...}`, `if`/`else`, blank identifier `_`. |
| **02** | [**OOP, 4-Quadrant Structs & Composition**](./02_oop_and_4_quadrant_structs.md) | Intermediate | [`02_oop_composition.vir`](../examples/02_oop_composition.vir) | 4-quadrant field encapsulation (`default`, `mut:`, `pub:`, `pub mut:`), struct embedding, promoted methods/fields, value vs pointer receivers, and static factory methods. |
| **03** | [**Structural Interfaces & Dynamic Dispatch**](./03_interfaces_and_polymorphism.md) | Intermediate | [`03_interfaces.vir`](../examples/03_interfaces.vir) | Implicit interface satisfaction (no `implements` keyword), 16-byte fat-pointer boxing, vtable dispatch, and automatic destructor chaining. |
| **04** | [**Zero-Cost Error Handling with Results (`!T`)**](./04_error_handling_and_results.md) | Intermediate | [`04_result_handling.vir`](../examples/04_result_handling.vir) | Result return types (`!T`), `return error('...')`, mandatory unwrap blocks `or { ... }`, implicit `err` binding, default fallbacks, and compiler enforcement. |
| **05** | [**Deterministic Autofree & Bump Arenas**](./05_autofree_and_bump_arenas.md) | Advanced | [`05_arenas_and_autofree.vir`](../examples/05_arenas_and_autofree.vir) | How static escape analysis manages memory without a GC, classification states (`ScopedLocal`, `EscapesViaReturn`, `MovedToContainer`), and 64-byte aligned bump arenas (`arena { ... }`). |
| **06** | [**First-Class AI Tensors & SIMD Acceleration**](./06_simd_tensors_and_ai_math.md) | Intermediate | [`06_simd_tensors.vir`](../examples/06_simd_tensors.vir) | 2D numerical tensors, `@` dot-product matrix multiplication, hardware vectorization (AVX2 + FMA, ARM NEON), and multi-layer forward passes. |
| **07** | [**Actor Concurrency & Fault Supervision**](./07_actor_concurrency_and_supervisors.md) | Advanced | [`07_actors_and_supervisors.vir`](../examples/07_actors_and_supervisors.vir) | Isolated actors (`actor`), asynchronous fan-out (`<-`), `Future[T]` `.await()`, Chase-Lev work-stealing pool, `panic` containment, O(1) arena wipes, and `OneForOne` supervisor restarts. |
| **08** | [**Building Production AI Pipelines**](./08_building_production_systems.md) | Advanced | [`08_production_pipeline.vir`](../examples/08_production_pipeline.vir) | Comprehensive end-to-end integration: OOP telemetry, dynamic interfaces, parallel actor inference, `-prod` optimization flags, and zero memory leaks. |
| **09** | [**File I/O, Buffer Operations & Streams**](./09_file_io_and_streams.md) | Advanced | [`09_file_io.vir`](../examples/09_file_io.vir) | Native file I/O (`file_exists`, `read_file`, `write_file`, `append_file`, `remove_file`), zero-leak buffer autofree, and concurrent actor processing. |
| **10** | [**Threads, Actors, Synchronization & Fault Tolerance**](./10_threads_actors_fault_tolerance.md) | Advanced | [`10_threads_actors_fault_tolerance.vir`](../examples/10_threads_actors_fault_tolerance.vir) | Multicore Chase-Lev work-stealing pool, actor isolation, non-blocking dispatch (`<-`), `Future[T]` `.await()`, panic containment with `setjmp`/`longjmp`, O(1) arena wipes, and `OneForOne` supervisor restarts. |
| **11** | [**Control Flow, Loops, and Operators**](./11_control_flow_loops_operators.md) | Beginner/Intermediate | [`11_control_flow_loops_operators.vir`](../examples/11_control_flow_loops_operators.vir) | Multi-way `if`/`else if`/`else`, unified `for` loops (3-part, while-style, infinite), `break`/`continue`, arithmetic, bitwise shifts (`<<`, `>>`), compound assignments (`+=`, `-=`, `*=`, `/=`, `%=`), logical (`&&`, `\|\|`, `!`), and `@` MatMul. |
| **12** | [**First-Class AI Tensors, SIMD & Language Comparison**](./12_deep_dive_ai_tensors_and_simd.md) | Advanced | [`12_deep_dive_tensors.vir`](../examples/12_deep_dive_tensors.vir) | Deep architectural comparison with Python/PyTorch, Mojo, Rust, C++, Julia, and Go; 64-byte aligned memory layouts, AVX2+FMA/ARM NEON vectorization, zero-allocation bump arenas, and zero-GIL actor serving. |
| **13** | [**Hardware Acceleration, GPUs & Heterogeneous Computing**](./13_hardware_acceleration_gpu_npu.md) | Advanced | [`13_hardware_acceleration_gpu.vir`](../examples/13_hardware_acceleration_gpu.vir) | Discrete GPUs (CUDA/ROCm), Apple Metal Unified DMA, 64-byte PCIe cache alignment, zero-copy C-ABI handoff, and Erlang-style supervisor containment for GPU OOM faults. |
| **14** | [**Neural Network Inference on CPU SIMD vs. GPU Compute Streams**](./14_neural_network_cpu_gpu_inference.md) | Advanced | [`14_neural_network_cpu_gpu.vir`](../examples/14_neural_network_cpu_gpu.vir) | 3-layer MLP classifier forward pass (Dense + Bias + ReLU + Softmax Scaling), AVX2/NEON SIMD execution, GPU device stream actors, numerical parity verification, and CPU fallback on GPU OOM. |
| **15** | [**Loading Pretrained Model Weights (PyTorch, SafeTensors & GGUF)**](./15_loading_pretrained_models_weights.md) | Advanced | [`15_pretrained_weights_loader.vir`](../examples/15_pretrained_weights_loader.vir) | Zero-copy binary tensor ingestion (`tensor_load_bin`), bit-exact forward pass matching PyTorch reference output, asynchronous actor serving, and bidirectional export (`tensor_save_bin`). |
| **16** | [**Compiling, Building & Deploying Vira Applications**](./16_compilation_and_deployment.md) | Beginner to Advanced | [`16_compilation_and_build_modes.vir`](../examples/16_compilation_and_build_modes.vir) | Multi-stage Ahead-Of-Time (AOT) pipeline, build flags (`-prod`, `-sanitize`, `-keep-c`, `-cc`), static linking, standalone binary distribution, and production Docker containerization. |
| **17** | [**High-Performance HTTP 1.1 Web Services & Fast JSON**](./17_http_web_service_and_json.md) | Advanced | [`17_http_web_server_and_json.vir`](../examples/17_http_web_server_and_json.vir) | High-performance HTTP 1.1 networking, fast zero-copy JSON parsing (`json_get_str`, `json_get_int`, `json_get_f32`), and async actor-driven REST APIs. |
| **18** | [**Pure Vira Autograd Engine & Native SafeTensors Loader**](./18_autograd_and_safetensors.md) | Advanced | [`18_autograd_and_safetensors.vir`](../examples/18_autograd_and_safetensors.vir) | Native Hugging Face `.safetensors` parser without Python, Autograd backward gradients (`tensor_matmul_grad_b`, `tensor_relu_backward`), and SIMD SGD optimization (`tensor_sgd_step`). |
| **19** | [**Developer Tooling, LSP, VS Code Extension & Package Management**](./19_tooling_lsp_and_package_management.md) | Beginner to Advanced | `vira init`, `vira.mod`, `vira fmt`, `vira test`, `vira lsp` | Project scaffolding (`vira init`), manifest packages (`vira.mod`), multi-file compilation, code formatter (`vira fmt`), automated test runner (`vira test`), Language Server Protocol (`vira lsp`), and VS Code extension. |
| **20** | [**Pure Vira Local LLM Inference Engine & GGUF Parser**](./20_gguf_llm_inference_engine.md) | Advanced / Expert | [`19_local_llm_inference.vir`](../examples/19_local_llm_inference.vir) | Native GGUF v2/v3 binary parser, Q4_0 and Q8_0 SIMD block dequantization, RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU feed-forward networks, Causal Multi-Head Attention, and temperature-based token sampling. |
| **21** | [**Native Desktop GUI, Immediate-Mode Rendering & Concurrent Actor Dashboards**](./21_native_gui_desktop_applications.md) | Intermediate to Adv | [`20_native_gui.vir`](../examples/20_native_gui.vir) | Hardware-accelerated Immediate-Mode GUI (IMGUI), zero-flicker double-buffered GDI/X11 rendering, interactive widgets (buttons, sliders, checkboxes), and non-blocking background actor dashboards. |
| **22** | [**Declarative GUI Auto-Layout Engine: VStacks, HStacks & Cards**](./22_declarative_gui_auto_layout.md) | Intermediate to Adv | [`21_declarative_gui.vir`](../examples/21_declarative_gui.vir) | Declarative layout cursor stack, vertical columns (`gui_vstack_begin`), horizontal rows (`gui_hstack_begin`), auto-sized cards, text inputs (`gui_input`), progress bars (`gui_progress`), radio buttons (`gui_radio`), separators, badges, and zero-math widgets (`gui_btn`, `gui_slide`, `gui_check`, `gui_lbl`). |
| **23** | [**Window Navigation, Multi-Tab Views & Lifecycle Management**](./23_gui_window_navigation_and_lifecycle.md) | Intermediate to Adv | [`22_gui_navigation_lifecycle.vir`](../examples/22_gui_navigation_lifecycle.vir) | Multi-tab navigation (`gui_tab_item`), modal dialogs (`gui_modal_begin`, `gui_modal_end`), OS window lifecycle, close cancellation (`gui_cancel_close`), dynamic viewport sizing (`gui_window_width`, `gui_window_height`), and zero-leak string interpolation. |
| **24** | [**Multi-Package Architecture, Scopes & Enterprise Scaffolding**](./24_multi_package_architecture_and_scopes.md) | Intermediate to Adv | [`23_multi_package_architecture.vir`](../examples/23_multi_package_architecture.vir) | Scalable project structures (`src/config/`, `src/models/`, `src/services/`), recursive package compilation, 5-tier lexical scoping, 4-quadrant cross-package encapsulation, and zero-leak cross-package actors. |
| **25** | [**Native Package Management, Publishing & Air-Gapped Vendoring**](./25_package_management_publishing_and_vendoring.md) | Intermediate to Adv | [`24_package_management_consumer.vir`](../examples/24_package_management_consumer.vir) | Manifests (`vira.mod`), deterministic lockfiles (`vira.lock`), SHA-256 tree hashing, adding dependencies (`vira add`), offline vendoring (`vira vendor`), GitHub publishing, and 6-tier import resolution. |
| **26** | [**Actor System, Fault Tolerance & High-Concurrency Benchmarks**](./26_actor_system_tradeoffs_and_benchmarks.md) | Advanced / Expert | [`25_actors_banking_fault_tolerance.vir`](../examples/25_actors_banking_fault_tolerance.vir) | Deep trade-off analysis (lightweight actors vs OS threads, per-actor bump arenas vs shared heap, lock-free Chase-Lev deques vs centralized mailboxes, `setjmp`/`longjmp` panic isolation), 10M actors/sec spawn, and reproducible high-concurrency stress benchmarks. |

---

## Quick Start: Testing the Toolchain

To verify that your compiler toolchain is ready:

```powershell
# Run the first tutorial example
.\vira.exe run docs/examples/01_basics.vir

# Check and inspect autofree escape analysis for an example
.\vira.exe check docs/examples/02_oop_composition.vir

# Compile an optimized standalone native binary
.\vira.exe build -prod -o app.exe docs/examples/08_production_pipeline.vir
.\app.exe
```
