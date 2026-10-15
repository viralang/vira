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
   └── [2. 4-Quadrant Structs & OOP]     └── [4. Error Handling & Results]        ├── [6. SIMD Tensors & Math]
                                                                                  │
                                                                                  ├── [7. Actors & Supervision]
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
| **19** | [**Developer Tooling, LSP, VS Code Extension & Package Management**](./19_tooling_lsp_and_package_management.md) | Beginner to Advanced | `vira init`, `vex.mod`, `vira fmt`, `vira test`, `vira lsp` | Project scaffolding (`vira init`), manifest packages (`vex.mod`), multi-file compilation, code formatter (`vira fmt`), automated test runner (`vira test`), Language Server Protocol (`vira lsp`), and VS Code extension. |
| **20** | [**Pure Vira Local LLM Inference Engine & GGUF Parser**](./20_gguf_llm_inference_engine.md) | Advanced / Expert | [`19_local_llm_inference.vir`](../examples/19_local_llm_inference.vir) | Native GGUF v2/v3 binary parser, Q4_0 and Q8_0 SIMD block dequantization, RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU feed-forward networks, Causal Multi-Head Attention, and temperature-based token sampling. |

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
