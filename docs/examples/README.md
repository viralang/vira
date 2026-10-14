# Vex Runnable Examples

This directory contains standalone, self-contained, runnable `.vex` example programs covering every feature and pattern in the language.

All examples can be executed directly using the `vex` compiler CLI:

```powershell
# Run directly
.\vex.exe run docs/examples/<filename>.vex

# Or build to an optimized native binary
.\vex.exe build -prod -o app.exe docs/examples/<filename>.vex
.\app.exe
```

---

## Catalog of Examples

| File | Topic & Focus Area | Features Demonstrated |
| :--- | :--- | :--- |
| **[`01_basics.vex`](./01_basics.vex)** | Syntax, Variables & Control Flow | Immutability by default, `mut`, primitives (`int`, `i64`, `f32`, `bool`, `string`), string interpolation `${...}`, `if`/`else`, and blank identifier `_`. |
| **[`02_oop_composition.vex`](./02_oop_composition.vex)** | OOP, Encapsulation & Embedding | 4-quadrant struct access levels (`default`, `mut:`, `pub:`, `pub mut:`), composition via struct embedding, promoted methods/fields, and value vs mutable pointer receivers. |
| **[`03_interfaces.vex`](./03_interfaces.vex)** | Structural Interfaces & VTables | Implicit interface satisfaction (no `implements` keyword), 16-byte fat-pointer boxing (`._object`, `._vtable`), dynamic polymorphic dispatch, and automatic destructor chaining. |
| **[`04_result_handling.vex`](./04_result_handling.vex)** | Zero-Cost Error Handling | Result types (`!T`), `return error('...')`, mandatory unwrap blocks `or { ... }`, default fallbacks, and compile-time unhandled error prevention. |
| **[`05_arenas_and_autofree.vex`](./05_arenas_and_autofree.vex)** | Deterministic Memory Management | Static LIFO scope-exit autofree without a GC, and high-performance 64-byte aligned bump arenas (`arena { ... }`) with O(1) bulk slab teardown. |
| **[`06_simd_tensors.vex`](./06_simd_tensors.vex)** | First-Class AI Tensors & SIMD | 2D tensor allocations (`tensor[f32]`), matrix multiplication with the `@` operator, hardware AVX2/NEON vectorization, and multi-layer forward passes. |
| **[`07_actors_and_supervisors.vex`](./07_actors_and_supervisors.vex)** | Actor Concurrency & Fault Tolerance | Multi-threaded actors (`actor`), non-blocking asynchronous message dispatch (`<-`), `Future[T]` `.await()`, Chase-Lev work-stealing, and `setjmp`/`longjmp` fault trapping with `OneForOne` supervisor restarts. |
| **[`08_production_pipeline.vex`](./08_production_pipeline.vex)** | End-to-End AI Production Pipeline | Integrated real-world pipeline combining OOP telemetry, dynamic interfaces, parallel actor inference, error propagation, and 0 memory leaks. |
| **[`09_file_io.vex`](./09_file_io.vex)** | Native File I/O & Buffers | File existence checking, atomic writes, appending records, reading into autofreed buffers, and file deletion with Result error handling. |
| **[`10_threads_actors_fault_tolerance.vex`](./10_threads_actors_fault_tolerance.vex)** | Threads, Actors, Synchronization & Fault Tolerance | Multicore Chase-Lev work-stealing pool, actor isolation, non-blocking dispatch (`<-`), `Future[T]` `.await()`, panic containment with `setjmp`/`longjmp`, O(1) arena wipes, and `OneForOne` supervisor restarts. |
| **[`11_control_flow_loops_operators.vex`](./11_control_flow_loops_operators.vex)** | Control Flow, Loops & Operators | Multi-way branching (`if`/`else if`/`else`), unified `for` loops (3-part C-style, while-style, infinite), `break`/`continue`, arithmetic, bitwise shifts (`<<`, `>>`), compound assignments (`+=`, `-=`, `*=`, `/=`, `%=`), logical (`&&`, `\|\|`, `!`), and `@` MatMul. |
| **[`12_deep_dive_tensors.vex`](./12_deep_dive_tensors.vex)** | Deep Dive AI Tensors & SIMD | Multi-layer MLP forward pass, 64-byte aligned tensors, hardware AVX2+FMA vectorization, zero-allocation bump arena inference, and zero-GIL multicore actor serving. |
| **[`13_hardware_acceleration_gpu.vex`](./13_hardware_acceleration_gpu.vex)** | Hardware Acceleration & GPU Queues | Discrete GPUs (CUDA/ROCm), Apple Metal Unified DMA, 64-byte PCIe cache alignment, zero-copy C-ABI handoff, and Erlang-style supervisor containment for GPU OOM faults. |
| **[`14_neural_network_cpu_gpu.vex`](./14_neural_network_cpu_gpu.vex)** | Neural Network CPU SIMD vs. GPU Streams | 3-layer MLP classifier forward pass (Dense + Bias + ReLU + Softmax Scaling), AVX2/NEON SIMD execution, GPU device stream actors, numerical parity verification, and CPU fallback on GPU OOM. |
| **[`15_pretrained_weights_loader.vex`](./15_pretrained_weights_loader.vex)** | Loading Pretrained PyTorch & SafeTensors Weights | Zero-copy binary tensor ingestion (`tensor_load_bin`), bit-exact forward pass matching PyTorch reference output, asynchronous actor serving, and bidirectional export (`tensor_save_bin`). |
| **[`16_compilation_and_build_modes.vex`](./16_compilation_and_build_modes.vex)** | Compilation, Build Modes & Production Deployment | Demonstrates build modes (dev `-O2`, prod `-O3`, sanitizer `-sanitize`), static linking, standalone binary deployment, and compiler telemetry. |
| **[`17_http_web_server_and_json.vex`](./17_http_web_server_and_json.vex)** | High-Performance HTTP 1.1 Web Engine & JSON Microservices | Fast JSON field parsing (`json_get_str`, `json_get_int`, `json_get_f32`), HTTP 1.1 response generation (`http_response`), actor-based request processing, and zero memory leaks. |
| **[`18_autograd_and_safetensors.vex`](./18_autograd_and_safetensors.vex)** | Pure Vex Autograd Engine & Native SafeTensors Loader | Native zero-copy loading of Hugging Face `.safetensors` files (`safetensors_load`), SIMD Autograd backpropagation (`tensor_matmul_grad_b`, `tensor_relu_backward`), and hardware in-place SGD optimization (`tensor_sgd_step`). |

---

## Verifying All Examples

You can run each example in sequence from your shell:

```powershell
.\vex.exe run docs/examples/01_basics.vex
.\vex.exe run docs/examples/02_oop_composition.vex
.\vex.exe run docs/examples/03_interfaces.vex
.\vex.exe run docs/examples/04_result_handling.vex
.\vex.exe run docs/examples/05_arenas_and_autofree.vex
.\vex.exe run docs/examples/06_simd_tensors.vex
.\vex.exe run docs/examples/07_actors_and_supervisors.vex
.\vex.exe run docs/examples/08_production_pipeline.vex
.\vex.exe run docs/examples/09_file_io.vex
.\vex.exe run docs/examples/10_threads_actors_fault_tolerance.vex
.\vex.exe run docs/examples/11_control_flow_loops_operators.vex
.\vex.exe run docs/examples/12_deep_dive_tensors.vex
.\vex.exe run docs/examples/13_hardware_acceleration_gpu.vex
.\vex.exe run docs/examples/14_neural_network_cpu_gpu.vex
.\vex.exe run docs/examples/15_pretrained_weights_loader.vex
.\vex.exe run docs/examples/16_compilation_and_build_modes.vex
.\vex.exe run docs/examples/17_http_web_server_and_json.vex
.\vex.exe run docs/examples/18_autograd_and_safetensors.vex
```
