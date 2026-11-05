# Tutorial 40: Cross-Platform GPU Compute Acceleration & Actor OOM Supervision (`v0.8`)

In **Milestone v0.8**, Vira introduces a **Zero-SDK-Dependency Cross-Platform GPU Compute Runtime** ([`runtime/vira_gpu.h`](../../runtime/vira_gpu.h) and [`packages/gpu`](../../packages/gpu/src/gpu.vir)) that offloads heavy tensor projections (`.gpu_linear(w)`), matrix multiplications (`gpu.matmul(a, b)`), and Transformer blocks to hardware GPUs while preserving Vira's strict **`active_leaks=0`** guarantee.

---

## 1. Architectural Highlights

1. **Zero Compile-Time SDK Dependencies (`LoadLibraryA` / `dlopen`):**
   - `vira run` and `vira build` never require a 4 GB CUDA Toolkit or Vulkan SDK at compile time.
   - At runtime, [`vira_gpu_init()`](../../runtime/vira_gpu.h) dynamically probes native OS GPU drivers (`OpenCL.dll` on Windows, `libOpenCL.so.1` on Linux, `OpenCL.framework` on macOS) and JIT-compiles hardware compute kernels (`vira_cl_linear_f32` and `vira_cl_matmul_f32`).
2. **64-Byte Aligned Unified DMA & Persistent Weight Caching:**
   - Every `tensor[f32]` in Vira is allocated on a 64-byte cache-line boundary (`((uintptr_t)raw + 63) & ~63`). On unified-memory GPUs (`CL_DEVICE_HOST_UNIFIED_MEMORY`, such as Intel Iris/Arc, AMD APUs, and Apple Silicon), Vira maps host tensors directly via `CL_MEM_USE_HOST_PTR` with **zero host-to-device copy**.
   - Model weight matrices (such as Qwen 2.5's `151,936 × 896` output classifier head) are indexed in a 512-slot **Persistent GPU Weight Cache** by host pointer address, uploading weights once and reusing them across all generated tokens.
   - When a host tensor is freed by Vira's static Autofree engine (`vira_tensor_f32_free`), its corresponding GPU device buffer is automatically evicted and released.
3. **Actor Fault-Tolerant GPU Supervision (VRAM OOM -> CPU SIMD Fallback):**
   - If a machine has no hardware GPU driver or if GPU VRAM runs out of memory (`CL_MEM_OBJECT_ALLOCATION_FAILURE`), Vira automatically evicts temporary device cache entries, increments `gpu.oom_fallbacks()`, and completes the tensor operation on its 16-worker Chase-Lev AVX2+FMA / ARM NEON CPU pool without crashing the actor.

---

## 2. Standard Library `import gpu` API Reference

| Function / Method | Signature | Description |
| :--- | :--- | :--- |
| **`gpu.info()`** | `fn info() DeviceInfo` | Returns complete hardware telemetry (`available`, `enabled`, `device_name`, `backend_name`, `vram_bytes`, `allocated_bytes`, `oom_fallbacks`) |
| **`gpu.is_available()`** | `fn is_available() bool` | Returns `true` if a hardware GPU compute driver is active |
| **`gpu.device_name()`** | `fn device_name() string` | Hardware GPU model name (e.g., `"Intel(R) HD Graphics 630"`, `"NVIDIA GeForce RTX 4090"`) |
| **`gpu.backend_name()`** | `fn backend_name() string` | Active compute backend (`"OpenCL"` or `"CPU-SIMD"`) |
| **`gpu.vram_mb()`** | `fn vram_mb() i64` | Total device VRAM in megabytes |
| **`gpu.allocated_kb()`** | `fn allocated_kb() i64` | Currently cached persistent weight buffer size in kilobytes |
| **`gpu.set_enabled(b)`** | `fn set_enabled(enabled bool)` | Dynamically enables or disables GPU offload at runtime |
| **`gpu.sync()`** | `fn sync()` | Waits for all in-flight GPU command queues to complete |
| **`x.gpu_linear(w)`** / **`gpu.linear(x, w)`** | `fn linear(x, w tensor[f32]) tensor[f32]` | Computes $y = x W^T$ on the GPU with persistent weight caching and OOM fallback |
| **`a.gpu_matmul(b)`** / **`gpu.matmul(a, b)`** | `fn matmul(a, b tensor[f32]) tensor[f32]` | Computes $C = A B$ on the GPU with persistent buffer caching and OOM fallback |
| **`gpu.rmsnorm(x, w, eps)`** | `fn rmsnorm(x, w tensor[f32], eps f32) tensor[f32]` | Root Mean Square Layer Normalization |
| **`gpu.rope(q, pos, base)`** | `fn rope(q tensor[f32], pos int, base f32) tensor[f32]` | Rotary Position Embeddings (Interleaved & NeoX Split-Half) |
| **`gpu.softmax(x)`** | `fn softmax(logits tensor[f32]) tensor[f32]` | Numerically stable row-wise Softmax |
| **`gpu.silu(x)`** | `fn silu(x tensor[f32]) tensor[f32]` | Sigmoid Linear Unit (`SiLU`) activation |
| **`gpu.simulate_oom(b)`** | `fn simulate_oom(enable bool)` | Injects simulated VRAM Out-of-Memory faults to test actor OOM fallback |

---

## 3. Persistent-Weight Linear Projections (`x.gpu_linear(w)`)

In LLM inference, weight matrices `w` have shape `(out_features, in_features)` and remain immutable across the entire session. Calling `x.gpu_linear(w)` caches `w` in GPU memory on the first call and reuses the device handle on all subsequent calls:

```vira
import gpu

x := tensor_fill(128, 512, 0.125)
w_proj := tensor_fill(1024, 512, 0.03125)

// Cold call: uploads `w_proj` (2048 KB) to the GPU weight cache
y_cold := x.gpu_linear(w_proj)

// Warm call: 0 bytes uploaded — reuses resident GPU buffer directly!
y_warm := x.gpu_linear(w_proj)
println('Cached GPU weights: ${gpu.allocated_kb()} KB')
```

---

## 4. Actor-Supervised GPU Fault Tolerance

When running GPU inference inside concurrent Vira actors, VRAM spikes from parallel requests can exhaust device memory. Vira's GPU runtime supervises every buffer allocation and kernel launch so actors automatically fall back to CPU SIMD without dropping requests:

```vira
import gpu

actor GpuComputeWorker {
pub:
    id int
mut:
    completed_batches int = 0
}

fn (mut w GpuComputeWorker) run_projection(scale f32, trigger_oom bool) f32 {
    w.completed_batches += 1
    if trigger_oom {
        gpu.simulate_oom(true)
    }
    mut result_sum := f32(0.0)
    arena {
        x := tensor_fill(64, 128, scale)
        weight := tensor_fill(128, 128, 0.0625)
        out := x.gpu_linear(weight)
        result_sum = tensor_sum(out)
    }
    if trigger_oom {
        gpu.simulate_oom(false)
    }
    return result_sum
}
```

---

## 5. Running Example 44 & Enabling `--gpu` in Qwen 2.5 Chat

Run the standalone GPU compute benchmark and OOM supervision suite:

```powershell
.\vira.exe run docs/examples/44_gpu_compute_acceleration.vir
```

### Verified Output

```text
========================================================================
   Vira v0.8 Cross-Platform GPU Compute Acceleration & OOM Supervision  
========================================================================
Compute Backend : OpenCL
Compute Device  : Intel(R) HD Graphics 630
Hardware GPU    : true (enabled=true, VRAM=13021 MB)
------------------------------------------------------------------------
=== 1. Persistent-Weight Linear Projection (128x512 @ 1024x512^T) ===
CPU SIMD (.linear)       : 2598 us | sum=262144.00
GPU Cold (.gpu_linear)   : 11771 us (initial weight upload)
GPU Warm (.gpu_linear)   : 9120 us (persistent weight cache hit, cached=2048 KB)
Numerical Equivalence    : |gpu_sum - cpu_sum| = 0.00
=== 2. Fused GPU Transformer Block inside O(1) Bump Arena ===
Transformer Block Output : sum=1073575936.00 (4139 us, 0 arena leaks)
=== 3. Actor Fault-Tolerant GPU Supervision (VRAM OOM -> CPU Fallback) ===
Worker #1 (Normal GPU)   : sum=65536.00
Worker #2 (OOM Fallback) : sum=65536.00 (oom_fallbacks=1)
------------------------------------------------------------------------
Vira v0.8 GPU Compute Acceleration Verified Successfully (active_leaks=0).
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle, Multi-Threaded)
[vira_rt] pool_workers=8 | actor_tasks_run=9 | work_steals=3
[vira_rt] active_leaks=0 | total_freed=41
```

You can also toggle GPU offload in the Qwen 2.5 GGUF Chat Engine ([`docs/examples/43_qwen2_5_gguf_chat_inference.vir`](../examples/43_qwen2_5_gguf_chat_inference.vir)) via the `--gpu` CLI flag or the `/gpu` interactive REPL command:

```powershell
.\vira.exe run docs/examples/43_qwen2_5_gguf_chat_inference.vir -autofree=false -- --gpu -i
```

