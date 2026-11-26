# Chapter 19: Zero-SDK GPU Acceleration & Local LLM Inference

Deploying GPU-accelerated AI applications traditionally forces users to install gigabytes of CUDA toolkits, `nvcc`, cuBLAS, or PyTorch wheels. Vira takes a radically simpler **zero-SDK** approach: the runtime embeds precompiled PTX and OpenCL kernel sources directly in C99 headers (`runtime/vira_gpu.h`), discovers system GPU drivers dynamically at runtime via `dlopen` / `LoadLibrary`, and falls back seamlessly to multi-threaded CPU SIMD whenever no GPU is present—or whenever VRAM runs out.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch19_gpu_and_local_llm.vir`](../code/ch19_gpu_and_local_llm.vir)

---

## 19.1. Zero-SDK GPU Architecture (`import gpu`, Dynamic CUDA Driver PTX JIT & OpenCL Fallback)

Importing `gpu` (`packages/gpu/src/gpu.vir`) exposes Vira's hardware discovery, telemetry, backend selection, and GPU-accelerated tensor operations:

```v
import gpu

dev := gpu.info()
println('backend=${dev.backend_name}, device=${dev.device_name}, available=${dev.available}, vram_mb=${gpu.vram_mb()}, tensor_cores=${dev.has_tensor_cores}')
```

### Three-Tier Runtime Discovery Pipeline

At initialization, `runtime/vira_gpu.h` probes backends in priority order without requiring any compile-time GPU SDK headers or libraries:

1. **CUDA Driver API (`libcuda.so` / `nvcuda.dll`):** Loads `cuInit` and `cuModuleLoadData` dynamically via `dlopen` / `LoadLibrary`, JIT-compiling Vira's embedded PTX modules (`vira_cuda_ptx.h`, `vira_cuda_mmvq2_ptx.h`, `vira_cuda_mmq_ptx.h`, `vira_cuda_fused_ptx.h`, `vira_cuda_attn_ptx.h`) onto the active NVIDIA GPU.
2. **OpenCL (`libOpenCL.so` / `OpenCL.dll`):** If no CUDA driver is present (such as on AMD, Intel, or Apple OpenCL devices), compiles `_vira_opencl_kernel_src` at runtime via `clCreateProgramWithSource` and `clBuildProgram`.
3. **CPU-SIMD Fallback (`CPU-SIMD`):** On headless CI runners, containers, or CPU-only servers, every `gpu.*` call and `.gpu_linear(w)` method executes on Vira's multi-threaded AVX2/SSE2/NEON kernels with identical numerical results (up to float rounding).

You can override device selection programmatically (`gpu.set_backend('cuda' | 'opencl' | 'cpu' | 'auto')`, `gpu.set_device('0')`, `gpu.set_enabled(false)`) or via environment variables (`VIRA_GPU_BACKEND`, `VIRA_GPU_DEVICE`, `VIRA_GPU_VRAM_BUDGET_MB`, `VIRA_GPU_STATS`).

---

## 19.2. VRAM Weight Caching, Tensor Core (`WMMA`) Dispatch, and Supervised OOM Fallbacks

Uploading weight matrices across PCIe on every token step would destroy inference throughput. Instead, Vira maintains a **persistent VRAM weight cache** keyed by host tensor pointer:

- The first call to `x.gpu_linear(w)` uploads `w` (or its raw quantized blocks for `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`) into VRAM and tracks usage via `gpu.allocated_kb()`.
- Subsequent calls reuse the resident device buffer with zero host-to-device copy.
- When `tensor_f32_free` runs on a weight tensor, `_vira_gpu_evict_ptr` automatically frees its VRAM allocation.
- Calling `gpu.sync()` flushes queued GPU work and synchronizes any GPU-modified host tensors (while `gpu.wait()` synchronizes timing without forcing a device-to-host copy).

### Advanced CUDA Kernels in `runtime/vira_gpu.h`

On supported NVIDIA GPUs, Vira's embedded PTX modules automatically dispatch specialized kernels behind standard `gpu.*` calls:
- **FP16 Tensor Cores (`WMMA`):** `cu_k_wmma_linear_h`, `cu_k_wmma_matmul`, and quantized `cu_k_wmma_gemm_*` kernels accelerate large prompt-prefill GEMMs on Volta+ Tensor Cores.
- **FlashAttention-2 / FlashDecoding:** `gpu.gqa_prefill` and `gpu.gqa_attention` run online-softmax causal grouped-query attention directly against resident KV caches (and `gpu.kv_f16(k_cache)` keeps KV caches in FP16 on CUDA to halve VRAM usage).
- **CUDA Graph Capture & Replay:** Autoregressive decode loops can capture static kernel launch graphs (`cuStreamEndCapture` / `cuGraphLaunch`) to eliminate per-token CPU launch overhead.
- **Fused Modern Architectures:** `vira_cuda_fused_ptx.h` also ships fused kernels for **Mamba-2** state-space models (`vira_cu_mamba2_conv_f32`, `vira_cu_mamba2_ssm_f32`, `vira_cu_mamba2_norm_f32`), **Mixture of Experts (MoE)** routing (`vira_cu_moe_act_f32`, `vira_cu_moe_sum_f32` with `resident` expert placement), and **DeepSeek MLA** latent attention (`vira_cu_mla_*`).

### Actor-Supervised OOM Fallback (`gpu.simulate_oom`)

When an actor runs out of VRAM (or when testing fault recovery with `gpu.simulate_oom(true)`), Vira increments `gpu.oom_fallbacks()` and transparently routes the operation to CPU SIMD instead of crashing the worker:

```v
fn (mut w GpuWorker) run_linear_step(scale f32, force_oom bool) f32 {
	w.batches += 1
	if force_oom {
		gpu.simulate_oom(true)
	}
	mut total := f32(0.0)
	arena {
		x := tensor_fill(64, 128, scale)
		wt := tensor_fill(128, 128, 0.0625)
		y := x.gpu_linear(wt)
		total = tensor_sum(y)
	}
	if force_oom {
		gpu.simulate_oom(false)
	}
	return total
}
```

---

## 19.3. Building a Quantized GGUF Local LLM Chat Engine (RoPE, RMSNorm, SwiGLU, GQA, KV Cache)

Combining `gguf_load_tensor` (Chapter 18), `arena { ... }`, and `gpu.*` primitives gives you everything needed to run real quantized LLMs—such as LLaMA 3 or Qwen 2.5 (see [`docs/examples/41_cli_llm_chat_engine.vir`](../../../examples/41_cli_llm_chat_engine.vir) and [`docs/examples/43_qwen2_5_gguf_chat_inference.vir`](../../../examples/43_qwen2_5_gguf_chat_inference.vir))—in pure Vira.

Each transformer step inside an `arena { ... }` block follows five clean stages:

1. **Embedding & Pre-Attention RMSNorm:**
   `h := tensor_embedding(tok_emb, tok_id)` followed by `hn := gpu.rmsnorm(h, attn_norm, 0.00001)`.
2. **QKV Linear Projections & Per-Head RoPE:**
   Project `q`, `k`, `v` with `gpu.linear`, reshape `q` into `(n_heads, head_dim)` and `k` into `(n_kv_heads, head_dim)`, and apply NeoX RoPE via `gpu.rope(..., pos, 10000.0)` (or `gpu.rope_norm` for adjacent-pair LLaMA RoPE).
3. **Stateful KV Cache Update & Grouped-Query Attention (GQA):**
   Write `k` and `v` into persistent cache rows with `k_cache.copy_row(pos, k_flat)` and `v_cache.copy_row(pos, v)`, then compute causal GQA over `[0..pos]` via `gpu.gqa_attention(q_heads, k_cache, v_cache, i64(pos), scale)`.
4. **SwiGLU Feed-Forward Network:**
   Normalize the residual state and compute fused SwiGLU gating with `gpu.silu_mul(gpu.linear(h2, w_gate), gpu.linear(h2, w_up))` (or single-launch `gpu.linear_swiglu` for quantized prompt prefill) followed by `w_down`.
5. **Final RMSNorm, LM Head & Sampling:**
   Project logits with `gpu.linear(xf, lm_head)` and decode the next token ID via `sample_token(logits, temperature)`.

Because every intermediate activation (`hn`, `q_lin`, `q_heads`, `attn_out`, `swiglu`, `logits`) is scoped to the `arena { ... }` iteration, all per-token temporary buffers are wiped in $O(1)$ at the end of every step with `active_leaks=0`.

---

## 19.4. Running the Complete Chapter 19 Example

Run [`docs/handbook/chapters/code/ch19_gpu_and_local_llm.vir`](../code/ch19_gpu_and_local_llm.vir) (executes cleanly both on GPU hosts and on CPU-only hosts via automatic CPU-SIMD fallback):

```bash
./vira run docs/handbook/chapters/code/ch19_gpu_and_local_llm.vir
```

Output (on an `x86_64` CPU-SIMD host without an attached GPU):
```text
[1] GPU Info: backend=CPU-SIMD, device=Multi-Threaded CPU SIMD (x86_64 SSSE3 + SSE2 (128-bit Integer/Float SIMD, Multi-Threaded)), available=false, enabled=true, vram_mb=0, tensor_cores=false
[2] Linear Cache: cpu_sum=32768.00, gpu_sum=32768.00, cached_kb=0, diff=0.00
    Actor OOM Supervision: normal=65536.00, oom_fallback=65536.00, oom_fallbacks=2
[3] Transformer GQA Block: steps=2, next_tok=0, logits_sum=128.00
[vira_rt] simd=x86_64 SSSE3 + SSE2 (128-bit Integer/Float SIMD, Multi-Threaded)
[vira_rt] pool_workers=4 | actor_tasks_run=2 | work_steals=2
[vira_rt] active_leaks=0 | total_freed=62
```

---

## Summary & Exercises

1. **GPU Telemetry Profiling:** Call `gpu.stats_reset()` before the `[2]` linear projection section and `gpu.stats_print('ch19', 1)` after `gpu.sync()` to inspect host-to-device byte counts, kernel launches, and CPU fallback counters.
2. **Multi-Turn Autoregressive Loop:** Extend the Transformer loop in `ch19_gpu_and_local_llm.vir` from `2` steps to `6` steps, feeding each sampled `next_tok` back as the next step's `tok_id`, and verify `active_leaks=0`.
3. **Batched Prefill (`gpu.gqa_prefill`):** Replace the single-token prefill loop with `tensor_embedding_batch`, `gpu.rope_seq`, `k_cache.copy_rows(0, k_flat)`, and `gpu.gqa_prefill` for a 4-token prompt chunk.
