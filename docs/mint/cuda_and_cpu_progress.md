# Progress Tracker: CUDA & CPU-SIMD Inference Parity (`docs/mint/cuda_and_cpu_plan.md`)

**Snapshot Date**: `2026-11-26`  
**Baseline Version**: `v0.7.9` (`main` post-`3bc29b0`)  
**Primary Plan Spec**: [`docs/mint/cuda_and_cpu_plan.md`](file:///mnt/flutter_drive/vira/docs/mint/cuda_and_cpu_plan.md)

---

## Overall Status Summary

| Phase | Scope | Status | Completion | Remaining Summary |
|---|---|---|---|---|
| **Phase 0** | Benchmark flags, chunked arena, tied `out_weight`, init ordering | Done | 100% | — |
| **Phase 1** | Batched prompt prefill + Int8 Tensor Core (`mmqt`/`mmqb`/`mmqg`) & WMMA GEMM (`M > 4`) | Done | 100% | — |
| **Phase 2** | Turing `Q8_1` activations + `sm_61+` `DP4A` MMVQ / `mmvqb` (`M <= 4`) | Done | 100% | — |
| **Phase 3** | Fused decode kernels, FlashAttention-2 (`mma.m16n8k8`), FP16 KV & Zero-SDK CUDA Graphs | Done | 100% | Dense `m == 1` decode captures & replays zero-toolkit CUDA Graphs |
| **Phase 4** | Host quant retention, lazy FP32, `AVX2`/`NEON` quant linear, 256-entry TLS quant cache | Done | 100% | GPU upload path consults raw quant registry & lazy FP32 |
| **Phase 5** | Mamba-2 SSM, Zero-Sync GPU MoE, GPU MLA, 64-bit `numel()`, Hardened GGUF & Parity Gates | In Progress | ~85% | ViraLama `70/70` reference parity passes; remaining: native `BF16`/`Q2_K`/`Q3_K`/`MXFP4` GPU kernels (`T-051`), `IQ*` (`T-030`/`T-031`), tighter `T-125` GPU parity & Windows RTX 3060 verification (`T-120`) |

---

## Post-`e386210` Completed Milestones (`fa23a43` .. `b0b7233`)

- **Int8 Tensor Core Prompt Kernels (`mmqt` / `mmqb` / fused `mmqg`)** (`fa23a43` .. `5ed90f7`):
  - `128x128` / `128x64` big-tile quantized prompt GEMM with `x4` activation layout, conflict-free `STS.64`, rasterized block order, FMA-folded scales, and fused gate+up SwiGLU (`mmqg`).
  - Measured on Tesla T4 (`Qwen3-14B Q4_K_M`): **`588 tok/s` (`pp512`)** and **`22 tok/s` (`tg128`)**.
- **FlashAttention-2 & FP16 KV Cache** (`24cda81`, `5e59e1e`):
  - Tiled prompt attention over FP16 KV cache (`gpu.kv_f16`) on `mma.m16n8k8` Tensor Cores.
- **Zero-SDK CUDA Graph Replay (`m == 1`)** (`ccd50fb`):
  - Direct CUDA Driver graph capture/replay (`dpos` kernels, zero CUDA Toolkit dependency) plus auto-`w1` for `>= 64` SMs.
  - Measured on NVIDIA A100 (`Qwen3-32B`): **`31 -> 45 tok/s`**.
- **GPU Precision & Kernel Coverage** (`976758c`, `df34a7a`, `1ad035f`, `9a0b46f`, `024d029`):
  - Keep `F32` GPU weights in `F32` instead of downcasting to `FP16` (`T-125`), and route `Q8_0` `K % 256 != 0` prefill cleanly (`df34a7a`).
  - Added CUDA kernels for **Mamba-2 SSM** (`conv1d`, `scan`, grouped RMSNorm), **`relu²`** and ungated `relu²` **MoE** (`9a0b46f`, reaching **`110 tok/s`** on A100 for Nemotron-Nano-9B-v2), **zero-sync GPU MoE residency** (`T-127`), and **GPU MLA attention** (`T-128`).
- **64-Bit Tensors, Thread-Safe GGUF & Lock-Free Quant Cache** (`7678fc7`, `20791f5`, `b0b7233`):
  - Widened `numel()` to `i64` (`T-011`); hardened `gguf_load_tensor` with extent bounds checks, single open file handle, 4-slot thread-safe LRU directory cache (`T-012`, `T-109`), and `fp16` subnormal decoding (`T-003`/`T-004`).
  - Added lock-free 256-entry TLS quant lookup (`_vira_tls_qcache`), AVX2 `vira_tensor_silu`, physical core detection in scheduler, and supervised `_vira_rt_fault` handling in `core.h`/`ops.h`/`llm.h`.

---

## Historical Snapshot: Detailed Verification Against Baseline `e386210` (`2026-11-09`)

### Phase 0: Benchmarking Discipline & Zero-Cost Cleanups
- [x] **Production `-prod` / `-quiet-rt` build support & CLI benchmark flags**
  - [`docs/examples/43_qwen2_5_gguf_chat_inference.vir`](file:///mnt/flutter_drive/vira/docs/examples/43_qwen2_5_gguf_chat_inference.vir#L747-L852) implements `run_benchmark` with `--bench-p <int>` and `--bench-n <int>` (`L888-889`, `L925-930`, `L1086-1092`).
- [x] **Per-layer arena churn eliminated**
  - Decode & prefill batch `6` layers per `arena { ... }` block in both [`43_qwen2_5_gguf_chat_inference.vir`](file:///mnt/flutter_drive/vira/docs/examples/43_qwen2_5_gguf_chat_inference.vir#L429-L469) (`L538-572`) and [`45_gui_qwen2_5_gguf_chat_studio.vir`](file:///mnt/flutter_drive/vira/docs/examples/45_gui_qwen2_5_gguf_chat_studio.vir#L392-L427) (`L479-515`).
  - `VIRA_GPU_ACT_RING_CAP` increased to `512` in [`runtime/vira_gpu.h`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L122).
- [x] **Tied LM-head weight deduplication**
  - [`43_qwen2_5_gguf_chat_inference.vir`](file:///mnt/flutter_drive/vira/docs/examples/43_qwen2_5_gguf_chat_inference.vir#L1012-L1016) sets `has_tied_out = true` and reuses `tok_emb` directly (`L478-482`, `L579-584`); [`45_gui_qwen2_5_gguf_chat_studio.vir`](file:///mnt/flutter_drive/vira/docs/examples/45_gui_qwen2_5_gguf_chat_studio.vir#L317-L322) returns a `1x1` sentinel and dispatches to `tok_emb` (`L436-440`).
- [ ] **Remaining bug fix — CPU default vs lazy `vira_gpu_init()` ordering**:
  - In [`43_qwen2_5_gguf_chat_inference.vir`](file:///mnt/flutter_drive/vira/docs/examples/43_qwen2_5_gguf_chat_inference.vir#L986-L987), `gpu.set_enabled(use_gpu)` runs *before* `gpu.device_name()` triggers [`vira_gpu_init`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L4288-L4305), which unconditionally sets `_vira_gpu.enabled = true`. Workaround today: pass `--gpu-backend cpu` to disable GPU on CUDA hosts. Fix needed: preserve pre-init `enabled` flag or move `gpu.set_enabled(use_gpu)` after init, and update interactive `/gpu` (`L1119-1125`) from legacy `gpu_set_enabled`/`gpu_allocated_bytes` + stale `gpu_dev` to `gpu.set_enabled` / `gpu.allocated_kb` / `gpu.device_name`.

---

### Phase 1: Batched Prompt Prefill + Quantized Tensor Cores (`Done`)
- [x] **Quantized Tensor Core WMMA (`sm_70+`) & Tiled Quantized GEMM (`M > 4`)**:
  - [`runtime/vira_cuda_ptx.h`](file:///mnt/flutter_drive/vira/runtime/vira_cuda_ptx.h#L10940-L13797) contains `vira_cu_wmma_gemm_q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q4_k`, `q5_k`, `q6_k`; loaded in [`_vira_gpu_init_cuda_locked`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L3861-L3885).
  - Non-WMMA fallback PTX (`vira_cu_gemm_*`) & OpenCL (`vira_cl_gemm_*`) kernels `23..30` are wired and dispatched in [`vira_gpu_linear`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L4885-L5150).
- [x] **Batched sequence operators & wiring**:
  - `tensor_embedding_batch` ([`runtime/rt/llm/llm.h`](file:///mnt/flutter_drive/vira/runtime/rt/llm/llm.h#L426-L480))
  - `gpu_rope_seq` ([`runtime/vira_gpu.h`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L5730), OpenCL/CUDA kernel `31`)
  - `.copy_rows(...)` ([`runtime/rt/tensor/ops.h`](file:///mnt/flutter_drive/vira/runtime/rt/tensor/ops.h#L224-L257), [`_vira_gpu_copy_rows`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L6900-L6979) async `DtoD`)
  - `gpu_gqa_prefill` ([`runtime/vira_gpu.h`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L6668), OpenCL/CUDA kernel `32`)
  - Compiler/stdlib support in [`internal/checker/checker.go`](file:///mnt/flutter_drive/vira/internal/checker/checker.go), [`internal/autofree/autofree.go`](file:///mnt/flutter_drive/vira/internal/autofree/autofree.go#L691-L692), [`internal/cgen/cgen.go`](file:///mnt/flutter_drive/vira/internal/cgen/cgen.go#L2772-L2859), and [`packages/gpu/src/gpu.vir`](file:///mnt/flutter_drive/vira/packages/gpu/src/gpu.vir#L111-L129).
  - `qwen2_forward_prefill` chunked at `512` tokens in both [`43_qwen2_5_gguf_chat_inference.vir`](file:///mnt/flutter_drive/vira/docs/examples/43_qwen2_5_gguf_chat_inference.vir#L492-L591) and [`45_gui_qwen2_5_gguf_chat_studio.vir`](file:///mnt/flutter_drive/vira/docs/examples/45_gui_qwen2_5_gguf_chat_studio.vir#L443-L550).

---

### Phase 2: Turing `DP4A` Quantized Decode Kernels (`Done`)
- [x] **Dynamic `Q8_1` activation cache & reuse**:
  - [`_vira_gpu_act_entry`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L213-L229) tracks `cu_q8_buf`, `q8_buf_cap`, `q8_bytes`, `q8_epoch`; quantized lazily by `_vira_gpu_ensure_act_q8_locked` and invalidated automatically when the source activation buffer changes epoch.
- [x] **`sm_61+` `DP4A` PTX (`_vira_cuda_dp4a_kernel_ptx`)**:
  - [`runtime/vira_cuda_ptx.h`](file:///mnt/flutter_drive/vira/runtime/vira_cuda_ptx.h#L13798-L15435) contains `vira_cu_quantize_q8_1` and `vira_cu_mmvq_q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q4_k`, `q5_k`, `q6_k`.
  - Loaded for `cc >= 6.1` in [`_vira_gpu_init_cuda_locked`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L3899-L3921) and dispatched for `M <= 4` in [`vira_gpu_linear`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L4937-L5053).

---

### Phase 3: Kernel Fusion & Zero-SDK CUDA Graph Replay (`~30% Complete`)
- [x] **CUDA PTX kernels present in `_vira_cuda_kernel_ptx`** ([`runtime/vira_cuda_ptx.h`](file:///mnt/flutter_drive/vira/runtime/vira_cuda_ptx.h#L7766-L10148)):
  - `vira_cu_add_rmsnorm_f32` (`L7766`) — grid `(rows, 1, 1)`, block `(256, 1, 1)`
  - `vira_cu_silu_mul_f32` (`L7981`) — grid `((total+255)/256, 1, 1)`, block `(256, 1, 1)`
  - `vira_cu_rope_kv_write_f32` (`L8021`) — grid `(n_heads + 2*n_kv_heads, M, 1)`, block `(64, 1, 1)`
  - `vira_cu_argmax_final_f32` (`L8297`) — grid `(1, 1, 1)`, block `(256, 1, 1)`
  - `vira_cu_emb_gather_f32` (`L8523`) & `vira_cu_emb_gather_indirect` (`L8987`) — grid `((cols+255)/256, 1, 1)`, block `(256, 1, 1)`, supports `wfmt` `0,1,2,3,6,7,8,12,13,14`
  - `vira_cu_rope_kv_write_indirect` (`L9456`) — grid `(n_heads + 2*n_kv_heads, 1, 1)`, block `(64, 1, 1)` (reads `kv_pos` at `cu_step_ctrl+4`)
  - `vira_cu_gqa_attention_indirect` (`L9733`) — grid `(n_heads, 1, 1)`, block `(256, 1, 1)` (reads `seq_len` at `cu_step_ctrl+8`)
- [x] **Forward declarations already in [`runtime/rt/tensor/core.h`](file:///mnt/flutter_drive/vira/runtime/rt/tensor/core.h#L482-L507)** (via [`runtime/vira_rt.h`](file:///mnt/flutter_drive/vira/runtime/vira_rt.h)):
  - `vira_gpu_add_rmsnorm`, `vira_gpu_silu_mul`, `vira_gpu_rope_kv_write`, `vira_gpu_emb_gather`, `vira_gpu_graph_try_replay`, `vira_gpu_graph_begin_capture`, `vira_gpu_graph_end_capture_argmax`, `vira_gpu_graph_captures`, `vira_gpu_graph_replays`
- [ ] **Remaining in [`runtime/vira_gpu.h`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h)**:
  1. Add OpenCL kernels `34..37` (`vira_cl_add_rmsnorm_f32`, `vira_cl_silu_mul_f32`, `vira_cl_rope_kv_write_f32`, `vira_cl_emb_gather_f32`) and split kernel counts (`VIRA_GPU_NUM_OPENCL_KERNELS = 37`, `VIRA_GPU_NUM_CUDA_KERNELS = 41`).
  2. Add dynamic CUDA Driver Graph API typedefs + loading in `_vira_gpu_init_cuda_locked` (`cuStreamBeginCapture_v2` / `cuStreamBeginCapture`, `cuStreamEndCapture`, `cuGraphInstantiateWithFlags` / `_v2` / legacy, `cuGraphLaunch`, `cuGraphExecDestroy`, `cuGraphDestroy`, `cuStreamIsCapturing`).
  3. Add device control block (`cu_step_ctrl`, `cu_best_tok_dev`), graph state (`cu_decode_graph`, `cu_decode_graph_exec`, bound signature pointers/caps), persistent graph-bound scratch slot tags (`graph_internal` / `graph_external` / `graph_bound` preserved across `_vira_gpu_act_entry_reset`), and strict invalidation on `set_enabled`, `set_backend`, `set_device`, `simulate_oom(true)`, `evict_ptr`, buffer frees, and `vira_gpu_shutdown`.
  4. Implement all 9 `vira_gpu_*` functions with CPU-SIMD / OpenCL / CUDA dispatch + OOM fallback accounting.
- [ ] **Remaining in Compiler & Stdlib**:
  - Wire the 9 `gpu_*` builtins across `internal/checker/checker.go`, `internal/autofree/autofree.go` (mark tensor-returning ops arena-managed), `internal/cgen/cgen.go`, `internal/lsp/server.go`, and `packages/gpu/src/gpu.vir`.
- [ ] **Remaining in Examples**:
  - Update `qwen2_forward_step` / decode loops in [`43_qwen2_5_gguf_chat_inference.vir`](file:///mnt/flutter_drive/vira/docs/examples/43_qwen2_5_gguf_chat_inference.vir) and [`45_gui_qwen2_5_gguf_chat_studio.vir`](file:///mnt/flutter_drive/vira/docs/examples/45_gui_qwen2_5_gguf_chat_studio.vir) to use `gpu_emb_gather`, `gpu_add_rmsnorm`, `gpu_rope_kv_write`, `gpu_silu_mul`, and CUDA Graph capture/replay for greedy decode (`temperature <= 0.05`).
  - *Autofree constraint*: keep `x` outside `arena { ... }`, do not reassign heap tensors in loops outside `arena`, and do not `return` an `arena`-allocated tensor from inside an `arena { ... }` block (`xf` and `logits` stay outside the layer-chunk `arena`).

---

### Phase 4: Quantized Host Weight Retention & Integer-SIMD CPU Engine (`~90% Complete`)
- [x] **Host quant registry & lazy FP32**:
  - `_vira_rt_quant_table` + bloom filter ([`runtime/rt/tensor/quant_registry.h`](file:///mnt/flutter_drive/vira/runtime/rt/tensor/quant_registry.h#L291-L625)) retains raw GGUF quant bytes, frees in `vira_tensor_f32_free` ([`runtime/rt/tensor/ops.h`](file:///mnt/flutter_drive/vira/runtime/rt/tensor/ops.h#L1104-L1113)), clones cleanly in `vira_tensor_f32_clone` ([`ops.h`](file:///mnt/flutter_drive/vira/runtime/rt/tensor/ops.h#L1121-L1140)), and cleans up at exit (`_vira_rt_quant_shutdown`).
  - `VIRA_LAZY_HOST_FP32=1` (default) skips upfront FP32 unpack for 2D quantized tensors with `total >= 65536` in [`vira_gguf_load_tensor`](file:///mnt/flutter_drive/vira/runtime/rt/weights/gguf.h#L333-L341) and materializes lazily via `_vira_tensor_sync_host_f32` -> `_vira_rt_ensure_fp32_host_ptr`.
- [x] **On-demand quantized embeddings**:
  - [`vira_tensor_embedding`](file:///mnt/flutter_drive/vira/runtime/rt/llm/llm.h#L375-L420) and [`vira_tensor_embedding_batch`](file:///mnt/flutter_drive/vira/runtime/rt/llm/llm.h#L426-L475) dequantize only selected rows from `raw_q`.
- [x] **Integer-SIMD quantized CPU linear (`_vira_rt_quant_cpu_linear`)**:
  - [`runtime/rt/tensor/quant_simd.h`](file:///mnt/flutter_drive/vira/runtime/rt/tensor/quant_simd.h#L1-L1330) implements `Q8_0` / `Q8_K` input quantization + multi-threaded single-row & 4-row blocked kernels with `AVX2`, `SSSE3`, and `ARM64 NEON` (`vdotq_s32` / `vmull_s8`) across `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`.
- [ ] **Remaining GPU/Lazy-FP32 integration hook in [`runtime/vira_gpu.h`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L3399-L3560)**:
  - [`_vira_gpu_get_or_upload_weight_locked`](file:///mnt/flutter_drive/vira/runtime/vira_gpu.h#L3399-L3560) reads `host_ptr` directly without consulting `_vira_rt_quant_try_get` or `_vira_rt_ensure_fp32_host_ptr`. If a lazy-FP32 weight (`fp32_state == 0`) was never pre-registered (e.g., model loaded while GPU was disabled and later enabled via `/gpu` or `gpu.set_backend`, cloned before GPU init, or evicted from VRAM), `_vira_gpu_get_or_upload_weight_locked` would upload uninitialized `host_ptr` floats instead of re-uploading `raw_q` or materializing host FP32. Fix: check `_vira_rt_quant_try_get` for raw quant upload first, and fall back to `_vira_rt_ensure_fp32_host_ptr(host_ptr)` before touching `host_ptr`.

---

### Phase 5: Verification Plan & Acceptance Gates (`50% Automated Tests / Pending Colab Gate`)
- [x] **Unit & parity tests in [`packages/gpu/tests/gpu_test.vir`](file:///mnt/flutter_drive/vira/packages/gpu/tests/gpu_test.vir)**:
  - `test_gpu_quant_wmma_prefill_parity` (`L674`)
  - `test_gpu_batched_prefill_exact_equivalence` (`L707`)
  - `test_gpu_dp4a_mmvq_parity` (`L888`)
- [ ] **Remaining tests in `packages/gpu/tests/gpu_test.vir`**:
  - `test_gpu_fused_ops_and_oom_fallback`
  - `test_cuda_graph_decode_lifecycle`
  - `test_cpu_quant_simd_linear_and_zero_leak`
- [ ] **Colab Tesla T4 `-prod` benchmark gates** (§5.2 of `cuda_and_cpu_plan.md`):
  - GPU (`--gpu-backend cuda --gpu`): `prefill >= 4,500 tok/s` (`128+` prompt tok), `gen >= 295 tok/s`
  - CPU (`--gpu-backend cpu --cpu`): `gen >= 38 tok/s`, load `< 500 ms`, host RAM `< 650 MiB`

---

## Verified Local Baseline (`2026-11-09` on `e386210`)
- `go test ./...` — **PASS** (`0` failures across all Go packages)
- `go run ./cmd/vira test packages/gpu/tests/gpu_test.vir` — **PASS** (`1 passed, 0 failed in 7.23s`, `active_leaks=0`; ran via CPU-SIMD fallback locally since dev host has no GPU)
- `go run ./cmd/vira check docs/examples/43_qwen2_5_gguf_chat_inference.vir` — **PASS** (`0 errors`)

---

## Remaining Open GPU & Quantization Backlog

1. **`T-125` — Tighter GPU Reference Parity**: Verify remaining GPU architecture parity tolerances across CUDA and OpenCL paths.
2. **`T-051` — Native GPU Kernels for Frontier Quants**: Add CUDA/OpenCL kernels for `BF16`, `Q2_K`, `Q3_K`, and `MXFP4` (currently dispatched via CPU-SIMD kernels).
3. **`T-030` / `T-031` — Importance Quantization (`IQ*`)**: Add dequantization and dot kernels for `IQ4_NL`, `IQ4_XS`, `IQ3`, `IQ2`, and `IQ1`.
4. **`T-120` — Windows RTX 3060 Verification**: Re-verify device scoring, CUDA kernel compilation, and runtime behavior on Windows hardware.

