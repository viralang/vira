# Chapter 18: First-Class AI Tensors, SIMD & Autograd

Most systems languages treat tensors and neural network math as third-party C++ bindings bolted onto the side. Vira builds **1D–4D strided tensors (`tensor[f32]`)** directly into the compiler and C99 runtime. Every tensor is backed by a 64-byte cache-line-aligned buffer, supports zero-copy reshaping inside `arena { ... }` blocks, accelerates `@` and `.linear(w)` across multi-threaded AVX2+FMA / SSE2 / ARM64 NEON kernels, includes reverse-mode autograd operators, and loads raw binary, Hugging Face `.safetensors`, and quantized `.gguf` weights with zero Python dependencies.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch18_ai_tensors_and_autograd.vir`](../code/ch18_ai_tensors_and_autograd.vir)

---

## 18.1. 1D–4D Strided Tensors (`tensor[f32]`), Views, and Reshaping

Under the hood (`runtime/rt/tensor/core.h`), `tensor[f32]` (`vira_tensor_f32`) stores a 64-byte aligned `data` pointer alongside 4-element `dims[4]` and `strides[4]` arrays, `n_dims` (`1..4`), and flattened `rows` / `cols` metadata for seamless 2D matrix compatibility.

You can allocate 2D, 3D, and 4D tensors directly with `tensor_fill`, `tensor_fill_3d`, and `tensor_fill_4d`, and inspect their shape and 64-bit element count at any time:

```v
q4 := tensor_fill_4d(2, 4, 8, 16, 0.25)
println('ndim=${q4.ndim}, shape=(${q4.dim(0)}x${q4.dim(1)}x${q4.dim(2)}x${q4.dim(3)}), numel=${q4.numel()}')
```

### Shape & Access API

| Property / Method | Return Type | Description |
| :--- | :--- | :--- |
| **`.ndim`** | `int` | Active rank (`1..4`). |
| **`.rows` / `.cols`** | `int` | Flattened outer product (`dims[0] * ... * dims[ndim-2]`) and trailing axis (`dims[ndim-1]`). |
| **`.numel()`** | `i64` | Total 64-bit element count across all dimensions (`vira_tensor_numel64`). |
| **`.dim(axis)`** / **`.stride(axis)`** | `int` | Size and element stride along `axis` (supports negative axis indexing). |
| **`.get(r, c)`** / **`.set(r, c, val)`** | `f32` / `void` | Bounds-checked scalar read and write (triggers host sync if dirty on GPU). |
| **`.copy_row(r, src)`** / **`.copy_rows(r0, src)`** | `void` | Copies 1D or 2D rows into a pre-allocated cache tensor. |
| **`.slice_rows(start, end)`** | `tensor[f32]` | Extracts rows `[start, end)` into a new tensor (arena-allocated inside `arena {}`). |

### Zero-Copy Views Inside `arena { ... }`

Reshaping a tensor with `.reshape_2d(d0, d1)`, `.reshape_3d(d0, d1, d2)`, or `.reshape_4d(d0, d1, d2, d3)` validates that the total element count matches. Inside an `arena { ... }` block, if the source tensor lives in the active arena (`vira_arena_contains`), `_vira_tensor_reshape_safe` returns a **zero-copy non-owning view** (`is_view = true`) pointing directly at the original buffer; outside an arena, it allocates a fresh owned copy so Autofree can free both tensors independently without double-free bugs:

```v
arena {
	seq3 := tensor_fill_3d(2, 8, 16, 0.5)
	flat_view := seq3.reshape_2d(16, 16) // O(1) zero-copy arena view
	trans_view := flat_view.transpose()  // Swaps trailing two axes (..., M, N) -> (..., N, M)
	_ := tensor_sum(trans_view)
}
```

---

## 18.2. Batched & Broadcast Matrix Multiplication (`@`) and Multi-Threaded AVX2 / NEON SIMD

Vira's `@` operator lowers directly to `vira_tensor_matmul_f32`, which handles three layouts uniformly:

1. **2D × 2D GEMM:** $(M, K) @ (K, N) \to (M, N)$
2. **Batched 3D / 4D Attention:** $(B, H, M, K) @ (B, H, K, N) \to (B, H, M, N)$ (with NumPy-style broadcasting along batch axes when a dimension is `1`)
3. **Broadcast N-D × 2D Projections:** $(B, S, K) @ (K, N) \to (B, S, N)$

Inside the CPU kernel, 64×64 L2-cache-blocked tiles execute 256-bit AVX2+FMA3 (`_mm256_fmadd_ps`), 128-bit SSE2, or ARM64 NEON (`vfmaq_n_f32`) instructions. When total multiply-accumulate ops reach $128^3$ (`2,097,152` ops) outside an active actor, `@` automatically shards row batches across Vira's work-stealing thread pool (`VIRA_THREADS` controls team size).

### Row-Major Linear Layers (`.linear(w)`) and Bias Broadcasting

Modern LLM weights (including GGUF files) are stored in row-major `[out_features, in_features]` (`[N, K]`) order. Calling `x.linear(w)` computes $x W^T$ directly without transposing `w`, parallelizing across output columns via `_vira_team_run` when $M \times N \times K \ge 65{,}536$, and automatically dispatching quantized integer dot kernels (`Q4_0 × Q8_0`, `Q4_K × Q8_K`, etc.) when `w` holds quantized GGUF blocks:

```v
k4 := q4.transpose()
attn_scores := q4 @ k4 // (2, 4, 8, 8)

seq := tensor_fill_3d(2, 8, 16, 0.5)
w_proj := tensor_fill(16, 8, 0.25)
bias := tensor_fill(1, 8, 1.0)
proj := (seq @ w_proj) + bias // Trailing-dimension bias broadcast

w_rm := tensor_fill(8, 16, 0.25)
seq_2d := seq.reshape_2d(16, 16)
lin_out := seq_2d.linear(w_rm) // Row-major x @ w_rm^T
```

---

## 18.3. Reverse-Mode Autograd and Training Neural Networks in Pure Vira

Vira's runtime (`runtime/rt/tensor/autograd.h`) ships built-in SIMD backward kernels and an in-place optimizer so you can train neural networks directly in Vira without Python or external frameworks:

| Operator | Mathematical Definition | Purpose |
| :--- | :--- | :--- |
| **`tensor_matmul_grad_a(grad_out, b)`** | $\frac{\partial L}{\partial A} = \text{grad\_out} @ B^T$ | Input activation gradient for $C = A @ B$. |
| **`tensor_matmul_grad_b(a, grad_out)`** | $\frac{\partial L}{\partial B} = A^T @ \text{grad\_out}$ | Weight matrix gradient for $C = A @ B$. |
| **`tensor_relu_backward(input, grad_out)`** | $\text{grad\_out} \odot \mathbb{1}(\text{input} > 0)$ | AVX2 masked ReLU backward pass. |
| **`tensor_sgd_step(mut param, grad, lr)`** | $\text{param} \leftarrow \text{param} - \text{lr} \cdot \text{grad}$ | In-place SIMD weight update (also evicts stale GPU/quant cache). |

Here is a complete 2-layer MLP training loop (`x -> w1 -> ReLU -> w2 -> MSE`) in pure Vira:

```v
mut step := 0
for step < 4 {
	z1 := x_train @ w1
	a1 := tensor_relu(z1)
	y_pred := a1 @ w2
	diff := y_pred - y_target
	loss := tensor_sum(tensor_elemwise_mul(diff, diff))
	println('    step=${step}: mse_sum=${loss}')

	grad_w2 := tensor_matmul_grad_b(a1, diff)
	grad_a1 := tensor_matmul_grad_a(diff, w2)
	grad_z1 := tensor_relu_backward(z1, grad_a1)
	grad_w1 := tensor_matmul_grad_b(x_train, grad_z1)

	tensor_sgd_step(mut w2, grad_w2, lr)
	tensor_sgd_step(mut w1, grad_w1, lr)
	step += 1
}
```

---

## 18.4. Loading Pretrained Weights (Binary, Hugging Face `.safetensors`, and Quantized GGUF)

Vira includes three native binary weight loaders returning `!tensor[f32]` Results:

1. **Raw Binary (`tensor_save_bin` / `tensor_load_bin`):** Writes and reads contiguous IEEE-754 `f32` bytes directly into 64-byte aligned buffers.
2. **Hugging Face SafeTensors (`safetensors_load(path, tensor_name)`):** Parses the 8-byte little-endian header length and JSON tensor directory, seeks to the exact byte offset, and loads the tensor shape and float data with zero Python dependencies.
3. **Quantized GGUF v2/v3 (`gguf_load_tensor(path, tensor_name)`):** Parses GGUF metadata and tensor headers (using a 4-slot thread-safe LRU directory cache) and reverses GGUF column-major shapes into row-major `[out, in]` (or `[n_expert, out, in]`) tensors. Supported formats include `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `MXFP4`, and `I32`. Quantized blocks stay compressed in Vira's quant registry and materialize a lazy FP32 host view (`VIRA_LAZY_HOST_FP32=1` by default for tensors $\ge 65{,}536$ elements) only when a non-quantized kernel touches them.

In the companion example, `ch18_ai_tensors_and_autograd.vir` writes tiny self-contained `.bin`, `.safetensors`, and `.gguf` files into `/tmp/`, registers `defer` cleanup with `os.remove`, and loads all three formats cleanly:

```v
bin_path := '/tmp/ch18_weights.bin'
st_path := '/tmp/ch18_model.safetensors'
gguf_path := '/tmp/ch18_tiny.gguf'
defer {
	_ := os.remove(bin_path) or { false }
	_ := os.remove(st_path) or { false }
	_ := os.remove(gguf_path) or { false }
}

_ := tensor_save_bin(w2, bin_path) or { panic('save_bin failed: ${err}') }
w2_bin := tensor_load_bin(4, 2, bin_path) or { panic('load_bin failed: ${err}') }
w_st := safetensors_load(st_path, 'fc1.weight') or { panic('safetensors_load failed: ${err}') }
w_gguf := gguf_load_tensor(gguf_path, 'blk.0.attn_norm.weight') or { panic('gguf_load_tensor failed: ${err}') }
```

---

## 18.5. Running the Complete Chapter 18 Example

Run [`docs/handbook/chapters/code/ch18_ai_tensors_and_autograd.vir`](../code/ch18_ai_tensors_and_autograd.vir):

```bash
./vira run docs/handbook/chapters/code/ch18_ai_tensors_and_autograd.vir
```

Output:
```text
[1] 4D Tensor: ndim=4, shape=(2x4x8x16), strides=(512,128,16,1), numel=1024
    Arena View: 3D->2D=(16x16), transpose=(16x16), sum=128.00
[2] Matmul & SIMD: attn_numel=512, proj_sum=384.00, linear_sum=256.00
[3] Autograd Training:
    step=0: mse_sum=11.14
    step=1: mse_sum=9.39
    step=2: mse_sum=7.55
    step=3: mse_sum=5.74
[4] Weights: bin_sum=2.79, safetensors(2x2)=10.00, gguf(ndim=1)=4.00
[vira_rt] simd=x86_64 SSSE3 + SSE2 (128-bit Integer/Float SIMD, Multi-Threaded)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=78
```

---

## Summary & Exercises

1. **Arena vs. Heap Reshapes:** Run `./vira check docs/handbook/chapters/code/ch18_ai_tensors_and_autograd.vir` and compare how `flat_view` inside `arena { ... }` is classified (`ARENA_MANAGED`) versus `seq_2d` outside the arena (`SCOPED_LOCAL`).
2. **Deeper Convergence:** Increase the autograd loop in `ch18_ai_tensors_and_autograd.vir` from `4` steps to `20` steps and print every 5th loss value to verify monotonic convergence toward `0.0`.
3. **3D SafeTensors Export:** Modify `write_demo_safetensors` to write a `shape` of `[1,2,2]` and verify that `safetensors_load` preserves `ndim == 3` and `dim(0) == 1, dim(1) == 2, dim(2) == 2`.
