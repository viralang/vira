# Tutorial 6: First-Class AI Tensors & SIMD Acceleration

> **Difficulty:** Intermediate  
> **Matching Code Example:** [`docs/examples/06_simd_tensors.vir`](../examples/06_simd_tensors.vir)

---

## 1. Overview

Most general-purpose programming languages treat matrix math and tensor operations as third-party library extensions (e.g. NumPy, PyTorch).

Vira integrates **1D–4D strided numerical tensors directly into the language syntax and type system**:
- Native `tensor[f32]` and `tensor[bf16]` types (`dims[4]`, `strides[4]`, `n_dims`, plus 2D `rows`/`cols` compatibility)
- First-class batched & broadcast matrix multiplication operator (`@`)
- Zero-copy `.reshape_2d/3d/4d()` views inside `arena { ... }` and trailing-axis `.transpose()`
- Multi-threaded CPU SIMD vectorization (AVX2+FMA, SSE2, ARM NEON) across the work-stealing thread pool
- Zero-leak lifecycle tracking via Autofree

---

## 2. Tensor Creation & Initialization (2D, 3D & 4D)

Create 2D, 3D, or 4D strided tensors using the built-in `tensor_fill*` primitives:

```v
// Allocate a 64x64 2D matrix filled with 0.50
w := tensor_fill(64, 64, 0.50)

// Allocate a 64x64 2D input matrix filled with 2.00
x := tensor_fill(64, 64, 2.00)

// Allocate a 3D sequence batch (Batch=2, Seq=16, Hidden=64)
seq := tensor_fill_3d(2, 16, 64, 0.25)

// Allocate a 4D multi-head attention tensor (Batch=2, Heads=4, Seq=16, HeadDim=64)
q := tensor_fill_4d(2, 4, 16, 64, 0.125)
println('Rank=${q.ndim}, Numel=${q.numel()}, HeadDim=${q.dim(3)}')
```

Tensors store data in a contiguous 64-byte aligned memory slab, enabling vector instructions to load memory directly without unaligned-load penalties.

---

## 3. Matrix Multiplication with the `@` Operator

In Vira, matrix multiplication is an infix binary operator:

```v
y := w @ x
```

The `@` operator computes the linear algebraic dot product:
$$\mathbf{Y}_{i,j} = \sum_{k} \mathbf{W}_{i,k} \cdot \mathbf{X}_{k,j}$$

If the inner dimensions mismatch, the compiler raises a compile-time semantic error.

---

## 4. Hardware SIMD Backends

The Vira single-header runtime (`vira_rt.h`) automatically vectorizes `@` based on target architecture:

### 1. x86_64 AVX2 + FMA (256-bit Vector Registers)
- Employs 8-wide single-precision float vectors (`__m256`).
- Executes 8 multiply-accumulate operations per CPU clock cycle using fused multiply-add (`_mm256_fmadd_ps`).

### 2. ARM64 NEON (128-bit Vector Registers)
- Employs 4-wide single-precision float vectors (`float32x4_t`).
- Uses `vmlaq_f32` fused multiply-accumulate instructions.

### 3. Portable Fallback
- Cache-blocked 8x8 tiling algorithm designed to keep L1 cache utilization near 100% on any standard C99 CPU.

---

## 5. Tensor Reductions

Calculate aggregate statistics using built-in reduction primitives:

```v
sum_w := tensor_sum(w) // Sums all 4,096 elements
sum_y := tensor_sum(y) // Sums the dot-product output
```

---

## 6. Multi-Layer Forward Pass Example

You can chain tensor operations to implement neural network layers:

```v
// Layer 1
w1 := tensor_fill(64, 64, 0.50)
x  := tensor_fill(64, 64, 2.00)
y1 := w1 @ x

// Layer 2
w2 := tensor_fill(64, 64, 0.10)
y2 := y1 @ w2

println('Final Layer Output Sum: ${tensor_sum(y2)}')
```

All intermediate matrices (`w1`, `x`, `y1`, `w2`, `y2`) are automatically freed by the autofree engine at function exit.

---

## 7. Running the Code

```powershell
.\vira.exe run docs/examples/06_simd_tensors.vir
```

Expected output:
```text
=== 1. TENSOR CONSTRUCTION & INITIALIZATION ===
Allocated weight matrix W (64x64) and input matrix X (64x64)
=== 2. HARDWARE-ACCELERATED MATRIX MULTIPLICATION (@) ===
Computed Y = W @ X via SIMD dot-product vectorization
=== 3. TENSOR REDUCTIONS & TELEMETRY ===
TensorSum(W): 2048.00
TensorSum(X): 8192.00
TensorSum(Y = W @ X): 262144.00
=== 4. MULTI-LAYER TENSOR FORWARD PASS ===
Layer 2 Output Sum: 1677714.75
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=9
```

---

## Next Steps

Proceed to [**Tutorial 7: Actor Concurrency & Fault Supervision**](./07_actor_concurrency_and_supervisors.md) to learn how to scale tensor computations across multicore work-stealing actor pools.
