# Tutorial 6: First-Class AI Tensors & SIMD Acceleration

> **Difficulty:** Intermediate  
> **Matching Code Example:** [`docs/examples/06_simd_tensors.vex`](../examples/06_simd_tensors.vex)

---

## 1. Overview

Most general-purpose programming languages treat matrix math and tensor operations as third-party library extensions (e.g. NumPy, PyTorch).

Vex integrates **2D numerical tensors directly into the language syntax and type system**:
- Native `tensor[f32]` and `tensor[bf16]` types
- First-class matrix multiplication operator (`@`)
- Automatic hardware SIMD vectorization (AVX2, FMA, ARM NEON)
- Zero-leak lifecycle tracking via Autofree

---

## 2. Tensor Creation & Initialization

Create 2D tensors using the built-in `tensor_fill` primitive:

```vex
// Allocate a 64x64 matrix filled with 0.50
w := tensor_fill(64, 64, 0.50)

// Allocate a 64x64 input matrix filled with 2.00
x := tensor_fill(64, 64, 2.00)
```

Tensors store data in a contiguous 64-byte aligned memory slab, enabling vector instructions to load memory directly without unaligned-load penalties.

---

## 3. Matrix Multiplication with the `@` Operator

In Vex, matrix multiplication is an infix binary operator:

```vex
y := w @ x
```

The `@` operator computes the linear algebraic dot product:
$$\mathbf{Y}_{i,j} = \sum_{k} \mathbf{W}_{i,k} \cdot \mathbf{X}_{k,j}$$

If the inner dimensions mismatch, the compiler raises a compile-time semantic error.

---

## 4. Hardware SIMD Backends

The Vex single-header runtime (`vex_rt.h`) automatically vectorizes `@` based on target architecture:

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

```vex
sum_w := tensor_sum(w) // Sums all 4,096 elements
sum_y := tensor_sum(y) // Sums the dot-product output
```

---

## 6. Multi-Layer Forward Pass Example

You can chain tensor operations to implement neural network layers:

```vex
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
.\vex.exe run docs/examples/06_simd_tensors.vex
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
[vex_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
[vex_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vex_rt] active_leaks=0 | total_freed=9
```

---

## Next Steps

Proceed to [**Tutorial 7: Actor Concurrency & Fault Supervision**](./07_actor_concurrency_and_supervisors.md) to learn how to scale tensor computations across multicore work-stealing actor pools.
