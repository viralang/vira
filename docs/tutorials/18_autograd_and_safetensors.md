# Tutorial 18: Pure Vira Autograd Engine & Native SafeTensors Loader

Modern machine learning workloads depend heavily on Python-based frameworks (such as PyTorch and Hugging Face `safetensors`), creating complex deployment environments, large container footprints (often exceeding several gigabytes), and runtime memory overheads.

Vira changes this paradigm by providing:
1. **A native zero-copy SafeTensors parser** that reads Hugging Face `.safetensors` files directly into 64-byte aligned SIMD memory without Python.
2. **First-class Autograd & backward gradient operators** for backpropagation (`tensor_matmul_grad_a`, `tensor_matmul_grad_b`, `tensor_relu_backward`).
3. **Hardware-vectorized in-place optimizers** (`tensor_sgd_step`) accelerated by AVX2+FMA on x86-64 and NEON on ARM64.

---

## 1. The SafeTensors File Format Architecture

Hugging Face's SafeTensors is an open, secure binary serialization format designed for fast model loading:

```
+-------------------------------------------------------------------+
| 8 Bytes (uint64_t Little-Endian): Header JSON length N            |
+-------------------------------------------------------------------+
| N Bytes: UTF-8 JSON Header String                                 |
|   {                                                               |
|     "fc1.weight": { "dtype": "F32", "shape": [4, 4],              |
|                     "data_offsets": [0, 64] },                    |
|     "__metadata__": { "format": "pt" }                            |
|   }                                                               |
+-------------------------------------------------------------------+
| Raw Tensor Data Buffer (Contiguous 32-bit floats)                 |
|   fc1.weight floats starting at offset: 8 + N + 0                 |
+-------------------------------------------------------------------+
```

In Vex, `safetensors_load(path, tensor_name)`:
1. Reads the 8-byte header size $N$.
2. Parses the JSON metadata to extract `"shape"` and `"data_offsets"`.
3. Seeks directly to `8 + N + offset_begin`.
4. Executes a single zero-copy DMA read directly into a 64-byte aligned `vira_tensor_f32` buffer ready for immediate AVX2 vector execution.

---

## 2. Mathematical Foundations of Vira Autograd

Given a linear transformation with input $X \in \mathbb{R}^{M \times K}$ and weight matrix $W \in \mathbb{R}^{K \times N}$:

$$Y = X @ W$$

During backward propagation, given upstream error gradient $\Delta = \frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{M \times N}$, the matrix gradients are computed via the matrix chain rule:

$$\frac{\partial \mathcal{L}}{\partial X} = \Delta @ W^T \quad (\text{shape } M \times K)$$

$$\frac{\partial \mathcal{L}}{\partial W} = X^T @ \Delta \quad (\text{shape } K \times N)$$

Vira provides optimized hardware primitives for both:
- `grad_x := tensor_matmul_grad_a(diff, w)`
- `grad_w := tensor_matmul_grad_b(x, diff)`

---

## 3. Activation Functions & In-Place Optimizers

### ReLU Backward Operator
For $A = \text{ReLU}(Z)$, the upstream gradient flows only through positive elements:

$$\frac{\partial \mathcal{L}}{\partial Z_i} = \begin{cases} \Delta_i & \text{if } Z_i > 0 \\ 0 & \text{otherwise} \end{cases}$$

In Vex:
```vex
relu_grad := tensor_relu_backward(input_z, grad_out)
```
On AVX2 hardware, this executes using SIMD vector comparison (`_mm256_cmp_ps(..., _CMP_GT_OQ)`) and bitwise masking (`_mm256_and_ps`) processing 8 floats per CPU cycle.

### SIMD Stochastic Gradient Descent (SGD)
Parameters are updated in-place using fused vector subtraction:

$$W \leftarrow W - \eta \cdot \nabla_W \mathcal{L}$$

```vex
tensor_sgd_step(mut w, grad_w, 0.05)
```

---

## 4. Complete Runnable Implementation

Here is the source code from [docs/examples/18_autograd_and_safetensors.vir](file:///c:/Users/hp/vex/docs/examples/18_autograd_and_safetensors.vir):

```vex
module main

fn main() {
	println('=== 1. NATIVE SAFETENSORS ZERO-COPY LOADER ===')
	mut w := safetensors_load('models/model.safetensors', 'fc1.weight') or {
		eprintln('Failed to load SafeTensors file: ${err}')
		return
	}

	w_sum := tensor_sum(w)
	println('Loaded SafeTensors weight \'fc1.weight\' (shape 4x4, sum=${w_sum})')

	println('=== 2. FORWARD EVALUATION ===')
	x := tensor_fill(4, 4, 1.0)
	y_init := x @ w
	init_sum := tensor_sum(y_init)
	println('Initial forward activation sum: ${init_sum}')

	println('=== 3. AUTOGRAD SIMD GRADIENT DESCENT ===')
	// Target desired activation sum is 306.00
	target_w := tensor_scale(w, 0.75)
	y_target := x @ target_w

	mut step := 0
	for step < 3 {
		y_pred := x @ w
		diff := y_pred - y_target
		current_loss := tensor_sum(diff)
		println('Step ${step}: loss=${current_loss}')

		// Compute backprop gradient: grad_W = X^T @ diff
		grad_w := tensor_matmul_grad_b(x, diff)

		// Apply SIMD SGD step: W = W - (lr * grad_W)
		tensor_sgd_step(mut w, grad_w, 0.05)

		step += 1
	}

	y_final := x @ w
	final_sum := tensor_sum(y_final)
	println('Converged activation sum: ${final_sum} (target: ${tensor_sum(y_target)})')
}
```

---

## 5. Verification & Numerical Convergence

To run the example:
```bash
python scripts/export_safetensors.py
vira run docs/examples/18_autograd_and_safetensors.vir
```

Expected execution output:
```text
=== 1. NATIVE SAFETENSORS ZERO-COPY LOADER ===
Loaded SafeTensors weight 'fc1.weight' (shape 4x4, sum=102.00)
=== 2. FORWARD EVALUATION ===
Initial forward activation sum: 408.00
=== 3. AUTOGRAD SIMD GRADIENT DESCENT ===
Step 0: loss=102.00
Step 1: loss=20.40
Step 2: loss=4.08
Converged activation sum: 306.82 (target: 306.00)
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=24
```

The loss decreases exponentially by $(1 - \eta \cdot \|X\|^2) = (1 - 0.05 \cdot 16) = 0.20$ each step:
- Step 0: $102.00$
- Step 1: $102.00 \times 0.20 = 20.40$
- Step 2: $20.40 \times 0.20 = 4.08$

The final activation sum reaches $306.82$, converging precisely to target $306.00$ with zero leaks (`active_leaks=0`).
