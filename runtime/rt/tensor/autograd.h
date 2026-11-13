#ifndef VIRA_RT_TENSOR_AUTOGRAD_H
#define VIRA_RT_TENSOR_AUTOGRAD_H

// ============================================================================
// 15. HARDWARE SIMD AUTOGRAD BACKWARD OPS & OPTIMIZERS
// ============================================================================

/**
 * Transposes a 2D float tensor matrix, swapping rows and columns.
 * @param a Source matrix tensor (M, K).
 * @return Newly allocated transposed tensor (K, M).
 * Usage: Matrix transpose operator (`matrix.transpose()`).
 */
static inline vira_tensor_f32 vira_tensor_transpose(vira_tensor_f32 a) {
  return vira_tensor_transpose_nd(NULL, a);
}

/**
 * Computes gradient with respect to matrix A during backpropagation: `grad_a = grad_out @ B^T`.
 * @param grad_out Upstream output gradient tensor.
 * @param b Forward pass matrix B tensor.
 * @return Gradient tensor for matrix A.
 * Usage: Autograd backward pass for matrix multiplication.
 */
static inline vira_tensor_f32
vira_tensor_matmul_grad_a(vira_tensor_f32 grad_out, vira_tensor_f32 b) {
  vira_tensor_f32 bt = vira_tensor_transpose(b);
  vira_tensor_f32 grad_a = vira_tensor_matmul_f32(NULL, grad_out, bt);
  vira_tensor_f32_free(&bt);
  return grad_a;
}

/**
 * Computes gradient with respect to matrix B during backpropagation: `grad_b = A^T @ grad_out`.
 * @param a Forward pass matrix A tensor.
 * @param grad_out Upstream output gradient tensor.
 * @return Gradient tensor for matrix B.
 * Usage: Autograd backward pass for matrix multiplication.
 */
static inline vira_tensor_f32
vira_tensor_matmul_grad_b(vira_tensor_f32 a, vira_tensor_f32 grad_out) {
  vira_tensor_f32 at = vira_tensor_transpose(a);
  vira_tensor_f32 grad_b = vira_tensor_matmul_f32(NULL, at, grad_out);
  vira_tensor_f32_free(&at);
  return grad_b;
}

/**
 * Backpropagates gradients through a ReLU non-linearity: `grad_in = grad_out * (input > 0)`.
 * @param input Forward pass input tensor.
 * @param grad_out Upstream output gradient tensor.
 * @return Gradient tensor with respect to input.
 * Usage: Autograd backward pass for ReLU activations.
 */
static inline vira_tensor_f32
vira_tensor_relu_backward(vira_tensor_f32 input, vira_tensor_f32 grad_out) {
  _vira_tensor_norm_shape(&input);
  _vira_tensor_sync_host_f32(input.data);
  _vira_tensor_sync_host_f32(grad_out.data);
  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(input.n_dims, input.dims);
  int64_t total = (int64_t)input.rows * input.cols;
  int64_t i = 0;

#if defined(__AVX2__)
  __m256 zero = _mm256_setzero_ps();
  for (; i + 7 < total; i += 8) {
    __m256 in_v = _mm256_loadu_ps(&input.data[i]);
    __m256 grad_v = _mm256_loadu_ps(&grad_out.data[i]);
    __m256 mask = _mm256_cmp_ps(in_v, zero, _CMP_GT_OQ);
    __m256 res_v = _mm256_and_ps(mask, grad_v);
    _mm256_storeu_ps(&out.data[i], res_v);
  }
#endif

  for (; i < total; ++i) {
    out.data[i] = (input.data[i] > 0.0f) ? grad_out.data[i] : 0.0f;
  }
  return out;
}

/**
 * Performs an in-place stochastic gradient descent optimizer update: `param -= lr * grad`.
 * @param param Pointer to parameter weight tensor to update in-place.
 * @param grad Gradient tensor with identical dimensions.
 * @param lr Learning rate multiplier.
 * Usage: Model parameter optimization during training steps.
 */
static inline void vira_tensor_sgd_step(vira_tensor_f32 *param,
                                        vira_tensor_f32 grad, float lr) {
  if (!param || !param->data || !grad.data)
    return;
  _vira_tensor_sync_host_f32(param->data);
  _vira_tensor_sync_host_f32(grad.data);
  int64_t total = (int64_t)param->rows * param->cols;
  int64_t i = 0;

#if defined(__AVX2__)
  __m256 lr_v = _mm256_set1_ps(lr);
  for (; i + 7 < total; i += 8) {
    __m256 p_v = _mm256_loadu_ps(&param->data[i]);
    __m256 g_v = _mm256_loadu_ps(&grad.data[i]);
    __m256 step = _mm256_mul_ps(lr_v, g_v);
    __m256 updated = _mm256_sub_ps(p_v, step);
    _mm256_storeu_ps(&param->data[i], updated);
  }
#endif

  for (; i < total; ++i) {
    param->data[i] -= lr * grad.data[i];
  }
  _vira_rt_quant_remove(param->data);
  _vira_gpu_evict_ptr(param->data);
}

#endif // VIRA_RT_TENSOR_AUTOGRAD_H
