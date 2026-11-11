#ifndef VIRA_RT_TENSOR_OPS_H
#define VIRA_RT_TENSOR_OPS_H

// ============================================================================
// 3.7 TENSOR SHAPE, VIEW, LINEAR, MATMUL & ELEMENTWISE SIMD OPERATIONS
// ============================================================================

/**
 * Reshapes a tensor safely, returning a view if non-arena or reallocating in arena if required.
 * @param opt_arena Optional arena pointer.
 * @param t Source tensor.
 * @param n_dims Target dimension count.
 * @param dims Target dimension sizes.
 * @return Reshaped tensor.
 * Usage: Internal safe reshape helper.
 */
static inline vira_tensor_f32
_vira_tensor_reshape_safe(vira_arena *opt_arena, vira_tensor_f32 t,
                          int32_t n_dims, const int32_t *dims) {
  _vira_tensor_norm_shape(&t);
  if (t.is_arena && opt_arena != NULL && vira_arena_contains(opt_arena, t.data)) {
    return vira_tensor_view_nd(t, n_dims, dims);
  }
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, n_dims, dims)
                : vira_tensor_f32_alloc_nd(n_dims, dims);
  if (out.rows * out.cols != t.rows * t.cols) {
    fprintf(stderr,
            "vira tensor error: reshape element count mismatch (%d -> %d)\n",
            t.rows * t.cols, out.rows * out.cols);
    exit(1);
  }
  _vira_tensor_sync_host_f32(t.data);
  memcpy(out.data, t.data, (size_t)out.rows * (size_t)out.cols * sizeof(vira_f32));
  return out;
}

/**
 * Reshapes a tensor into 2D dimensions (d0, d1).
 * @param opt_arena Optional arena pointer.
 * @param t Source tensor.
 * @param d0 Rows.
 * @param d1 Columns.
 * @return Reshaped 2D tensor.
 * Usage: Reshaping tensors to 2D matrices (`tensor.reshape(rows, cols)`).
 */
static inline vira_tensor_f32 vira_tensor_reshape_2d(vira_arena *opt_arena,
                                                     vira_tensor_f32 t,
                                                     int32_t d0, int32_t d1) {
  int32_t dims[2] = {d0, d1};
  return _vira_tensor_reshape_safe(opt_arena, t, 2, dims);
}

/**
 * Reshapes a tensor into 3D dimensions (d0, d1, d2).
 * @param opt_arena Optional arena pointer.
 * @param t Source tensor.
 * @param d0 Dimension 0.
 * @param d1 Dimension 1.
 * @param d2 Dimension 2.
 * @return Reshaped 3D tensor.
 * Usage: Reshaping tensors to 3D.
 */
static inline vira_tensor_f32 vira_tensor_reshape_3d(vira_arena *opt_arena,
                                                     vira_tensor_f32 t,
                                                     int32_t d0, int32_t d1,
                                                     int32_t d2) {
  int32_t dims[3] = {d0, d1, d2};
  return _vira_tensor_reshape_safe(opt_arena, t, 3, dims);
}

/**
 * Reshapes a tensor into 4D dimensions (d0, d1, d2, d3).
 * @param opt_arena Optional arena pointer.
 * @param t Source tensor.
 * @param d0 Dimension 0.
 * @param d1 Dimension 1.
 * @param d2 Dimension 2.
 * @param d3 Dimension 3.
 * @return Reshaped 4D tensor.
 * Usage: Reshaping tensors to 4D (batch, heads, seq, dim).
 */
static inline vira_tensor_f32 vira_tensor_reshape_4d(vira_arena *opt_arena,
                                                     vira_tensor_f32 t,
                                                     int32_t d0, int32_t d1,
                                                     int32_t d2, int32_t d3) {
  int32_t dims[4] = {d0, d1, d2, d3};
  return _vira_tensor_reshape_safe(opt_arena, t, 4, dims);
}

// Transposes the last two dimensions `(..., R, C) -> (..., C, R)` for 2D/3D/4D tensors
/**
 * Transposes the trailing two dimensions of an N-D tensor.
 * @param opt_arena Optional arena pointer.
 * @param a Source tensor.
 * @return Transposed tensor.
 * Usage: Batched matrix transpose operations.
 */
static inline vira_tensor_f32 vira_tensor_transpose_nd(vira_arena *opt_arena,
                                                       vira_tensor_f32 a) {
  _vira_tensor_norm_shape(&a);
  _vira_tensor_sync_host_f32(a.data);
  if (a.n_dims == 1) {
    int32_t dims[2] = {a.dims[0], 1};
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, dims)
                  : vira_tensor_f32_alloc_nd(2, dims);
    memcpy(out.data, a.data, (size_t)a.dims[0] * sizeof(vira_f32));
    return out;
  }
  int32_t out_dims[4] = {a.dims[0], a.dims[1], a.dims[2], a.dims[3]};
  int32_t R = a.dims[a.n_dims - 2];
  int32_t C = a.dims[a.n_dims - 1];
  out_dims[a.n_dims - 2] = C;
  out_dims[a.n_dims - 1] = R;
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, a.n_dims, out_dims)
                : vira_tensor_f32_alloc_nd(a.n_dims, out_dims);
  int32_t batches = a.rows / R;
  size_t slice_elems = (size_t)R * (size_t)C;
  for (int32_t b = 0; b < batches; b++) {
    const vira_f32 *src = a.data + (size_t)b * slice_elems;
    vira_f32 *dst = out.data + (size_t)b * slice_elems;
    for (int32_t r = 0; r < R; r++) {
      for (int32_t c = 0; c < C; c++) {
        dst[c * R + r] = src[r * C + c];
      }
    }
  }
  return out;
}

/**
 * Retrieves a scalar float element from a 2D tensor with bounds checking.
 * @param t 2D tensor.
 * @param row Zero-based row index.
 * @param col Zero-based column index.
 * @return Float value at (row, col).
 * Usage: Element access on matrices (`matrix.get(r, c)`).
 */
static inline vira_f32 vira_tensor_f32_get(vira_tensor_f32 t, int32_t row,
                                           int32_t col) {
  _vira_tensor_norm_shape(&t);
  if (row < 0 || row >= t.rows || col < 0 || col >= t.cols) {
    fprintf(stderr,
            "vira tensor error: .get(%d, %d) out of bounds for shape (%dx%d)\n",
            row, col, t.rows, t.cols);
    exit(1);
  }
  _vira_tensor_sync_host_f32(t.data);
  return t.data[(size_t)row * (size_t)t.cols + (size_t)col];
}

/**
 * Sets a scalar float element in a 2D tensor with bounds checking.
 * @param t 2D tensor.
 * @param row Zero-based row index.
 * @param col Zero-based column index.
 * @param val Float value to assign.
 * Usage: Element assignment on matrices (`matrix.set(r, c, val)`).
 */
static inline void vira_tensor_f32_set(vira_tensor_f32 t, int32_t row,
                                       int32_t col, vira_f32 val) {
  _vira_tensor_norm_shape(&t);
  if (row < 0 || row >= t.rows || col < 0 || col >= t.cols) {
    fprintf(stderr,
            "vira tensor error: .set(%d, %d) out of bounds for shape (%dx%d)\n",
            row, col, t.rows, t.cols);
    exit(1);
  }
  _vira_tensor_sync_host_f32(t.data);
  t.data[(size_t)row * (size_t)t.cols + (size_t)col] = val;
  _vira_rt_quant_remove(t.data);
  _vira_gpu_evict_ptr(t.data);
}

/**
 * Copies a source 1D tensor row vector into a target tensor row using SIMD instructions.
 * @param dst Destination 2D tensor.
 * @param dst_row Row index in destination.
 * @param src Source 1D tensor row.
 * Usage: Efficient vector row staging.
 */
static inline void vira_tensor_f32_copy_row(vira_tensor_f32 dst,
                                            int32_t dst_row,
                                            vira_tensor_f32 src) {
  _vira_tensor_norm_shape(&dst);
  _vira_tensor_norm_shape(&src);
  if (dst_row < 0 || dst_row >= dst.rows) {
    fprintf(stderr,
            "vira tensor error: .copy_row(%d) row out of bounds [0, %d)\n",
            dst_row, dst.rows);
    exit(1);
  }
  if (src.cols != dst.cols || src.rows < 1) {
    fprintf(stderr,
            "vira tensor error: .copy_row column mismatch dst=(%dx%d) vs src=(%dx%d)\n",
            dst.rows, dst.cols, src.rows, src.cols);
    exit(1);
  }
  if (!_vira_tls_gpu_fallback && _vira_gpu_copy_row(dst, dst_row, src)) {
    return;
  }
  _vira_tensor_sync_host_f32(dst.data);
  _vira_tensor_sync_host_f32(src.data);
  if (dst.data == src.data) {
    memmove(&dst.data[(size_t)dst_row * (size_t)dst.cols], src.data,
            (size_t)dst.cols * sizeof(vira_f32));
  } else {
    memcpy(&dst.data[(size_t)dst_row * (size_t)dst.cols], src.data,
           (size_t)dst.cols * sizeof(vira_f32));
  }
  _vira_rt_quant_remove(dst.data);
  _vira_gpu_evict_ptr(dst.data);
}

/**
 * Copies multiple contiguous rows from a 2D source tensor into a target tensor starting at `dst_row_start`.
 * @param dst Destination 2D tensor.
 * @param dst_row_start Starting row index in destination.
 * @param src Source 2D tensor of shape (rows, dst.cols).
 * Usage: Batched KV-cache row scatter (`k_cache.copy_rows(start_pos, k_seq)`).
 */
static inline void vira_tensor_f32_copy_rows(vira_tensor_f32 dst,
                                             int32_t dst_row_start,
                                             vira_tensor_f32 src) {
  _vira_tensor_norm_shape(&dst);
  _vira_tensor_norm_shape(&src);
  if (dst_row_start < 0 || src.rows <= 0 ||
      dst_row_start + src.rows > dst.rows) {
    fprintf(stderr,
            "vira tensor error: .copy_rows(%d, %d rows) out of bounds [0, %d)\n",
            dst_row_start, src.rows, dst.rows);
    exit(1);
  }
  if (src.cols != dst.cols) {
    fprintf(stderr,
            "vira tensor error: .copy_rows column mismatch dst=(%dx%d) vs src=(%dx%d)\n",
            dst.rows, dst.cols, src.rows, src.cols);
    exit(1);
  }
  if (!_vira_tls_gpu_fallback &&
      _vira_gpu_copy_rows(dst, dst_row_start, src)) {
    return;
  }
  _vira_tensor_sync_host_f32(dst.data);
  _vira_tensor_sync_host_f32(src.data);
  size_t copy_bytes = (size_t)src.rows * (size_t)src.cols * sizeof(vira_f32);
  float *dst_ptr = &dst.data[(size_t)dst_row_start * (size_t)dst.cols];
  if (dst.data == src.data) {
    memmove(dst_ptr, src.data, copy_bytes);
  } else {
    memcpy(dst_ptr, src.data, copy_bytes);
  }
  _vira_rt_quant_remove(dst.data);
  _vira_gpu_evict_ptr(dst.data);
}

/**
 * Extracts a contiguous sub-range of rows from a 2D tensor into a new tensor.
 * @param opt_arena Optional arena pointer.
 * @param t Source 2D tensor.
 * @param start_row Starting row index (inclusive).
 * @param end_row Ending row index (exclusive).
 * @return Sliced 2D tensor.
 * Usage: Row slicing and mini-batch extraction.
 */
static inline vira_tensor_f32 vira_tensor_slice_rows(vira_arena *opt_arena,
                                                     vira_tensor_f32 t,
                                                     int32_t start_row,
                                                     int32_t end_row) {
  _vira_tensor_norm_shape(&t);
  if (start_row < 0 || end_row <= start_row || end_row > t.rows) {
    fprintf(stderr,
            "vira tensor error: .slice_rows(%d, %d) invalid for rows=%d\n",
            start_row, end_row, t.rows);
    exit(1);
  }
  if (!_vira_tls_gpu_fallback) {
    vira_tensor_f32 gpu_out = {0};
    if (_vira_gpu_slice_rows(opt_arena, t, start_row, end_row, &gpu_out)) {
      return gpu_out;
    }
  }
  _vira_tensor_sync_host_f32(t.data);
  int32_t n_rows = end_row - start_row;
  int32_t dims[2] = {n_rows, t.cols};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, dims)
                : vira_tensor_f32_alloc_nd(2, dims);
  memcpy(out.data, &t.data[(size_t)start_row * (size_t)t.cols],
         (size_t)n_rows * (size_t)t.cols * sizeof(vira_f32));
  return out;
}

/**
 * Blocked SIMD inner kernel computing matrix multiplication for a contiguous range of rows.
 * @param a_data Matrix A float data.
 * @param b_data Matrix B float data.
 * @param c_data Matrix C destination float data.
 * @param r_start Start row index in C.
 * @param r_end End row index in C.
 * @param K Inner dimension.
 * @param N Columns in C.
 * Usage: High-performance AVX2/SSE2/NEON blocked matrix multiply worker kernel.
 */
static inline void _vira_matmul_kernel_2d_rows(const vira_f32 *a_data,
                                               const vira_f32 *b_data,
                                               vira_f32 *c_data,
                                               int32_t r_start, int32_t r_end,
                                               int32_t K, int32_t N) {
  for (int32_t si = r_start; si < r_end; si += 64) {
    int32_t imax = (si + 64 < r_end) ? (si + 64) : r_end;
    for (int32_t sk = 0; sk < K; sk += 64) {
      int32_t kmax = (sk + 64 < K) ? (sk + 64) : K;
      for (int32_t sj = 0; sj < N; sj += 64) {
        int32_t jmax = (sj + 64 < N) ? (sj + 64) : N;

        for (int32_t i = si; i < imax; i++) {
          vira_f32 *c_row = &c_data[(size_t)i * (size_t)N];
          for (int32_t k = sk; k < kmax; k++) {
            const vira_f32 a_ik = a_data[(size_t)i * (size_t)K + (size_t)k];
            const vira_f32 *b_row = &b_data[(size_t)k * (size_t)N];
            int32_t j = sj;

#if defined(__AVX2__)
            __m256 va = _mm256_set1_ps(a_ik);
            for (; j + 8 <= jmax; j += 8) {
              __m256 vc = _mm256_loadu_ps(&c_row[j]);
              __m256 vb = _mm256_loadu_ps(&b_row[j]);
#if defined(__FMA__)
              vc = _mm256_fmadd_ps(va, vb, vc);
#else
              vc = _mm256_add_ps(_mm256_mul_ps(va, vb), vc);
#endif
              _mm256_storeu_ps(&c_row[j], vc);
            }
#elif defined(__ARM_NEON) || defined(__aarch64__)
            for (; j + 8 <= jmax; j += 8) {
              float32x4_t vc0 = vld1q_f32(&c_row[j]);
              float32x4_t vc1 = vld1q_f32(&c_row[j + 4]);
              float32x4_t vb0 = vld1q_f32(&b_row[j]);
              float32x4_t vb1 = vld1q_f32(&b_row[j + 4]);
              vc0 = vfmaq_n_f32(vc0, vb0, a_ik);
              vc1 = vfmaq_n_f32(vc1, vb1, a_ik);
              vst1q_f32(&c_row[j], vc0);
              vst1q_f32(&c_row[j + 4], vc1);
            }
            for (; j + 4 <= jmax; j += 4) {
              float32x4_t vc0 = vld1q_f32(&c_row[j]);
              float32x4_t vb0 = vld1q_f32(&b_row[j]);
              vc0 = vfmaq_n_f32(vc0, vb0, a_ik);
              vst1q_f32(&c_row[j], vc0);
            }
#elif defined(__SSE2__)
            __m128 va = _mm_set1_ps(a_ik);
            for (; j + 4 <= jmax; j += 4) {
              __m128 vc = _mm_loadu_ps(&c_row[j]);
              __m128 vb = _mm_loadu_ps(&b_row[j]);
              vc = _mm_add_ps(_mm_mul_ps(va, vb), vc);
              _mm_storeu_ps(&c_row[j], vc);
            }
#endif
            for (; j < jmax; j++) {
              c_row[j] += a_ik * b_row[j];
            }
          }
        }
      }
    }
  }
}

typedef struct {
  const vira_f32 *a_base;
  const vira_f32 *b_base;
  vira_f32 *c_base;
  int32_t M;
  int32_t K;
  int32_t N;
  int32_t B1;
  int32_t A_B0;
  int32_t A_B1;
  int32_t B_B0;
  int32_t B_B1;
  int32_t gr_start; // global row start in [0, total_batches * M)
  int32_t gr_end;   // global row end
} _vira_matmul_job;

/**
 * Worker thread task callback executing partitioned matrix multiplication jobs.
 * @param raw Pointer to `_vira_matmul_job` environment structure.
 * Usage: Invoked by scheduler worker threads for multi-threaded parallel `@` matrix multiplies.
 */
static void _vira_matmul_job_fn(void *raw) {
  const _vira_matmul_job *job = (const _vira_matmul_job *)raw;
  const int32_t M = job->M;
  const int32_t K = job->K;
  const int32_t N = job->N;
  int32_t cur = job->gr_start;
  while (cur < job->gr_end) {
    int32_t b_idx = cur / M;
    int32_t r_start = cur % M;
    int32_t rem_in_batch = M - r_start;
    int32_t rem_total = job->gr_end - cur;
    int32_t r_end = r_start + ((rem_total < rem_in_batch) ? rem_total : rem_in_batch);

    int32_t i0 = b_idx / job->B1;
    int32_t i1 = b_idx % job->B1;
    int32_t a_b = ((job->A_B0 == 1) ? 0 : i0) * job->A_B1 +
                  ((job->A_B1 == 1) ? 0 : i1);
    int32_t b_b = ((job->B_B0 == 1) ? 0 : i0) * job->B_B1 +
                  ((job->B_B1 == 1) ? 0 : i1);

    const vira_f32 *a_ptr = job->a_base + (size_t)a_b * (size_t)M * (size_t)K;
    const vira_f32 *b_ptr = job->b_base + (size_t)b_b * (size_t)K * (size_t)N;
    vira_f32 *c_ptr = job->c_base + (size_t)b_idx * (size_t)M * (size_t)N;

    _vira_matmul_kernel_2d_rows(a_ptr, b_ptr, c_ptr, r_start, r_end, K, N);
    cur += (r_end - r_start);
  }
}

#define VIRA_MATMUL_MT_THRESHOLD (128LL * 128LL * 128LL)

// Hardware-accelerated 2D/3D/4D Batched + Multi-Threaded `@` Matrix Multiplication
/**
 * Executes batched multi-threaded matrix multiplication `a @ b` with hardware SIMD and automatic GPU auto-dispatch.
 * @param opt_arena Optional memory arena for result allocation (or NULL for heap).
 * @param a Left tensor matrix.
 * @param b Right tensor matrix.
 * @return Result tensor of matrix multiplication.
 * Usage: Implements the `@` operator in Vira code (`c = a @ b`).
 */
static inline vira_tensor_f32 vira_tensor_matmul_f32(vira_arena *opt_arena,
                                                     vira_tensor_f32 a,
                                                     vira_tensor_f32 b) {
  _vira_tensor_norm_shape(&a);
  _vira_tensor_norm_shape(&b);

  int32_t a_nd = a.n_dims;
  int32_t b_nd = b.n_dims;
  int32_t M = (a_nd == 1) ? 1 : a.dims[a_nd - 2];
  int32_t K_a = a.dims[a_nd - 1];
  int32_t K_b = (b_nd == 1) ? b.dims[0] : b.dims[b_nd - 2];
  int32_t N = (b_nd == 1) ? 1 : b.dims[b_nd - 1];

  if (K_a != K_b) {
    fprintf(stderr,
            "vira tensor error: matmul '@' inner dimension mismatch (%d vs %d)\n",
            K_a, K_b);
    exit(1);
  }

  if (!_vira_tls_gpu_fallback && a_nd <= 2 && b_nd <= 2 &&
      ((int64_t)M * (int64_t)K_a * (int64_t)N >= 65536LL ||
       _vira_gpu_is_ptr_dirty(a.data) || _vira_gpu_is_ptr_dirty(b.data)) &&
      vira_gpu_is_available() && vira_gpu_is_enabled()) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 gpu_out = vira_gpu_matmul(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return gpu_out;
  }

  _vira_tensor_sync_host_f32(a.data);
  _vira_tensor_sync_host_f32(b.data);

  int32_t a_batch_nd = (a_nd > 2) ? (a_nd - 2) : 0;
  int32_t b_batch_nd = (b_nd > 2) ? (b_nd - 2) : 0;
  int32_t batch_nd = (a_batch_nd > b_batch_nd) ? a_batch_nd : b_batch_nd;

  // Normalize leading batch dims into 2 slots [B0, B1]
  int32_t A_B0 = 1, A_B1 = 1;
  if (a_batch_nd == 1) {
    A_B1 = a.dims[0];
  } else if (a_batch_nd == 2) {
    A_B0 = a.dims[0];
    A_B1 = a.dims[1];
  }

  int32_t B_B0 = 1, B_B1 = 1;
  if (b_batch_nd == 1) {
    B_B1 = b.dims[0];
  } else if (b_batch_nd == 2) {
    B_B0 = b.dims[0];
    B_B1 = b.dims[1];
  }

  if ((A_B0 != B_B0 && A_B0 != 1 && B_B0 != 1) ||
      (A_B1 != B_B1 && A_B1 != 1 && B_B1 != 1)) {
    fprintf(stderr,
            "vira tensor error: matmul '@' batch broadcast mismatch (%dx%d) vs (%dx%d)\n",
            A_B0, A_B1, B_B0, B_B1);
    exit(1);
  }

  int32_t B0 = (A_B0 > B_B0) ? A_B0 : B_B0;
  int32_t B1 = (A_B1 > B_B1) ? A_B1 : B_B1;
  int32_t total_batches = B0 * B1;

  int32_t out_nd = batch_nd + 2;
  int32_t out_dims[4] = {0, 0, 0, 0};
  if (batch_nd == 0) {
    out_dims[0] = M;
    out_dims[1] = N;
  } else if (batch_nd == 1) {
    out_dims[0] = B1;
    out_dims[1] = M;
    out_dims[2] = N;
  } else {
    out_dims[0] = B0;
    out_dims[1] = B1;
    out_dims[2] = M;
    out_dims[3] = N;
  }

  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, out_nd, out_dims)
                : vira_tensor_f32_alloc_nd(out_nd, out_dims);

  int32_t total_rows = total_batches * M;
  int64_t total_ops =
      (int64_t)total_rows * (int64_t)K_a * (int64_t)N;

  if (total_ops >= VIRA_MATMUL_MT_THRESHOLD && total_rows >= 4 &&
      !_vira_in_active_actor()) {
    vira_scheduler_ensure_init();
    int32_t nw = vira_scheduler_num_workers();
    if (nw > 16)
      nw = 16;
    if (nw > total_rows)
      nw = total_rows;
    if (nw > 1) {
      _vira_matmul_job jobs[16];
      vira_task tasks[16];
      int32_t base_rows = total_rows / nw;
      int32_t extra = total_rows % nw;
      int32_t cur = 0;
      for (int32_t w = 0; w < nw; w++) {
        int32_t cnt = base_rows + (w < extra ? 1 : 0);
        jobs[w] = (_vira_matmul_job){.a_base = a.data,
                                     .b_base = b.data,
                                     .c_base = out.data,
                                     .M = M,
                                     .K = K_a,
                                     .N = N,
                                     .B1 = B1,
                                     .A_B0 = A_B0,
                                     .A_B1 = A_B1,
                                     .B_B0 = B_B0,
                                     .B_B1 = B_B1,
                                     .gr_start = cur,
                                     .gr_end = cur + cnt};
        cur += cnt;
        if (w > 0) {
          tasks[w] = (vira_task){.fn = _vira_matmul_job_fn,
                                 .env = &jobs[w],
                                 .next_actor_msg = NULL,
                                 .done = 0};
          vira_scheduler_submit_async(&tasks[w]);
        }
      }
      _vira_matmul_job_fn(&jobs[0]);
      for (int32_t w = 1; w < nw; w++) {
        vira_scheduler_await(&tasks[w]);
      }
      return out;
    }
  }

  _vira_matmul_job single = {.a_base = a.data,
                             .b_base = b.data,
                             .c_base = out.data,
                             .M = M,
                             .K = K_a,
                             .N = N,
                             .B1 = B1,
                             .A_B0 = A_B0,
                             .A_B1 = A_B1,
                             .B_B0 = B_B0,
                             .B_B1 = B_B1,
                             .gr_start = 0,
                             .gr_end = total_rows};
  _vira_matmul_job_fn(&single);
  return out;
}

typedef struct {
  const vira_f32 *x_base;
  const vira_f32 *w_base;
  vira_f32 *y_base;
  int32_t M;
  int32_t K;
  int32_t N;
  int32_t n_start;
  int32_t n_end;
} _vira_linear_job;

/**
 * Worker thread task callback executing parallel linear layer projection `x @ w^T`.
 * @param raw_env Pointer to `_vira_linear_job` environment structure.
 * Usage: Invoked by scheduler worker threads for parallel multi-threaded linear projections.
 */
static void _vira_linear_job_fn(void *raw_env) {
  _vira_linear_job *j = (_vira_linear_job *)raw_env;
  int32_t M = j->M;
  int32_t K = j->K;
  int32_t N = j->N;
  int32_t n_start = j->n_start;
  int32_t n_end = j->n_end;

  for (int32_t m = 0; m < M; ++m) {
    const vira_f32 *x_row = &j->x_base[(size_t)m * (size_t)K];
    vira_f32 *y_row = &j->y_base[(size_t)m * (size_t)N];
    for (int32_t n = n_start; n < n_end; ++n) {
      const vira_f32 *w_row = &j->w_base[(size_t)n * (size_t)K];
      int32_t k = 0;
      float acc = 0.0f;
#if defined(__AVX2__)
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();
      __m256 sum2 = _mm256_setzero_ps();
      __m256 sum3 = _mm256_setzero_ps();
      for (; k + 32 <= K; k += 32) {
        __m256 x0 = _mm256_loadu_ps(&x_row[k]);
        __m256 w0 = _mm256_loadu_ps(&w_row[k]);
        __m256 x1 = _mm256_loadu_ps(&x_row[k + 8]);
        __m256 w1 = _mm256_loadu_ps(&w_row[k + 8]);
        __m256 x2 = _mm256_loadu_ps(&x_row[k + 16]);
        __m256 w2 = _mm256_loadu_ps(&w_row[k + 16]);
        __m256 x3 = _mm256_loadu_ps(&x_row[k + 24]);
        __m256 w3 = _mm256_loadu_ps(&w_row[k + 24]);
#if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(x0, w0, sum0);
        sum1 = _mm256_fmadd_ps(x1, w1, sum1);
        sum2 = _mm256_fmadd_ps(x2, w2, sum2);
        sum3 = _mm256_fmadd_ps(x3, w3, sum3);
#else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(x0, w0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(x1, w1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(x2, w2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(x3, w3));
#endif
      }
      sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1),
                           _mm256_add_ps(sum2, sum3));
      for (; k + 8 <= K; k += 8) {
        __m256 x0 = _mm256_loadu_ps(&x_row[k]);
        __m256 w0 = _mm256_loadu_ps(&w_row[k]);
#if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(x0, w0, sum0);
#else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(x0, w0));
#endif
      }
      __m128 lo = _mm256_castps256_ps128(sum0);
      __m128 hi = _mm256_extractf128_ps(sum0, 1);
      __m128 s128 = _mm_add_ps(lo, hi);
      s128 = _mm_hadd_ps(s128, s128);
      s128 = _mm_hadd_ps(s128, s128);
      acc = _mm_cvtss_f32(s128);
#elif defined(__ARM_NEON) || defined(__aarch64__)
      float32x4_t sum0 = vdupq_n_f32(0.0f);
      float32x4_t sum1 = vdupq_n_f32(0.0f);
      for (; k + 8 <= K; k += 8) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(&x_row[k]), vld1q_f32(&w_row[k]));
        sum1 =
            vfmaq_f32(sum1, vld1q_f32(&x_row[k + 4]), vld1q_f32(&w_row[k + 4]));
      }
      sum0 = vaddq_f32(sum0, sum1);
      acc = vaddvq_f32(sum0);
#endif
      for (; k < K; ++k) {
        acc += x_row[k] * w_row[k];
      }
      y_row[n] = acc;
    }
  }
}

// Multi-threaded AVX2+FMA / NEON Linear Projection `x.linear(w)` computing `x @ w^T`
// where `x` has shape `(M, K)` and `w` has shape `(N, K)` (GGUF row-major weight layout),
// producing `(M, N)` without transposing `w`.
/**
 * Computes hardware-accelerated linear layer projection `y = x @ w^T` with SIMD and GPU auto-dispatch.
 * @param opt_arena Optional memory arena for output allocation.
 * @param x Input activation tensor (M, K).
 * @param w Weight matrix tensor (N, K).
 * @return Output tensor (M, N).
 * Usage: Core forward projection operator in dense neural networks and Transformers (`x.linear(w)`).
 */
static inline vira_tensor_f32 vira_tensor_linear(vira_arena *opt_arena,
                                                 vira_tensor_f32 x,
                                                 vira_tensor_f32 w) {
  _vira_tensor_norm_shape(&x);
  _vira_tensor_norm_shape(&w);
  int32_t M = x.rows;
  int32_t K = x.cols;
  int32_t N = w.rows;
  if (K != w.cols) {
    fprintf(stderr,
            "vira tensor error: .linear() inner dimension mismatch x=(%dx%d) vs w=(%dx%d)\n",
            M, K, N, w.cols);
    exit(1);
  }

  int64_t total_ops = (int64_t)M * (int64_t)N * (int64_t)K;
  if (!_vira_tls_gpu_fallback &&
      (total_ops >= 65536LL || _vira_gpu_is_ptr_dirty(x.data) ||
       _vira_gpu_is_ptr_dirty(w.data)) &&
      vira_gpu_is_available() && vira_gpu_is_enabled()) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 gpu_out = vira_gpu_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return gpu_out;
  }

  {
    vira_tensor_f32 q_out = {0};
    if (_vira_rt_quant_cpu_linear(opt_arena, x, w, &q_out)) {
      return q_out;
    }
  }

  _vira_tensor_sync_host_f32(x.data);
  _vira_tensor_sync_host_f32(w.data);

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, out_dims)
                : vira_tensor_f32_alloc_nd(2, out_dims);

  if (total_ops >= 262144LL && N >= 32 && !_vira_in_active_actor()) {
    vira_scheduler_ensure_init();
    int32_t nw = vira_scheduler_num_workers();
    if (nw > 16)
      nw = 16;
    if (nw > N)
      nw = N;
    if (nw > 1) {
      _vira_linear_job jobs[16];
      vira_task tasks[16];
      int32_t base_n = N / nw;
      int32_t extra = N % nw;
      int32_t cur = 0;
      for (int32_t wi = 0; wi < nw; ++wi) {
        int32_t cnt = base_n + (wi < extra ? 1 : 0);
        jobs[wi] = (_vira_linear_job){.x_base = x.data,
                                      .w_base = w.data,
                                      .y_base = out.data,
                                      .M = M,
                                      .K = K,
                                      .N = N,
                                      .n_start = cur,
                                      .n_end = cur + cnt};
        cur += cnt;
        if (wi > 0) {
          tasks[wi] = (vira_task){.fn = _vira_linear_job_fn,
                                  .env = &jobs[wi],
                                  .next_actor_msg = NULL,
                                  .done = 0};
          vira_scheduler_submit_async(&tasks[wi]);
        }
      }
      _vira_linear_job_fn(&jobs[0]);
      for (int32_t wi = 1; wi < nw; ++wi) {
        vira_scheduler_await(&tasks[wi]);
      }
      return out;
    }
  }

  _vira_linear_job single = {.x_base = x.data,
                             .w_base = w.data,
                             .y_base = out.data,
                             .M = M,
                             .K = K,
                             .N = N,
                             .n_start = 0,
                             .n_end = N};
  _vira_linear_job_fn(&single);
  return out;
}

// SIMD-accelerated tensor sum reduction
/**
 * Computes the scalar sum reduction across all elements of a tensor using SIMD vector instructions.
 * @param t Tensor instance.
 * @return Sum of all elements as float.
 * Usage: Vectorized tensor reduction (`tensor.sum()`).
 */
static inline vira_f32 vira_tensor_sum(vira_tensor_f32 t) {
  _vira_tensor_norm_shape(&t);
  _vira_tensor_sync_host_f32(t.data);
  int32_t total = t.rows * t.cols;
  int32_t i = 0;
  vira_f32 sum = 0.0f;

#if defined(__AVX2__)
  __m256 vsum = _mm256_setzero_ps();
  for (; i + 8 <= total; i += 8) {
    vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(&t.data[i]));
  }
  __m128 lo = _mm256_castps256_ps128(vsum);
  __m128 hi = _mm256_extractf128_ps(vsum, 1);
  __m128 s128 = _mm_add_ps(lo, hi);
  s128 = _mm_hadd_ps(s128, s128);
  s128 = _mm_hadd_ps(s128, s128);
  sum += _mm_cvtss_f32(s128);
#elif defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t vsum = vdupq_n_f32(0.0f);
  for (; i + 4 <= total; i += 4) {
    vsum = vaddq_f32(vsum, vld1q_f32(&t.data[i]));
  }
  sum += vaddvq_f32(vsum);
#endif
  for (; i < total; i++) {
    sum += t.data[i];
  }
  return sum;
}

// SIMD-accelerated ReLU activation: out = max(0, t) (preserves 1D..4D shape)
/**
 * Applies the rectified linear unit activation `max(0, x)` elementwise across a tensor using SIMD.
 * @param opt_arena Optional memory arena for output allocation.
 * @param t Input tensor.
 * @return Activated tensor.
 * Usage: Standard neural network non-linear activation (`tensor.relu()`).
 */
static inline vira_tensor_f32 vira_tensor_relu(vira_arena *opt_arena,
                                               vira_tensor_f32 t) {
  _vira_tensor_norm_shape(&t);
  _vira_tensor_sync_host_f32(t.data);
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, t.n_dims, t.dims)
                : vira_tensor_f32_alloc_nd(t.n_dims, t.dims);
  int32_t total = t.rows * t.cols;
  int32_t i = 0;

#if defined(__AVX2__)
  __m256 vzero = _mm256_setzero_ps();
  for (; i + 8 <= total; i += 8) {
    __m256 v = _mm256_loadu_ps(&t.data[i]);
    __m256 vr = _mm256_max_ps(v, vzero);
    _mm256_storeu_ps(&out.data[i], vr);
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t vzero = vdupq_n_f32(0.0f);
  for (; i + 4 <= total; i += 4) {
    float32x4_t v = vld1q_f32(&t.data[i]);
    float32x4_t vr = vmaxq_f32(v, vzero);
    vst1q_f32(&out.data[i], vr);
  }
#elif defined(__SSE2__)
  __m128 vzero = _mm_setzero_ps();
  for (; i + 4 <= total; i += 4) {
    __m128 v = _mm_loadu_ps(&t.data[i]);
    __m128 vr = _mm_max_ps(v, vzero);
    _mm_storeu_ps(&out.data[i], vr);
  }
#endif
  for (; i < total; i++) {
    out.data[i] = t.data[i] > 0.0f ? t.data[i] : 0.0f;
  }
  return out;
}

/**
 * Vectorized SIMD addition kernel adding arrays `pa` and `pb` into `dst`.
 * @param dst Destination buffer.
 * @param pa First input buffer.
 * @param pb Second input buffer.
 * @param count Number of elements.
 * Usage: Internal SIMD helper for elementwise tensor addition.
 */
static inline void _vira_vec_add_f32(vira_f32 *dst, const vira_f32 *pa,
                                     const vira_f32 *pb, int32_t count) {
  int32_t i = 0;
#if defined(__AVX2__)
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(&pa[i]);
    __m256 vb = _mm256_loadu_ps(&pb[i]);
    _mm256_storeu_ps(&dst[i], _mm256_add_ps(va, vb));
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  for (; i + 4 <= count; i += 4) {
    float32x4_t va = vld1q_f32(&pa[i]);
    float32x4_t vb = vld1q_f32(&pb[i]);
    vst1q_f32(&dst[i], vaddq_f32(va, vb));
  }
#elif defined(__SSE2__)
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(&pa[i]);
    __m128 vb = _mm_loadu_ps(&pb[i]);
    _mm_storeu_ps(&dst[i], _mm_add_ps(va, vb));
  }
#endif
  for (; i < count; i++) {
    dst[i] = pa[i] + pb[i];
  }
}

/**
 * Vectorized SIMD subtraction kernel subtracting `pb` from `pa` into `dst`.
 * @param dst Destination buffer.
 * @param pa First input buffer.
 * @param pb Second input buffer.
 * @param count Number of elements.
 * Usage: Internal SIMD helper for elementwise tensor subtraction.
 */
static inline void _vira_vec_sub_f32(vira_f32 *dst, const vira_f32 *pa,
                                     const vira_f32 *pb, int32_t count) {
  int32_t i = 0;
#if defined(__AVX2__)
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(&pa[i]);
    __m256 vb = _mm256_loadu_ps(&pb[i]);
    _mm256_storeu_ps(&dst[i], _mm256_sub_ps(va, vb));
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  for (; i + 4 <= count; i += 4) {
    float32x4_t va = vld1q_f32(&pa[i]);
    float32x4_t vb = vld1q_f32(&pb[i]);
    vst1q_f32(&dst[i], vsubq_f32(va, vb));
  }
#elif defined(__SSE2__)
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(&pa[i]);
    __m128 vb = _mm_loadu_ps(&pb[i]);
    _mm_storeu_ps(&dst[i], _mm_sub_ps(va, vb));
  }
#endif
  for (; i < count; i++) {
    dst[i] = pa[i] - pb[i];
  }
}

// SIMD-accelerated tensor addition: out = a + b (supports 1D..4D + trailing-dim bias broadcast)
/**
 * Computes elementwise addition of two tensors `a + b` with SIMD acceleration.
 * @param opt_arena Optional memory arena for output allocation.
 * @param a First tensor.
 * @param b Second tensor.
 * @return Result tensor.
 * Usage: Implements the `+` operator on tensors (`a + b`).
 */
static inline vira_tensor_f32
vira_tensor_add(vira_arena *opt_arena, vira_tensor_f32 a, vira_tensor_f32 b) {
  _vira_tensor_norm_shape(&a);
  _vira_tensor_norm_shape(&b);
  size_t a_elems = (size_t)a.rows * (size_t)a.cols;
  size_t b_elems = (size_t)b.rows * (size_t)b.cols;
  int64_t max_elems = (int64_t)(a_elems > b_elems ? a_elems : b_elems);
  if (!_vira_tls_gpu_fallback &&
      (max_elems >= 16384LL || _vira_gpu_is_ptr_dirty(a.data) ||
       _vira_gpu_is_ptr_dirty(b.data)) &&
      vira_gpu_is_available() && vira_gpu_is_enabled()) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 gpu_out = vira_gpu_add(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return gpu_out;
  }
  _vira_tensor_sync_host_f32(a.data);
  _vira_tensor_sync_host_f32(b.data);
  if (a.rows == b.rows && a.cols == b.cols) {
    const vira_tensor_f32 *sh = (a.n_dims >= b.n_dims) ? &a : &b;
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, sh->n_dims, sh->dims)
                  : vira_tensor_f32_alloc_nd(sh->n_dims, sh->dims);
    _vira_vec_add_f32(out.data, a.data, b.data, a.rows * a.cols);
    return out;
  }
  if (b.rows == 1 && b.cols == a.cols) {
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, a.n_dims, a.dims)
                  : vira_tensor_f32_alloc_nd(a.n_dims, a.dims);
    for (int32_t r = 0; r < a.rows; r++) {
      _vira_vec_add_f32(&out.data[(size_t)r * (size_t)a.cols],
                        &a.data[(size_t)r * (size_t)a.cols], b.data, a.cols);
    }
    return out;
  }
  if (a.rows == 1 && a.cols == b.cols) {
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, b.n_dims, b.dims)
                  : vira_tensor_f32_alloc_nd(b.n_dims, b.dims);
    for (int32_t r = 0; r < b.rows; r++) {
      _vira_vec_add_f32(&out.data[(size_t)r * (size_t)b.cols], a.data,
                        &b.data[(size_t)r * (size_t)b.cols], b.cols);
    }
    return out;
  }
  fprintf(stderr, "vira tensor error: add shape mismatch (%dx%d) + (%dx%d)\n",
          a.rows, a.cols, b.rows, b.cols);
  exit(1);
}

// SIMD-accelerated tensor subtraction: out = a - b (supports 1D..4D + trailing-dim bias broadcast)
/**
 * Computes elementwise subtraction of two tensors `a - b` with SIMD acceleration.
 * @param opt_arena Optional memory arena for output allocation.
 * @param a First tensor.
 * @param b Second tensor.
 * @return Result tensor.
 * Usage: Implements the `-` operator on tensors (`a - b`).
 */
static inline vira_tensor_f32
vira_tensor_sub(vira_arena *opt_arena, vira_tensor_f32 a, vira_tensor_f32 b) {
  _vira_tensor_norm_shape(&a);
  _vira_tensor_norm_shape(&b);
  _vira_tensor_sync_host_f32(a.data);
  _vira_tensor_sync_host_f32(b.data);
  if (a.rows == b.rows && a.cols == b.cols) {
    const vira_tensor_f32 *sh = (a.n_dims >= b.n_dims) ? &a : &b;
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, sh->n_dims, sh->dims)
                  : vira_tensor_f32_alloc_nd(sh->n_dims, sh->dims);
    _vira_vec_sub_f32(out.data, a.data, b.data, a.rows * a.cols);
    return out;
  }
  if (b.rows == 1 && b.cols == a.cols) {
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, a.n_dims, a.dims)
                  : vira_tensor_f32_alloc_nd(a.n_dims, a.dims);
    for (int32_t r = 0; r < a.rows; r++) {
      _vira_vec_sub_f32(&out.data[(size_t)r * (size_t)a.cols],
                        &a.data[(size_t)r * (size_t)a.cols], b.data, a.cols);
    }
    return out;
  }
  if (a.rows == 1 && a.cols == b.cols) {
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, b.n_dims, b.dims)
                  : vira_tensor_f32_alloc_nd(b.n_dims, b.dims);
    for (int32_t r = 0; r < b.rows; r++) {
      _vira_vec_sub_f32(&out.data[(size_t)r * (size_t)b.cols], a.data,
                        &b.data[(size_t)r * (size_t)b.cols], b.cols);
    }
    return out;
  }
  fprintf(stderr, "vira tensor error: sub shape mismatch (%dx%d) - (%dx%d)\n",
          a.rows, a.cols, b.rows, b.cols);
  exit(1);
}

// SIMD-accelerated tensor scalar scaling: out = t * scale (preserves 1D..4D shape)
/**
 * Scales all elements in a tensor by a scalar float factor using SIMD instructions.
 * @param opt_arena Optional memory arena for output allocation.
 * @param t Input tensor.
 * @param scale Scalar scaling factor.
 * @return Scaled tensor.
 * Usage: Scalar multiplication on tensors (`tensor * scale`).
 */
static inline vira_tensor_f32
vira_tensor_scale(vira_arena *opt_arena, vira_tensor_f32 t, vira_f32 scale) {
  _vira_tensor_norm_shape(&t);
  int32_t total = t.rows * t.cols;
  if (!_vira_tls_gpu_fallback &&
      (total >= 16384 || _vira_gpu_is_ptr_dirty(t.data)) &&
      vira_gpu_is_available() && vira_gpu_is_enabled()) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 gpu_out = vira_gpu_scale(opt_arena, t, scale);
    _vira_tls_gpu_fallback = false;
    return gpu_out;
  }
  _vira_tensor_sync_host_f32(t.data);
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, t.n_dims, t.dims)
                : vira_tensor_f32_alloc_nd(t.n_dims, t.dims);
  int32_t i = 0;

#if defined(__AVX2__)
  __m256 vs = _mm256_set1_ps(scale);
  for (; i + 8 <= total; i += 8) {
    __m256 v = _mm256_loadu_ps(&t.data[i]);
    _mm256_storeu_ps(&out.data[i], _mm256_mul_ps(v, vs));
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t vs = vdupq_n_f32(scale);
  for (; i + 4 <= total; i += 4) {
    float32x4_t v = vld1q_f32(&t.data[i]);
    vst1q_f32(&out.data[i], vmulq_f32(v, vs));
  }
#elif defined(__SSE2__)
  __m128 vs = _mm_set1_ps(scale);
  for (; i + 4 <= total; i += 4) {
    __m128 v = _mm_loadu_ps(&t.data[i]);
    _mm_storeu_ps(&out.data[i], _mm_mul_ps(v, vs));
  }
#endif
  for (; i < total; i++) {
    out.data[i] = t.data[i] * scale;
  }
  return out;
}

/**
 * Releases tensor backing memory and evicts its address from the GPU weight cache.
 * @param t Pointer to tensor struct.
 * Usage: Destructor for heap-allocated tensors; no-op if tensor is arena-allocated.
 */
static inline void vira_tensor_f32_free(vira_tensor_f32 *t) {
  if (t != NULL && !t->is_arena && !t->is_view && t->raw_alloc != NULL) {
    _vira_gpu_evict_ptr(t->data);
    _vira_rt_quant_free(t->data);
#ifdef _WIN32
    if (t->is_vmem) {
      VirtualFree(t->raw_alloc, 0, MEM_RELEASE);
    } else {
      free(t->raw_alloc);
    }
#else
    free(t->raw_alloc);
#endif
    t->raw_alloc = NULL;
    t->data = NULL;
    t->is_vmem = false;
    _vira_record_free();
  }
}

/**
 * Creates an independent heap-allocated deep clone of a tensor.
 * @param t Source tensor.
 * @return Cloned tensor with its own 64-byte aligned memory buffer.
 * Usage: Deep copy operations on tensors (`tensor.clone()`).
 */
static inline vira_tensor_f32 vira_tensor_f32_clone(vira_tensor_f32 t) {
  if (t.data == NULL || t.rows <= 0 || t.cols <= 0) {
    return (vira_tensor_f32){0};
  }
  _vira_tensor_norm_shape(&t);
  if (!_vira_gpu_is_ptr_dirty(t.data)) {
    vira_tensor_f32 lazy_copy = {0};
    if (_vira_rt_quant_try_clone_lazy(t, &lazy_copy)) {
      _vira_gpu_clone_quant(lazy_copy.data, t.data);
      return lazy_copy;
    }
  }
  _vira_tensor_sync_host_f32(t.data);
  vira_tensor_f32 copy = vira_tensor_f32_alloc_nd(t.n_dims, t.dims);
  size_t bytes = (size_t)t.rows * (size_t)t.cols * sizeof(vira_f32);
  memcpy(copy.data, t.data, bytes);
  _vira_rt_quant_clone_entry(copy.data, t.data);
  _vira_gpu_clone_quant(copy.data, t.data);
  return copy;
}

#endif // VIRA_RT_TENSOR_OPS_H
