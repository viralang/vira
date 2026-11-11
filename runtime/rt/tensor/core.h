#ifndef VIRA_RT_TENSOR_CORE_H
#define VIRA_RT_TENSOR_CORE_H

// ============================================================================
// 3. N-D AI TENSORS + MULTI-THREADED CPU SIMD (`AVX2` / `SSE2` / `ARM NEON`)
// ============================================================================

#ifndef _VIRA_TASK_DEFINED
#define _VIRA_TASK_DEFINED
typedef void (*vira_task_fn)(void *env);

typedef struct _vira_task {
  vira_task_fn fn;
  void *env;
  struct _vira_task *next_actor_msg; // Intrusive MPSC link
  volatile int32_t done;             // 0 = pending, 1 = completed on worker pool
} vira_task;
#endif

typedef struct _vira_actor_ctx vira_actor_ctx;
static __thread vira_actor_ctx *_vira_tls_actor;
static inline bool _vira_in_active_actor(void);
static inline void vira_scheduler_ensure_init(void);
static inline void vira_scheduler_submit_async(vira_task *task);
static inline void vira_scheduler_await(vira_task *task);
static inline int32_t vira_scheduler_num_workers(void);

typedef void (*_vira_team_fn)(void *ctx, int32_t tid, int32_t n_threads);
static inline void _vira_team_run(_vira_team_fn fn, void *ctx, int32_t max_threads);
static inline void _vira_team_shutdown(void);
static inline void vira_scheduler_shutdown(void);
static inline int32_t vira_sched_get_threads(void);
static inline void vira_sched_set_threads(int32_t n);
static inline void _vira_llm_shutdown(void);

#if defined(__AVX2__)
static inline __m256 _vira_avx2_expf(__m256 x) {
  const __m256 exp_hi = _mm256_set1_ps(88.0f);
  const __m256 exp_lo = _mm256_set1_ps(-87.3f);
  const __m256 valid = _mm256_cmp_ps(x, exp_lo, _CMP_GT_OS);
  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);

  const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
  __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2ef),
                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
  const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
#if defined(__FMA__)
  x = _mm256_fnmadd_ps(fx, ln2_hi, x);
  x = _mm256_fnmadd_ps(fx, ln2_lo, x);
#else
  x = _mm256_sub_ps(x, _mm256_mul_ps(fx, ln2_hi));
  x = _mm256_sub_ps(x, _mm256_mul_ps(fx, ln2_lo));
#endif

  const __m256 z = _mm256_mul_ps(x, x);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
#if defined(__FMA__)
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_fmadd_ps(y, z, x);
#else
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_add_ps(_mm256_mul_ps(y, z), x);
#endif
  y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));

  __m256i emm0 = _mm256_cvtps_epi32(fx);
  emm0 = _mm256_slli_epi32(_mm256_add_epi32(emm0, _mm256_set1_epi32(127)), 23);
  __m256 pow2n = _mm256_castsi256_ps(emm0);
  return _mm256_and_ps(_mm256_mul_ps(y, pow2n), valid);
}
#endif

/**
 * Returns the human-readable identifier of the active CPU SIMD instruction set.
 * @return C-string ("x86_64 AVX2", "x86_64 SSE2", "ARM NEON", or "Scalar Fallback").
 * Usage: Telemetry diagnostics and hardware capability reporting.
 */
static inline const char *vira_simd_backend_name(void) {
#if defined(__AVX2__)
  return "x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle, Multi-Threaded)";
#elif defined(__ARM_NEON) || defined(__aarch64__)
  return "ARM64 NEON (128-bit 2x-Unrolled / 8x f32 per step, Multi-Threaded)";
#elif defined(__SSSE3__)
  return "x86_64 SSSE3 + SSE2 (128-bit Integer/Float SIMD, Multi-Threaded)";
#elif defined(__SSE2__)
  return "x86_64 SSE2 (128-bit / 4x f32 per cycle, Multi-Threaded)";
#else
  return "Scalar C99 Fallback (Multi-Threaded)";
#endif
}

typedef struct {
  vira_f32 *data;  // 64-byte cache-line aligned pointer
  void *raw_alloc; // underlying pointer for free()
  int32_t rows;    // flattened outer product (1 for 1D) for 2D compatibility
  int32_t cols;    // trailing dimension dims[n_dims - 1] for 2D compatibility
  int32_t dims[4];
  int32_t strides[4];
  int32_t n_dims;
  bool is_arena;
  bool is_view;
  bool is_vmem;
} vira_tensor_f32;
static inline void vira_tensor_f32_free(vira_tensor_f32 *t);

/**
 * Internal helper to initialize shape dimensions and stride metadata for an N-D tensor.
 * @param t Pointer to tensor struct.
 * @param n_dims Number of dimensions (1 to 4).
 * @param dims Array of dimension sizes.
 * Usage: Common shape initialization helper across tensor allocation functions.
 */
static inline void _vira_tensor_init_shape(vira_tensor_f32 *t, int32_t n_dims,
                                           const int32_t *dims) {
  if (n_dims < 1 || n_dims > 4 || dims == NULL) {
    fprintf(stderr, "vira tensor error: unsupported rank %d (expected 1..4)\n",
            n_dims);
    exit(1);
  }
  t->n_dims = n_dims;
  for (int32_t d = 0; d < 4; d++) {
    t->dims[d] = 0;
    t->strides[d] = 0;
  }
  for (int32_t d = 0; d < n_dims; d++) {
    if (dims[d] <= 0) {
      fprintf(stderr,
              "vira tensor error: invalid dimension %d at axis %d (rank %d)\n",
              dims[d], d, n_dims);
      exit(1);
    }
    t->dims[d] = dims[d];
  }
  int32_t st = 1;
  for (int32_t d = n_dims - 1; d >= 0; d--) {
    t->strides[d] = st;
    st *= dims[d];
  }
  if (n_dims == 1) {
    t->rows = 1;
    t->cols = dims[0];
  } else {
    int32_t outer = 1;
    for (int32_t d = 0; d < n_dims - 1; d++) {
      outer *= dims[d];
    }
    t->rows = outer;
    t->cols = dims[n_dims - 1];
  }
}

/**
 * Normalizes tensor dimensions, computing elements, rows, and columns for 2D/3D/4D matrix operations.
 * @param t Pointer to tensor to normalize.
 * Usage: Invoked before GEMM or linear operations to prepare batched dimension metadata.
 */
static inline void _vira_tensor_norm_shape(vira_tensor_f32 *t) {
  if (t->n_dims <= 0 && t->rows > 0 && t->cols > 0) {
    int32_t d[2] = {t->rows, t->cols};
    _vira_tensor_init_shape(t, 2, d);
  }
}

/**
 * Returns the total number of scalar float elements contained in a tensor.
 * @param t Tensor instance.
 * @return Total element count.
 * Usage: Element count query (`tensor.numel()`).
 */
static inline int32_t vira_tensor_numel(vira_tensor_f32 t) {
  _vira_tensor_norm_shape(&t);
  return t.rows * t.cols;
}

/**
 * Returns the size of a tensor along the specified axis dimension.
 * @param t Tensor instance.
 * @param axis Dimension axis index (0 to 3).
 * @return Dimension size in elements.
 * Usage: Inspecting individual tensor dimensions (`tensor.dim(0)`).
 */
static inline int32_t vira_tensor_dim(vira_tensor_f32 t, int32_t axis) {
  _vira_tensor_norm_shape(&t);
  int32_t ax = axis < 0 ? (t.n_dims + axis) : axis;
  if (ax < 0 || ax >= t.n_dims) {
    fprintf(stderr,
            "vira tensor error: dim axis %d out of bounds for %dD tensor\n",
            axis, t.n_dims);
    exit(1);
  }
  return t.dims[ax];
}

/**
 * Returns the element stride of a tensor along the specified axis.
 * @param t Tensor instance.
 * @param axis Dimension axis index.
 * @return Stride multiplier.
 * Usage: Stride calculation for tensor views and indexing.
 */
static inline int32_t vira_tensor_stride(vira_tensor_f32 t, int32_t axis) {
  _vira_tensor_norm_shape(&t);
  int32_t ax = axis < 0 ? (t.n_dims + axis) : axis;
  if (ax < 0 || ax >= t.n_dims) {
    fprintf(stderr,
            "vira tensor error: stride axis %d out of bounds for %dD tensor\n",
            axis, t.n_dims);
    exit(1);
  }
  return t.strides[ax];
}

/**
 * Allocates an N-dimensional float tensor on the heap with 64-byte aligned SIMD storage.
 * @param n_dims Number of dimensions (1 to 4).
 * @param dims Array of dimension sizes.
 * @return Freshly allocated heap tensor.
 * Usage: General N-D tensor creation (`Tensor.new([B, S, H, D])`).
 */
static inline vira_tensor_f32 vira_tensor_f32_alloc_nd(int32_t n_dims,
                                                       const int32_t *dims) {
  vira_tensor_f32 t = {0};
  _vira_tensor_init_shape(&t, n_dims, dims);
  size_t bytes = (size_t)t.rows * (size_t)t.cols * sizeof(vira_f32);
  void *raw = malloc(bytes + 64);
  if (!raw) {
    fprintf(stderr, "vira panic: tensor allocation failed (%zu bytes)\n",
            bytes);
    exit(1);
  }
  uintptr_t aligned = ((uintptr_t)raw + 63) & ~(uintptr_t)63;
  t.data = (vira_f32 *)aligned;
  t.raw_alloc = raw;
  t.is_arena = false;
  t.is_view = false;
  memset(t.data, 0, bytes);
  _vira_record_alloc();
  return t;
}

/**
 * Allocates an N-dimensional float tensor inside a bump-pointer memory arena.
 * @param a Pointer to arena.
 * @param n_dims Number of dimensions (1 to 4).
 * @param dims Array of dimension sizes.
 * @return Arena-allocated tensor with zero individual deallocation overhead.
 * Usage: Fast zero-leak intermediate activations in Transformer inference loops.
 */
static inline vira_tensor_f32
vira_tensor_f32_arena_alloc_nd(vira_arena *a, int32_t n_dims,
                               const int32_t *dims) {
  vira_tensor_f32 t = {0};
  _vira_tensor_init_shape(&t, n_dims, dims);
  size_t bytes = (size_t)t.rows * (size_t)t.cols * sizeof(vira_f32);
  t.data = (vira_f32 *)vira_arena_alloc(a, bytes);
  t.raw_alloc = NULL;
  t.is_arena = true;
  t.is_view = false;
  memset(t.data, 0, bytes);
  return t;
}

/**
 * Uninitialized N-D heap tensor allocation for GPU output wrappers where kernel populates elements.
 */
static inline vira_tensor_f32 _vira_tensor_f32_alloc_uninit_nd(int32_t n_dims,
                                                               const int32_t *dims) {
  vira_tensor_f32 t = {0};
  _vira_tensor_init_shape(&t, n_dims, dims);
  size_t bytes = (size_t)t.rows * (size_t)t.cols * sizeof(vira_f32);
  void *raw = malloc(bytes + 64);
  if (!raw) {
    fprintf(stderr, "vira panic: tensor allocation failed (%zu bytes)\n",
            bytes);
    exit(1);
  }
  uintptr_t aligned = ((uintptr_t)raw + 63) & ~(uintptr_t)63;
  t.data = (vira_f32 *)aligned;
  t.raw_alloc = raw;
  t.is_arena = false;
  t.is_view = false;
  t.is_vmem = false;
  _vira_record_alloc();
  return t;
}

/**
 * Reserves virtual address space without committing physical memory or pagefile
 * for deferred host-quantized GGUF tensors; pages are committed on demand if
 * host FP32 materialization is ever triggered.
 */
static inline vira_tensor_f32
_vira_tensor_f32_alloc_reserved_nd(int32_t n_dims, const int32_t *dims) {
#ifdef _WIN32
  vira_tensor_f32 t = {0};
  _vira_tensor_init_shape(&t, n_dims, dims);
  size_t bytes = (size_t)t.rows * (size_t)t.cols * sizeof(vira_f32);
  if (bytes == 0)
    bytes = sizeof(vira_f32);
  void *raw = VirtualAlloc(NULL, bytes, MEM_RESERVE, PAGE_READWRITE);
  if (!raw) {
    fprintf(stderr, "vira panic: tensor virtual reserve failed (%zu bytes)\n",
            bytes);
    exit(1);
  }
  t.data = (vira_f32 *)raw;
  t.raw_alloc = raw;
  t.is_arena = false;
  t.is_view = false;
  t.is_vmem = true;
  _vira_record_alloc();
  return t;
#else
  return _vira_tensor_f32_alloc_uninit_nd(n_dims, dims);
#endif
}

/**
 * Uninitialized N-D arena tensor allocation for GPU output wrappers where kernel populates elements.
 */
static inline vira_tensor_f32
_vira_tensor_f32_arena_alloc_uninit_nd(vira_arena *a, int32_t n_dims,
                                       const int32_t *dims) {
  vira_tensor_f32 t = {0};
  _vira_tensor_init_shape(&t, n_dims, dims);
  size_t bytes = (size_t)t.rows * (size_t)t.cols * sizeof(vira_f32);
  t.data = (vira_f32 *)vira_arena_alloc(a, bytes);
  t.raw_alloc = NULL;
  t.is_arena = true;
  t.is_view = false;
  return t;
}

/**
 * Allocates a 2D float tensor (matrix) on the heap with 64-byte aligned storage.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Freshly allocated 2D tensor.
 * Usage: 2D matrix allocation (`Tensor.new(rows, cols)`).
 */
static inline vira_tensor_f32 vira_tensor_f32_alloc(int32_t rows,
                                                    int32_t cols) {
  if (rows <= 0 || cols <= 0) {
    fprintf(stderr, "vira tensor error: invalid dimensions (%d x %d)\n", rows,
            cols);
    exit(1);
  }
  int32_t dims[2] = {rows, cols};
  return vira_tensor_f32_alloc_nd(2, dims);
}

/**
 * Allocates a 2D float tensor (matrix) inside a bump-pointer memory arena.
 * @param a Pointer to arena.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Arena-allocated 2D matrix.
 * Usage: High-throughput matrix creation in arena blocks.
 */
static inline vira_tensor_f32
vira_tensor_f32_arena_alloc(vira_arena *a, int32_t rows, int32_t cols) {
  if (rows <= 0 || cols <= 0) {
    fprintf(stderr, "vira tensor error: invalid dimensions (%d x %d)\n", rows,
            cols);
    exit(1);
  }
  int32_t dims[2] = {rows, cols};
  return vira_tensor_f32_arena_alloc_nd(a, 2, dims);
}

/**
 * Fills a contiguous float array with a constant value using SIMD vector instructions.
 * @param data Pointer to float buffer.
 * @param total Total number of elements.
 * @param val Value to write into every element.
 * Usage: Internal vectorized fill helper for tensor initialization.
 */
static inline void _vira_tensor_fill_data(vira_f32 *data, int32_t total,
                                          vira_f32 val) {
  int32_t i = 0;
#if defined(__AVX2__)
  __m256 vval = _mm256_set1_ps(val);
  for (; i + 8 <= total; i += 8) {
    _mm256_storeu_ps(&data[i], vval);
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t vval = vdupq_n_f32(val);
  for (; i + 4 <= total; i += 4) {
    vst1q_f32(&data[i], vval);
  }
#endif
  for (; i < total; i++) {
    data[i] = val;
  }
}

/**
 * Creates a 2D float tensor initialized with a uniform constant value.
 * @param opt_arena Optional arena (or NULL for heap).
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param val Fill value.
 * @return Initialized 2D tensor.
 * Usage: Creating constant or zero-initialized matrices (`Tensor.fill(rows, cols, 0.0)`).
 */
static inline vira_tensor_f32 vira_tensor_f32_fill(vira_arena *opt_arena,
                                                   int32_t rows, int32_t cols,
                                                   vira_f32 val) {
  vira_tensor_f32 t = opt_arena
                          ? vira_tensor_f32_arena_alloc(opt_arena, rows, cols)
                          : vira_tensor_f32_alloc(rows, cols);
  _vira_tensor_fill_data(t.data, t.rows * t.cols, val);
  return t;
}

/**
 * Creates a 3D float tensor initialized with a uniform constant value.
 * @param opt_arena Optional arena (or NULL for heap).
 * @param d0 Dimension 0.
 * @param d1 Dimension 1.
 * @param d2 Dimension 2.
 * @param val Fill value.
 * @return Initialized 3D tensor.
 * Usage: Creating constant 3D tensors.
 */
static inline vira_tensor_f32 vira_tensor_f32_fill_3d(vira_arena *opt_arena,
                                                      int32_t d0, int32_t d1,
                                                      int32_t d2,
                                                      vira_f32 val) {
  int32_t dims[3] = {d0, d1, d2};
  vira_tensor_f32 t =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 3, dims)
                : vira_tensor_f32_alloc_nd(3, dims);
  _vira_tensor_fill_data(t.data, t.rows * t.cols, val);
  return t;
}

/**
 * Creates a 4D float tensor initialized with a uniform constant value.
 * @param opt_arena Optional arena (or NULL for heap).
 * @param d0 Dimension 0.
 * @param d1 Dimension 1.
 * @param d2 Dimension 2.
 * @param d3 Dimension 3.
 * @param val Fill value.
 * @return Initialized 4D tensor.
 * Usage: Creating constant 4D tensors (e.g. multi-head attention caches).
 */
static inline vira_tensor_f32 vira_tensor_f32_fill_4d(vira_arena *opt_arena,
                                                      int32_t d0, int32_t d1,
                                                      int32_t d2, int32_t d3,
                                                      vira_f32 val) {
  int32_t dims[4] = {d0, d1, d2, d3};
  vira_tensor_f32 t =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 4, dims)
                : vira_tensor_f32_alloc_nd(4, dims);
  _vira_tensor_fill_data(t.data, t.rows * t.cols, val);
  return t;
}

// Low-level C FFI zero-copy view (is_view = true; caller ensures `t` outlives view)
/**
 * Creates a zero-copy reshaped view sharing the underlying data buffer of an existing tensor.
 * @param t Source tensor.
 * @param n_dims Target number of dimensions.
 * @param dims Array of target dimension sizes.
 * @return Non-owning reshaped view tensor.
 * Usage: Zero-cost tensor reshaping when total element count is preserved (`tensor.view(...)`).
 */
static inline vira_tensor_f32 vira_tensor_view_nd(vira_tensor_f32 t,
                                                  int32_t n_dims,
                                                  const int32_t *dims) {
  _vira_tensor_norm_shape(&t);
  vira_tensor_f32 v = {0};
  _vira_tensor_init_shape(&v, n_dims, dims);
  if (v.rows * v.cols != t.rows * t.cols) {
    fprintf(stderr,
            "vira tensor error: reshape element count mismatch (%d -> %d)\n",
            t.rows * t.cols, v.rows * v.cols);
    exit(1);
  }
  v.data = t.data;
  v.raw_alloc = NULL;
  v.is_arena = t.is_arena;
  v.is_view = true;
  return v;
}

static inline void _vira_gpu_evict_ptr(const vira_f32 *host_ptr);
static inline bool _vira_gpu_is_ptr_dirty(const vira_f32 *ptr);
static inline void _vira_gpu_sync_host(const vira_f32 *ptr);
static inline void _vira_gpu_clone_quant(const vira_f32 *dst_ptr,
                                         const vira_f32 *src_ptr);
static inline void vira_gpu_shutdown(void);
static inline bool _vira_gpu_copy_row(vira_tensor_f32 dst, int32_t dst_row,
                                      vira_tensor_f32 src);
static inline bool _vira_gpu_copy_rows(vira_tensor_f32 dst,
                                       int32_t dst_row_start,
                                       vira_tensor_f32 src);
static inline bool _vira_gpu_slice_rows(vira_arena *opt_arena,
                                        vira_tensor_f32 src,
                                        int32_t start_row, int32_t end_row,
                                        vira_tensor_f32 *out_tensor);
static __thread bool _vira_tls_gpu_fallback = false;

// --- GPU AUTO-DISPATCH FORWARD DECLARATIONS ---
static inline bool vira_gpu_is_available(void);
static inline bool vira_gpu_is_enabled(void);
static inline bool vira_gpu_is_unified_memory(void);
static inline bool vira_gpu_has_tensor_cores(void);
static inline vira_string vira_gpu_device_name(void);
static inline vira_string vira_gpu_backend_name(void);
static inline int64_t vira_gpu_vram_bytes(void);
static inline int64_t vira_gpu_allocated_bytes(void);
static inline int64_t vira_gpu_oom_fallbacks(void);
static inline void vira_gpu_set_backend(vira_string backend);
static inline void vira_gpu_set_device(vira_string device);
static inline vira_tensor_f32 vira_gpu_linear(vira_arena *opt_arena,
                                              vira_tensor_f32 x,
                                              vira_tensor_f32 w);
static inline vira_tensor_f32 vira_gpu_matmul(vira_arena *opt_arena,
                                              vira_tensor_f32 a,
                                              vira_tensor_f32 b);
static inline vira_tensor_f32 vira_gpu_rmsnorm(vira_arena *opt_arena,
                                               vira_tensor_f32 x,
                                               vira_tensor_f32 weight,
                                               float eps);
static inline vira_tensor_f32 vira_gpu_rope(vira_arena *opt_arena,
                                            vira_tensor_f32 q, int64_t pos,
                                            float freq_base);
static inline vira_tensor_f32 vira_gpu_rope_norm(vira_arena *opt_arena,
                                                 vira_tensor_f32 q, int64_t pos,
                                                 float freq_base);
static inline vira_tensor_f32 vira_gpu_rope_seq(vira_arena *opt_arena,
                                                vira_tensor_f32 q,
                                                int64_t start_pos,
                                                int32_t n_heads,
                                                float freq_base);
static inline vira_tensor_f32 vira_gpu_rope_seq_norm(vira_arena *opt_arena,
                                                     vira_tensor_f32 q,
                                                     int64_t start_pos,
                                                     int32_t n_heads,
                                                     float freq_base);
static inline vira_tensor_f32 vira_gpu_softmax(vira_arena *opt_arena,
                                               vira_tensor_f32 logits);
static inline vira_tensor_f32 vira_gpu_silu(vira_arena *opt_arena,
                                            vira_tensor_f32 x);
static inline vira_tensor_f32 vira_gpu_add(vira_arena *opt_arena,
                                           vira_tensor_f32 a,
                                           vira_tensor_f32 b);
static inline vira_tensor_f32 vira_gpu_elemwise_mul(vira_arena *opt_arena,
                                                    vira_tensor_f32 a,
                                                    vira_tensor_f32 b);
static inline vira_tensor_f32 vira_gpu_scale(vira_arena *opt_arena,
                                             vira_tensor_f32 t,
                                             vira_f32 scale);
static inline vira_tensor_f32 vira_gpu_gqa_attention(
    vira_arena *opt_arena, vira_tensor_f32 q_heads, vira_tensor_f32 k_cache,
    vira_tensor_f32 v_cache, int64_t kv_pos, float scale);
static inline vira_tensor_f32 vira_gpu_gqa_prefill(
    vira_arena *opt_arena, vira_tensor_f32 q_seq, vira_tensor_f32 k_cache,
    vira_tensor_f32 v_cache, int64_t start_pos, int32_t seq_len,
    int32_t n_heads, float scale);
static inline int64_t _vira_gpu_argmax_f32(vira_tensor_f32 logits);
static inline void _vira_gpu_register_quant(const vira_f32 *host_ptr,
                                            uint32_t gguf_dtype,
                                            const void *raw_blocks,
                                            size_t raw_bytes, size_t elems);
static inline vira_tensor_f32 vira_gpu_add_rmsnorm(vira_arena *opt_arena,
                                                  vira_tensor_f32 mut_res,
                                                  vira_tensor_f32 delta,
                                                  vira_tensor_f32 weight,
                                                  float eps);
static inline vira_tensor_f32 vira_gpu_silu_mul(vira_arena *opt_arena,
                                               vira_tensor_f32 gate,
                                               vira_tensor_f32 up);
static inline vira_tensor_f32 vira_gpu_rope_kv_write(
    vira_arena *opt_arena, vira_tensor_f32 q_lin, vira_tensor_f32 bq,
    vira_tensor_f32 k_lin, vira_tensor_f32 bk, vira_tensor_f32 v_lin,
    vira_tensor_f32 bv, vira_tensor_f32 k_cache, vira_tensor_f32 v_cache,
    int64_t pos, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
    float freq_base);
static inline vira_tensor_f32 vira_gpu_emb_gather(vira_arena *opt_arena,
                                                 vira_tensor_f32 table,
                                                 int64_t token_id);
static inline int64_t vira_gpu_graph_try_replay(
    int32_t token_id, int32_t kv_pos, int32_t seq_len, vira_tensor_f32 tok_emb,
    vira_tensor_f32 k_cache_sample, vira_tensor_f32 v_cache_sample);
static inline void vira_gpu_graph_begin_capture(
    int32_t token_id, int32_t kv_pos, int32_t seq_len, vira_tensor_f32 tok_emb,
    vira_tensor_f32 k_cache_sample, vira_tensor_f32 v_cache_sample);
static inline int64_t vira_gpu_graph_end_capture_argmax(vira_tensor_f32 logits);
static inline int64_t vira_gpu_graph_captures(void);
static inline int64_t vira_gpu_graph_replays(void);

#endif // VIRA_RT_TENSOR_CORE_H
