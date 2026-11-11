#ifndef VIRA_RT_LLM_H
#define VIRA_RT_LLM_H

// ============================================================================
// 17. LLM TRANSFORMER OPERATORS (RMSNORM, ROPE, SILU, EMBEDDING, SAMPLING)
// ============================================================================

/**
 * Computes Root Mean Square Layer Normalization: `y = (x / sqrt(mean(x^2) + eps)) * weight`.
 * @param x Input activation tensor.
 * @param weight Normalization scale weights.
 * @param eps Epsilon value for numerical stability.
 * @return Normalized tensor.
 * Usage: Pre- and post-attention layer normalization in modern LLMs (Llama, Qwen, Mistral).
 */
static inline vira_tensor_f32
vira_tensor_rmsnorm(vira_arena *opt_arena, vira_tensor_f32 x,
                    vira_tensor_f32 weight, float eps) {
  _vira_tensor_norm_shape(&x);
  _vira_tensor_norm_shape(&weight);
  if (!_vira_tls_gpu_fallback &&
      ((int64_t)x.rows * (int64_t)x.cols >= 16384LL ||
       _vira_gpu_is_ptr_dirty(x.data) || _vira_gpu_is_ptr_dirty(weight.data)) &&
      vira_gpu_is_available() && vira_gpu_is_enabled()) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 gpu_out = vira_gpu_rmsnorm(opt_arena, x, weight, eps);
    _vira_tls_gpu_fallback = false;
    return gpu_out;
  }
  _vira_tensor_sync_host_f32(x.data);
  _vira_tensor_sync_host_f32(weight.data);
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, x.n_dims, x.dims)
                : vira_tensor_f32_alloc_nd(x.n_dims, x.dims);
  for (int32_t r = 0; r < x.rows; ++r) {
    float *x_row = &x.data[r * x.cols];
    float *out_row = &out.data[r * x.cols];
    float sum_sq = 0.0f;

#if defined(__AVX2__)
    __m256 v_sum_sq = _mm256_setzero_ps();
    int32_t c = 0;
    for (; c + 7 < x.cols; c += 8) {
      __m256 vx = _mm256_loadu_ps(&x_row[c]);
      v_sum_sq = _mm256_add_ps(v_sum_sq, _mm256_mul_ps(vx, vx));
    }
    __m128 lo = _mm256_castps256_ps128(v_sum_sq);
    __m128 hi = _mm256_extractf128_ps(v_sum_sq, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    sum_sq += _mm_cvtss_f32(s);
    for (; c < x.cols; ++c) {
      sum_sq += x_row[c] * x_row[c];
    }
#else
    for (int32_t c = 0; c < x.cols; ++c) {
      sum_sq += x_row[c] * x_row[c];
    }
#endif
    float mean_sq = sum_sq / (float)x.cols;
    float inv_rms = 1.0f / sqrtf(mean_sq + eps);

#if defined(__AVX2__)
    __m256 v_inv = _mm256_set1_ps(inv_rms);
    c = 0;
    for (; c + 7 < x.cols; c += 8) {
      __m256 vx = _mm256_loadu_ps(&x_row[c]);
      __m256 vw = _mm256_loadu_ps(&weight.data[c]);
      __m256 res = _mm256_mul_ps(_mm256_mul_ps(vx, v_inv), vw);
      _mm256_storeu_ps(&out_row[c], res);
    }
    for (; c < x.cols; ++c) {
      out_row[c] = x_row[c] * inv_rms * weight.data[c];
    }
#else
    for (int32_t c = 0; c < x.cols; ++c) {
      out_row[c] = x_row[c] * inv_rms * weight.data[c];
    }
#endif
  }
  return out;
}

/**
 * Applies Rotary Position Embeddings (RoPE) to attention query or key vectors.
 * @param q Query or key tensor.
 * @param pos Sequence token position offset.
 * @param freq_base Frequency base constant (e.g. 10000.0f).
 * @return Rotated tensor with position encodings applied.
 * Usage: Position encoding in modern Transformer self-attention blocks.
 */
static inline vira_tensor_f32 vira_tensor_rope(vira_arena *opt_arena,
                                               vira_tensor_f32 q, int64_t pos,
                                               float freq_base) {
  _vira_tensor_norm_shape(&q);
  if (!_vira_tls_gpu_fallback && (q.cols & 1) == 0 &&
      ((int64_t)q.rows * (int64_t)q.cols >= 16384LL ||
       _vira_gpu_is_ptr_dirty(q.data)) &&
      vira_gpu_is_available() && vira_gpu_is_enabled()) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 gpu_out = vira_gpu_rope(opt_arena, q, pos, freq_base);
    _vira_tls_gpu_fallback = false;
    return gpu_out;
  }
  _vira_tensor_sync_host_f32(q.data);
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, q.n_dims, q.dims)
                : vira_tensor_f32_alloc_nd(q.n_dims, q.dims);
  int32_t half_dim = q.cols / 2;
  float cos_stack[256];
  float sin_stack[256];
  float *cos_tab =
      (half_dim <= 256) ? cos_stack : (float *)malloc(half_dim * sizeof(float));
  float *sin_tab =
      (half_dim <= 256) ? sin_stack : (float *)malloc(half_dim * sizeof(float));

  for (int32_t i = 0; i < half_dim; ++i) {
    float exponent = -(2.0f * (float)i) / (float)q.cols;
    float freq = powf(freq_base, exponent);
    float angle = (float)pos * freq;
    cos_tab[i] = cosf(angle);
    sin_tab[i] = sinf(angle);
  }

  for (int32_t r = 0; r < q.rows; ++r) {
    const float *q_row = &q.data[r * q.cols];
    float *out_row = &out.data[r * q.cols];
    for (int32_t i = 0; i < half_dim; ++i) {
      float cos_a = cos_tab[i];
      float sin_a = sin_tab[i];
      float q0 = q_row[i];
      float q1 = q_row[i + half_dim];
      out_row[i] = q0 * cos_a - q1 * sin_a;
      out_row[i + half_dim] = q0 * sin_a + q1 * cos_a;
    }
  }

  if (half_dim > 256) {
    free(cos_tab);
    free(sin_tab);
  }
  return out;
}

/**
 * Applies classic/interleaved Rotary Position Embeddings (RoPE, `(2*i, 2*i+1)` pairs)
 * to attention query or key vectors (standard LLaMA / DeepSeek GGUF layout).
 * @param q Query or key tensor.
 * @param pos Sequence token position offset.
 * @param freq_base Frequency base constant (e.g. 10000.0f).
 * @return Rotated tensor with position encodings applied.
 * Usage: Position encoding for LLaMA-style GGUF models (`tensor_rope_norm(q, pos, freq_base)`).
 */
static inline vira_tensor_f32 vira_tensor_rope_norm(vira_arena *opt_arena,
                                                    vira_tensor_f32 q, int64_t pos,
                                                    float freq_base) {
  _vira_tensor_norm_shape(&q);
  if (!_vira_tls_gpu_fallback && (q.cols & 1) == 0 &&
      ((int64_t)q.rows * (int64_t)q.cols >= 16384LL ||
       _vira_gpu_is_ptr_dirty(q.data)) &&
      vira_gpu_is_available() && vira_gpu_is_enabled()) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 gpu_out = vira_gpu_rope_norm(opt_arena, q, pos, freq_base);
    _vira_tls_gpu_fallback = false;
    return gpu_out;
  }
  _vira_tensor_sync_host_f32(q.data);
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, q.n_dims, q.dims)
                : vira_tensor_f32_alloc_nd(q.n_dims, q.dims);
  int32_t half_dim = q.cols / 2;
  float cos_stack[256];
  float sin_stack[256];
  float *cos_tab =
      (half_dim <= 256) ? cos_stack : (float *)malloc(half_dim * sizeof(float));
  float *sin_tab =
      (half_dim <= 256) ? sin_stack : (float *)malloc(half_dim * sizeof(float));

  for (int32_t i = 0; i < half_dim; ++i) {
    float exponent = -(2.0f * (float)i) / (float)q.cols;
    float freq = powf(freq_base, exponent);
    float angle = (float)pos * freq;
    cos_tab[i] = cosf(angle);
    sin_tab[i] = sinf(angle);
  }

  for (int32_t r = 0; r < q.rows; ++r) {
    const float *q_row = &q.data[r * q.cols];
    float *out_row = &out.data[r * q.cols];
    for (int32_t i = 0; i < half_dim; ++i) {
      float cos_a = cos_tab[i];
      float sin_a = sin_tab[i];
      float q0 = q_row[2 * i];
      float q1 = q_row[2 * i + 1];
      out_row[2 * i] = q0 * cos_a - q1 * sin_a;
      out_row[2 * i + 1] = q0 * sin_a + q1 * cos_a;
    }
  }

  if (half_dim > 256) {
    free(cos_tab);
    free(sin_tab);
  }
  return out;
}

/**
 * Computes the SiLU (Swish) non-linear activation function `f(x) = x * sigmoid(x)` with SIMD.
 * @param x Input tensor.
 * @return Activated tensor.
 * Usage: MLP feed-forward gate activation in Llama/Qwen architectures.
 */
static inline vira_tensor_f32 vira_tensor_silu(vira_arena *opt_arena,
                                               vira_tensor_f32 x) {
  _vira_tensor_norm_shape(&x);
  int32_t total = x.rows * x.cols;
  if (!_vira_tls_gpu_fallback &&
      (total >= 16384 || _vira_gpu_is_ptr_dirty(x.data)) &&
      vira_gpu_is_available() && vira_gpu_is_enabled()) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 gpu_out = vira_gpu_silu(opt_arena, x);
    _vira_tls_gpu_fallback = false;
    return gpu_out;
  }
  _vira_tensor_sync_host_f32(x.data);
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, x.n_dims, x.dims)
                : vira_tensor_f32_alloc_nd(x.n_dims, x.dims);
  for (int32_t i = 0; i < total; ++i) {
    float v = x.data[i];
    out.data[i] = v / (1.0f + expf(-v));
  }
  return out;
}

/**
 * SIMD vector kernel computing elementwise multiplication of arrays `pa` and `pb` into `dst`.
 * @param dst Destination buffer.
 * @param pa First input array.
 * @param pb Second input array.
 * @param count Number of elements.
 * Usage: Internal vectorized helper for Hadamard product.
 */
static inline void _vira_vec_mul_f32(vira_f32 *dst, const vira_f32 *pa,
                                     const vira_f32 *pb, int32_t count) {
  int32_t i = 0;
#if defined(__AVX2__)
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(&pa[i]);
    __m256 vb = _mm256_loadu_ps(&pb[i]);
    _mm256_storeu_ps(&dst[i], _mm256_mul_ps(va, vb));
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  for (; i + 4 <= count; i += 4) {
    float32x4_t va = vld1q_f32(&pa[i]);
    float32x4_t vb = vld1q_f32(&pb[i]);
    vst1q_f32(&dst[i], vmulq_f32(va, vb));
  }
#elif defined(__SSE2__)
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(&pa[i]);
    __m128 vb = _mm_loadu_ps(&pb[i]);
    _mm_storeu_ps(&dst[i], _mm_mul_ps(va, vb));
  }
#endif
  for (; i < count; i++) {
    dst[i] = pa[i] * pb[i];
  }
}

/**
 * Computes elementwise Hadamard multiplication of two tensors `a * b`.
 * @param a First tensor.
 * @param b Second tensor.
 * @return Elementwise product tensor.
 * Usage: Fused gate-up projection multiplication in SwiGLU MLP blocks.
 */
static inline vira_tensor_f32
vira_tensor_elemwise_mul(vira_arena *opt_arena, vira_tensor_f32 a,
                         vira_tensor_f32 b) {
  _vira_tensor_norm_shape(&a);
  _vira_tensor_norm_shape(&b);
  int64_t elems_a = (int64_t)a.rows * (int64_t)a.cols;
  int64_t elems_b = (int64_t)b.rows * (int64_t)b.cols;
  int64_t max_elems = (elems_a > elems_b) ? elems_a : elems_b;
  if (!_vira_tls_gpu_fallback &&
      (max_elems >= 16384LL || _vira_gpu_is_ptr_dirty(a.data) ||
       _vira_gpu_is_ptr_dirty(b.data)) &&
      vira_gpu_is_available() && vira_gpu_is_enabled()) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 gpu_out = vira_gpu_elemwise_mul(opt_arena, a, b);
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
    _vira_vec_mul_f32(out.data, a.data, b.data, a.rows * a.cols);
    return out;
  }
  if (b.rows == 1 && b.cols == a.cols) {
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, a.n_dims, a.dims)
                  : vira_tensor_f32_alloc_nd(a.n_dims, a.dims);
    for (int32_t r = 0; r < a.rows; r++) {
      _vira_vec_mul_f32(&out.data[(size_t)r * (size_t)a.cols],
                        &a.data[(size_t)r * (size_t)a.cols], b.data, a.cols);
    }
    return out;
  }
  if (a.rows == 1 && a.cols == b.cols) {
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, b.n_dims, b.dims)
                  : vira_tensor_f32_alloc_nd(b.n_dims, b.dims);
    for (int32_t r = 0; r < b.rows; r++) {
      _vira_vec_mul_f32(&out.data[(size_t)r * (size_t)b.cols], a.data,
                        &b.data[(size_t)r * (size_t)b.cols], b.cols);
    }
    return out;
  }
  fprintf(stderr,
          "vira tensor error: elemwise_mul shape mismatch (%dx%d) * (%dx%d)\n",
          a.rows, a.cols, b.rows, b.cols);
  exit(1);
}

/**
 * Computes numerically stable Softmax probability distribution across rows of a tensor.
 * @param logits Input logits tensor.
 * @return Softmax probability distribution tensor summing to 1.0 along the trailing dimension.
 * Usage: Attention score normalization and next-token generation probability calculation.
 */
static inline vira_tensor_f32 vira_tensor_softmax(vira_arena *opt_arena,
                                                  vira_tensor_f32 logits) {
  _vira_tensor_norm_shape(&logits);
  if (!_vira_tls_gpu_fallback &&
      ((int64_t)logits.rows * (int64_t)logits.cols >= 16384LL ||
       _vira_gpu_is_ptr_dirty(logits.data)) &&
      vira_gpu_is_available() && vira_gpu_is_enabled()) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 gpu_out = vira_gpu_softmax(opt_arena, logits);
    _vira_tls_gpu_fallback = false;
    return gpu_out;
  }
  _vira_tensor_sync_host_f32(logits.data);
  vira_tensor_f32 out =
      opt_arena
          ? vira_tensor_f32_arena_alloc_nd(opt_arena, logits.n_dims, logits.dims)
          : vira_tensor_f32_alloc_nd(logits.n_dims, logits.dims);
  for (int32_t r = 0; r < logits.rows; ++r) {
    float *in_row = &logits.data[r * logits.cols];
    float *out_row = &out.data[r * logits.cols];

    float max_val = in_row[0];
    for (int32_t c = 1; c < logits.cols; ++c) {
      if (in_row[c] > max_val)
        max_val = in_row[c];
    }

    float sum = 0.0f;
    for (int32_t c = 0; c < logits.cols; ++c) {
      out_row[c] = expf(in_row[c] - max_val);
      sum += out_row[c];
    }

    float inv_sum = sum > 0.0f ? (1.0f / sum) : 1.0f;
    for (int32_t c = 0; c < logits.cols; ++c) {
      out_row[c] *= inv_sum;
    }
  }
  return out;
}

/**
 * Samples a next-token ID from a logit distribution using temperature scaling and probabilistic selection.
 * @param logits 1D tensor of unnormalized vocabulary logits.
 * @param temperature Temperature scaling factor (e.g. 0.7f).
 * @return Sampled token vocabulary ID.
 * Usage: Autoregressive text generation token sampling (`sample_token(logits, 0.7)`).
 */
static inline int64_t vira_sample_token(vira_tensor_f32 logits,
                                        float temperature) {
  _vira_tensor_norm_shape(&logits);
  int32_t total = logits.rows * logits.cols;
  if (total == 0)
    return 0;

  if (temperature <= 0.05f) {
    int64_t gpu_idx = _vira_gpu_argmax_f32(logits);
    if (gpu_idx >= 0) {
      return gpu_idx;
    }
    _vira_tensor_sync_host_f32(logits.data);
    int64_t best_idx = 0;
    float best_val = logits.data[0];
    for (int32_t i = 1; i < total; ++i) {
      if (logits.data[i] > best_val) {
        best_val = logits.data[i];
        best_idx = (int64_t)i;
      }
    }
    return best_idx;
  }

  vira_tensor_f32 scaled = vira_tensor_scale(NULL, logits, 1.0f / temperature);
  vira_tensor_f32 probs = vira_tensor_softmax(NULL, scaled);
  vira_tensor_f32_free(&scaled);
  _vira_tensor_sync_host_f32(probs.data);

  float r = (float)rand() / (float)RAND_MAX;
  float cdf = 0.0f;
  int64_t chosen = total - 1;
  for (int32_t i = 0; i < total; ++i) {
    cdf += probs.data[i];
    if (r <= cdf) {
      chosen = (int64_t)i;
      break;
    }
  }
  vira_tensor_f32_free(&probs);
  return chosen;
}

/**
 * Retrieves a token embedding row vector from an embedding weight table matrix.
 * @param opt_arena Optional memory arena for output allocation.
 * @param table Embedding table tensor of shape (vocab_size, hidden_dim).
 * @param token_id Integer token vocabulary identifier.
 * @return 1D row tensor containing token embedding.
 * Usage: Initial token embedding lookup in language model inference.
 */
static inline vira_tensor_f32 vira_tensor_embedding(vira_arena *opt_arena,
                                                    vira_tensor_f32 table,
                                                    int64_t token_id) {
  _vira_tensor_norm_shape(&table);
  if (token_id < 0 || token_id >= table.rows) {
    fprintf(stderr,
            "vira tensor error: embedding index %lld out of range [0, %d)\n",
            (long long)token_id, table.rows);
    exit(1);
  }
  if (!_vira_gpu_is_ptr_dirty(table.data)) {
    uint32_t q_dtype = 0;
    const void *q_raw = NULL;
    size_t q_bytes = 0, q_elems = 0;
    if (_vira_rt_quant_try_get(table.data, &q_dtype, &q_raw, &q_bytes,
                               &q_elems)) {
      size_t qk = _vira_gguf_dtype_qk(q_dtype);
      size_t bs = _vira_gguf_dtype_bs(q_dtype);
      size_t cols = (size_t)table.cols;
      if (qk > 0 && bs > 0 && (qk == 1 || (cols % qk) == 0)) {
        size_t row_blocks = (qk == 1) ? cols : (cols / qk);
        size_t start_blk = (size_t)token_id * row_blocks;
        if ((start_blk + row_blocks) * bs <= q_bytes) {
          int32_t odims[2] = {1, table.cols};
          vira_tensor_f32 out =
              opt_arena ? _vira_tensor_f32_arena_alloc_uninit_nd(opt_arena, 2, odims)
                        : _vira_tensor_f32_alloc_uninit_nd(2, odims);
          _vira_rt_dequant_blocks_to_fp32(q_dtype, (const uint8_t *)q_raw,
                                          start_blk, row_blocks, out.data);
          return out;
        }
      }
    }
  }
  _vira_tensor_sync_host_f32(table.data);
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc(opt_arena, 1, table.cols)
                : vira_tensor_f32_alloc(1, table.cols);
  memcpy(out.data, &table.data[token_id * table.cols],
         (size_t)table.cols * sizeof(float));
  return out;
}

/**
 * Retrieves a batch of token embedding row vectors from an embedding weight table matrix.
 * @param opt_arena Optional memory arena for output allocation.
 * @param table Embedding table tensor of shape (vocab_size, hidden_dim).
 * @param token_ids Slice of integer token vocabulary identifiers.
 * @return 2D tensor of shape (token_ids.len, hidden_dim).
 * Usage: Batched prompt token embedding lookup (`tensor_embedding_batch(tok_emb, token_ids)`).
 */
static inline vira_tensor_f32 vira_tensor_embedding_batch(
    vira_arena *opt_arena, vira_tensor_f32 table, vira_slice_int token_ids) {
  _vira_tensor_norm_shape(&table);
  if (token_ids.len <= 0 || token_ids.data == NULL) {
    fprintf(stderr,
            "vira tensor error: embedding_batch token_ids slice cannot be empty (len=%d)\n",
            token_ids.len);
    exit(1);
  }
  for (int32_t i = 0; i < token_ids.len; ++i) {
    int32_t tid = token_ids.data[i];
    if (tid < 0 || tid >= table.rows) {
      fprintf(stderr,
              "vira tensor error: embedding_batch index %d (pos %d) out of range [0, %d)\n",
              tid, i, table.rows);
      exit(1);
    }
  }
  if (!_vira_gpu_is_ptr_dirty(table.data)) {
    uint32_t q_dtype = 0;
    const void *q_raw = NULL;
    size_t q_bytes = 0, q_elems = 0;
    if (_vira_rt_quant_try_get(table.data, &q_dtype, &q_raw, &q_bytes,
                               &q_elems)) {
      size_t qk = _vira_gguf_dtype_qk(q_dtype);
      size_t bs = _vira_gguf_dtype_bs(q_dtype);
      size_t cols = (size_t)table.cols;
      if (qk > 0 && bs > 0 && (qk == 1 || (cols % qk) == 0)) {
        size_t row_blocks = (qk == 1) ? cols : (cols / qk);
        if ((size_t)table.rows * row_blocks * bs <= q_bytes) {
          int32_t odims[2] = {token_ids.len, table.cols};
          vira_tensor_f32 out =
              opt_arena ? _vira_tensor_f32_arena_alloc_uninit_nd(opt_arena, 2, odims)
                        : _vira_tensor_f32_alloc_uninit_nd(2, odims);
          for (int32_t i = 0; i < token_ids.len; ++i) {
            size_t start_blk = (size_t)token_ids.data[i] * row_blocks;
            _vira_rt_dequant_blocks_to_fp32(
                q_dtype, (const uint8_t *)q_raw, start_blk, row_blocks,
                &out.data[(size_t)i * cols]);
          }
          return out;
        }
      }
    }
  }
  _vira_tensor_sync_host_f32(table.data);
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc(opt_arena, token_ids.len, table.cols)
                : vira_tensor_f32_alloc(token_ids.len, table.cols);
  size_t row_bytes = (size_t)table.cols * sizeof(float);
  for (int32_t i = 0; i < token_ids.len; ++i) {
    int32_t tid = token_ids.data[i];
    memcpy(&out.data[(size_t)i * (size_t)table.cols],
           &table.data[(size_t)tid * (size_t)table.cols], row_bytes);
  }
  return out;
}

#endif // VIRA_RT_LLM_H
