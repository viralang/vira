#ifndef VIRA_RT_TENSOR_QUANT_SIMD_H
#define VIRA_RT_TENSOR_QUANT_SIMD_H

// ============================================================================
// 3.6 QUANTIZED INTEGER-SIMD CPU LINEAR ENGINE (PHASE 4.2)
// ============================================================================

typedef struct {
  float d;
  float sum_x;
  int32_t sum_q;
  int8_t qs[32];
} _vira_block_q8_0;

typedef struct {
  float d;
  int8_t qs[256];
  int16_t bsums[16];
} _vira_block_q8_K;

#if defined(__AVX2__)
static inline int32_t _vira_avx2_hsum_i32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(s);
}

static inline float _vira_avx2_hsum_f32(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}
#endif
#if defined(__SSSE3__)
static inline int32_t _vira_sse_hsum_i32(__m128i v) {
  v = _mm_hadd_epi32(v, v);
  v = _mm_hadd_epi32(v, v);
  return _mm_cvtsi128_si32(v);
}
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
static inline int32x4_t _vira_neon_dot_s8(int32x4_t acc, int8x16_t a,
                                          int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
  return vdotq_s32(acc, a, b);
#else
  int16x8_t p = vmull_s8(vget_low_s8(a), vget_low_s8(b));
  p = vmlal_s8(p, vget_high_s8(a), vget_high_s8(b));
  return vpadalq_s16(acc, p);
#endif
}
static inline int32_t _vira_neon_hsum_i32(int32x4_t v) {
#if defined(__aarch64__)
  return vaddvq_s32(v);
#else
  int32x2_t r = vpadd_s32(vget_low_s32(v), vget_high_s32(v));
  return vget_lane_s32(vpadd_s32(r, r), 0);
#endif
}
#endif

static inline void _vira_quantize_row_q8_0(const float *x, int32_t K,
                                           _vira_block_q8_0 *y) {
  int32_t nb = K / 32;
#if defined(__AVX2__)
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 v_min127 = _mm256_set1_ps(-127.0f);
  const __m256 v_max127 = _mm256_set1_ps(127.0f);
  const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  for (int32_t b = 0; b < nb; ++b) {
    const float *xb = x + (size_t)b * 32u;
    if (b + 2 < nb) {
      _mm_prefetch((const char *)(xb + 64), _MM_HINT_T0);
    }
    __m256 x0 = _mm256_loadu_ps(xb + 0);
    __m256 x1 = _mm256_loadu_ps(xb + 8);
    __m256 x2 = _mm256_loadu_ps(xb + 16);
    __m256 x3 = _mm256_loadu_ps(xb + 24);
    __m256 vs = _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3));
    __m256 vm = _mm256_max_ps(
        _mm256_max_ps(_mm256_andnot_ps(sign_mask, x0),
                      _mm256_andnot_ps(sign_mask, x1)),
        _mm256_max_ps(_mm256_andnot_ps(sign_mask, x2),
                      _mm256_andnot_ps(sign_mask, x3)));
    __m128 vm128 = _mm_max_ps(_mm256_castps256_ps128(vm),
                              _mm256_extractf128_ps(vm, 1));
    vm128 = _mm_max_ps(vm128, _mm_movehl_ps(vm128, vm128));
    vm128 = _mm_max_ss(vm128, _mm_shuffle_ps(vm128, vm128, 1));
    float amax = _mm_cvtss_f32(vm128);
    __m128 vs128 = _mm_add_ps(_mm256_castps256_ps128(vs),
                              _mm256_extractf128_ps(vs, 1));
    vs128 = _mm_add_ps(vs128, _mm_movehl_ps(vs128, vs128));
    vs128 = _mm_add_ss(vs128, _mm_shuffle_ps(vs128, vs128, 1));
    float sum_x = _mm_cvtss_f32(vs128);
    float d = amax / 127.0f;
    float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    __m256 vid = _mm256_set1_ps(id);
    __m256i i0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x0, vid), v_min127), v_max127));
    __m256i i1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x1, vid), v_min127), v_max127));
    __m256i i2 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x2, vid), v_min127), v_max127));
    __m256i i3 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x3, vid), v_min127), v_max127));
    __m256i vsq = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
    __m256i p16_01 = _mm256_packs_epi32(i0, i1);
    __m256i p16_23 = _mm256_packs_epi32(i2, i3);
    __m256i p8 = _mm256_packs_epi16(p16_01, p16_23);
    __m256i p8_ord = _mm256_permutevar8x32_epi32(p8, perm);
    _mm256_storeu_si256((__m256i *)y[b].qs, p8_ord);
    y[b].d = d;
    y[b].sum_x = sum_x;
    y[b].sum_q = _vira_avx2_hsum_i32(vsq);
  }
#else
  for (int32_t b = 0; b < nb; ++b) {
    const float *xb = x + (size_t)b * 32u;
    float amax = 0.0f;
    float sum_x = 0.0f;
    for (int32_t j = 0; j < 32; ++j) {
      float v = xb[j];
      float av = v < 0.0f ? -v : v;
      if (av > amax)
        amax = av;
      sum_x += v;
    }
    float d = amax / 127.0f;
    float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int32_t sum_q = 0;
    for (int32_t j = 0; j < 32; ++j) {
      float scaled = xb[j] * id;
      int32_t q = (scaled >= 0.0f) ? (int32_t)(scaled + 0.5f)
                                   : (int32_t)(scaled - 0.5f);
      if (q > 127) q = 127;
      else if (q < -127) q = -127;
      y[b].qs[j] = (int8_t)q;
      sum_q += q;
    }
    y[b].d = d;
    y[b].sum_x = sum_x;
    y[b].sum_q = sum_q;
  }
#endif
}

static inline void _vira_quantize_row_q8_K(const float *x, int32_t K,
                                           _vira_block_q8_K *y) {
  int32_t nb = K / 256;
  for (int32_t b = 0; b < nb; ++b) {
    const float *xb = x + (size_t)b * 256u;
    float amax = 0.0f;
    for (int32_t j = 0; j < 256; ++j) {
      float av = xb[j] < 0.0f ? -xb[j] : xb[j];
      if (av > amax)
        amax = av;
    }
    float d = amax / 127.0f;
    float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    y[b].d = d;
    for (int32_t s = 0; s < 16; ++s) {
      const float *xs = xb + s * 16;
      int8_t *qs = y[b].qs + s * 16;
      int32_t bs = 0;
      for (int32_t j = 0; j < 16; ++j) {
        float scaled = xs[j] * id;
        int32_t q = (scaled >= 0.0f) ? (int32_t)(scaled + 0.5f)
                                     : (int32_t)(scaled - 0.5f);
        if (q > 127) q = 127;
        else if (q < -127) q = -127;
        qs[j] = (int8_t)q;
        bs += q;
      }
      y[b].bsums[s] = (int16_t)bs;
    }
  }
}

static inline float _vira_cpu_dot_q4_0_q8_0(const uint8_t *w_row,
                                            const _vira_block_q8_0 *x,
                                            int32_t nb) {
#if defined(__AVX2__)
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 vacc = _mm256_setzero_ps();
  float corr = 0.0f;
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 18u;
    if (b + 4 < nb) _mm_prefetch((const char *)(blk + 4u * 18u), _MM_HINT_T0);
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float sc = _vira_fp16_to_fp32(dh) * x[b].d;
    corr += sc * (float)x[b].sum_q;
    __m128i wq = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m256i wu = _mm256_and_si256(
        _mm256_inserti128_si256(_mm256_castsi128_si256(wq),
                                _mm_srli_epi16(wq, 4), 1),
        m4);
    __m256i qv = _mm256_loadu_si256((const __m256i *)x[b].qs);
    __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(wu, qv), ones);
    __m256 pf = _mm256_cvtepi32_ps(p32);
    __m256 vsc = _mm256_set1_ps(sc);
#if defined(__FMA__)
    vacc = _mm256_fmadd_ps(vsc, pf, vacc);
#else
    vacc = _mm256_add_ps(vacc, _mm256_mul_ps(vsc, pf));
#endif
  }
  return _vira_avx2_hsum_f32(vacc) - 8.0f * corr;
#else
  float sumf = 0.0f;
#if defined(__SSSE3__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i ones = _mm_set1_epi16(1);
#elif defined(__ARM_NEON) || defined(__aarch64__)
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 18u;
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float d = _vira_fp16_to_fp32(dh);
    int32_t dot = 0;
#if defined(__SSSE3__)
    __m128i wq = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m128i u0 = _mm_and_si128(wq, m4);
    __m128i u1 = _mm_and_si128(_mm_srli_epi16(wq, 4), m4);
    __m128i p0 = _mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)x[b].qs));
    __m128i p1 =
        _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x[b].qs + 16)));
    dot = _vira_sse_hsum_i32(_mm_madd_epi16(_mm_add_epi16(p0, p1), ones));
#elif defined(__ARM_NEON) || defined(__aarch64__)
    uint8x16_t wq = vld1q_u8(blk + 2);
    int8x16_t u0 = vreinterpretq_s8_u8(vandq_u8(wq, m4));
    int8x16_t u1 = vreinterpretq_s8_u8(vshrq_n_u8(wq, 4));
    int32x4_t acc = _vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(x[b].qs));
    acc = _vira_neon_dot_s8(acc, u1, vld1q_s8(x[b].qs + 16));
    dot = _vira_neon_hsum_i32(acc);
#else
    for (int32_t j = 0; j < 16; ++j) {
      uint8_t byte = blk[2 + j];
      dot += (int32_t)(byte & 0x0Fu) * (int32_t)x[b].qs[j];
      dot += (int32_t)(byte >> 4) * (int32_t)x[b].qs[j + 16];
    }
#endif
    sumf += d * x[b].d * (float)(dot - 8 * x[b].sum_q);
  }
  return sumf;
#endif
}

static inline void _vira_cpu_dot_4w_q4_0_q8_0(
    const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
    const _vira_block_q8_0 *x, int32_t nb, float *out4) {
#if defined(__AVX2__)
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  __m128 corr4 = _mm_setzero_ps();
  for (int32_t b = 0; b < nb; ++b) {
    size_t off = (size_t)b * 18u;
    const uint8_t *b0 = w0 + off;
    const uint8_t *b1 = w1 + off;
    const uint8_t *b2 = w2 + off;
    const uint8_t *b3 = w3 + off;
    if (b + 4 < nb) {
      _mm_prefetch((const char *)(b0 + 4u * 18u), _MM_HINT_T0);
      _mm_prefetch((const char *)(b1 + 4u * 18u), _MM_HINT_T0);
      _mm_prefetch((const char *)(b2 + 4u * 18u), _MM_HINT_T0);
      _mm_prefetch((const char *)(b3 + 4u * 18u), _MM_HINT_T0);
    }
    uint16_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
    memcpy(&h0, b0, 2); memcpy(&h1, b1, 2); memcpy(&h2, b2, 2); memcpy(&h3, b3, 2);
#if defined(__F16C__)
    uint64_t hpack = (uint64_t)h0 | ((uint64_t)h1 << 16) |
                     ((uint64_t)h2 << 32) | ((uint64_t)h3 << 48);
    __m128 vd = _mm_cvtph_ps(_mm_cvtsi64_si128((long long)hpack));
#else
    __m128 vd = _mm_set_ps(_vira_fp16_to_fp32(h3), _vira_fp16_to_fp32(h2),
                           _vira_fp16_to_fp32(h1), _vira_fp16_to_fp32(h0));
#endif
    __m128 sc4 = _mm_mul_ps(vd, _mm_set1_ps(x[b].d));
    __m128 xsq = _mm_set1_ps((float)x[b].sum_q);
#if defined(__FMA__)
    corr4 = _mm_fmadd_ps(sc4, xsq, corr4);
#else
    corr4 = _mm_add_ps(corr4, _mm_mul_ps(sc4, xsq));
#endif
    __m256 s256 = _mm256_insertf128_ps(_mm256_castps128_ps256(sc4), sc4, 1);
    __m256i qv = _mm256_loadu_si256((const __m256i *)x[b].qs);

    __m128i wq0 = _mm_loadu_si128((const __m128i *)(b0 + 2));
    __m256i wu0 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(wq0), _mm_srli_epi16(wq0, 4), 1), m4);
    __m256 p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu0, qv), ones));

    __m128i wq1 = _mm_loadu_si128((const __m128i *)(b1 + 2));
    __m256i wu1 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(wq1), _mm_srli_epi16(wq1, 4), 1), m4);
    __m256 p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu1, qv), ones));

    __m128i wq2 = _mm_loadu_si128((const __m128i *)(b2 + 2));
    __m256i wu2 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(wq2), _mm_srli_epi16(wq2, 4), 1), m4);
    __m256 p2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu2, qv), ones));

    __m128i wq3 = _mm_loadu_si128((const __m128i *)(b3 + 2));
    __m256i wu3 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(wq3), _mm_srli_epi16(wq3, 4), 1), m4);
    __m256 p3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu3, qv), ones));

#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x00), p0, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x55), p1, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xAA), p2, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xFF), p3, acc3);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_permute_ps(s256, 0x00), p0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_permute_ps(s256, 0x55), p1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_permute_ps(s256, 0xAA), p2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_permute_ps(s256, 0xFF), p3));
#endif
  }
  __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
  __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1));
  __m128 r2 = _mm_add_ps(_mm256_castps256_ps128(acc2), _mm256_extractf128_ps(acc2, 1));
  __m128 r3 = _mm_add_ps(_mm256_castps256_ps128(acc3), _mm256_extractf128_ps(acc3, 1));
  _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
  __m128 tot = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3));
#if defined(__FMA__)
  tot = _mm_fnmadd_ps(_mm_set1_ps(8.0f), corr4, tot);
#else
  tot = _mm_sub_ps(tot, _mm_mul_ps(_mm_set1_ps(8.0f), corr4));
#endif
  _mm_storeu_ps(out4, tot);
#else
  out4[0] = _vira_cpu_dot_q4_0_q8_0(w0, x, nb);
  out4[1] = _vira_cpu_dot_q4_0_q8_0(w1, x, nb);
  out4[2] = _vira_cpu_dot_q4_0_q8_0(w2, x, nb);
  out4[3] = _vira_cpu_dot_q4_0_q8_0(w3, x, nb);
#endif
}

static inline void _vira_cpu_dot_4rows_q4_0_q8_0(
    const uint8_t *w_row, const _vira_block_q8_0 *x0,
    const _vira_block_q8_0 *x1, const _vira_block_q8_0 *x2,
    const _vira_block_q8_0 *x3, int32_t nb, float *out4) {
#if defined(__AVX2__)
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  __m128 corr4 = _mm_setzero_ps();
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 18u;
    if (b + 4 < nb) _mm_prefetch((const char *)(blk + 4u * 18u), _MM_HINT_T0);
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float d = _vira_fp16_to_fp32(dh);
    __m128 xd4 = _mm_set_ps(x3[b].d, x2[b].d, x1[b].d, x0[b].d);
    __m128 sc4 = _mm_mul_ps(_mm_set1_ps(d), xd4);
    __m128 sq4 = _mm_set_ps((float)x3[b].sum_q, (float)x2[b].sum_q,
                            (float)x1[b].sum_q, (float)x0[b].sum_q);
#if defined(__FMA__)
    corr4 = _mm_fmadd_ps(sc4, sq4, corr4);
#else
    corr4 = _mm_add_ps(corr4, _mm_mul_ps(sc4, sq4));
#endif
    __m256 s256 = _mm256_insertf128_ps(_mm256_castps128_ps256(sc4), sc4, 1);
    __m128i wq = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m256i wu = _mm256_and_si256(
        _mm256_inserti128_si256(_mm256_castsi128_si256(wq),
                                _mm_srli_epi16(wq, 4), 1),
        m4);
    __m256 p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x0[b].qs)), ones));
    __m256 p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x1[b].qs)), ones));
    __m256 p2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x2[b].qs)), ones));
    __m256 p3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x3[b].qs)), ones));
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x00), p0, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x55), p1, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xAA), p2, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xFF), p3, acc3);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_permute_ps(s256, 0x00), p0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_permute_ps(s256, 0x55), p1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_permute_ps(s256, 0xAA), p2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_permute_ps(s256, 0xFF), p3));
#endif
  }
  __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
  __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1));
  __m128 r2 = _mm_add_ps(_mm256_castps256_ps128(acc2), _mm256_extractf128_ps(acc2, 1));
  __m128 r3 = _mm_add_ps(_mm256_castps256_ps128(acc3), _mm256_extractf128_ps(acc3, 1));
  _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
  __m128 tot = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3));
#if defined(__FMA__)
  tot = _mm_fnmadd_ps(_mm_set1_ps(8.0f), corr4, tot);
#else
  tot = _mm_sub_ps(tot, _mm_mul_ps(_mm_set1_ps(8.0f), corr4));
#endif
  _mm_storeu_ps(out4, tot);
#else
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#if defined(__SSSE3__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i ones = _mm_set1_epi16(1);
#elif defined(__ARM_NEON) || defined(__aarch64__)
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 18u;
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float d = _vira_fp16_to_fp32(dh);
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__SSSE3__)
    __m128i wq = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m128i u0 = _mm_and_si128(wq, m4);
    __m128i u1 = _mm_and_si128(_mm_srli_epi16(wq, 4), m4);
    d0 = _vira_sse_hsum_i32(_mm_madd_epi16(
        _mm_add_epi16(_mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)x0[b].qs)),
                      _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x0[b].qs + 16)))),
        ones));
    d1 = _vira_sse_hsum_i32(_mm_madd_epi16(
        _mm_add_epi16(_mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)x1[b].qs)),
                      _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x1[b].qs + 16)))),
        ones));
    d2 = _vira_sse_hsum_i32(_mm_madd_epi16(
        _mm_add_epi16(_mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)x2[b].qs)),
                      _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x2[b].qs + 16)))),
        ones));
    d3 = _vira_sse_hsum_i32(_mm_madd_epi16(
        _mm_add_epi16(_mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)x3[b].qs)),
                      _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x3[b].qs + 16)))),
        ones));
#elif defined(__ARM_NEON) || defined(__aarch64__)
    uint8x16_t wq = vld1q_u8(blk + 2);
    int8x16_t u0 = vreinterpretq_s8_u8(vandq_u8(wq, m4));
    int8x16_t u1 = vreinterpretq_s8_u8(vshrq_n_u8(wq, 4));
    d0 = _vira_neon_hsum_i32(_vira_neon_dot_s8(
        _vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(x0[b].qs)), u1,
        vld1q_s8(x0[b].qs + 16)));
    d1 = _vira_neon_hsum_i32(_vira_neon_dot_s8(
        _vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(x1[b].qs)), u1,
        vld1q_s8(x1[b].qs + 16)));
    d2 = _vira_neon_hsum_i32(_vira_neon_dot_s8(
        _vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(x2[b].qs)), u1,
        vld1q_s8(x2[b].qs + 16)));
    d3 = _vira_neon_hsum_i32(_vira_neon_dot_s8(
        _vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(x3[b].qs)), u1,
        vld1q_s8(x3[b].qs + 16)));
#else
    for (int32_t j = 0; j < 16; ++j) {
      int32_t v0 = (int32_t)(blk[2 + j] & 0x0Fu);
      int32_t v1 = (int32_t)(blk[2 + j] >> 4);
      d0 += v0 * x0[b].qs[j] + v1 * x0[b].qs[j + 16];
      d1 += v0 * x1[b].qs[j] + v1 * x1[b].qs[j + 16];
      d2 += v0 * x2[b].qs[j] + v1 * x2[b].qs[j + 16];
      d3 += v0 * x3[b].qs[j] + v1 * x3[b].qs[j + 16];
    }
#endif
    s0 += d * x0[b].d * (float)(d0 - 8 * x0[b].sum_q);
    s1 += d * x1[b].d * (float)(d1 - 8 * x1[b].sum_q);
    s2 += d * x2[b].d * (float)(d2 - 8 * x2[b].sum_q);
    s3 += d * x3[b].d * (float)(d3 - 8 * x3[b].sum_q);
  }
  out4[0] = s0;
  out4[1] = s1;
  out4[2] = s2;
  out4[3] = s3;
#endif
}

static inline void _vira_cpu_dot_2w4m_q4_0_q8_0(
    const uint8_t *w0, const uint8_t *w1,
    const _vira_block_q8_0 *x0, const _vira_block_q8_0 *x1,
    const _vira_block_q8_0 *x2, const _vira_block_q8_0 *x3,
    int32_t nb, float *out0_4, float *out1_4) {
#if defined(__AVX2__)
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
  __m256 acc02 = _mm256_setzero_ps(), acc03 = _mm256_setzero_ps();
  __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
  __m256 acc12 = _mm256_setzero_ps(), acc13 = _mm256_setzero_ps();
  __m128 c0 = _mm_setzero_ps(), c1 = _mm_setzero_ps();
  for (int32_t b = 0; b < nb; ++b) {
    size_t off = (size_t)b * 18u;
    const uint8_t *bw0 = w0 + off;
    const uint8_t *bw1 = w1 + off;
    if (b + 4 < nb) {
      _mm_prefetch((const char *)(bw0 + 4u * 18u), _MM_HINT_T0);
      _mm_prefetch((const char *)(bw1 + 4u * 18u), _MM_HINT_T0);
    }
    uint16_t h0 = 0, h1 = 0;
    memcpy(&h0, bw0, 2); memcpy(&h1, bw1, 2);
    float d0 = _vira_fp16_to_fp32(h0);
    float d1 = _vira_fp16_to_fp32(h1);
    __m128 xd4 = _mm_set_ps(x3[b].d, x2[b].d, x1[b].d, x0[b].d);
    __m128 sq4 = _mm_set_ps((float)x3[b].sum_q, (float)x2[b].sum_q,
                            (float)x1[b].sum_q, (float)x0[b].sum_q);
    {
      __m128 sc0 = _mm_mul_ps(_mm_set1_ps(d0), xd4);
#if defined(__FMA__)
      c0 = _mm_fmadd_ps(sc0, sq4, c0);
#else
      c0 = _mm_add_ps(c0, _mm_mul_ps(sc0, sq4));
#endif
      __m256 s256 = _mm256_insertf128_ps(_mm256_castps128_ps256(sc0), sc0, 1);
      __m128i wq = _mm_loadu_si128((const __m128i *)(bw0 + 2));
      __m256i wu = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(wq), _mm_srli_epi16(wq, 4), 1), m4);
      __m256 p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x0[b].qs)), ones));
      __m256 p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x1[b].qs)), ones));
      __m256 p2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x2[b].qs)), ones));
      __m256 p3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x3[b].qs)), ones));
#if defined(__FMA__)
      acc00 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x00), p0, acc00);
      acc01 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x55), p1, acc01);
      acc02 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xAA), p2, acc02);
      acc03 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xFF), p3, acc03);
#else
      acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(_mm256_permute_ps(s256, 0x00), p0));
      acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(_mm256_permute_ps(s256, 0x55), p1));
      acc02 = _mm256_add_ps(acc02, _mm256_mul_ps(_mm256_permute_ps(s256, 0xAA), p2));
      acc03 = _mm256_add_ps(acc03, _mm256_mul_ps(_mm256_permute_ps(s256, 0xFF), p3));
#endif
    }
    {
      __m128 sc1 = _mm_mul_ps(_mm_set1_ps(d1), xd4);
#if defined(__FMA__)
      c1 = _mm_fmadd_ps(sc1, sq4, c1);
#else
      c1 = _mm_add_ps(c1, _mm_mul_ps(sc1, sq4));
#endif
      __m256 s256 = _mm256_insertf128_ps(_mm256_castps128_ps256(sc1), sc1, 1);
      __m128i wq = _mm_loadu_si128((const __m128i *)(bw1 + 2));
      __m256i wu = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(wq), _mm_srli_epi16(wq, 4), 1), m4);
      __m256 p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x0[b].qs)), ones));
      __m256 p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x1[b].qs)), ones));
      __m256 p2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x2[b].qs)), ones));
      __m256 p3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x3[b].qs)), ones));
#if defined(__FMA__)
      acc10 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x00), p0, acc10);
      acc11 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x55), p1, acc11);
      acc12 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xAA), p2, acc12);
      acc13 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xFF), p3, acc13);
#else
      acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(_mm256_permute_ps(s256, 0x00), p0));
      acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(_mm256_permute_ps(s256, 0x55), p1));
      acc12 = _mm256_add_ps(acc12, _mm256_mul_ps(_mm256_permute_ps(s256, 0xAA), p2));
      acc13 = _mm256_add_ps(acc13, _mm256_mul_ps(_mm256_permute_ps(s256, 0xFF), p3));
#endif
    }
  }
  {
    __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(acc00), _mm256_extractf128_ps(acc00, 1));
    __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(acc01), _mm256_extractf128_ps(acc01, 1));
    __m128 r2 = _mm_add_ps(_mm256_castps256_ps128(acc02), _mm256_extractf128_ps(acc02, 1));
    __m128 r3 = _mm_add_ps(_mm256_castps256_ps128(acc03), _mm256_extractf128_ps(acc03, 1));
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    __m128 tot = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3));
#if defined(__FMA__)
    tot = _mm_fnmadd_ps(_mm_set1_ps(8.0f), c0, tot);
#else
    tot = _mm_sub_ps(tot, _mm_mul_ps(_mm_set1_ps(8.0f), c0));
#endif
    _mm_storeu_ps(out0_4, tot);
  }
  {
    __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(acc10), _mm256_extractf128_ps(acc10, 1));
    __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(acc11), _mm256_extractf128_ps(acc11, 1));
    __m128 r2 = _mm_add_ps(_mm256_castps256_ps128(acc12), _mm256_extractf128_ps(acc12, 1));
    __m128 r3 = _mm_add_ps(_mm256_castps256_ps128(acc13), _mm256_extractf128_ps(acc13, 1));
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    __m128 tot = _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3));
#if defined(__FMA__)
    tot = _mm_fnmadd_ps(_mm_set1_ps(8.0f), c1, tot);
#else
    tot = _mm_sub_ps(tot, _mm_mul_ps(_mm_set1_ps(8.0f), c1));
#endif
    _mm_storeu_ps(out1_4, tot);
  }
#else
  _vira_cpu_dot_4rows_q4_0_q8_0(w0, x0, x1, x2, x3, nb, out0_4);
  _vira_cpu_dot_4rows_q4_0_q8_0(w1, x0, x1, x2, x3, nb, out1_4);
#endif
}

static inline float _vira_cpu_dot_q4_1_q8_0(const uint8_t *w_row,
                                            const _vira_block_q8_0 *x,
                                            int32_t nb) {
  float sumf = 0.0f;
#if defined(__AVX2__)
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
#elif defined(__SSSE3__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i ones = _mm_set1_epi16(1);
#elif defined(__ARM_NEON) || defined(__aarch64__)
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 20u;
    uint16_t dh = 0, mh = 0;
    memcpy(&dh, blk, 2);
    memcpy(&mh, blk + 2, 2);
    float d = _vira_fp16_to_fp32(dh);
    float m = _vira_fp16_to_fp32(mh);
    int32_t dot = 0;
#if defined(__AVX2__)
    __m128i wq = _mm_loadu_si128((const __m128i *)(blk + 4));
    __m256i wu = _mm256_and_si256(
        _mm256_inserti128_si256(_mm256_castsi128_si256(wq),
                                _mm_srli_epi16(wq, 4), 1),
        m4);
    dot = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x[b].qs)),
        ones));
#elif defined(__SSSE3__)
    __m128i wq = _mm_loadu_si128((const __m128i *)(blk + 4));
    __m128i u0 = _mm_and_si128(wq, m4);
    __m128i u1 = _mm_and_si128(_mm_srli_epi16(wq, 4), m4);
    dot = _vira_sse_hsum_i32(_mm_madd_epi16(
        _mm_add_epi16(_mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)x[b].qs)),
                      _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x[b].qs + 16)))),
        ones));
#elif defined(__ARM_NEON) || defined(__aarch64__)
    uint8x16_t wq = vld1q_u8(blk + 4);
    int32x4_t acc = _vira_neon_dot_s8(
        vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(wq, m4)),
        vld1q_s8(x[b].qs));
    acc = _vira_neon_dot_s8(acc, vreinterpretq_s8_u8(vshrq_n_u8(wq, 4)),
                            vld1q_s8(x[b].qs + 16));
    dot = _vira_neon_hsum_i32(acc);
#else
    for (int32_t j = 0; j < 16; ++j) {
      uint8_t byte = blk[4 + j];
      dot += (int32_t)(byte & 0x0Fu) * (int32_t)x[b].qs[j];
      dot += (int32_t)(byte >> 4) * (int32_t)x[b].qs[j + 16];
    }
#endif
    sumf += d * x[b].d * (float)dot + m * x[b].sum_x;
  }
  return sumf;
}

static inline float _vira_cpu_dot_q5_0_q8_0(const uint8_t *w_row,
                                            const _vira_block_q8_0 *x,
                                            int32_t nb) {
  float sumf = 0.0f;
#if defined(__AVX2__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i sh0 = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  const __m128i sh1 = _mm_set_epi8(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2);
  const __m128i bmask = _mm_set_epi8((char)-128, 64, 32, 16, 8, 4, 2, 1,
                                     (char)-128, 64, 32, 16, 8, 4, 2, 1);
  const __m128i hi16 = _mm_set1_epi8(16);
  const __m256i ones = _mm256_set1_epi16(1);
#elif defined(__SSSE3__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i sh0 = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  const __m128i sh1 = _mm_set_epi8(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2);
  const __m128i bmask = _mm_set_epi8((char)-128, 64, 32, 16, 8, 4, 2, 1,
                                     (char)-128, 64, 32, 16, 8, 4, 2, 1);
  const __m128i hi16 = _mm_set1_epi8(16);
  const __m128i ones = _mm_set1_epi16(1);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 22u;
    uint16_t dh = 0;
    uint32_t qh = 0;
    memcpy(&dh, blk, 2);
    memcpy(&qh, blk + 2, 4);
    float d = _vira_fp16_to_fp32(dh);
    int32_t dot = 0;
#if defined(__AVX2__) || defined(__SSSE3__)
    __m128i vqh = _mm_set1_epi32((int32_t)qh);
    __m128i h0 = _mm_and_si128(
        _mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(vqh, sh0), bmask), bmask),
        hi16);
    __m128i h1 = _mm_and_si128(
        _mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(vqh, sh1), bmask), bmask),
        hi16);
    __m128i wq = _mm_loadu_si128((const __m128i *)(blk + 6));
    __m128i u0 = _mm_or_si128(_mm_and_si128(wq, m4), h0);
    __m128i u1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(wq, 4), m4), h1);
#if defined(__AVX2__)
    __m256i wu = _mm256_inserti128_si256(_mm256_castsi128_si256(u0), u1, 1);
    dot = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x[b].qs)),
        ones));
#else
    __m128i p0 = _mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)x[b].qs));
    __m128i p1 =
        _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x[b].qs + 16)));
    dot = _vira_sse_hsum_i32(_mm_madd_epi16(_mm_add_epi16(p0, p1), ones));
#endif
#else
    for (int32_t j = 0; j < 16; ++j) {
      uint8_t byte = blk[6 + j];
      int32_t v0 = (int32_t)((uint32_t)(byte & 0x0Fu) | (((qh >> j) & 1u) << 4));
      int32_t v1 =
          (int32_t)((uint32_t)(byte >> 4) | (((qh >> (j + 16)) & 1u) << 4));
      dot += v0 * (int32_t)x[b].qs[j] + v1 * (int32_t)x[b].qs[j + 16];
    }
#endif
    sumf += d * x[b].d * (float)(dot - 16 * x[b].sum_q);
  }
  return sumf;
}

static inline float _vira_cpu_dot_q5_1_q8_0(const uint8_t *w_row,
                                            const _vira_block_q8_0 *x,
                                            int32_t nb) {
  float sumf = 0.0f;
#if defined(__AVX2__) || defined(__SSSE3__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i sh0 = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  const __m128i sh1 = _mm_set_epi8(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2);
  const __m128i bmask = _mm_set_epi8((char)-128, 64, 32, 16, 8, 4, 2, 1,
                                     (char)-128, 64, 32, 16, 8, 4, 2, 1);
  const __m128i hi16 = _mm_set1_epi8(16);
#if defined(__AVX2__)
  const __m256i ones = _mm256_set1_epi16(1);
#else
  const __m128i ones = _mm_set1_epi16(1);
#endif
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 24u;
    uint16_t dh = 0, mh = 0;
    uint32_t qh = 0;
    memcpy(&dh, blk, 2);
    memcpy(&mh, blk + 2, 2);
    memcpy(&qh, blk + 4, 4);
    float d = _vira_fp16_to_fp32(dh);
    float m = _vira_fp16_to_fp32(mh);
    int32_t dot = 0;
#if defined(__AVX2__) || defined(__SSSE3__)
    __m128i vqh = _mm_set1_epi32((int32_t)qh);
    __m128i h0 = _mm_and_si128(
        _mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(vqh, sh0), bmask), bmask),
        hi16);
    __m128i h1 = _mm_and_si128(
        _mm_cmpeq_epi8(_mm_and_si128(_mm_shuffle_epi8(vqh, sh1), bmask), bmask),
        hi16);
    __m128i wq = _mm_loadu_si128((const __m128i *)(blk + 8));
    __m128i u0 = _mm_or_si128(_mm_and_si128(wq, m4), h0);
    __m128i u1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(wq, 4), m4), h1);
#if defined(__AVX2__)
    __m256i wu = _mm256_inserti128_si256(_mm256_castsi128_si256(u0), u1, 1);
    dot = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x[b].qs)),
        ones));
#else
    __m128i p0 = _mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)x[b].qs));
    __m128i p1 =
        _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x[b].qs + 16)));
    dot = _vira_sse_hsum_i32(_mm_madd_epi16(_mm_add_epi16(p0, p1), ones));
#endif
#else
    for (int32_t j = 0; j < 16; ++j) {
      uint8_t byte = blk[8 + j];
      int32_t v0 = (int32_t)((uint32_t)(byte & 0x0Fu) | (((qh >> j) & 1u) << 4));
      int32_t v1 =
          (int32_t)((uint32_t)(byte >> 4) | (((qh >> (j + 16)) & 1u) << 4));
      dot += v0 * (int32_t)x[b].qs[j] + v1 * (int32_t)x[b].qs[j + 16];
    }
#endif
    sumf += d * x[b].d * (float)dot + m * x[b].sum_x;
  }
  return sumf;
}

static inline float _vira_cpu_dot_q8_0_q8_0(const uint8_t *w_row,
                                            const _vira_block_q8_0 *x,
                                            int32_t nb) {
#if defined(__AVX2__)
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 vacc = _mm256_setzero_ps();
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 34u;
    if (b + 4 < nb) _mm_prefetch((const char *)(blk + 4u * 34u), _MM_HINT_T0);
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float sc = _vira_fp16_to_fp32(dh) * x[b].d;
    __m256i wv = _mm256_loadu_si256((const __m256i *)(blk + 2));
    __m256i qv = _mm256_loadu_si256((const __m256i *)x[b].qs);
    __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(qv, wv));
    __m256 pf = _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones));
#if defined(__FMA__)
    vacc = _mm256_fmadd_ps(_mm256_set1_ps(sc), pf, vacc);
#else
    vacc = _mm256_add_ps(vacc, _mm256_mul_ps(_mm256_set1_ps(sc), pf));
#endif
  }
  return _vira_avx2_hsum_f32(vacc);
#else
  float sumf = 0.0f;
#if defined(__SSSE3__)
  const __m128i ones = _mm_set1_epi16(1);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 34u;
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float d = _vira_fp16_to_fp32(dh);
    int32_t dot = 0;
#if defined(__SSSE3__)
    __m128i w0 = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m128i w1 = _mm_loadu_si128((const __m128i *)(blk + 18));
    __m128i q0 = _mm_loadu_si128((const __m128i *)x[b].qs);
    __m128i q1 = _mm_loadu_si128((const __m128i *)(x[b].qs + 16));
    __m128i p0 = _mm_maddubs_epi16(_mm_abs_epi8(w0), _mm_sign_epi8(q0, w0));
    __m128i p1 = _mm_maddubs_epi16(_mm_abs_epi8(w1), _mm_sign_epi8(q1, w1));
    dot = _vira_sse_hsum_i32(
        _mm_add_epi32(_mm_madd_epi16(p0, ones), _mm_madd_epi16(p1, ones)));
#elif defined(__ARM_NEON) || defined(__aarch64__)
    const int8_t *wqs = (const int8_t *)(blk + 2);
    int32x4_t acc = _vira_neon_dot_s8(vdupq_n_s32(0), vld1q_s8(wqs), vld1q_s8(x[b].qs));
    acc = _vira_neon_dot_s8(acc, vld1q_s8(wqs + 16), vld1q_s8(x[b].qs + 16));
    dot = _vira_neon_hsum_i32(acc);
#else
    const int8_t *wqs = (const int8_t *)(blk + 2);
    for (int32_t j = 0; j < 32; ++j) {
      dot += (int32_t)wqs[j] * (int32_t)x[b].qs[j];
    }
#endif
    sumf += d * x[b].d * (float)dot;
  }
  return sumf;
#endif
}

static inline void _vira_cpu_dot_4w_q8_0_q8_0(
    const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
    const _vira_block_q8_0 *x, int32_t nb, float *out4) {
#if defined(__AVX2__)
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  for (int32_t b = 0; b < nb; ++b) {
    size_t off = (size_t)b * 34u;
    const uint8_t *b0 = w0 + off;
    const uint8_t *b1 = w1 + off;
    const uint8_t *b2 = w2 + off;
    const uint8_t *b3 = w3 + off;
    if (b + 4 < nb) {
      _mm_prefetch((const char *)(b0 + 4u * 34u), _MM_HINT_T0);
      _mm_prefetch((const char *)(b1 + 4u * 34u), _MM_HINT_T0);
      _mm_prefetch((const char *)(b2 + 4u * 34u), _MM_HINT_T0);
      _mm_prefetch((const char *)(b3 + 4u * 34u), _MM_HINT_T0);
    }
    uint16_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
    memcpy(&h0, b0, 2); memcpy(&h1, b1, 2); memcpy(&h2, b2, 2); memcpy(&h3, b3, 2);
#if defined(__F16C__)
    uint64_t hpack = (uint64_t)h0 | ((uint64_t)h1 << 16) |
                     ((uint64_t)h2 << 32) | ((uint64_t)h3 << 48);
    __m128 vd = _mm_cvtph_ps(_mm_cvtsi64_si128((long long)hpack));
#else
    __m128 vd = _mm_set_ps(_vira_fp16_to_fp32(h3), _vira_fp16_to_fp32(h2),
                           _vira_fp16_to_fp32(h1), _vira_fp16_to_fp32(h0));
#endif
    __m128 sc4 = _mm_mul_ps(vd, _mm_set1_ps(x[b].d));
    __m256 s256 = _mm256_insertf128_ps(_mm256_castps128_ps256(sc4), sc4, 1);
    __m256i qv = _mm256_loadu_si256((const __m256i *)x[b].qs);

    __m256i wv0 = _mm256_loadu_si256((const __m256i *)(b0 + 2));
    __m256 p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(wv0), _mm256_sign_epi8(qv, wv0)), ones));
    __m256i wv1 = _mm256_loadu_si256((const __m256i *)(b1 + 2));
    __m256 p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(wv1), _mm256_sign_epi8(qv, wv1)), ones));
    __m256i wv2 = _mm256_loadu_si256((const __m256i *)(b2 + 2));
    __m256 p2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(wv2), _mm256_sign_epi8(qv, wv2)), ones));
    __m256i wv3 = _mm256_loadu_si256((const __m256i *)(b3 + 2));
    __m256 p3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(wv3), _mm256_sign_epi8(qv, wv3)), ones));

#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x00), p0, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x55), p1, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xAA), p2, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xFF), p3, acc3);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_permute_ps(s256, 0x00), p0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_permute_ps(s256, 0x55), p1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_permute_ps(s256, 0xAA), p2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_permute_ps(s256, 0xFF), p3));
#endif
  }
  __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
  __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1));
  __m128 r2 = _mm_add_ps(_mm256_castps256_ps128(acc2), _mm256_extractf128_ps(acc2, 1));
  __m128 r3 = _mm_add_ps(_mm256_castps256_ps128(acc3), _mm256_extractf128_ps(acc3, 1));
  _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
  _mm_storeu_ps(out4, _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3)));
#else
  out4[0] = _vira_cpu_dot_q8_0_q8_0(w0, x, nb);
  out4[1] = _vira_cpu_dot_q8_0_q8_0(w1, x, nb);
  out4[2] = _vira_cpu_dot_q8_0_q8_0(w2, x, nb);
  out4[3] = _vira_cpu_dot_q8_0_q8_0(w3, x, nb);
#endif
}

static inline void _vira_cpu_dot_4rows_q8_0_q8_0(
    const uint8_t *w_row, const _vira_block_q8_0 *x0,
    const _vira_block_q8_0 *x1, const _vira_block_q8_0 *x2,
    const _vira_block_q8_0 *x3, int32_t nb, float *out4) {
#if defined(__AVX2__)
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 34u;
    if (b + 4 < nb) _mm_prefetch((const char *)(blk + 4u * 34u), _MM_HINT_T0);
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float d = _vira_fp16_to_fp32(dh);
    __m128 sc4 = _mm_mul_ps(_mm_set1_ps(d), _mm_set_ps(x3[b].d, x2[b].d, x1[b].d, x0[b].d));
    __m256 s256 = _mm256_insertf128_ps(_mm256_castps128_ps256(sc4), sc4, 1);
    __m256i wv = _mm256_loadu_si256((const __m256i *)(blk + 2));
    __m256i aw = _mm256_abs_epi8(wv);
    __m256 p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x0[b].qs), wv)), ones));
    __m256 p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x1[b].qs), wv)), ones));
    __m256 p2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x2[b].qs), wv)), ones));
    __m256 p3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x3[b].qs), wv)), ones));
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x00), p0, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x55), p1, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xAA), p2, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xFF), p3, acc3);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_permute_ps(s256, 0x00), p0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_permute_ps(s256, 0x55), p1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_permute_ps(s256, 0xAA), p2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_permute_ps(s256, 0xFF), p3));
#endif
  }
  __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
  __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(acc1), _mm256_extractf128_ps(acc1, 1));
  __m128 r2 = _mm_add_ps(_mm256_castps256_ps128(acc2), _mm256_extractf128_ps(acc2, 1));
  __m128 r3 = _mm_add_ps(_mm256_castps256_ps128(acc3), _mm256_extractf128_ps(acc3, 1));
  _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
  _mm_storeu_ps(out4, _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3)));
#else
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#if defined(__SSSE3__)
  const __m128i ones = _mm_set1_epi16(1);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 34u;
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float d = _vira_fp16_to_fp32(dh);
#if defined(__SSSE3__)
    __m128i w0 = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m128i w1 = _mm_loadu_si128((const __m128i *)(blk + 18));
    __m128i aw0 = _mm_abs_epi8(w0);
    __m128i aw1 = _mm_abs_epi8(w1);
    int32_t d0 = _vira_sse_hsum_i32(_mm_add_epi32(
        _mm_madd_epi16(_mm_maddubs_epi16(aw0, _mm_sign_epi8(_mm_loadu_si128((const __m128i *)x0[b].qs), w0)), ones),
        _mm_madd_epi16(_mm_maddubs_epi16(aw1, _mm_sign_epi8(_mm_loadu_si128((const __m128i *)(x0[b].qs + 16)), w1)), ones)));
    int32_t d1 = _vira_sse_hsum_i32(_mm_add_epi32(
        _mm_madd_epi16(_mm_maddubs_epi16(aw0, _mm_sign_epi8(_mm_loadu_si128((const __m128i *)x1[b].qs), w0)), ones),
        _mm_madd_epi16(_mm_maddubs_epi16(aw1, _mm_sign_epi8(_mm_loadu_si128((const __m128i *)(x1[b].qs + 16)), w1)), ones)));
    int32_t d2 = _vira_sse_hsum_i32(_mm_add_epi32(
        _mm_madd_epi16(_mm_maddubs_epi16(aw0, _mm_sign_epi8(_mm_loadu_si128((const __m128i *)x2[b].qs), w0)), ones),
        _mm_madd_epi16(_mm_maddubs_epi16(aw1, _mm_sign_epi8(_mm_loadu_si128((const __m128i *)(x2[b].qs + 16)), w1)), ones)));
    int32_t d3 = _vira_sse_hsum_i32(_mm_add_epi32(
        _mm_madd_epi16(_mm_maddubs_epi16(aw0, _mm_sign_epi8(_mm_loadu_si128((const __m128i *)x3[b].qs), w0)), ones),
        _mm_madd_epi16(_mm_maddubs_epi16(aw1, _mm_sign_epi8(_mm_loadu_si128((const __m128i *)(x3[b].qs + 16)), w1)), ones)));
#elif defined(__ARM_NEON) || defined(__aarch64__)
    const int8_t *wqs = (const int8_t *)(blk + 2);
    int8x16_t w0 = vld1q_s8(wqs), w1 = vld1q_s8(wqs + 16);
    int32_t d0 = _vira_neon_hsum_i32(_vira_neon_dot_s8(_vira_neon_dot_s8(vdupq_n_s32(0), w0, vld1q_s8(x0[b].qs)), w1, vld1q_s8(x0[b].qs + 16)));
    int32_t d1 = _vira_neon_hsum_i32(_vira_neon_dot_s8(_vira_neon_dot_s8(vdupq_n_s32(0), w0, vld1q_s8(x1[b].qs)), w1, vld1q_s8(x1[b].qs + 16)));
    int32_t d2 = _vira_neon_hsum_i32(_vira_neon_dot_s8(_vira_neon_dot_s8(vdupq_n_s32(0), w0, vld1q_s8(x2[b].qs)), w1, vld1q_s8(x2[b].qs + 16)));
    int32_t d3 = _vira_neon_hsum_i32(_vira_neon_dot_s8(_vira_neon_dot_s8(vdupq_n_s32(0), w0, vld1q_s8(x3[b].qs)), w1, vld1q_s8(x3[b].qs + 16)));
#else
    const int8_t *wqs = (const int8_t *)(blk + 2);
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int32_t j = 0; j < 32; ++j) {
      int32_t wv = wqs[j];
      d0 += wv * x0[b].qs[j];
      d1 += wv * x1[b].qs[j];
      d2 += wv * x2[b].qs[j];
      d3 += wv * x3[b].qs[j];
    }
#endif
    s0 += d * x0[b].d * (float)d0;
    s1 += d * x1[b].d * (float)d1;
    s2 += d * x2[b].d * (float)d2;
    s3 += d * x3[b].d * (float)d3;
  }
  out4[0] = s0;
  out4[1] = s1;
  out4[2] = s2;
  out4[3] = s3;
#endif
}

static inline void _vira_cpu_dot_2w4m_q8_0_q8_0(
    const uint8_t *w0, const uint8_t *w1,
    const _vira_block_q8_0 *x0, const _vira_block_q8_0 *x1,
    const _vira_block_q8_0 *x2, const _vira_block_q8_0 *x3,
    int32_t nb, float *out0_4, float *out1_4) {
#if defined(__AVX2__)
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
  __m256 acc02 = _mm256_setzero_ps(), acc03 = _mm256_setzero_ps();
  __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
  __m256 acc12 = _mm256_setzero_ps(), acc13 = _mm256_setzero_ps();
  for (int32_t b = 0; b < nb; ++b) {
    size_t off = (size_t)b * 34u;
    const uint8_t *bw0 = w0 + off;
    const uint8_t *bw1 = w1 + off;
    if (b + 4 < nb) {
      _mm_prefetch((const char *)(bw0 + 4u * 34u), _MM_HINT_T0);
      _mm_prefetch((const char *)(bw1 + 4u * 34u), _MM_HINT_T0);
    }
    uint16_t h0 = 0, h1 = 0;
    memcpy(&h0, bw0, 2); memcpy(&h1, bw1, 2);
    float d0 = _vira_fp16_to_fp32(h0);
    float d1 = _vira_fp16_to_fp32(h1);
    __m128 xd4 = _mm_set_ps(x3[b].d, x2[b].d, x1[b].d, x0[b].d);
    {
      __m128 sc0 = _mm_mul_ps(_mm_set1_ps(d0), xd4);
      __m256 s256 = _mm256_insertf128_ps(_mm256_castps128_ps256(sc0), sc0, 1);
      __m256i wv = _mm256_loadu_si256((const __m256i *)(bw0 + 2));
      __m256i aw = _mm256_abs_epi8(wv);
      __m256 p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x0[b].qs), wv)), ones));
      __m256 p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x1[b].qs), wv)), ones));
      __m256 p2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x2[b].qs), wv)), ones));
      __m256 p3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x3[b].qs), wv)), ones));
#if defined(__FMA__)
      acc00 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x00), p0, acc00);
      acc01 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x55), p1, acc01);
      acc02 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xAA), p2, acc02);
      acc03 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xFF), p3, acc03);
#else
      acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(_mm256_permute_ps(s256, 0x00), p0));
      acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(_mm256_permute_ps(s256, 0x55), p1));
      acc02 = _mm256_add_ps(acc02, _mm256_mul_ps(_mm256_permute_ps(s256, 0xAA), p2));
      acc03 = _mm256_add_ps(acc03, _mm256_mul_ps(_mm256_permute_ps(s256, 0xFF), p3));
#endif
    }
    {
      __m128 sc1 = _mm_mul_ps(_mm_set1_ps(d1), xd4);
      __m256 s256 = _mm256_insertf128_ps(_mm256_castps128_ps256(sc1), sc1, 1);
      __m256i wv = _mm256_loadu_si256((const __m256i *)(bw1 + 2));
      __m256i aw = _mm256_abs_epi8(wv);
      __m256 p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x0[b].qs), wv)), ones));
      __m256 p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x1[b].qs), wv)), ones));
      __m256 p2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x2[b].qs), wv)), ones));
      __m256 p3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x3[b].qs), wv)), ones));
#if defined(__FMA__)
      acc10 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x00), p0, acc10);
      acc11 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0x55), p1, acc11);
      acc12 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xAA), p2, acc12);
      acc13 = _mm256_fmadd_ps(_mm256_permute_ps(s256, 0xFF), p3, acc13);
#else
      acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(_mm256_permute_ps(s256, 0x00), p0));
      acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(_mm256_permute_ps(s256, 0x55), p1));
      acc12 = _mm256_add_ps(acc12, _mm256_mul_ps(_mm256_permute_ps(s256, 0xAA), p2));
      acc13 = _mm256_add_ps(acc13, _mm256_mul_ps(_mm256_permute_ps(s256, 0xFF), p3));
#endif
    }
  }
  {
    __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(acc00), _mm256_extractf128_ps(acc00, 1));
    __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(acc01), _mm256_extractf128_ps(acc01, 1));
    __m128 r2 = _mm_add_ps(_mm256_castps256_ps128(acc02), _mm256_extractf128_ps(acc02, 1));
    __m128 r3 = _mm_add_ps(_mm256_castps256_ps128(acc03), _mm256_extractf128_ps(acc03, 1));
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    _mm_storeu_ps(out0_4, _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3)));
  }
  {
    __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(acc10), _mm256_extractf128_ps(acc10, 1));
    __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(acc11), _mm256_extractf128_ps(acc11, 1));
    __m128 r2 = _mm_add_ps(_mm256_castps256_ps128(acc12), _mm256_extractf128_ps(acc12, 1));
    __m128 r3 = _mm_add_ps(_mm256_castps256_ps128(acc13), _mm256_extractf128_ps(acc13, 1));
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    _mm_storeu_ps(out1_4, _mm_add_ps(_mm_add_ps(r0, r1), _mm_add_ps(r2, r3)));
  }
#else
  _vira_cpu_dot_4rows_q8_0_q8_0(w0, x0, x1, x2, x3, nb, out0_4);
  _vira_cpu_dot_4rows_q8_0_q8_0(w1, x0, x1, x2, x3, nb, out1_4);
#endif
}

static inline float _vira_cpu_dot_q4_k_q8_K(const uint8_t *w_row,
                                            const _vira_block_q8_K *x,
                                            int32_t nb) {
  float sumf = 0.0f;
#if defined(__AVX2__)
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
#elif defined(__SSSE3__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i ones = _mm_set1_epi16(1);
#elif defined(__ARM_NEON) || defined(__aarch64__)
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 144u;
    uint16_t dh = 0, dmh = 0;
    memcpy(&dh, blk, 2);
    memcpy(&dmh, blk + 2, 2);
    float d = _vira_fp16_to_fp32(dh);
    float dmin = _vira_fp16_to_fp32(dmh);
    const uint8_t *scales = blk + 4;
    const uint8_t *qs = blk + 16;
    int32_t sumi_d = 0, sumi_m = 0;
    for (int32_t ib = 0; ib < 8; ++ib) {
      uint8_t sc = 0, m_sc = 0;
      _vira_gguf_k4_scale_min(ib, scales, &sc, &m_sc);
      const uint8_t *qs_sub = qs + ((ib >> 1) * 32);
      const int8_t *xq = x[b].qs + ib * 32;
      int32_t dot = 0;
#if defined(__AVX2__)
      __m256i qv = _mm256_loadu_si256((const __m256i *)qs_sub);
      __m256i wu = _mm256_and_si256((ib & 1) ? _mm256_srli_epi16(qv, 4) : qv, m4);
      dot = _vira_avx2_hsum_i32(_mm256_madd_epi16(
          _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)xq)),
          ones));
#elif defined(__SSSE3__)
      __m128i w0 = _mm_loadu_si128((const __m128i *)qs_sub);
      __m128i w1 = _mm_loadu_si128((const __m128i *)(qs_sub + 16));
      __m128i u0 = _mm_and_si128((ib & 1) ? _mm_srli_epi16(w0, 4) : w0, m4);
      __m128i u1 = _mm_and_si128((ib & 1) ? _mm_srli_epi16(w1, 4) : w1, m4);
      __m128i p0 = _mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)xq));
      __m128i p1 =
          _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(xq + 16)));
      dot = _vira_sse_hsum_i32(_mm_madd_epi16(_mm_add_epi16(p0, p1), ones));
#elif defined(__ARM_NEON) || defined(__aarch64__)
      uint8x16_t w0 = vld1q_u8(qs_sub), w1 = vld1q_u8(qs_sub + 16);
      int8x16_t u0 = vreinterpretq_s8_u8((ib & 1) ? vshrq_n_u8(w0, 4) : vandq_u8(w0, m4));
      int8x16_t u1 = vreinterpretq_s8_u8((ib & 1) ? vshrq_n_u8(w1, 4) : vandq_u8(w1, m4));
      dot = _vira_neon_hsum_i32(_vira_neon_dot_s8(
          _vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(xq)), u1,
          vld1q_s8(xq + 16)));
#else
      int32_t shift = (ib & 1) * 4;
      for (int32_t l = 0; l < 32; ++l) {
        dot += (int32_t)((qs_sub[l] >> shift) & 0x0Fu) * (int32_t)xq[l];
      }
#endif
      sumi_d += (int32_t)sc * dot;
      sumi_m +=
          (int32_t)m_sc * ((int32_t)x[b].bsums[2 * ib] + (int32_t)x[b].bsums[2 * ib + 1]);
    }
    sumf += x[b].d * (d * (float)sumi_d - dmin * (float)sumi_m);
  }
  return sumf;
}

static inline void _vira_cpu_dot_4rows_q4_k_q8_K(
    const uint8_t *w_row, const _vira_block_q8_K *x0,
    const _vira_block_q8_K *x1, const _vira_block_q8_K *x2,
    const _vira_block_q8_K *x3, int32_t nb, float *out4) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#if defined(__AVX2__)
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
#elif defined(__SSSE3__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i ones = _mm_set1_epi16(1);
#elif defined(__ARM_NEON) || defined(__aarch64__)
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 144u;
    uint16_t dh = 0, dmh = 0;
    memcpy(&dh, blk, 2);
    memcpy(&dmh, blk + 2, 2);
    float d = _vira_fp16_to_fp32(dh);
    float dmin = _vira_fp16_to_fp32(dmh);
    const uint8_t *scales = blk + 4;
    const uint8_t *qs = blk + 16;
    int32_t sd0 = 0, sd1 = 0, sd2 = 0, sd3 = 0;
    int32_t sm0 = 0, sm1 = 0, sm2 = 0, sm3 = 0;
    for (int32_t ib = 0; ib < 8; ++ib) {
      uint8_t sc = 0, m_sc = 0;
      _vira_gguf_k4_scale_min(ib, scales, &sc, &m_sc);
      const uint8_t *qs_sub = qs + ((ib >> 1) * 32);
      int32_t off = ib * 32;
      int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__AVX2__)
      __m256i qv = _mm256_loadu_si256((const __m256i *)qs_sub);
      __m256i wu = _mm256_and_si256((ib & 1) ? _mm256_srli_epi16(qv, 4) : qv, m4);
      d0 = _vira_avx2_hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)(x0[b].qs + off))), ones));
      d1 = _vira_avx2_hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)(x1[b].qs + off))), ones));
      d2 = _vira_avx2_hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)(x2[b].qs + off))), ones));
      d3 = _vira_avx2_hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)(x3[b].qs + off))), ones));
#elif defined(__SSSE3__)
      __m128i w0 = _mm_loadu_si128((const __m128i *)qs_sub);
      __m128i w1 = _mm_loadu_si128((const __m128i *)(qs_sub + 16));
      __m128i u0 = _mm_and_si128((ib & 1) ? _mm_srli_epi16(w0, 4) : w0, m4);
      __m128i u1 = _mm_and_si128((ib & 1) ? _mm_srli_epi16(w1, 4) : w1, m4);
      d0 = _vira_sse_hsum_i32(_mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)(x0[b].qs + off))), _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x0[b].qs + off + 16)))), ones));
      d1 = _vira_sse_hsum_i32(_mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)(x1[b].qs + off))), _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x1[b].qs + off + 16)))), ones));
      d2 = _vira_sse_hsum_i32(_mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)(x2[b].qs + off))), _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x2[b].qs + off + 16)))), ones));
      d3 = _vira_sse_hsum_i32(_mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)(x3[b].qs + off))), _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(x3[b].qs + off + 16)))), ones));
#elif defined(__ARM_NEON) || defined(__aarch64__)
      uint8x16_t w0 = vld1q_u8(qs_sub), w1 = vld1q_u8(qs_sub + 16);
      int8x16_t u0 = vreinterpretq_s8_u8((ib & 1) ? vshrq_n_u8(w0, 4) : vandq_u8(w0, m4));
      int8x16_t u1 = vreinterpretq_s8_u8((ib & 1) ? vshrq_n_u8(w1, 4) : vandq_u8(w1, m4));
      d0 = _vira_neon_hsum_i32(_vira_neon_dot_s8(_vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(x0[b].qs + off)), u1, vld1q_s8(x0[b].qs + off + 16)));
      d1 = _vira_neon_hsum_i32(_vira_neon_dot_s8(_vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(x1[b].qs + off)), u1, vld1q_s8(x1[b].qs + off + 16)));
      d2 = _vira_neon_hsum_i32(_vira_neon_dot_s8(_vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(x2[b].qs + off)), u1, vld1q_s8(x2[b].qs + off + 16)));
      d3 = _vira_neon_hsum_i32(_vira_neon_dot_s8(_vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(x3[b].qs + off)), u1, vld1q_s8(x3[b].qs + off + 16)));
#else
      int32_t shift = (ib & 1) * 4;
      for (int32_t l = 0; l < 32; ++l) {
        int32_t wv = (int32_t)((qs_sub[l] >> shift) & 0x0Fu);
        d0 += wv * x0[b].qs[off + l];
        d1 += wv * x1[b].qs[off + l];
        d2 += wv * x2[b].qs[off + l];
        d3 += wv * x3[b].qs[off + l];
      }
#endif
      sd0 += (int32_t)sc * d0; sd1 += (int32_t)sc * d1;
      sd2 += (int32_t)sc * d2; sd3 += (int32_t)sc * d3;
      sm0 += (int32_t)m_sc * ((int32_t)x0[b].bsums[2 * ib] + (int32_t)x0[b].bsums[2 * ib + 1]);
      sm1 += (int32_t)m_sc * ((int32_t)x1[b].bsums[2 * ib] + (int32_t)x1[b].bsums[2 * ib + 1]);
      sm2 += (int32_t)m_sc * ((int32_t)x2[b].bsums[2 * ib] + (int32_t)x2[b].bsums[2 * ib + 1]);
      sm3 += (int32_t)m_sc * ((int32_t)x3[b].bsums[2 * ib] + (int32_t)x3[b].bsums[2 * ib + 1]);
    }
    s0 += x0[b].d * (d * (float)sd0 - dmin * (float)sm0);
    s1 += x1[b].d * (d * (float)sd1 - dmin * (float)sm1);
    s2 += x2[b].d * (d * (float)sd2 - dmin * (float)sm2);
    s3 += x3[b].d * (d * (float)sd3 - dmin * (float)sm3);
  }
  out4[0] = s0; out4[1] = s1; out4[2] = s2; out4[3] = s3;
}

static inline float _vira_cpu_dot_q5_k_q8_K(const uint8_t *w_row,
                                            const _vira_block_q8_K *x,
                                            int32_t nb) {
  float sumf = 0.0f;
#if defined(__AVX2__)
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i hi16 = _mm256_set1_epi8(16);
  const __m256i ones = _mm256_set1_epi16(1);
#elif defined(__SSSE3__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i hi16 = _mm_set1_epi8(16);
  const __m128i ones = _mm_set1_epi16(1);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 176u;
    uint16_t dh = 0, dmh = 0;
    memcpy(&dh, blk, 2);
    memcpy(&dmh, blk + 2, 2);
    float d = _vira_fp16_to_fp32(dh);
    float dmin = _vira_fp16_to_fp32(dmh);
    const uint8_t *scales = blk + 4;
    const uint8_t *qh = blk + 16;
    const uint8_t *qs = blk + 48;
    int32_t sumi_d = 0, sumi_m = 0;
    for (int32_t ib = 0; ib < 8; ++ib) {
      uint8_t sc = 0, m_sc = 0;
      _vira_gguf_k4_scale_min(ib, scales, &sc, &m_sc);
      const uint8_t *qs_sub = qs + ((ib >> 1) * 32);
      const int8_t *xq = x[b].qs + ib * 32;
      int32_t dot = 0;
#if defined(__AVX2__)
      __m256i um = _mm256_set1_epi8((char)(int8_t)(uint8_t)(1u << ib));
      __m256i qv = _mm256_loadu_si256((const __m256i *)qs_sub);
      __m256i lo = _mm256_and_si256((ib & 1) ? _mm256_srli_epi16(qv, 4) : qv, m4);
      __m256i qhv = _mm256_loadu_si256((const __m256i *)qh);
      __m256i hi = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhv, um), um), hi16);
      __m256i wu = _mm256_or_si256(lo, hi);
      dot = _vira_avx2_hsum_i32(_mm256_madd_epi16(
          _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)xq)),
          ones));
#elif defined(__SSSE3__)
      __m128i um = _mm_set1_epi8((char)(int8_t)(uint8_t)(1u << ib));
      __m128i w0 = _mm_loadu_si128((const __m128i *)qs_sub);
      __m128i w1 = _mm_loadu_si128((const __m128i *)(qs_sub + 16));
      __m128i lo0 = _mm_and_si128((ib & 1) ? _mm_srli_epi16(w0, 4) : w0, m4);
      __m128i lo1 = _mm_and_si128((ib & 1) ? _mm_srli_epi16(w1, 4) : w1, m4);
      __m128i qh0 = _mm_loadu_si128((const __m128i *)qh);
      __m128i qh1 = _mm_loadu_si128((const __m128i *)(qh + 16));
      __m128i hi0 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh0, um), um), hi16);
      __m128i hi1 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh1, um), um), hi16);
      __m128i p0 = _mm_maddubs_epi16(_mm_or_si128(lo0, hi0), _mm_loadu_si128((const __m128i *)xq));
      __m128i p1 = _mm_maddubs_epi16(_mm_or_si128(lo1, hi1), _mm_loadu_si128((const __m128i *)(xq + 16)));
      dot = _vira_sse_hsum_i32(_mm_madd_epi16(_mm_add_epi16(p0, p1), ones));
#else
      int32_t shift = (ib & 1) * 4;
      uint8_t um = (uint8_t)(1u << ib);
      for (int32_t l = 0; l < 32; ++l) {
        int32_t q = (int32_t)((qs_sub[l] >> shift) & 0x0Fu) + ((qh[l] & um) ? 16 : 0);
        dot += q * (int32_t)xq[l];
      }
#endif
      sumi_d += (int32_t)sc * dot;
      sumi_m += (int32_t)m_sc * ((int32_t)x[b].bsums[2 * ib] + (int32_t)x[b].bsums[2 * ib + 1]);
    }
    sumf += x[b].d * (d * (float)sumi_d - dmin * (float)sumi_m);
  }
  return sumf;
}

static inline float _vira_cpu_dot_q6_k_q8_K(const uint8_t *w_row,
                                            const _vira_block_q8_K *x,
                                            int32_t nb) {
  float sumf = 0.0f;
#if defined(__AVX2__)
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i m3 = _mm256_set1_epi8(0x03);
  const __m128i ones128 = _mm_set1_epi16(1);
#elif defined(__SSSE3__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i m3 = _mm_set1_epi8(0x03);
  const __m128i ones = _mm_set1_epi16(1);
#elif defined(__ARM_NEON) || defined(__aarch64__)
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
  const uint8x16_t m3 = vdupq_n_u8(0x03);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 210u;
    uint16_t dh = 0;
    memcpy(&dh, blk + 208, 2);
    float d = _vira_fp16_to_fp32(dh);
    int32_t sumi = 0;
    for (int32_t h = 0; h < 2; ++h) {
      const uint8_t *ql_h = blk + h * 64;
      const uint8_t *qh_h = blk + 128 + h * 32;
      const int8_t *sc_h = (const int8_t *)(blk + 192) + h * 8;
      const int8_t *q_h = x[b].qs + h * 128;
      const int16_t *bs_h = x[b].bsums + h * 8;
#if defined(__AVX2__)
      __m256i qh32 = _mm256_loadu_si256((const __m256i *)qh_h);
      __m256i ql_a = _mm256_loadu_si256((const __m256i *)ql_h);
      __m256i ql_b = _mm256_loadu_si256((const __m256i *)(ql_h + 32));
      for (int32_t c = 0; c < 4; ++c) {
        __m256i ql_use = (c & 1) ? ql_b : ql_a;
        __m256i lo4 = _mm256_and_si256((c >= 2) ? _mm256_srli_epi16(ql_use, 4) : ql_use, m4);
        __m256i hi2 = _mm256_slli_epi16(_mm256_and_si256((c == 0) ? qh32 : _mm256_srli_epi16(qh32, c * 2), m3), 4);
        __m256i wu = _mm256_or_si256(lo4, hi2);
        __m256i p16 = _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)(q_h + c * 32)));
        __m128i s0 = _mm_madd_epi16(_mm256_castsi256_si128(p16), ones128);
        __m128i s1 = _mm_madd_epi16(_mm256_extracti128_si256(p16, 1), ones128);
        __m128i hab = _mm_hadd_epi32(s0, s1);
        hab = _mm_hadd_epi32(hab, hab);
        int32_t dot0 = _mm_cvtsi128_si32(hab);
        int32_t dot1 = _mm_cvtsi128_si32(_mm_srli_si128(hab, 4));
        sumi += (int32_t)sc_h[2 * c + 0] * (dot0 - 32 * (int32_t)bs_h[2 * c + 0]);
        sumi += (int32_t)sc_h[2 * c + 1] * (dot1 - 32 * (int32_t)bs_h[2 * c + 1]);
      }
#elif defined(__SSSE3__)
      __m128i qh0 = _mm_loadu_si128((const __m128i *)qh_h);
      __m128i qh1 = _mm_loadu_si128((const __m128i *)(qh_h + 16));
      __m128i ql0a = _mm_loadu_si128((const __m128i *)ql_h);
      __m128i ql1a = _mm_loadu_si128((const __m128i *)(ql_h + 16));
      __m128i ql0b = _mm_loadu_si128((const __m128i *)(ql_h + 32));
      __m128i ql1b = _mm_loadu_si128((const __m128i *)(ql_h + 48));
      for (int32_t c = 0; c < 4; ++c) {
        __m128i qla = (c & 1) ? ql0b : ql0a;
        __m128i qlb = (c & 1) ? ql1b : ql1a;
        __m128i lo0 = _mm_and_si128((c >= 2) ? _mm_srli_epi16(qla, 4) : qla, m4);
        __m128i lo1 = _mm_and_si128((c >= 2) ? _mm_srli_epi16(qlb, 4) : qlb, m4);
        __m128i hi0 = _mm_slli_epi16(_mm_and_si128((c == 0) ? qh0 : _mm_srli_epi16(qh0, c * 2), m3), 4);
        __m128i hi1 = _mm_slli_epi16(_mm_and_si128((c == 0) ? qh1 : _mm_srli_epi16(qh1, c * 2), m3), 4);
        __m128i p0 = _mm_maddubs_epi16(_mm_or_si128(lo0, hi0), _mm_loadu_si128((const __m128i *)(q_h + c * 32)));
        __m128i p1 = _mm_maddubs_epi16(_mm_or_si128(lo1, hi1), _mm_loadu_si128((const __m128i *)(q_h + c * 32 + 16)));
        __m128i hab = _mm_hadd_epi32(_mm_madd_epi16(p0, ones), _mm_madd_epi16(p1, ones));
        hab = _mm_hadd_epi32(hab, hab);
        int32_t dot0 = _mm_cvtsi128_si32(hab);
        int32_t dot1 = _mm_cvtsi128_si32(_mm_srli_si128(hab, 4));
        sumi += (int32_t)sc_h[2 * c + 0] * (dot0 - 32 * (int32_t)bs_h[2 * c + 0]);
        sumi += (int32_t)sc_h[2 * c + 1] * (dot1 - 32 * (int32_t)bs_h[2 * c + 1]);
      }
#elif defined(__ARM_NEON) || defined(__aarch64__)
      uint8x16_t qh0 = vld1q_u8(qh_h), qh1 = vld1q_u8(qh_h + 16);
      uint8x16_t ql0a = vld1q_u8(ql_h), ql1a = vld1q_u8(ql_h + 16);
      uint8x16_t ql0b = vld1q_u8(ql_h + 32), ql1b = vld1q_u8(ql_h + 48);
      for (int32_t c = 0; c < 4; ++c) {
        uint8x16_t qla = (c & 1) ? ql0b : ql0a;
        uint8x16_t qlb = (c & 1) ? ql1b : ql1a;
        uint8x16_t lo0 = (c >= 2) ? vshrq_n_u8(qla, 4) : vandq_u8(qla, m4);
        uint8x16_t lo1 = (c >= 2) ? vshrq_n_u8(qlb, 4) : vandq_u8(qlb, m4);
        uint8x16_t hr0 = (c == 0) ? qh0 : ((c == 1) ? vshrq_n_u8(qh0, 2) : ((c == 2) ? vshrq_n_u8(qh0, 4) : vshrq_n_u8(qh0, 6)));
        uint8x16_t hr1 = (c == 0) ? qh1 : ((c == 1) ? vshrq_n_u8(qh1, 2) : ((c == 2) ? vshrq_n_u8(qh1, 4) : vshrq_n_u8(qh1, 6)));
        int8x16_t u0 = vreinterpretq_s8_u8(vorrq_u8(lo0, vshlq_n_u8(vandq_u8(hr0, m3), 4)));
        int8x16_t u1 = vreinterpretq_s8_u8(vorrq_u8(lo1, vshlq_n_u8(vandq_u8(hr1, m3), 4)));
        int32_t dot0 = _vira_neon_hsum_i32(_vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(q_h + c * 32)));
        int32_t dot1 = _vira_neon_hsum_i32(_vira_neon_dot_s8(vdupq_n_s32(0), u1, vld1q_s8(q_h + c * 32 + 16)));
        sumi += (int32_t)sc_h[2 * c + 0] * (dot0 - 32 * (int32_t)bs_h[2 * c + 0]);
        sumi += (int32_t)sc_h[2 * c + 1] * (dot1 - 32 * (int32_t)bs_h[2 * c + 1]);
      }
#else
      for (int32_t l = 0; l < 32; ++l) {
        int32_t is = l >> 4;
        int32_t q1 = (int32_t)((ql_h[l + 0] & 0x0Fu) | (((qh_h[l] >> 0) & 3u) << 4)) - 32;
        int32_t q2 = (int32_t)((ql_h[l + 32] & 0x0Fu) | (((qh_h[l] >> 2) & 3u) << 4)) - 32;
        int32_t q3 = (int32_t)((ql_h[l + 0] >> 4) | (((qh_h[l] >> 4) & 3u) << 4)) - 32;
        int32_t q4 = (int32_t)((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3u) << 4)) - 32;
        sumi += (int32_t)sc_h[is + 0] * q1 * (int32_t)q_h[l + 0];
        sumi += (int32_t)sc_h[is + 2] * q2 * (int32_t)q_h[l + 32];
        sumi += (int32_t)sc_h[is + 4] * q3 * (int32_t)q_h[l + 64];
        sumi += (int32_t)sc_h[is + 6] * q4 * (int32_t)q_h[l + 96];
      }
      (void)bs_h;
#endif
    }
    sumf += d * x[b].d * (float)sumi;
  }
  return sumf;
}

/* Q2_K / Q3_K . Q8_K: each 256-wide super-block is dequantized to floats
 * and dotted with the int8 activations (scalar, auto-vectorized). */
static inline void _vira_rt_dequant_blocks_to_fp32(uint32_t gguf_dtype, const uint8_t *raw_q,
                                                   size_t start_block, size_t n_blocks, float *dst_out);
static inline float _vira_cpu_dot_kq_dequant_q8_K(uint32_t dt, const uint8_t *w_row,
                                                  const _vira_block_q8_K *x, int32_t nb) {
  float w[256];
  float sum = 0.0f;
  for (int32_t b = 0; b < nb; ++b) {
    _vira_rt_dequant_blocks_to_fp32(dt, w_row, (size_t)b, 1, w);
    float acc = 0.0f;
    for (int32_t i = 0; i < 256; ++i) acc += w[i] * (float)x[b].qs[i];
    sum += acc * x[b].d;
  }
  return sum;
}

static inline float _vira_cpu_dot_bf16_f32(const uint8_t *w_row, const float *x, int32_t K) {
  float acc = 0.0f;
  for (int32_t k = 0; k < K; ++k) {
    uint16_t h;
    memcpy(&h, w_row + (size_t)k * 2u, 2);
    uint32_t bits = (uint32_t)h << 16;
    float wf;
    memcpy(&wf, &bits, 4);
    acc += wf * x[k];
  }
  return acc;
}

static inline void _vira_cpu_dot_4rows_q6_k_q8_K(
    const uint8_t *w_row, const _vira_block_q8_K *x0,
    const _vira_block_q8_K *x1, const _vira_block_q8_K *x2,
    const _vira_block_q8_K *x3, int32_t nb, float *out4) {
  float sf0 = 0.0f, sf1 = 0.0f, sf2 = 0.0f, sf3 = 0.0f;
#if defined(__AVX2__)
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i m3 = _mm256_set1_epi8(0x03);
  const __m128i ones128 = _mm_set1_epi16(1);
#elif defined(__SSSE3__)
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m128i m3 = _mm_set1_epi8(0x03);
  const __m128i ones = _mm_set1_epi16(1);
#elif defined(__ARM_NEON) || defined(__aarch64__)
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
  const uint8x16_t m3 = vdupq_n_u8(0x03);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 210u;
    uint16_t dh = 0;
    memcpy(&dh, blk + 208, 2);
    float d = _vira_fp16_to_fp32(dh);
    int32_t si0 = 0, si1 = 0, si2 = 0, si3 = 0;
    for (int32_t h = 0; h < 2; ++h) {
      const uint8_t *ql_h = blk + h * 64;
      const uint8_t *qh_h = blk + 128 + h * 32;
      const int8_t *sc_h = (const int8_t *)(blk + 192) + h * 8;
      int32_t qoff = h * 128, boff = h * 8;
#if defined(__AVX2__)
      __m256i qh32 = _mm256_loadu_si256((const __m256i *)qh_h);
      __m256i ql_a = _mm256_loadu_si256((const __m256i *)ql_h);
      __m256i ql_b = _mm256_loadu_si256((const __m256i *)(ql_h + 32));
      for (int32_t c = 0; c < 4; ++c) {
        __m256i ql_use = (c & 1) ? ql_b : ql_a;
        __m256i lo4 = _mm256_and_si256((c >= 2) ? _mm256_srli_epi16(ql_use, 4) : ql_use, m4);
        __m256i hi2 = _mm256_slli_epi16(_mm256_and_si256((c == 0) ? qh32 : _mm256_srli_epi16(qh32, c * 2), m3), 4);
        __m256i wu = _mm256_or_si256(lo4, hi2);
        const _vira_block_q8_K *xs[4] = {&x0[b], &x1[b], &x2[b], &x3[b]};
        int32_t *sis[4] = {&si0, &si1, &si2, &si3};
        for (int32_t r = 0; r < 4; ++r) {
          __m256i p16 = _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)(xs[r]->qs + qoff + c * 32)));
          __m128i hab = _mm_hadd_epi32(_mm_madd_epi16(_mm256_castsi256_si128(p16), ones128), _mm_madd_epi16(_mm256_extracti128_si256(p16, 1), ones128));
          hab = _mm_hadd_epi32(hab, hab);
          int32_t d0 = _mm_cvtsi128_si32(hab);
          int32_t d1 = _mm_cvtsi128_si32(_mm_srli_si128(hab, 4));
          *(sis[r]) += (int32_t)sc_h[2 * c + 0] * (d0 - 32 * (int32_t)xs[r]->bsums[boff + 2 * c + 0]) +
                       (int32_t)sc_h[2 * c + 1] * (d1 - 32 * (int32_t)xs[r]->bsums[boff + 2 * c + 1]);
        }
      }
#elif defined(__SSSE3__)
      __m128i qh0 = _mm_loadu_si128((const __m128i *)qh_h);
      __m128i qh1 = _mm_loadu_si128((const __m128i *)(qh_h + 16));
      __m128i ql0a = _mm_loadu_si128((const __m128i *)ql_h);
      __m128i ql1a = _mm_loadu_si128((const __m128i *)(ql_h + 16));
      __m128i ql0b = _mm_loadu_si128((const __m128i *)(ql_h + 32));
      __m128i ql1b = _mm_loadu_si128((const __m128i *)(ql_h + 48));
      for (int32_t c = 0; c < 4; ++c) {
        __m128i qla = (c & 1) ? ql0b : ql0a, qlb = (c & 1) ? ql1b : ql1a;
        __m128i u0 = _mm_or_si128(_mm_and_si128((c >= 2) ? _mm_srli_epi16(qla, 4) : qla, m4),
                                  _mm_slli_epi16(_mm_and_si128((c == 0) ? qh0 : _mm_srli_epi16(qh0, c * 2), m3), 4));
        __m128i u1 = _mm_or_si128(_mm_and_si128((c >= 2) ? _mm_srli_epi16(qlb, 4) : qlb, m4),
                                  _mm_slli_epi16(_mm_and_si128((c == 0) ? qh1 : _mm_srli_epi16(qh1, c * 2), m3), 4));
        const _vira_block_q8_K *xs[4] = {&x0[b], &x1[b], &x2[b], &x3[b]};
        int32_t *sis[4] = {&si0, &si1, &si2, &si3};
        for (int32_t r = 0; r < 4; ++r) {
          __m128i p0 = _mm_maddubs_epi16(u0, _mm_loadu_si128((const __m128i *)(xs[r]->qs + qoff + c * 32)));
          __m128i p1 = _mm_maddubs_epi16(u1, _mm_loadu_si128((const __m128i *)(xs[r]->qs + qoff + c * 32 + 16)));
          __m128i hab = _mm_hadd_epi32(_mm_madd_epi16(p0, ones), _mm_madd_epi16(p1, ones));
          hab = _mm_hadd_epi32(hab, hab);
          int32_t d0 = _mm_cvtsi128_si32(hab);
          int32_t d1 = _mm_cvtsi128_si32(_mm_srli_si128(hab, 4));
          *(sis[r]) += (int32_t)sc_h[2 * c + 0] * (d0 - 32 * (int32_t)xs[r]->bsums[boff + 2 * c + 0]) +
                       (int32_t)sc_h[2 * c + 1] * (d1 - 32 * (int32_t)xs[r]->bsums[boff + 2 * c + 1]);
        }
      }
#elif defined(__ARM_NEON) || defined(__aarch64__)
      uint8x16_t qh0 = vld1q_u8(qh_h), qh1 = vld1q_u8(qh_h + 16);
      uint8x16_t ql0a = vld1q_u8(ql_h), ql1a = vld1q_u8(ql_h + 16);
      uint8x16_t ql0b = vld1q_u8(ql_h + 32), ql1b = vld1q_u8(ql_h + 48);
      for (int32_t c = 0; c < 4; ++c) {
        uint8x16_t qla = (c & 1) ? ql0b : ql0a, qlb = (c & 1) ? ql1b : ql1a;
        uint8x16_t lo0 = (c >= 2) ? vshrq_n_u8(qla, 4) : vandq_u8(qla, m4);
        uint8x16_t lo1 = (c >= 2) ? vshrq_n_u8(qlb, 4) : vandq_u8(qlb, m4);
        uint8x16_t hr0 = (c == 0) ? qh0 : ((c == 1) ? vshrq_n_u8(qh0, 2) : ((c == 2) ? vshrq_n_u8(qh0, 4) : vshrq_n_u8(qh0, 6)));
        uint8x16_t hr1 = (c == 0) ? qh1 : ((c == 1) ? vshrq_n_u8(qh1, 2) : ((c == 2) ? vshrq_n_u8(qh1, 4) : vshrq_n_u8(qh1, 6)));
        int8x16_t u0 = vreinterpretq_s8_u8(vorrq_u8(lo0, vshlq_n_u8(vandq_u8(hr0, m3), 4)));
        int8x16_t u1 = vreinterpretq_s8_u8(vorrq_u8(lo1, vshlq_n_u8(vandq_u8(hr1, m3), 4)));
        const _vira_block_q8_K *xs[4] = {&x0[b], &x1[b], &x2[b], &x3[b]};
        int32_t *sis[4] = {&si0, &si1, &si2, &si3};
        for (int32_t r = 0; r < 4; ++r) {
          int32_t d0 = _vira_neon_hsum_i32(_vira_neon_dot_s8(vdupq_n_s32(0), u0, vld1q_s8(xs[r]->qs + qoff + c * 32)));
          int32_t d1 = _vira_neon_hsum_i32(_vira_neon_dot_s8(vdupq_n_s32(0), u1, vld1q_s8(xs[r]->qs + qoff + c * 32 + 16)));
          *(sis[r]) += (int32_t)sc_h[2 * c + 0] * (d0 - 32 * (int32_t)xs[r]->bsums[boff + 2 * c + 0]) +
                       (int32_t)sc_h[2 * c + 1] * (d1 - 32 * (int32_t)xs[r]->bsums[boff + 2 * c + 1]);
        }
      }
#else
      for (int32_t l = 0; l < 32; ++l) {
        int32_t is = l >> 4;
        int32_t q1 = (int32_t)((ql_h[l + 0] & 0x0Fu) | (((qh_h[l] >> 0) & 3u) << 4)) - 32;
        int32_t q2 = (int32_t)((ql_h[l + 32] & 0x0Fu) | (((qh_h[l] >> 2) & 3u) << 4)) - 32;
        int32_t q3 = (int32_t)((ql_h[l + 0] >> 4) | (((qh_h[l] >> 4) & 3u) << 4)) - 32;
        int32_t q4 = (int32_t)((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3u) << 4)) - 32;
        si0 += (int32_t)sc_h[is + 0] * q1 * x0[b].qs[qoff + l] + (int32_t)sc_h[is + 2] * q2 * x0[b].qs[qoff + l + 32] +
               (int32_t)sc_h[is + 4] * q3 * x0[b].qs[qoff + l + 64] + (int32_t)sc_h[is + 6] * q4 * x0[b].qs[qoff + l + 96];
        si1 += (int32_t)sc_h[is + 0] * q1 * x1[b].qs[qoff + l] + (int32_t)sc_h[is + 2] * q2 * x1[b].qs[qoff + l + 32] +
               (int32_t)sc_h[is + 4] * q3 * x1[b].qs[qoff + l + 64] + (int32_t)sc_h[is + 6] * q4 * x1[b].qs[qoff + l + 96];
        si2 += (int32_t)sc_h[is + 0] * q1 * x2[b].qs[qoff + l] + (int32_t)sc_h[is + 2] * q2 * x2[b].qs[qoff + l + 32] +
               (int32_t)sc_h[is + 4] * q3 * x2[b].qs[qoff + l + 64] + (int32_t)sc_h[is + 6] * q4 * x2[b].qs[qoff + l + 96];
        si3 += (int32_t)sc_h[is + 0] * q1 * x3[b].qs[qoff + l] + (int32_t)sc_h[is + 2] * q2 * x3[b].qs[qoff + l + 32] +
               (int32_t)sc_h[is + 4] * q3 * x3[b].qs[qoff + l + 64] + (int32_t)sc_h[is + 6] * q4 * x3[b].qs[qoff + l + 96];
      }
#endif
    }
    sf0 += d * x0[b].d * (float)si0;
    sf1 += d * x1[b].d * (float)si1;
    sf2 += d * x2[b].d * (float)si2;
    sf3 += d * x3[b].d * (float)si3;
  }
  out4[0] = sf0; out4[1] = sf1; out4[2] = sf2; out4[3] = sf3;
}

static inline float _vira_cpu_dot_f16_f32(const uint8_t *w_row,
                                          const float *x_row, int32_t K) {
  float sumf = 0.0f;
  int32_t k = 0;
#if defined(__AVX2__) && defined(__F16C__)
  __m256 vs = _mm256_setzero_ps();
  for (; k + 8 <= K; k += 8) {
    __m128i wh = _mm_loadu_si128((const __m128i *)(w_row + (size_t)k * 2u));
    __m256 wf = _mm256_cvtph_ps(wh);
    __m256 xf = _mm256_loadu_ps(&x_row[k]);
#if defined(__FMA__)
    vs = _mm256_fmadd_ps(wf, xf, vs);
#else
    vs = _mm256_add_ps(vs, _mm256_mul_ps(wf, xf));
#endif
  }
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(vs), _mm256_extractf128_ps(vs, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  sumf = _mm_cvtss_f32(s);
#endif
  for (; k < K; ++k) {
    uint16_t h = 0;
    memcpy(&h, w_row + (size_t)k * 2u, 2);
    sumf += _vira_fp16_to_fp32(h) * x_row[k];
  }
  return sumf;
}

static inline void _vira_cpu_dot_4rows_f16_f32(const uint8_t *w_row,
                                               const float *x0, const float *x1,
                                               const float *x2, const float *x3,
                                               int32_t K, float *out4) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  int32_t k = 0;
#if defined(__AVX2__) && defined(__F16C__)
  __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
  __m256 v2 = _mm256_setzero_ps(), v3 = _mm256_setzero_ps();
  for (; k + 8 <= K; k += 8) {
    __m256 wf = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w_row + (size_t)k * 2u)));
#if defined(__FMA__)
    v0 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(&x0[k]), v0);
    v1 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(&x1[k]), v1);
    v2 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(&x2[k]), v2);
    v3 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(&x3[k]), v3);
#else
    v0 = _mm256_add_ps(v0, _mm256_mul_ps(wf, _mm256_loadu_ps(&x0[k])));
    v1 = _mm256_add_ps(v1, _mm256_mul_ps(wf, _mm256_loadu_ps(&x1[k])));
    v2 = _mm256_add_ps(v2, _mm256_mul_ps(wf, _mm256_loadu_ps(&x2[k])));
    v3 = _mm256_add_ps(v3, _mm256_mul_ps(wf, _mm256_loadu_ps(&x3[k])));
#endif
  }
  __m128 a0 = _mm_add_ps(_mm256_castps256_ps128(v0), _mm256_extractf128_ps(v0, 1));
  __m128 a1 = _mm_add_ps(_mm256_castps256_ps128(v1), _mm256_extractf128_ps(v1, 1));
  __m128 a2 = _mm_add_ps(_mm256_castps256_ps128(v2), _mm256_extractf128_ps(v2, 1));
  __m128 a3 = _mm_add_ps(_mm256_castps256_ps128(v3), _mm256_extractf128_ps(v3, 1));
  a0 = _mm_hadd_ps(a0, a0); a0 = _mm_hadd_ps(a0, a0); s0 = _mm_cvtss_f32(a0);
  a1 = _mm_hadd_ps(a1, a1); a1 = _mm_hadd_ps(a1, a1); s1 = _mm_cvtss_f32(a1);
  a2 = _mm_hadd_ps(a2, a2); a2 = _mm_hadd_ps(a2, a2); s2 = _mm_cvtss_f32(a2);
  a3 = _mm_hadd_ps(a3, a3); a3 = _mm_hadd_ps(a3, a3); s3 = _mm_cvtss_f32(a3);
#endif
  for (; k < K; ++k) {
    uint16_t h = 0;
    memcpy(&h, w_row + (size_t)k * 2u, 2);
    float wf = _vira_fp16_to_fp32(h);
    s0 += wf * x0[k]; s1 += wf * x1[k]; s2 += wf * x2[k]; s3 += wf * x3[k];
  }
  out4[0] = s0; out4[1] = s1; out4[2] = s2; out4[3] = s3;
}

typedef struct {
  uint32_t gguf_dtype;
  const uint8_t *w_raw;
  const void *x_q8;
  const float *x_f32;
  const float *bias;
  float *y_base;
  int32_t M;
  int32_t K;
  int32_t N;
  int32_t nb;
  size_t row_bytes;
  int32_t n_start;
  int32_t n_end;
  bool accum;
} _vira_quant_linear_job;

static inline void _vira_q_write1(float *dst, int32_t n, float val,
                                  const float *bias, bool accum) {
  if (bias) val += bias[n];
  if (accum) dst[n] += val;
  else dst[n] = val;
}

static inline void _vira_q_write4_n(float *dst, int32_t n, const float *acc4,
                                    const float *bias, bool accum) {
  float v0 = acc4[0], v1 = acc4[1], v2 = acc4[2], v3 = acc4[3];
  if (bias) {
    v0 += bias[n + 0];
    v1 += bias[n + 1];
    v2 += bias[n + 2];
    v3 += bias[n + 3];
  }
  if (accum) {
    dst[n + 0] += v0;
    dst[n + 1] += v1;
    dst[n + 2] += v2;
    dst[n + 3] += v3;
  } else {
    dst[n + 0] = v0;
    dst[n + 1] = v1;
    dst[n + 2] = v2;
    dst[n + 3] = v3;
  }
}

static inline void _vira_q_write4_m(float *y0, float *y1, float *y2, float *y3,
                                    int32_t n, const float *acc4,
                                    const float *bias, bool accum) {
  float bval = bias ? bias[n] : 0.0f;
  float v0 = acc4[0] + bval;
  float v1 = acc4[1] + bval;
  float v2 = acc4[2] + bval;
  float v3 = acc4[3] + bval;
  if (accum) {
    y0[n] += v0;
    y1[n] += v1;
    y2[n] += v2;
    y3[n] += v3;
  } else {
    y0[n] = v0;
    y1[n] = v1;
    y2[n] = v2;
    y3[n] = v3;
  }
}

static void _vira_quant_linear_job_fn(void *raw_env) {
  _vira_quant_linear_job *j = (_vira_quant_linear_job *)raw_env;
  const int32_t M = j->M, K = j->K, N = j->N, nb = j->nb;
  const int32_t n0 = j->n_start, n1 = j->n_end;
  const size_t rb = j->row_bytes;
  const uint32_t dt = j->gguf_dtype;
  const float *bias = j->bias;
  const bool accum = j->accum;

  if (n0 >= n1 || M <= 0) return;

  if (dt == 30) {
    for (int32_t m = 0; m < M; ++m) {
      const float *xr = j->x_f32 + (size_t)m * (size_t)K;
      float *yr = j->y_base + (size_t)m * (size_t)N;
      for (int32_t n = n0; n < n1; ++n)
        _vira_q_write1(yr, n, _vira_cpu_dot_bf16_f32(j->w_raw + (size_t)n * rb, xr, K), bias, accum);
    }
    return;
  }

  if (dt == 1) {
    int32_t m = 0;
    for (; m + 4 <= M; m += 4) {
      const float *xr0 = j->x_f32 + (size_t)(m + 0) * (size_t)K;
      const float *xr1 = j->x_f32 + (size_t)(m + 1) * (size_t)K;
      const float *xr2 = j->x_f32 + (size_t)(m + 2) * (size_t)K;
      const float *xr3 = j->x_f32 + (size_t)(m + 3) * (size_t)K;
      float *y0 = j->y_base + (size_t)(m + 0) * (size_t)N;
      float *y1 = j->y_base + (size_t)(m + 1) * (size_t)N;
      float *y2 = j->y_base + (size_t)(m + 2) * (size_t)N;
      float *y3 = j->y_base + (size_t)(m + 3) * (size_t)N;
      for (int32_t n = n0; n < n1; ++n) {
        float acc4[4];
        _vira_cpu_dot_4rows_f16_f32(j->w_raw + (size_t)n * rb, xr0, xr1, xr2, xr3, K, acc4);
        _vira_q_write4_m(y0, y1, y2, y3, n, acc4, bias, accum);
      }
    }
    for (; m < M; ++m) {
      const float *xr = j->x_f32 + (size_t)m * (size_t)K;
      float *yr = j->y_base + (size_t)m * (size_t)N;
      for (int32_t n = n0; n < n1; ++n) {
        float v = _vira_cpu_dot_f16_f32(j->w_raw + (size_t)n * rb, xr, K);
        _vira_q_write1(yr, n, v, bias, accum);
      }
    }
    return;
  }

  if (dt <= 8) {
    const _vira_block_q8_0 *xq = (const _vira_block_q8_0 *)j->x_q8;
    if (dt == 2 || dt == 8) {
      if (M == 1) {
        float *y0 = j->y_base;
        int32_t n = n0;
        for (; n + 4 <= n1; n += 4) {
          const uint8_t *w0 = j->w_raw + (size_t)(n + 0) * rb;
          const uint8_t *w1 = j->w_raw + (size_t)(n + 1) * rb;
          const uint8_t *w2 = j->w_raw + (size_t)(n + 2) * rb;
          const uint8_t *w3 = j->w_raw + (size_t)(n + 3) * rb;
          float acc4[4];
          if (dt == 2) _vira_cpu_dot_4w_q4_0_q8_0(w0, w1, w2, w3, xq, nb, acc4);
          else         _vira_cpu_dot_4w_q8_0_q8_0(w0, w1, w2, w3, xq, nb, acc4);
          _vira_q_write4_n(y0, n, acc4, bias, accum);
        }
        for (; n < n1; ++n) {
          const uint8_t *wr = j->w_raw + (size_t)n * rb;
          float v = (dt == 2) ? _vira_cpu_dot_q4_0_q8_0(wr, xq, nb)
                              : _vira_cpu_dot_q8_0_q8_0(wr, xq, nb);
          _vira_q_write1(y0, n, v, bias, accum);
        }
        return;
      }

      int32_t n_blk = (nb <= 64) ? 64 : ((nb <= 192) ? 32 : 16);
      for (int32_t nb_s = n0; nb_s < n1; nb_s += n_blk) {
        int32_t nb_e = nb_s + n_blk;
        if (nb_e > n1) nb_e = n1;
        int32_t m = 0;
        for (; m + 4 <= M; m += 4) {
          const _vira_block_q8_0 *x0 = xq + (size_t)(m + 0) * (size_t)nb;
          const _vira_block_q8_0 *x1 = xq + (size_t)(m + 1) * (size_t)nb;
          const _vira_block_q8_0 *x2 = xq + (size_t)(m + 2) * (size_t)nb;
          const _vira_block_q8_0 *x3 = xq + (size_t)(m + 3) * (size_t)nb;
          float *y0 = j->y_base + (size_t)(m + 0) * (size_t)N;
          float *y1 = j->y_base + (size_t)(m + 1) * (size_t)N;
          float *y2 = j->y_base + (size_t)(m + 2) * (size_t)N;
          float *y3 = j->y_base + (size_t)(m + 3) * (size_t)N;
          int32_t n = nb_s;
          for (; n + 2 <= nb_e; n += 2) {
            const uint8_t *w0 = j->w_raw + (size_t)(n + 0) * rb;
            const uint8_t *w1 = j->w_raw + (size_t)(n + 1) * rb;
            float a0[4], a1[4];
            if (dt == 2) {
              _vira_cpu_dot_2w4m_q4_0_q8_0(w0, w1, x0, x1, x2, x3, nb, a0, a1);
            } else {
              _vira_cpu_dot_2w4m_q8_0_q8_0(w0, w1, x0, x1, x2, x3, nb, a0, a1);
            }
            _vira_q_write4_m(y0, y1, y2, y3, n + 0, a0, bias, accum);
            _vira_q_write4_m(y0, y1, y2, y3, n + 1, a1, bias, accum);
          }
          for (; n < nb_e; ++n) {
            const uint8_t *wr = j->w_raw + (size_t)n * rb;
            float acc4[4];
            if (dt == 2) _vira_cpu_dot_4rows_q4_0_q8_0(wr, x0, x1, x2, x3, nb, acc4);
            else         _vira_cpu_dot_4rows_q8_0_q8_0(wr, x0, x1, x2, x3, nb, acc4);
            _vira_q_write4_m(y0, y1, y2, y3, n, acc4, bias, accum);
          }
        }
        for (; m < M; ++m) {
          const _vira_block_q8_0 *xm = xq + (size_t)m * (size_t)nb;
          float *yr = j->y_base + (size_t)m * (size_t)N;
          int32_t n = nb_s;
          for (; n + 4 <= nb_e; n += 4) {
            const uint8_t *w0 = j->w_raw + (size_t)(n + 0) * rb;
            const uint8_t *w1 = j->w_raw + (size_t)(n + 1) * rb;
            const uint8_t *w2 = j->w_raw + (size_t)(n + 2) * rb;
            const uint8_t *w3 = j->w_raw + (size_t)(n + 3) * rb;
            float acc4[4];
            if (dt == 2) _vira_cpu_dot_4w_q4_0_q8_0(w0, w1, w2, w3, xm, nb, acc4);
            else         _vira_cpu_dot_4w_q8_0_q8_0(w0, w1, w2, w3, xm, nb, acc4);
            _vira_q_write4_n(yr, n, acc4, bias, accum);
          }
          for (; n < nb_e; ++n) {
            const uint8_t *wr = j->w_raw + (size_t)n * rb;
            float v = (dt == 2) ? _vira_cpu_dot_q4_0_q8_0(wr, xm, nb)
                                : _vira_cpu_dot_q8_0_q8_0(wr, xm, nb);
            _vira_q_write1(yr, n, v, bias, accum);
          }
        }
      }
      return;
    }

    for (int32_t m = 0; m < M; ++m) {
      const _vira_block_q8_0 *xm = xq + (size_t)m * (size_t)nb;
      float *yr = j->y_base + (size_t)m * (size_t)N;
      for (int32_t n = n0; n < n1; ++n) {
        const uint8_t *wr = j->w_raw + (size_t)n * rb;
        float v = 0.0f;
        if (dt == 3)      v = _vira_cpu_dot_q4_1_q8_0(wr, xm, nb);
        else if (dt == 6) v = _vira_cpu_dot_q5_0_q8_0(wr, xm, nb);
        else              v = _vira_cpu_dot_q5_1_q8_0(wr, xm, nb);
        _vira_q_write1(yr, n, v, bias, accum);
      }
    }
  } else {
    const _vira_block_q8_K *xq = (const _vira_block_q8_K *)j->x_q8;
    int32_t m = 0;
    if (dt == 12 || dt == 14) {
      for (; m + 4 <= M; m += 4) {
        const _vira_block_q8_K *x0 = xq + (size_t)(m + 0) * (size_t)nb;
        const _vira_block_q8_K *x1 = xq + (size_t)(m + 1) * (size_t)nb;
        const _vira_block_q8_K *x2 = xq + (size_t)(m + 2) * (size_t)nb;
        const _vira_block_q8_K *x3 = xq + (size_t)(m + 3) * (size_t)nb;
        float *y0 = j->y_base + (size_t)(m + 0) * (size_t)N;
        float *y1 = j->y_base + (size_t)(m + 1) * (size_t)N;
        float *y2 = j->y_base + (size_t)(m + 2) * (size_t)N;
        float *y3 = j->y_base + (size_t)(m + 3) * (size_t)N;
        for (int32_t n = n0; n < n1; ++n) {
          const uint8_t *wr = j->w_raw + (size_t)n * rb;
          float acc4[4];
          if (dt == 12) _vira_cpu_dot_4rows_q4_k_q8_K(wr, x0, x1, x2, x3, nb, acc4);
          else          _vira_cpu_dot_4rows_q6_k_q8_K(wr, x0, x1, x2, x3, nb, acc4);
          _vira_q_write4_m(y0, y1, y2, y3, n, acc4, bias, accum);
        }
      }
    }
    for (; m < M; ++m) {
      const _vira_block_q8_K *xm = xq + (size_t)m * (size_t)nb;
      float *yr = j->y_base + (size_t)m * (size_t)N;
      for (int32_t n = n0; n < n1; ++n) {
        const uint8_t *wr = j->w_raw + (size_t)n * rb;
        float v = 0.0f;
        if (dt == 12)      v = _vira_cpu_dot_q4_k_q8_K(wr, xm, nb);
        else if (dt == 13) v = _vira_cpu_dot_q5_k_q8_K(wr, xm, nb);
        else if (dt == 14) v = _vira_cpu_dot_q6_k_q8_K(wr, xm, nb);
        else               v = _vira_cpu_dot_kq_dequant_q8_K(dt, wr, xm, nb);
        _vira_q_write1(yr, n, v, bias, accum);
      }
    }
  }
}

typedef struct {
  vira_tensor_f32 w;
  const float *bias;
  float *y_base;
  bool accum;
} _vira_multi_qmat_desc;

typedef struct {
  uint32_t dtype;
  const void *raw_q;
  const float *bias;
  float *y_base;
  int32_t N;
  int32_t nb;
  size_t row_bytes;
  bool accum;
} _vira_multi_qmat_resolved;

typedef struct {
  const float *x_f32;
  const void *x_q8_0;
  const void *x_q8_K;
  int32_t M;
  int32_t K;
  int32_t n_mats;
  int32_t total_N;
  _vira_multi_qmat_resolved mats[3];
} _vira_multi_qctx;

static void _vira_multi_qlinear_team_fn(void *raw_ctx, int32_t tid, int32_t nt) {
  const _vira_multi_qctx *ctx = (const _vira_multi_qctx *)raw_ctx;
  int32_t total = ctx->total_N;
  int32_t base = total / nt;
  int32_t extra = total % nt;
  int32_t g_start = tid * base + (tid < extra ? tid : extra);
  int32_t g_end = g_start + base + (tid < extra ? 1 : 0);
  if (g_start >= g_end) return;

  int32_t off = 0;
  for (int32_t i = 0; i < ctx->n_mats; ++i) {
    const _vira_multi_qmat_resolved *m = &ctx->mats[i];
    int32_t m_end = off + m->N;
    int32_t s = g_start > off ? (g_start - off) : 0;
    int32_t e = g_end < m_end ? (g_end - off) : m->N;
    if (s < e) {
      _vira_quant_linear_job job = {
          .gguf_dtype = m->dtype,
          .w_raw = (const uint8_t *)m->raw_q,
          .x_q8 = (m->dtype <= 8) ? ctx->x_q8_0 : ctx->x_q8_K,
          .x_f32 = ctx->x_f32,
          .bias = m->bias,
          .y_base = m->y_base,
          .M = ctx->M,
          .K = ctx->K,
          .N = m->N,
          .nb = m->nb,
          .row_bytes = m->row_bytes,
          .n_start = s,
          .n_end = e,
          .accum = m->accum,
      };
      _vira_quant_linear_job_fn(&job);
    }
    off = m_end;
  }
}

static inline bool _vira_rt_quant_cpu_multi_linear_write(
    const float *x_data, int32_t M, int32_t K, int32_t n_mats,
    const _vira_multi_qmat_desc *descs, void *scratch_q8,
    size_t scratch_q8_cap) {
  if (!x_data || M <= 0 || K <= 0 || n_mats <= 0 || n_mats > 3 || !descs) {
    return false;
  }
  _vira_multi_qctx ctx = {0};
  ctx.x_f32 = x_data;
  ctx.M = M;
  ctx.K = K;
  ctx.n_mats = n_mats;

  bool need_q8_0 = false;
  bool need_q8_K = false;
  int32_t total_N = 0;
  for (int32_t i = 0; i < n_mats; ++i) {
    vira_tensor_f32 w = descs[i].w;
    if (!w.data || w.cols != K || _vira_gpu_is_ptr_dirty(w.data)) {
      return false;
    }
    uint32_t dtype = 0;
    const void *raw_q = NULL;
    size_t raw_bytes = 0, elems = 0;
    if (!_vira_rt_quant_try_get(w.data, &dtype, &raw_q, &raw_bytes, &elems)) {
      return false;
    }
    int32_t N = w.rows;
    size_t qk = _vira_gguf_dtype_qk(dtype);
    size_t bs = _vira_gguf_dtype_bs(dtype);
    if (qk == 0 || bs == 0 || elems != (size_t)N * (size_t)K ||
        ((size_t)K % qk) != 0) {
      return false;
    }
    int32_t nb = (int32_t)((size_t)K / qk);
    size_t row_bytes = (size_t)nb * bs;
    if ((size_t)N * row_bytes > raw_bytes) {
      return false;
    }
    if (!(dtype == 1 || dtype == 2 || dtype == 3 || dtype == 6 || dtype == 7 ||
          dtype == 8 || dtype == 10 || dtype == 11 || dtype == 12 || dtype == 13 || dtype == 14 || dtype == 30)) {
      return false;
    }
    if (dtype != 1 && dtype != 30) {
      if (qk == 32) need_q8_0 = true;
      else need_q8_K = true;
    }
    ctx.mats[i] = (_vira_multi_qmat_resolved){
        .dtype = dtype,
        .raw_q = raw_q,
        .bias = descs[i].bias,
        .y_base = descs[i].y_base,
        .N = N,
        .nb = nb,
        .row_bytes = row_bytes,
        .accum = descs[i].accum,
    };
    total_N += N;
  }
  ctx.total_N = total_N;

  uint8_t stack_q8_0[16384];
  void *heap_q8_0 = NULL;
  void *heap_q8_K = NULL;

  if (need_q8_0) {
    int32_t nb32 = K / 32;
    size_t need = (size_t)M * (size_t)nb32 * sizeof(_vira_block_q8_0);
    void *buf = NULL;
    if (scratch_q8 != NULL && !need_q8_K && need <= scratch_q8_cap) {
      buf = scratch_q8;
    } else if (need <= sizeof(stack_q8_0)) {
      buf = stack_q8_0;
    } else {
      heap_q8_0 = malloc(need);
      if (!heap_q8_0) return false;
      buf = heap_q8_0;
    }
    for (int32_t m = 0; m < M; ++m) {
      _vira_quantize_row_q8_0(x_data + (size_t)m * (size_t)K, K,
                              (_vira_block_q8_0 *)buf + (size_t)m * (size_t)nb32);
    }
    ctx.x_q8_0 = buf;
  }
  if (need_q8_K) {
    int32_t nb256 = K / 256;
    size_t need = (size_t)M * (size_t)nb256 * sizeof(_vira_block_q8_K);
    void *buf = NULL;
    if (scratch_q8 != NULL && !need_q8_0 && need <= scratch_q8_cap) {
      buf = scratch_q8;
    } else {
      heap_q8_K = malloc(need);
      if (!heap_q8_K) {
        if (heap_q8_0) free(heap_q8_0);
        return false;
      }
      buf = heap_q8_K;
    }
    for (int32_t m = 0; m < M; ++m) {
      _vira_quantize_row_q8_K(x_data + (size_t)m * (size_t)K, K,
                              (_vira_block_q8_K *)buf + (size_t)m * (size_t)nb256);
    }
    ctx.x_q8_K = buf;
  }

  int64_t total_ops = (int64_t)M * (int64_t)total_N * (int64_t)K;
  if (total_ops >= 32768LL && total_N >= 32) {
    _vira_team_run(_vira_multi_qlinear_team_fn, &ctx, total_N);
  } else {
    _vira_multi_qlinear_team_fn(&ctx, 0, 1);
  }

  if (heap_q8_0) free(heap_q8_0);
  if (heap_q8_K) free(heap_q8_K);
  return true;
}

static inline bool _vira_rt_quant_cpu_linear(vira_arena *opt_arena,
                                             vira_tensor_f32 x,
                                             vira_tensor_f32 w,
                                             vira_tensor_f32 *out_tensor) {
  if (!out_tensor || _vira_gpu_is_ptr_dirty(w.data))
    return false;
  uint32_t dtype = 0;
  const void *raw_q = NULL;
  size_t raw_bytes = 0, elems = 0;
  if (!_vira_rt_quant_try_get(w.data, &dtype, &raw_q, &raw_bytes, &elems))
    return false;
  int32_t M = x.rows, K = x.cols, N = w.rows;
  size_t qk = _vira_gguf_dtype_qk(dtype);
  size_t bs = _vira_gguf_dtype_bs(dtype);
  if (qk == 0 || bs == 0 || elems != (size_t)N * (size_t)K ||
      ((size_t)K % qk) != 0)
    return false;
  int32_t nb = (int32_t)((size_t)K / qk);
  size_t row_bytes = (size_t)nb * bs;
  if ((size_t)N * row_bytes > raw_bytes)
    return false;
  if (!(dtype == 1 || dtype == 2 || dtype == 3 || dtype == 6 || dtype == 7 ||
        dtype == 8 || dtype == 10 || dtype == 11 || dtype == 12 || dtype == 13 || dtype == 14 || dtype == 30))
    return false;

  _vira_tensor_sync_host_f32(x.data);

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? _vira_tensor_f32_arena_alloc_uninit_nd(opt_arena, 2, out_dims)
                : _vira_tensor_f32_alloc_uninit_nd(2, out_dims);

  _vira_multi_qmat_desc desc = {
      .w = w,
      .bias = NULL,
      .y_base = out.data,
      .accum = false,
  };
  if (!_vira_rt_quant_cpu_multi_linear_write(x.data, M, K, 1, &desc, NULL, 0)) {
    if (!opt_arena) vira_tensor_f32_free(&out);
    return false;
  }
  *out_tensor = out;
  return true;
}

#endif // VIRA_RT_TENSOR_QUANT_SIMD_H
