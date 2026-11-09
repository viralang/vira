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

static inline void _vira_quantize_row_q8_0(const float *x, int32_t K,
                                           _vira_block_q8_0 *y) {
  int32_t nb = K / 32;
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

#if defined(__AVX2__)
static inline int32_t _vira_avx2_hsum_i32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_hadd_epi32(s, s);
  s = _mm_hadd_epi32(s, s);
  return _mm_cvtsi128_si32(s);
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

static inline float _vira_cpu_dot_q4_0_q8_0(const uint8_t *w_row,
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
    const uint8_t *blk = w_row + (size_t)b * 18u;
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float d = _vira_fp16_to_fp32(dh);
    int32_t dot = 0;
#if defined(__AVX2__)
    __m128i wq = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m256i wu = _mm256_and_si256(
        _mm256_inserti128_si256(_mm256_castsi128_si256(wq),
                                _mm_srli_epi16(wq, 4), 1),
        m4);
    __m256i qv = _mm256_loadu_si256((const __m256i *)x[b].qs);
    dot = _vira_avx2_hsum_i32(
        _mm256_madd_epi16(_mm256_maddubs_epi16(wu, qv), ones));
#elif defined(__SSSE3__)
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
}

static inline void _vira_cpu_dot_4rows_q4_0_q8_0(
    const uint8_t *w_row, const _vira_block_q8_0 *x0,
    const _vira_block_q8_0 *x1, const _vira_block_q8_0 *x2,
    const _vira_block_q8_0 *x3, int32_t nb, float *out4) {
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
    const uint8_t *blk = w_row + (size_t)b * 18u;
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float d = _vira_fp16_to_fp32(dh);
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if defined(__AVX2__)
    __m128i wq = _mm_loadu_si128((const __m128i *)(blk + 2));
    __m256i wu = _mm256_and_si256(
        _mm256_inserti128_si256(_mm256_castsi128_si256(wq),
                                _mm_srli_epi16(wq, 4), 1),
        m4);
    d0 = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x0[b].qs)),
        ones));
    d1 = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x1[b].qs)),
        ones));
    d2 = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x2[b].qs)),
        ones));
    d3 = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(wu, _mm256_loadu_si256((const __m256i *)x3[b].qs)),
        ones));
#elif defined(__SSSE3__)
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
  float sumf = 0.0f;
#if defined(__AVX2__)
  const __m256i ones = _mm256_set1_epi16(1);
#elif defined(__SSSE3__)
  const __m128i ones = _mm_set1_epi16(1);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 34u;
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float d = _vira_fp16_to_fp32(dh);
    int32_t dot = 0;
#if defined(__AVX2__)
    __m256i wv = _mm256_loadu_si256((const __m256i *)(blk + 2));
    __m256i qv = _mm256_loadu_si256((const __m256i *)x[b].qs);
    __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(qv, wv));
    dot = _vira_avx2_hsum_i32(_mm256_madd_epi16(p, ones));
#elif defined(__SSSE3__)
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
}

static inline void _vira_cpu_dot_4rows_q8_0_q8_0(
    const uint8_t *w_row, const _vira_block_q8_0 *x0,
    const _vira_block_q8_0 *x1, const _vira_block_q8_0 *x2,
    const _vira_block_q8_0 *x3, int32_t nb, float *out4) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#if defined(__AVX2__)
  const __m256i ones = _mm256_set1_epi16(1);
#elif defined(__SSSE3__)
  const __m128i ones = _mm_set1_epi16(1);
#endif
  for (int32_t b = 0; b < nb; ++b) {
    const uint8_t *blk = w_row + (size_t)b * 34u;
    uint16_t dh = 0;
    memcpy(&dh, blk, 2);
    float d = _vira_fp16_to_fp32(dh);
#if defined(__AVX2__)
    __m256i wv = _mm256_loadu_si256((const __m256i *)(blk + 2));
    __m256i aw = _mm256_abs_epi8(wv);
    int32_t d0 = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x0[b].qs), wv)), ones));
    int32_t d1 = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x1[b].qs), wv)), ones));
    int32_t d2 = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x2[b].qs), wv)), ones));
    int32_t d3 = _vira_avx2_hsum_i32(_mm256_madd_epi16(
        _mm256_maddubs_epi16(aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)x3[b].qs), wv)), ones));
#elif defined(__SSSE3__)
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
  float *y_base;
  int32_t M;
  int32_t K;
  int32_t N;
  int32_t nb;
  size_t row_bytes;
  int32_t n_start;
  int32_t n_end;
} _vira_quant_linear_job;

static void _vira_quant_linear_job_fn(void *raw_env) {
  _vira_quant_linear_job *j = (_vira_quant_linear_job *)raw_env;
  const int32_t M = j->M, K = j->K, N = j->N, nb = j->nb;
  const int32_t n0 = j->n_start, n1 = j->n_end;
  const size_t rb = j->row_bytes;
  const uint32_t dt = j->gguf_dtype;

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
        y0[n] = acc4[0]; y1[n] = acc4[1]; y2[n] = acc4[2]; y3[n] = acc4[3];
      }
    }
    for (; m < M; ++m) {
      const float *xr = j->x_f32 + (size_t)m * (size_t)K;
      float *yr = j->y_base + (size_t)m * (size_t)N;
      for (int32_t n = n0; n < n1; ++n) {
        yr[n] = _vira_cpu_dot_f16_f32(j->w_raw + (size_t)n * rb, xr, K);
      }
    }
    return;
  }

  if (dt <= 8) {
    const _vira_block_q8_0 *xq = (const _vira_block_q8_0 *)j->x_q8;
    int32_t m = 0;
    if (dt == 2 || dt == 8) {
      for (; m + 4 <= M; m += 4) {
        const _vira_block_q8_0 *x0 = xq + (size_t)(m + 0) * (size_t)nb;
        const _vira_block_q8_0 *x1 = xq + (size_t)(m + 1) * (size_t)nb;
        const _vira_block_q8_0 *x2 = xq + (size_t)(m + 2) * (size_t)nb;
        const _vira_block_q8_0 *x3 = xq + (size_t)(m + 3) * (size_t)nb;
        float *y0 = j->y_base + (size_t)(m + 0) * (size_t)N;
        float *y1 = j->y_base + (size_t)(m + 1) * (size_t)N;
        float *y2 = j->y_base + (size_t)(m + 2) * (size_t)N;
        float *y3 = j->y_base + (size_t)(m + 3) * (size_t)N;
        for (int32_t n = n0; n < n1; ++n) {
          const uint8_t *wr = j->w_raw + (size_t)n * rb;
          float acc4[4];
          if (dt == 2) _vira_cpu_dot_4rows_q4_0_q8_0(wr, x0, x1, x2, x3, nb, acc4);
          else         _vira_cpu_dot_4rows_q8_0_q8_0(wr, x0, x1, x2, x3, nb, acc4);
          y0[n] = acc4[0]; y1[n] = acc4[1]; y2[n] = acc4[2]; y3[n] = acc4[3];
        }
      }
    }
    for (; m < M; ++m) {
      const _vira_block_q8_0 *xm = xq + (size_t)m * (size_t)nb;
      float *yr = j->y_base + (size_t)m * (size_t)N;
      for (int32_t n = n0; n < n1; ++n) {
        const uint8_t *wr = j->w_raw + (size_t)n * rb;
        if (dt == 2)      yr[n] = _vira_cpu_dot_q4_0_q8_0(wr, xm, nb);
        else if (dt == 3) yr[n] = _vira_cpu_dot_q4_1_q8_0(wr, xm, nb);
        else if (dt == 6) yr[n] = _vira_cpu_dot_q5_0_q8_0(wr, xm, nb);
        else if (dt == 7) yr[n] = _vira_cpu_dot_q5_1_q8_0(wr, xm, nb);
        else              yr[n] = _vira_cpu_dot_q8_0_q8_0(wr, xm, nb);
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
          y0[n] = acc4[0]; y1[n] = acc4[1]; y2[n] = acc4[2]; y3[n] = acc4[3];
        }
      }
    }
    for (; m < M; ++m) {
      const _vira_block_q8_K *xm = xq + (size_t)m * (size_t)nb;
      float *yr = j->y_base + (size_t)m * (size_t)N;
      for (int32_t n = n0; n < n1; ++n) {
        const uint8_t *wr = j->w_raw + (size_t)n * rb;
        if (dt == 12)      yr[n] = _vira_cpu_dot_q4_k_q8_K(wr, xm, nb);
        else if (dt == 13) yr[n] = _vira_cpu_dot_q5_k_q8_K(wr, xm, nb);
        else               yr[n] = _vira_cpu_dot_q6_k_q8_K(wr, xm, nb);
      }
    }
  }
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
        dtype == 8 || dtype == 12 || dtype == 13 || dtype == 14))
    return false;

  _vira_tensor_sync_host_f32(x.data);

  uint8_t stack_q8[16384];
  void *heap_q8 = NULL;
  void *x_q8 = NULL;
  if (dtype != 1) {
    size_t blk_sz = (qk == 32) ? sizeof(_vira_block_q8_0) : sizeof(_vira_block_q8_K);
    size_t need = (size_t)M * (size_t)nb * blk_sz;
    if (need <= sizeof(stack_q8)) {
      x_q8 = stack_q8;
    } else {
      heap_q8 = malloc(need);
      if (!heap_q8)
        return false;
      x_q8 = heap_q8;
    }
    for (int32_t m = 0; m < M; ++m) {
      const float *xr = x.data + (size_t)m * (size_t)K;
      if (qk == 32) {
        _vira_quantize_row_q8_0(xr, K,
                                (_vira_block_q8_0 *)x_q8 + (size_t)m * (size_t)nb);
      } else {
        _vira_quantize_row_q8_K(xr, K,
                                (_vira_block_q8_K *)x_q8 + (size_t)m * (size_t)nb);
      }
    }
  }

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? _vira_tensor_f32_arena_alloc_uninit_nd(opt_arena, 2, out_dims)
                : _vira_tensor_f32_alloc_uninit_nd(2, out_dims);

  int64_t total_ops = (int64_t)M * (int64_t)N * (int64_t)K;
  if (total_ops >= 131072LL && N >= 32 && !_vira_in_active_actor()) {
    vira_scheduler_ensure_init();
    int32_t nw = vira_scheduler_num_workers();
    if (nw > 16) nw = 16;
    if (nw > N) nw = N;
    if (nw > 1) {
      _vira_quant_linear_job jobs[16];
      vira_task tasks[16];
      int32_t base_n = N / nw;
      int32_t extra = N % nw;
      int32_t cur = 0;
      for (int32_t wi = 0; wi < nw; ++wi) {
        int32_t cnt = base_n + (wi < extra ? 1 : 0);
        jobs[wi] = (_vira_quant_linear_job){.gguf_dtype = dtype,
                                            .w_raw = (const uint8_t *)raw_q,
                                            .x_q8 = x_q8,
                                            .x_f32 = x.data,
                                            .y_base = out.data,
                                            .M = M,
                                            .K = K,
                                            .N = N,
                                            .nb = nb,
                                            .row_bytes = row_bytes,
                                            .n_start = cur,
                                            .n_end = cur + cnt};
        cur += cnt;
        if (wi > 0) {
          tasks[wi] = (vira_task){.fn = _vira_quant_linear_job_fn,
                                  .env = &jobs[wi],
                                  .next_actor_msg = NULL,
                                  .done = 0};
          vira_scheduler_submit_async(&tasks[wi]);
        }
      }
      _vira_quant_linear_job_fn(&jobs[0]);
      for (int32_t wi = 1; wi < nw; ++wi) {
        vira_scheduler_await(&tasks[wi]);
      }
      if (heap_q8) free(heap_q8);
      *out_tensor = out;
      return true;
    }
  }

  _vira_quant_linear_job single = {.gguf_dtype = dtype,
                                   .w_raw = (const uint8_t *)raw_q,
                                   .x_q8 = x_q8,
                                   .x_f32 = x.data,
                                   .y_base = out.data,
                                   .M = M,
                                   .K = K,
                                   .N = N,
                                   .nb = nb,
                                   .row_bytes = row_bytes,
                                   .n_start = 0,
                                   .n_end = N};
  _vira_quant_linear_job_fn(&single);
  if (heap_q8) free(heap_q8);
  *out_tensor = out;
  return true;
}

#endif // VIRA_RT_TENSOR_QUANT_SIMD_H
