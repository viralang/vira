// Shared by mmq.cu (dp4a) and mmqt.cu (int8 tensor cores): the per-format decoders of the
// prompt kernels. For a 256-value slice of a weight row: NAME_q gives int i (values
// 4i..4i+3, signed bytes), NAME_s the slice's scales a[16] (one per 16 values) and biases
// c[8] (one per 32 values), so that value v = a[v / 16] * q[v] + c[v / 32].
#ifndef VL_MMQ_FMT_CUH
#define VL_MMQ_FMT_CUH

#ifdef VL_EMULATE
#include "emulate.h"
#else
#define VL_SYNC() __syncthreads()
__device__ __forceinline__ int vl_dp4a(int a, int b, int c) {
  int d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ int vl_load_int_b2(const unsigned char *p) {
  const unsigned short *s = (const unsigned short *)p;
  return (int)((unsigned int)__ldg(s) | ((unsigned int)__ldg(s + 1) << 16));
}
__device__ __forceinline__ float vl_half_to_float(unsigned short h) {
  float f;
  asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
  return f;
}
#endif

__device__ __forceinline__ float vl_h(const unsigned char *p) { return vl_half_to_float(__ldg((const unsigned short *)p)); }
__device__ __forceinline__ int vl_ld4(const unsigned char *p) { return __ldg((const int *)p); }
// bytes q in 0..127 minus o (o in 0..127) as signed bytes, no borrow between bytes
__device__ __forceinline__ int vl_sub_bytes(int q, unsigned int o4) {
  return (int)((((unsigned int)q | 0x80808080u) - o4) ^ 0x80808080u);
}
__device__ __forceinline__ int vl_qh4(unsigned int qh, int base) { // bits base..base+3 -> bit 4 of bytes
  const unsigned int b = (qh >> base) & 0xFu;
  return (int)(((b & 1u) << 4) | ((b & 2u) << 11) | ((b & 4u) << 18) | ((b & 8u) << 25));
}
__device__ __forceinline__ void vl_k4sm(const unsigned char *sc, int j, int *s, int *m) {
  if (j < 4) {
    *s = __ldg(sc + j) & 63;
    *m = __ldg(sc + j + 4) & 63;
  } else {
    *s = (__ldg(sc + j + 4) & 0xF) | ((__ldg(sc + j - 4) >> 6) << 4);
    *m = (__ldg(sc + j + 4) >> 4) | ((__ldg(sc + j) >> 6) << 4);
  }
}

// ---- per format: int i (values 4i..4i+3) of the 256-value slice starting at unit u of a
// row; and the slice's scales a[16] (per 16 values) and bias c[8] (per 32 values) ----
// 32-value formats: the slice is 8 blocks, block u + i / 8, int i % 8 of it.
#define VL_LEGACY(NAME, BYTES, QOFF, BODY, SC)                                               \
  __device__ __forceinline__ int NAME##_q(const unsigned char *row, int u, int i) {          \
    const unsigned char *b = row + (size_t)(u + (i >> 3)) * (BYTES);                         \
    const int ii = i & 7;                                                                    \
    BODY                                                                                     \
  }                                                                                          \
  __device__ __forceinline__ void NAME##_s(const unsigned char *row, int u, float *a, float *c) { \
    _Pragma("unroll") for (int j = 0; j < 8; j++) {                                          \
      const unsigned char *b = row + (size_t)(u + j) * (BYTES);                              \
      SC                                                                                     \
    }                                                                                        \
  }
// Q4_0: values 0..15 low nibbles of qs, 16..31 high; v = d * (q - 8)
VL_LEGACY(mq_q4_0, 18, 2,
  const int q = (vl_load_int_b2(b + 2 + 4 * (ii & 3)) >> ((ii >> 2) * 4)) & 0x0F0F0F0F;
  return vl_sub_bytes(q, 0x08080808u);,
  a[2 * j] = a[2 * j + 1] = vl_h(b); c[j] = 0.0f;)
// Q4_1: v = d * q + m
VL_LEGACY(mq_q4_1, 20, 4,
  return (vl_ld4(b + 4 + 4 * (ii & 3)) >> ((ii >> 2) * 4)) & 0x0F0F0F0F;,
  a[2 * j] = a[2 * j + 1] = vl_h(b); c[j] = vl_h(b + 2);)
// Q5_0: v = d * ((nibble | qh bit << 4) - 16)
VL_LEGACY(mq_q5_0, 22, 6,
  const int q = ((vl_load_int_b2(b + 6 + 4 * (ii & 3)) >> ((ii >> 2) * 4)) & 0x0F0F0F0F) |
                vl_qh4((unsigned int)vl_load_int_b2(b + 2), 4 * ii);
  return vl_sub_bytes(q, 0x10101010u);,
  a[2 * j] = a[2 * j + 1] = vl_h(b); c[j] = 0.0f;)
// Q5_1: v = d * (nibble | qh bit << 4) + m
VL_LEGACY(mq_q5_1, 24, 8,
  return ((vl_ld4(b + 8 + 4 * (ii & 3)) >> ((ii >> 2) * 4)) & 0x0F0F0F0F) |
         vl_qh4((unsigned int)vl_ld4(b + 4), 4 * ii);,
  a[2 * j] = a[2 * j + 1] = vl_h(b); c[j] = vl_h(b + 2);)
// Q8_0: v = d * q
VL_LEGACY(mq_q8_0, 34, 2,
  return vl_load_int_b2(b + 2 + 4 * ii);,
  a[2 * j] = a[2 * j + 1] = vl_h(b); c[j] = 0.0f;)

// K-quants: the slice is one super-block (unit u); sub-block j = i / 8 holds values 32j..
// Q4_K (144 B): d, dmin, scales[12], qs[128]; sub-block j = nibble (j & 1) of qs[32 (j/2)..]
__device__ __forceinline__ int mq_q4_k_q(const unsigned char *row, int u, int i) {
  const unsigned char *b = row + (size_t)u * 144u;
  const int j = i >> 3, ii = i & 7;
  return (vl_ld4(b + 16 + 32 * (j >> 1) + 4 * ii) >> (4 * (j & 1))) & 0x0F0F0F0F;
}
__device__ __forceinline__ void mq_q4_k_s(const unsigned char *row, int u, float *a, float *c) {
  const unsigned char *b = row + (size_t)u * 144u;
  const float d = vl_h(b), dm = vl_h(b + 2);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int s, m;
    vl_k4sm(b + 4, j, &s, &m);
    a[2 * j] = a[2 * j + 1] = d * (float)s;
    c[j] = -dm * (float)m;
  }
}
// Q5_K (176 B): d, dmin, scales[12], qh[32], qs[128]; bit j of qh[l] is bit 4 of value l
__device__ __forceinline__ int mq_q5_k_q(const unsigned char *row, int u, int i) {
  const unsigned char *b = row + (size_t)u * 176u;
  const int j = i >> 3, ii = i & 7;
  const int lo = (vl_ld4(b + 48 + 32 * (j >> 1) + 4 * ii) >> (4 * (j & 1))) & 0x0F0F0F0F;
  const int hi = ((vl_ld4(b + 16 + 4 * ii) >> j) & 0x01010101) << 4;
  return lo | hi;
}
__device__ __forceinline__ void mq_q5_k_s(const unsigned char *row, int u, float *a, float *c) {
  const unsigned char *b = row + (size_t)u * 176u;
  const float d = vl_h(b), dm = vl_h(b + 2);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int s, m;
    vl_k4sm(b + 4, j, &s, &m);
    a[2 * j] = a[2 * j + 1] = d * (float)s;
    c[j] = -dm * (float)m;
  }
}
// Q6_K (210 B): ql[128], qh[64], int8 scales[16], d at 208. Sub-block j: half ip = j / 4,
// quarter k = j % 4; value l (0..31): ql[64 ip + 32 (k & 1) + l] nibble (k >> 1), bits 2k of
// qh[32 ip + l]; v = d * scales[8 ip + 2k + l / 16] * (q - 32)
__device__ __forceinline__ int mq_q6_k_q(const unsigned char *row, int u, int i) {
  const unsigned char *b = row + (size_t)u * 210u;
  const int j = i >> 3, ii = i & 7, ip = j >> 2, k = j & 3;
  const int lo = (vl_load_int_b2(b + 64 * ip + 32 * (k & 1) + 4 * ii) >> (4 * (k >> 1))) & 0x0F0F0F0F;
  const int hi = ((vl_load_int_b2(b + 128 + 32 * ip + 4 * ii) >> (2 * k)) & 0x03030303) << 4;
  return vl_sub_bytes(lo | hi, 0x20202020u);
}
__device__ __forceinline__ void mq_q6_k_s(const unsigned char *row, int u, float *a, float *c) {
  const unsigned char *b = row + (size_t)u * 210u;
  const float d = vl_h(b + 208);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const int ip = j >> 2, k = j & 3;
    a[2 * j] = d * (float)(signed char)__ldg(b + 192 + 8 * ip + 2 * k);
    a[2 * j + 1] = d * (float)(signed char)__ldg(b + 192 + 8 * ip + 2 * k + 1);
    c[j] = 0.0f;
  }
}

#endif
