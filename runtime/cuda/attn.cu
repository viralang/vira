// Attention over an FP16 KV cache, and the f32 <-> f16 row conversions it needs.
// Source of runtime/vira_cuda_attn_ptx.h (python3 runtime/cuda/build_ptx.py).
//
// Cache layout as the f32 kernels: K and V [ctx, kv_dim] row-major, kv_dim = n_kv_heads *
// head_dim, query head h reads kv head h / kv_mul. Stored as IEEE half (cvt.rn.f16.f32), so
// a cache costs half the memory of the f32 one.

#define WARP 32
#define AT_THREADS 256
#define AT_WARPS (AT_THREADS / WARP)
#define AT_MAX_D 8 // head_dim / 32 values per lane: head_dim <= 256
#define AT_NEG_INF __int_as_float((int)0xff800000)

#ifdef VL_EMULATE
#include "emulate.h"
#else
#define VL_SYNC() __syncthreads()
__device__ __forceinline__ float vl_h2f(unsigned short h) {
  float f;
  asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short vl_f2h(float f) {
  unsigned short h;
  asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ float vl_wsum(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
#endif

// dst[(row0 + r) * cols + c] = half(src[r * cols + c]) for r < rows. Grid (n + 255) / 256.
extern "C" __global__ void vira_cu_kv_store_f16(const float *__restrict__ src, unsigned short *__restrict__ dst,
                                               long long row0, int cols, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[row0 * cols + i] = vl_f2h(src[i]);
}
// dst[i] = float(src[i]) for i < n (cache read back to the host).
extern "C" __global__ void vira_cu_kv_load_f32(const unsigned short *__restrict__ src, float *__restrict__ dst,
                                              long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = vl_h2f(src[i]);
}

// out[t, h, :] = softmax_j(scale * q[t, h] . K[j, kvh]) V[j, kvh] over j <= start + t (causal),
// q / out rows t * n_heads + h (decode: one row of n_heads). One block per (head, query row):
// grid (n_heads, M), block 256. Each warp walks every 8th key with an online softmax (its own
// running max and sum), lanes hold head_dim / 32 dimensions; warps merge in shared memory.
extern "C" __global__ void __launch_bounds__(AT_THREADS)
vira_cu_attn_f16kv(const float *__restrict__ q, const unsigned short *__restrict__ kc,
                   const unsigned short *__restrict__ vc, float *__restrict__ out, int n_heads, int kv_mul,
                   int head_dim, int kv_dim, int M, long long start, float scale) {
  __shared__ float s_m[AT_WARPS], s_l[AT_WARPS];
  __shared__ float s_acc[AT_WARPS][AT_MAX_D * WARP];
  const int h = blockIdx.x, t = blockIdx.y;
  if (h >= n_heads || t >= M) return; // whole block
  const int tid = threadIdx.x, lane = tid & (WARP - 1), wp = tid / WARP;
  const int nd = (head_dim + WARP - 1) / WARP; // values per lane
  const int kvh = h / kv_mul;
  const float *qr = q + ((size_t)t * n_heads + h) * head_dim;
  float qv[AT_MAX_D], acc[AT_MAX_D];
#pragma unroll
  for (int i = 0; i < AT_MAX_D; i++) {
    const int d = lane + WARP * i;
    qv[i] = (i < nd && d < head_dim) ? qr[d] * scale : 0.0f;
    acc[i] = 0.0f;
  }
  float m = AT_NEG_INF, l = 0.0f;
  const long long n_keys = start + t + 1;
  for (long long j = wp; j < n_keys; j += AT_WARPS) {
    const unsigned short *kr = kc + (size_t)j * kv_dim + (size_t)kvh * head_dim;
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < AT_MAX_D; i++) {
      const int d = lane + WARP * i;
      if (i < nd && d < head_dim) s += qv[i] * vl_h2f(kr[d]);
    }
    s = vl_wsum(s);
    const float mn = fmaxf(m, s);
    const float corr = __expf(m - mn), p = __expf(s - mn);
    l = l * corr + p;
    const unsigned short *vr = vc + (size_t)j * kv_dim + (size_t)kvh * head_dim;
#pragma unroll
    for (int i = 0; i < AT_MAX_D; i++) {
      const int d = lane + WARP * i;
      if (i < nd && d < head_dim) acc[i] = acc[i] * corr + p * vl_h2f(vr[d]);
    }
    m = mn;
  }
  if (lane == 0) {
    s_m[wp] = m;
    s_l[wp] = l;
  }
#pragma unroll
  for (int i = 0; i < AT_MAX_D; i++) s_acc[wp][lane + WARP * i] = acc[i];
  VL_SYNC();
  if (wp == 0) {
    float M_ = AT_NEG_INF;
    for (int w = 0; w < AT_WARPS; w++) M_ = fmaxf(M_, s_m[w]);
    float L = 0.0f, o[AT_MAX_D];
#pragma unroll
    for (int i = 0; i < AT_MAX_D; i++) o[i] = 0.0f;
    for (int w = 0; w < AT_WARPS; w++) {
      if (s_l[w] == 0.0f) continue; // a warp with no keys (n_keys < 8)
      const float c = __expf(s_m[w] - M_);
      L += s_l[w] * c;
#pragma unroll
      for (int i = 0; i < AT_MAX_D; i++) o[i] += s_acc[w][lane + WARP * i] * c;
    }
    float *orow = out + ((size_t)t * n_heads + h) * head_dim;
    const float inv = 1.0f / L;
#pragma unroll
    for (int i = 0; i < AT_MAX_D; i++) {
      const int d = lane + WARP * i;
      if (i < nd && d < head_dim) orow[d] = o[i] * inv;
    }
  }
}

// dst[i] = 0 for i < n (32-bit words): a new FP16 cache starts at zero without an upload.
extern "C" __global__ void vira_cu_kv_zero(unsigned int *__restrict__ dst, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = 0u;
}
