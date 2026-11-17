// Fused small kernels for decode. Source of runtime/vira_cuda_fused_ptx.h
// (python3 runtime/cuda/build_ptx.py; same toolchain notes as mmvq2.cu).

#define WARP 32

#ifdef VL_EMULATE
#include "emulate.h"
#else
__device__ __forceinline__ float vl_rsqrt_approx(float x) {
  float r;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float vl_fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ float vl_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float vl_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float vl_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float vl_warp_max(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ int vl_warp_sum_i(int v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
#define VL_SYNC() __syncthreads()
#endif

// RMSNorm of one row (block of 256 threads per row, grid = rows) that also writes the row
// as Q8_1 blocks {f32 d, f32 d * sum(q), int8 q[32]}, so the next matrix-vector product
// does not need its own quantize launch. Both halves repeat the arithmetic of the separate
// kernels exactly (vira_cu_rmsnorm_f32: strided fma sum, shared-memory tree, sum / cols,
// + eps, rsqrt.approx, (r * x) * w; vira_cu_quantize_q8_1: warp absmax, q = rni(v * (127 /
// amax)) clamped to +-127, d = amax * (1 / 127)), so results do not change. cols % 32 == 0.
extern "C" __global__ void __launch_bounds__(256)
vira_cu_rmsnorm_q8_1_f32(const float *__restrict__ x, const float *__restrict__ w,
                         float *__restrict__ y, unsigned char *__restrict__ q8, int cols, float eps) {
  __shared__ float sdata[256];
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const float *xr = x + (size_t)row * (size_t)cols;
  float *yr = y + (size_t)row * (size_t)cols;
  float acc = 0.0f;
  for (int i = tid; i < cols; i += 256) {
    const float v = __ldg(xr + i);
    acc = vl_fma(v, v, acc);
  }
  sdata[tid] = acc;
  VL_SYNC();
  for (int o = 128; o > 0; o >>= 1) {
    if (tid < o) sdata[tid] = vl_add(sdata[tid + o], sdata[tid]);
    VL_SYNC();
  }
  const float r = vl_rsqrt_approx(vl_add(eps, vl_div(sdata[0], (float)cols)));
  // Q8_1: warp k quantises blocks k, k + 8, ...; lane l owns value 32 * blk + l
  const int lane = tid & (WARP - 1);
  const int nblk = cols / 32;
  unsigned char *qr = q8 + (size_t)row * (size_t)nblk * 40u;
  for (int blk = tid / WARP; blk < nblk; blk += 256 / WARP) {
    const int i = blk * 32 + lane;
    const float v = vl_mul(vl_mul(r, __ldg(xr + i)), __ldg(w + i));
    yr[i] = v;
    const float amax = vl_warp_max(fabsf(v));
    int q = 0;
    if (amax > 0.0f) {
      q = __float2int_rn(vl_mul(v, vl_div(127.0f, amax)));
      q = max(min(q, 127), -127);
    }
    const int sum = vl_warp_sum_i(q);
    unsigned char *qb = qr + (size_t)blk * 40u;
    qb[8 + lane] = (unsigned char)(signed char)q;
    if (lane == 0) {
      const float d = vl_mul(amax, __int_as_float(0x3C010204));
      ((float *)qb)[0] = d;
      ((float *)qb)[1] = vl_mul(d, (float)sum);
    }
  }
}
