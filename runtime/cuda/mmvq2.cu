// Matrix x vector kernels for decode (M <= 4 rows of activations), quantized weights.
//
// Source of runtime/vira_cuda_mmvq2_ptx.h. Regenerate after editing:
//   python3 runtime/cuda/build_ptx.py
// (NVRTC 11.8, target compute_61: dp4a; the PTX loads on any driver from CUDA 11.8 up,
// and the driver compiles it for the GPU it runs on.)
//
// Layout (one CUDA block per output row, 128 threads = 4 warps):
//   - Weights are read in their GGUF layout, untouched (the prefill GEMM kernels share
//     the same device buffer).
//   - Every 4 consecutive threads share one 32-weight block, 8 bytes of quants each, so
//     one warp reads 8 consecutive blocks (a contiguous stretch of the row) per step and
//     128 threads read 32 blocks. The old kernel gave each row 4 threads and each warp 8
//     rows, so a warp read 8 scattered places and one thread did a whole block alone.
//   - Activations come quantized to Q8_1 by vira_cu_quantize_q8_1:
//     40-byte blocks {f32 d, f32 sum, int8 qs[32]}.
//   - Partial sums: warp shuffle, then 4 values through shared memory.
//
// Signature matches vira_cu_mmvq_q8_0, so the host passes the same arguments:
//   (xq8_1, w, y, M, nb = K / 32, N); grid (N, M), block (128, 1, 1).

#define WARP 32
#define THREADS 128

#ifdef VL_EMULATE
#include "emulate.h" // CPU test harness: runtime/cuda/test_emulate.py
#else
#define VL_WRITER 0      // the thread that writes the row's result
#define VL_WARP_WRITER 0 // the lane that writes in the warp-per-row kernel

__device__ __forceinline__ int vl_dp4a(int a, int b, int c) {
  int d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); // sm_61+
  return d;
}

// Two 16-bit loads: a Q8_0 block is 34 bytes, so its quants are only 2-byte aligned.
__device__ __forceinline__ int vl_load_int_b2(const unsigned char *p) {
  const unsigned short *s = (const unsigned short *)p;
  return (int)((unsigned int)__ldg(s) | ((unsigned int)__ldg(s + 1) << 16));
}

__device__ __forceinline__ float vl_half_to_float(unsigned short h) {
  float f;
  asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
  return f;
}

__device__ __forceinline__ float vl_block_reduce(float v) {
  __shared__ float part[THREADS / WARP];
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  const int tid = threadIdx.x;
  if ((tid & (WARP - 1)) == 0) part[tid / WARP] = v;
  __syncthreads();
  float s = 0.0f;
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < THREADS / WARP; i++) s += part[i];
  }
  return s;
}

__device__ __forceinline__ float vl_warp_reduce(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}
#endif // VL_EMULATE

extern "C" __global__ void __launch_bounds__(THREADS)
vira_cu_mmvq2_q8_0(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,
                   float *__restrict__ y, int M, int nb, int N) {
  const int row = blockIdx.x;
  const int m = blockIdx.y;
  if (row >= N || m >= M) return; // the whole block leaves together
  const int tid = threadIdx.x;
  const int part = tid & 3; // which 8 bytes of the block's 32 quants
  const unsigned char *wrow = w + (size_t)row * (size_t)nb * 34u;
  const unsigned char *xrow = xq + (size_t)m * (size_t)nb * 40u;
  float acc = 0.0f;
#pragma unroll 4
  for (int b = tid >> 2; b < nb; b += THREADS / 4) {
    const unsigned char *wb = wrow + (size_t)b * 34u;
    const unsigned char *xb = xrow + (size_t)b * 40u;
    const int w0 = vl_load_int_b2(wb + 2 + part * 8);
    const int w1 = vl_load_int_b2(wb + 6 + part * 8);
    const int *xi = (const int *)(xb + 8 + part * 8); // 40 * b + 8 + 8 * part: 4-byte aligned
    int s = vl_dp4a(w0, __ldg(xi), 0);
    s = vl_dp4a(w1, __ldg(xi + 1), s);
    const float dw = vl_half_to_float(__ldg((const unsigned short *)wb));
    const float dx = __ldg((const float *)xb);
    acc += dw * dx * (float)s;
  }
  const float total = vl_block_reduce(acc);
  if (tid == VL_WRITER) y[(size_t)m * (size_t)N + (size_t)row] = total;
}

// Default since the T4 A/B (Qwen3-8B Q8_0: warp 26 tok/s, row kernel above 25, old 13):
// one warp per output row, 4 rows per block of (32, 4). VIRA_GPU_MMVQ=row selects the one above.
// For K = 4096 a 128-thread row block gives each thread one step; here each thread does
// four, so more loads are in flight per thread. Same arguments; grid ((N + 3) / 4, M).
extern "C" __global__ void __launch_bounds__(THREADS)
vira_cu_mmvq2w_q8_0(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,
                    float *__restrict__ y, int M, int nb, int N) {
  const int row = blockIdx.x * (THREADS / WARP) + threadIdx.y;
  const int m = blockIdx.y;
  if (row >= N || m >= M) return; // a whole warp leaves together; no block barrier below
  const int lane = threadIdx.x;
  const int part = lane & 3;
  const unsigned char *wrow = w + (size_t)row * (size_t)nb * 34u;
  const unsigned char *xrow = xq + (size_t)m * (size_t)nb * 40u;
  float acc = 0.0f;
#pragma unroll 4
  for (int b = lane >> 2; b < nb; b += WARP / 4) {
    const unsigned char *wb = wrow + (size_t)b * 34u;
    const unsigned char *xb = xrow + (size_t)b * 40u;
    const int w0 = vl_load_int_b2(wb + 2 + part * 8);
    const int w1 = vl_load_int_b2(wb + 6 + part * 8);
    const int *xi = (const int *)(xb + 8 + part * 8);
    int s = vl_dp4a(w0, __ldg(xi), 0);
    s = vl_dp4a(w1, __ldg(xi + 1), s);
    const float dw = vl_half_to_float(__ldg((const unsigned short *)wb));
    const float dx = __ldg((const float *)xb);
    acc += dw * dx * (float)s;
  }
  const float total = vl_warp_reduce(acc);
  if (lane == VL_WARP_WRITER) y[(size_t)m * (size_t)N + (size_t)row] = total;
}
