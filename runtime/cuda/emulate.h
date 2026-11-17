// CPU stand-in for the CUDA features the kernels in this folder use, so their index
// arithmetic can be tested without a GPU (runtime/cuda/test_emulate.py). Threads of a
// block run one after another; the block reduction adds every thread's value and hands
// the total to the last thread, which is therefore the writer here.
#include <stdint.h>
#include <string.h>
#include <math.h>
#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(x)
struct vl_dim3 { int x, y, z; };
static vl_dim3 threadIdx, blockIdx;
template <typename T> static inline T __ldg(const T *p) { return *p; }
#define VL_WRITER (THREADS - 1)
#define VL_WARP_WRITER 31
static inline int vl_dp4a(int a, int b, int c) {
  const signed char *x = (const signed char *)&a, *y = (const signed char *)&b;
  return c + x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}
static inline int vl_load_int_b2(const unsigned char *p) {
  unsigned short a, b;
  memcpy(&a, p, 2);
  memcpy(&b, p + 2, 2);
  return (int)((unsigned int)a | ((unsigned int)b << 16));
}
static inline float vl_half_to_float(unsigned short h) {
  unsigned int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float v = (e == 0) ? ldexpf((float)m, -24) : ldexpf((float)(m | 1024), (int)e - 25);
  return s ? -v : v;
}
static float vl_emul_sum = 0.0f;
static inline float vl_block_reduce(float v) {
  if (threadIdx.x == 0) vl_emul_sum = 0.0f;
  vl_emul_sum += v;
  return vl_emul_sum; // complete at the last thread
}
static float vl_emul_warp[64];
static inline float vl_warp_reduce(float v) { // threads of a warp run lane 0..31 in order
  if (threadIdx.x == 0) vl_emul_warp[threadIdx.y] = 0.0f;
  vl_emul_warp[threadIdx.y] += v;
  return vl_emul_warp[threadIdx.y]; // complete at lane 31
}
// Runs kernel `k` over grid (gx, gy) with `threads` threads per block.
#define VL_LAUNCH(k, gx, gy, threads, ...)                       \
  for (int by = 0; by < (gy); by++)                              \
    for (int bx = 0; bx < (gx); bx++)                            \
      for (int t = 0; t < (threads); t++) {                      \
        blockIdx.x = bx; blockIdx.y = by; threadIdx.x = t;       \
        k(__VA_ARGS__);                                          \
      }
// Same with a 2-D block (tx, ty).
#define VL_LAUNCH2(k, gx, gy, tx, ty, ...)                       \
  for (int by = 0; by < (gy); by++)                              \
    for (int bx = 0; bx < (gx); bx++)                            \
      for (int j = 0; j < (ty); j++)                             \
        for (int t = 0; t < (tx); t++) {                         \
          blockIdx.x = bx; blockIdx.y = by;                      \
          threadIdx.x = t; threadIdx.y = j;                      \
          k(__VA_ARGS__);                                        \
        }
