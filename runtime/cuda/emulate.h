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
static thread_local vl_dim3 threadIdx, blockIdx;
static thread_local vl_dim3 blockDim = {256, 1, 1};
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
static float vl_emul_warp_r[8][64];
static inline float vl_warp_reduce_r(float v, int r) { // value r of R per warp (multi-row kernels)
  if (threadIdx.x == 0) vl_emul_warp_r[r][threadIdx.y] = 0.0f;
  vl_emul_warp_r[r][threadIdx.y] += v;
  return vl_emul_warp_r[r][threadIdx.y];
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

// ---- threaded mode, for kernels with barriers and warp shuffles (fused.cu) ----
// One std::thread per CUDA thread; one block at a time. __shared__ arrays become statics.
#include <barrier>
#include <thread>
#include <vector>
#include <algorithm>
#define __shared__ static
#define WARP_EMU 32
static std::barrier<> *vl_block_bar = nullptr;
static std::barrier<> *vl_warp_bar[64];
static float vl_wbuf_f[2048];
static int vl_wbuf_i[2048];
#define VL_SYNC() vl_block_bar->arrive_and_wait()
static inline float vl_rsqrt_approx(float x) { return 1.0f / sqrtf(x); }
static inline float vl_fma(float a, float b, float c) { return fmaf(a, b, c); }
static inline float vl_mul(float a, float b) { return a * b; }
static inline float vl_add(float a, float b) { return a + b; }
static inline float vl_div(float a, float b) { return a / b; }
static inline int __float2int_rn(float x) { return (int)nearbyintf(x); }
static inline float __int_as_float(int i) { float f; memcpy(&f, &i, 4); return f; }
using std::max;
using std::min;
static inline float vl_warp_max(float v) {
  const int t = threadIdx.x, w = t / WARP_EMU;
  vl_wbuf_f[t] = v;
  vl_warp_bar[w]->arrive_and_wait();
  float m = vl_wbuf_f[w * WARP_EMU];
  for (int i = 1; i < WARP_EMU; i++) m = fmaxf(m, vl_wbuf_f[w * WARP_EMU + i]);
  vl_warp_bar[w]->arrive_and_wait();
  return m;
}
static inline int vl_warp_sum_i(int v) {
  const int t = threadIdx.x, w = t / WARP_EMU;
  vl_wbuf_i[t] = v;
  vl_warp_bar[w]->arrive_and_wait();
  int s = 0;
  for (int i = 0; i < WARP_EMU; i++) s += vl_wbuf_i[w * WARP_EMU + i];
  vl_warp_bar[w]->arrive_and_wait();
  return s;
}
// 1-D blocks of `threads` threads, grid (gx, gy).
#define VL_LAUNCH_THREADED(k, gx, gy, threads, ...)                              \
  for (int by_ = 0; by_ < (gy); by_++)                                           \
  for (int bx_ = 0; bx_ < (gx); bx_++) {                                         \
    std::barrier<> bb_(threads);                                                 \
    std::vector<std::barrier<> *> wb_;                                           \
    for (int w_ = 0; w_ < (threads) / WARP_EMU; w_++) wb_.push_back(new std::barrier<>(WARP_EMU)); \
    vl_block_bar = &bb_;                                                         \
    for (int w_ = 0; w_ < (threads) / WARP_EMU; w_++) vl_warp_bar[w_] = wb_[w_]; \
    std::vector<std::thread> th_;                                                \
    for (int t_ = 0; t_ < (threads); t_++)                                       \
      th_.emplace_back([&, t_] { threadIdx.x = t_; threadIdx.y = 0; blockIdx.x = bx_; blockIdx.y = by_; k(__VA_ARGS__); }); \
    for (auto &t_ : th_) t_.join();                                              \
    for (auto *p_ : wb_) delete p_;                                              \
  }

// attn.cu helpers
#define __expf(x) expf(x)
static inline float vl_h2f(unsigned short h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }
static inline unsigned short vl_f2h(float x) { _Float16 f = (_Float16)x; unsigned short h; memcpy(&h, &f, 2); return h; }
static float vl_wbuf_s[2048];
static inline float vl_wsum(float v) {
  const int t = threadIdx.x, w = t / WARP_EMU;
  vl_wbuf_s[t] = v;
  vl_warp_bar[w]->arrive_and_wait();
  float s = 0.0f;
  for (int i = 0; i < WARP_EMU; i++) s += vl_wbuf_s[w * WARP_EMU + i];
  vl_warp_bar[w]->arrive_and_wait();
  return s;
}

// mmqt.cu: mma.sync.m8n8k16.row.col.s32.s8.s8.s32 (warp-wide). Written from the PTX ISA
// fragment tables (llama.cpp ggml-cuda/mma.cuh tile<8, 4, int> / tile<16, 8, int>), not from
// the kernel: lane 4 r + j holds int j (values 4j..4j+3) of A row r and of B column r;
// the lane's D elements are row lane / 4, columns 2 (lane % 4) and 2 (lane % 4) + 1.
static int vl_mma_a[2048], vl_mma_b[2048];
static inline void vl_mma(int &d0, int &d1, int a, int b) {
  const int t = threadIdx.x, w = t / WARP_EMU, lane = t % WARP_EMU, base = w * WARP_EMU;
  vl_mma_a[t] = a;
  vl_mma_b[t] = b;
  vl_warp_bar[w]->arrive_and_wait();
  const int row = lane / 4;
  int s0 = 0, s1 = 0;
  for (int j = 0; j < 4; j++) {
    s0 = vl_dp4a(vl_mma_a[base + 4 * row + j], vl_mma_b[base + 4 * (2 * (lane % 4)) + j], s0);
    s1 = vl_dp4a(vl_mma_a[base + 4 * row + j], vl_mma_b[base + 4 * (2 * (lane % 4) + 1) + j], s1);
  }
  vl_warp_bar[w]->arrive_and_wait();
  d0 += s0;
  d1 += s1;
}
// vector types and helpers of mmqt.cu
struct int2 { int x, y; };
struct int4 { int x, y, z, w; };
struct float2 { float x, y; };
#define __align__(n)
static inline int __funnelshift_r(int lo, int hi, int sh) {
  const unsigned long long v = ((unsigned long long)(unsigned int)hi << 32) | (unsigned int)lo;
  return (int)(unsigned int)(v >> (sh & 31));
}
