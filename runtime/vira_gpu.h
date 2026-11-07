#ifndef VIRA_GPU_H
#define VIRA_GPU_H

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "vira_rt.h"

// ============================================================================
// VIRA v1.1 ULTRA-FAST GPU RUNTIME (NATIVE Q4_0 / Q8_0 / FP16 + WARP GEMV)
// ============================================================================
// - Native GGUF Q4_0 (18B / 32 weights) & Q8_0 (34B / 32 weights) VRAM storage:
//   cuts VRAM usage by up to 7.1x vs FP32 so 3B-7B models fit in 4GB VRAM!
// - Multi-Row Warp GEMV (32 threads/warp, 8 rows/workgroup): all 8 warps share
//   activation vector `x` in L1 cache and reduce without slow 256-thread barriers.
// - Redundant PCIe Upload Elimination: skips re-uploading `x` across Q/K/V and
//   Gate/Up projections when `x.data` is unchanged.
// ============================================================================

#if defined(_WIN32) && !defined(_WIN64)
#define VIRA_CL_API __stdcall
#else
#define VIRA_CL_API
#endif

typedef int32_t vira_cl_int;
typedef uint32_t vira_cl_uint;
typedef uint64_t vira_cl_ulong;
typedef uint64_t vira_cl_bitfield;
typedef void *vira_cl_platform_id;
typedef void *vira_cl_device_id;
typedef void *vira_cl_context;
typedef void *vira_cl_command_queue;
typedef void *vira_cl_mem;
typedef void *vira_cl_program;
typedef void *vira_cl_kernel;
typedef void *vira_cl_event;

#define VIRA_CL_SUCCESS 0
#define VIRA_CL_DEVICE_TYPE_GPU (1ULL << 2)
#define VIRA_CL_DEVICE_TYPE_ALL 0xFFFFFFFFULL

#define VIRA_CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define VIRA_CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define VIRA_CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define VIRA_CL_DEVICE_NAME 0x102B
#define VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#define VIRA_CL_PROGRAM_BUILD_LOG 0x1183

#define VIRA_CL_MEM_READ_WRITE (1ULL << 0)
#define VIRA_CL_MEM_WRITE_ONLY (1ULL << 1)
#define VIRA_CL_MEM_READ_ONLY (1ULL << 2)
#define VIRA_CL_MEM_COPY_HOST_PTR (1ULL << 5)

typedef vira_cl_int(VIRA_CL_API *pfn_clGetPlatformIDs)(vira_cl_uint,
                                                       vira_cl_platform_id *,
                                                       vira_cl_uint *);
typedef vira_cl_int(VIRA_CL_API *pfn_clGetDeviceIDs)(vira_cl_platform_id,
                                                     vira_cl_bitfield,
                                                     vira_cl_uint,
                                                     vira_cl_device_id *,
                                                     vira_cl_uint *);
typedef vira_cl_int(VIRA_CL_API *pfn_clGetDeviceInfo)(vira_cl_device_id,
                                                      vira_cl_uint, size_t,
                                                      void *, size_t *);
typedef vira_cl_context(VIRA_CL_API *pfn_clCreateContext)(
    const void *, vira_cl_uint, const vira_cl_device_id *, void *, void *,
    vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseContext)(vira_cl_context);
typedef vira_cl_command_queue(VIRA_CL_API *pfn_clCreateCommandQueue)(
    vira_cl_context, vira_cl_device_id, vira_cl_bitfield, vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseCommandQueue)(
    vira_cl_command_queue);
typedef vira_cl_program(VIRA_CL_API *pfn_clCreateProgramWithSource)(
    vira_cl_context, vira_cl_uint, const char **, const size_t *,
    vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clBuildProgram)(
    vira_cl_program, vira_cl_uint, const vira_cl_device_id *, const char *,
    void *, void *);
typedef vira_cl_int(VIRA_CL_API *pfn_clGetProgramBuildInfo)(
    vira_cl_program, vira_cl_device_id, vira_cl_uint, size_t, void *,
    size_t *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseProgram)(vira_cl_program);
typedef vira_cl_kernel(VIRA_CL_API *pfn_clCreateKernel)(vira_cl_program,
                                                        const char *,
                                                        vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseKernel)(vira_cl_kernel);
typedef vira_cl_mem(VIRA_CL_API *pfn_clCreateBuffer)(vira_cl_context,
                                                     vira_cl_bitfield, size_t,
                                                     void *, vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseMemObject)(vira_cl_mem);
typedef vira_cl_int(VIRA_CL_API *pfn_clEnqueueWriteBuffer)(
    vira_cl_command_queue, vira_cl_mem, vira_cl_uint, size_t, size_t,
    const void *, vira_cl_uint, const vira_cl_event *, vira_cl_event *);
typedef vira_cl_int(VIRA_CL_API *pfn_clEnqueueReadBuffer)(
    vira_cl_command_queue, vira_cl_mem, vira_cl_uint, size_t, size_t, void *,
    vira_cl_uint, const vira_cl_event *, vira_cl_event *);
typedef vira_cl_int(VIRA_CL_API *pfn_clSetKernelArg)(vira_cl_kernel,
                                                     vira_cl_uint, size_t,
                                                     const void *);
typedef vira_cl_int(VIRA_CL_API *pfn_clEnqueueNDRangeKernel)(
    vira_cl_command_queue, vira_cl_kernel, vira_cl_uint, const size_t *,
    const size_t *, const size_t *, vira_cl_uint, const vira_cl_event *,
    vira_cl_event *);
typedef vira_cl_int(VIRA_CL_API *pfn_clFinish)(vira_cl_command_queue);

#define VIRA_GPU_WEIGHT_CACHE_CAP 4096
#define VIRA_GPU_WEIGHT_PROBE_LEN 64
#define VIRA_GPU_NUM_KERNELS 10
#define VIRA_GPU_NUM_REQUIRED_KERNELS 7
#define VIRA_GPU_UPLOAD_CHUNK_ELEMS (1u << 22)

#define VIRA_CL_ERR_NO_ROOM (-4)
#define VIRA_CL_ERR_FP16_RANGE (-5)

typedef enum {
  VIRA_WFMT_F32 = 0,
  VIRA_WFMT_F16 = 1,
  VIRA_WFMT_Q4_0 = 2,
  VIRA_WFMT_Q8_0 = 8,
} _vira_gpu_wfmt;

typedef struct {
  const vira_f32 *host_ptr;
  size_t elems;
  size_t bytes;
  vira_cl_mem dev_buf;
  uint64_t last_used;
  _vira_gpu_wfmt fmt;
  bool uncacheable;
} _vira_gpu_cache_entry;

typedef struct {
  volatile int32_t initialized;
  bool available;
  bool enabled;
  bool unified_mem;
  bool vram_saturated;
  volatile bool simulate_oom;
  char device_name[256];
  char backend_name[64];
  int64_t vram_bytes;
  int64_t vram_budget_bytes;
  volatile int64_t allocated_bytes;
  volatile int64_t oom_fallbacks;
  int32_t compute_units;
  size_t max_wg_size;
  uint64_t access_clock;
  pthread_mutex_t lock;

  void *lib_handle;
  pfn_clGetPlatformIDs clGetPlatformIDs;
  pfn_clGetDeviceIDs clGetDeviceIDs;
  pfn_clGetDeviceInfo clGetDeviceInfo;
  pfn_clCreateContext clCreateContext;
  pfn_clReleaseContext clReleaseContext;
  pfn_clCreateCommandQueue clCreateCommandQueue;
  pfn_clReleaseCommandQueue clReleaseCommandQueue;
  pfn_clCreateProgramWithSource clCreateProgramWithSource;
  pfn_clBuildProgram clBuildProgram;
  pfn_clGetProgramBuildInfo clGetProgramBuildInfo;
  pfn_clReleaseProgram clReleaseProgram;
  pfn_clCreateKernel clCreateKernel;
  pfn_clReleaseKernel clReleaseKernel;
  pfn_clCreateBuffer clCreateBuffer;
  pfn_clReleaseMemObject clReleaseMemObject;
  pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer;
  pfn_clEnqueueReadBuffer clEnqueueReadBuffer;
  pfn_clSetKernelArg clSetKernelArg;
  pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
  pfn_clFinish clFinish;

  vira_cl_device_id device;
  vira_cl_context context;
  vira_cl_command_queue queue;
  vira_cl_program program;

  // Compute kernels
  vira_cl_kernel kernel_gemv_q4_0;
  vira_cl_kernel kernel_gemv_q8_0;
  vira_cl_kernel kernel_gemv_vec8;
  vira_cl_kernel kernel_gemv_vec4;
  vira_cl_kernel kernel_gemv_scalar;
  vira_cl_kernel kernel_linear_tiled;
  vira_cl_kernel kernel_matmul_tiled;
  vira_cl_kernel kernel_rmsnorm;
  vira_cl_kernel kernel_rope;
  vira_cl_kernel kernel_silu;

  // Persistent scratch buffers + redundant upload tracker
  vira_cl_mem scratch_a;
  size_t scratch_a_cap;
  const vira_f32 *last_x_ptr;
  size_t last_x_bytes;

  vira_cl_mem scratch_b;
  size_t scratch_b_cap;
  vira_cl_mem scratch_c;
  size_t scratch_c_cap;

  _vira_gpu_cache_entry weight_cache[VIRA_GPU_WEIGHT_CACHE_CAP];
} _vira_gpu_state;

static _vira_gpu_state _vira_gpu = {
    .initialized = 0,
    .available = false,
    .enabled = true,
    .unified_mem = false,
    .vram_saturated = false,
    .simulate_oom = false,
    .device_name = "Multi-Threaded CPU SIMD Fallback",
    .backend_name = "CPU-SIMD",
    .vram_bytes = 0,
    .vram_budget_bytes = 0,
    .allocated_bytes = 0,
    .oom_fallbacks = 0,
    .compute_units = 0,
    .max_wg_size = 256,
    .access_clock = 0,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .lib_handle = NULL,
};

// ============================================================================
// OPENCL KERNELS: NATIVE Q4_0, Q8_0, & FP16 MULTI-ROW WARP GEMV (8 ROWS/WG)
// ============================================================================
static const char *_vira_opencl_kernel_src =
    "#define WARP_SIZE 32\n"
    "#define ROWS_PER_WG 8\n"
    "#define WG_SIZE 256\n"
    "#define TILE_DIM 16\n"
    "\n"
    "// 1. Native GGUF Q4_0 Multi-Row Warp GEMV (18 bytes per 32 weights)\n"
    "//    32 threads per row (1 warp), 8 rows per work-group.\n"
    "__kernel void vira_cl_gemv_q4_0(\n"
    "    __global const float* restrict x,\n"
    "    __global const uchar* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int num_blocks_k, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K    = num_blocks_k * 32;\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    __global const uchar* w_row = w + (size_t)n * (size_t)num_blocks_k * 18ULL;\n"
    "    for (int b = lane; b < num_blocks_k; b += WARP_SIZE) {\n"
    "      __global const uchar* blk = w_row + (size_t)b * 18ULL;\n"
    "      const float scale = vload_half(0, (__global const half*)blk);\n"
    "      __global const uchar* qs = blk + 2;\n"
    "      __global const float* xb = x_row + (size_t)b * 32ULL;\n"
    "      float bsum = 0.0f;\n"
    "      #pragma unroll 4\n"
    "      for (int j = 0; j < 16; ++j) {\n"
    "        const uchar byte = qs[j];\n"
    "        const float v0 = (float)((int)(byte & 0x0Fu) - 8);\n"
    "        const float v1 = (float)((int)(byte >> 4) - 8);\n"
    "        bsum = fma(xb[j],      v0, bsum);\n"
    "        bsum = fma(xb[j + 16], v1, bsum);\n"
    "      }\n"
    "      sum = fma(scale, bsum, sum);\n"
    "    }\n"
    "  }\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 2. Native GGUF Q8_0 Multi-Row Warp GEMV (34 bytes per 32 weights)\n"
    "__kernel void vira_cl_gemv_q8_0(\n"
    "    __global const float* restrict x,\n"
    "    __global const uchar* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int num_blocks_k, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K    = num_blocks_k * 32;\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    __global const uchar* w_row = w + (size_t)n * (size_t)num_blocks_k * 34ULL;\n"
    "    for (int b = lane; b < num_blocks_k; b += WARP_SIZE) {\n"
    "      __global const uchar* blk = w_row + (size_t)b * 34ULL;\n"
    "      const float scale = vload_half(0, (__global const half*)blk);\n"
    "      __global const char* qs = (__global const char*)(blk + 2);\n"
    "      __global const float* xb = x_row + (size_t)b * 32ULL;\n"
    "      float bsum = 0.0f;\n"
    "      #pragma unroll 8\n"
    "      for (int j = 0; j < 32; ++j) {\n"
    "        bsum = fma(xb[j], (float)qs[j], bsum);\n"
    "      }\n"
    "      sum = fma(scale, bsum, sum);\n"
    "    }\n"
    "  }\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 3. FP16-weight Multi-Row Warp GEMV, float4 loads (K % 8 == 0)\n"
    "__kernel void vira_cl_gemv_vec8_h(\n"
    "    __global const float4* restrict x,\n"
    "    __global const half* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K8, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K4   = K8 * 2;\n"
    "\n"
    "  float4 acc0 = (float4)(0.0f);\n"
    "  float4 acc1 = (float4)(0.0f);\n"
    "  if (n < N && m < M) {\n"
    "    __global const float4* x_row = x + (size_t)m * (size_t)K4;\n"
    "    const size_t w_off = (size_t)n * (size_t)K4;\n"
    "    int k = lane;\n"
    "    for (; k + WARP_SIZE < K4; k += (WARP_SIZE * 2)) {\n"
    "      acc0 = fma(x_row[k],             vload_half4(w_off + (size_t)k, w),             acc0);\n"
    "      acc1 = fma(x_row[k + WARP_SIZE], vload_half4(w_off + (size_t)(k + WARP_SIZE), w), acc1);\n"
    "    }\n"
    "    if (k < K4) {\n"
    "      acc0 = fma(x_row[k], vload_half4(w_off + (size_t)k, w), acc0);\n"
    "    }\n"
    "  }\n"
    "  const float4 s4 = acc0 + acc1;\n"
    "  const float sum = (s4.x + s4.y) + (s4.z + s4.w);\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 4. FP16-weight Multi-Row Warp GEMV, float4 loads (K % 4 == 0)\n"
    "__kernel void vira_cl_gemv_vec4_h(\n"
    "    __global const float4* restrict x,\n"
    "    __global const half* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K4, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "\n"
    "  float4 acc0 = (float4)(0.0f);\n"
    "  if (n < N && m < M) {\n"
    "    __global const float4* x_row = x + (size_t)m * (size_t)K4;\n"
    "    const size_t w_off = (size_t)n * (size_t)K4;\n"
    "    for (int k = lane; k < K4; k += WARP_SIZE) {\n"
    "      acc0 = fma(x_row[k], vload_half4(w_off + (size_t)k, w), acc0);\n"
    "    }\n"
    "  }\n"
    "  const float sum = (acc0.x + acc0.y) + (acc0.z + acc0.w);\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 5. Scalar FP16-weight Multi-Row Warp GEMV fallback\n"
    "__kernel void vira_cl_gemv_scalar_h(\n"
    "    __global const float* restrict x,\n"
    "    __global const half* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    const size_t w_off = (size_t)n * (size_t)K;\n"
    "    for (int k = lane; k < K; k += WARP_SIZE) {\n"
    "      sum = fma(x_row[k], vload_half(w_off + (size_t)k, w), sum);\n"
    "    }\n"
    "  }\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 6. 16x16 tiled linear projection with FP16 weights\n"
    "__kernel void vira_cl_linear_tiled_h(\n"
    "    __global const float* restrict x,\n"
    "    __global const half* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int tx = (int)get_local_id(0);\n"
    "  const int ty = (int)get_local_id(1);\n"
    "  const int col = (int)get_group_id(0) * TILE_DIM + tx;\n"
    "  const int row = (int)get_group_id(1) * TILE_DIM + ty;\n"
    "  __local float tile_x[TILE_DIM][TILE_DIM];\n"
    "  __local float tile_w[TILE_DIM][TILE_DIM + 1];\n"
    "  float acc = 0.0f;\n"
    "  const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;\n"
    "  for (int t = 0; t < num_tiles; ++t) {\n"
    "    const int kx = t * TILE_DIM + tx;\n"
    "    tile_x[ty][tx] = (row < M && kx < K) ? x[(size_t)row * (size_t)K + (size_t)kx] : 0.0f;\n"
    "    const int w_row_idx = (int)get_group_id(0) * TILE_DIM + ty;\n"
    "    const int w_col_idx = t * TILE_DIM + tx;\n"
    "    tile_w[ty][tx] = (w_row_idx < N && w_col_idx < K)\n"
    "                         ? vload_half((size_t)w_row_idx * (size_t)K + (size_t)w_col_idx, w)\n"
    "                         : 0.0f;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    #pragma unroll\n"
    "    for (int k = 0; k < TILE_DIM; ++k) {\n"
    "      acc = fma(tile_x[ty][k], tile_w[tx][k], acc);\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  if (row < M && col < N) {\n"
    "    y[(size_t)row * (size_t)N + (size_t)col] = acc;\n"
    "  }\n"
    "}\n"
    "\n"
    "// 7. 16x16 tiled FP32 matmul\n"
    "__kernel void vira_cl_matmul_tiled_f32(\n"
    "    __global const float* restrict a,\n"
    "    __global const float* restrict b,\n"
    "    __global float* restrict c,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int tx = (int)get_local_id(0);\n"
    "  const int ty = (int)get_local_id(1);\n"
    "  const int col = (int)get_group_id(0) * TILE_DIM + tx;\n"
    "  const int row = (int)get_group_id(1) * TILE_DIM + ty;\n"
    "  __local float tile_a[TILE_DIM][TILE_DIM];\n"
    "  __local float tile_b[TILE_DIM][TILE_DIM];\n"
    "  float acc = 0.0f;\n"
    "  const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;\n"
    "  for (int t = 0; t < num_tiles; ++t) {\n"
    "    const int ka = t * TILE_DIM + tx;\n"
    "    const int kb = t * TILE_DIM + ty;\n"
    "    tile_a[ty][tx] = (row < M && ka < K) ? a[(size_t)row * (size_t)K + (size_t)ka] : 0.0f;\n"
    "    tile_b[ty][tx] = (kb < K && col < N) ? b[(size_t)kb * (size_t)N + (size_t)col] : 0.0f;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    #pragma unroll\n"
    "    for (int k = 0; k < TILE_DIM; ++k) {\n"
    "      acc = fma(tile_a[ty][k], tile_b[k][tx], acc);\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  if (row < M && col < N) {\n"
    "    c[(size_t)row * (size_t)N + (size_t)col] = acc;\n"
    "  }\n"
    "}\n"
    "\n"
    "// 8. Work-group RMSNorm\n"
    "__kernel void vira_cl_rmsnorm_f32(\n"
    "    __global const float* restrict x,\n"
    "    __global const float* restrict weight,\n"
    "    __global float* restrict out,\n"
    "    const int cols, const float eps) {\n"
    "  const int lid = (int)get_local_id(0);\n"
    "  const int row = (int)get_group_id(0);\n"
    "  __global const float* x_row = x + (size_t)row * (size_t)cols;\n"
    "  __global float* out_row = out + (size_t)row * (size_t)cols;\n"
    "  float sum_sq = 0.0f;\n"
    "  for (int c = lid; c < cols; c += WG_SIZE) {\n"
    "    const float v = x_row[c];\n"
    "    sum_sq = fma(v, v, sum_sq);\n"
    "  }\n"
    "  __local float sdata[WG_SIZE];\n"
    "  sdata[lid] = sum_sq;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {\n"
    "    if (lid < s) sdata[lid] += sdata[lid + s];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  const float inv_rms = rsqrt(sdata[0] / (float)cols + eps);\n"
    "  for (int c = lid; c < cols; c += WG_SIZE) {\n"
    "    out_row[c] = x_row[c] * inv_rms * weight[c];\n"
    "  }\n"
    "}\n"
    "\n"
    "// 9. Rotary Position Embedding (RoPE)\n"
    "__kernel void vira_cl_rope_f32(\n"
    "    __global const float* restrict q,\n"
    "    __global float* restrict out,\n"
    "    const int rows, const int cols, const int half_dim,\n"
    "    const float pos, const float freq_base) {\n"
    "  const int i = (int)get_global_id(0);\n"
    "  const int r = (int)get_global_id(1);\n"
    "  if (i >= half_dim || r >= rows) return;\n"
    "  const float exponent = -(2.0f * (float)i) / (float)cols;\n"
    "  const float angle = pos * pow(freq_base, exponent);\n"
    "  const float cos_a = cos(angle);\n"
    "  const float sin_a = sin(angle);\n"
    "  __global const float* q_row = q + (size_t)r * (size_t)cols;\n"
    "  __global float* o_row = out + (size_t)r * (size_t)cols;\n"
    "  const float q0 = q_row[i];\n"
    "  const float q1 = q_row[i + half_dim];\n"
    "  o_row[i]            = fma(q0, cos_a, -q1 * sin_a);\n"
    "  o_row[i + half_dim] = fma(q0, sin_a,  q1 * cos_a);\n"
    "}\n"
    "\n"
    "// 10. SiLU activation\n"
    "__kernel void vira_cl_silu_f32(\n"
    "    __global const float* restrict x,\n"
    "    __global float* restrict out,\n"
    "    const int total) {\n"
    "  const int idx = (int)get_global_id(0);\n"
    "  if (idx >= total) return;\n"
    "  const float v = x[idx];\n"
    "  out[idx] = v / (1.0f + native_exp(-v));\n"
    "}\n";

static const char *_vira_opencl_kernel_names[VIRA_GPU_NUM_KERNELS] = {
    "vira_cl_gemv_q4_0",        "vira_cl_gemv_q8_0",
    "vira_cl_gemv_vec8_h",      "vira_cl_gemv_vec4_h",
    "vira_cl_gemv_scalar_h",    "vira_cl_linear_tiled_h",
    "vira_cl_matmul_tiled_f32", "vira_cl_rmsnorm_f32",
    "vira_cl_rope_f32",         "vira_cl_silu_f32",
};

static inline bool _vira_gpu_debug(void) {
  const char *e = getenv("VIRA_GPU_DEBUG");
  return e != NULL && e[0] != '\0' && e[0] != '0';
}

static inline bool _vira_gpu_load_sym(void *lib, const char *name,
                                      void *fn_out) {
#ifdef _WIN32
  void *p = (void *)(uintptr_t)GetProcAddress((HMODULE)lib, name);
#else
  void *p = dlsym(lib, name);
#endif
  if (p == NULL) {
    return false;
  }
  memcpy(fn_out, &p, sizeof(void *));
  return true;
}

static inline uint16_t _vira_f32_bits_to_f16(uint32_t x) {
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t exp = (x >> 23) & 0xFFu;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp == 0xFFu) {
    return (uint16_t)(sign | 0x7C00u | (mant ? (0x200u | (mant >> 13)) : 0u));
  }
  const int32_t e = (int32_t)exp - 127 + 15;
  if (e >= 31) {
    return (uint16_t)(sign | 0x7C00u);
  }
  if (e <= 0) {
    if (e < -10) {
      return (uint16_t)sign;
    }
    mant |= 0x800000u;
    const uint32_t shift = (uint32_t)(14 - e);
    uint32_t h = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (h & 1u))) {
      h++;
    }
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
    h++;
  }
  return (uint16_t)(sign | h);
}

static inline bool _vira_f32_to_f16_block(const vira_f32 *src, uint16_t *dst,
                                          size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits;
    memcpy(&bits, &src[i], sizeof(bits));
    const uint16_t h = _vira_f32_bits_to_f16(bits);
    if (((bits >> 23) & 0xFFu) != 0xFFu && (h & 0x7FFFu) == 0x7C00u) {
      return false;
    }
    dst[i] = h;
  }
  return true;
}

static inline vira_cl_mem _vira_gpu_ensure_scratch_locked(
    vira_cl_mem *buf_ptr, size_t *cap_ptr, size_t needed_bytes,
    vira_cl_bitfield flags, vira_cl_int *out_err) {
  if (*buf_ptr != NULL && *cap_ptr >= needed_bytes) {
    *out_err = VIRA_CL_SUCCESS;
    return *buf_ptr;
  }
  size_t new_cap = (needed_bytes < (4ULL * 1024ULL * 1024ULL))
                       ? (4ULL * 1024ULL * 1024ULL)
                       : ((needed_bytes + 65535ULL) & ~65535ULL);
  if (*buf_ptr != NULL) {
    _vira_gpu.clReleaseMemObject(*buf_ptr);
    __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)(*cap_ptr),
                       __ATOMIC_RELAXED);
    *buf_ptr = NULL;
    *cap_ptr = 0;
  }
  vira_cl_int err = 0;
  vira_cl_mem new_buf =
      _vira_gpu.clCreateBuffer(_vira_gpu.context, flags, new_cap, NULL, &err);
  if (err != VIRA_CL_SUCCESS || new_buf == NULL) {
    *out_err = (err != VIRA_CL_SUCCESS) ? err : VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }
  *buf_ptr = new_buf;
  *cap_ptr = new_cap;
  __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)new_cap,
                     __ATOMIC_RELAXED);
  *out_err = VIRA_CL_SUCCESS;
  return new_buf;
}

static inline vira_cl_int _vira_gpu_set_gemm_args(vira_cl_kernel k,
                                                  vira_cl_mem *a,
                                                  vira_cl_mem *b,
                                                  vira_cl_mem *c, int32_t d0,
                                                  int32_t d1, int32_t d2) {
  vira_cl_int e = 0;
  e |= _vira_gpu.clSetKernelArg(k, 0, sizeof(vira_cl_mem), a);
  e |= _vira_gpu.clSetKernelArg(k, 1, sizeof(vira_cl_mem), b);
  e |= _vira_gpu.clSetKernelArg(k, 2, sizeof(vira_cl_mem), c);
  e |= _vira_gpu.clSetKernelArg(k, 3, sizeof(int32_t), &d0);
  e |= _vira_gpu.clSetKernelArg(k, 4, sizeof(int32_t), &d1);
  e |= _vira_gpu.clSetKernelArg(k, 5, sizeof(int32_t), &d2);
  return e;
}

static inline void _vira_gpu_dump_build_log(vira_cl_program prog,
                                            vira_cl_device_id dev) {
  if (!_vira_gpu_debug() || _vira_gpu.clGetProgramBuildInfo == NULL) {
    return;
  }
  char blog[8192] = {0};
  if (_vira_gpu.clGetProgramBuildInfo(prog, dev, VIRA_CL_PROGRAM_BUILD_LOG,
                                      sizeof(blog) - 1, blog,
                                      NULL) == VIRA_CL_SUCCESS) {
    fprintf(stderr, "vira gpu: OpenCL build failed:\n%s\n", blog);
  }
}

static inline bool _vira_gpu_init_opencl_locked(void *lib) {
  _vira_gpu_state *g = &_vira_gpu;
  bool ok = true;
#define VIRA_GPU_LOAD(sym) ok = ok && _vira_gpu_load_sym(lib, #sym, &g->sym)
  VIRA_GPU_LOAD(clGetPlatformIDs);
  VIRA_GPU_LOAD(clGetDeviceIDs);
  VIRA_GPU_LOAD(clGetDeviceInfo);
  VIRA_GPU_LOAD(clCreateContext);
  VIRA_GPU_LOAD(clReleaseContext);
  VIRA_GPU_LOAD(clCreateCommandQueue);
  VIRA_GPU_LOAD(clReleaseCommandQueue);
  VIRA_GPU_LOAD(clCreateProgramWithSource);
  VIRA_GPU_LOAD(clBuildProgram);
  VIRA_GPU_LOAD(clReleaseProgram);
  VIRA_GPU_LOAD(clCreateKernel);
  VIRA_GPU_LOAD(clReleaseKernel);
  VIRA_GPU_LOAD(clCreateBuffer);
  VIRA_GPU_LOAD(clReleaseMemObject);
  VIRA_GPU_LOAD(clEnqueueWriteBuffer);
  VIRA_GPU_LOAD(clEnqueueReadBuffer);
  VIRA_GPU_LOAD(clSetKernelArg);
  VIRA_GPU_LOAD(clEnqueueNDRangeKernel);
  VIRA_GPU_LOAD(clFinish);
#undef VIRA_GPU_LOAD
  if (!ok) {
    return false;
  }
  (void)_vira_gpu_load_sym(lib, "clGetProgramBuildInfo",
                           &g->clGetProgramBuildInfo);

  vira_cl_platform_id platforms[16];
  vira_cl_uint num_platforms = 0;
  if (g->clGetPlatformIDs(16, platforms, &num_platforms) != VIRA_CL_SUCCESS ||
      num_platforms == 0) {
    return false;
  }
  if (num_platforms > 16) {
    num_platforms = 16;
  }

  vira_cl_device_id best_dev = NULL;
  int64_t best_score = -1;
  for (vira_cl_uint p = 0; p < num_platforms; ++p) {
    vira_cl_device_id devs[8];
    vira_cl_uint num_devs = 0;
    if (g->clGetDeviceIDs(platforms[p], VIRA_CL_DEVICE_TYPE_GPU, 8, devs,
                          &num_devs) != VIRA_CL_SUCCESS ||
        num_devs == 0) {
      continue;
    }
    if (num_devs > 8) {
      num_devs = 8;
    }
    for (vira_cl_uint d = 0; d < num_devs; ++d) {
      vira_cl_ulong gmem = 0;
      vira_cl_uint cu = 0;
      vira_cl_uint um = 0;
      g->clGetDeviceInfo(devs[d], VIRA_CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem),
                         &gmem, NULL);
      g->clGetDeviceInfo(devs[d], VIRA_CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu),
                         &cu, NULL);
      g->clGetDeviceInfo(devs[d], VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY,
                         sizeof(um), &um, NULL);
      int64_t score =
          (int64_t)(gmem >> 20) * (um == 0 ? 4LL : 1LL) + (int64_t)cu * 64LL;
      if (score > best_score) {
        best_score = score;
        best_dev = devs[d];
      }
    }
  }
  if (best_dev == NULL) {
    return false;
  }

  vira_cl_int err = 0;
  vira_cl_context ctx = g->clCreateContext(NULL, 1, &best_dev, NULL, NULL, &err);
  if (err != VIRA_CL_SUCCESS || ctx == NULL) {
    return false;
  }
  vira_cl_command_queue q = g->clCreateCommandQueue(ctx, best_dev, 0, &err);
  if (err != VIRA_CL_SUCCESS || q == NULL) {
    g->clReleaseContext(ctx);
    return false;
  }

  size_t src_len = strlen(_vira_opencl_kernel_src);
  vira_cl_program prog = g->clCreateProgramWithSource(
      ctx, 1, &_vira_opencl_kernel_src, &src_len, &err);
  if (err != VIRA_CL_SUCCESS || prog == NULL) {
    g->clReleaseCommandQueue(q);
    g->clReleaseContext(ctx);
    return false;
  }
  if (g->clBuildProgram(prog, 1, &best_dev,
                        "-cl-fast-relaxed-math -cl-mad-enable", NULL,
                        NULL) != VIRA_CL_SUCCESS) {
    _vira_gpu_dump_build_log(prog, best_dev);
    g->clReleaseProgram(prog);
    g->clReleaseCommandQueue(q);
    g->clReleaseContext(ctx);
    return false;
  }

  vira_cl_kernel ks[VIRA_GPU_NUM_KERNELS] = {0};
  bool required_ok = true;
  for (int i = 0; i < VIRA_GPU_NUM_KERNELS; ++i) {
    vira_cl_int ke = 0;
    ks[i] = g->clCreateKernel(prog, _vira_opencl_kernel_names[i], &ke);
    if (ke != VIRA_CL_SUCCESS) {
      if (ks[i] != NULL) {
        g->clReleaseKernel(ks[i]);
      }
      ks[i] = NULL;
    }
    if (i < VIRA_GPU_NUM_REQUIRED_KERNELS && ks[i] == NULL) {
      required_ok = false;
    }
  }
  if (!required_ok) {
    for (int i = 0; i < VIRA_GPU_NUM_KERNELS; ++i) {
      if (ks[i] != NULL) {
        g->clReleaseKernel(ks[i]);
      }
    }
    g->clReleaseProgram(prog);
    g->clReleaseCommandQueue(q);
    g->clReleaseContext(ctx);
    return false;
  }

  g->lib_handle = lib;
  g->device = best_dev;
  g->context = ctx;
  g->queue = q;
  g->program = prog;
  g->kernel_gemv_q4_0 = ks[0];
  g->kernel_gemv_q8_0 = ks[1];
  g->kernel_gemv_vec8 = ks[2];
  g->kernel_gemv_vec4 = ks[3];
  g->kernel_gemv_scalar = ks[4];
  g->kernel_linear_tiled = ks[5];
  g->kernel_matmul_tiled = ks[6];
  g->kernel_rmsnorm = ks[7];
  g->kernel_rope = ks[8];
  g->kernel_silu = ks[9];
  g->available = true;
  snprintf(g->backend_name, sizeof(g->backend_name), "OpenCL");

  char dname[256] = {0};
  if (g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_NAME, sizeof(dname) - 1,
                         dname, NULL) == VIRA_CL_SUCCESS) {
    snprintf(g->device_name, sizeof(g->device_name), "%s", dname);
  }
  vira_cl_ulong gmem = 0;
  if (g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_GLOBAL_MEM_SIZE,
                         sizeof(gmem), &gmem, NULL) == VIRA_CL_SUCCESS) {
    g->vram_bytes = (int64_t)gmem;
    g->vram_budget_bytes = (int64_t)((gmem * 88ULL) / 100ULL);
  }
  vira_cl_uint cu = 0;
  if (g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_MAX_COMPUTE_UNITS,
                         sizeof(cu), &cu, NULL) == VIRA_CL_SUCCESS) {
    g->compute_units = (int32_t)cu;
  }
  size_t mwg = 256;
  if (g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_MAX_WORK_GROUP_SIZE,
                         sizeof(mwg), &mwg, NULL) == VIRA_CL_SUCCESS &&
      mwg > 0) {
    g->max_wg_size = mwg;
  }
  vira_cl_uint um = 0;
  if (g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY,
                         sizeof(um), &um, NULL) == VIRA_CL_SUCCESS) {
    g->unified_mem = (um != 0);
  }
  if (_vira_gpu_debug()) {
    fprintf(stderr,
            "vira gpu: %s | %lld MiB VRAM | %d CU | max_wg=%zu | Q4_0/Q8_0/FP16 Warp GEMV\n",
            g->device_name, (long long)(g->vram_bytes >> 20), g->compute_units,
            g->max_wg_size);
  }
  return true;
}

static inline void vira_gpu_init(void) {
  if (__atomic_load_n(&_vira_gpu.initialized, __ATOMIC_ACQUIRE) == 2) {
    return;
  }
  pthread_mutex_lock(&_vira_gpu.lock);
  if (_vira_gpu.initialized == 2) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    return;
  }

  snprintf(_vira_gpu.device_name, sizeof(_vira_gpu.device_name),
           "Multi-Threaded CPU SIMD (%s)", vira_simd_backend_name());
  snprintf(_vira_gpu.backend_name, sizeof(_vira_gpu.backend_name), "CPU-SIMD");
  _vira_gpu.available = false;
  _vira_gpu.enabled = true;

  void *lib = NULL;
#ifdef _WIN32
  lib = (void *)LoadLibraryA("OpenCL.dll");
#elif defined(__APPLE__)
  lib = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_LAZY);
#else
  lib = dlopen("libOpenCL.so.1", RTLD_LAZY);
  if (!lib) {
    lib = dlopen("libOpenCL.so", RTLD_LAZY);
  }
#endif
  if (lib != NULL) {
    (void)_vira_gpu_init_opencl_locked(lib);
  }

  __atomic_store_n(&_vira_gpu.initialized, 2, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline void _vira_gpu_cache_entry_reset(_vira_gpu_cache_entry *e) {
  e->host_ptr = NULL;
  e->elems = 0;
  e->bytes = 0;
  e->dev_buf = NULL;
  e->last_used = 0;
  e->fmt = VIRA_WFMT_F32;
  e->uncacheable = false;
}

static inline void _vira_gpu_evict_ptr_locked(const vira_f32 *host_ptr) {
  if (!_vira_gpu.available || host_ptr == NULL) {
    return;
  }
  if (_vira_gpu.last_x_ptr == host_ptr) {
    _vira_gpu.last_x_ptr = NULL;
    _vira_gpu.last_x_bytes = 0;
  }
  uint64_t h = ((uint64_t)(uintptr_t)host_ptr >> 6) * 11400714819323198485ULL;
  int32_t start_slot = (int32_t)(h & (VIRA_GPU_WEIGHT_CACHE_CAP - 1));

  for (int32_t probe = 0; probe < VIRA_GPU_WEIGHT_PROBE_LEN; ++probe) {
    int32_t idx = (start_slot + probe) & (VIRA_GPU_WEIGHT_CACHE_CAP - 1);
    _vira_gpu_cache_entry *e = &_vira_gpu.weight_cache[idx];
    if (e->host_ptr == host_ptr) {
      if (e->dev_buf != NULL) {
        _vira_gpu.clReleaseMemObject(e->dev_buf);
        __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)e->bytes,
                           __ATOMIC_RELAXED);
      }
      _vira_gpu_cache_entry_reset(e);
    }
  }
}

static inline void _vira_gpu_evict_ptr(const vira_f32 *host_ptr) {
  if (__atomic_load_n(&_vira_gpu.initialized, __ATOMIC_ACQUIRE) != 2 ||
      !_vira_gpu.available || host_ptr == NULL) {
    return;
  }
  pthread_mutex_lock(&_vira_gpu.lock);
  _vira_gpu_evict_ptr_locked(host_ptr);
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline void _vira_gpu_evict_all_locked(void) {
  if (!_vira_gpu.available) {
    return;
  }
  for (int32_t i = 0; i < VIRA_GPU_WEIGHT_CACHE_CAP; ++i) {
    if (_vira_gpu.weight_cache[i].dev_buf != NULL) {
      _vira_gpu.clReleaseMemObject(_vira_gpu.weight_cache[i].dev_buf);
    }
    _vira_gpu_cache_entry_reset(&_vira_gpu.weight_cache[i]);
  }
  if (_vira_gpu.scratch_a != NULL) {
    _vira_gpu.clReleaseMemObject(_vira_gpu.scratch_a);
    _vira_gpu.scratch_a = NULL;
    _vira_gpu.scratch_a_cap = 0;
  }
  _vira_gpu.last_x_ptr = NULL;
  _vira_gpu.last_x_bytes = 0;
  if (_vira_gpu.scratch_b != NULL) {
    _vira_gpu.clReleaseMemObject(_vira_gpu.scratch_b);
    _vira_gpu.scratch_b = NULL;
    _vira_gpu.scratch_b_cap = 0;
  }
  if (_vira_gpu.scratch_c != NULL) {
    _vira_gpu.clReleaseMemObject(_vira_gpu.scratch_c);
    _vira_gpu.scratch_c = NULL;
    _vira_gpu.scratch_c_cap = 0;
  }
  _vira_gpu.vram_saturated = false;
  __atomic_store_n(&_vira_gpu.allocated_bytes, 0, __ATOMIC_RELAXED);
}

// Called directly by `vira_gguf_load_tensor` in `vira_rt.h` to upload raw Q4_0 / Q8_0
// blocks into VRAM immediately upon model load (7.1x smaller than FP32!).
static inline void _vira_gpu_register_quant(const vira_f32 *host_ptr,
                                            uint32_t gguf_dtype,
                                            const void *raw_blocks,
                                            size_t raw_bytes, size_t elems) {
  vira_gpu_init();
  if (!_vira_gpu.available || !_vira_gpu.enabled || host_ptr == NULL ||
      raw_blocks == NULL || raw_bytes == 0 || (elems & 31) != 0) {
    return;
  }
  _vira_gpu_wfmt fmt = (gguf_dtype == 2)
                           ? VIRA_WFMT_Q4_0
                           : ((gguf_dtype == 8 || gguf_dtype == 7)
                                  ? VIRA_WFMT_Q8_0
                                  : VIRA_WFMT_F32);
  if (fmt != VIRA_WFMT_Q4_0 && fmt != VIRA_WFMT_Q8_0) {
    return;
  }

  pthread_mutex_lock(&_vira_gpu.lock);
  int64_t cur_alloc =
      __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
  if (_vira_gpu.vram_budget_bytes > 0 &&
      cur_alloc + (int64_t)raw_bytes > _vira_gpu.vram_budget_bytes) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    return;
  }

  uint64_t h = ((uint64_t)(uintptr_t)host_ptr >> 6) * 11400714819323198485ULL;
  int32_t start_slot = (int32_t)(h & (VIRA_GPU_WEIGHT_CACHE_CAP - 1));
  int32_t empty_slot = -1;
  for (int32_t probe = 0; probe < VIRA_GPU_WEIGHT_PROBE_LEN; ++probe) {
    int32_t idx = (start_slot + probe) & (VIRA_GPU_WEIGHT_CACHE_CAP - 1);
    if (_vira_gpu.weight_cache[idx].host_ptr == host_ptr) {
      pthread_mutex_unlock(&_vira_gpu.lock);
      return;
    }
    if (empty_slot < 0 && _vira_gpu.weight_cache[idx].host_ptr == NULL) {
      empty_slot = idx;
    }
  }
  if (empty_slot < 0) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    return;
  }

  vira_cl_int err = 0;
  vira_cl_mem buf = _vira_gpu.clCreateBuffer(
      _vira_gpu.context, VIRA_CL_MEM_READ_ONLY | VIRA_CL_MEM_COPY_HOST_PTR,
      raw_bytes, (void *)raw_blocks, &err);
  if (err == VIRA_CL_SUCCESS && buf != NULL) {
    _vira_gpu_cache_entry *slot = &_vira_gpu.weight_cache[empty_slot];
    slot->host_ptr = host_ptr;
    slot->elems = elems;
    slot->bytes = raw_bytes;
    slot->dev_buf = buf;
    slot->last_used = ++_vira_gpu.access_clock;
    slot->fmt = fmt;
    slot->uncacheable = false;
    __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)raw_bytes,
                       __ATOMIC_RELAXED);
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline vira_cl_mem _vira_gpu_upload_half_locked(const vira_f32 *src,
                                                       size_t elems,
                                                       vira_cl_int *out_err) {
  const size_t bytes = elems * sizeof(uint16_t);
  vira_cl_int err = 0;
  vira_cl_mem buf = _vira_gpu.clCreateBuffer(
      _vira_gpu.context, VIRA_CL_MEM_READ_ONLY, bytes, NULL, &err);
  if (err != VIRA_CL_SUCCESS || buf == NULL) {
    *out_err = (err != VIRA_CL_SUCCESS) ? err : VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }
  const size_t chunk =
      elems < VIRA_GPU_UPLOAD_CHUNK_ELEMS ? elems : VIRA_GPU_UPLOAD_CHUNK_ELEMS;
  uint16_t *tmp = (uint16_t *)malloc(chunk * sizeof(uint16_t));
  if (tmp == NULL) {
    _vira_gpu.clReleaseMemObject(buf);
    *out_err = VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }
  for (size_t off = 0; off < elems; off += chunk) {
    const size_t n = (elems - off < chunk) ? (elems - off) : chunk;
    if (!_vira_f32_to_f16_block(src + off, tmp, n)) {
      free(tmp);
      _vira_gpu.clReleaseMemObject(buf);
      *out_err = VIRA_CL_ERR_FP16_RANGE;
      return NULL;
    }
    err = _vira_gpu.clEnqueueWriteBuffer(_vira_gpu.queue, buf, 1,
                                         off * sizeof(uint16_t),
                                         n * sizeof(uint16_t), tmp, 0, NULL,
                                         NULL);
    if (err != VIRA_CL_SUCCESS) {
      free(tmp);
      _vira_gpu.clReleaseMemObject(buf);
      *out_err = err;
      return NULL;
    }
  }
  free(tmp);
  *out_err = VIRA_CL_SUCCESS;
  return buf;
}

static inline vira_cl_mem _vira_gpu_get_or_upload_weight_locked(
    const vira_f32 *host_ptr, size_t elems, bool prefer_half,
    _vira_gpu_wfmt *out_fmt, vira_cl_int *out_err) {
  uint64_t h = ((uint64_t)(uintptr_t)host_ptr >> 6) * 11400714819323198485ULL;
  int32_t start_slot = (int32_t)(h & (VIRA_GPU_WEIGHT_CACHE_CAP - 1));
  uint64_t now = ++_vira_gpu.access_clock;

  int32_t empty_slot = -1;
  for (int32_t probe = 0; probe < VIRA_GPU_WEIGHT_PROBE_LEN; ++probe) {
    int32_t idx = (start_slot + probe) & (VIRA_GPU_WEIGHT_CACHE_CAP - 1);
    _vira_gpu_cache_entry *e = &_vira_gpu.weight_cache[idx];
    if (e->host_ptr == host_ptr && e->elems == elems) {
      if (e->uncacheable) {
        *out_err = VIRA_CL_ERR_NO_ROOM;
        return NULL;
      }
      if (e->dev_buf != NULL) {
        e->last_used = now;
        if (out_fmt) *out_fmt = e->fmt;
        *out_err = VIRA_CL_SUCCESS;
        return e->dev_buf;
      }
    }
    if (empty_slot < 0 && e->host_ptr == NULL) {
      empty_slot = idx;
    }
  }

  if (empty_slot < 0) {
    *out_err = VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }

  const size_t bytes =
      elems * (prefer_half ? sizeof(uint16_t) : sizeof(vira_f32));
  _vira_gpu_cache_entry *slot = &_vira_gpu.weight_cache[empty_slot];
  slot->host_ptr = host_ptr;
  slot->elems = elems;
  slot->bytes = bytes;
  slot->dev_buf = NULL;
  slot->last_used = now;
  slot->fmt = prefer_half ? VIRA_WFMT_F16 : VIRA_WFMT_F32;
  slot->uncacheable = true;

  int64_t cur_alloc =
      __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
  if (_vira_gpu.vram_budget_bytes > 0 &&
      cur_alloc + (int64_t)bytes > _vira_gpu.vram_budget_bytes) {
    *out_err = VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }

  vira_cl_int err = 0;
  vira_cl_mem buf = NULL;
  if (prefer_half) {
    buf = _vira_gpu_upload_half_locked(host_ptr, elems, &err);
  } else {
    buf = _vira_gpu.clCreateBuffer(
        _vira_gpu.context,
        VIRA_CL_MEM_READ_ONLY | VIRA_CL_MEM_COPY_HOST_PTR, bytes,
        (void *)host_ptr, &err);
    if (err == VIRA_CL_SUCCESS && buf == NULL) {
      err = VIRA_CL_ERR_NO_ROOM;
    }
  }
  if (err != VIRA_CL_SUCCESS || buf == NULL) {
    if (err != VIRA_CL_ERR_FP16_RANGE) {
      _vira_gpu.vram_saturated = true;
    }
    *out_err = (err != VIRA_CL_SUCCESS) ? err : VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }

  slot->dev_buf = buf;
  slot->uncacheable = false;
  if (out_fmt) *out_fmt = slot->fmt;
  __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)bytes,
                     __ATOMIC_RELAXED);
  *out_err = VIRA_CL_SUCCESS;
  return buf;
}

static inline void vira_gpu_shutdown(void) {
  if (__atomic_load_n(&_vira_gpu.initialized, __ATOMIC_ACQUIRE) != 2) {
    return;
  }
  pthread_mutex_lock(&_vira_gpu.lock);
  if (_vira_gpu.available) {
    if (_vira_gpu.queue) {
      _vira_gpu.clFinish(_vira_gpu.queue);
    }
    _vira_gpu_evict_all_locked();
    vira_cl_kernel ks[VIRA_GPU_NUM_KERNELS] = {
        _vira_gpu.kernel_gemv_q4_0,    _vira_gpu.kernel_gemv_q8_0,
        _vira_gpu.kernel_gemv_vec8,    _vira_gpu.kernel_gemv_vec4,
        _vira_gpu.kernel_gemv_scalar,  _vira_gpu.kernel_linear_tiled,
        _vira_gpu.kernel_matmul_tiled, _vira_gpu.kernel_rmsnorm,
        _vira_gpu.kernel_rope,         _vira_gpu.kernel_silu};
    for (int i = 0; i < VIRA_GPU_NUM_KERNELS; ++i) {
      if (ks[i]) {
        _vira_gpu.clReleaseKernel(ks[i]);
      }
    }
    if (_vira_gpu.program) _vira_gpu.clReleaseProgram(_vira_gpu.program);
    if (_vira_gpu.queue) _vira_gpu.clReleaseCommandQueue(_vira_gpu.queue);
    if (_vira_gpu.context) _vira_gpu.clReleaseContext(_vira_gpu.context);
    _vira_gpu.available = false;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline bool vira_gpu_is_available(void) {
  vira_gpu_init();
  return _vira_gpu.available;
}

static inline vira_string vira_gpu_device_name(void) {
  vira_gpu_init();
  return vira_str_lit(_vira_gpu.device_name,
                      (int32_t)strlen(_vira_gpu.device_name));
}

static inline vira_string vira_gpu_backend_name(void) {
  vira_gpu_init();
  return vira_str_lit(_vira_gpu.backend_name,
                      (int32_t)strlen(_vira_gpu.backend_name));
}

static inline int64_t vira_gpu_vram_bytes(void) {
  vira_gpu_init();
  return _vira_gpu.vram_bytes;
}

static inline int64_t vira_gpu_allocated_bytes(void) {
  vira_gpu_init();
  return __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
}

static inline int64_t vira_gpu_oom_fallbacks(void) {
  vira_gpu_init();
  return __atomic_load_n(&_vira_gpu.oom_fallbacks, __ATOMIC_RELAXED);
}

static inline void vira_gpu_set_enabled(bool enabled) {
  vira_gpu_init();
  _vira_gpu.enabled = enabled;
}

static inline bool vira_gpu_is_enabled(void) {
  vira_gpu_init();
  return _vira_gpu.enabled;
}

static inline void vira_gpu_simulate_oom(bool enable) {
  vira_gpu_init();
  _vira_gpu.simulate_oom = enable;
}

static inline void vira_gpu_sync(void) {
  vira_gpu_init();
  if (_vira_gpu.available && _vira_gpu.queue != NULL) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu.clFinish(_vira_gpu.queue);
    pthread_mutex_unlock(&_vira_gpu.lock);
  }
}

// ============================================================================
// GPU LINEAR PROJECTION: `y = x @ w^T` (Q4_0 / Q8_0 / FP16 + WARP GEMV)
// ============================================================================
static inline vira_tensor_f32 vira_gpu_linear(vira_arena *opt_arena,
                                              vira_tensor_f32 x,
                                              vira_tensor_f32 w) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&x);
  _vira_tensor_norm_shape(&w);
  int32_t M = x.rows;
  int32_t K = x.cols;
  int32_t N = w.rows;
  if (K != w.cols) {
    fprintf(stderr,
            "vira gpu_linear error: inner dimension mismatch x=(%dx%d) vs w=(%dx%d)\n",
            M, K, N, w.cols);
    exit(1);
  }

  if (_vira_gpu.simulate_oom) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  int64_t total_ops = (int64_t)M * (int64_t)N * (int64_t)K;
  if (!_vira_gpu.available || !_vira_gpu.enabled || total_ops < 65536LL ||
      _vira_gpu.max_wg_size < 256) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  size_t x_bytes = (size_t)M * (size_t)K * sizeof(vira_f32);
  size_t y_bytes = (size_t)M * (size_t)N * sizeof(vira_f32);

  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  _vira_gpu_wfmt wfmt = VIRA_WFMT_F16;

  // 1. Lookup Q4_0 / Q8_0 / FP16 weight in VRAM (or upload as FP16)
  vira_cl_mem w_buf = _vira_gpu_get_or_upload_weight_locked(
      w.data, (size_t)N * (size_t)K, true, &wfmt, &err);
  if (err != VIRA_CL_SUCCESS || w_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  // 2. Ensure scratch buffers
  vira_cl_mem x_buf = _vira_gpu_ensure_scratch_locked(
      &_vira_gpu.scratch_a, &_vira_gpu.scratch_a_cap, x_bytes,
      VIRA_CL_MEM_READ_WRITE, &err);
  vira_cl_mem y_buf =
      (err == VIRA_CL_SUCCESS)
          ? _vira_gpu_ensure_scratch_locked(
                &_vira_gpu.scratch_c, &_vira_gpu.scratch_c_cap, y_bytes,
                VIRA_CL_MEM_READ_WRITE, &err)
          : NULL;
  if (err != VIRA_CL_SUCCESS || x_buf == NULL || y_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  // 3. Skip PCIe write if `x` is the exact same immutable tensor as the last call
  //    (e.g. wq, wk, wv or w_gate, w_up projections sharing input `x`)
  if (_vira_gpu.last_x_ptr != x.data || _vira_gpu.last_x_bytes != x_bytes ||
      x.is_arena) {
    err = _vira_gpu.clEnqueueWriteBuffer(_vira_gpu.queue, x_buf, 0, 0, x_bytes,
                                         x.data, 0, NULL, NULL);
    _vira_gpu.last_x_ptr = x.data;
    _vira_gpu.last_x_bytes = x_bytes;
  }

  if (err == VIRA_CL_SUCCESS) {
    if (wfmt == VIRA_WFMT_Q4_0 || wfmt == VIRA_WFMT_Q8_0) {
      vira_cl_kernel k = (wfmt == VIRA_WFMT_Q4_0) ? _vira_gpu.kernel_gemv_q4_0
                                                  : _vira_gpu.kernel_gemv_q8_0;
      int32_t num_blocks_k = K >> 5;
      err = _vira_gpu_set_gemm_args(k, &x_buf, &w_buf, &y_buf, M, num_blocks_k, N);
      size_t l_work[2] = {32, 8};
      size_t num_groups_n = ((size_t)N + 7) >> 3;
      size_t g_work[2] = {num_groups_n * 32, (size_t)M * 8};
      if (err == VIRA_CL_SUCCESS) {
        err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, k, 2, NULL,
                                               g_work, l_work, 0, NULL, NULL);
      }
    } else if (M <= 4) {
      vira_cl_kernel k;
      int32_t kdim;
      if ((K & 7) == 0) {
        k = _vira_gpu.kernel_gemv_vec8;
        kdim = K >> 3;
      } else if ((K & 3) == 0) {
        k = _vira_gpu.kernel_gemv_vec4;
        kdim = K >> 2;
      } else {
        k = _vira_gpu.kernel_gemv_scalar;
        kdim = K;
      }
      err = _vira_gpu_set_gemm_args(k, &x_buf, &w_buf, &y_buf, M, kdim, N);
      size_t l_work[2] = {32, 8};
      size_t num_groups_n = ((size_t)N + 7) >> 3;
      size_t g_work[2] = {num_groups_n * 32, (size_t)M * 8};
      if (err == VIRA_CL_SUCCESS) {
        err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, k, 2, NULL,
                                               g_work, l_work, 0, NULL, NULL);
      }
    } else {
      err = _vira_gpu_set_gemm_args(_vira_gpu.kernel_linear_tiled, &x_buf,
                                    &w_buf, &y_buf, M, K, N);
      size_t l_work[2] = {16, 16};
      size_t g_work[2] = {((size_t)N + 15) & ~15ULL, ((size_t)M + 15) & ~15ULL};
      if (err == VIRA_CL_SUCCESS) {
        err = _vira_gpu.clEnqueueNDRangeKernel(
            _vira_gpu.queue, _vira_gpu.kernel_linear_tiled, 2, NULL, g_work,
            l_work, 0, NULL, NULL);
      }
    }
  }

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, out_dims)
                : vira_tensor_f32_alloc_nd(2, out_dims);

  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_buf, 1, 0, y_bytes,
                                        out.data, 0, NULL, NULL);
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  return out;
}

// ============================================================================
// GPU MATRIX MULTIPLICATION: `c = a @ b` (FP32, 16x16 tiled)
// ============================================================================
static inline vira_tensor_f32 vira_gpu_matmul(vira_arena *opt_arena,
                                              vira_tensor_f32 a,
                                              vira_tensor_f32 b) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&a);
  _vira_tensor_norm_shape(&b);

  if (_vira_gpu.simulate_oom) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  int32_t M = a.rows;
  int32_t K = a.cols;
  int32_t N = b.cols;
  int64_t total_ops = (int64_t)M * (int64_t)K * (int64_t)N;
  if (!_vira_gpu.available || !_vira_gpu.enabled || a.n_dims > 2 ||
      b.n_dims > 2 || K != b.rows || total_ops < 65536LL ||
      _vira_gpu.max_wg_size < 256) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  size_t a_bytes = (size_t)M * (size_t)K * sizeof(vira_f32);
  size_t b_bytes = (size_t)K * (size_t)N * sizeof(vira_f32);
  size_t c_bytes = (size_t)M * (size_t)N * sizeof(vira_f32);

  pthread_mutex_lock(&_vira_gpu.lock);
  _vira_gpu.last_x_ptr = NULL;
  vira_cl_int err = 0;

  vira_cl_mem b_buf = _vira_gpu_ensure_scratch_locked(
      &_vira_gpu.scratch_b, &_vira_gpu.scratch_b_cap, b_bytes,
      VIRA_CL_MEM_READ_WRITE, &err);
  if (err == VIRA_CL_SUCCESS && b_buf != NULL) {
    err = _vira_gpu.clEnqueueWriteBuffer(_vira_gpu.queue, b_buf, 0, 0, b_bytes,
                                         b.data, 0, NULL, NULL);
  }
  vira_cl_mem a_buf =
      (err == VIRA_CL_SUCCESS)
          ? _vira_gpu_ensure_scratch_locked(
                &_vira_gpu.scratch_a, &_vira_gpu.scratch_a_cap, a_bytes,
                VIRA_CL_MEM_READ_WRITE, &err)
          : NULL;
  vira_cl_mem c_buf =
      (err == VIRA_CL_SUCCESS)
          ? _vira_gpu_ensure_scratch_locked(
                &_vira_gpu.scratch_c, &_vira_gpu.scratch_c_cap, c_bytes,
                VIRA_CL_MEM_READ_WRITE, &err)
          : NULL;
  if (err != VIRA_CL_SUCCESS || a_buf == NULL || b_buf == NULL ||
      c_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  err = _vira_gpu.clEnqueueWriteBuffer(_vira_gpu.queue, a_buf, 0, 0, a_bytes,
                                       a.data, 0, NULL, NULL);
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu_set_gemm_args(_vira_gpu.kernel_matmul_tiled, &a_buf, &b_buf,
                                  &c_buf, M, K, N);
  }
  size_t l_work[2] = {16, 16};
  size_t g_work[2] = {((size_t)N + 15) & ~15ULL, ((size_t)M + 15) & ~15ULL};
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(
        _vira_gpu.queue, _vira_gpu.kernel_matmul_tiled, 2, NULL, g_work, l_work,
        0, NULL, NULL);
  }

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, out_dims)
                : vira_tensor_f32_alloc_nd(2, out_dims);

  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, c_buf, 1, 0, c_bytes,
                                        out.data, 0, NULL, NULL);
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  return out;
}

static inline vira_tensor_f32 vira_gpu_rmsnorm(vira_tensor_f32 x,
                                               vira_tensor_f32 weight,
                                               float eps) {
  return vira_tensor_rmsnorm(x, weight, eps);
}

static inline vira_tensor_f32 vira_gpu_rope(vira_tensor_f32 q, int64_t pos,
                                            float freq_base) {
  return vira_tensor_rope(q, pos, freq_base);
}

static inline vira_tensor_f32 vira_gpu_softmax(vira_tensor_f32 logits) {
  return vira_tensor_softmax(logits);
}

static inline vira_tensor_f32 vira_gpu_silu(vira_tensor_f32 x) {
  return vira_tensor_silu(x);
}

#endif // VIRA_GPU_H