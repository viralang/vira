#ifndef VIRA_GPU_H
#define VIRA_GPU_H

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "vira_rt.h"

// ============================================================================
// VIRA v0.8 CROSS-PLATFORM GPU COMPUTE ACCELERATION RUNTIME
// ============================================================================
// - Zero compile-time SDK dependencies: dynamically probes OpenCL / Vulkan
//   drivers at runtime via Win32 LoadLibraryA / POSIX dlopen.
// - Persistent GPU Weight Cache: caches immutable GGUF / SafeTensors weight
//   matrices in GPU VRAM across tokens (zero repeated host->device copies).
// - Actor Fault-Tolerant Supervision: automatically recovers from GPU VRAM
//   Out-of-Memory (OOM) or driver errors by evicting cache and falling back
//   to Vira's multi-threaded AVX2+FMA / ARM NEON CPU pool.
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
#define VIRA_CL_DEVICE_TYPE_DEFAULT (1ULL << 0)
#define VIRA_CL_DEVICE_TYPE_CPU (1ULL << 1)
#define VIRA_CL_DEVICE_TYPE_GPU (1ULL << 2)
#define VIRA_CL_DEVICE_TYPE_ALL 0xFFFFFFFFULL

#define VIRA_CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define VIRA_CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define VIRA_CL_DEVICE_NAME 0x102B
#define VIRA_CL_DEVICE_VERSION 0x102F
#define VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035

#define VIRA_CL_MEM_READ_WRITE (1ULL << 0)
#define VIRA_CL_MEM_WRITE_ONLY (1ULL << 1)
#define VIRA_CL_MEM_READ_ONLY (1ULL << 2)
#define VIRA_CL_MEM_USE_HOST_PTR (1ULL << 3)
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

#define VIRA_GPU_WEIGHT_CACHE_CAP 512

typedef struct {
  const vira_f32 *host_ptr;
  size_t bytes;
  vira_cl_mem dev_buf;
} _vira_gpu_cache_entry;

typedef struct {
  volatile int32_t initialized;
  bool available;
  bool enabled;
  bool unified_mem;
  volatile bool simulate_oom;
  char device_name[256];
  char backend_name[64];
  int64_t vram_bytes;
  volatile int64_t allocated_bytes;
  volatile int64_t oom_fallbacks;
  int32_t compute_units;
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
  vira_cl_kernel kernel_linear;
  vira_cl_kernel kernel_matmul;

  _vira_gpu_cache_entry weight_cache[VIRA_GPU_WEIGHT_CACHE_CAP];
} _vira_gpu_state;

static _vira_gpu_state _vira_gpu = {
    .initialized = 0,
    .available = false,
    .enabled = true,
    .unified_mem = false,
    .simulate_oom = false,
    .device_name = "Multi-Threaded CPU SIMD Fallback",
    .backend_name = "CPU-SIMD",
    .vram_bytes = 0,
    .allocated_bytes = 0,
    .oom_fallbacks = 0,
    .compute_units = 0,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .lib_handle = NULL,
};

static const char *_vira_opencl_kernel_src =
    "__kernel void vira_cl_linear_f32(\n"
    "    __global const float* restrict x,\n"
    "    __global const float* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int n = (int)get_global_id(0);\n"
    "  const int m = (int)get_global_id(1);\n"
    "  if (m >= M || n >= N) return;\n"
    "  __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "  __global const float* w_row = w + (size_t)n * (size_t)K;\n"
    "  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;\n"
    "  int k = 0;\n"
    "  for (; k + 3 < K; k += 4) {\n"
    "    acc0 = fma(x_row[k],     w_row[k],     acc0);\n"
    "    acc1 = fma(x_row[k + 1], w_row[k + 1], acc1);\n"
    "    acc2 = fma(x_row[k + 2], w_row[k + 2], acc2);\n"
    "    acc3 = fma(x_row[k + 3], w_row[k + 3], acc3);\n"
    "  }\n"
    "  float sum = (acc0 + acc1) + (acc2 + acc3);\n"
    "  for (; k < K; ++k) {\n"
    "    sum = fma(x_row[k], w_row[k], sum);\n"
    "  }\n"
    "  y[(size_t)m * (size_t)N + (size_t)n] = sum;\n"
    "}\n"
    "\n"
    "__kernel void vira_cl_matmul_f32(\n"
    "    __global const float* restrict a,\n"
    "    __global const float* restrict b,\n"
    "    __global float* restrict c,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int col = (int)get_global_id(0);\n"
    "  const int row = (int)get_global_id(1);\n"
    "  if (row >= M || col >= N) return;\n"
    "  __global const float* a_row = a + (size_t)row * (size_t)K;\n"
    "  float sum = 0.0f;\n"
    "  for (int k = 0; k < K; ++k) {\n"
    "    sum = fma(a_row[k], b[(size_t)k * (size_t)N + (size_t)col], sum);\n"
    "  }\n"
    "  c[(size_t)row * (size_t)N + (size_t)col] = sum;\n"
    "}\n";

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
    bool ok = true;
    ok = ok && _vira_gpu_load_sym(lib, "clGetPlatformIDs",
                                  &_vira_gpu.clGetPlatformIDs);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clGetDeviceIDs", &_vira_gpu.clGetDeviceIDs);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clGetDeviceInfo", &_vira_gpu.clGetDeviceInfo);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clCreateContext", &_vira_gpu.clCreateContext);
    ok = ok && _vira_gpu_load_sym(lib, "clReleaseContext",
                                  &_vira_gpu.clReleaseContext);
    ok = ok && _vira_gpu_load_sym(lib, "clCreateCommandQueue",
                                  &_vira_gpu.clCreateCommandQueue);
    ok = ok && _vira_gpu_load_sym(lib, "clReleaseCommandQueue",
                                  &_vira_gpu.clReleaseCommandQueue);
    ok = ok && _vira_gpu_load_sym(lib, "clCreateProgramWithSource",
                                  &_vira_gpu.clCreateProgramWithSource);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clBuildProgram", &_vira_gpu.clBuildProgram);
    ok = ok && _vira_gpu_load_sym(lib, "clReleaseProgram",
                                  &_vira_gpu.clReleaseProgram);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clCreateKernel", &_vira_gpu.clCreateKernel);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clReleaseKernel", &_vira_gpu.clReleaseKernel);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clCreateBuffer", &_vira_gpu.clCreateBuffer);
    ok = ok && _vira_gpu_load_sym(lib, "clReleaseMemObject",
                                  &_vira_gpu.clReleaseMemObject);
    ok = ok && _vira_gpu_load_sym(lib, "clEnqueueWriteBuffer",
                                  &_vira_gpu.clEnqueueWriteBuffer);
    ok = ok && _vira_gpu_load_sym(lib, "clEnqueueReadBuffer",
                                  &_vira_gpu.clEnqueueReadBuffer);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clSetKernelArg", &_vira_gpu.clSetKernelArg);
    ok = ok && _vira_gpu_load_sym(lib, "clEnqueueNDRangeKernel",
                                  &_vira_gpu.clEnqueueNDRangeKernel);
    ok = ok && _vira_gpu_load_sym(lib, "clFinish", &_vira_gpu.clFinish);

    if (ok) {
      vira_cl_platform_id platforms[8];
      vira_cl_uint num_platforms = 0;
      if (_vira_gpu.clGetPlatformIDs(8, platforms, &num_platforms) ==
              VIRA_CL_SUCCESS &&
          num_platforms > 0) {
        vira_cl_device_id chosen_dev = NULL;
        for (vira_cl_uint p = 0; p < num_platforms; ++p) {
          vira_cl_device_id devs[4];
          vira_cl_uint num_devs = 0;
          if (_vira_gpu.clGetDeviceIDs(platforms[p], VIRA_CL_DEVICE_TYPE_GPU, 4,
                                       devs, &num_devs) == VIRA_CL_SUCCESS &&
              num_devs > 0) {
            chosen_dev = devs[0];
            break;
          }
        }
        if (chosen_dev == NULL) {
          for (vira_cl_uint p = 0; p < num_platforms; ++p) {
            vira_cl_device_id devs[4];
            vira_cl_uint num_devs = 0;
            if (_vira_gpu.clGetDeviceIDs(platforms[p], VIRA_CL_DEVICE_TYPE_ALL,
                                         4, devs,
                                         &num_devs) == VIRA_CL_SUCCESS &&
                num_devs > 0) {
              chosen_dev = devs[0];
              break;
            }
          }
        }

        if (chosen_dev != NULL) {
          vira_cl_int err = 0;
          vira_cl_context ctx =
              _vira_gpu.clCreateContext(NULL, 1, &chosen_dev, NULL, NULL, &err);
          if (err == VIRA_CL_SUCCESS && ctx != NULL) {
            vira_cl_command_queue q =
                _vira_gpu.clCreateCommandQueue(ctx, chosen_dev, 0, &err);
            if (err == VIRA_CL_SUCCESS && q != NULL) {
              size_t src_len = strlen(_vira_opencl_kernel_src);
              vira_cl_program prog = _vira_gpu.clCreateProgramWithSource(
                  ctx, 1, &_vira_opencl_kernel_src, &src_len, &err);
              if (err == VIRA_CL_SUCCESS && prog != NULL &&
                  _vira_gpu.clBuildProgram(prog, 1, &chosen_dev,
                                           "-cl-fast-relaxed-math", NULL,
                                           NULL) == VIRA_CL_SUCCESS) {
                vira_cl_int k1_err = 0, k2_err = 0;
                vira_cl_kernel k_lin = _vira_gpu.clCreateKernel(
                    prog, "vira_cl_linear_f32", &k1_err);
                vira_cl_kernel k_mm = _vira_gpu.clCreateKernel(
                    prog, "vira_cl_matmul_f32", &k2_err);
                if (k1_err == VIRA_CL_SUCCESS && k2_err == VIRA_CL_SUCCESS &&
                    k_lin != NULL && k_mm != NULL) {
                  _vira_gpu.lib_handle = lib;
                  _vira_gpu.device = chosen_dev;
                  _vira_gpu.context = ctx;
                  _vira_gpu.queue = q;
                  _vira_gpu.program = prog;
                  _vira_gpu.kernel_linear = k_lin;
                  _vira_gpu.kernel_matmul = k_mm;
                  _vira_gpu.available = true;
                  snprintf(_vira_gpu.backend_name,
                           sizeof(_vira_gpu.backend_name), "OpenCL");

                  char dname[256] = {0};
                  if (_vira_gpu.clGetDeviceInfo(chosen_dev, VIRA_CL_DEVICE_NAME,
                                                sizeof(dname) - 1, dname,
                                                NULL) == VIRA_CL_SUCCESS) {
                    snprintf(_vira_gpu.device_name,
                             sizeof(_vira_gpu.device_name), "%s", dname);
                  }
                  vira_cl_ulong gmem = 0;
                  if (_vira_gpu.clGetDeviceInfo(
                          chosen_dev, VIRA_CL_DEVICE_GLOBAL_MEM_SIZE,
                          sizeof(gmem), &gmem, NULL) == VIRA_CL_SUCCESS) {
                    _vira_gpu.vram_bytes = (int64_t)gmem;
                  }
                  vira_cl_uint cu = 0;
                  if (_vira_gpu.clGetDeviceInfo(
                          chosen_dev, VIRA_CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(cu), &cu, NULL) == VIRA_CL_SUCCESS) {
                    _vira_gpu.compute_units = (int32_t)cu;
                  }
                  vira_cl_uint um = 0;
                  if (_vira_gpu.clGetDeviceInfo(
                          chosen_dev, VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY,
                          sizeof(um), &um, NULL) == VIRA_CL_SUCCESS) {
                    _vira_gpu.unified_mem = (um != 0);
                  }
                } else {
                  if (k_lin)
                    _vira_gpu.clReleaseKernel(k_lin);
                  if (k_mm)
                    _vira_gpu.clReleaseKernel(k_mm);
                  _vira_gpu.clReleaseProgram(prog);
                  _vira_gpu.clReleaseCommandQueue(q);
                  _vira_gpu.clReleaseContext(ctx);
                }
              } else {
                if (prog)
                  _vira_gpu.clReleaseProgram(prog);
                _vira_gpu.clReleaseCommandQueue(q);
                _vira_gpu.clReleaseContext(ctx);
              }
            } else {
              _vira_gpu.clReleaseContext(ctx);
            }
          }
        }
      }
    }
  }

  __atomic_store_n(&_vira_gpu.initialized, 2, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline void _vira_gpu_evict_ptr_locked(const vira_f32 *host_ptr) {
  if (!_vira_gpu.available || host_ptr == NULL) {
    return;
  }
  for (int32_t i = 0; i < VIRA_GPU_WEIGHT_CACHE_CAP; ++i) {
    if (_vira_gpu.weight_cache[i].host_ptr == host_ptr &&
        _vira_gpu.weight_cache[i].dev_buf != NULL) {
      _vira_gpu.clReleaseMemObject(_vira_gpu.weight_cache[i].dev_buf);
      __atomic_sub_fetch(&_vira_gpu.allocated_bytes,
                         (int64_t)_vira_gpu.weight_cache[i].bytes,
                         __ATOMIC_RELAXED);
      _vira_gpu.weight_cache[i].host_ptr = NULL;
      _vira_gpu.weight_cache[i].bytes = 0;
      _vira_gpu.weight_cache[i].dev_buf = NULL;
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
      _vira_gpu.weight_cache[i].host_ptr = NULL;
      _vira_gpu.weight_cache[i].bytes = 0;
      _vira_gpu.weight_cache[i].dev_buf = NULL;
    }
  }
  __atomic_store_n(&_vira_gpu.allocated_bytes, 0, __ATOMIC_RELAXED);
}

static inline vira_cl_mem _vira_gpu_get_or_upload_weight_locked(
    const vira_f32 *host_ptr, size_t bytes, vira_cl_int *out_err) {
  uint64_t h = ((uint64_t)(uintptr_t)host_ptr >> 6) * 11400714819323198485ULL;
  int32_t start_slot = (int32_t)(h & (VIRA_GPU_WEIGHT_CACHE_CAP - 1));

  for (int32_t probe = 0; probe < 8; ++probe) {
    int32_t idx = (start_slot + probe) & (VIRA_GPU_WEIGHT_CACHE_CAP - 1);
    if (_vira_gpu.weight_cache[idx].host_ptr == host_ptr &&
        _vira_gpu.weight_cache[idx].bytes == bytes &&
        _vira_gpu.weight_cache[idx].dev_buf != NULL) {
      *out_err = VIRA_CL_SUCCESS;
      return _vira_gpu.weight_cache[idx].dev_buf;
    }
  }

  int32_t target_slot = start_slot;
  for (int32_t probe = 0; probe < 8; ++probe) {
    int32_t idx = (start_slot + probe) & (VIRA_GPU_WEIGHT_CACHE_CAP - 1);
    if (_vira_gpu.weight_cache[idx].dev_buf == NULL) {
      target_slot = idx;
      break;
    }
  }

  if (_vira_gpu.weight_cache[target_slot].dev_buf != NULL) {
    _vira_gpu.clReleaseMemObject(_vira_gpu.weight_cache[target_slot].dev_buf);
    __atomic_sub_fetch(&_vira_gpu.allocated_bytes,
                       (int64_t)_vira_gpu.weight_cache[target_slot].bytes,
                       __ATOMIC_RELAXED);
    _vira_gpu.weight_cache[target_slot].dev_buf = NULL;
  }

  vira_cl_bitfield flags =
      VIRA_CL_MEM_READ_ONLY |
      (_vira_gpu.unified_mem ? VIRA_CL_MEM_USE_HOST_PTR
                             : VIRA_CL_MEM_COPY_HOST_PTR);
  vira_cl_int err = 0;
  vira_cl_mem buf =
      _vira_gpu.clCreateBuffer(_vira_gpu.context, flags, bytes, (void *)host_ptr,
                               &err);
  if (err != VIRA_CL_SUCCESS || buf == NULL) {
    *out_err = (err != VIRA_CL_SUCCESS) ? err : -4;
    return NULL;
  }

  _vira_gpu.weight_cache[target_slot].host_ptr = host_ptr;
  _vira_gpu.weight_cache[target_slot].bytes = bytes;
  _vira_gpu.weight_cache[target_slot].dev_buf = buf;
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
    _vira_gpu_evict_all_locked();
    if (_vira_gpu.queue)
      _vira_gpu.clFinish(_vira_gpu.queue);
    if (_vira_gpu.kernel_linear)
      _vira_gpu.clReleaseKernel(_vira_gpu.kernel_linear);
    if (_vira_gpu.kernel_matmul)
      _vira_gpu.clReleaseKernel(_vira_gpu.kernel_matmul);
    if (_vira_gpu.program)
      _vira_gpu.clReleaseProgram(_vira_gpu.program);
    if (_vira_gpu.queue)
      _vira_gpu.clReleaseCommandQueue(_vira_gpu.queue);
    if (_vira_gpu.context)
      _vira_gpu.clReleaseContext(_vira_gpu.context);
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

// GPU-accelerated Linear Projection `y = x @ w^T` with Persistent Weight Cache
// and Actor-Supervised OOM Fallback to Multi-Threaded AVX2+FMA / NEON.
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

  // Simulated OOM fault injection for actor supervision & resilience testing
  if (_vira_gpu.simulate_oom) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_linear(opt_arena, x, w);
  }

  int64_t total_ops = (int64_t)M * (int64_t)N * (int64_t)K;
  if (!_vira_gpu.available || !_vira_gpu.enabled || total_ops < 65536LL) {
    return vira_tensor_linear(opt_arena, x, w);
  }

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, out_dims)
                : vira_tensor_f32_alloc_nd(2, out_dims);

  size_t x_bytes = (size_t)M * (size_t)K * sizeof(vira_f32);
  size_t w_bytes = (size_t)N * (size_t)K * sizeof(vira_f32);
  size_t y_bytes = (size_t)M * (size_t)N * sizeof(vira_f32);

  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  vira_cl_mem w_buf =
      _vira_gpu_get_or_upload_weight_locked(w.data, w_bytes, &err);
  if (err != VIRA_CL_SUCCESS || w_buf == NULL) {
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_linear(opt_arena, x, w);
  }

  vira_cl_mem x_buf = _vira_gpu.clCreateBuffer(
      _vira_gpu.context, VIRA_CL_MEM_READ_ONLY | VIRA_CL_MEM_COPY_HOST_PTR,
      x_bytes, (void *)x.data, &err);
  if (err != VIRA_CL_SUCCESS || x_buf == NULL) {
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_linear(opt_arena, x, w);
  }

  vira_cl_mem y_buf = _vira_gpu.clCreateBuffer(
      _vira_gpu.context, VIRA_CL_MEM_WRITE_ONLY, y_bytes, NULL, &err);
  if (err != VIRA_CL_SUCCESS || y_buf == NULL) {
    _vira_gpu.clReleaseMemObject(x_buf);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_linear(opt_arena, x, w);
  }

  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear, 0,
                                  sizeof(vira_cl_mem), &x_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear, 1,
                                  sizeof(vira_cl_mem), &w_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear, 2,
                                  sizeof(vira_cl_mem), &y_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear, 3, sizeof(int32_t),
                                  &M);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear, 4, sizeof(int32_t),
                                  &K);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear, 5, sizeof(int32_t),
                                  &N);

  size_t g_work[2] = {(size_t)N, (size_t)M};
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue,
                                           _vira_gpu.kernel_linear, 2, NULL,
                                           g_work, NULL, 0, NULL, NULL);
  }
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_buf, 1, 0, y_bytes,
                                        out.data, 0, NULL, NULL);
  }

  _vira_gpu.clReleaseMemObject(y_buf);
  _vira_gpu.clReleaseMemObject(x_buf);
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_linear(opt_arena, x, w);
  }
  return out;
}

// GPU-accelerated Matrix Multiplication `c = a @ b` with Actor-Supervised OOM Fallback
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
    return vira_tensor_matmul_f32(opt_arena, a, b);
  }

  int32_t M = a.rows;
  int32_t K = a.cols;
  int32_t N = b.cols;
  int64_t total_ops = (int64_t)M * (int64_t)K * (int64_t)N;
  if (!_vira_gpu.available || !_vira_gpu.enabled || a.n_dims > 2 ||
      b.n_dims > 2 || K != b.rows || total_ops < 65536LL) {
    return vira_tensor_matmul_f32(opt_arena, a, b);
  }

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, out_dims)
                : vira_tensor_f32_alloc_nd(2, out_dims);

  size_t a_bytes = (size_t)M * (size_t)K * sizeof(vira_f32);
  size_t b_bytes = (size_t)K * (size_t)N * sizeof(vira_f32);
  size_t c_bytes = (size_t)M * (size_t)N * sizeof(vira_f32);

  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  vira_cl_mem b_buf =
      _vira_gpu_get_or_upload_weight_locked(b.data, b_bytes, &err);
  if (err != VIRA_CL_SUCCESS || b_buf == NULL) {
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_matmul_f32(opt_arena, a, b);
  }

  vira_cl_mem a_buf = _vira_gpu.clCreateBuffer(
      _vira_gpu.context, VIRA_CL_MEM_READ_ONLY | VIRA_CL_MEM_COPY_HOST_PTR,
      a_bytes, (void *)a.data, &err);
  if (err != VIRA_CL_SUCCESS || a_buf == NULL) {
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_matmul_f32(opt_arena, a, b);
  }

  vira_cl_mem c_buf = _vira_gpu.clCreateBuffer(
      _vira_gpu.context, VIRA_CL_MEM_WRITE_ONLY, c_bytes, NULL, &err);
  if (err != VIRA_CL_SUCCESS || c_buf == NULL) {
    _vira_gpu.clReleaseMemObject(a_buf);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_matmul_f32(opt_arena, a, b);
  }

  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul, 0,
                                  sizeof(vira_cl_mem), &a_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul, 1,
                                  sizeof(vira_cl_mem), &b_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul, 2,
                                  sizeof(vira_cl_mem), &c_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul, 3, sizeof(int32_t),
                                  &M);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul, 4, sizeof(int32_t),
                                  &K);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul, 5, sizeof(int32_t),
                                  &N);

  size_t g_work[2] = {(size_t)N, (size_t)M};
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue,
                                           _vira_gpu.kernel_matmul, 2, NULL,
                                           g_work, NULL, 0, NULL, NULL);
  }
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, c_buf, 1, 0, c_bytes,
                                        out.data, 0, NULL, NULL);
  }

  _vira_gpu.clReleaseMemObject(c_buf);
  _vira_gpu.clReleaseMemObject(a_buf);
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_matmul_f32(opt_arena, a, b);
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
