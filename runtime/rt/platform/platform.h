#ifndef VIRA_RT_PLATFORM_H
#define VIRA_RT_PLATFORM_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef _WIN32
#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#if !defined(WINVER) || WINVER < 0x0600
#undef WINVER
#define WINVER 0x0601
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <direct.h>
#include <sys/stat.h>
#include <process.h>
#include <ctype.h>
typedef SOCKET vira_socket_t;
#define VIRA_INVALID_SOCKET INVALID_SOCKET
#define VIRA_SOCKET_ERROR SOCKET_ERROR
#define vira_poll WSAPoll
#else
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <dirent.h>
#include <ctype.h>
#if defined(__linux__)
#include <sys/mount.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#endif

typedef int vira_socket_t;
#define VIRA_INVALID_SOCKET -1
#define VIRA_SOCKET_ERROR -1
#define closesocket close
#define vira_poll poll
#endif

// --- HARDWARE SIMD INTRINSIC HEADERS ---
#if defined(__AVX2__) || defined(__SSSE3__) || defined(__SSE2__) ||           \
    defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define VIRA_CACHELINE_ALIGNED __attribute__((aligned(64)))
#elif defined(_MSC_VER)
#define VIRA_CACHELINE_ALIGNED __declspec(align(64))
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define VIRA_CACHELINE_ALIGNED _Alignas(64)
#else
#define VIRA_CACHELINE_ALIGNED
#endif

/**
 * Emits an architecture-specific CPU pause or yield instruction.
 * Usage: Reduces CPU pipeline contention and power consumption in spin-wait loops.
 */
static inline void vira_cpu_relax(void) {
#if defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#else
  __asm__ __volatile__("" ::: "memory");
#endif
}

/**
 * Detects the number of online logical CPU hardware threads.
 */
static inline int32_t _vira_detect_logical_cores(void) {
#ifdef _WIN32
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  int32_t n = (int32_t)si.dwNumberOfProcessors;
  return n > 0 ? n : 4;
#else
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int32_t)n : 4;
#endif
}

/**
 * Detects the number of physical CPU cores (excluding SMT/Hyper-Threading siblings when possible).
 */
static inline int32_t _vira_detect_physical_cores(void) {
#ifdef _WIN32
  uint8_t stack_buf[16384];
  DWORD len = (DWORD)sizeof(stack_buf);
  if (GetLogicalProcessorInformationEx(
          RelationProcessorCore,
          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)stack_buf, &len)) {
    int32_t cores = 0;
    DWORD off = 0;
    while (off + 8 <= len) {
      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(stack_buf + off);
      if (info->Size == 0 || off + info->Size > len) {
        break;
      }
      if (info->Relationship == RelationProcessorCore) {
        cores++;
      }
      off += info->Size;
    }
    if (cores > 0) {
      return cores;
    }
  }
#elif defined(__linux__)
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (f) {
    char line[256];
    int64_t seen[256];
    int32_t seen_cnt = 0;
    int32_t cur_phys = -1;
    int32_t cur_core = -1;
    while (fgets(line, (int)sizeof(line), f)) {
      if (strncmp(line, "physical id", 11) == 0) {
        const char *colon = strchr(line, ':');
        if (colon) cur_phys = atoi(colon + 1);
      } else if (strncmp(line, "core id", 7) == 0) {
        const char *colon = strchr(line, ':');
        if (colon) cur_core = atoi(colon + 1);
      } else if (line[0] == '\n' || line[0] == '\r') {
        if (cur_phys >= 0 && cur_core >= 0 && seen_cnt < 256) {
          int64_t key = ((int64_t)cur_phys << 32) | (uint32_t)cur_core;
          bool found = false;
          for (int32_t i = 0; i < seen_cnt; i++) {
            if (seen[i] == key) { found = true; break; }
          }
          if (!found) seen[seen_cnt++] = key;
        }
        cur_phys = -1;
        cur_core = -1;
      }
    }
    if (cur_phys >= 0 && cur_core >= 0 && seen_cnt < 256) {
      int64_t key = ((int64_t)cur_phys << 32) | (uint32_t)cur_core;
      bool found = false;
      for (int32_t i = 0; i < seen_cnt; i++) {
        if (seen[i] == key) { found = true; break; }
      }
      if (!found) seen[seen_cnt++] = key;
    }
    fclose(f);
    if (seen_cnt > 0) {
      return seen_cnt;
    }
  }
#endif
  return _vira_detect_logical_cores();
}

// --- THREAD-SAFE TELEMETRY COUNTERS FOR ZERO-LEAK VERIFICATION ---
static VIRA_CACHELINE_ALIGNED volatile int64_t _vira_active_heap_allocs = 0;
static VIRA_CACHELINE_ALIGNED volatile int64_t _vira_total_heap_frees = 0;
static VIRA_CACHELINE_ALIGNED volatile int64_t _vira_total_tasks_run = 0;
static VIRA_CACHELINE_ALIGNED volatile int64_t _vira_total_work_steals = 0;

static __thread int64_t _vira_tls_alloc_delta = 0;
static __thread int64_t _vira_tls_free_count = 0;

/**
 * Flushes thread-local allocation and deallocation telemetry counters to global atomic totals.
 * Usage: Internal helper called periodically and at thread exit to ensure accurate memory leak tracking.
 */
static inline void _vira_flush_tls_mem_stats(void) {
  if (_vira_tls_alloc_delta != 0) {
    __atomic_add_fetch(&_vira_active_heap_allocs, _vira_tls_alloc_delta, __ATOMIC_RELAXED);
    _vira_tls_alloc_delta = 0;
  }
  if (_vira_tls_free_count != 0) {
    __atomic_add_fetch(&_vira_total_heap_frees, _vira_tls_free_count, __ATOMIC_RELAXED);
    _vira_tls_free_count = 0;
  }
}

/**
 * Records a dynamic memory allocation in thread-local telemetry counters.
 * Usage: Invoked automatically by runtime allocation functions; flushes to global stats every 64 calls.
 */
static inline void _vira_record_alloc(void) {
  if (++_vira_tls_alloc_delta >= 64) {
    _vira_flush_tls_mem_stats();
  }
}

/**
 * Records a dynamic memory deallocation in thread-local telemetry counters.
 * Usage: Invoked automatically by runtime free functions; flushes to global stats every 64 calls.
 */
static inline void _vira_record_free(void) {
  --_vira_tls_alloc_delta;
  if (++_vira_tls_free_count >= 64) {
    _vira_flush_tls_mem_stats();
  }
}

// Runtime fault entry point (bounds errors, division by zero, pop from empty slice, ...).
// Defined in rt/scheduler/scheduler.h: inside an actor it is trapped by the supervisor like an
// explicit `panic(...)`; elsewhere it prints the message and exits.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn, format(printf, 1, 2)))
#endif
static void _vira_rt_fault(const char *fmt, ...);

// --- CHECKED INTEGER DIVISION -------------------------------------------------
// C leaves `x / 0`, `x % 0` and `INT_MIN / -1` undefined (SIGFPE on x86). The compiler lowers
// integer `/` and `%` to these helpers so a zero divisor is a supervised runtime fault and the
// `MIN / -1` overflow wraps like every other integer operation.
static inline int32_t vira_div_i32(int32_t a, int32_t b) {
  if (b == 0) _vira_rt_fault("vira panic: integer division by zero");
  if (b == -1) return (int32_t)(0u - (uint32_t)a);
  return a / b;
}
static inline int32_t vira_mod_i32(int32_t a, int32_t b) {
  if (b == 0) _vira_rt_fault("vira panic: integer modulo by zero");
  if (b == -1) return 0;
  return a % b;
}
static inline int64_t vira_div_i64(int64_t a, int64_t b) {
  if (b == 0) _vira_rt_fault("vira panic: integer division by zero");
  if (b == -1) return (int64_t)(0ull - (uint64_t)a);
  return a / b;
}
static inline int64_t vira_mod_i64(int64_t a, int64_t b) {
  if (b == 0) _vira_rt_fault("vira panic: integer modulo by zero");
  if (b == -1) return 0;
  return a % b;
}
static inline uint64_t vira_div_u64(uint64_t a, uint64_t b) {
  if (b == 0) _vira_rt_fault("vira panic: integer division by zero");
  return a / b;
}
static inline uint64_t vira_mod_u64(uint64_t a, uint64_t b) {
  if (b == 0) _vira_rt_fault("vira panic: integer modulo by zero");
  return a % b;
}

typedef float vira_f32;
typedef uint16_t vira_bf16;

static int _vira_argc = 0;
static char **_vira_argv = NULL;
static bool _vira_quiet_rt = false;

/**
 * Captures process command-line arguments into runtime global storage.
 * @param argc Number of command-line arguments.
 * @param argv Array of argument strings.
 * Usage: Called by the runtime entry point before executing user code.
 */
static inline void _vira_rt_init_args(int argc, char **argv) {
  _vira_argc = argc;
  _vira_argv = argv;
}

/**
 * Releases cached command-line argument structures.
 * Usage: Invoked during runtime teardown.
 */
static inline void _vira_rt_free_args(void) {
  // Global argv pointers are borrowed from OS process heap
}

/**
 * Enables or suppresses runtime telemetry and memory leak reporting at process termination.
 * @param q True to suppress diagnostic reports, false to display them.
 * Usage: Used by CLI utilities and silent tests to disable exit banners (`os::set_quiet_rt(true)`).
 */
static inline void vira_os_set_quiet_rt(bool q) {
  _vira_quiet_rt = q;
}

/**
 * Flushes memory statistics, conditionally prints diagnostic banners, and terminates the process.
 * @param code Process exit status code.
 * Usage: Standard application exit mechanism in Vira programs (`os::exit(code)`).
 */
static inline void vira_os_exit(int32_t code) {
  _vira_rt_free_args();
  exit(code);
}

#endif // VIRA_RT_PLATFORM_H
