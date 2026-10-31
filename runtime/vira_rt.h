#ifndef VIRA_RT_H
#define VIRA_RT_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
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
#if defined(__AVX2__) || defined(__SSE2__) || defined(__x86_64__) ||           \
    defined(_M_X64)
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

static inline void vira_cpu_relax(void) {
#if defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#else
  __asm__ __volatile__("" ::: "memory");
#endif
}

// --- THREAD-SAFE TELEMETRY COUNTERS FOR ZERO-LEAK VERIFICATION ---
static VIRA_CACHELINE_ALIGNED volatile int64_t _vira_active_heap_allocs = 0;
static VIRA_CACHELINE_ALIGNED volatile int64_t _vira_total_heap_frees = 0;
static VIRA_CACHELINE_ALIGNED volatile int64_t _vira_total_tasks_run = 0;
static VIRA_CACHELINE_ALIGNED volatile int64_t _vira_total_work_steals = 0;

static __thread int64_t _vira_tls_alloc_delta = 0;
static __thread int64_t _vira_tls_free_count = 0;

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

static inline void _vira_record_alloc(void) {
  if (++_vira_tls_alloc_delta >= 64) {
    _vira_flush_tls_mem_stats();
  }
}

static inline void _vira_record_free(void) {
  --_vira_tls_alloc_delta;
  if (++_vira_tls_free_count >= 64) {
    _vira_flush_tls_mem_stats();
  }
}

typedef float vira_f32;
typedef uint16_t vira_bf16;

static int _vira_argc = 0;
static char **_vira_argv = NULL;
static bool _vira_quiet_rt = false;

static inline void _vira_rt_init_args(int argc, char **argv) {
  _vira_argc = argc;
  _vira_argv = argv;
}

static inline void _vira_rt_free_args(void) {
  // Global argv pointers are borrowed from OS process heap
}

static inline void vira_os_set_quiet_rt(bool q) {
  _vira_quiet_rt = q;
}

static inline void vira_os_exit(int32_t code) {
  _vira_rt_free_args();
  exit(code);
}

// ============================================================================
// 1. VEX STRING (16 BYTES: ZERO-ALLOC LITERALS + HEAP STRINGS)
// ============================================================================

typedef struct {
  const uint8_t *data;
  int32_t len;
  bool is_lit;
} vira_string;

static inline vira_string vira_str_lit(const char *cstr, int32_t len) {
  return (vira_string){
      .data = (const uint8_t *)cstr, .len = len, .is_lit = true};
}

static inline bool vira_str_eq(vira_string a, vira_string b) {
  if (a.len != b.len)
    return false;
  if (a.len == 0)
    return true;
  if (a.data == b.data)
    return true;
  if (a.data == NULL || b.data == NULL)
    return false;
  return memcmp(a.data, b.data, (size_t)a.len) == 0;
}

static inline int32_t vira_str_cmp(vira_string a, vira_string b) {
  int32_t min_len = a.len < b.len ? a.len : b.len;
  if (min_len > 0 && a.data && b.data) {
    int res = memcmp(a.data, b.data, (size_t)min_len);
    if (res != 0) return res < 0 ? -1 : 1;
  }
  if (a.len < b.len) return -1;
  if (a.len > b.len) return 1;
  return 0;
}

static inline vira_string vira_str_format(const char *fmt, ...) {
  va_list args, args_copy;
  va_start(args, fmt);
  va_copy(args_copy, args);
  int32_t needed = vsnprintf(NULL, 0, fmt, args);
  va_end(args);

  if (needed <= 0) {
    va_end(args_copy);
    return vira_str_lit("", 0);
  }

  char *buf = (char *)malloc((size_t)needed + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in vira_str_format\n");
    exit(1);
  }
  vsnprintf(buf, (size_t)needed + 1, fmt, args_copy);
  va_end(args_copy);

  _vira_record_alloc();
  return (vira_string){
      .data = (const uint8_t *)buf, .len = needed, .is_lit = false};
}

static inline vira_string vira_str_clone(vira_string s) {
  if (s.is_lit || s.data == NULL || s.len <= 0) {
    return s;
  }
  uint8_t *copy = (uint8_t *)malloc((size_t)s.len + 1);
  if (!copy) {
    fprintf(stderr, "vira panic: out of memory in vira_str_clone\n");
    exit(1);
  }
  memcpy(copy, s.data, (size_t)s.len);
  copy[s.len] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = copy, .len = s.len, .is_lit = false};
}

static inline void vira_str_free(vira_string *s) {
  if (s != NULL && !s->is_lit && s->data != NULL) {
    free((void *)s->data);
    s->data = NULL;
    s->len = 0;
    s->is_lit = true;
    _vira_record_free();
  }
}

// --- C FFI NUL-TERMINATED STRING MARSHALING HELPERS ---
// Fast-path uses caller-provided stack buffer when `s.len + 1 <= stack_cap`;
// falls back to exact-sized heap allocation for arbitrarily long paths/strings.
typedef struct {
  char *ptr;
  bool heap;
} vira_cstr_buf;

static inline vira_cstr_buf vira_str_to_cstr_buf(vira_string s, char *stack_buf,
                                                 size_t stack_cap) {
  size_t len = (s.data != NULL && s.len > 0) ? (size_t)s.len : 0;
  if (stack_buf != NULL && len + 1 <= stack_cap) {
    if (len > 0) {
      memcpy(stack_buf, s.data, len);
    }
    stack_buf[len] = '\0';
    return (vira_cstr_buf){.ptr = stack_buf, .heap = false};
  }
  char *h = (char *)malloc(len + 1);
  if (!h) {
    fprintf(stderr, "vira panic: out of memory in vira_str_to_cstr_buf\n");
    exit(1);
  }
  if (len > 0) {
    memcpy(h, s.data, len);
  }
  h[len] = '\0';
  _vira_record_alloc();
  return (vira_cstr_buf){.ptr = h, .heap = true};
}

static inline void vira_cstr_buf_free(vira_cstr_buf *b) {
  if (b != NULL && b->heap && b->ptr != NULL) {
    free(b->ptr);
    b->ptr = NULL;
    b->heap = false;
    _vira_record_free();
  }
}

// Public C FFI helper: allocates a NUL-terminated heap C string copy of `s`.
// Release with `vira_cstr_free(ptr)`.
static inline char *vira_str_cstr(vira_string s) {
  size_t len = (s.data != NULL && s.len > 0) ? (size_t)s.len : 0;
  char *h = (char *)malloc(len + 1);
  if (!h) {
    fprintf(stderr, "vira panic: out of memory in vira_str_cstr\n");
    exit(1);
  }
  if (len > 0) {
    memcpy(h, s.data, len);
  }
  h[len] = '\0';
  _vira_record_alloc();
  return h;
}

static inline void vira_cstr_free(char *cstr) {
  if (cstr != NULL) {
    free(cstr);
    _vira_record_free();
  }
}

static inline vira_string vira_str_from_cstr(const char *cstr) {
  if (cstr == NULL) return vira_str_lit("", 0);
  size_t len = strlen(cstr);
  char *buf = (char *)malloc(len + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in vira_str_from_cstr\n");
    exit(1);
  }
  memcpy(buf, cstr, len);
  buf[len] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = (const uint8_t *)buf, .len = (int32_t)len, .is_lit = false};
}

// ============================================================================
// 1.5. VIRA DYNAMIC SLICE RUNTIME (`[]T`) & FIXED ARRAYS (`[N]T`)
// ============================================================================

static inline void vira_bounds_check(int32_t index, int32_t len,
                                     const char *file, int line) {
  if (index < 0 || index >= len) {
    fprintf(stderr,
            "%s:%d: vira panic: index out of bounds: index %d with len %d\n",
            file, line, index, len);
    exit(1);
  }
}

static inline void _vira_slice_grow(void **data_ptr, int32_t *cap_ptr,
                                    size_t elem_size, bool is_arena) {
  int32_t old_cap = *cap_ptr;
  int32_t new_cap = (old_cap <= 0) ? 4 : (old_cap * 2);
  void *new_buf;
  if (old_cap == 0 || *data_ptr == NULL) {
    new_buf = malloc((size_t)new_cap * elem_size);
    if (new_buf)
      _vira_record_alloc();
  } else {
    new_buf = realloc(*data_ptr, (size_t)new_cap * elem_size);
  }
  if (!new_buf) {
    fprintf(stderr, "vira panic: out of memory growing slice buffer\n");
    exit(1);
  }
  *data_ptr = new_buf;
  *cap_ptr = new_cap;
}

static inline void _vira_slice_free_raw(void **data_ptr, int32_t *len_ptr,
                                        int32_t *cap_ptr, bool is_arena) {
  if (data_ptr != NULL && *data_ptr != NULL && !is_arena) {
    free(*data_ptr);
    *data_ptr = NULL;
    _vira_record_free();
  }
  if (len_ptr)
    *len_ptr = 0;
  if (cap_ptr)
    *cap_ptr = 0;
}

#define VIRA_DEFINE_SLICE_TYPE(TYPE_NAME, C_TYPE)                              \
  typedef struct {                                                             \
    C_TYPE *data;                                                              \
    int32_t len;                                                               \
    int32_t cap;                                                               \
    bool is_arena;                                                             \
  } vira_slice_##TYPE_NAME;                                                    \
                                                                               \
  static inline vira_slice_##TYPE_NAME vira_slice_##TYPE_NAME##_new(           \
      int32_t initial_cap) {                                                   \
    vira_slice_##TYPE_NAME s = {                                               \
        .data = NULL, .len = 0, .cap = 0, .is_arena = false};                  \
    if (initial_cap > 0) {                                                     \
      s.data = (C_TYPE *)malloc((size_t)initial_cap * sizeof(C_TYPE));         \
      if (s.data) {                                                            \
        s.cap = initial_cap;                                                   \
        _vira_record_alloc();                                                  \
      }                                                                        \
    }                                                                          \
    return s;                                                                  \
  }                                                                            \
                                                                               \
  static inline void vira_slice_##TYPE_NAME##_push(vira_slice_##TYPE_NAME *s,  \
                                                   C_TYPE val) {               \
    if (s->len >= s->cap) {                                                    \
      _vira_slice_grow((void **)&s->data, &s->cap, sizeof(C_TYPE),             \
                       s->is_arena);                                           \
    }                                                                          \
    s->data[s->len++] = val;                                                   \
  }                                                                            \
                                                                               \
  static inline C_TYPE vira_slice_##TYPE_NAME##_pop(                           \
      vira_slice_##TYPE_NAME *s) {                                             \
    if (s->len <= 0) {                                                         \
      fprintf(stderr, "vira panic: pop from empty slice\n");                   \
      exit(1);                                                                 \
    }                                                                          \
    return s->data[--s->len];                                                  \
  }                                                                            \
                                                                               \
  static inline void vira_slice_##TYPE_NAME##_set(vira_slice_##TYPE_NAME *s,   \
                                                  int32_t idx, C_TYPE val) {   \
    vira_bounds_check(idx, s->len, __FILE__, __LINE__);                        \
    s->data[idx] = val;                                                        \
  }                                                                            \
                                                                               \
  static inline void vira_slice_##TYPE_NAME##_clear(                           \
      vira_slice_##TYPE_NAME *s) {                                             \
    s->len = 0;                                                                \
  }                                                                            \
                                                                               \
  static inline bool vira_slice_##TYPE_NAME##_is_empty(                        \
      const vira_slice_##TYPE_NAME *s) {                                       \
    return s->len == 0;                                                        \
  }                                                                            \
                                                                               \
  static inline vira_slice_##TYPE_NAME vira_slice_##TYPE_NAME##_clone(         \
      const vira_slice_##TYPE_NAME *s) {                                       \
    vira_slice_##TYPE_NAME res = vira_slice_##TYPE_NAME##_new(s->len);         \
    if (s->len > 0 && s->data != NULL) {                                       \
      memcpy(res.data, s->data, (size_t)s->len * sizeof(C_TYPE));              \
      res.len = s->len;                                                        \
    }                                                                          \
    return res;                                                                \
  }                                                                            \
                                                                               \
  static inline vira_slice_##TYPE_NAME vira_slice_##TYPE_NAME##_subslice(      \
      const vira_slice_##TYPE_NAME *s, int32_t start, int32_t end) {          \
    if (start < 0) start = 0;                                                  \
    if (end > s->len) end = s->len;                                            \
    if (start > end) start = end;                                              \
    int32_t count = end - start;                                               \
    vira_slice_##TYPE_NAME res = vira_slice_##TYPE_NAME##_new(count);          \
    if (count > 0 && s->data != NULL) {                                        \
      memcpy(res.data, s->data + start, (size_t)count * sizeof(C_TYPE));       \
      res.len = count;                                                         \
    }                                                                          \
    return res;                                                                \
  }                                                                            \
                                                                               \
  static inline bool vira_slice_##TYPE_NAME##_contains(                        \
      const vira_slice_##TYPE_NAME *s, C_TYPE val) {                           \
    for (int32_t i = 0; i < s->len; i++) {                                     \
      if (s->data[i] == val)                                                   \
        return true;                                                           \
    }                                                                          \
    return false;                                                              \
  }                                                                            \
                                                                               \
  static inline void vira_slice_##TYPE_NAME##_free(                            \
      vira_slice_##TYPE_NAME *s) {                                             \
    _vira_slice_free_raw((void **)&s->data, &s->len, &s->cap, s->is_arena);    \
  }

#ifndef _vira_slice_u8_DEFINED
#define _vira_slice_u8_DEFINED
VIRA_DEFINE_SLICE_TYPE(u8, uint8_t)
#endif
#ifndef _vira_slice_int_DEFINED
#define _vira_slice_int_DEFINED
VIRA_DEFINE_SLICE_TYPE(int, int32_t)
#endif
#ifndef _vira_slice_i64_DEFINED
#define _vira_slice_i64_DEFINED
VIRA_DEFINE_SLICE_TYPE(i64, int64_t)
#endif
#ifndef _vira_slice_f32_DEFINED
#define _vira_slice_f32_DEFINED
VIRA_DEFINE_SLICE_TYPE(f32, vira_f32)
#endif
#ifndef _vira_slice_f64_DEFINED
#define _vira_slice_f64_DEFINED
VIRA_DEFINE_SLICE_TYPE(f64, double)
#endif
#ifndef _vira_slice_bool_DEFINED
#define _vira_slice_bool_DEFINED
VIRA_DEFINE_SLICE_TYPE(bool, bool)
#endif

static inline vira_string vira_str_char_at(vira_string s, int32_t idx) {
  vira_bounds_check(idx, s.len, __FILE__, __LINE__);
  return (vira_string){.data = &s.data[idx], .len = 1, .is_lit = true};
}

#ifndef _vira_slice_string_DEFINED
#define _vira_slice_string_DEFINED
// Slice for vira_string
typedef struct {
  vira_string *data;
  int32_t len;
  int32_t cap;
  bool is_arena;
} vira_slice_string;
#endif

static inline vira_slice_string vira_slice_string_new(int32_t initial_cap) {
  vira_slice_string s = {.data = NULL, .len = 0, .cap = 0, .is_arena = false};
  if (initial_cap > 0) {
    s.data = (vira_string *)malloc((size_t)initial_cap * sizeof(vira_string));
    if (s.data) {
      s.cap = initial_cap;
      _vira_record_alloc();
    }
  }
  return s;
}

static inline void vira_slice_string_push(vira_slice_string *s,
                                          vira_string val) {
  if (s->len >= s->cap) {
    _vira_slice_grow((void **)&s->data, &s->cap, sizeof(vira_string),
                     s->is_arena);
  }
  s->data[s->len++] = val;
}

static inline vira_string vira_slice_string_pop(vira_slice_string *s) {
  if (s->len <= 0) {
    fprintf(stderr, "vira panic: pop from empty slice\n");
    exit(1);
  }
  return s->data[--s->len];
}

static inline void vira_slice_string_set(vira_slice_string *s, int32_t idx,
                                         vira_string val) {
  vira_bounds_check(idx, s->len, __FILE__, __LINE__);
  vira_str_free(&s->data[idx]);
  s->data[idx] = val;
}

static inline void vira_slice_string_clear(vira_slice_string *s) {
  for (int32_t i = 0; i < s->len; i++) {
    vira_str_free(&s->data[i]);
  }
  s->len = 0;
}

static inline bool vira_slice_string_is_empty(const vira_slice_string *s) {
  return s->len == 0;
}

static inline bool vira_slice_string_contains(const vira_slice_string *s,
                                              vira_string val) {
  for (int32_t i = 0; i < s->len; i++) {
    if (vira_str_eq(s->data[i], val))
      return true;
  }
  return false;
}

static inline vira_slice_string
vira_slice_string_clone(const vira_slice_string *s) {
  vira_slice_string res = vira_slice_string_new(s->len);
  for (int32_t i = 0; i < s->len; i++) {
    vira_slice_string_push(&res, vira_str_clone(s->data[i]));
  }
  return res;
}

static inline vira_slice_string
vira_slice_string_subslice(const vira_slice_string *s, int32_t start, int32_t end) {
  if (start < 0) start = 0;
  if (end > s->len) end = s->len;
  if (start > end) start = end;
  int32_t count = end - start;
  vira_slice_string res = vira_slice_string_new(count);
  for (int32_t i = 0; i < count; i++) {
    vira_slice_string_push(&res, vira_str_clone(s->data[start + i]));
  }
  return res;
}

static inline void vira_slice_string_free(vira_slice_string *s) {
  if (s->data != NULL && !s->is_arena) {
    for (int32_t i = 0; i < s->len; i++) {
      vira_str_free(&s->data[i]);
    }
    free(s->data);
    s->data = NULL;
    _vira_record_free();
  }
  s->len = 0;
  s->cap = 0;
}

#ifndef _vira_slice_vira_slice_string_DEFINED
#define _vira_slice_vira_slice_string_DEFINED
typedef struct {
  vira_slice_string *data;
  int32_t len;
  int32_t cap;
  bool is_arena;
} vira_slice_vira_slice_string;
#endif

static inline vira_slice_vira_slice_string vira_slice_vira_slice_string_new(int32_t initial_cap) {
  vira_slice_vira_slice_string s = {.data = NULL, .len = 0, .cap = 0, .is_arena = false};
  if (initial_cap > 0) {
    s.data = (vira_slice_string *)malloc((size_t)initial_cap * sizeof(vira_slice_string));
    if (s.data) {
      s.cap = initial_cap;
      _vira_record_alloc();
    }
  }
  return s;
}

static inline void vira_slice_vira_slice_string_push(vira_slice_vira_slice_string *s, vira_slice_string val) {
  if (s->len >= s->cap) {
    _vira_slice_grow((void **)&s->data, &s->cap, sizeof(vira_slice_string), s->is_arena);
  }
  s->data[s->len++] = val;
}

static inline vira_slice_string vira_slice_vira_slice_string_pop(vira_slice_vira_slice_string *s) {
  if (s->len <= 0) {
    fprintf(stderr, "vira panic: pop from empty slice\n");
    exit(1);
  }
  return s->data[--s->len];
}

static inline void vira_slice_vira_slice_string_set(vira_slice_vira_slice_string *s, int32_t idx, vira_slice_string val) {
  vira_bounds_check(idx, s->len, __FILE__, __LINE__);
  vira_slice_string_free(&s->data[idx]);
  s->data[idx] = val;
}

static inline void vira_slice_vira_slice_string_clear(vira_slice_vira_slice_string *s) {
  for (int32_t i = 0; i < s->len; i++) {
    vira_slice_string_free(&s->data[i]);
  }
  s->len = 0;
}

static inline bool vira_slice_vira_slice_string_is_empty(const vira_slice_vira_slice_string *s) {
  return s->len == 0;
}

static inline vira_slice_vira_slice_string vira_slice_vira_slice_string_clone(const vira_slice_vira_slice_string *s) {
  vira_slice_vira_slice_string res = vira_slice_vira_slice_string_new(s->len);
  for (int32_t i = 0; i < s->len; i++) {
    vira_slice_string clone_item = vira_slice_string_clone(&s->data[i]);
    vira_slice_vira_slice_string_push(&res, clone_item);
  }
  return res;
}

static inline vira_slice_vira_slice_string vira_slice_vira_slice_string_subslice(const vira_slice_vira_slice_string *s, int32_t start, int32_t end) {
  if (start < 0) start = 0;
  if (end > s->len) end = s->len;
  if (start > end) start = end;
  int32_t count = end - start;
  vira_slice_vira_slice_string res = vira_slice_vira_slice_string_new(count);
  for (int32_t i = 0; i < count; i++) {
    vira_slice_string clone_item = vira_slice_string_clone(&s->data[start + i]);
    vira_slice_vira_slice_string_push(&res, clone_item);
  }
  return res;
}

static inline void vira_slice_vira_slice_string_free(vira_slice_vira_slice_string *s) {
  if (s->data != NULL && !s->is_arena) {
    for (int32_t i = 0; i < s->len; i++) {
      vira_slice_string_free(&s->data[i]);
    }
    free(s->data);
    s->data = NULL;
    _vira_record_free();
  }
  s->len = 0;
  s->cap = 0;
}

static inline vira_slice_string vira_os_args(void) {
  vira_slice_string s = vira_slice_string_new(_vira_argc);
  for (int i = 0; i < _vira_argc; i++) {
    vira_slice_string_push(&s, vira_str_from_cstr(_vira_argv[i]));
  }
  return s;
}

static inline vira_string vira_os_platform(void) {
#if defined(_WIN32)
  return vira_str_lit("windows", 7);
#elif defined(__APPLE__)
  return vira_str_lit("darwin", 6);
#elif defined(__linux__)
  return vira_str_lit("linux", 5);
#else
  return vira_str_lit("unknown", 7);
#endif
}

#ifndef _Result_u8_DEFINED
#define _Result_u8_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  uint8_t val;
} Result_u8;
#endif

#ifndef _Result_f32_DEFINED
#define _Result_f32_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  vira_f32 val;
} Result_f32;
#endif

#ifndef _Result_f64_DEFINED
#define _Result_f64_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  double val;
} Result_f64;
#endif

#ifndef _Result_int_DEFINED
#define _Result_int_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  int32_t val;
} Result_int;
#endif

#ifndef _Result_i64_DEFINED
#define _Result_i64_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  int64_t val;
} Result_i64;
#endif

static inline bool vira_str_contains(vira_string s, vira_string sub) {
  if (sub.len == 0)
    return true;
  if (s.len < sub.len || s.data == NULL || sub.data == NULL)
    return false;
  for (int32_t i = 0; i <= s.len - sub.len; i++) {
    if (memcmp(&s.data[i], sub.data, (size_t)sub.len) == 0) {
      return true;
    }
  }
  return false;
}

static inline int32_t vira_str_index_of(vira_string s, vira_string sub) {
  if (sub.len == 0)
    return 0;
  if (s.len < sub.len || s.data == NULL || sub.data == NULL)
    return -1;
  for (int32_t i = 0; i <= s.len - sub.len; i++) {
    if (memcmp(&s.data[i], sub.data, (size_t)sub.len) == 0) {
      return i;
    }
  }
  return -1;
}

static inline bool vira_str_starts_with(vira_string s, vira_string prefix) {
  if (prefix.len == 0)
    return true;
  if (s.len < prefix.len || s.data == NULL || prefix.data == NULL)
    return false;
  return memcmp(s.data, prefix.data, (size_t)prefix.len) == 0;
}

static inline bool vira_str_ends_with(vira_string s, vira_string suffix) {
  if (suffix.len == 0)
    return true;
  if (s.len < suffix.len || s.data == NULL || suffix.data == NULL)
    return false;
  return memcmp(&s.data[s.len - suffix.len], suffix.data, (size_t)suffix.len) ==
         0;
}

static inline vira_string vira_str_substr(vira_string s, int32_t start,
                                          int32_t end) {
  if (start < 0)
    start = 0;
  if (end > s.len)
    end = s.len;
  if (start >= end || s.data == NULL) {
    return vira_str_lit("", 0);
  }
  int32_t len = end - start;
  uint8_t *buf = (uint8_t *)malloc((size_t)len + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in vira_str_substr\n");
    exit(1);
  }
  memcpy(buf, &s.data[start], (size_t)len);
  buf[len] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = buf, .len = len, .is_lit = false};
}

static inline vira_string vira_str_trim(vira_string s) {
  if (s.len == 0 || s.data == NULL)
    return vira_str_lit("", 0);
  int32_t start = 0;
  while (start < s.len && (s.data[start] == ' ' || s.data[start] == '\t' ||
                           s.data[start] == '\r' || s.data[start] == '\n')) {
    start++;
  }
  int32_t end = s.len;
  while (end > start && (s.data[end - 1] == ' ' || s.data[end - 1] == '\t' ||
                         s.data[end - 1] == '\r' || s.data[end - 1] == '\n')) {
    end--;
  }
  return vira_str_substr(s, start, end);
}

static inline vira_string vira_str_to_lower(vira_string s) {
  if (s.len <= 0 || s.data == NULL)
    return vira_str_lit("", 0);
  size_t sz = (size_t)s.len;
  uint8_t *buf = (uint8_t *)malloc(sz + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in vira_str_to_lower\n");
    exit(1);
  }
  for (size_t i = 0; i < sz; i++) {
    uint8_t c = s.data[i];
    if (c >= 'A' && c <= 'Z')
      c = (uint8_t)(c + ('a' - 'A'));
    buf[i] = c;
  }
  buf[sz] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = buf, .len = (int32_t)sz, .is_lit = false};
}

static inline vira_string vira_str_to_upper(vira_string s) {
  if (s.len <= 0 || s.data == NULL)
    return vira_str_lit("", 0);
  size_t sz = (size_t)s.len;
  uint8_t *buf = (uint8_t *)malloc(sz + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in vira_str_to_upper\n");
    exit(1);
  }
  for (size_t i = 0; i < sz; i++) {
    uint8_t c = s.data[i];
    if (c >= 'a' && c <= 'z')
      c = (uint8_t)(c - ('a' - 'A'));
    buf[i] = c;
  }
  buf[sz] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = buf, .len = (int32_t)sz, .is_lit = false};
}

static inline Result_int vira_str_to_int(vira_string s) {
  if (s.len == 0 || s.data == NULL) {
    return (Result_int){.is_err = true,
                        .err = vira_str_lit("empty string", 12)};
  }
  char stack_buf[128];
  vira_cstr_buf b = vira_str_to_cstr_buf(s, stack_buf, sizeof(stack_buf));
  char *endptr = NULL;
  long val = strtol(b.ptr, &endptr, 10);
  bool valid = (endptr != b.ptr && *endptr == '\0');
  vira_cstr_buf_free(&b);
  if (!valid) {
    return (Result_int){.is_err = true,
                        .err = vira_str_lit("invalid integer format", 22)};
  }
  return (Result_int){.is_err = false, .val = (int32_t)val};
}

static inline Result_i64 vira_str_to_i64(vira_string s) {
  if (s.len == 0 || s.data == NULL) {
    return (Result_i64){.is_err = true,
                        .err = vira_str_lit("empty string", 12)};
  }
  char stack_buf[128];
  vira_cstr_buf b = vira_str_to_cstr_buf(s, stack_buf, sizeof(stack_buf));
  char *endptr = NULL;
  long long val = strtoll(b.ptr, &endptr, 10);
  bool valid = (endptr != b.ptr && *endptr == '\0');
  vira_cstr_buf_free(&b);
  if (!valid) {
    return (Result_i64){.is_err = true,
                        .err = vira_str_lit("invalid integer format", 22)};
  }
  return (Result_i64){.is_err = false, .val = (int64_t)val};
}

static inline Result_f64 vira_str_to_f64(vira_string s) {
  if (s.len == 0 || s.data == NULL) {
    return (Result_f64){.is_err = true, .err = vira_str_lit("empty string", 12)};
  }
  char stack_buf[128];
  vira_cstr_buf b = vira_str_to_cstr_buf(s, stack_buf, sizeof(stack_buf));
  char *endptr = NULL;
  double val = strtod(b.ptr, &endptr);
  bool valid = (endptr != b.ptr && *endptr == '\0');
  vira_cstr_buf_free(&b);
  if (!valid) {
    return (Result_f64){.is_err = true, .err = vira_str_lit("invalid float format", 20)};
  }
  return (Result_f64){.is_err = false, .val = val};
}

static inline Result_f32 vira_str_to_f32(vira_string s) {
  Result_f64 r = vira_str_to_f64(s);
  if (r.is_err) {
    return (Result_f32){.is_err = true, .err = r.err};
  }
  return (Result_f32){.is_err = false, .val = (vira_f32)r.val};
}

static inline Result_i64 vira_str_parse_i64_base(vira_string s, int32_t base) {
  if (s.len == 0 || s.data == NULL || base < 2 || base > 36) {
    return (Result_i64){.is_err = true, .err = vira_str_lit("invalid base or empty string", 28)};
  }
  char stack_buf[128];
  vira_cstr_buf b = vira_str_to_cstr_buf(s, stack_buf, sizeof(stack_buf));
  char *endptr = NULL;
  errno = 0;
  long long val = strtoll(b.ptr, &endptr, base);
  bool valid = (endptr != b.ptr && *endptr == '\0' && errno == 0);
  vira_cstr_buf_free(&b);
  if (!valid) {
    return (Result_i64){.is_err = true, .err = vira_str_lit("invalid integer format for base", 31)};
  }
  return (Result_i64){.is_err = false, .val = (int64_t)val};
}

static inline vira_string vira_str_format_i64_base(int64_t v, int32_t base) {
  if (base < 2 || base > 36) base = 10;
  if (base == 10) {
    return vira_str_format("%lld", (long long)v);
  }
  if (base == 16) {
    return vira_str_format("%llx", (unsigned long long)v);
  }
  if (base == 8) {
    return vira_str_format("%llo", (unsigned long long)v);
  }
  char buf[70];
  int idx = 69;
  buf[idx--] = '\0';
  bool neg = (v < 0 && base == 10);
  uint64_t uv = neg ? (uint64_t)(-v) : (uint64_t)v;
  if (uv == 0) {
    buf[idx--] = '0';
  } else {
    const char *digits = "0123456789abcdefghijklmnopqrstuvwxyz";
    while (uv > 0) {
      buf[idx--] = digits[uv % (uint64_t)base];
      uv /= (uint64_t)base;
    }
    if (neg) {
      buf[idx--] = '-';
    }
  }
  return vira_str_from_cstr(&buf[idx + 1]);
}

static inline Result_u8 vira_str_byte_at(vira_string s, int32_t idx) {
  if (idx < 0 || idx >= s.len || s.data == NULL) {
    return (Result_u8){.is_err = true, .err = vira_str_lit("index out of bounds", 19)};
  }
  return (Result_u8){.is_err = false, .val = (uint8_t)s.data[idx]};
}

static inline int32_t vira_str_last_index_of(vira_string s, vira_string sub) {
  if (sub.len == 0) return s.len;
  if (s.len < sub.len || s.data == NULL || sub.data == NULL) return -1;
  for (int32_t i = s.len - sub.len; i >= 0; i--) {
    if (memcmp(s.data + i, sub.data, (size_t)sub.len) == 0) {
      return i;
    }
  }
  return -1;
}

static inline vira_string vira_str_trim_left(vira_string s) {
  if (s.len == 0 || s.data == NULL) return s;
  int32_t start = 0;
  while (start < s.len && (s.data[start] == ' ' || s.data[start] == '\t' || s.data[start] == '\n' || s.data[start] == '\r')) {
    start++;
  }
  return (vira_string){.data = s.data + start, .len = s.len - start, .is_lit = true};
}

static inline vira_string vira_str_trim_right(vira_string s) {
  if (s.len == 0 || s.data == NULL) return s;
  int32_t end = s.len;
  while (end > 0 && (s.data[end - 1] == ' ' || s.data[end - 1] == '\t' || s.data[end - 1] == '\n' || s.data[end - 1] == '\r')) {
    end--;
  }
  return (vira_string){.data = s.data, .len = end, .is_lit = true};
}

static inline vira_string vira_str_repeat(vira_string s, int32_t count) {
  if (count <= 0 || s.len == 0 || s.data == NULL) {
    return vira_str_lit("", 0);
  }
  size_t total = (size_t)s.len * (size_t)count;
  char *buf = (char *)malloc(total + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in str.repeat\n");
    exit(1);
  }
  for (int32_t i = 0; i < count; i++) {
    memcpy(buf + ((size_t)i * (size_t)s.len), s.data, (size_t)s.len);
  }
  buf[total] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = (const uint8_t *)buf, .len = (int32_t)total, .is_lit = false};
}

static inline vira_string vira_str_replace(vira_string s, vira_string old_sub, vira_string new_sub) {
  if (s.len == 0 || s.data == NULL || old_sub.len == 0 || old_sub.data == NULL) {
    return vira_str_clone(s);
  }
  int32_t count = 0;
  int32_t idx = 0;
  while (idx <= s.len - old_sub.len) {
    if (memcmp(s.data + idx, old_sub.data, (size_t)old_sub.len) == 0) {
      count++;
      idx += old_sub.len;
    } else {
      idx++;
    }
  }
  if (count == 0) {
    return vira_str_clone(s);
  }
  size_t new_len = (size_t)s.len + (size_t)count * (size_t)(new_sub.len - old_sub.len);
  char *buf = (char *)malloc(new_len + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in str.replace\n");
    exit(1);
  }
  size_t dst = 0;
  idx = 0;
  while (idx < s.len) {
    if (idx <= s.len - old_sub.len && memcmp(s.data + idx, old_sub.data, (size_t)old_sub.len) == 0) {
      if (new_sub.len > 0) {
        memcpy(buf + dst, new_sub.data, (size_t)new_sub.len);
        dst += (size_t)new_sub.len;
      }
      idx += old_sub.len;
    } else {
      buf[dst++] = s.data[idx++];
    }
  }
  buf[new_len] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = (const uint8_t *)buf, .len = (int32_t)new_len, .is_lit = false};
}

static inline vira_slice_string vira_str_lines(vira_string s) {
  vira_slice_string res = vira_slice_string_new(8);
  if (s.len == 0 || s.data == NULL) {
    return res;
  }
  int32_t start = 0;
  for (int32_t i = 0; i < s.len; i++) {
    if (s.data[i] == '\n') {
      int32_t line_end = i;
      if (line_end > start && s.data[line_end - 1] == '\r') {
        line_end--;
      }
      vira_slice_string_push(&res, vira_str_substr(s, start, line_end));
      start = i + 1;
    }
  }
  if (start <= s.len) {
    int32_t line_end = s.len;
    if (line_end > start && s.data[line_end - 1] == '\r') {
      line_end--;
    }
    vira_slice_string_push(&res, vira_str_substr(s, start, line_end));
  }
  return res;
}

static inline vira_slice_string vira_str_split(vira_string s, vira_string sep) {
  vira_slice_string res = vira_slice_string_new(4);
  if (s.len == 0 || s.data == NULL) {
    vira_slice_string_push(&res, vira_str_lit("", 0));
    return res;
  }
  if (sep.len == 0) {
    for (int32_t i = 0; i < s.len; i++) {
      vira_slice_string_push(&res, vira_str_substr(s, i, i + 1));
    }
    return res;
  }
  int32_t cur = 0;
  while (cur <= s.len) {
    int32_t next = -1;
    if (cur <= s.len - sep.len) {
      for (int32_t i = cur; i <= s.len - sep.len; i++) {
        if (memcmp(&s.data[i], sep.data, (size_t)sep.len) == 0) {
          next = i;
          break;
        }
      }
    }
    if (next == -1) {
      vira_slice_string_push(&res, vira_str_substr(s, cur, s.len));
      break;
    } else {
      vira_slice_string_push(&res, vira_str_substr(s, cur, next));
      cur = next + sep.len;
    }
  }
  return res;
}

static inline vira_slice_u8 vira_str_to_bytes(vira_string s) {
  vira_slice_u8 res = vira_slice_u8_new(s.len);
  if (s.len > 0 && s.data != NULL) {
    memcpy(res.data, s.data, (size_t)s.len);
    res.len = s.len;
  }
  return res;
}

static inline vira_string vira_bytes_to_string(const vira_slice_u8 *s) {
  if (!s || s->len <= 0 || s->data == NULL) {
    return vira_str_lit("", 0);
  }
  uint8_t *copy = (uint8_t *)malloc((size_t)s->len + 1);
  if (!copy) {
    fprintf(stderr, "vira panic: out of memory in vira_bytes_to_string\n");
    exit(1);
  }
  memcpy(copy, s->data, (size_t)s->len);
  copy[s->len] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = copy, .len = s->len, .is_lit = false};
}

// ============================================================================
// 2. 64-BYTE ALIGNED BUMP-POINTER ARENA ALLOCATOR (`arena { ... }`)
// ============================================================================

typedef struct _vira_arena_slab {
  uint8_t *buf;
  size_t cap;
  struct _vira_arena_slab *next;
} vira_arena_slab;

typedef struct {
  uint8_t *buf;
  size_t cap;
  size_t offset;
  vira_arena_slab *overflow_slabs;
} vira_arena;

static inline vira_arena vira_arena_init(size_t capacity) {
  uint8_t *slab = (uint8_t *)malloc(capacity + 64);
  if (!slab) {
    fprintf(stderr, "vira panic: failed to allocate arena slab (%zu bytes)\n",
            capacity);
    exit(1);
  }
  _vira_record_alloc();
  return (vira_arena){
      .buf = slab, .cap = capacity, .offset = 0, .overflow_slabs = NULL};
}

static inline void *vira_arena_alloc(vira_arena *a, size_t size) {
  if (a->buf == NULL) {
    size_t initial_cap =
        (size < 65536) ? 65536 : ((size + 4095) & ~(size_t)4095);
    *a = vira_arena_init(initial_cap);
  }
  // 64-byte alignment guarantees AVX-512 / AVX2 / NEON aligned vector loads
  uintptr_t base = (uintptr_t)a->buf + a->offset;
  uintptr_t aligned = (base + 63) & ~(uintptr_t)63;
  size_t pad = (size_t)(aligned - base);
  size_t total = pad + ((size + 63) & ~(size_t)63);

  if (a->offset + total > a->cap) {
    vira_arena_slab *old_slab =
        (vira_arena_slab *)malloc(sizeof(vira_arena_slab));
    if (!old_slab) {
      fprintf(stderr, "vira panic: failed to allocate arena slab header\n");
      exit(1);
    }
    old_slab->buf = a->buf;
    old_slab->cap = a->cap;
    old_slab->next = a->overflow_slabs;

    size_t min_cap = ((size + 63) & ~(size_t)63) + 64;
    size_t doubled = a->cap * 2;
    size_t new_cap =
        (doubled >= min_cap) ? doubled : ((min_cap + 4095) & ~(size_t)4095);
    uint8_t *new_buf = (uint8_t *)malloc(new_cap + 64);
    if (!new_buf) {
      fprintf(stderr, "vira panic: failed to allocate chained arena slab (%zu bytes)\n",
              new_cap);
      exit(1);
    }
    _vira_record_alloc();
    a->overflow_slabs = old_slab;
    a->buf = new_buf;
    a->cap = new_cap;
    a->offset = 0;

    base = (uintptr_t)a->buf;
    aligned = (base + 63) & ~(uintptr_t)63;
    pad = (size_t)(aligned - base);
    total = pad + ((size + 63) & ~(size_t)63);
  }
  a->offset += total;
  return (void *)aligned;
}

static inline bool vira_arena_contains(const vira_arena *a, const void *ptr) {
  if (!a || !ptr)
    return false;
  uintptr_t p = (uintptr_t)ptr;
  if (a->buf != NULL) {
    uintptr_t start = (uintptr_t)a->buf;
    if (p >= start && p < start + a->cap + 64) {
      return true;
    }
  }
  for (const vira_arena_slab *s = a->overflow_slabs; s != NULL; s = s->next) {
    uintptr_t start = (uintptr_t)s->buf;
    if (p >= start && p < start + s->cap + 64) {
      return true;
    }
  }
  return false;
}

static inline void vira_arena_reset(vira_arena *a) {
  if (!a)
    return;
  vira_arena_slab *cur = a->overflow_slabs;
  while (cur != NULL) {
    vira_arena_slab *next = cur->next;
    if (cur->buf != NULL) {
      free(cur->buf);
      _vira_record_free();
    }
    free(cur);
    cur = next;
  }
  a->overflow_slabs = NULL;
  a->offset = 0;
}

static inline vira_string vira_arena_str_format(vira_arena *a, const char *fmt,
                                                ...) {
  va_list args, args_copy;
  va_start(args, fmt);
  va_copy(args_copy, args);
  int32_t needed = vsnprintf(NULL, 0, fmt, args);
  va_end(args);

  char *buf = (char *)vira_arena_alloc(a, (size_t)needed + 1);
  vsnprintf(buf, (size_t)needed + 1, fmt, args_copy);
  va_end(args_copy);

  return (vira_string){
      .data = (const uint8_t *)buf, .len = needed, .is_lit = true};
}

static inline void vira_arena_destroy(vira_arena *a) {
  if (a == NULL)
    return;
  vira_arena_reset(a);
  if (a->buf != NULL) {
    free(a->buf);
    a->buf = NULL;
    a->cap = 0;
    a->offset = 0;
    _vira_record_free();
  }
}

// ============================================================================
// 3. N-D AI TENSORS + MULTI-THREADED CPU SIMD (`AVX2` / `SSE2` / `ARM NEON`)
// ============================================================================

#ifndef _VIRA_TASK_DEFINED
#define _VIRA_TASK_DEFINED
typedef void (*vira_task_fn)(void *env);

typedef struct _vira_task {
  vira_task_fn fn;
  void *env;
  struct _vira_task *next_actor_msg; // Intrusive MPSC link
  volatile int32_t done;             // 0 = pending, 1 = completed on worker pool
} vira_task;
#endif

typedef struct _vira_actor_ctx vira_actor_ctx;
static __thread vira_actor_ctx *_vira_tls_actor;
static inline bool _vira_in_active_actor(void);
static inline void vira_scheduler_ensure_init(void);
static inline void vira_scheduler_submit_async(vira_task *task);
static inline void vira_scheduler_await(vira_task *task);
static inline int32_t vira_scheduler_num_workers(void);

static inline const char *vira_simd_backend_name(void) {
#if defined(__AVX2__)
  return "x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle, Multi-Threaded)";
#elif defined(__ARM_NEON) || defined(__aarch64__)
  return "ARM64 NEON (128-bit 2x-Unrolled / 8x f32 per step, Multi-Threaded)";
#elif defined(__SSE2__)
  return "x86_64 SSE2 (128-bit / 4x f32 per cycle, Multi-Threaded)";
#else
  return "Scalar C99 Fallback (Multi-Threaded)";
#endif
}

typedef struct {
  vira_f32 *data;  // 64-byte cache-line aligned pointer
  void *raw_alloc; // underlying pointer for free()
  int32_t rows;    // flattened outer product (1 for 1D) for 2D compatibility
  int32_t cols;    // trailing dimension dims[n_dims - 1] for 2D compatibility
  int32_t dims[4];
  int32_t strides[4];
  int32_t n_dims;
  bool is_arena;
  bool is_view;
} vira_tensor_f32;

static inline void _vira_tensor_init_shape(vira_tensor_f32 *t, int32_t n_dims,
                                           const int32_t *dims) {
  if (n_dims < 1 || n_dims > 4 || dims == NULL) {
    fprintf(stderr, "vira tensor error: unsupported rank %d (expected 1..4)\n",
            n_dims);
    exit(1);
  }
  t->n_dims = n_dims;
  for (int32_t d = 0; d < 4; d++) {
    t->dims[d] = 0;
    t->strides[d] = 0;
  }
  for (int32_t d = 0; d < n_dims; d++) {
    if (dims[d] <= 0) {
      fprintf(stderr,
              "vira tensor error: invalid dimension %d at axis %d (rank %d)\n",
              dims[d], d, n_dims);
      exit(1);
    }
    t->dims[d] = dims[d];
  }
  int32_t st = 1;
  for (int32_t d = n_dims - 1; d >= 0; d--) {
    t->strides[d] = st;
    st *= dims[d];
  }
  if (n_dims == 1) {
    t->rows = 1;
    t->cols = dims[0];
  } else {
    int32_t outer = 1;
    for (int32_t d = 0; d < n_dims - 1; d++) {
      outer *= dims[d];
    }
    t->rows = outer;
    t->cols = dims[n_dims - 1];
  }
}

static inline void _vira_tensor_norm_shape(vira_tensor_f32 *t) {
  if (t->n_dims <= 0 && t->rows > 0 && t->cols > 0) {
    int32_t d[2] = {t->rows, t->cols};
    _vira_tensor_init_shape(t, 2, d);
  }
}

static inline int32_t vira_tensor_numel(vira_tensor_f32 t) {
  _vira_tensor_norm_shape(&t);
  return t.rows * t.cols;
}

static inline int32_t vira_tensor_dim(vira_tensor_f32 t, int32_t axis) {
  _vira_tensor_norm_shape(&t);
  int32_t ax = axis < 0 ? (t.n_dims + axis) : axis;
  if (ax < 0 || ax >= t.n_dims) {
    fprintf(stderr,
            "vira tensor error: dim axis %d out of bounds for %dD tensor\n",
            axis, t.n_dims);
    exit(1);
  }
  return t.dims[ax];
}

static inline int32_t vira_tensor_stride(vira_tensor_f32 t, int32_t axis) {
  _vira_tensor_norm_shape(&t);
  int32_t ax = axis < 0 ? (t.n_dims + axis) : axis;
  if (ax < 0 || ax >= t.n_dims) {
    fprintf(stderr,
            "vira tensor error: stride axis %d out of bounds for %dD tensor\n",
            axis, t.n_dims);
    exit(1);
  }
  return t.strides[ax];
}

static inline vira_tensor_f32 vira_tensor_f32_alloc_nd(int32_t n_dims,
                                                       const int32_t *dims) {
  vira_tensor_f32 t = {0};
  _vira_tensor_init_shape(&t, n_dims, dims);
  size_t bytes = (size_t)t.rows * (size_t)t.cols * sizeof(vira_f32);
  void *raw = malloc(bytes + 64);
  if (!raw) {
    fprintf(stderr, "vira panic: tensor allocation failed (%zu bytes)\n",
            bytes);
    exit(1);
  }
  uintptr_t aligned = ((uintptr_t)raw + 63) & ~(uintptr_t)63;
  t.data = (vira_f32 *)aligned;
  t.raw_alloc = raw;
  t.is_arena = false;
  t.is_view = false;
  memset(t.data, 0, bytes);
  _vira_record_alloc();
  return t;
}

static inline vira_tensor_f32
vira_tensor_f32_arena_alloc_nd(vira_arena *a, int32_t n_dims,
                               const int32_t *dims) {
  vira_tensor_f32 t = {0};
  _vira_tensor_init_shape(&t, n_dims, dims);
  size_t bytes = (size_t)t.rows * (size_t)t.cols * sizeof(vira_f32);
  t.data = (vira_f32 *)vira_arena_alloc(a, bytes);
  t.raw_alloc = NULL;
  t.is_arena = true;
  t.is_view = false;
  memset(t.data, 0, bytes);
  return t;
}

static inline vira_tensor_f32 vira_tensor_f32_alloc(int32_t rows,
                                                    int32_t cols) {
  if (rows <= 0 || cols <= 0) {
    fprintf(stderr, "vira tensor error: invalid dimensions (%d x %d)\n", rows,
            cols);
    exit(1);
  }
  int32_t dims[2] = {rows, cols};
  return vira_tensor_f32_alloc_nd(2, dims);
}

static inline vira_tensor_f32
vira_tensor_f32_arena_alloc(vira_arena *a, int32_t rows, int32_t cols) {
  if (rows <= 0 || cols <= 0) {
    fprintf(stderr, "vira tensor error: invalid dimensions (%d x %d)\n", rows,
            cols);
    exit(1);
  }
  int32_t dims[2] = {rows, cols};
  return vira_tensor_f32_arena_alloc_nd(a, 2, dims);
}

static inline void _vira_tensor_fill_data(vira_f32 *data, int32_t total,
                                          vira_f32 val) {
  int32_t i = 0;
#if defined(__AVX2__)
  __m256 vval = _mm256_set1_ps(val);
  for (; i + 8 <= total; i += 8) {
    _mm256_storeu_ps(&data[i], vval);
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t vval = vdupq_n_f32(val);
  for (; i + 4 <= total; i += 4) {
    vst1q_f32(&data[i], vval);
  }
#endif
  for (; i < total; i++) {
    data[i] = val;
  }
}

static inline vira_tensor_f32 vira_tensor_f32_fill(vira_arena *opt_arena,
                                                   int32_t rows, int32_t cols,
                                                   vira_f32 val) {
  vira_tensor_f32 t = opt_arena
                          ? vira_tensor_f32_arena_alloc(opt_arena, rows, cols)
                          : vira_tensor_f32_alloc(rows, cols);
  _vira_tensor_fill_data(t.data, t.rows * t.cols, val);
  return t;
}

static inline vira_tensor_f32 vira_tensor_f32_fill_3d(vira_arena *opt_arena,
                                                      int32_t d0, int32_t d1,
                                                      int32_t d2,
                                                      vira_f32 val) {
  int32_t dims[3] = {d0, d1, d2};
  vira_tensor_f32 t =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 3, dims)
                : vira_tensor_f32_alloc_nd(3, dims);
  _vira_tensor_fill_data(t.data, t.rows * t.cols, val);
  return t;
}

static inline vira_tensor_f32 vira_tensor_f32_fill_4d(vira_arena *opt_arena,
                                                      int32_t d0, int32_t d1,
                                                      int32_t d2, int32_t d3,
                                                      vira_f32 val) {
  int32_t dims[4] = {d0, d1, d2, d3};
  vira_tensor_f32 t =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 4, dims)
                : vira_tensor_f32_alloc_nd(4, dims);
  _vira_tensor_fill_data(t.data, t.rows * t.cols, val);
  return t;
}

// Low-level C FFI zero-copy view (is_view = true; caller ensures `t` outlives view)
static inline vira_tensor_f32 vira_tensor_view_nd(vira_tensor_f32 t,
                                                  int32_t n_dims,
                                                  const int32_t *dims) {
  _vira_tensor_norm_shape(&t);
  vira_tensor_f32 v = {0};
  _vira_tensor_init_shape(&v, n_dims, dims);
  if (v.rows * v.cols != t.rows * t.cols) {
    fprintf(stderr,
            "vira tensor error: reshape element count mismatch (%d -> %d)\n",
            t.rows * t.cols, v.rows * v.cols);
    exit(1);
  }
  v.data = t.data;
  v.raw_alloc = NULL;
  v.is_arena = t.is_arena;
  v.is_view = true;
  return v;
}

static inline vira_tensor_f32
_vira_tensor_reshape_safe(vira_arena *opt_arena, vira_tensor_f32 t,
                          int32_t n_dims, const int32_t *dims) {
  _vira_tensor_norm_shape(&t);
  if (t.is_arena && opt_arena != NULL && vira_arena_contains(opt_arena, t.data)) {
    return vira_tensor_view_nd(t, n_dims, dims);
  }
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, n_dims, dims)
                : vira_tensor_f32_alloc_nd(n_dims, dims);
  if (out.rows * out.cols != t.rows * t.cols) {
    fprintf(stderr,
            "vira tensor error: reshape element count mismatch (%d -> %d)\n",
            t.rows * t.cols, out.rows * out.cols);
    exit(1);
  }
  memcpy(out.data, t.data, (size_t)out.rows * (size_t)out.cols * sizeof(vira_f32));
  return out;
}

static inline vira_tensor_f32 vira_tensor_reshape_2d(vira_arena *opt_arena,
                                                     vira_tensor_f32 t,
                                                     int32_t d0, int32_t d1) {
  int32_t dims[2] = {d0, d1};
  return _vira_tensor_reshape_safe(opt_arena, t, 2, dims);
}

static inline vira_tensor_f32 vira_tensor_reshape_3d(vira_arena *opt_arena,
                                                     vira_tensor_f32 t,
                                                     int32_t d0, int32_t d1,
                                                     int32_t d2) {
  int32_t dims[3] = {d0, d1, d2};
  return _vira_tensor_reshape_safe(opt_arena, t, 3, dims);
}

static inline vira_tensor_f32 vira_tensor_reshape_4d(vira_arena *opt_arena,
                                                     vira_tensor_f32 t,
                                                     int32_t d0, int32_t d1,
                                                     int32_t d2, int32_t d3) {
  int32_t dims[4] = {d0, d1, d2, d3};
  return _vira_tensor_reshape_safe(opt_arena, t, 4, dims);
}

// Transposes the last two dimensions `(..., R, C) -> (..., C, R)` for 2D/3D/4D tensors
static inline vira_tensor_f32 vira_tensor_transpose_nd(vira_arena *opt_arena,
                                                       vira_tensor_f32 a) {
  _vira_tensor_norm_shape(&a);
  if (a.n_dims == 1) {
    int32_t dims[2] = {a.dims[0], 1};
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, dims)
                  : vira_tensor_f32_alloc_nd(2, dims);
    memcpy(out.data, a.data, (size_t)a.dims[0] * sizeof(vira_f32));
    return out;
  }
  int32_t out_dims[4] = {a.dims[0], a.dims[1], a.dims[2], a.dims[3]};
  int32_t R = a.dims[a.n_dims - 2];
  int32_t C = a.dims[a.n_dims - 1];
  out_dims[a.n_dims - 2] = C;
  out_dims[a.n_dims - 1] = R;
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, a.n_dims, out_dims)
                : vira_tensor_f32_alloc_nd(a.n_dims, out_dims);
  int32_t batches = a.rows / R;
  size_t slice_elems = (size_t)R * (size_t)C;
  for (int32_t b = 0; b < batches; b++) {
    const vira_f32 *src = a.data + (size_t)b * slice_elems;
    vira_f32 *dst = out.data + (size_t)b * slice_elems;
    for (int32_t r = 0; r < R; r++) {
      for (int32_t c = 0; c < C; c++) {
        dst[c * R + r] = src[r * C + c];
      }
    }
  }
  return out;
}

static inline void _vira_matmul_kernel_2d_rows(const vira_f32 *a_data,
                                               const vira_f32 *b_data,
                                               vira_f32 *c_data,
                                               int32_t r_start, int32_t r_end,
                                               int32_t K, int32_t N) {
  for (int32_t si = r_start; si < r_end; si += 64) {
    int32_t imax = (si + 64 < r_end) ? (si + 64) : r_end;
    for (int32_t sk = 0; sk < K; sk += 64) {
      int32_t kmax = (sk + 64 < K) ? (sk + 64) : K;
      for (int32_t sj = 0; sj < N; sj += 64) {
        int32_t jmax = (sj + 64 < N) ? (sj + 64) : N;

        for (int32_t i = si; i < imax; i++) {
          vira_f32 *c_row = &c_data[(size_t)i * (size_t)N];
          for (int32_t k = sk; k < kmax; k++) {
            const vira_f32 a_ik = a_data[(size_t)i * (size_t)K + (size_t)k];
            const vira_f32 *b_row = &b_data[(size_t)k * (size_t)N];
            int32_t j = sj;

#if defined(__AVX2__)
            __m256 va = _mm256_set1_ps(a_ik);
            for (; j + 8 <= jmax; j += 8) {
              __m256 vc = _mm256_loadu_ps(&c_row[j]);
              __m256 vb = _mm256_loadu_ps(&b_row[j]);
#if defined(__FMA__)
              vc = _mm256_fmadd_ps(va, vb, vc);
#else
              vc = _mm256_add_ps(_mm256_mul_ps(va, vb), vc);
#endif
              _mm256_storeu_ps(&c_row[j], vc);
            }
#elif defined(__ARM_NEON) || defined(__aarch64__)
            for (; j + 8 <= jmax; j += 8) {
              float32x4_t vc0 = vld1q_f32(&c_row[j]);
              float32x4_t vc1 = vld1q_f32(&c_row[j + 4]);
              float32x4_t vb0 = vld1q_f32(&b_row[j]);
              float32x4_t vb1 = vld1q_f32(&b_row[j + 4]);
              vc0 = vfmaq_n_f32(vc0, vb0, a_ik);
              vc1 = vfmaq_n_f32(vc1, vb1, a_ik);
              vst1q_f32(&c_row[j], vc0);
              vst1q_f32(&c_row[j + 4], vc1);
            }
            for (; j + 4 <= jmax; j += 4) {
              float32x4_t vc0 = vld1q_f32(&c_row[j]);
              float32x4_t vb0 = vld1q_f32(&b_row[j]);
              vc0 = vfmaq_n_f32(vc0, vb0, a_ik);
              vst1q_f32(&c_row[j], vc0);
            }
#elif defined(__SSE2__)
            __m128 va = _mm_set1_ps(a_ik);
            for (; j + 4 <= jmax; j += 4) {
              __m128 vc = _mm_loadu_ps(&c_row[j]);
              __m128 vb = _mm_loadu_ps(&b_row[j]);
              vc = _mm_add_ps(_mm_mul_ps(va, vb), vc);
              _mm_storeu_ps(&c_row[j], vc);
            }
#endif
            for (; j < jmax; j++) {
              c_row[j] += a_ik * b_row[j];
            }
          }
        }
      }
    }
  }
}

typedef struct {
  const vira_f32 *a_base;
  const vira_f32 *b_base;
  vira_f32 *c_base;
  int32_t M;
  int32_t K;
  int32_t N;
  int32_t B1;
  int32_t A_B0;
  int32_t A_B1;
  int32_t B_B0;
  int32_t B_B1;
  int32_t gr_start; // global row start in [0, total_batches * M)
  int32_t gr_end;   // global row end
} _vira_matmul_job;

static void _vira_matmul_job_fn(void *raw) {
  const _vira_matmul_job *job = (const _vira_matmul_job *)raw;
  const int32_t M = job->M;
  const int32_t K = job->K;
  const int32_t N = job->N;
  int32_t cur = job->gr_start;
  while (cur < job->gr_end) {
    int32_t b_idx = cur / M;
    int32_t r_start = cur % M;
    int32_t rem_in_batch = M - r_start;
    int32_t rem_total = job->gr_end - cur;
    int32_t r_end = r_start + ((rem_total < rem_in_batch) ? rem_total : rem_in_batch);

    int32_t i0 = b_idx / job->B1;
    int32_t i1 = b_idx % job->B1;
    int32_t a_b = ((job->A_B0 == 1) ? 0 : i0) * job->A_B1 +
                  ((job->A_B1 == 1) ? 0 : i1);
    int32_t b_b = ((job->B_B0 == 1) ? 0 : i0) * job->B_B1 +
                  ((job->B_B1 == 1) ? 0 : i1);

    const vira_f32 *a_ptr = job->a_base + (size_t)a_b * (size_t)M * (size_t)K;
    const vira_f32 *b_ptr = job->b_base + (size_t)b_b * (size_t)K * (size_t)N;
    vira_f32 *c_ptr = job->c_base + (size_t)b_idx * (size_t)M * (size_t)N;

    _vira_matmul_kernel_2d_rows(a_ptr, b_ptr, c_ptr, r_start, r_end, K, N);
    cur += (r_end - r_start);
  }
}

#define VIRA_MATMUL_MT_THRESHOLD (128LL * 128LL * 128LL)

// Hardware-accelerated 2D/3D/4D Batched + Multi-Threaded `@` Matrix Multiplication
static inline vira_tensor_f32 vira_tensor_matmul_f32(vira_arena *opt_arena,
                                                     vira_tensor_f32 a,
                                                     vira_tensor_f32 b) {
  _vira_tensor_norm_shape(&a);
  _vira_tensor_norm_shape(&b);

  int32_t a_nd = a.n_dims;
  int32_t b_nd = b.n_dims;
  int32_t M = (a_nd == 1) ? 1 : a.dims[a_nd - 2];
  int32_t K_a = a.dims[a_nd - 1];
  int32_t K_b = (b_nd == 1) ? b.dims[0] : b.dims[b_nd - 2];
  int32_t N = (b_nd == 1) ? 1 : b.dims[b_nd - 1];

  if (K_a != K_b) {
    fprintf(stderr,
            "vira tensor error: matmul '@' inner dimension mismatch (%d vs %d)\n",
            K_a, K_b);
    exit(1);
  }

  int32_t a_batch_nd = (a_nd > 2) ? (a_nd - 2) : 0;
  int32_t b_batch_nd = (b_nd > 2) ? (b_nd - 2) : 0;
  int32_t batch_nd = (a_batch_nd > b_batch_nd) ? a_batch_nd : b_batch_nd;

  // Normalize leading batch dims into 2 slots [B0, B1]
  int32_t A_B0 = 1, A_B1 = 1;
  if (a_batch_nd == 1) {
    A_B1 = a.dims[0];
  } else if (a_batch_nd == 2) {
    A_B0 = a.dims[0];
    A_B1 = a.dims[1];
  }

  int32_t B_B0 = 1, B_B1 = 1;
  if (b_batch_nd == 1) {
    B_B1 = b.dims[0];
  } else if (b_batch_nd == 2) {
    B_B0 = b.dims[0];
    B_B1 = b.dims[1];
  }

  if ((A_B0 != B_B0 && A_B0 != 1 && B_B0 != 1) ||
      (A_B1 != B_B1 && A_B1 != 1 && B_B1 != 1)) {
    fprintf(stderr,
            "vira tensor error: matmul '@' batch broadcast mismatch (%dx%d) vs (%dx%d)\n",
            A_B0, A_B1, B_B0, B_B1);
    exit(1);
  }

  int32_t B0 = (A_B0 > B_B0) ? A_B0 : B_B0;
  int32_t B1 = (A_B1 > B_B1) ? A_B1 : B_B1;
  int32_t total_batches = B0 * B1;

  int32_t out_nd = batch_nd + 2;
  int32_t out_dims[4] = {0, 0, 0, 0};
  if (batch_nd == 0) {
    out_dims[0] = M;
    out_dims[1] = N;
  } else if (batch_nd == 1) {
    out_dims[0] = B1;
    out_dims[1] = M;
    out_dims[2] = N;
  } else {
    out_dims[0] = B0;
    out_dims[1] = B1;
    out_dims[2] = M;
    out_dims[3] = N;
  }

  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, out_nd, out_dims)
                : vira_tensor_f32_alloc_nd(out_nd, out_dims);

  int32_t total_rows = total_batches * M;
  int64_t total_ops =
      (int64_t)total_rows * (int64_t)K_a * (int64_t)N;

  if (total_ops >= VIRA_MATMUL_MT_THRESHOLD && total_rows >= 4 &&
      !_vira_in_active_actor()) {
    vira_scheduler_ensure_init();
    int32_t nw = vira_scheduler_num_workers();
    if (nw > 16)
      nw = 16;
    if (nw > total_rows)
      nw = total_rows;
    if (nw > 1) {
      _vira_matmul_job jobs[16];
      vira_task tasks[16];
      int32_t base_rows = total_rows / nw;
      int32_t extra = total_rows % nw;
      int32_t cur = 0;
      for (int32_t w = 0; w < nw; w++) {
        int32_t cnt = base_rows + (w < extra ? 1 : 0);
        jobs[w] = (_vira_matmul_job){.a_base = a.data,
                                     .b_base = b.data,
                                     .c_base = out.data,
                                     .M = M,
                                     .K = K_a,
                                     .N = N,
                                     .B1 = B1,
                                     .A_B0 = A_B0,
                                     .A_B1 = A_B1,
                                     .B_B0 = B_B0,
                                     .B_B1 = B_B1,
                                     .gr_start = cur,
                                     .gr_end = cur + cnt};
        cur += cnt;
        if (w > 0) {
          tasks[w] = (vira_task){.fn = _vira_matmul_job_fn,
                                 .env = &jobs[w],
                                 .next_actor_msg = NULL,
                                 .done = 0};
          vira_scheduler_submit_async(&tasks[w]);
        }
      }
      _vira_matmul_job_fn(&jobs[0]);
      for (int32_t w = 1; w < nw; w++) {
        vira_scheduler_await(&tasks[w]);
      }
      return out;
    }
  }

  _vira_matmul_job single = {.a_base = a.data,
                             .b_base = b.data,
                             .c_base = out.data,
                             .M = M,
                             .K = K_a,
                             .N = N,
                             .B1 = B1,
                             .A_B0 = A_B0,
                             .A_B1 = A_B1,
                             .B_B0 = B_B0,
                             .B_B1 = B_B1,
                             .gr_start = 0,
                             .gr_end = total_rows};
  _vira_matmul_job_fn(&single);
  return out;
}

// SIMD-accelerated tensor sum reduction
static inline vira_f32 vira_tensor_sum(vira_tensor_f32 t) {
  _vira_tensor_norm_shape(&t);
  int32_t total = t.rows * t.cols;
  int32_t i = 0;
  vira_f32 sum = 0.0f;

#if defined(__AVX2__)
  __m256 vsum = _mm256_setzero_ps();
  for (; i + 8 <= total; i += 8) {
    vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(&t.data[i]));
  }
  __m128 lo = _mm256_castps256_ps128(vsum);
  __m128 hi = _mm256_extractf128_ps(vsum, 1);
  __m128 s128 = _mm_add_ps(lo, hi);
  s128 = _mm_hadd_ps(s128, s128);
  s128 = _mm_hadd_ps(s128, s128);
  sum += _mm_cvtss_f32(s128);
#elif defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t vsum = vdupq_n_f32(0.0f);
  for (; i + 4 <= total; i += 4) {
    vsum = vaddq_f32(vsum, vld1q_f32(&t.data[i]));
  }
  sum += vaddvq_f32(vsum);
#endif
  for (; i < total; i++) {
    sum += t.data[i];
  }
  return sum;
}

// SIMD-accelerated ReLU activation: out = max(0, t) (preserves 1D..4D shape)
static inline vira_tensor_f32 vira_tensor_relu(vira_arena *opt_arena,
                                               vira_tensor_f32 t) {
  _vira_tensor_norm_shape(&t);
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, t.n_dims, t.dims)
                : vira_tensor_f32_alloc_nd(t.n_dims, t.dims);
  int32_t total = t.rows * t.cols;
  int32_t i = 0;

#if defined(__AVX2__)
  __m256 vzero = _mm256_setzero_ps();
  for (; i + 8 <= total; i += 8) {
    __m256 v = _mm256_loadu_ps(&t.data[i]);
    __m256 vr = _mm256_max_ps(v, vzero);
    _mm256_storeu_ps(&out.data[i], vr);
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t vzero = vdupq_n_f32(0.0f);
  for (; i + 4 <= total; i += 4) {
    float32x4_t v = vld1q_f32(&t.data[i]);
    float32x4_t vr = vmaxq_f32(v, vzero);
    vst1q_f32(&out.data[i], vr);
  }
#elif defined(__SSE2__)
  __m128 vzero = _mm_setzero_ps();
  for (; i + 4 <= total; i += 4) {
    __m128 v = _mm_loadu_ps(&t.data[i]);
    __m128 vr = _mm_max_ps(v, vzero);
    _mm_storeu_ps(&out.data[i], vr);
  }
#endif
  for (; i < total; i++) {
    out.data[i] = t.data[i] > 0.0f ? t.data[i] : 0.0f;
  }
  return out;
}

static inline void _vira_vec_add_f32(vira_f32 *dst, const vira_f32 *pa,
                                     const vira_f32 *pb, int32_t count) {
  int32_t i = 0;
#if defined(__AVX2__)
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(&pa[i]);
    __m256 vb = _mm256_loadu_ps(&pb[i]);
    _mm256_storeu_ps(&dst[i], _mm256_add_ps(va, vb));
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  for (; i + 4 <= count; i += 4) {
    float32x4_t va = vld1q_f32(&pa[i]);
    float32x4_t vb = vld1q_f32(&pb[i]);
    vst1q_f32(&dst[i], vaddq_f32(va, vb));
  }
#elif defined(__SSE2__)
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(&pa[i]);
    __m128 vb = _mm_loadu_ps(&pb[i]);
    _mm_storeu_ps(&dst[i], _mm_add_ps(va, vb));
  }
#endif
  for (; i < count; i++) {
    dst[i] = pa[i] + pb[i];
  }
}

static inline void _vira_vec_sub_f32(vira_f32 *dst, const vira_f32 *pa,
                                     const vira_f32 *pb, int32_t count) {
  int32_t i = 0;
#if defined(__AVX2__)
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(&pa[i]);
    __m256 vb = _mm256_loadu_ps(&pb[i]);
    _mm256_storeu_ps(&dst[i], _mm256_sub_ps(va, vb));
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  for (; i + 4 <= count; i += 4) {
    float32x4_t va = vld1q_f32(&pa[i]);
    float32x4_t vb = vld1q_f32(&pb[i]);
    vst1q_f32(&dst[i], vsubq_f32(va, vb));
  }
#elif defined(__SSE2__)
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(&pa[i]);
    __m128 vb = _mm_loadu_ps(&pb[i]);
    _mm_storeu_ps(&dst[i], _mm_sub_ps(va, vb));
  }
#endif
  for (; i < count; i++) {
    dst[i] = pa[i] - pb[i];
  }
}

// SIMD-accelerated tensor addition: out = a + b (supports 1D..4D + trailing-dim bias broadcast)
static inline vira_tensor_f32
vira_tensor_add(vira_arena *opt_arena, vira_tensor_f32 a, vira_tensor_f32 b) {
  _vira_tensor_norm_shape(&a);
  _vira_tensor_norm_shape(&b);
  if (a.rows == b.rows && a.cols == b.cols) {
    const vira_tensor_f32 *sh = (a.n_dims >= b.n_dims) ? &a : &b;
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, sh->n_dims, sh->dims)
                  : vira_tensor_f32_alloc_nd(sh->n_dims, sh->dims);
    _vira_vec_add_f32(out.data, a.data, b.data, a.rows * a.cols);
    return out;
  }
  if (b.rows == 1 && b.cols == a.cols) {
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, a.n_dims, a.dims)
                  : vira_tensor_f32_alloc_nd(a.n_dims, a.dims);
    for (int32_t r = 0; r < a.rows; r++) {
      _vira_vec_add_f32(&out.data[(size_t)r * (size_t)a.cols],
                        &a.data[(size_t)r * (size_t)a.cols], b.data, a.cols);
    }
    return out;
  }
  if (a.rows == 1 && a.cols == b.cols) {
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, b.n_dims, b.dims)
                  : vira_tensor_f32_alloc_nd(b.n_dims, b.dims);
    for (int32_t r = 0; r < b.rows; r++) {
      _vira_vec_add_f32(&out.data[(size_t)r * (size_t)b.cols], a.data,
                        &b.data[(size_t)r * (size_t)b.cols], b.cols);
    }
    return out;
  }
  fprintf(stderr, "vira tensor error: add shape mismatch (%dx%d) + (%dx%d)\n",
          a.rows, a.cols, b.rows, b.cols);
  exit(1);
}

// SIMD-accelerated tensor subtraction: out = a - b (supports 1D..4D + trailing-dim bias broadcast)
static inline vira_tensor_f32
vira_tensor_sub(vira_arena *opt_arena, vira_tensor_f32 a, vira_tensor_f32 b) {
  _vira_tensor_norm_shape(&a);
  _vira_tensor_norm_shape(&b);
  if (a.rows == b.rows && a.cols == b.cols) {
    const vira_tensor_f32 *sh = (a.n_dims >= b.n_dims) ? &a : &b;
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, sh->n_dims, sh->dims)
                  : vira_tensor_f32_alloc_nd(sh->n_dims, sh->dims);
    _vira_vec_sub_f32(out.data, a.data, b.data, a.rows * a.cols);
    return out;
  }
  if (b.rows == 1 && b.cols == a.cols) {
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, a.n_dims, a.dims)
                  : vira_tensor_f32_alloc_nd(a.n_dims, a.dims);
    for (int32_t r = 0; r < a.rows; r++) {
      _vira_vec_sub_f32(&out.data[(size_t)r * (size_t)a.cols],
                        &a.data[(size_t)r * (size_t)a.cols], b.data, a.cols);
    }
    return out;
  }
  if (a.rows == 1 && a.cols == b.cols) {
    vira_tensor_f32 out =
        opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, b.n_dims, b.dims)
                  : vira_tensor_f32_alloc_nd(b.n_dims, b.dims);
    for (int32_t r = 0; r < b.rows; r++) {
      _vira_vec_sub_f32(&out.data[(size_t)r * (size_t)b.cols], a.data,
                        &b.data[(size_t)r * (size_t)b.cols], b.cols);
    }
    return out;
  }
  fprintf(stderr, "vira tensor error: sub shape mismatch (%dx%d) - (%dx%d)\n",
          a.rows, a.cols, b.rows, b.cols);
  exit(1);
}

// SIMD-accelerated tensor scalar scaling: out = t * scale (preserves 1D..4D shape)
static inline vira_tensor_f32
vira_tensor_scale(vira_arena *opt_arena, vira_tensor_f32 t, vira_f32 scale) {
  _vira_tensor_norm_shape(&t);
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, t.n_dims, t.dims)
                : vira_tensor_f32_alloc_nd(t.n_dims, t.dims);
  int32_t total = t.rows * t.cols;
  int32_t i = 0;

#if defined(__AVX2__)
  __m256 vs = _mm256_set1_ps(scale);
  for (; i + 8 <= total; i += 8) {
    __m256 v = _mm256_loadu_ps(&t.data[i]);
    _mm256_storeu_ps(&out.data[i], _mm256_mul_ps(v, vs));
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t vs = vdupq_n_f32(scale);
  for (; i + 4 <= total; i += 4) {
    float32x4_t v = vld1q_f32(&t.data[i]);
    vst1q_f32(&out.data[i], vmulq_f32(v, vs));
  }
#elif defined(__SSE2__)
  __m128 vs = _mm_set1_ps(scale);
  for (; i + 4 <= total; i += 4) {
    __m128 v = _mm_loadu_ps(&t.data[i]);
    _mm_storeu_ps(&out.data[i], _mm_mul_ps(v, vs));
  }
#endif
  for (; i < total; i++) {
    out.data[i] = t.data[i] * scale;
  }
  return out;
}

static inline void vira_tensor_f32_free(vira_tensor_f32 *t) {
  if (t != NULL && !t->is_arena && !t->is_view && t->raw_alloc != NULL) {
    free(t->raw_alloc);
    t->raw_alloc = NULL;
    t->data = NULL;
    _vira_record_free();
  }
}

static inline vira_tensor_f32 vira_tensor_f32_clone(vira_tensor_f32 t) {
  if (t.data == NULL || t.rows <= 0 || t.cols <= 0) {
    return (vira_tensor_f32){0};
  }
  _vira_tensor_norm_shape(&t);
  vira_tensor_f32 copy = vira_tensor_f32_alloc_nd(t.n_dims, t.dims);
  size_t bytes = (size_t)t.rows * (size_t)t.cols * sizeof(vira_f32);
  memcpy(copy.data, t.data, bytes);
  return copy;
}

// ============================================================================
// 4. MULTI-THREADED LOCK-FREE WORK-STEALING ACTOR POOL & SUPERVISOR
// ============================================================================

#define VIRA_WS_CAP 4096
#define VIRA_WS_MASK (VIRA_WS_CAP - 1)
#define VIRA_MAX_WORKERS 32
#define VIRA_ACTOR_MAX_DEFERS 128

#ifndef _VIRA_TASK_DEFINED
#define _VIRA_TASK_DEFINED
typedef void (*vira_task_fn)(void *env);

typedef struct _vira_task {
  vira_task_fn fn;
  void *env;
  struct _vira_task *next_actor_msg; // Intrusive MPSC link
  volatile int32_t done;             // 0 = pending, 1 = completed on worker pool
} vira_task;
#endif

typedef void (*vira_defer_fn)(void *ptr);

typedef struct {
  vira_defer_fn fn;
  void *ptr;
  bool is_arena;
} vira_defer_entry;

static void _vira_actor_drain_task(void *env);
static inline void vira_scheduler_await(vira_task *task);

typedef enum {
  VIRA_ACTOR_RUNNING = 0,
  VIRA_ACTOR_FAULTED = 1,
  VIRA_ACTOR_STOPPED = 2,
} vira_actor_state;

struct _vira_actor_ctx {
  int32_t id;
  const char *name;
  vira_actor_state state;
  int32_t restarts;
  int32_t max_restarts;
  vira_arena isolated_heap; // Wiped in O(1) if the actor panics
  jmp_buf fault_boundary;   // Thread-safe fault jump target when active
  vira_string last_error;
  bool in_actor;
  volatile uint8_t mailbox_lock; // Ensures single-actor sequential state safety
  int32_t last_worker_id;        // Tracks which worker thread executed the actor
  vira_task * volatile mbox_head;
  vira_task * volatile mbox_tail;
  volatile int32_t scheduled;
  vira_task drain_task;
  vira_defer_entry defer_stack[VIRA_ACTOR_MAX_DEFERS];
  int32_t defer_top;
};

// Thread-Local Storage (TLS): Each worker pthread tracks its own active actor!
static __thread vira_actor_ctx *_vira_tls_actor = NULL;
static __thread int32_t _vira_tls_worker_id = -1;

static inline bool _vira_in_active_actor(void) {
  return _vira_tls_actor != NULL && _vira_tls_actor->in_actor;
}

static inline int32_t _vira_actor_defer_mark(void) {
  if (_vira_tls_actor != NULL && _vira_tls_actor->in_actor) {
    return _vira_tls_actor->defer_top;
  }
  return 0;
}

static inline void _vira_actor_defer_restore(int32_t base) {
  if (_vira_tls_actor != NULL && _vira_tls_actor->in_actor) {
    if (base >= 0 && base <= _vira_tls_actor->defer_top) {
      _vira_tls_actor->defer_top = base;
    }
  }
}

static inline void _vira_actor_defer_push_ex(vira_defer_fn fn, void *ptr,
                                             bool is_arena) {
  if (_vira_tls_actor != NULL && _vira_tls_actor->in_actor && ptr != NULL &&
      fn != NULL) {
    if (_vira_tls_actor->defer_top < VIRA_ACTOR_MAX_DEFERS) {
      _vira_tls_actor->defer_stack[_vira_tls_actor->defer_top++] =
          (vira_defer_entry){.fn = fn, .ptr = ptr, .is_arena = is_arena};
    }
  }
}

static inline void _vira_actor_defer_push(vira_defer_fn fn, void *ptr) {
  _vira_actor_defer_push_ex(fn, ptr, false);
}

static inline void _vira_actor_defer_pop(const void *ptr) {
  if (_vira_tls_actor != NULL && _vira_tls_actor->in_actor && ptr != NULL) {
    for (int32_t i = _vira_tls_actor->defer_top - 1; i >= 0; i--) {
      if (_vira_tls_actor->defer_stack[i].ptr == ptr) {
        for (int32_t j = i; j < _vira_tls_actor->defer_top - 1; j++) {
          _vira_tls_actor->defer_stack[j] = _vira_tls_actor->defer_stack[j + 1];
        }
        _vira_tls_actor->defer_top--;
        return;
      }
    }
  }
}

static inline bool _vira_actor_defer_arenas_contain(const vira_actor_ctx *ctx,
                                                    const void *ptr) {
  if (ctx == NULL || ptr == NULL)
    return false;
  for (int32_t i = 0; i < ctx->defer_top; i++) {
    if (ctx->defer_stack[i].is_arena && ctx->defer_stack[i].ptr != NULL) {
      if (vira_arena_contains((const vira_arena *)ctx->defer_stack[i].ptr,
                              ptr)) {
        return true;
      }
    }
  }
  return false;
}

static inline void _vira_actor_defer_unwind(vira_actor_ctx *ctx) {
  if (ctx == NULL)
    return;
  while (ctx->defer_top > 0) {
    ctx->defer_top--;
    vira_defer_entry e = ctx->defer_stack[ctx->defer_top];
    if (e.fn != NULL && e.ptr != NULL) {
      e.fn(e.ptr);
    }
  }
}

static inline void vira_actor_ctx_init(vira_actor_ctx *ctx, int32_t id,
                                       const char *name, int32_t max_restarts) {
  ctx->id = id;
  ctx->name = name;
  ctx->state = VIRA_ACTOR_RUNNING;
  ctx->restarts = 0;
  ctx->max_restarts = max_restarts;
  ctx->isolated_heap =
      (vira_arena){.buf = NULL,
                   .cap = 0,
                   .offset = 0,
                   .overflow_slabs = NULL}; // Lazy SIMD-aligned scratch arena
  ctx->last_error = vira_str_lit("", 0);
  ctx->in_actor = false;
  ctx->mailbox_lock = 0;
  ctx->last_worker_id = -1;
  ctx->mbox_head = NULL;
  ctx->mbox_tail = NULL;
  ctx->scheduled = 0;
  ctx->drain_task.fn = _vira_actor_drain_task;
  ctx->drain_task.env = ctx;
  ctx->drain_task.next_actor_msg = NULL;
  ctx->drain_task.done = 0;
  ctx->defer_top = 0;
}

static inline void vira_actor_ctx_destroy(vira_actor_ctx *ctx) {
  if (__atomic_load_n(&ctx->scheduled, __ATOMIC_ACQUIRE)) {
    vira_scheduler_await(&ctx->drain_task);
    while (__atomic_load_n(&ctx->scheduled, __ATOMIC_ACQUIRE)) {
      vira_cpu_relax();
    }
  }
  if (!ctx->last_error.is_lit && ctx->last_error.data != NULL) {
    vira_str_free(&ctx->last_error);
  }
  ctx->defer_top = 0;
  vira_arena_destroy(&ctx->isolated_heap);
}

static inline void vira_panic(vira_string msg) {
  if (_vira_tls_actor != NULL && _vira_tls_actor->in_actor) {
    vira_actor_ctx *ctx = _vira_tls_actor;
    ctx->state = VIRA_ACTOR_FAULTED;
    if (!ctx->last_error.is_lit && ctx->last_error.data != NULL) {
      vira_str_free(&ctx->last_error);
    }
    bool in_arena = vira_arena_contains(&ctx->isolated_heap, msg.data) ||
                    _vira_actor_defer_arenas_contain(ctx, msg.data);
    vira_string saved_err;
    if (msg.data == NULL || msg.len <= 0) {
      saved_err = vira_str_lit("", 0);
    } else if (msg.is_lit && !in_arena) {
      saved_err = msg;
    } else {
      char *buf = (char *)malloc((size_t)msg.len + 1);
      if (buf != NULL) {
        memcpy(buf, msg.data, (size_t)msg.len);
        buf[msg.len] = '\0';
        _vira_record_alloc();
        saved_err = (vira_string){
            .data = (const uint8_t *)buf, .len = msg.len, .is_lit = false};
      } else {
        saved_err = vira_str_lit("out of memory during panic", 26);
      }
    }
    // Unwind all live stack-frame heap allocations BEFORE longjmp!
    _vira_actor_defer_unwind(ctx);
    ctx->last_error = saved_err;
    longjmp(ctx->fault_boundary, 1);
  }
  fprintf(stderr, "vira unrecoverable panic: %.*s\n", msg.len,
          (const char *)msg.data);
  exit(1);
}

// --- CHASE-LEV STYLE LOCK-FREE WORK-STEALING DEQUE ---
typedef struct _vira_overflow_node {
  vira_task *task;
  struct _vira_overflow_node *next;
} vira_overflow_node;

typedef struct {
  vira_overflow_node *head;
  vira_overflow_node *tail;
  pthread_mutex_t mu;
} vira_overflow_queue;

typedef struct {
  int32_t worker_id;
  pthread_t thread;
  VIRA_CACHELINE_ALIGNED volatile int64_t top;    // Modified via CAS by thieves (FIFO steal)
  VIRA_CACHELINE_ALIGNED volatile int64_t bottom; // Modified by owner (LIFO push/pop)
  VIRA_CACHELINE_ALIGNED volatile uint8_t push_lock; // Serializes external producers (multi-producer safe)
  VIRA_CACHELINE_ALIGNED vira_task *ring[VIRA_WS_CAP];
} vira_ws_deque;

typedef struct {
  int32_t num_workers;
  volatile bool initialized;
  volatile int32_t shutdown;
  VIRA_CACHELINE_ALIGNED volatile uint32_t rr_cursor; // Isolate on its own cache line!
  VIRA_CACHELINE_ALIGNED vira_overflow_queue overflow;
  VIRA_CACHELINE_ALIGNED vira_ws_deque workers[VIRA_MAX_WORKERS];
} vira_scheduler;

static vira_scheduler _vira_sched = {0};
static pthread_mutex_t _vira_sched_init_mu = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
  pthread_mutex_t mu;
  pthread_cond_t cv;
  volatile int32_t sleepers;
} vira_parker;

static vira_parker _vira_parker = {
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0
};

static inline void _vira_overflow_push(vira_task *t) {
  vira_overflow_node *node = (vira_overflow_node *)malloc(sizeof(vira_overflow_node));
  if (!node) {
    fprintf(stderr, "vira panic: out of memory in scheduler overflow queue\n");
    exit(1);
  }
  node->task = t;
  node->next = NULL;
  pthread_mutex_lock(&_vira_sched.overflow.mu);
  if (_vira_sched.overflow.tail) {
    _vira_sched.overflow.tail->next = node;
    _vira_sched.overflow.tail = node;
  } else {
    _vira_sched.overflow.head = node;
    _vira_sched.overflow.tail = node;
  }
  pthread_mutex_unlock(&_vira_sched.overflow.mu);
}

static inline vira_task *_vira_overflow_pop(void) {
  if (!__atomic_load_n(&_vira_sched.overflow.head, __ATOMIC_ACQUIRE))
    return NULL;
  pthread_mutex_lock(&_vira_sched.overflow.mu);
  vira_overflow_node *node = _vira_sched.overflow.head;
  vira_task *task = NULL;
  if (node) {
    task = node->task;
    _vira_sched.overflow.head = node->next;
    if (!_vira_sched.overflow.head) {
      _vira_sched.overflow.tail = NULL;
    }
    free(node);
  }
  pthread_mutex_unlock(&_vira_sched.overflow.mu);
  return task;
}

// Push task onto a deque's bottom
static inline bool vira_ws_push(vira_ws_deque *q, vira_task *t) {
  int64_t b = __atomic_load_n(&q->bottom, __ATOMIC_RELAXED);
  int64_t top = __atomic_load_n(&q->top, __ATOMIC_ACQUIRE);
  if (b - top >= VIRA_WS_CAP - 1) {
    return false; // Queue full
  }
  __atomic_store_n(&q->ring[b & VIRA_WS_MASK], t, __ATOMIC_RELAXED);
  __atomic_thread_fence(__ATOMIC_RELEASE);
  __atomic_store_n(&q->bottom, b + 1, __ATOMIC_RELEASE);
  return true;
}

// Multi-producer-safe push (consumers are still lock-free via CAS on top)
static inline bool vira_ws_push_mp(vira_ws_deque *q, vira_task *t) {
  while (__atomic_exchange_n(&q->push_lock, 1, __ATOMIC_ACQUIRE)) {
    vira_cpu_relax();
  }
  bool ok = vira_ws_push(q, t);
  __atomic_store_n(&q->push_lock, 0, __ATOMIC_RELEASE);
  if (!ok) {
    _vira_overflow_push(t);
  }
  return true;
}

// Local LIFO pop by queue owner
static inline vira_task *vira_ws_pop(vira_ws_deque *q) {
  int64_t b = __atomic_load_n(&q->bottom, __ATOMIC_RELAXED);
  int64_t t = __atomic_load_n(&q->top, __ATOMIC_ACQUIRE);
  if (t >= b) {
    return NULL; // Fast exit if empty
  }

  while (__atomic_exchange_n(&q->push_lock, 1, __ATOMIC_ACQUIRE)) {
    vira_cpu_relax();
  }
  b = __atomic_load_n(&q->bottom, __ATOMIC_RELAXED) - 1;
  __atomic_store_n(&q->bottom, b, __ATOMIC_RELAXED);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  t = __atomic_load_n(&q->top, __ATOMIC_RELAXED);
  vira_task *task = NULL;
  if (t <= b) {
    task = __atomic_load_n(&q->ring[b & VIRA_WS_MASK], __ATOMIC_RELAXED);
    if (t == b) {
      if (!__atomic_compare_exchange_n(&q->top, &t, t + 1, false,
                                       __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {
        task = NULL;
      }
      __atomic_store_n(&q->bottom, b + 1, __ATOMIC_RELAXED);
    }
  } else {
    __atomic_store_n(&q->bottom, b + 1, __ATOMIC_RELAXED);
  }
  __atomic_store_n(&q->push_lock, 0, __ATOMIC_RELEASE);
  return task;
}

// Pop or steal a task from the top of any deque using lock-free CAS
static inline vira_task *vira_ws_steal(vira_ws_deque *q) {
  int64_t t = __atomic_load_n(&q->top, __ATOMIC_ACQUIRE);
  int64_t b = __atomic_load_n(&q->bottom, __ATOMIC_ACQUIRE);
  if (t >= b) {
    return NULL; // Fast exit: no SEQ_CST fence when empty!
  }

  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  t = __atomic_load_n(&q->top, __ATOMIC_ACQUIRE);
  b = __atomic_load_n(&q->bottom, __ATOMIC_ACQUIRE);
  if (t >= b) {
    return NULL;
  }

  vira_task *task =
      __atomic_load_n(&q->ring[t & VIRA_WS_MASK], __ATOMIC_ACQUIRE);
  if (!__atomic_compare_exchange_n(&q->top, &t, t + 1, false, __ATOMIC_SEQ_CST,
                                   __ATOMIC_RELAXED)) {
    return NULL; // Lost race to another thief
  }
  return task;
}

static inline bool _vira_any_work(void) {
  if (__atomic_load_n(&_vira_sched.overflow.head, __ATOMIC_ACQUIRE) != NULL) {
    return true;
  }
  for (int32_t i = 0; i < _vira_sched.num_workers; i++) {
    vira_ws_deque *q = &_vira_sched.workers[i];
    if (__atomic_load_n(&q->top, __ATOMIC_ACQUIRE) <
        __atomic_load_n(&q->bottom, __ATOMIC_ACQUIRE)) {
      return true;
    }
  }
  return false;
}

static inline void vira_sched_wake_one(void) {
  // Fast check without a full fence if no one is sleeping
  if (__atomic_load_n(&_vira_parker.sleepers, __ATOMIC_ACQUIRE) <= 0) {
    return;
  }
  // Atomically claim one sleeper so concurrent producers don't pile onto the mutex
  int32_t s = __atomic_load_n(&_vira_parker.sleepers, __ATOMIC_ACQUIRE);
  while (s > 0) {
    if (__atomic_compare_exchange_n(&_vira_parker.sleepers, &s, s - 1, true,
                                    __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE)) {
      pthread_mutex_lock(&_vira_parker.mu);
      pthread_cond_signal(&_vira_parker.cv);
      pthread_mutex_unlock(&_vira_parker.mu);
      return;
    }
  }
}

static void _vira_park(void) {
  pthread_mutex_lock(&_vira_parker.mu);
  __atomic_add_fetch(&_vira_parker.sleepers, 1, __ATOMIC_SEQ_CST);
  if (!_vira_any_work() && !__atomic_load_n(&_vira_sched.shutdown, __ATOMIC_ACQUIRE)) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC) && !defined(__APPLE__)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    ts.tv_nsec += 2000000;
    if (ts.tv_nsec >= 1000000000L) {
      ts.tv_sec++;
      ts.tv_nsec -= 1000000000L;
    }
    pthread_cond_timedwait(&_vira_parker.cv, &_vira_parker.mu, &ts);
  }
  // If we timed out (weren't claimed by wake_one), decrement sleepers safely
  int32_t s = __atomic_load_n(&_vira_parker.sleepers, __ATOMIC_RELAXED);
  while (s > 0 && !__atomic_compare_exchange_n(&_vira_parker.sleepers, &s, s - 1,
                                               true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
  }
  pthread_mutex_unlock(&_vira_parker.mu);
}

static inline uint32_t _vira_xorshift(uint32_t *s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return *s = x;
}

// Own deque first, then sweep victims starting at a random offset, then overflow queue
static inline vira_task *_vira_find_task(vira_ws_deque *self, uint32_t *rng,
                                         bool *stolen) {
  *stolen = false;
  vira_task *task = vira_ws_pop(self);
  if (task) {
    return task;
  }

  int32_t n = _vira_sched.num_workers;
  if (n > 1) {
    int32_t start = (int32_t)(_vira_xorshift(rng) % (uint32_t)n);
    for (int32_t i = 0; i < n; i++) {
      int32_t v = (start + i) % n;
      if (v == self->worker_id) {
        continue;
      }
      task = vira_ws_steal(&_vira_sched.workers[v]);
      if (task) {
        *stolen = true;
        return task;
      }
    }
  }

  task = _vira_overflow_pop();
  if (task) {
    *stolen = true;
    return task;
  }
  return NULL;
}

#define VIRA_SPIN_ITERS 64
#define VIRA_YIELD_ITERS 128

static void *_vira_worker_loop(void *arg) {
  vira_ws_deque *self = (vira_ws_deque *)arg;
  _vira_tls_worker_id = self->worker_id;

  uint32_t rng = 0x9E3779B9u * (uint32_t)(self->worker_id + 1);
  int64_t local_run = 0, local_steals = 0;
  int32_t idle = 0;

  while (true) {
    bool stolen;
    vira_task *task = _vira_find_task(self, &rng, &stolen);

    if (task) {
      idle = 0;
      if (stolen) {
        local_steals++;
      }

      vira_task_fn fn = task->fn; // read before publishing done
      void *env = task->env;
      fn(env);
      local_run++;
      __atomic_store_n(&task->done, 1, __ATOMIC_RELEASE); // task may be freed after this
      continue;
    }

    if (__atomic_load_n(&_vira_sched.shutdown, __ATOMIC_ACQUIRE)) {
      if (!_vira_any_work()) {
        break; // All worker deques and overflow drained! Clean termination
      }
    }

    // Idle: spin -> yield -> park
    if (idle < VIRA_SPIN_ITERS) {
      vira_cpu_relax();
    } else if (idle < VIRA_SPIN_ITERS + VIRA_YIELD_ITERS) {
      sched_yield();
    } else {
      if (local_run) {
        __atomic_add_fetch(&_vira_total_tasks_run, local_run, __ATOMIC_RELAXED);
        local_run = 0;
      }
      if (local_steals) {
        __atomic_add_fetch(&_vira_total_work_steals, local_steals, __ATOMIC_RELAXED);
        local_steals = 0;
      }
      _vira_park();
      idle = VIRA_SPIN_ITERS; // after waking, skip straight to a short yield phase
      continue;
    }
    idle++;
  }

  if (local_run) {
    __atomic_add_fetch(&_vira_total_tasks_run, local_run, __ATOMIC_RELAXED);
  }
  if (local_steals) {
    __atomic_add_fetch(&_vira_total_work_steals, local_steals, __ATOMIC_RELAXED);
  }
  _vira_flush_tls_mem_stats();
  return NULL;
}

static inline void vira_scheduler_init(int32_t num_workers) {
  if (_vira_sched.initialized)
    return;
  if (num_workers <= 0 || num_workers > VIRA_MAX_WORKERS) {
    num_workers = 4;
  }
  _vira_sched.num_workers = num_workers;
  _vira_sched.shutdown = 0;
  _vira_sched.rr_cursor = 0;

  _vira_sched.overflow.head = NULL;
  _vira_sched.overflow.tail = NULL;
  pthread_mutex_init(&_vira_sched.overflow.mu, NULL);

  pthread_condattr_t cattr;
  pthread_condattr_init(&cattr);
#if defined(CLOCK_MONOTONIC) && !defined(__APPLE__)
  pthread_condattr_setclock(&cattr, CLOCK_MONOTONIC);
#endif
  pthread_cond_init(&_vira_parker.cv, &cattr);
  pthread_condattr_destroy(&cattr);

  for (int32_t i = 0; i < num_workers; i++) {
    _vira_sched.workers[i].worker_id = i;
    _vira_sched.workers[i].top = 0;
    _vira_sched.workers[i].bottom = 0;
    _vira_sched.workers[i].push_lock = 0;
    pthread_create(&_vira_sched.workers[i].thread, NULL, _vira_worker_loop,
                   &_vira_sched.workers[i]);
  }
  __atomic_store_n(&_vira_sched.initialized, true, __ATOMIC_RELEASE);
}

static inline void vira_scheduler_shutdown(void) {
  if (!__atomic_load_n(&_vira_sched.initialized, __ATOMIC_ACQUIRE))
    return;
  pthread_mutex_lock(&_vira_sched_init_mu);
  if (!_vira_sched.initialized) {
    pthread_mutex_unlock(&_vira_sched_init_mu);
    return;
  }
  __atomic_store_n(&_vira_sched.shutdown, 1, __ATOMIC_SEQ_CST);
  pthread_mutex_lock(&_vira_parker.mu);
  pthread_cond_broadcast(&_vira_parker.cv);
  pthread_mutex_unlock(&_vira_parker.mu);
  for (int32_t i = 0; i < _vira_sched.num_workers; i++) {
    pthread_join(_vira_sched.workers[i].thread, NULL);
  }

  // Drain any remaining overflow queue items
  pthread_mutex_lock(&_vira_sched.overflow.mu);
  vira_overflow_node *cur = _vira_sched.overflow.head;
  while (cur) {
    vira_overflow_node *next = cur->next;
    free(cur);
    cur = next;
  }
  _vira_sched.overflow.head = NULL;
  _vira_sched.overflow.tail = NULL;
  pthread_mutex_unlock(&_vira_sched.overflow.mu);

  _vira_flush_tls_mem_stats();
  __atomic_store_n(&_vira_sched.initialized, false, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&_vira_sched_init_mu);
}

// Thread-safe lazy initialization guard
static inline void vira_scheduler_ensure_init(void) {
  if (__atomic_load_n(&_vira_sched.initialized, __ATOMIC_ACQUIRE)) {
    return;
  }
  pthread_mutex_lock(&_vira_sched_init_mu);
  if (!_vira_sched.initialized) {
    vira_scheduler_init(4);
  }
  pthread_mutex_unlock(&_vira_sched_init_mu);
}

static inline int32_t vira_scheduler_num_workers(void) {
  return _vira_sched.num_workers;
}

// --- ASYNCHRONOUS FUTURE[T] PRIMITIVES (FAN-OUT + WORK-HELPING AWAIT) ---

typedef struct {
  vira_task *task;
  void *env;
  void (*drop_fn)(void *env);
} vira_future;

// Pushes a task onto a worker deque and returns immediately (non-blocking)
static inline void vira_scheduler_submit_async(vira_task *task) {
  vira_scheduler_ensure_init();
  __atomic_store_n(&task->done, 0, __ATOMIC_RELAXED);

  int32_t queue_idx = _vira_tls_worker_id;
  if (queue_idx < 0 || queue_idx >= _vira_sched.num_workers) {
    uint32_t seq =
        __atomic_fetch_add(&_vira_sched.rr_cursor, 1, __ATOMIC_RELAXED);
    queue_idx = (int32_t)(seq % (uint32_t)_vira_sched.num_workers);
  }

  vira_ws_push_mp(&_vira_sched.workers[queue_idx], task);
  vira_sched_wake_one();
}

static inline vira_task *vira_actor_pop_msg(vira_actor_ctx *ctx) {
  vira_task *head = ctx->mbox_head;
  if (!head) {
    if (__atomic_load_n(&ctx->mbox_tail, __ATOMIC_ACQUIRE) != NULL) {
      while ((head = __atomic_load_n(&ctx->mbox_head, __ATOMIC_ACQUIRE)) == NULL) {
        vira_cpu_relax();
      }
    } else {
      return NULL;
    }
  }

  vira_task *next = __atomic_load_n(&head->next_actor_msg, __ATOMIC_ACQUIRE);
  if (next != NULL) {
    ctx->mbox_head = next;
    return head;
  }

  vira_task *expected = head;
  if (__atomic_compare_exchange_n(&ctx->mbox_tail, &expected, NULL, false,
                                  __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    ctx->mbox_head = NULL;
    return head;
  }

  while ((next = __atomic_load_n(&head->next_actor_msg, __ATOMIC_ACQUIRE)) == NULL) {
    vira_cpu_relax();
  }
  ctx->mbox_head = next;
  return head;
}

static void _vira_actor_drain_task(void *env) {
  vira_actor_ctx *ctx = (vira_actor_ctx *)env;
  int32_t quota = 64;
  int32_t count = 0;

  while (quota-- > 0) {
    vira_task *msg = vira_actor_pop_msg(ctx);
    if (!msg) {
      break;
    }
    vira_task_fn fn = msg->fn;
    void *menv = msg->env;
    bool is_detached = (__atomic_load_n(&msg->done, __ATOMIC_RELAXED) == -1);
    fn(menv);
    count++;
    if (is_detached) {
      free(menv);
      _vira_record_free();
    } else {
      __atomic_store_n(&msg->done, 1, __ATOMIC_RELEASE);
    }
  }

  if (count > 1) {
    __atomic_add_fetch(&_vira_total_tasks_run, count - 1, __ATOMIC_RELAXED);
  }

  if (__atomic_load_n(&ctx->mbox_tail, __ATOMIC_ACQUIRE) != NULL ||
      ctx->mbox_head != NULL) {
    ctx->drain_task.fn = _vira_actor_drain_task;
    ctx->drain_task.env = ctx;
    ctx->drain_task.next_actor_msg = NULL;
    ctx->drain_task.done = 0;
    vira_scheduler_submit_async(&ctx->drain_task);
  } else {
    __atomic_store_n(&ctx->scheduled, 0, __ATOMIC_SEQ_CST);
    if (__atomic_load_n(&ctx->mbox_tail, __ATOMIC_SEQ_CST) != NULL ||
        __atomic_load_n(&ctx->mbox_head, __ATOMIC_SEQ_CST) != NULL) {
      if (__atomic_exchange_n(&ctx->scheduled, 1, __ATOMIC_SEQ_CST) == 0) {
        ctx->drain_task.fn = _vira_actor_drain_task;
        ctx->drain_task.env = ctx;
        ctx->drain_task.next_actor_msg = NULL;
        ctx->drain_task.done = 0;
        vira_scheduler_submit_async(&ctx->drain_task);
      }
    }
  }
}

static inline void vira_actor_send(vira_actor_ctx *ctx, vira_task *task) {
  vira_scheduler_ensure_init();
  task->next_actor_msg = NULL;
  if (__atomic_load_n(&task->done, __ATOMIC_RELAXED) != -1) {
    __atomic_store_n(&task->done, 0, __ATOMIC_RELAXED);
  }

  vira_task *prev = __atomic_exchange_n(&ctx->mbox_tail, task, __ATOMIC_ACQ_REL);
  if (prev != NULL) {
    __atomic_store_n(&prev->next_actor_msg, task, __ATOMIC_RELEASE);
  } else {
    __atomic_store_n(&ctx->mbox_head, task, __ATOMIC_RELEASE);
  }

  if (__atomic_exchange_n(&ctx->scheduled, 1, __ATOMIC_ACQ_REL) == 0) {
    ctx->drain_task.fn = _vira_actor_drain_task;
    ctx->drain_task.env = ctx;
    ctx->drain_task.next_actor_msg = NULL;
    ctx->drain_task.done = 0;
    vira_scheduler_submit_async(&ctx->drain_task);
  }
}

// Waits for a future's task to complete; while waiting, the caller helps steal
// pending tasks!
static inline void vira_scheduler_await(vira_task *task) {
  if (task == NULL)
    return;
  uint32_t rng = 0x85EBCA6Bu;
  int32_t idle = 0;
  while (__atomic_load_n(&task->done, __ATOMIC_ACQUIRE) == 0) {
    bool helped = false;
    int32_t n = _vira_sched.num_workers;
    if (n > 0) {
      int32_t start = (int32_t)(_vira_xorshift(&rng) % (uint32_t)n);
      for (int32_t i = 0; i < n; i++) {
        int32_t v = (start + i) % n;
        vira_task *stolen = vira_ws_steal(&_vira_sched.workers[v]);
        if (stolen == NULL) {
          stolen = _vira_overflow_pop();
        }
        if (stolen != NULL) {
          __atomic_add_fetch(&_vira_total_work_steals, 1, __ATOMIC_RELAXED);
          vira_task_fn fn = stolen->fn;
          void *env = stolen->env;
          fn(env);
          __atomic_add_fetch(&_vira_total_tasks_run, 1, __ATOMIC_RELAXED);
          __atomic_store_n(&stolen->done, 1, __ATOMIC_RELEASE);
          helped = true;
          idle = 0;
          break;
        }
      }
    }
    if (!helped) {
      if (idle < 32) {
        vira_cpu_relax();
      } else {
        sched_yield();
      }
      idle++;
    }
  }
}

// Synchronous helper (spawn + immediate await)
static inline void vira_scheduler_submit_and_wait(vira_task_fn fn, void *env) {
  vira_scheduler_ensure_init();
  vira_task task = {.fn = fn, .env = env, .done = 0};
  vira_scheduler_submit_async(&task);
  vira_scheduler_await(&task);
}

// Idempotent scope-exit cleanup injected by `internal/autofree` for Future[T]
static inline void vira_future_free(vira_future *f) {
  if (f != NULL && f->env != NULL) {
    // If an early return skipped `.await()`, wait for the worker thread to
    // finish before freeing unconsumed result and envelope
    if (f->task != NULL) {
      vira_scheduler_await(f->task);
    }
    if (f->drop_fn != NULL) {
      f->drop_fn(f->env);
      f->drop_fn = NULL;
    }
    free(f->env);
    f->env = NULL;
    f->task = NULL;
    _vira_record_free();
  }
}

// ============================================================================
// 5. BUILTIN I/O & DIAGNOSTICS
// ============================================================================

static inline void vira_print(vira_string s) {
  if (s.data != NULL && s.len > 0) {
    fwrite(s.data, 1, (size_t)s.len, stdout);
    fflush(stdout);
  }
}

static inline void vira_print_temp(vira_string s) {
  vira_print(s);
  if (!s.is_lit && s.data != NULL) {
    free((void *)s.data);
    _vira_record_free();
  }
}

static inline void vira_println(vira_string s) {
  if (s.data != NULL && s.len > 0) {
    fwrite(s.data, 1, (size_t)s.len, stdout);
  }
  fputc('\n', stdout);
  fflush(stdout);
}

static inline void vira_println_temp(vira_string s) {
  vira_println(s);
  if (!s.is_lit && s.data != NULL) {
    free((void *)s.data);
    _vira_record_free();
  }
}

static inline void vira_eprint(vira_string s) {
  if (s.data != NULL && s.len > 0) {
    fwrite(s.data, 1, (size_t)s.len, stderr);
    fflush(stderr);
  }
}

static inline void vira_eprint_temp(vira_string s) {
  vira_eprint(s);
  if (!s.is_lit && s.data != NULL) {
    free((void *)s.data);
    _vira_record_free();
  }
}

static inline void vira_eprintln(vira_string s) {
  if (s.data != NULL && s.len > 0) {
    fwrite(s.data, 1, (size_t)s.len, stderr);
  }
  fputc('\n', stderr);
  fflush(stderr);
}

static inline void vira_eprintln_temp(vira_string s) {
  vira_eprintln(s);
  if (!s.is_lit && s.data != NULL) {
    free((void *)s.data);
    _vira_record_free();
  }
}

static inline void vira_time_sleep_ms(int64_t ms) {
  if (ms <= 0) return;
#ifdef _WIN32
  Sleep((DWORD)ms);
#else
  struct timespec ts;
  ts.tv_sec = ms / 1000;
  ts.tv_nsec = (ms % 1000) * 1000000L;
  nanosleep(&ts, NULL);
#endif
}

static inline void vira_time_sleep_us(int64_t us) {
  if (us <= 0) return;
#ifdef _WIN32
  Sleep((DWORD)((us + 999) / 1000));
#else
  struct timespec ts;
  ts.tv_sec = us / 1000000;
  ts.tv_nsec = (us % 1000000) * 1000L;
  nanosleep(&ts, NULL);
#endif
}

static inline int64_t vira_time_now_us(void) {
#if defined(_WIN32)
  static LARGE_INTEGER freq;
  static int init = 0;
  if (!init) {
    QueryPerformanceFrequency(&freq);
    init = 1;
  }
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  return (int64_t)((counter.QuadPart * 1000000LL) / freq.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000LL + (int64_t)(ts.tv_nsec / 1000LL);
#endif
}

static inline int64_t vira_time_now_ms(void) {
  return vira_time_now_us() / 1000LL;
}

// ============================================================================
// 6. NATIVE FILE I/O & BUFFER OPERATIONS
// ============================================================================

#ifndef _Result_string_DEFINED
#define _Result_string_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  vira_string val;
} Result_string;
#endif

#ifndef _Result_bool_DEFINED
#define _Result_bool_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  bool val;
} Result_bool;
#endif

static inline bool vira_file_exists(vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));
  FILE *f = fopen(pbuf.ptr, "rb");
  vira_cstr_buf_free(&pbuf);
  if (f) {
    fclose(f);
    return true;
  }
  return false;
}

static inline bool vira_dir_exists(vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));
#ifdef _WIN32
  DWORD attr = GetFileAttributesA(pbuf.ptr);
  vira_cstr_buf_free(&pbuf);
  return (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
  struct stat st;
  int res = stat(pbuf.ptr, &st);
  vira_cstr_buf_free(&pbuf);
  return (res == 0 && S_ISDIR(st.st_mode));
#endif
}

static inline Result_string vira_read_file(vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "rb");
  if (!f) {
    vira_string err = vira_str_format("failed to open file '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_string){.is_err = true, .err = err};
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 0) {
    fclose(f);
    vira_string err = vira_str_format("failed to seek in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_string){.is_err = true, .err = err};
  }
  vira_cstr_buf_free(&pbuf);

  char *buf = (char *)malloc((size_t)sz + 1);
  if (!buf) {
    fclose(f);
    return (Result_string){
        .is_err = true, .err = vira_str_lit("out of memory reading file", 26)};
  }
  size_t read_bytes = fread(buf, 1, (size_t)sz, f);
  buf[read_bytes] = '\0';
  fclose(f);
  _vira_record_alloc();
  return (Result_string){.is_err = false,
                         .val = (vira_string){.data = (const uint8_t *)buf,
                                              .len = (int32_t)read_bytes,
                                              .is_lit = false}};
}

static inline Result_bool vira_write_file(vira_string path,
                                          vira_string content) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "wb");
  if (!f) {
    vira_string err =
        vira_str_format("failed to open file '%s' for writing", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_bool){.is_err = true, .err = err};
  }
  if (content.len > 0 && content.data != NULL) {
    size_t written = fwrite(content.data, 1, (size_t)content.len, f);
    if (written != (size_t)content.len) {
      fclose(f);
      vira_string err = vira_str_format(
          "failed to write complete content to '%s'", pbuf.ptr);
      vira_cstr_buf_free(&pbuf);
      return (Result_bool){.is_err = true, .err = err};
    }
  }
  fclose(f);
  vira_cstr_buf_free(&pbuf);
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_bool vira_append_file(vira_string path,
                                           vira_string content) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "ab");
  if (!f) {
    vira_string err =
        vira_str_format("failed to open file '%s' for appending", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_bool){.is_err = true, .err = err};
  }
  if (content.len > 0 && content.data != NULL) {
    size_t written = fwrite(content.data, 1, (size_t)content.len, f);
    if (written != (size_t)content.len) {
      fclose(f);
      vira_string err = vira_str_format(
          "failed to append complete content to '%s'", pbuf.ptr);
      vira_cstr_buf_free(&pbuf);
      return (Result_bool){.is_err = true, .err = err};
    }
  }
  fclose(f);
  vira_cstr_buf_free(&pbuf);
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_bool vira_remove_file(vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  if (remove(pbuf.ptr) == 0) {
    vira_cstr_buf_free(&pbuf);
    return (Result_bool){.is_err = false, .val = true};
  }
  vira_string err = vira_str_format("failed to remove file '%s'", pbuf.ptr);
  vira_cstr_buf_free(&pbuf);
  return (Result_bool){.is_err = true, .err = err};
}

static inline int64_t vira_io_file_open(vira_string path, vira_string mode) {
  if (path.len == 0 || path.data == NULL || mode.len == 0 ||
      mode.data == NULL) {
    return 0;
  }
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  char stack_mode[16];
  vira_cstr_buf mbuf = vira_str_to_cstr_buf(mode, stack_mode, sizeof(stack_mode));

  FILE *f = fopen(pbuf.ptr, mbuf.ptr);
  vira_cstr_buf_free(&mbuf);
  vira_cstr_buf_free(&pbuf);
  return (int64_t)(uintptr_t)f;
}

static inline int32_t vira_io_file_read(int64_t handle, vira_slice_u8 *buf,
                                        int32_t max_len) {
  if (handle == 0 || buf == NULL || buf->data == NULL || max_len <= 0) {
    return 0;
  }
  FILE *f = (FILE *)(uintptr_t)handle;
  int32_t can_read = buf->cap < max_len ? buf->cap : max_len;
  if (can_read <= 0)
    return 0;
  size_t n = fread(buf->data, 1, (size_t)can_read, f);
  buf->len = (int32_t)n;
  return (int32_t)n;
}

static inline int32_t vira_io_file_write(int64_t handle, vira_slice_u8 buf,
                                         int32_t len) {
  if (handle == 0 || buf.data == NULL || len <= 0) {
    return 0;
  }
  FILE *f = (FILE *)(uintptr_t)handle;
  int32_t to_write = buf.len < len ? buf.len : len;
  if (to_write <= 0)
    return 0;
  size_t n = fwrite(buf.data, 1, (size_t)to_write, f);
  return (int32_t)n;
}

static inline int64_t vira_io_file_seek(int64_t handle, int64_t offset,
                                        int32_t whence) {
  if (handle == 0)
    return -1;
  FILE *f = (FILE *)(uintptr_t)handle;
  int c_whence = SEEK_SET;
  if (whence == 1)
    c_whence = SEEK_CUR;
  else if (whence == 2)
    c_whence = SEEK_END;
  if (fseek(f, (long)offset, c_whence) != 0) {
    return -1;
  }
  long pos = ftell(f);
  return (int64_t)pos;
}

static inline int32_t vira_io_file_flush(int64_t handle) {
  if (handle == 0)
    return -1;
  FILE *f = (FILE *)(uintptr_t)handle;
  return fflush(f) == 0 ? 0 : -1;
}

static inline int32_t vira_io_file_close(int64_t handle) {
  if (handle == 0)
    return 0;
  FILE *f = (FILE *)(uintptr_t)handle;
  return fclose(f) == 0 ? 0 : -1;
}

static inline int64_t vira_io_file_size(int64_t handle) {
  if (handle == 0)
    return -1;
  FILE *f = (FILE *)(uintptr_t)handle;
  long cur = ftell(f);
  if (fseek(f, 0, SEEK_END) != 0)
    return -1;
  long sz = ftell(f);
  fseek(f, cur, SEEK_SET);
  return (int64_t)sz;
}

// ============================================================================
// 7. PRETRAINED TENSOR BINARY WEIGHT I/O (PYTORCH / SAFETENSORS / GGUF COMPAT)
// ============================================================================

#ifndef _Result_tensor_f32_DEFINED
#define _Result_tensor_f32_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  vira_tensor_f32 val;
} Result_tensor_f32;
#endif

static inline Result_tensor_f32 vira_tensor_load_bin(vira_arena *opt_arena,
                                                     int32_t rows, int32_t cols,
                                                     vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "rb");
  if (!f) {
    vira_string err =
        vira_str_format("could not open weight file '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  vira_tensor_f32 t = opt_arena
                          ? vira_tensor_f32_arena_alloc(opt_arena, rows, cols)
                          : vira_tensor_f32_alloc(rows, cols);

  size_t expected = (size_t)rows * (size_t)cols;
  size_t read_count = fread(t.data, sizeof(vira_f32), expected, f);
  fclose(f);

  if (read_count != expected) {
    if (!opt_arena)
      vira_tensor_f32_free(&t);
    vira_string err = vira_str_format(
        "weight file '%s' size mismatch: expected %zu floats, read %zu",
        pbuf.ptr, expected, read_count);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  vira_cstr_buf_free(&pbuf);
  return (Result_tensor_f32){
      .is_err = false, .err = vira_str_lit("", 0), .val = t};
}

static inline Result_bool vira_tensor_save_bin(vira_tensor_f32 t,
                                               vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "wb");
  if (!f) {
    vira_string err = vira_str_format(
        "failed to open file '%s' for writing weights", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_bool){.is_err = true, .err = err, .val = false};
  }

  size_t total = (size_t)t.rows * (size_t)t.cols;
  size_t written = fwrite(t.data, sizeof(vira_f32), total, f);
  fclose(f);

  if (written != total) {
    vira_string err =
        vira_str_format("failed to write complete weights to '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_bool){.is_err = true, .err = err, .val = false};
  }

  vira_cstr_buf_free(&pbuf);
  return (Result_bool){
      .is_err = false, .err = vira_str_lit("", 0), .val = true};
}

// ============================================================================
// 12. HIGH-PERFORMANCE NETWORKING & HTTP 1.1 WEB ENGINE
// ============================================================================

static inline void _vira_net_init(void) {
#ifdef _WIN32
  static int initialized = 0;
  if (!initialized) {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    initialized = 1;
  }
#else
  static int initialized = 0;
  if (!initialized) {
    signal(SIGPIPE, SIG_IGN);
    initialized = 1;
  }
#endif
}

// --- LEGACY MINIMAL SOCKET WRAPPERS ---
static inline int64_t vira_net_listen(int64_t port) {
  _vira_net_init();
  vira_socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd == VIRA_INVALID_SOCKET)
    return -1;

  int opt = 1;
#ifdef _WIN32
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));
#else
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons((uint16_t)port);

  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) == VIRA_SOCKET_ERROR) {
    closesocket(fd);
    return -1;
  }

  if (listen(fd, 128) == VIRA_SOCKET_ERROR) {
    closesocket(fd);
    return -1;
  }

  return (int64_t)fd;
}

static inline int64_t vira_net_accept(int64_t server_fd) {
  _vira_net_init();
  struct sockaddr_in client_addr;
  socklen_t addr_len = sizeof(client_addr);
  vira_socket_t client = accept((vira_socket_t)server_fd,
                                (struct sockaddr *)&client_addr, &addr_len);
  if (client == VIRA_INVALID_SOCKET)
    return -1;
  return (int64_t)client;
}

static inline vira_string vira_net_read(int64_t client_fd, int64_t max_bytes) {
  if (max_bytes <= 0)
    max_bytes = 4096;
  char *buf = (char *)malloc(max_bytes + 1);
  if (!buf)
    return vira_str_lit("", 0);
  _vira_record_alloc();

  int n = recv((vira_socket_t)client_fd, buf, (int)max_bytes, 0);
  if (n <= 0) {
    free(buf);
    _vira_record_free();
    return vira_str_lit("", 0);
  }
  buf[n] = '\0';
  return (vira_string){.data = (const uint8_t *)buf, .len = n, .is_lit = false};
}

static inline int64_t vira_net_write(int64_t client_fd, vira_string msg) {
  if (msg.len <= 0 || msg.data == NULL)
    return 0;
  int sent = send((vira_socket_t)client_fd, (const char *)msg.data, msg.len, 0);
  return (int64_t)sent;
}

static inline void vira_net_close(int64_t fd) {
  if (fd >= 0) {
    closesocket((vira_socket_t)fd);
  }
}

// --- STANDARD VIRA NETWORKING (TCP, UDP, MULTIPLEXING) ---

static inline int64_t vira_net_tcp_listen(vira_string host, int32_t port,
                                          int32_t backlog) {
  _vira_net_init();
  if (port < 0 || port > 65535)
    return -1;
  if (backlog <= 0)
    backlog = 128;

  char stack_host[256];
  vira_cstr_buf hbuf = vira_str_to_cstr_buf(host, stack_host, sizeof(stack_host));

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL, *p = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  const char *node =
      (host.len > 0 && strcmp(hbuf.ptr, "0.0.0.0") != 0) ? hbuf.ptr : NULL;
  int gai = getaddrinfo(node, port_buf, &hints, &res);
  vira_cstr_buf_free(&hbuf);
  if (gai != 0) {
    return -1;
  }

  vira_socket_t listen_fd = VIRA_INVALID_SOCKET;
  for (p = res; p != NULL; p = p->ai_next) {
    listen_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (listen_fd == VIRA_INVALID_SOCKET)
      continue;

    int opt = 1;
#ifdef _WIN32
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt,
               sizeof(opt));
#else
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
#endif

    if (bind(listen_fd, p->ai_addr, (socklen_t)p->ai_addrlen) == 0) {
      break;
    }

    closesocket(listen_fd);
    listen_fd = VIRA_INVALID_SOCKET;
  }

  freeaddrinfo(res);

  if (listen_fd == VIRA_INVALID_SOCKET)
    return -1;
  if (listen(listen_fd, (int)backlog) == VIRA_SOCKET_ERROR) {
    closesocket(listen_fd);
    return -1;
  }

  return (int64_t)listen_fd;
}

static inline int64_t vira_net_tcp_accept(int64_t server_fd) {
  _vira_net_init();
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  vira_socket_t client =
      accept((vira_socket_t)server_fd, (struct sockaddr *)&addr, &addr_len);
  if (client == VIRA_INVALID_SOCKET)
    return -1;
  return (int64_t)client;
}

static inline vira_string vira_net_peer_addr(int64_t fd) {
  if (fd < 0)
    return vira_str_lit("", 0);
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  if (getpeername((vira_socket_t)fd, (struct sockaddr *)&addr, &addr_len) !=
      0) {
    return vira_str_lit("", 0);
  }
  if (addr.ss_family == AF_INET) {
    struct sockaddr_in *s = (struct sockaddr_in *)&addr;
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &s->sin_addr, ip, sizeof(ip));
    int port = ntohs(s->sin_port);
    return vira_str_format("%s:%d", ip, port);
  } else if (addr.ss_family == AF_INET6) {
    struct sockaddr_in6 *s = (struct sockaddr_in6 *)&addr;
    char ip[INET6_ADDRSTRLEN];
    inet_ntop(AF_INET6, &s->sin6_addr, ip, sizeof(ip));
    int port = ntohs(s->sin6_port);
    return vira_str_format("[%s]:%d", ip, port);
  }
  return vira_str_lit("", 0);
}

static inline vira_string vira_net_local_addr(int64_t fd) {
  if (fd < 0)
    return vira_str_lit("", 0);
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  if (getsockname((vira_socket_t)fd, (struct sockaddr *)&addr, &addr_len) !=
      0) {
    return vira_str_lit("", 0);
  }
  if (addr.ss_family == AF_INET) {
    struct sockaddr_in *s = (struct sockaddr_in *)&addr;
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &s->sin_addr, ip, sizeof(ip));
    int port = ntohs(s->sin_port);
    return vira_str_format("%s:%d", ip, port);
  } else if (addr.ss_family == AF_INET6) {
    struct sockaddr_in6 *s = (struct sockaddr_in6 *)&addr;
    char ip[INET6_ADDRSTRLEN];
    inet_ntop(AF_INET6, &s->sin6_addr, ip, sizeof(ip));
    int port = ntohs(s->sin6_port);
    return vira_str_format("[%s]:%d", ip, port);
  }
  return vira_str_lit("", 0);
}

static inline int64_t vira_net_tcp_connect(vira_string host, int32_t port,
                                           int64_t timeout_ms) {
  _vira_net_init();
  if (port <= 0 || port > 65535)
    return -1;

  char stack_host[256];
  vira_cstr_buf hbuf = vira_str_to_cstr_buf(host, stack_host, sizeof(stack_host));

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL, *p = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int gai = getaddrinfo(hbuf.ptr, port_buf, &hints, &res);
  vira_cstr_buf_free(&hbuf);
  if (gai != 0) {
    return -1;
  }

  vira_socket_t conn_fd = VIRA_INVALID_SOCKET;
  for (p = res; p != NULL; p = p->ai_next) {
    conn_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (conn_fd == VIRA_INVALID_SOCKET)
      continue;

    if (timeout_ms <= 0) {
      if (connect(conn_fd, p->ai_addr, (socklen_t)p->ai_addrlen) == 0) {
        break;
      }
    } else {
#ifdef _WIN32
      u_long mode = 1;
      ioctlsocket(conn_fd, FIONBIO, &mode);
#else
      int orig_flags = fcntl((int)conn_fd, F_GETFL, 0);
      fcntl((int)conn_fd, F_SETFL, orig_flags | O_NONBLOCK);
#endif
      int ret = connect(conn_fd, p->ai_addr, (socklen_t)p->ai_addrlen);
      bool ok = false;
      if (ret == 0) {
        ok = true;
      } else {
#ifdef _WIN32
        int err = WSAGetLastError();
        if (err == WSAEWOULDBLOCK) {
#else
        if (errno == EINPROGRESS) {
#endif
          struct pollfd pfd;
          pfd.fd = conn_fd;
          pfd.events = POLLOUT;
          pfd.revents = 0;
          int pr = vira_poll(&pfd, 1, (int)timeout_ms);
          if (pr > 0 && (pfd.revents & POLLOUT)) {
            int sock_err = 0;
            socklen_t errlen = sizeof(sock_err);
            getsockopt(conn_fd, SOL_SOCKET, SO_ERROR, (char *)&sock_err,
                       &errlen);
            if (sock_err == 0) {
              ok = true;
            }
          }
        }
      }

#ifdef _WIN32
      mode = 0;
      ioctlsocket(conn_fd, FIONBIO, &mode);
#else
      fcntl((int)conn_fd, F_SETFL, orig_flags);
#endif
      if (ok) {
        break;
      }
    }

    closesocket(conn_fd);
    conn_fd = VIRA_INVALID_SOCKET;
  }

  freeaddrinfo(res);
  if (conn_fd == VIRA_INVALID_SOCKET)
    return -1;
  return (int64_t)conn_fd;
}

static inline int32_t vira_net_tcp_read(int64_t fd, vira_slice_u8 *buf,
                                        int32_t max_len) {
  if (fd < 0 || buf == NULL || buf->data == NULL || max_len <= 0) {
    return 0;
  }
  int32_t can_read = buf->cap < max_len ? buf->cap : max_len;
  if (can_read <= 0)
    return 0;

  int n = recv((vira_socket_t)fd, (char *)buf->data, (int)can_read, 0);
  if (n > 0) {
    buf->len = (int32_t)n;
    return (int32_t)n;
  }
  if (n == 0) {
    buf->len = 0;
    return 0;
  }
#ifdef _WIN32
  int werr = WSAGetLastError();
  if (werr == WSAEWOULDBLOCK)
    return -2;
#else
  if (errno == EAGAIN || errno == EWOULDBLOCK)
    return -2;
#endif
  return -1;
}

static inline int32_t vira_net_tcp_write(int64_t fd, vira_slice_u8 buf,
                                         int32_t len) {
  if (fd < 0 || buf.data == NULL || len <= 0) {
    return 0;
  }
  int32_t to_write = buf.len < len ? buf.len : len;
  if (to_write <= 0)
    return 0;

  int flags = 0;
#ifdef MSG_NOSIGNAL
  flags |= MSG_NOSIGNAL;
#endif
  int sent =
      send((vira_socket_t)fd, (const char *)buf.data, (int)to_write, flags);
  return (int32_t)sent;
}

static inline int32_t vira_net_tcp_close(int64_t fd) {
  if (fd < 0)
    return 0;
  return closesocket((vira_socket_t)fd) == 0 ? 0 : -1;
}

static inline int32_t vira_net_tcp_shutdown(int64_t fd, int32_t how) {
  if (fd < 0)
    return 0;
#ifdef _WIN32
  int c_how = (how == 0) ? SD_RECEIVE : ((how == 1) ? SD_SEND : SD_BOTH);
#else
  int c_how = (how == 0) ? SHUT_RD : ((how == 1) ? SHUT_WR : SHUT_RDWR);
#endif
  return shutdown((vira_socket_t)fd, c_how) == 0 ? 0 : -1;
}

static inline int32_t vira_net_set_nonblocking(int64_t fd, bool nonblocking) {
  if (fd < 0)
    return -1;
#ifdef _WIN32
  u_long mode = nonblocking ? 1 : 0;
  return ioctlsocket((vira_socket_t)fd, FIONBIO, &mode) == 0 ? 0 : -1;
#else
  int flags = fcntl((int)fd, F_GETFL, 0);
  if (flags < 0)
    return -1;
  flags = nonblocking ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
  return fcntl((int)fd, F_SETFL, flags) == 0 ? 0 : -1;
#endif
}

static inline int32_t vira_net_set_nodelay(int64_t fd, bool nodelay) {
  if (fd < 0)
    return -1;
  int opt = nodelay ? 1 : 0;
  return setsockopt((vira_socket_t)fd, IPPROTO_TCP, TCP_NODELAY,
                    (const char *)&opt, sizeof(opt)) == 0
             ? 0
             : -1;
}

static inline int32_t vira_net_set_reuseaddr(int64_t fd, bool reuse) {
  if (fd < 0)
    return -1;
  int opt = reuse ? 1 : 0;
  return setsockopt((vira_socket_t)fd, SOL_SOCKET, SO_REUSEADDR,
                    (const char *)&opt, sizeof(opt)) == 0
             ? 0
             : -1;
}

static inline int32_t vira_net_set_timeout(int64_t fd, int64_t read_timeout_ms,
                                           int64_t write_timeout_ms) {
  if (fd < 0)
    return -1;
#ifdef _WIN32
  DWORD r_to = (DWORD)read_timeout_ms;
  DWORD w_to = (DWORD)write_timeout_ms;
  if (read_timeout_ms >= 0)
    setsockopt((vira_socket_t)fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&r_to,
               sizeof(r_to));
  if (write_timeout_ms >= 0)
    setsockopt((vira_socket_t)fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&w_to,
               sizeof(w_to));
#else
  if (read_timeout_ms >= 0) {
    struct timeval tv;
    tv.tv_sec = read_timeout_ms / 1000;
    tv.tv_usec = (read_timeout_ms % 1000) * 1000;
    setsockopt((int)fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  }
  if (write_timeout_ms >= 0) {
    struct timeval tv;
    tv.tv_sec = write_timeout_ms / 1000;
    tv.tv_usec = (write_timeout_ms % 1000) * 1000;
    setsockopt((int)fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  }
#endif
  return 0;
}

static inline int32_t vira_net_poll(int64_t fd, int32_t events,
                                    int32_t timeout_ms) {
  if (fd < 0)
    return -1;
  struct pollfd pfd;
  pfd.fd = (vira_socket_t)fd;
  pfd.events = 0;
  if (events & 1)
    pfd.events |= POLLIN;
  if (events & 2)
    pfd.events |= POLLOUT;
  pfd.revents = 0;

  int n = vira_poll(&pfd, 1, (int)timeout_ms);
  if (n <= 0)
    return n;

  int32_t res = 0;
  if (pfd.revents & POLLIN)
    res |= 1;
  if (pfd.revents & POLLOUT)
    res |= 2;
  if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
    res |= 4;
  return res;
}

#if defined(__GNUC__) || defined(__clang__)
static __thread struct sockaddr_storage _vira_last_udp_sender;
static __thread socklen_t _vira_last_udp_sender_len = 0;
#else
static struct sockaddr_storage _vira_last_udp_sender;
static socklen_t _vira_last_udp_sender_len = 0;
#endif

static inline int64_t vira_net_udp_bind(vira_string host, int32_t port) {
  _vira_net_init();
  if (port < 0 || port > 65535)
    return -1;

  char stack_host[256];
  vira_cstr_buf hbuf = vira_str_to_cstr_buf(host, stack_host, sizeof(stack_host));

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL, *p = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_flags = AI_PASSIVE;

  const char *node =
      (host.len > 0 && strcmp(hbuf.ptr, "0.0.0.0") != 0) ? hbuf.ptr : NULL;
  int gai = getaddrinfo(node, port_buf, &hints, &res);
  vira_cstr_buf_free(&hbuf);
  if (gai != 0) {
    return -1;
  }

  vira_socket_t udp_fd = VIRA_INVALID_SOCKET;
  for (p = res; p != NULL; p = p->ai_next) {
    udp_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (udp_fd == VIRA_INVALID_SOCKET)
      continue;

    int opt = 1;
#ifdef _WIN32
    setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt,
               sizeof(opt));
#else
    setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
    setsockopt(udp_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
#endif

    if (bind(udp_fd, p->ai_addr, (socklen_t)p->ai_addrlen) == 0) {
      break;
    }

    closesocket(udp_fd);
    udp_fd = VIRA_INVALID_SOCKET;
  }

  freeaddrinfo(res);
  if (udp_fd == VIRA_INVALID_SOCKET)
    return -1;
  return (int64_t)udp_fd;
}

static inline int32_t vira_net_udp_send_to(int64_t fd, vira_slice_u8 buf,
                                           int32_t len, vira_string host,
                                           int32_t port) {
  if (fd < 0 || buf.data == NULL || len <= 0 || port < 0 || port > 65535) {
    return 0;
  }
  int32_t to_write = buf.len < len ? buf.len : len;
  if (to_write <= 0)
    return 0;

  char stack_host[256];
  vira_cstr_buf hbuf = vira_str_to_cstr_buf(host, stack_host, sizeof(stack_host));

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;

  int gai = getaddrinfo(hbuf.ptr, port_buf, &hints, &res);
  vira_cstr_buf_free(&hbuf);
  if (gai != 0 || !res) {
    return -1;
  }

  int sent = sendto((vira_socket_t)fd, (const char *)buf.data, (int)to_write, 0,
                    res->ai_addr, (socklen_t)res->ai_addrlen);
  freeaddrinfo(res);
  return (int32_t)sent;
}

static inline int32_t vira_net_udp_recv_from(int64_t fd, vira_slice_u8 *buf,
                                             int32_t max_len) {
  if (fd < 0 || buf == NULL || buf->data == NULL || max_len <= 0) {
    return 0;
  }
  int32_t can_read = buf->cap < max_len ? buf->cap : max_len;
  if (can_read <= 0)
    return 0;

  _vira_last_udp_sender_len = sizeof(_vira_last_udp_sender);
  memset(&_vira_last_udp_sender, 0, sizeof(_vira_last_udp_sender));

  int n = recvfrom((vira_socket_t)fd, (char *)buf->data, (int)can_read, 0,
                   (struct sockaddr *)&_vira_last_udp_sender,
                   &_vira_last_udp_sender_len);
  if (n >= 0) {
    buf->len = (int32_t)n;
    return (int32_t)n;
  }
  return -1;
}

static inline vira_string vira_net_udp_last_sender(void) {
  if (_vira_last_udp_sender_len == 0) {
    return vira_str_lit("", 0);
  }
  if (_vira_last_udp_sender.ss_family == AF_INET) {
    struct sockaddr_in *s = (struct sockaddr_in *)&_vira_last_udp_sender;
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &s->sin_addr, ip, sizeof(ip));
    int port = ntohs(s->sin_port);
    return vira_str_format("%s:%d", ip, port);
  } else if (_vira_last_udp_sender.ss_family == AF_INET6) {
    struct sockaddr_in6 *s = (struct sockaddr_in6 *)&_vira_last_udp_sender;
    char ip[INET6_ADDRSTRLEN];
    inet_ntop(AF_INET6, &s->sin6_addr, ip, sizeof(ip));
    int port = ntohs(s->sin6_port);
    return vira_str_format("[%s]:%d", ip, port);
  }
  return vira_str_lit("", 0);
}

static inline vira_string vira_http_response(int64_t status_code,
                                             vira_string content_type,
                                             vira_string body) {
  const char *status_text = "OK";
  if (status_code == 201)
    status_text = "Created";
  else if (status_code == 400)
    status_text = "Bad Request";
  else if (status_code == 404)
    status_text = "Not Found";
  else if (status_code == 500)
    status_text = "Internal Server Error";

  char stack_ctype[128];
  vira_cstr_buf cbuf =
      vira_str_to_cstr_buf(content_type, stack_ctype, sizeof(stack_ctype));

  int32_t body_len = body.len > 0 ? body.len : 0;
  size_t ctype_len = content_type.len > 0 ? (size_t)content_type.len : 0;
  size_t hdr_cap = 256 + ctype_len;
  size_t alloc_sz = hdr_cap + (size_t)body_len + 1;
  char *res = (char *)malloc(alloc_sz);
  if (!res) {
    vira_cstr_buf_free(&cbuf);
    return vira_str_lit("", 0);
  }
  _vira_record_alloc();

  int n = snprintf(res, hdr_cap,
                   "HTTP/1.1 %lld %s\r\n"
                   "Content-Type: %s\r\n"
                   "Content-Length: %d\r\n"
                   "Connection: close\r\n"
                   "\r\n",
                   (long long)status_code, status_text, cbuf.ptr, body_len);
  vira_cstr_buf_free(&cbuf);
  if (n < 0)
    n = 0;
  if ((size_t)n >= hdr_cap)
    n = (int)(hdr_cap - 1);

  if (body_len > 0 && body.data) {
    memcpy(res + n, body.data, (size_t)body_len);
    n += body_len;
  }
  res[n] = '\0';
  return (vira_string){.data = (const uint8_t *)res, .len = n, .is_lit = false};
}

// ============================================================================
// 13. DEPTH- & ESCAPE-AWARE JSON FIELD HELPERS (SEE ALSO `import json`)
// ============================================================================

static inline size_t _vira_json_skip_string(const char *s, size_t len,
                                            size_t open_quote_idx) {
  size_t i = open_quote_idx + 1;
  while (i < len) {
    if (s[i] == '\\') {
      i += 2;
      continue;
    }
    if (s[i] == '"') {
      return i + 1;
    }
    i++;
  }
  return len;
}

static inline bool _vira_json_key_matches(const char *s, size_t len,
                                          size_t open_quote_idx,
                                          vira_string key) {
  size_t i = open_quote_idx + 1;
  int32_t ki = 0;
  while (i < len && s[i] != '"') {
    char c = s[i];
    if (c == '\\') {
      i++;
      if (i >= len)
        return false;
      switch (s[i]) {
      case '"':
        c = '"';
        break;
      case '\\':
        c = '\\';
        break;
      case '/':
        c = '/';
        break;
      case 'n':
        c = '\n';
        break;
      case 'r':
        c = '\r';
        break;
      case 't':
        c = '\t';
        break;
      default:
        c = s[i];
        break;
      }
    }
    if (ki >= key.len || (uint8_t)c != key.data[ki]) {
      return false;
    }
    ki++;
    i++;
  }
  return (i < len && s[i] == '"' && ki == key.len);
}

// Finds the value start offset for `key`. Prefers top-level object (`target_depth == 1`);
// falls back to nested object match when `target_depth == 0`.
static inline const char *_vira_json_find_val_at_depth(vira_string json,
                                                       vira_string key,
                                                       int32_t target_depth) {
  if (json.data == NULL || json.len <= 0 || key.data == NULL || key.len <= 0) {
    return NULL;
  }
  const char *s = (const char *)json.data;
  size_t len = (size_t)json.len;
  int32_t obj_depth = 0;
  int32_t arr_depth = 0;
  size_t i = 0;

  while (i < len) {
    char c = s[i];
    if (c == '{') {
      obj_depth++;
      i++;
    } else if (c == '}') {
      if (obj_depth > 0)
        obj_depth--;
      i++;
    } else if (c == '[') {
      arr_depth++;
      i++;
    } else if (c == ']') {
      if (arr_depth > 0)
        arr_depth--;
      i++;
    } else if (c == '"') {
      size_t after_str = _vira_json_skip_string(s, len, i);
      size_t j = after_str;
      while (j < len &&
             (s[j] == ' ' || s[j] == '\t' || s[j] == '\r' || s[j] == '\n')) {
        j++;
      }
      bool is_obj_key = (j < len && s[j] == ':' && obj_depth >= 1);
      bool depth_ok = (target_depth == 0) ||
                      (obj_depth == target_depth && arr_depth == 0);
      if (is_obj_key && depth_ok && _vira_json_key_matches(s, len, i, key)) {
        j++; // skip ':'
        while (j < len &&
               (s[j] == ' ' || s[j] == '\t' || s[j] == '\r' || s[j] == '\n')) {
          j++;
        }
        return (j < len) ? (s + j) : NULL;
      }
      i = after_str;
    } else {
      i++;
    }
  }
  return NULL;
}

static inline const char *_vira_json_find_val(vira_string json,
                                              vira_string key) {
  const char *v = _vira_json_find_val_at_depth(json, key, 1);
  if (v != NULL)
    return v;
  return _vira_json_find_val_at_depth(json, key, 0);
}

static inline vira_string vira_json_get_str(vira_string json, vira_string key) {
  const char *val = _vira_json_find_val(json, key);
  if (!val || *val != '"')
    return vira_str_lit("", 0);
  const char *end = (const char *)json.data + json.len;
  const char *p = val + 1;

  size_t max_len = (size_t)(end - p);
  char *out = (char *)malloc(max_len + 1);
  if (!out)
    return vira_str_lit("", 0);

  int32_t out_len = 0;
  bool closed = false;
  while (p < end) {
    char c = *p++;
    if (c == '"') {
      closed = true;
      break;
    }
    if (c == '\\' && p < end) {
      char esc = *p++;
      switch (esc) {
      case '"':
        out[out_len++] = '"';
        break;
      case '\\':
        out[out_len++] = '\\';
        break;
      case '/':
        out[out_len++] = '/';
        break;
      case 'b':
        out[out_len++] = '\b';
        break;
      case 'f':
        out[out_len++] = '\f';
        break;
      case 'n':
        out[out_len++] = '\n';
        break;
      case 'r':
        out[out_len++] = '\r';
        break;
      case 't':
        out[out_len++] = '\t';
        break;
      default:
        out[out_len++] = esc;
        break;
      }
    } else {
      out[out_len++] = c;
    }
  }
  if (!closed) {
    free(out);
    return vira_str_lit("", 0);
  }
  out[out_len] = '\0';
  _vira_record_alloc();
  return (vira_string){
      .data = (const uint8_t *)out, .len = out_len, .is_lit = false};
}

static inline int64_t vira_json_get_int(vira_string json, vira_string key) {
  const char *val = _vira_json_find_val(json, key);
  if (!val)
    return 0;
  const char *end = (const char *)json.data + json.len;
  char num_buf[64];
  int nlen = 0;
  while (val + nlen < end && nlen < 63) {
    char c = val[nlen];
    if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
      num_buf[nlen++] = c;
    } else {
      break;
    }
  }
  if (nlen == 0)
    return 0;
  num_buf[nlen] = '\0';
  return (int64_t)strtoll(num_buf, NULL, 10);
}

static inline float vira_json_get_f32(vira_string json, vira_string key) {
  const char *val = _vira_json_find_val(json, key);
  if (!val)
    return 0.0f;
  const char *end = (const char *)json.data + json.len;
  char num_buf[64];
  int nlen = 0;
  while (val + nlen < end && nlen < 63) {
    char c = val[nlen];
    if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' ||
        c == 'e' || c == 'E') {
      num_buf[nlen++] = c;
    } else {
      break;
    }
  }
  if (nlen == 0)
    return 0.0f;
  num_buf[nlen] = '\0';
  return (float)strtod(num_buf, NULL);
}

// ============================================================================
// 14. NATIVE SAFETENSORS ZERO-COPY PARSER & LOADER
// ============================================================================

static inline Result_tensor_f32 vira_safetensors_load(vira_string path,
                                                      vira_string tensor_name) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  char stack_name[256];
  vira_cstr_buf nbuf =
      vira_str_to_cstr_buf(tensor_name, stack_name, sizeof(stack_name));

  FILE *f = fopen(pbuf.ptr, "rb");
  if (!f) {
    vira_string err =
        vira_str_format("failed to open safetensors file '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  uint64_t header_len = 0;
  if (fread(&header_len, sizeof(uint64_t), 1, f) != 1 ||
      header_len > 100 * 1024 * 1024) {
    fclose(f);
    vira_string err =
        vira_str_format("corrupt safetensors header in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  char *header_json = (char *)malloc(header_len + 1);
  if (!header_json) {
    fclose(f);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true,
        .err =
            vira_str_format("failed to allocate memory for header (%llu bytes)",
                            (unsigned long long)header_len),
        .val = (vira_tensor_f32){0}};
  }

  if (fread(header_json, 1, header_len, f) != header_len) {
    free(header_json);
    fclose(f);
    vira_string err = vira_str_format(
        "incomplete read of safetensors header in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }
  header_json[header_len] = '\0';

  vira_string hstr = {.data = (const uint8_t *)header_json,
                      .len = (int32_t)header_len,
                      .is_lit = true};
  const char *t_entry = _vira_json_find_val_at_depth(hstr, tensor_name, 1);
  if (!t_entry) {
    free(header_json);
    fclose(f);
    vira_string err =
        vira_str_format("tensor '%s' not found in '%s'", nbuf.ptr, pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  int32_t parsed_dims[4] = {1, 1, 1, 1};
  int32_t parsed_ndims = 0;
  const char *shape_ptr = strstr(t_entry, "\"shape\"");
  if (shape_ptr) {
    const char *cur = strchr(shape_ptr, '[');
    if (cur) {
      cur++;
      while (*cur != '\0' && *cur != ']' && parsed_ndims < 4) {
        while (*cur == ' ' || *cur == ',' || *cur == '\t' || *cur == '\n' ||
               *cur == '\r') {
          cur++;
        }
        if (*cur == ']' || *cur == '\0')
          break;
        char *endp = NULL;
        long dval = strtol(cur, &endp, 10);
        if (endp == cur)
          break;
        parsed_dims[parsed_ndims++] = (dval > 0) ? (int32_t)dval : 1;
        cur = endp;
      }
    }
  }
  if (parsed_ndims == 0) {
    parsed_ndims = 1;
    parsed_dims[0] = 1;
  }

  uint64_t begin_off = 0;
  const char *off_ptr = strstr(t_entry, "\"data_offsets\"");
  if (off_ptr) {
    const char *lbracket = strchr(off_ptr, '[');
    if (lbracket) {
      begin_off = (uint64_t)strtoull(lbracket + 1, NULL, 10);
    }
  }
  free(header_json);

  uint64_t file_offset = 8 + header_len + begin_off;
#ifdef _WIN32
  _fseeki64(f, file_offset, SEEK_SET);
#else
  fseeko(f, file_offset, SEEK_SET);
#endif

  vira_tensor_f32 t = vira_tensor_f32_alloc_nd(parsed_ndims, parsed_dims);
  size_t expected_bytes = (size_t)t.rows * (size_t)t.cols * sizeof(float);
  size_t read_bytes = fread(t.data, 1, expected_bytes, f);
  fclose(f);

  if (read_bytes != expected_bytes) {
    vira_tensor_f32_free(&t);
    vira_string err = vira_str_format(
        "tensor '%s' data truncated: expected %zu bytes, read %zu", nbuf.ptr,
        expected_bytes, read_bytes);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  vira_cstr_buf_free(&nbuf);
  vira_cstr_buf_free(&pbuf);
  return (Result_tensor_f32){
      .is_err = false, .err = vira_str_lit("", 0), .val = t};
}

// ============================================================================
// 15. HARDWARE SIMD AUTOGRAD BACKWARD OPS & OPTIMIZERS
// ============================================================================

static inline vira_tensor_f32 vira_tensor_transpose(vira_tensor_f32 a) {
  return vira_tensor_transpose_nd(NULL, a);
}

static inline vira_tensor_f32
vira_tensor_matmul_grad_a(vira_tensor_f32 grad_out, vira_tensor_f32 b) {
  vira_tensor_f32 bt = vira_tensor_transpose(b);
  vira_tensor_f32 grad_a = vira_tensor_matmul_f32(NULL, grad_out, bt);
  vira_tensor_f32_free(&bt);
  return grad_a;
}

static inline vira_tensor_f32
vira_tensor_matmul_grad_b(vira_tensor_f32 a, vira_tensor_f32 grad_out) {
  vira_tensor_f32 at = vira_tensor_transpose(a);
  vira_tensor_f32 grad_b = vira_tensor_matmul_f32(NULL, at, grad_out);
  vira_tensor_f32_free(&at);
  return grad_b;
}

static inline vira_tensor_f32
vira_tensor_relu_backward(vira_tensor_f32 input, vira_tensor_f32 grad_out) {
  _vira_tensor_norm_shape(&input);
  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(input.n_dims, input.dims);
  int total = input.rows * input.cols;
  int i = 0;

#if defined(__AVX2__)
  __m256 zero = _mm256_setzero_ps();
  for (; i + 7 < total; i += 8) {
    __m256 in_v = _mm256_loadu_ps(&input.data[i]);
    __m256 grad_v = _mm256_loadu_ps(&grad_out.data[i]);
    __m256 mask = _mm256_cmp_ps(in_v, zero, _CMP_GT_OQ);
    __m256 res_v = _mm256_and_ps(mask, grad_v);
    _mm256_storeu_ps(&out.data[i], res_v);
  }
#endif

  for (; i < total; ++i) {
    out.data[i] = (input.data[i] > 0.0f) ? grad_out.data[i] : 0.0f;
  }
  return out;
}

static inline void vira_tensor_sgd_step(vira_tensor_f32 *param,
                                        vira_tensor_f32 grad, float lr) {
  if (!param || !param->data || !grad.data)
    return;
  int total = param->rows * param->cols;
  int i = 0;

#if defined(__AVX2__)
  __m256 lr_v = _mm256_set1_ps(lr);
  for (; i + 7 < total; i += 8) {
    __m256 p_v = _mm256_loadu_ps(&param->data[i]);
    __m256 g_v = _mm256_loadu_ps(&grad.data[i]);
    __m256 step = _mm256_mul_ps(lr_v, g_v);
    __m256 updated = _mm256_sub_ps(p_v, step);
    _mm256_storeu_ps(&param->data[i], updated);
  }
#endif

  for (; i < total; ++i) {
    param->data[i] -= lr * grad.data[i];
  }
}

// ============================================================================
// 16. NATIVE GGUF V2/V3 ZERO-COPY BINARY PARSER (Q4_0, Q8_0, F16, F32)
// ============================================================================

static inline float _vira_fp16_to_fp32(uint16_t h) {
#if defined(__F16C__)
  return _cvtsh_ss(h);
#else
  uint32_t sign = ((uint32_t)h & 0x8000) << 16;
  uint32_t exp = ((uint32_t)h & 0x7C00) >> 10;
  uint32_t mant = (uint32_t)h & 0x03FF;
  if (exp == 0) {
    if (mant == 0) {
      float res;
      uint32_t bits = sign;
      memcpy(&res, &bits, sizeof(float));
      return res;
    }
    while ((mant & 0x0400) == 0) {
      mant <<= 1;
      exp--;
    }
    exp++;
    mant &= ~0x0400;
  } else if (exp == 31) {
    exp = 255;
  } else {
    exp = exp - 15 + 127;
  }
  uint32_t bits = sign | (exp << 23) | (mant << 13);
  float res;
  memcpy(&res, &bits, sizeof(float));
  return res;
#endif
}

static inline void _vira_gguf_skip_val(FILE *f, uint32_t val_type) {
  switch (val_type) {
  case 0:
  case 1:
  case 7:
    fseek(f, 1, SEEK_CUR);
    break;
  case 2:
  case 3:
    fseek(f, 2, SEEK_CUR);
    break;
  case 4:
  case 5:
  case 6:
    fseek(f, 4, SEEK_CUR);
    break;
  case 10:
  case 11:
  case 12:
    fseek(f, 8, SEEK_CUR);
    break;
  case 8: {
    uint64_t slen = 0;
    if (fread(&slen, sizeof(uint64_t), 1, f) == 1) {
      fseek(f, (long)slen, SEEK_CUR);
    }
    break;
  }
  case 9: {
    uint32_t elem_type = 0;
    uint64_t elem_count = 0;
    if (fread(&elem_type, sizeof(uint32_t), 1, f) == 1 &&
        fread(&elem_count, sizeof(uint64_t), 1, f) == 1) {
      for (uint64_t i = 0; i < elem_count; ++i) {
        _vira_gguf_skip_val(f, elem_type);
      }
    }
    break;
  }
  }
}

static inline Result_tensor_f32 vira_gguf_load_tensor(vira_string path,
                                                      vira_string tensor_name) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  char stack_name[256];
  vira_cstr_buf nbuf =
      vira_str_to_cstr_buf(tensor_name, stack_name, sizeof(stack_name));

  FILE *f = fopen(pbuf.ptr, "rb");
  if (!f) {
    vira_string err =
        vira_str_format("failed to open GGUF file '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t tensor_count = 0;
  uint64_t kv_count = 0;

  if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) {
    fclose(f);
    vira_string err = vira_str_format("invalid GGUF magic in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }
  if (fread(&version, 4, 1, f) != 1 || (version != 2 && version != 3)) {
    fclose(f);
    vira_string err = vira_str_format("unsupported GGUF version %u in '%s'",
                                      version, pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }
  if (fread(&tensor_count, 8, 1, f) != 1 || fread(&kv_count, 8, 1, f) != 1) {
    fclose(f);
    vira_string err = vira_str_format("corrupt GGUF header in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  uint32_t alignment = 32;

  for (uint64_t k = 0; k < kv_count; ++k) {
    uint64_t klen = 0;
    if (fread(&klen, 8, 1, f) != 1)
      break;
    char key_buf[256];
    size_t to_read = klen < 255 ? klen : 255;
    if (fread(key_buf, 1, to_read, f) != to_read)
      break;
    key_buf[to_read] = '\0';
    if (klen > to_read)
      fseek(f, (long)(klen - to_read), SEEK_CUR);

    uint32_t val_type = 0;
    if (fread(&val_type, 4, 1, f) != 1)
      break;

    if (strcmp(key_buf, "general.alignment") == 0 && val_type == 4) {
      uint32_t a = 32;
      if (fread(&a, 4, 1, f) == 1 && a > 0)
        alignment = a;
    } else {
      _vira_gguf_skip_val(f, val_type);
    }
  }

  bool found = false;
  uint32_t matched_type = 0;
  uint64_t matched_offset = 0;
  int32_t matched_ndims = 2;
  int32_t matched_dims[4] = {1, 1, 1, 1};

  for (uint64_t t = 0; t < tensor_count; ++t) {
    uint64_t n_len = 0;
    if (fread(&n_len, 8, 1, f) != 1)
      break;
    char cur_name[256];
    size_t to_read = n_len < 255 ? n_len : 255;
    if (fread(cur_name, 1, to_read, f) != to_read)
      break;
    cur_name[to_read] = '\0';
    if (n_len > to_read)
      fseek(f, (long)(n_len - to_read), SEEK_CUR);

    uint32_t n_dims = 0;
    if (fread(&n_dims, 4, 1, f) != 1)
      break;

    uint64_t dims[4] = {1, 1, 1, 1};
    for (uint32_t d = 0; d < n_dims && d < 4; ++d) {
      if (fread(&dims[d], 8, 1, f) != 1)
        break;
    }
    if (n_dims > 4) {
      fseek(f, (long)(8 * (n_dims - 4)), SEEK_CUR);
    }

    uint32_t dtype = 0;
    uint64_t offset = 0;
    if (fread(&dtype, 4, 1, f) != 1)
      break;
    if (fread(&offset, 8, 1, f) != 1)
      break;

    if (n_len == (uint64_t)tensor_name.len &&
        strcmp(cur_name, nbuf.ptr) == 0) {
      found = true;
      matched_type = dtype;
      matched_offset = offset;
      uint32_t clamped_nd = (n_dims == 0) ? 1 : ((n_dims > 4) ? 4 : n_dims);
      matched_ndims = (int32_t)clamped_nd;
      // GGUF stores dimensions inner-to-outer (dims[0] = trailing cols)
      for (uint32_t d = 0; d < clamped_nd; ++d) {
        int32_t dv = (int32_t)dims[clamped_nd - 1 - d];
        matched_dims[d] = (dv > 0) ? dv : 1;
      }
    }
  }

  if (!found) {
    fclose(f);
    vira_string err = vira_str_format(
        "tensor '%s' not found in GGUF file '%s'", nbuf.ptr, pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  vira_cstr_buf_free(&nbuf);
  vira_cstr_buf_free(&pbuf);

  int64_t cur_pos = ftell(f);
  int64_t pad = (alignment - (cur_pos % alignment)) % alignment;
  int64_t data_start = cur_pos + pad;

  int64_t tensor_file_pos = data_start + matched_offset;
#ifdef _WIN32
  _fseeki64(f, tensor_file_pos, SEEK_SET);
#else
  fseeko(f, tensor_file_pos, SEEK_SET);
#endif

  vira_tensor_f32 t = vira_tensor_f32_alloc_nd(matched_ndims, matched_dims);
  int32_t total = t.rows * t.cols;

  if (matched_type == 0) {
    fread(t.data, sizeof(float), total, f);
  } else if (matched_type == 1) {
    uint16_t *hbuf = (uint16_t *)malloc(total * sizeof(uint16_t));
    if (hbuf) {
      fread(hbuf, sizeof(uint16_t), total, f);
      for (int32_t i = 0; i < total; ++i) {
        t.data[i] = _vira_fp16_to_fp32(hbuf[i]);
      }
      free(hbuf);
    }
  } else if (matched_type == 7) {
    int32_t blocks = total / 32;
    for (int32_t b = 0; b < blocks; ++b) {
      uint16_t scale_h = 0;
      int8_t qs[32];
      fread(&scale_h, 2, 1, f);
      fread(qs, 1, 32, f);
      float scale = _vira_fp16_to_fp32(scale_h);
      int32_t base = b * 32;
      for (int32_t j = 0; j < 32; ++j) {
        t.data[base + j] = scale * (float)qs[j];
      }
    }
  } else if (matched_type == 2) {
    int32_t blocks = total / 32;
    for (int32_t b = 0; b < blocks; ++b) {
      uint16_t scale_h = 0;
      uint8_t nibbles[16];
      fread(&scale_h, 2, 1, f);
      fread(nibbles, 1, 16, f);
      float scale = _vira_fp16_to_fp32(scale_h);
      int32_t base = b * 32;
      for (int32_t j = 0; j < 16; ++j) {
        uint8_t byte = nibbles[j];
        int8_t v0 = (int8_t)(byte & 0x0F) - 8;
        int8_t v1 = (int8_t)(byte >> 4) - 8;
        t.data[base + j] = scale * (float)v0;
        t.data[base + j + 16] = scale * (float)v1;
      }
    }
  } else {
    fread(t.data, sizeof(float), total, f);
  }

  fclose(f);
  return (Result_tensor_f32){
      .is_err = false, .err = vira_str_lit("", 0), .val = t};
}

// ============================================================================
// 17. LLM TRANSFORMER OPERATORS (RMSNORM, ROPE, SILU, EMBEDDING, SAMPLING)
// ============================================================================

static inline vira_tensor_f32
vira_tensor_rmsnorm(vira_tensor_f32 x, vira_tensor_f32 weight, float eps) {
  _vira_tensor_norm_shape(&x);
  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(x.n_dims, x.dims);
  for (int32_t r = 0; r < x.rows; ++r) {
    float *x_row = &x.data[r * x.cols];
    float *out_row = &out.data[r * x.cols];
    float sum_sq = 0.0f;

#if defined(__AVX2__)
    __m256 v_sum_sq = _mm256_setzero_ps();
    int32_t c = 0;
    for (; c + 7 < x.cols; c += 8) {
      __m256 vx = _mm256_loadu_ps(&x_row[c]);
      v_sum_sq = _mm256_add_ps(v_sum_sq, _mm256_mul_ps(vx, vx));
    }
    __m128 lo = _mm256_castps256_ps128(v_sum_sq);
    __m128 hi = _mm256_extractf128_ps(v_sum_sq, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    sum_sq += _mm_cvtss_f32(s);
    for (; c < x.cols; ++c) {
      sum_sq += x_row[c] * x_row[c];
    }
#else
    for (int32_t c = 0; c < x.cols; ++c) {
      sum_sq += x_row[c] * x_row[c];
    }
#endif
    float mean_sq = sum_sq / (float)x.cols;
    float inv_rms = 1.0f / sqrtf(mean_sq + eps);

#if defined(__AVX2__)
    __m256 v_inv = _mm256_set1_ps(inv_rms);
    c = 0;
    for (; c + 7 < x.cols; c += 8) {
      __m256 vx = _mm256_loadu_ps(&x_row[c]);
      __m256 vw = _mm256_loadu_ps(&weight.data[c]);
      __m256 res = _mm256_mul_ps(_mm256_mul_ps(vx, v_inv), vw);
      _mm256_storeu_ps(&out_row[c], res);
    }
    for (; c < x.cols; ++c) {
      out_row[c] = x_row[c] * inv_rms * weight.data[c];
    }
#else
    for (int32_t c = 0; c < x.cols; ++c) {
      out_row[c] = x_row[c] * inv_rms * weight.data[c];
    }
#endif
  }
  return out;
}

static inline vira_tensor_f32 vira_tensor_rope(vira_tensor_f32 q, int64_t pos,
                                               float freq_base) {
  _vira_tensor_norm_shape(&q);
  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(q.n_dims, q.dims);
  int32_t half_dim = q.cols / 2;

  for (int32_t r = 0; r < q.rows; ++r) {
    float *q_row = &q.data[r * q.cols];
    float *out_row = &out.data[r * q.cols];

    for (int32_t i = 0; i < half_dim; ++i) {
      float exponent = -(2.0f * (float)i) / (float)q.cols;
      float freq = powf(freq_base, exponent);
      float angle = (float)pos * freq;
      float cos_a = cosf(angle);
      float sin_a = sinf(angle);

      float q0 = q_row[i];
      float q1 = q_row[i + half_dim];

      out_row[i] = q0 * cos_a - q1 * sin_a;
      out_row[i + half_dim] = q0 * sin_a + q1 * cos_a;
    }
  }
  return out;
}

static inline vira_tensor_f32 vira_tensor_silu(vira_tensor_f32 x) {
  _vira_tensor_norm_shape(&x);
  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(x.n_dims, x.dims);
  int32_t total = x.rows * x.cols;
  for (int32_t i = 0; i < total; ++i) {
    float v = x.data[i];
    out.data[i] = v / (1.0f + expf(-v));
  }
  return out;
}

static inline void _vira_vec_mul_f32(vira_f32 *dst, const vira_f32 *pa,
                                     const vira_f32 *pb, int32_t count) {
  int32_t i = 0;
#if defined(__AVX2__)
  for (; i + 8 <= count; i += 8) {
    __m256 va = _mm256_loadu_ps(&pa[i]);
    __m256 vb = _mm256_loadu_ps(&pb[i]);
    _mm256_storeu_ps(&dst[i], _mm256_mul_ps(va, vb));
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  for (; i + 4 <= count; i += 4) {
    float32x4_t va = vld1q_f32(&pa[i]);
    float32x4_t vb = vld1q_f32(&pb[i]);
    vst1q_f32(&dst[i], vmulq_f32(va, vb));
  }
#elif defined(__SSE2__)
  for (; i + 4 <= count; i += 4) {
    __m128 va = _mm_loadu_ps(&pa[i]);
    __m128 vb = _mm_loadu_ps(&pb[i]);
    _mm_storeu_ps(&dst[i], _mm_mul_ps(va, vb));
  }
#endif
  for (; i < count; i++) {
    dst[i] = pa[i] * pb[i];
  }
}

static inline vira_tensor_f32 vira_tensor_elemwise_mul(vira_tensor_f32 a,
                                                       vira_tensor_f32 b) {
  _vira_tensor_norm_shape(&a);
  _vira_tensor_norm_shape(&b);
  if (a.rows == b.rows && a.cols == b.cols) {
    const vira_tensor_f32 *sh = (a.n_dims >= b.n_dims) ? &a : &b;
    vira_tensor_f32 out = vira_tensor_f32_alloc_nd(sh->n_dims, sh->dims);
    _vira_vec_mul_f32(out.data, a.data, b.data, a.rows * a.cols);
    return out;
  }
  if (b.rows == 1 && b.cols == a.cols) {
    vira_tensor_f32 out = vira_tensor_f32_alloc_nd(a.n_dims, a.dims);
    for (int32_t r = 0; r < a.rows; r++) {
      _vira_vec_mul_f32(&out.data[(size_t)r * (size_t)a.cols],
                        &a.data[(size_t)r * (size_t)a.cols], b.data, a.cols);
    }
    return out;
  }
  if (a.rows == 1 && a.cols == b.cols) {
    vira_tensor_f32 out = vira_tensor_f32_alloc_nd(b.n_dims, b.dims);
    for (int32_t r = 0; r < b.rows; r++) {
      _vira_vec_mul_f32(&out.data[(size_t)r * (size_t)b.cols], a.data,
                        &b.data[(size_t)r * (size_t)b.cols], b.cols);
    }
    return out;
  }
  fprintf(stderr,
          "vira tensor error: elemwise_mul shape mismatch (%dx%d) * (%dx%d)\n",
          a.rows, a.cols, b.rows, b.cols);
  exit(1);
}

static inline vira_tensor_f32 vira_tensor_softmax(vira_tensor_f32 logits) {
  _vira_tensor_norm_shape(&logits);
  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(logits.n_dims, logits.dims);
  for (int32_t r = 0; r < logits.rows; ++r) {
    float *in_row = &logits.data[r * logits.cols];
    float *out_row = &out.data[r * logits.cols];

    float max_val = in_row[0];
    for (int32_t c = 1; c < logits.cols; ++c) {
      if (in_row[c] > max_val)
        max_val = in_row[c];
    }

    float sum = 0.0f;
    for (int32_t c = 0; c < logits.cols; ++c) {
      out_row[c] = expf(in_row[c] - max_val);
      sum += out_row[c];
    }

    float inv_sum = sum > 0.0f ? (1.0f / sum) : 1.0f;
    for (int32_t c = 0; c < logits.cols; ++c) {
      out_row[c] *= inv_sum;
    }
  }
  return out;
}

static inline int64_t vira_sample_token(vira_tensor_f32 logits,
                                        float temperature) {
  int32_t total = logits.rows * logits.cols;
  if (total == 0)
    return 0;

  if (temperature <= 0.05f) {
    int64_t best_idx = 0;
    float best_val = logits.data[0];
    for (int32_t i = 1; i < total; ++i) {
      if (logits.data[i] > best_val) {
        best_val = logits.data[i];
        best_idx = (int64_t)i;
      }
    }
    return best_idx;
  }

  vira_tensor_f32 scaled = vira_tensor_f32_alloc(logits.rows, logits.cols);
  for (int32_t i = 0; i < total; ++i) {
    scaled.data[i] = logits.data[i] / temperature;
  }
  vira_tensor_f32 probs = vira_tensor_softmax(scaled);
  vira_tensor_f32_free(&scaled);

  float r = (float)rand() / (float)RAND_MAX;
  float cdf = 0.0f;
  int64_t chosen = total - 1;
  for (int32_t i = 0; i < total; ++i) {
    cdf += probs.data[i];
    if (r <= cdf) {
      chosen = (int64_t)i;
      break;
    }
  }
  vira_tensor_f32_free(&probs);
  return chosen;
}

static inline vira_tensor_f32 vira_tensor_embedding(vira_tensor_f32 table,
                                                    int64_t token_id) {
  if (token_id < 0 || token_id >= table.rows) {
    fprintf(stderr,
            "vira tensor error: embedding index %lld out of range [0, %d)\n",
            (long long)token_id, table.rows);
    exit(1);
  }
  vira_tensor_f32 out = vira_tensor_f32_alloc(1, table.cols);
  memcpy(out.data, &table.data[token_id * table.cols],
         (size_t)table.cols * sizeof(float));
  return out;
}

// ============================================================================
// 8. C STANDARD LIBRARY SHIMS (STDIO, CTYPE, MATH, TIME)
// ============================================================================

static inline Result_string vira_io_read_line(void) {
  size_t cap = 128;
  size_t len = 0;
  char *buf = (char *)malloc(cap);
  if (!buf) {
    return (Result_string){.is_err = true, .err = vira_str_lit("out of memory reading stdin", 27)};
  }
  int c;
  while ((c = fgetc(stdin)) != EOF) {
    if (c == '\r') continue;
    if (c == '\n') break;
    if (len + 1 >= cap) {
      cap *= 2;
      char *nb = (char *)realloc(buf, cap);
      if (!nb) {
        free(buf);
        return (Result_string){.is_err = true, .err = vira_str_lit("out of memory reading line", 26)};
      }
      buf = nb;
    }
    buf[len++] = (char)c;
  }
  if (len == 0 && c == EOF) {
    free(buf);
    return (Result_string){.is_err = true, .err = vira_str_lit("EOF", 3)};
  }
  buf[len] = '\0';
  _vira_record_alloc();
  return (Result_string){.is_err = false, .val = (vira_string){.data = (const uint8_t *)buf, .len = (int32_t)len, .is_lit = false}};
}

static inline Result_string vira_io_read_all_stdin(void) {
  size_t cap = 1024;
  size_t len = 0;
  char *buf = (char *)malloc(cap);
  if (!buf) {
    return (Result_string){.is_err = true, .err = vira_str_lit("out of memory", 13)};
  }
  char chunk[4096];
  size_t n;
  while ((n = fread(chunk, 1, sizeof(chunk), stdin)) > 0) {
    if (len + n + 1 > cap) {
      while (len + n + 1 > cap) cap *= 2;
      char *nb = (char *)realloc(buf, cap);
      if (!nb) {
        free(buf);
        return (Result_string){.is_err = true, .err = vira_str_lit("out of memory", 13)};
      }
      buf = nb;
    }
    memcpy(buf + len, chunk, n);
    len += n;
  }
  buf[len] = '\0';
  _vira_record_alloc();
  return (Result_string){.is_err = false, .val = (vira_string){.data = (const uint8_t *)buf, .len = (int32_t)len, .is_lit = false}};
}

static inline Result_bool vira_io_rename(vira_string old_p, vira_string new_p) {
  char s_old[1024], s_new[1024];
  vira_cstr_buf b_old = vira_str_to_cstr_buf(old_p, s_old, sizeof(s_old));
  vira_cstr_buf b_new = vira_str_to_cstr_buf(new_p, s_new, sizeof(s_new));
  int res = rename(b_old.ptr, b_new.ptr);
  vira_cstr_buf_free(&b_old);
  vira_cstr_buf_free(&b_new);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("rename error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_bool vira_io_remove(vira_string path) {
  char s_p[1024];
  vira_cstr_buf b = vira_str_to_cstr_buf(path, s_p, sizeof(s_p));
  int res = remove(b.ptr);
  vira_cstr_buf_free(&b);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("remove error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_string vira_io_temp_file(void) {
#ifdef _WIN32
  char tmp_dir[MAX_PATH];
  GetTempPathA(MAX_PATH, tmp_dir);
  char tmp_file[MAX_PATH];
  if (GetTempFileNameA(tmp_dir, "vir", 0, tmp_file) == 0) {
    return (Result_string){.is_err = true, .err = vira_str_lit("failed to create temp file", 26)};
  }
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(tmp_file)};
#else
  char pattern[] = "/tmp/vira_tmp_XXXXXX";
  int fd = mkstemp(pattern);
  if (fd < 0) {
    return (Result_string){.is_err = true, .err = vira_str_format("mkstemp error: %s", strerror(errno))};
  }
  close(fd);
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(pattern)};
#endif
}

static inline bool vira_ctype_is_alpha(int32_t c) { return isalpha((unsigned char)c) != 0; }
static inline bool vira_ctype_is_digit(int32_t c) { return isdigit((unsigned char)c) != 0; }
static inline bool vira_ctype_is_alnum(int32_t c) { return isalnum((unsigned char)c) != 0; }
static inline bool vira_ctype_is_space(int32_t c) { return isspace((unsigned char)c) != 0; }
static inline bool vira_ctype_is_upper(int32_t c) { return isupper((unsigned char)c) != 0; }
static inline bool vira_ctype_is_lower(int32_t c) { return islower((unsigned char)c) != 0; }
static inline bool vira_ctype_is_xdigit(int32_t c) { return isxdigit((unsigned char)c) != 0; }
static inline bool vira_ctype_is_print(int32_t c) { return isprint((unsigned char)c) != 0; }
static inline bool vira_ctype_is_punct(int32_t c) { return ispunct((unsigned char)c) != 0; }
static inline bool vira_ctype_is_cntrl(int32_t c) { return iscntrl((unsigned char)c) != 0; }
static inline int32_t vira_ctype_to_lower(int32_t c) { return tolower((unsigned char)c); }
static inline int32_t vira_ctype_to_upper(int32_t c) { return toupper((unsigned char)c); }

static inline double vira_math_sin(double x) { return sin(x); }
static inline double vira_math_cos(double x) { return cos(x); }
static inline double vira_math_tan(double x) { return tan(x); }
static inline double vira_math_asin(double x) { return asin(x); }
static inline double vira_math_acos(double x) { return acos(x); }
static inline double vira_math_atan(double x) { return atan(x); }
static inline double vira_math_atan2(double y, double x) { return atan2(y, x); }
static inline double vira_math_sinh(double x) { return sinh(x); }
static inline double vira_math_cosh(double x) { return cosh(x); }
static inline double vira_math_tanh(double x) { return tanh(x); }
static inline double vira_math_exp(double x) { return exp(x); }
static inline double vira_math_exp2(double x) { return exp2(x); }
static inline double vira_math_log(double x) { return log(x); }
static inline double vira_math_log2(double x) { return log2(x); }
static inline double vira_math_log10(double x) { return log10(x); }
static inline double vira_math_pow(double x, double y) { return pow(x, y); }
static inline double vira_math_sqrt(double x) { return sqrt(x); }
static inline double vira_math_cbrt(double x) { return cbrt(x); }
static inline double vira_math_hypot(double x, double y) { return hypot(x, y); }
static inline double vira_math_abs(double x) { return fabs(x); }
static inline int32_t vira_math_abs_int(int32_t x) { return x < 0 ? -x : x; }
static inline int64_t vira_math_abs_i64(int64_t x) { return x < 0 ? -x : x; }
static inline double vira_math_floor(double x) { return floor(x); }
static inline double vira_math_ceil(double x) { return ceil(x); }
static inline double vira_math_round(double x) { return round(x); }
static inline double vira_math_trunc(double x) { return trunc(x); }
static inline double vira_math_fmod(double x, double y) { return fmod(x, y); }
static inline bool vira_math_is_nan(double x) { return isnan(x) != 0; }
static inline bool vira_math_is_inf(double x) { return isinf(x) != 0; }
static inline void vira_math_srand(uint32_t seed) { srand((unsigned int)seed); }
static inline int32_t vira_math_rand_int(int32_t min_v, int32_t max_v) {
  if (min_v >= max_v) return min_v;
  return min_v + (rand() % (max_v - min_v + 1));
}
static inline double vira_math_rand_f64(void) {
  return (double)rand() / (double)RAND_MAX;
}

static int _cmp_vira_int(const void *a, const void *b) {
  int32_t ia = *(const int32_t *)a;
  int32_t ib = *(const int32_t *)b;
  return (ia > ib) - (ia < ib);
}
static inline void vira_math_sort_ints(vira_slice_int *s) {
  if (s != NULL && s->data != NULL && s->len > 1) {
    qsort(s->data, (size_t)s->len, sizeof(int32_t), _cmp_vira_int);
  }
}

static int _cmp_vira_f64(const void *a, const void *b) {
  double fa = *(const double *)a;
  double fb = *(const double *)b;
  return (fa > fb) - (fa < fb);
}
static inline void vira_math_sort_f64s(vira_slice_f64 *s) {
  if (s != NULL && s->data != NULL && s->len > 1) {
    qsort(s->data, (size_t)s->len, sizeof(double), _cmp_vira_f64);
  }
}

static int _cmp_vira_str(const void *a, const void *b) {
  const vira_string *sa = (const vira_string *)a;
  const vira_string *sb = (const vira_string *)b;
  int32_t min_len = sa->len < sb->len ? sa->len : sb->len;
  if (min_len > 0 && sa->data != NULL && sb->data != NULL) {
    int c = memcmp(sa->data, sb->data, (size_t)min_len);
    if (c != 0) return c;
  }
  return sa->len - sb->len;
}
static inline void vira_strings_sort(vira_slice_string *s) {
  if (s != NULL && s->data != NULL && s->len > 1) {
    qsort(s->data, (size_t)s->len, sizeof(vira_string), _cmp_vira_str);
  }
}

static inline int64_t vira_time_now_unix(void) {
  return (int64_t)time(NULL);
}

static inline int64_t vira_time_now_ns(void) {
#if defined(_WIN32)
  static LARGE_INTEGER freq;
  static int init = 0;
  if (!init) {
    QueryPerformanceFrequency(&freq);
    init = 1;
  }
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  return (int64_t)((counter.QuadPart * 1000000000LL) / freq.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
#endif
}

static inline int64_t vira_time_monotonic_ns(void) {
#if defined(_WIN32)
  static LARGE_INTEGER freq;
  static int init = 0;
  if (!init) {
    QueryPerformanceFrequency(&freq);
    init = 1;
  }
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  return (int64_t)((counter.QuadPart * 1000000000LL) / freq.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
#endif
}

static inline Result_string vira_time_strftime(vira_string fmt, int64_t unix_sec, bool utc) {
  char stack_fmt[256];
  vira_cstr_buf fb = vira_str_to_cstr_buf(fmt, stack_fmt, sizeof(stack_fmt));
  time_t t = (time_t)unix_sec;
  struct tm tm_buf;
#ifdef _WIN32
  if (utc) gmtime_s(&tm_buf, &t);
  else localtime_s(&tm_buf, &t);
#else
  if (utc) gmtime_r(&t, &tm_buf);
  else localtime_r(&t, &tm_buf);
#endif
  char out_buf[256];
  size_t n = strftime(out_buf, sizeof(out_buf), fb.ptr, &tm_buf);
  vira_cstr_buf_free(&fb);
  if (n == 0) {
    return (Result_string){.is_err = true, .err = vira_str_lit("strftime formatting failed", 26)};
  }
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(out_buf)};
}

// ============================================================================
// 9. SYSTEMS, PROCESS, SIGNALS & CONTAINER RUNTIME SHIMS
// ============================================================================

#define _VIRA_SIG_QUEUE_CAP 16
static volatile sig_atomic_t _vira_sig_queue[_VIRA_SIG_QUEUE_CAP];
static volatile sig_atomic_t _vira_sig_head = 0;
static volatile sig_atomic_t _vira_sig_tail = 0;

static void _vira_sig_handler(int sig) {
  int next = (_vira_sig_head + 1) % _VIRA_SIG_QUEUE_CAP;
  if (next != _vira_sig_tail) {
    _vira_sig_queue[_vira_sig_head] = sig;
    _vira_sig_head = next;
  }
}

static inline Result_bool vira_sys_signal_notify(vira_slice_int sigs) {
  for (int32_t i = 0; i < sigs.len; i++) {
    signal(sigs.data[i], _vira_sig_handler);
  }
  return (Result_bool){.is_err = false, .val = true};
}

static inline int32_t vira_sys_signal_poll(void) {
  if (_vira_sig_tail == _vira_sig_head) {
    return 0;
  }
  int sig = _vira_sig_queue[_vira_sig_tail];
  _vira_sig_tail = (_vira_sig_tail + 1) % _VIRA_SIG_QUEUE_CAP;
  return sig;
}

static inline int32_t vira_sys_signal_wait(void) {
  while (1) {
    int sig = vira_sys_signal_poll();
    if (sig != 0) return sig;
    vira_time_sleep_ms(10);
  }
}

static inline int32_t vira_sys_errno(void) {
  return (int32_t)errno;
}

static inline vira_string vira_sys_strerror(int32_t err) {
  return vira_str_from_cstr(strerror(err));
}

#define VIRA_O_RDONLY   0
#define VIRA_O_WRONLY   1
#define VIRA_O_RDWR     2
#define VIRA_O_CREAT    0x40
#define VIRA_O_EXCL     0x80
#define VIRA_O_TRUNC    0x200
#define VIRA_O_APPEND   0x400

static inline Result_int vira_sys_open(vira_string path, int32_t flags, int32_t mode) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
  int native_flags = 0;
  int acc = flags & 3;
#ifdef _WIN32
  if (acc == VIRA_O_WRONLY) native_flags |= _O_WRONLY;
  else if (acc == VIRA_O_RDWR) native_flags |= _O_RDWR;
  else native_flags |= _O_RDONLY;
  if (flags & VIRA_O_CREAT) native_flags |= _O_CREAT;
  if (flags & VIRA_O_EXCL) native_flags |= _O_EXCL;
  if (flags & VIRA_O_TRUNC) native_flags |= _O_TRUNC;
  if (flags & VIRA_O_APPEND) native_flags |= _O_APPEND;
  native_flags |= _O_BINARY;
  int fd = _open(p.ptr, native_flags, mode == 0 ? 0666 : mode);
#else
  if (acc == VIRA_O_WRONLY) native_flags |= O_WRONLY;
  else if (acc == VIRA_O_RDWR) native_flags |= O_RDWR;
  else native_flags |= O_RDONLY;
  if (flags & VIRA_O_CREAT) native_flags |= O_CREAT;
  if (flags & VIRA_O_EXCL) native_flags |= O_EXCL;
  if (flags & VIRA_O_TRUNC) native_flags |= O_TRUNC;
  if (flags & VIRA_O_APPEND) native_flags |= O_APPEND;
  int fd = open(p.ptr, native_flags, mode == 0 ? 0666 : (mode_t)mode);
#endif
  vira_cstr_buf_free(&p);
  if (fd < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("open error: %s", strerror(errno))};
  }
  return (Result_int){.is_err = false, .val = fd};
}

static inline Result_int vira_sys_read(int32_t fd, vira_slice_u8 *buf, int32_t max_bytes) {
  if (buf == NULL || max_bytes <= 0) return (Result_int){.is_err = false, .val = 0};
  if (buf->cap < max_bytes) {
    while (buf->cap < max_bytes) {
      _vira_slice_grow((void**)&buf->data, &buf->cap, sizeof(uint8_t), buf->is_arena);
    }
  }
#ifdef _WIN32
  int n = _read(fd, buf->data, (unsigned int)max_bytes);
#else
  ssize_t n = read(fd, buf->data, (size_t)max_bytes);
#endif
  if (n < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("read error: %s", strerror(errno))};
  }
  buf->len = (int32_t)n;
  return (Result_int){.is_err = false, .val = (int32_t)n};
}

static inline Result_int vira_sys_write(int32_t fd, vira_slice_u8 buf, int32_t count) {
  if (buf.data == NULL || count <= 0) return (Result_int){.is_err = false, .val = 0};
  if (count > buf.len) count = buf.len;
#ifdef _WIN32
  int n = _write(fd, buf.data, (unsigned int)count);
#else
  ssize_t n = write(fd, buf.data, (size_t)count);
#endif
  if (n < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("write error: %s", strerror(errno))};
  }
  return (Result_int){.is_err = false, .val = (int32_t)n};
}

static inline Result_bool vira_sys_close(int32_t fd) {
#ifdef _WIN32
  int res = _close(fd);
#else
  int res = close(fd);
#endif
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("close error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_i64 vira_sys_lseek(int32_t fd, int64_t offset, int32_t whence) {
#ifdef _WIN32
  __int64 res = _lseeki64(fd, offset, whence);
#else
  off_t res = lseek(fd, (off_t)offset, whence);
#endif
  if (res < 0) {
    return (Result_i64){.is_err = true, .err = vira_str_format("lseek error: %s", strerror(errno))};
  }
  return (Result_i64){.is_err = false, .val = (int64_t)res};
}

static inline Result_bool vira_sys_pipe(vira_slice_int *fds) {
  if (fds == NULL) return (Result_bool){.is_err = true, .err = vira_str_lit("null slice", 10)};
  int p[2];
#ifdef _WIN32
  if (_pipe(p, 4096, _O_BINARY) != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("pipe error: %s", strerror(errno))};
  }
#else
  if (pipe(p) != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("pipe error: %s", strerror(errno))};
  }
#endif
  vira_slice_int_clear(fds);
  vira_slice_int_push(fds, p[0]);
  vira_slice_int_push(fds, p[1]);
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_int vira_sys_dup(int32_t fd) {
#ifdef _WIN32
  int res = _dup(fd);
#else
  int res = dup(fd);
#endif
  if (res < 0) return (Result_int){.is_err = true, .err = vira_str_format("dup error: %s", strerror(errno))};
  return (Result_int){.is_err = false, .val = res};
}

static inline Result_int vira_sys_dup2(int32_t oldfd, int32_t newfd) {
#ifdef _WIN32
  int res = _dup2(oldfd, newfd);
#else
  int res = dup2(oldfd, newfd);
#endif
  if (res < 0) return (Result_int){.is_err = true, .err = vira_str_format("dup2 error: %s", strerror(errno))};
  return (Result_int){.is_err = false, .val = res};
}

static inline bool vira_sys_isatty(int32_t fd) {
#ifdef _WIN32
  return _isatty(fd) != 0;
#else
  return isatty(fd) != 0;
#endif
}

static inline int32_t vira_sys_getpid(void) {
#ifdef _WIN32
  return (int32_t)GetCurrentProcessId();
#else
  return (int32_t)getpid();
#endif
}

static inline int32_t vira_sys_getppid(void) {
#ifdef _WIN32
  return 0;
#else
  return (int32_t)getppid();
#endif
}

static inline int32_t vira_sys_getuid(void) {
#ifdef _WIN32
  return 0;
#else
  return (int32_t)getuid();
#endif
}

static inline int32_t vira_sys_getgid(void) {
#ifdef _WIN32
  return 0;
#else
  return (int32_t)getgid();
#endif
}

static inline Result_bool vira_sys_kill(int32_t pid, int32_t sig) {
#ifdef _WIN32
  HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD)pid);
  if (!h) return (Result_bool){.is_err = true, .err = vira_str_lit("failed to open process", 22)};
  BOOL ok = TerminateProcess(h, (UINT)sig);
  CloseHandle(h);
  if (!ok) return (Result_bool){.is_err = true, .err = vira_str_lit("failed to terminate process", 27)};
  return (Result_bool){.is_err = false, .val = true};
#else
  if (kill(pid, sig) != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("kill error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#endif
}

static inline int32_t vira_sys_system(vira_string cmd) {
  char stack_buf[1024];
  vira_cstr_buf c = vira_str_to_cstr_buf(cmd, stack_buf, sizeof(stack_buf));
  int res = system(c.ptr);
  vira_cstr_buf_free(&c);
  return res;
}

static inline Result_string vira_sys_getenv(vira_string name) {
  char stack_buf[256];
  vira_cstr_buf n = vira_str_to_cstr_buf(name, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  char win_buf[32767];
  DWORD win_len = GetEnvironmentVariableA(n.ptr, win_buf, sizeof(win_buf));
  if (win_len > 0) {
    vira_cstr_buf_free(&n);
    return (Result_string){.is_err = false, .val = vira_str_from_cstr(win_buf)};
  }
#endif
  const char *val = getenv(n.ptr);
  vira_cstr_buf_free(&n);
  if (!val) {
    return (Result_string){.is_err = true, .err = vira_str_lit("environment variable not found", 30)};
  }
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(val)};
}

static inline Result_bool vira_sys_setenv(vira_string name, vira_string val) {
  char stack_n[256], stack_v[1024];
  vira_cstr_buf n = vira_str_to_cstr_buf(name, stack_n, sizeof(stack_n));
  vira_cstr_buf v = vira_str_to_cstr_buf(val, stack_v, sizeof(stack_v));
#ifdef _WIN32
  _putenv_s(n.ptr, v.ptr);
  BOOL ok = SetEnvironmentVariableA(n.ptr, v.ptr);
  vira_cstr_buf_free(&n);
  vira_cstr_buf_free(&v);
  if (!ok) return (Result_bool){.is_err = true, .err = vira_str_lit("SetEnvironmentVariable failed", 29)};
  return (Result_bool){.is_err = false, .val = true};
#else
  int res = setenv(n.ptr, v.ptr, 1);
  vira_cstr_buf_free(&n);
  vira_cstr_buf_free(&v);
  if (res != 0) return (Result_bool){.is_err = true, .err = vira_str_format("setenv error: %s", strerror(errno))};
  return (Result_bool){.is_err = false, .val = true};
#endif
}

static inline Result_bool vira_sys_unsetenv(vira_string name) {
  char stack_n[256];
  vira_cstr_buf n = vira_str_to_cstr_buf(name, stack_n, sizeof(stack_n));
#ifdef _WIN32
  _putenv_s(n.ptr, "");
  SetEnvironmentVariableA(n.ptr, NULL);
  vira_cstr_buf_free(&n);
  return (Result_bool){.is_err = false, .val = true};
#else
  int res = unsetenv(n.ptr);
  vira_cstr_buf_free(&n);
  if (res != 0) return (Result_bool){.is_err = true, .err = vira_str_format("unsetenv error: %s", strerror(errno))};
  return (Result_bool){.is_err = false, .val = true};
#endif
}

static inline vira_slice_string vira_sys_environ(void) {
  vira_slice_string res = vira_slice_string_new(16);
#ifdef _WIN32
  char *env_block = GetEnvironmentStringsA();
  if (env_block) {
    char *p = env_block;
    while (*p) {
      vira_slice_string_push(&res, vira_str_from_cstr(p));
      p += strlen(p) + 1;
    }
    FreeEnvironmentStringsA(env_block);
  }
#else
  extern char **environ;
  if (environ) {
    for (char **p = environ; *p != NULL; p++) {
      vira_slice_string_push(&res, vira_str_from_cstr(*p));
    }
  }
#endif
  return res;
}

static inline Result_string vira_sys_hostname(void) {
  char buf[256];
  if (gethostname(buf, sizeof(buf)) != 0) {
    return (Result_string){.is_err = true, .err = vira_str_lit("failed to get hostname", 22)};
  }
  buf[sizeof(buf) - 1] = '\0';
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(buf)};
}

static inline Result_string vira_sys_executable_path(void) {
#ifdef _WIN32
  char buf[MAX_PATH];
  DWORD len = GetModuleFileNameA(NULL, buf, MAX_PATH);
  if (len == 0 || len >= MAX_PATH) {
    return (Result_string){.is_err = true, .err = vira_str_lit("GetModuleFileName failed", 24)};
  }
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(buf)};
#elif defined(__linux__)
  char buf[4096];
  ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len <= 0) {
    return (Result_string){.is_err = true, .err = vira_str_lit("readlink /proc/self/exe failed", 30)};
  }
  buf[len] = '\0';
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(buf)};
#else
  return (Result_string){.is_err = true, .err = vira_str_lit("executable_path not supported on this platform", 46)};
#endif
}

static inline Result_bool vira_sys_mkdir(vira_string path, int32_t perm) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  (void)perm;
  int res = _mkdir(p.ptr);
#else
  int res = mkdir(p.ptr, (mode_t)perm);
#endif
  vira_cstr_buf_free(&p);
  if (res != 0 && errno != EEXIST) {
    return (Result_bool){.is_err = true, .err = vira_str_format("mkdir error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_bool vira_sys_rmdir(vira_string path) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  int res = _rmdir(p.ptr);
#else
  int res = rmdir(p.ptr);
#endif
  vira_cstr_buf_free(&p);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("rmdir error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_bool vira_sys_unlink(vira_string path) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  int res = _unlink(p.ptr);
#else
  int res = unlink(p.ptr);
#endif
  vira_cstr_buf_free(&p);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("unlink error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_bool vira_sys_chdir(vira_string path) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  int res = _chdir(p.ptr);
#else
  int res = chdir(p.ptr);
#endif
  vira_cstr_buf_free(&p);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("chdir error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_string vira_sys_getcwd(void) {
  char buf[4096];
#ifdef _WIN32
  char *res = _getcwd(buf, sizeof(buf));
#else
  char *res = getcwd(buf, sizeof(buf));
#endif
  if (!res) {
    return (Result_string){.is_err = true, .err = vira_str_format("getcwd error: %s", strerror(errno))};
  }
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(buf)};
}

static inline Result_bool vira_sys_chmod(vira_string path, int32_t perm) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  int res = _chmod(p.ptr, perm);
#else
  int res = chmod(p.ptr, (mode_t)perm);
#endif
  vira_cstr_buf_free(&p);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("chmod error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

static inline vira_slice_string vira_sys_readdir(vira_string path) {
  vira_slice_string res = vira_slice_string_new(16);
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  char search_path[MAX_PATH];
  snprintf(search_path, sizeof(search_path), "%s\\*", p.ptr);
  WIN32_FIND_DATAA fd;
  HANDLE hFind = FindFirstFileA(search_path, &fd);
  if (hFind != INVALID_HANDLE_VALUE) {
    do {
      if (strcmp(fd.cFileName, ".") != 0 && strcmp(fd.cFileName, "..") != 0) {
        vira_slice_string_push(&res, vira_str_from_cstr(fd.cFileName));
      }
    } while (FindNextFileA(hFind, &fd));
    FindClose(hFind);
  }
#else
  DIR *d = opendir(p.ptr);
  if (d) {
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
      if (strcmp(de->d_name, ".") != 0 && strcmp(de->d_name, "..") != 0) {
        vira_slice_string_push(&res, vira_str_from_cstr(de->d_name));
      }
    }
    closedir(d);
  }
#endif
  vira_cstr_buf_free(&p);
  return res;
}

static inline Result_bool vira_sys_mount(vira_string source, vira_string target, vira_string fstype, int64_t flags, vira_string data) {
#if defined(__linux__)
  char s_src[1024], s_tgt[1024], s_fst[64], s_dat[1024];
  vira_cstr_buf b_src = vira_str_to_cstr_buf(source, s_src, sizeof(s_src));
  vira_cstr_buf b_tgt = vira_str_to_cstr_buf(target, s_tgt, sizeof(s_tgt));
  vira_cstr_buf b_fst = vira_str_to_cstr_buf(fstype, s_fst, sizeof(s_fst));
  vira_cstr_buf b_dat = vira_str_to_cstr_buf(data, s_dat, sizeof(s_dat));
  const char *p_src = (source.len > 0) ? b_src.ptr : NULL;
  const char *p_fst = (fstype.len > 0) ? b_fst.ptr : NULL;
  const void *p_dat = (data.len > 0) ? b_dat.ptr : NULL;
  int res = mount(p_src, b_tgt.ptr, p_fst, (unsigned long)flags, p_dat);
  vira_cstr_buf_free(&b_src);
  vira_cstr_buf_free(&b_tgt);
  vira_cstr_buf_free(&b_fst);
  vira_cstr_buf_free(&b_dat);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("mount error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)source; (void)target; (void)fstype; (void)flags; (void)data;
  return (Result_bool){.is_err = true, .err = vira_str_lit("mount: only supported on Linux", 30)};
#endif
}

static inline Result_bool vira_sys_unmount(vira_string target, int32_t flags) {
#if defined(__linux__)
  char stack_buf[1024];
  vira_cstr_buf b = vira_str_to_cstr_buf(target, stack_buf, sizeof(stack_buf));
  int res = umount2(b.ptr, flags);
  vira_cstr_buf_free(&b);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("umount2 error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)target; (void)flags;
  return (Result_bool){.is_err = true, .err = vira_str_lit("umount2: only supported on Linux", 32)};
#endif
}

static inline Result_bool vira_sys_chroot(vira_string path) {
#if defined(__linux__)
  char stack_buf[1024];
  vira_cstr_buf b = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
  int res = chroot(b.ptr);
  vira_cstr_buf_free(&b);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("chroot error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)path;
  return (Result_bool){.is_err = true, .err = vira_str_lit("chroot: only supported on Linux", 31)};
#endif
}

static inline Result_bool vira_sys_mknod(vira_string path, int32_t mode, int64_t dev) {
#if defined(__linux__)
  char stack_buf[1024];
  vira_cstr_buf b = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
  int res = mknod(b.ptr, (mode_t)mode, (dev_t)dev);
  vira_cstr_buf_free(&b);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("mknod error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)path; (void)mode; (void)dev;
  return (Result_bool){.is_err = true, .err = vira_str_lit("mknod: only supported on Linux", 30)};
#endif
}

static inline int64_t vira_sys_makedev(int32_t major, int32_t minor) {
#if defined(__linux__)
  return (int64_t)gnu_dev_makedev((unsigned int)major, (unsigned int)minor);
#else
  return ((int64_t)(major) << 8) | (int64_t)(minor);
#endif
}

static inline Result_bool vira_sys_unshare(int32_t flags) {
#if defined(__linux__)
  if (unshare(flags) != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("unshare error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)flags;
  return (Result_bool){.is_err = true, .err = vira_str_lit("unshare: only supported on Linux", 32)};
#endif
}

static inline Result_bool vira_sys_sethostname(vira_string name) {
#if defined(__linux__)
  char stack_buf[256];
  vira_cstr_buf b = vira_str_to_cstr_buf(name, stack_buf, sizeof(stack_buf));
  int res = sethostname(b.ptr, strlen(b.ptr));
  vira_cstr_buf_free(&b);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("sethostname error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)name;
  return (Result_bool){.is_err = true, .err = vira_str_lit("sethostname: only supported on Linux", 36)};
#endif
}

static inline Result_int vira_sys_exec_run(vira_slice_string args, vira_string dir, bool inherit_stdio) {
  if (args.data == NULL || args.len <= 0) {
    return (Result_int){.is_err = true, .err = vira_str_lit("empty command", 13)};
  }
#ifdef _WIN32
  size_t cmd_len = 0;
  for (int32_t i = 0; i < args.len; i++) {
    cmd_len += (size_t)args.data[i].len + 3;
  }
  size_t cmd_cap = (cmd_len < 4096) ? 4096 : (cmd_len + 1024);
  char *cmd_line = (char *)calloc(1, cmd_cap);
  for (int32_t i = 0; i < args.len; i++) {
    char sbuf[256];
    vira_cstr_buf b = vira_str_to_cstr_buf(args.data[i], sbuf, sizeof(sbuf));
    if (i > 0) strcat(cmd_line, " ");
    bool need_quotes = (strchr(b.ptr, ' ') != NULL || b.ptr[0] == '\0');
    if (need_quotes) strcat(cmd_line, "\"");
    strcat(cmd_line, b.ptr);
    if (need_quotes) strcat(cmd_line, "\"");
    vira_cstr_buf_free(&b);
  }

  STARTUPINFOA si;
  PROCESS_INFORMATION pi;
  memset(&si, 0, sizeof(si));
  memset(&pi, 0, sizeof(pi));
  si.cb = sizeof(si);

  char dstack[1024];
  vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
  const char *cdir = (dir.len > 0) ? db.ptr : NULL;

  BOOL ok = CreateProcessA(NULL, cmd_line, NULL, NULL, TRUE, 0, NULL, cdir, &si, &pi);
  free(cmd_line);
  vira_cstr_buf_free(&db);
  if (!ok) {
    return (Result_int){.is_err = true, .err = vira_str_lit("CreateProcess failed", 20)};
  }
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD exit_code = 0;
  GetExitCodeProcess(pi.hProcess, &exit_code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  return (Result_int){.is_err = false, .val = (int32_t)exit_code};
#else
  (void)inherit_stdio;
  pid_t pid = fork();
  if (pid == 0) {
    if (dir.len > 0) {
      char dstack[1024];
      vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
      if (chdir(db.ptr) != 0) { /* continue */ }
      vira_cstr_buf_free(&db);
    }
    char **argv = (char **)malloc((size_t)(args.len + 1) * sizeof(char *));
    for (int32_t i = 0; i < args.len; i++) {
      argv[i] = vira_str_cstr(args.data[i]);
    }
    argv[args.len] = NULL;
    execvp(argv[0], argv);
    _exit(127);
  }
  if (pid < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("fork failed: %s", strerror(errno))};
  }
  int status = 0;
  waitpid(pid, &status, 0);
  int code = WIFEXITED(status) ? WEXITSTATUS(status) : (WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1);
  return (Result_int){.is_err = false, .val = code};
#endif
}

static inline Result_string vira_sys_exec_output(vira_slice_string args, vira_string dir, vira_string stdin_text, bool combine_err) {
  if (args.data == NULL || args.len <= 0) {
    return (Result_string){.is_err = true, .err = vira_str_lit("empty command", 13)};
  }
#ifdef _WIN32
  (void)stdin_text;
  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(SECURITY_ATTRIBUTES);
  sa.bInheritHandle = TRUE;
  sa.lpSecurityDescriptor = NULL;

  HANDLE hRead, hWrite;
  if (!CreatePipe(&hRead, &hWrite, &sa, 0)) {
    return (Result_string){.is_err = true, .err = vira_str_lit("CreatePipe failed", 17)};
  }
  SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

  size_t cmd_len = 0;
  for (int32_t i = 0; i < args.len; i++) {
    cmd_len += (size_t)args.data[i].len + 3;
  }
  size_t cmd_cap = (cmd_len < 4096) ? 4096 : (cmd_len + 1024);
  char *cmd_line = (char *)calloc(1, cmd_cap);
  for (int32_t i = 0; i < args.len; i++) {
    char sbuf[256];
    vira_cstr_buf b = vira_str_to_cstr_buf(args.data[i], sbuf, sizeof(sbuf));
    if (i > 0) strcat(cmd_line, " ");
    bool need_quotes = (strchr(b.ptr, ' ') != NULL || b.ptr[0] == '\0');
    if (need_quotes) strcat(cmd_line, "\"");
    strcat(cmd_line, b.ptr);
    if (need_quotes) strcat(cmd_line, "\"");
    vira_cstr_buf_free(&b);
  }

  STARTUPINFOA si;
  PROCESS_INFORMATION pi;
  memset(&si, 0, sizeof(si));
  memset(&pi, 0, sizeof(pi));
  si.cb = sizeof(si);
  si.dwFlags |= STARTF_USESTDHANDLES;
  si.hStdOutput = hWrite;
  si.hStdError = combine_err ? hWrite : GetStdHandle(STD_ERROR_HANDLE);
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

  char dstack[1024];
  vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
  const char *cdir = (dir.len > 0) ? db.ptr : NULL;

  BOOL ok = CreateProcessA(NULL, cmd_line, NULL, NULL, TRUE, 0, NULL, cdir, &si, &pi);
  free(cmd_line);
  vira_cstr_buf_free(&db);
  CloseHandle(hWrite);

  if (!ok) {
    CloseHandle(hRead);
    return (Result_string){.is_err = true, .err = vira_str_lit("CreateProcess failed", 20)};
  }

  size_t cap = 1024;
  size_t len = 0;
  char *out = (char *)malloc(cap);
  DWORD bytes_read = 0;
  char chunk[4096];
  while (ReadFile(hRead, chunk, sizeof(chunk), &bytes_read, NULL) && bytes_read > 0) {
    if (len + bytes_read + 1 > cap) {
      while (len + bytes_read + 1 > cap) cap *= 2;
      char *nb = (char *)realloc(out, cap);
      if (!nb) break;
      out = nb;
    }
    memcpy(out + len, chunk, bytes_read);
    len += bytes_read;
  }
  out[len] = '\0';
  CloseHandle(hRead);
  WaitForSingleObject(pi.hProcess, INFINITE);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  _vira_record_alloc();
  return (Result_string){.is_err = false, .val = (vira_string){.data = (const uint8_t *)out, .len = (int32_t)len, .is_lit = false}};
#else
  (void)stdin_text;
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    return (Result_string){.is_err = true, .err = vira_str_format("pipe error: %s", strerror(errno))};
  }
  pid_t pid = fork();
  if (pid == 0) {
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    if (combine_err) {
      dup2(pipefd[1], STDERR_FILENO);
    }
    close(pipefd[1]);
    if (dir.len > 0) {
      char dstack[1024];
      vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
      if (chdir(db.ptr) != 0) { /* continue */ }
      vira_cstr_buf_free(&db);
    }
    char **argv = (char **)malloc((size_t)(args.len + 1) * sizeof(char *));
    for (int32_t i = 0; i < args.len; i++) {
      argv[i] = vira_str_cstr(args.data[i]);
    }
    argv[args.len] = NULL;
    execvp(argv[0], argv);
    _exit(127);
  }
  close(pipefd[1]);
  if (pid < 0) {
    close(pipefd[0]);
    return (Result_string){.is_err = true, .err = vira_str_format("fork failed: %s", strerror(errno))};
  }
  size_t cap = 1024;
  size_t len = 0;
  char *out = (char *)malloc(cap);
  char chunk[4096];
  ssize_t n;
  while ((n = read(pipefd[0], chunk, sizeof(chunk))) > 0) {
    if (len + (size_t)n + 1 > cap) {
      while (len + (size_t)n + 1 > cap) cap *= 2;
      char *nb = (char *)realloc(out, cap);
      if (!nb) break;
      out = nb;
    }
    memcpy(out + len, chunk, (size_t)n);
    len += (size_t)n;
  }
  out[len] = '\0';
  close(pipefd[0]);
  waitpid(pid, NULL, 0);
  _vira_record_alloc();
  return (Result_string){.is_err = false, .val = (vira_string){.data = (const uint8_t *)out, .len = (int32_t)len, .is_lit = false}};
#endif
}

static inline Result_int vira_sys_proc_start(vira_slice_string args, vira_string dir, int32_t clone_flags, bool inherit_stdio) {
  if (args.data == NULL || args.len <= 0) {
    return (Result_int){.is_err = true, .err = vira_str_lit("empty command", 13)};
  }
#ifdef _WIN32
  (void)clone_flags;
  (void)inherit_stdio;
  size_t cmd_len = 0;
  for (int32_t i = 0; i < args.len; i++) {
    cmd_len += (size_t)args.data[i].len + 3;
  }
  size_t cmd_cap = (cmd_len < 4096) ? 4096 : (cmd_len + 1024);
  char *cmd_line = (char *)calloc(1, cmd_cap);
  for (int32_t i = 0; i < args.len; i++) {
    char sbuf[256];
    vira_cstr_buf b = vira_str_to_cstr_buf(args.data[i], sbuf, sizeof(sbuf));
    if (i > 0) strcat(cmd_line, " ");
    bool need_quotes = (strchr(b.ptr, ' ') != NULL || b.ptr[0] == '\0');
    if (need_quotes) strcat(cmd_line, "\"");
    strcat(cmd_line, b.ptr);
    if (need_quotes) strcat(cmd_line, "\"");
    vira_cstr_buf_free(&b);
  }

  STARTUPINFOA si;
  PROCESS_INFORMATION pi;
  memset(&si, 0, sizeof(si));
  memset(&pi, 0, sizeof(pi));
  si.cb = sizeof(si);

  char dstack[1024];
  vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
  const char *cdir = (dir.len > 0) ? db.ptr : NULL;

  BOOL ok = CreateProcessA(NULL, cmd_line, NULL, NULL, TRUE, 0, NULL, cdir, &si, &pi);
  free(cmd_line);
  vira_cstr_buf_free(&db);
  if (!ok) {
    return (Result_int){.is_err = true, .err = vira_str_lit("CreateProcess failed", 20)};
  }
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  return (Result_int){.is_err = false, .val = (int32_t)pi.dwProcessId};
#else
  pid_t pid;
  if (clone_flags != 0) {
#if defined(__linux__) && defined(SYS_clone)
    pid = (pid_t)syscall(SYS_clone, clone_flags | SIGCHLD, NULL, NULL, NULL, 0);
#else
    pid = fork();
#endif
  } else {
    pid = fork();
  }
  if (pid == 0) {
    if (dir.len > 0) {
      char dstack[1024];
      vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
      if (chdir(db.ptr) != 0) { /* continue */ }
      vira_cstr_buf_free(&db);
    }
    char **argv = (char **)malloc((size_t)(args.len + 1) * sizeof(char *));
    for (int32_t i = 0; i < args.len; i++) {
      argv[i] = vira_str_cstr(args.data[i]);
    }
    argv[args.len] = NULL;
    execvp(argv[0], argv);
    _exit(127);
  }
  if (pid < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("proc_start failed: %s", strerror(errno))};
  }
  return (Result_int){.is_err = false, .val = (int32_t)pid};
#endif
}

static inline Result_int vira_sys_proc_wait(int32_t pid) {
#ifdef _WIN32
  HANDLE h = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_INFORMATION, FALSE, (DWORD)pid);
  if (!h) {
    return (Result_int){.is_err = true, .err = vira_str_lit("failed to open process handle for wait", 39)};
  }
  WaitForSingleObject(h, INFINITE);
  DWORD exit_code = 0;
  GetExitCodeProcess(h, &exit_code);
  CloseHandle(h);
  return (Result_int){.is_err = false, .val = (int32_t)exit_code};
#else
  int status = 0;
  pid_t w = waitpid((pid_t)pid, &status, 0);
  if (w < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("waitpid error: %s", strerror(errno))};
  }
  int code = WIFEXITED(status) ? WEXITSTATUS(status) : (WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1);
  return (Result_int){.is_err = false, .val = code};
#endif
}

static inline void vira_report_mem_stats(void) {
  _vira_flush_tls_mem_stats();
  vira_scheduler_shutdown();
#if defined(VIRA_QUIET_RT)
  return;
#else
  if (_vira_quiet_rt) {
    return;
  }
  const char *env_stats = getenv("VIRA_RT_STATS");
  if (env_stats != NULL && strcmp(env_stats, "0") == 0) {
    return;
  }
  printf("[vira_rt] simd=%s\n", vira_simd_backend_name());
  printf(
      "[vira_rt] pool_workers=%d | actor_tasks_run=%lld | work_steals=%lld\n",
      _vira_sched.num_workers,
      (long long)__atomic_load_n(&_vira_total_tasks_run, __ATOMIC_RELAXED),
      (long long)__atomic_load_n(&_vira_total_work_steals, __ATOMIC_RELAXED));
  printf(
      "[vira_rt] active_leaks=%lld | total_freed=%lld\n",
      (long long)__atomic_load_n(&_vira_active_heap_allocs, __ATOMIC_RELAXED),
      (long long)__atomic_load_n(&_vira_total_heap_frees, __ATOMIC_RELAXED));
  fflush(stdout);
#endif
}

#include "vira_gui.h"

#endif // VIRA_RT_H
