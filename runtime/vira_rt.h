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
    fprintf(stderr, "vex panic: out of memory in vira_str_format\n");
    exit(1);
  }
  vsnprintf(buf, (size_t)needed + 1, fmt, args_copy);
  va_end(args_copy);

  _vira_record_alloc();
  return (vira_string){
      .data = (const uint8_t *)buf, .len = needed, .is_lit = false};
}

static inline vira_string vira_str_clone(vira_string s) {
  if (s.is_lit || s.data == NULL || s.len == 0) {
    return s;
  }
  uint8_t *copy = (uint8_t *)malloc((size_t)s.len + 1);
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

VIRA_DEFINE_SLICE_TYPE(u8, uint8_t)
VIRA_DEFINE_SLICE_TYPE(int, int32_t)
VIRA_DEFINE_SLICE_TYPE(i64, int64_t)
VIRA_DEFINE_SLICE_TYPE(f32, vira_f32)
VIRA_DEFINE_SLICE_TYPE(f64, double)
VIRA_DEFINE_SLICE_TYPE(bool, bool)

static inline vira_string vira_str_char_at(vira_string s, int32_t idx) {
  vira_bounds_check(idx, s.len, __FILE__, __LINE__);
  return (vira_string){.data = &s.data[idx], .len = 1, .is_lit = true};
}

// Slice for vira_string
typedef struct {
  vira_string *data;
  int32_t len;
  int32_t cap;
  bool is_arena;
} vira_slice_string;

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
  char buf[64];
  int32_t copy_len = s.len < 63 ? s.len : 63;
  memcpy(buf, s.data, (size_t)copy_len);
  buf[copy_len] = '\0';
  char *endptr = NULL;
  long val = strtol(buf, &endptr, 10);
  if (endptr == buf || *endptr != '\0') {
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
  char buf[64];
  int32_t copy_len = s.len < 63 ? s.len : 63;
  memcpy(buf, s.data, (size_t)copy_len);
  buf[copy_len] = '\0';
  char *endptr = NULL;
  long long val = strtoll(buf, &endptr, 10);
  if (endptr == buf || *endptr != '\0') {
    return (Result_i64){.is_err = true,
                        .err = vira_str_lit("invalid integer format", 22)};
  }
  return (Result_i64){.is_err = false, .val = (int64_t)val};
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

typedef struct {
  uint8_t *buf;
  size_t cap;
  size_t offset;
} vira_arena;

static inline vira_arena vira_arena_init(size_t capacity) {
  uint8_t *slab = (uint8_t *)malloc(capacity + 64);
  if (!slab) {
    fprintf(stderr, "vex panic: failed to allocate arena slab (%zu bytes)\n",
            capacity);
    exit(1);
  }
  _vira_record_alloc();
  return (vira_arena){.buf = slab, .cap = capacity, .offset = 0};
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
    fprintf(stderr, "vex panic: arena slab exhausted (%zu / %zu bytes)\n",
            a->offset + total, a->cap);
    exit(1);
  }
  a->offset += total;
  return (void *)aligned;
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
  if (a != NULL && a->buf != NULL) {
    free(a->buf);
    a->buf = NULL;
    a->cap = 0;
    a->offset = 0;
    _vira_record_free();
  }
}

// ============================================================================
// 3. AI TENSORS + HARDWARE SIMD (`AVX2` / `SSE2` / `ARM NEON`) `@` MATMUL
// ============================================================================

static inline const char *vira_simd_backend_name(void) {
#if defined(__AVX2__)
  return "x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)";
#elif defined(__ARM_NEON) || defined(__aarch64__)
  return "ARM64 NEON (128-bit 2x-Unrolled / 8x f32 per step)";
#elif defined(__SSE2__)
  return "x86_64 SSE2 (128-bit / 4x f32 per cycle)";
#else
  return "Scalar C99 Fallback";
#endif
}

typedef struct {
  vira_f32 *data;  // 64-byte cache-line aligned pointer
  void *raw_alloc; // underlying pointer for free()
  int32_t rows;
  int32_t cols;
  bool is_arena;
} vira_tensor_f32;

static inline vira_tensor_f32 vira_tensor_f32_alloc(int32_t rows,
                                                    int32_t cols) {
  size_t bytes = (size_t)rows * (size_t)cols * sizeof(vira_f32);
  void *raw = malloc(bytes + 64);
  if (!raw) {
    fprintf(stderr, "vira panic: tensor allocation failed (%d x %d)\n", rows,
            cols);
    exit(1);
  }
  uintptr_t aligned = ((uintptr_t)raw + 63) & ~(uintptr_t)63;
  vira_f32 *data = (vira_f32 *)aligned;
  memset(data, 0, bytes);
  _vira_record_alloc();
  return (vira_tensor_f32){.data = data,
                           .raw_alloc = raw,
                           .rows = rows,
                           .cols = cols,
                           .is_arena = false};
}

static inline vira_tensor_f32
vira_tensor_f32_arena_alloc(vira_arena *a, int32_t rows, int32_t cols) {
  size_t bytes = (size_t)rows * (size_t)cols * sizeof(vira_f32);
  vira_f32 *data = (vira_f32 *)vira_arena_alloc(a, bytes);
  memset(data, 0, bytes);
  return (vira_tensor_f32){.data = data,
                           .raw_alloc = NULL,
                           .rows = rows,
                           .cols = cols,
                           .is_arena = true};
}

static inline vira_tensor_f32 vira_tensor_f32_fill(vira_arena *opt_arena,
                                                   int32_t rows, int32_t cols,
                                                   vira_f32 val) {
  vira_tensor_f32 t = opt_arena
                          ? vira_tensor_f32_arena_alloc(opt_arena, rows, cols)
                          : vira_tensor_f32_alloc(rows, cols);
  int32_t total = rows * cols;
  int32_t i = 0;

#if defined(__AVX2__)
  __m256 vval = _mm256_set1_ps(val);
  for (; i + 8 <= total; i += 8) {
    _mm256_storeu_ps(&t.data[i], vval);
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t vval = vdupq_n_f32(val);
  for (; i + 4 <= total; i += 4) {
    vst1q_f32(&t.data[i], vval);
  }
#endif
  for (; i < total; i++) {
    t.data[i] = val;
  }
  return t;
}

// Hardware-accelerated `@` Matrix Multiplication (C = A @ B)
static inline vira_tensor_f32 vira_tensor_matmul_f32(vira_arena *opt_arena,
                                                     vira_tensor_f32 a,
                                                     vira_tensor_f32 b) {
  if (a.cols != b.rows) {
    fprintf(stderr,
            "vex tensor error: matmul '@' shape mismatch (%dx%d) @ (%dx%d)\n",
            a.rows, a.cols, b.rows, b.cols);
    exit(1);
  }
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc(opt_arena, a.rows, b.cols)
                : vira_tensor_f32_alloc(a.rows, b.cols);

  const int32_t M = a.rows;
  const int32_t K = a.cols;
  const int32_t N = b.cols;

  for (int32_t si = 0; si < M; si += 64) {
    int32_t imax = (si + 64 < M) ? (si + 64) : M;
    for (int32_t sk = 0; sk < K; sk += 64) {
      int32_t kmax = (sk + 64 < K) ? (sk + 64) : K;
      for (int32_t sj = 0; sj < N; sj += 64) {
        int32_t jmax = (sj + 64 < N) ? (sj + 64) : N;

        for (int32_t i = si; i < imax; i++) {
          vira_f32 *c_row = &out.data[i * N];
          for (int32_t k = sk; k < kmax; k++) {
            const vira_f32 a_ik = a.data[i * K + k];
            const vira_f32 *b_row = &b.data[k * N];
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
  return out;
}

// SIMD-accelerated tensor sum reduction
static inline vira_f32 vira_tensor_sum(vira_tensor_f32 t) {
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

// SIMD-accelerated ReLU activation: out = max(0, t)
static inline vira_tensor_f32 vira_tensor_relu(vira_arena *opt_arena,
                                               vira_tensor_f32 t) {
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc(opt_arena, t.rows, t.cols)
                : vira_tensor_f32_alloc(t.rows, t.cols);
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

// SIMD-accelerated tensor addition: out = a + b
static inline vira_tensor_f32
vira_tensor_add(vira_arena *opt_arena, vira_tensor_f32 a, vira_tensor_f32 b) {
  if (a.rows != b.rows || a.cols != b.cols) {
    fprintf(stderr, "vex tensor error: add shape mismatch (%dx%d) + (%dx%d)\n",
            a.rows, a.cols, b.rows, b.cols);
    exit(1);
  }
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc(opt_arena, a.rows, a.cols)
                : vira_tensor_f32_alloc(a.rows, a.cols);
  int32_t total = a.rows * a.cols;
  int32_t i = 0;

#if defined(__AVX2__)
  for (; i + 8 <= total; i += 8) {
    __m256 va = _mm256_loadu_ps(&a.data[i]);
    __m256 vb = _mm256_loadu_ps(&b.data[i]);
    _mm256_storeu_ps(&out.data[i], _mm256_add_ps(va, vb));
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  for (; i + 4 <= total; i += 4) {
    float32x4_t va = vld1q_f32(&a.data[i]);
    float32x4_t vb = vld1q_f32(&b.data[i]);
    vst1q_f32(&out.data[i], vaddq_f32(va, vb));
  }
#elif defined(__SSE2__)
  for (; i + 4 <= total; i += 4) {
    __m128 va = _mm_loadu_ps(&a.data[i]);
    __m128 vb = _mm_loadu_ps(&b.data[i]);
    _mm_storeu_ps(&out.data[i], _mm_add_ps(va, vb));
  }
#endif
  for (; i < total; i++) {
    out.data[i] = a.data[i] + b.data[i];
  }
  return out;
}

// SIMD-accelerated tensor subtraction: out = a - b
static inline vira_tensor_f32
vira_tensor_sub(vira_arena *opt_arena, vira_tensor_f32 a, vira_tensor_f32 b) {
  if (a.rows != b.rows || a.cols != b.cols) {
    fprintf(stderr, "vex tensor error: sub shape mismatch (%dx%d) - (%dx%d)\n",
            a.rows, a.cols, b.rows, b.cols);
    exit(1);
  }
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc(opt_arena, a.rows, a.cols)
                : vira_tensor_f32_alloc(a.rows, a.cols);
  int32_t total = a.rows * a.cols;
  int32_t i = 0;

#if defined(__AVX2__)
  for (; i + 8 <= total; i += 8) {
    __m256 va = _mm256_loadu_ps(&a.data[i]);
    __m256 vb = _mm256_loadu_ps(&b.data[i]);
    _mm256_storeu_ps(&out.data[i], _mm256_sub_ps(va, vb));
  }
#elif defined(__ARM_NEON) || defined(__aarch64__)
  for (; i + 4 <= total; i += 4) {
    float32x4_t va = vld1q_f32(&a.data[i]);
    float32x4_t vb = vld1q_f32(&b.data[i]);
    vst1q_f32(&out.data[i], vsubq_f32(va, vb));
  }
#elif defined(__SSE2__)
  for (; i + 4 <= total; i += 4) {
    __m128 va = _mm_loadu_ps(&a.data[i]);
    __m128 vb = _mm_loadu_ps(&b.data[i]);
    _mm_storeu_ps(&out.data[i], _mm_sub_ps(va, vb));
  }
#endif
  for (; i < total; i++) {
    out.data[i] = a.data[i] - b.data[i];
  }
  return out;
}

// SIMD-accelerated tensor scalar scaling: out = t * scale
static inline vira_tensor_f32
vira_tensor_scale(vira_arena *opt_arena, vira_tensor_f32 t, vira_f32 scale) {
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc(opt_arena, t.rows, t.cols)
                : vira_tensor_f32_alloc(t.rows, t.cols);
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
  if (t != NULL && !t->is_arena && t->raw_alloc != NULL) {
    free(t->raw_alloc);
    t->raw_alloc = NULL;
    t->data = NULL;
    _vira_record_free();
  }
}

// ============================================================================
// 4. MULTI-THREADED LOCK-FREE WORK-STEALING ACTOR POOL & SUPERVISOR
// ============================================================================

#define VIRA_WS_CAP 4096
#define VIRA_WS_MASK (VIRA_WS_CAP - 1)
#define VIRA_MAX_WORKERS 32

typedef void (*vira_task_fn)(void *env);

typedef struct _vira_task {
  vira_task_fn fn;
  void *env;
  struct _vira_task *next_actor_msg; // Intrusive MPSC link
  volatile int32_t done;             // 0 = pending, 1 = completed on worker pool
} vira_task;

static void _vira_actor_drain_task(void *env);

typedef enum {
  VIRA_ACTOR_RUNNING = 0,
  VIRA_ACTOR_FAULTED = 1,
  VIRA_ACTOR_STOPPED = 2,
} vira_actor_state;

typedef struct {
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
} vira_actor_ctx;

// Thread-Local Storage (TLS): Each worker pthread tracks its own active actor!
static __thread vira_actor_ctx *_vira_tls_actor = NULL;
static __thread int32_t _vira_tls_worker_id = -1;

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
                   .offset = 0}; // Lazy SIMD-aligned scratch arena per actor
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
}

static inline void vira_actor_ctx_destroy(vira_actor_ctx *ctx) {
  while (__atomic_load_n(&ctx->scheduled, __ATOMIC_ACQUIRE)) {
    vira_cpu_relax();
  }
  vira_arena_destroy(&ctx->isolated_heap);
}

static inline void vira_panic(vira_string msg) {
  if (_vira_tls_actor != NULL && _vira_tls_actor->in_actor) {
    _vira_tls_actor->state = VIRA_ACTOR_FAULTED;
    if (!_vira_tls_actor->last_error.is_lit && _vira_tls_actor->last_error.data != NULL) {
      vira_str_free(&_vira_tls_actor->last_error);
    }
    _vira_tls_actor->last_error = vira_str_clone(msg);
    longjmp(_vira_tls_actor->fault_boundary, 1);
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

// --- ASYNCHRONOUS FUTURE[T] PRIMITIVES (FAN-OUT + WORK-HELPING AWAIT) ---

typedef struct {
  vira_task *task;
  void *env;
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
    // finish before freeing
    if (f->task != NULL) {
      vira_scheduler_await(f->task);
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

static inline void vira_eprintln(vira_string s) {
  if (s.data != NULL && s.len > 0) {
    fwrite(s.data, 1, (size_t)s.len, stderr);
  }
  fputc('\n', stderr);
}

static inline void vira_eprintln_temp(vira_string s) {
  vira_eprintln(s);
  if (!s.is_lit && s.data != NULL) {
    free((void *)s.data);
    _vira_record_free();
  }
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
  char path_buf[1024];
  int len = path.len < 1023 ? path.len : 1023;
  if (path.data != NULL && len > 0) {
    memcpy(path_buf, path.data, len);
  }
  path_buf[len] = '\0';
  FILE *f = fopen(path_buf, "rb");
  if (f) {
    fclose(f);
    return true;
  }
  return false;
}

static inline Result_string vira_read_file(vira_string path) {
  char path_buf[1024];
  int len = path.len < 1023 ? path.len : 1023;
  if (path.data != NULL && len > 0) {
    memcpy(path_buf, path.data, len);
  }
  path_buf[len] = '\0';

  FILE *f = fopen(path_buf, "rb");
  if (!f) {
    return (Result_string){
        .is_err = true,
        .err = vira_str_format("failed to open file '%s'", path_buf)};
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 0) {
    fclose(f);
    return (Result_string){
        .is_err = true,
        .err = vira_str_format("failed to seek in '%s'", path_buf)};
  }
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
  char path_buf[1024];
  int len = path.len < 1023 ? path.len : 1023;
  if (path.data != NULL && len > 0) {
    memcpy(path_buf, path.data, len);
  }
  path_buf[len] = '\0';

  FILE *f = fopen(path_buf, "wb");
  if (!f) {
    return (Result_bool){.is_err = true,
                         .err = vira_str_format(
                             "failed to open file '%s' for writing", path_buf)};
  }
  if (content.len > 0 && content.data != NULL) {
    size_t written = fwrite(content.data, 1, (size_t)content.len, f);
    if (written != (size_t)content.len) {
      fclose(f);
      return (Result_bool){
          .is_err = true,
          .err = vira_str_format("failed to write complete content to '%s'",
                                 path_buf)};
    }
  }
  fclose(f);
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_bool vira_append_file(vira_string path,
                                           vira_string content) {
  char path_buf[1024];
  int len = path.len < 1023 ? path.len : 1023;
  if (path.data != NULL && len > 0) {
    memcpy(path_buf, path.data, len);
  }
  path_buf[len] = '\0';

  FILE *f = fopen(path_buf, "ab");
  if (!f) {
    return (Result_bool){
        .is_err = true,
        .err = vira_str_format("failed to open file '%s' for appending",
                               path_buf)};
  }
  if (content.len > 0 && content.data != NULL) {
    size_t written = fwrite(content.data, 1, (size_t)content.len, f);
    if (written != (size_t)content.len) {
      fclose(f);
      return (Result_bool){
          .is_err = true,
          .err = vira_str_format("failed to append complete content to '%s'",
                                 path_buf)};
    }
  }
  fclose(f);
  return (Result_bool){.is_err = false, .val = true};
}

static inline Result_bool vira_remove_file(vira_string path) {
  char path_buf[1024];
  int len = path.len < 1023 ? path.len : 1023;
  if (path.data != NULL && len > 0) {
    memcpy(path_buf, path.data, len);
  }
  path_buf[len] = '\0';

  if (remove(path_buf) == 0) {
    return (Result_bool){.is_err = false, .val = true};
  }
  return (Result_bool){
      .is_err = true,
      .err = vira_str_format("failed to remove file '%s'", path_buf)};
}

static inline int64_t vira_io_file_open(vira_string path, vira_string mode) {
  if (path.len == 0 || path.data == NULL || mode.len == 0 ||
      mode.data == NULL) {
    return 0;
  }
  char path_buf[1024];
  int32_t plen = path.len < 1023 ? path.len : 1023;
  memcpy(path_buf, path.data, (size_t)plen);
  path_buf[plen] = '\0';

  char mode_buf[16];
  int32_t mlen = mode.len < 15 ? mode.len : 15;
  memcpy(mode_buf, mode.data, (size_t)mlen);
  mode_buf[mlen] = '\0';

  FILE *f = fopen(path_buf, mode_buf);
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
  char path_buf[1024];
  int len = path.len < 1023 ? path.len : 1023;
  if (path.data != NULL && len > 0) {
    memcpy(path_buf, path.data, len);
  }
  path_buf[len] = '\0';

  FILE *f = fopen(path_buf, "rb");
  if (!f) {
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format("could not open weight file '%s'", path_buf),
        .val = (vira_tensor_f32){0}};
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
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format(
            "weight file '%s' size mismatch: expected %zu floats, read %zu",
            path_buf, expected, read_count),
        .val = (vira_tensor_f32){0}};
  }

  return (Result_tensor_f32){
      .is_err = false, .err = vira_str_lit("", 0), .val = t};
}

static inline Result_bool vira_tensor_save_bin(vira_tensor_f32 t,
                                               vira_string path) {
  char path_buf[1024];
  int len = path.len < 1023 ? path.len : 1023;
  if (path.data != NULL && len > 0) {
    memcpy(path_buf, path.data, len);
  }
  path_buf[len] = '\0';

  FILE *f = fopen(path_buf, "wb");
  if (!f) {
    return (Result_bool){
        .is_err = true,
        .err = vira_str_format("failed to open file '%s' for writing weights",
                               path_buf),
        .val = false};
  }

  size_t total = (size_t)t.rows * (size_t)t.cols;
  size_t written = fwrite(t.data, sizeof(vira_f32), total, f);
  fclose(f);

  if (written != total) {
    return (Result_bool){
        .is_err = true,
        .err = vira_str_format("failed to write complete weights to '%s'",
                               path_buf),
        .val = false};
  }

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

  char host_buf[256];
  int hlen = host.len < 255 ? host.len : 255;
  if (host.data && hlen > 0) {
    memcpy(host_buf, host.data, hlen);
    host_buf[hlen] = '\0';
  } else {
    host_buf[0] = '\0';
  }

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL, *p = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  const char *node =
      (hlen > 0 && strcmp(host_buf, "0.0.0.0") != 0) ? host_buf : NULL;
  if (getaddrinfo(node, port_buf, &hints, &res) != 0) {
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

  char host_buf[256];
  int hlen = host.len < 255 ? host.len : 255;
  if (host.data && hlen > 0) {
    memcpy(host_buf, host.data, hlen);
    host_buf[hlen] = '\0';
  } else {
    host_buf[0] = '\0';
  }

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL, *p = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(host_buf, port_buf, &hints, &res) != 0) {
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

  char host_buf[256];
  int hlen = host.len < 255 ? host.len : 255;
  if (host.data && hlen > 0) {
    memcpy(host_buf, host.data, hlen);
    host_buf[hlen] = '\0';
  } else {
    host_buf[0] = '\0';
  }

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL, *p = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_flags = AI_PASSIVE;

  const char *node =
      (hlen > 0 && strcmp(host_buf, "0.0.0.0") != 0) ? host_buf : NULL;
  if (getaddrinfo(node, port_buf, &hints, &res) != 0) {
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

  char host_buf[256];
  int hlen = host.len < 255 ? host.len : 255;
  if (host.data && hlen > 0) {
    memcpy(host_buf, host.data, hlen);
    host_buf[hlen] = '\0';
  } else {
    host_buf[0] = '\0';
  }

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;

  if (getaddrinfo(host_buf, port_buf, &hints, &res) != 0 || !res) {
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

  char ctype_buf[128];
  int ctype_len = content_type.len < 127 ? content_type.len : 127;
  if (content_type.data && ctype_len > 0) {
    memcpy(ctype_buf, content_type.data, ctype_len);
  }
  ctype_buf[ctype_len] = '\0';

  int header_est = 256 + body.len;
  char *res = (char *)malloc(header_est);
  if (!res)
    return vira_str_lit("", 0);
  _vira_record_alloc();

  int n = snprintf(res, header_est,
                   "HTTP/1.1 %lld %s\r\n"
                   "Content-Type: %s\r\n"
                   "Content-Length: %d\r\n"
                   "Connection: close\r\n"
                   "\r\n",
                   (long long)status_code, status_text, ctype_buf, body.len);

  if (body.len > 0 && body.data) {
    memcpy(res + n, body.data, body.len);
    n += body.len;
  }
  res[n] = '\0';
  return (vira_string){.data = (const uint8_t *)res, .len = n, .is_lit = false};
}

// ============================================================================
// 13. FAST JSON FIELD PARSER
// ============================================================================

static inline const char *_vira_memmem(const char *haystack, size_t hlen,
                                       const char *needle, size_t nlen) {
  if (nlen == 0)
    return haystack;
  if (hlen < nlen || !haystack || !needle)
    return NULL;
  for (size_t i = 0; i <= hlen - nlen; i++) {
    if (haystack[i] == needle[0] && memcmp(haystack + i, needle, nlen) == 0) {
      return haystack + i;
    }
  }
  return NULL;
}

static inline vira_string vira_json_get_str(vira_string json, vira_string key) {
  if (json.data == NULL || json.len <= 0 || key.data == NULL || key.len <= 0) {
    return vira_str_lit("", 0);
  }
  char search_quote[260];
  int klen = key.len < 250 ? key.len : 250;
  int sq_len = snprintf(search_quote, sizeof(search_quote), "\"%.*s\"", klen,
                        (const char *)key.data);
  if (sq_len <= 0)
    return vira_str_lit("", 0);

  const char *hay = (const char *)json.data;
  size_t hlen = (size_t)json.len;

  const char *found = _vira_memmem(hay, hlen, search_quote, (size_t)sq_len);
  if (!found)
    return vira_str_lit("", 0);

  size_t offset = (size_t)(found - hay) + (size_t)sq_len;
  if (offset >= hlen)
    return vira_str_lit("", 0);

  const char *colon = (const char *)memchr(hay + offset, ':', hlen - offset);
  if (!colon)
    return vira_str_lit("", 0);

  offset = (size_t)(colon - hay) + 1;
  if (offset >= hlen)
    return vira_str_lit("", 0);

  const char *val_start =
      (const char *)memchr(hay + offset, '"', hlen - offset);
  if (!val_start)
    return vira_str_lit("", 0);
  val_start++;

  offset = (size_t)(val_start - hay);
  if (offset >= hlen)
    return vira_str_lit("", 0);

  const char *val_end =
      (const char *)memchr(hay + offset, '"', hlen - offset);
  if (!val_end)
    return vira_str_lit("", 0);

  int val_len = (int)(val_end - val_start);
  char *out = (char *)malloc((size_t)val_len + 1);
  if (!out)
    return vira_str_lit("", 0);
  _vira_record_alloc();
  memcpy(out, val_start, (size_t)val_len);
  out[val_len] = '\0';
  return (vira_string){
      .data = (const uint8_t *)out, .len = val_len, .is_lit = false};
}

static inline int64_t vira_json_get_int(vira_string json, vira_string key) {
  if (json.data == NULL || json.len <= 0 || key.data == NULL || key.len <= 0) {
    return 0;
  }
  char search_quote[260];
  int klen = key.len < 250 ? key.len : 250;
  int sq_len = snprintf(search_quote, sizeof(search_quote), "\"%.*s\"", klen,
                        (const char *)key.data);
  if (sq_len <= 0)
    return 0;

  const char *hay = (const char *)json.data;
  size_t hlen = (size_t)json.len;

  const char *found = _vira_memmem(hay, hlen, search_quote, (size_t)sq_len);
  if (!found)
    return 0;

  size_t offset = (size_t)(found - hay) + (size_t)sq_len;
  if (offset >= hlen)
    return 0;

  const char *colon = (const char *)memchr(hay + offset, ':', hlen - offset);
  if (!colon)
    return 0;
  colon++;

  while (colon < hay + hlen && (*colon == ' ' || *colon == '\t' ||
                                *colon == '\r' || *colon == '\n')) {
    colon++;
  }
  if (colon >= hay + hlen)
    return 0;

  char num_buf[64];
  int nlen = 0;
  while (colon + nlen < hay + hlen && nlen < 63) {
    char c = colon[nlen];
    if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
      num_buf[nlen] = c;
      nlen++;
    } else {
      break;
    }
  }
  num_buf[nlen] = '\0';
  return (int64_t)strtoll(num_buf, NULL, 10);
}

static inline float vira_json_get_f32(vira_string json, vira_string key) {
  if (json.data == NULL || json.len <= 0 || key.data == NULL || key.len <= 0) {
    return 0.0f;
  }
  char search_quote[260];
  int klen = key.len < 250 ? key.len : 250;
  int sq_len = snprintf(search_quote, sizeof(search_quote), "\"%.*s\"", klen,
                        (const char *)key.data);
  if (sq_len <= 0)
    return 0.0f;

  const char *hay = (const char *)json.data;
  size_t hlen = (size_t)json.len;

  const char *found = _vira_memmem(hay, hlen, search_quote, (size_t)sq_len);
  if (!found)
    return 0.0f;

  size_t offset = (size_t)(found - hay) + (size_t)sq_len;
  if (offset >= hlen)
    return 0.0f;

  const char *colon = (const char *)memchr(hay + offset, ':', hlen - offset);
  if (!colon)
    return 0.0f;
  colon++;

  while (colon < hay + hlen && (*colon == ' ' || *colon == '\t' ||
                                *colon == '\r' || *colon == '\n')) {
    colon++;
  }
  if (colon >= hay + hlen)
    return 0.0f;

  char num_buf[64];
  int nlen = 0;
  while (colon + nlen < hay + hlen && nlen < 63) {
    char c = colon[nlen];
    if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' ||
        c == 'e' || c == 'E') {
      num_buf[nlen] = c;
      nlen++;
    } else {
      break;
    }
  }
  num_buf[nlen] = '\0';
  return (float)strtod(num_buf, NULL);
}

// ============================================================================
// 14. NATIVE SAFETENSORS ZERO-COPY PARSER & LOADER
// ============================================================================

static inline Result_tensor_f32 vira_safetensors_load(vira_string path,
                                                      vira_string tensor_name) {
  char path_buf[1024];
  int len = path.len < 1023 ? path.len : 1023;
  if (path.data != NULL && len > 0) {
    memcpy(path_buf, path.data, len);
  }
  path_buf[len] = '\0';

  char name_buf[256];
  int nlen = tensor_name.len < 255 ? tensor_name.len : 255;
  if (tensor_name.data != NULL && nlen > 0) {
    memcpy(name_buf, tensor_name.data, nlen);
  }
  name_buf[nlen] = '\0';

  FILE *f = fopen(path_buf, "rb");
  if (!f) {
    return (Result_tensor_f32){
        .is_err = true,
        .err =
            vira_str_format("failed to open safetensors file '%s'", path_buf),
        .val = (vira_tensor_f32){0}};
  }

  uint64_t header_len = 0;
  if (fread(&header_len, sizeof(uint64_t), 1, f) != 1 ||
      header_len > 100 * 1024 * 1024) {
    fclose(f);
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format("corrupt safetensors header in '%s'", path_buf),
        .val = (vira_tensor_f32){0}};
  }

  char *header_json = (char *)malloc(header_len + 1);
  if (!header_json) {
    fclose(f);
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
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format("incomplete read of safetensors header in '%s'",
                               path_buf),
        .val = (vira_tensor_f32){0}};
  }
  header_json[header_len] = '\0';

  char tensor_key[300];
  snprintf(tensor_key, sizeof(tensor_key), "\"%s\"", name_buf);
  char *t_entry = strstr(header_json, tensor_key);
  if (!t_entry) {
    free(header_json);
    fclose(f);
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format("tensor '%s' not found in '%s'", name_buf,
                               path_buf),
        .val = (vira_tensor_f32){0}};
  }

  int rows = 1, cols = 1;
  char *shape_ptr = strstr(t_entry, "\"shape\"");
  if (shape_ptr) {
    char *lbracket = strchr(shape_ptr, '[');
    if (lbracket) {
      rows = (int)strtol(lbracket + 1, &lbracket, 10);
      while (*lbracket == ' ' || *lbracket == ',')
        lbracket++;
      if (*lbracket != ']' && *lbracket != '\0') {
        cols = (int)strtol(lbracket, NULL, 10);
      }
    }
  }
  if (rows <= 0)
    rows = 1;
  if (cols <= 0)
    cols = 1;

  uint64_t begin_off = 0;
  char *off_ptr = strstr(t_entry, "\"data_offsets\"");
  if (off_ptr) {
    char *lbracket = strchr(off_ptr, '[');
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

  vira_tensor_f32 t = vira_tensor_f32_alloc(rows, cols);
  size_t expected_bytes = (size_t)rows * (size_t)cols * sizeof(float);
  size_t read_bytes = fread(t.data, 1, expected_bytes, f);
  fclose(f);

  if (read_bytes != expected_bytes) {
    vira_tensor_f32_free(&t);
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format(
            "tensor '%s' data truncated: expected %zu bytes, read %zu",
            name_buf, expected_bytes, read_bytes),
        .val = (vira_tensor_f32){0}};
  }

  return (Result_tensor_f32){
      .is_err = false, .err = vira_str_lit("", 0), .val = t};
}

// ============================================================================
// 15. HARDWARE SIMD AUTOGRAD BACKWARD OPS & OPTIMIZERS
// ============================================================================

static inline vira_tensor_f32 vira_tensor_transpose(vira_tensor_f32 a) {
  vira_tensor_f32 out = vira_tensor_f32_alloc(a.cols, a.rows);
  for (int r = 0; r < a.rows; ++r) {
    for (int c = 0; c < a.cols; ++c) {
      out.data[c * out.cols + r] = a.data[r * a.cols + c];
    }
  }
  return out;
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
  vira_tensor_f32 out = vira_tensor_f32_alloc(input.rows, input.cols);
  int total = input.rows * input.cols;
  int i = 0;

#if defined(__AVX2__)
  __m256 zero = _mm256_setzero_ps();
  for (; i + 7 < total; i += 8) {
    __m256 in_v = _mm256_load_ps(&input.data[i]);
    __m256 grad_v = _mm256_load_ps(&grad_out.data[i]);
    __m256 mask = _mm256_cmp_ps(in_v, zero, _CMP_GT_OQ);
    __m256 res_v = _mm256_and_ps(mask, grad_v);
    _mm256_store_ps(&out.data[i], res_v);
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
    __m256 p_v = _mm256_load_ps(&param->data[i]);
    __m256 g_v = _mm256_load_ps(&grad.data[i]);
    __m256 step = _mm256_mul_ps(lr_v, g_v);
    __m256 updated = _mm256_sub_ps(p_v, step);
    _mm256_store_ps(&param->data[i], updated);
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
  char path_buf[1024];
  int len = path.len < 1023 ? path.len : 1023;
  if (path.data && len > 0)
    memcpy(path_buf, path.data, len);
  path_buf[len] = '\0';

  char name_buf[256];
  int nlen = tensor_name.len < 255 ? tensor_name.len : 255;
  if (tensor_name.data && nlen > 0)
    memcpy(name_buf, tensor_name.data, nlen);
  name_buf[nlen] = '\0';

  FILE *f = fopen(path_buf, "rb");
  if (!f) {
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format("failed to open GGUF file '%s'", path_buf),
        .val = (vira_tensor_f32){0}};
  }

  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t tensor_count = 0;
  uint64_t kv_count = 0;

  if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) {
    fclose(f);
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format("invalid GGUF magic in '%s'", path_buf),
        .val = (vira_tensor_f32){0}};
  }
  if (fread(&version, 4, 1, f) != 1 || (version != 2 && version != 3)) {
    fclose(f);
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format("unsupported GGUF version %u in '%s'", version,
                               path_buf),
        .val = (vira_tensor_f32){0}};
  }
  if (fread(&tensor_count, 8, 1, f) != 1 || fread(&kv_count, 8, 1, f) != 1) {
    fclose(f);
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format("corrupt GGUF header in '%s'", path_buf),
        .val = (vira_tensor_f32){0}};
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
  int32_t t_rows = 1, t_cols = 1;

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

    if (strcmp(cur_name, name_buf) == 0) {
      found = true;
      matched_type = dtype;
      matched_offset = offset;
      if (n_dims == 1) {
        t_rows = 1;
        t_cols = (int32_t)dims[0];
      } else {
        t_cols = (int32_t)dims[0];
        t_rows = (int32_t)dims[1];
      }
    }
  }

  if (!found) {
    fclose(f);
    return (Result_tensor_f32){
        .is_err = true,
        .err = vira_str_format("tensor '%s' not found in GGUF file '%s'",
                               name_buf, path_buf),
        .val = (vira_tensor_f32){0}};
  }

  int64_t cur_pos = ftell(f);
  int64_t pad = (alignment - (cur_pos % alignment)) % alignment;
  int64_t data_start = cur_pos + pad;

  int64_t tensor_file_pos = data_start + matched_offset;
#ifdef _WIN32
  _fseeki64(f, tensor_file_pos, SEEK_SET);
#else
  fseeko(f, tensor_file_pos, SEEK_SET);
#endif

  vira_tensor_f32 t = vira_tensor_f32_alloc(t_rows, t_cols);
  int32_t total = t_rows * t_cols;

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
  vira_tensor_f32 out = vira_tensor_f32_alloc(x.rows, x.cols);
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
  vira_tensor_f32 out = vira_tensor_f32_alloc(q.rows, q.cols);
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
  vira_tensor_f32 out = vira_tensor_f32_alloc(x.rows, x.cols);
  int32_t total = x.rows * x.cols;
  for (int32_t i = 0; i < total; ++i) {
    float v = x.data[i];
    out.data[i] = v / (1.0f + expf(-v));
  }
  return out;
}

static inline vira_tensor_f32 vira_tensor_elemwise_mul(vira_tensor_f32 a,
                                                       vira_tensor_f32 b) {
  if (a.rows != b.rows || a.cols != b.cols) {
    fprintf(stderr,
            "vex tensor error: elemwise_mul shape mismatch (%dx%d) * (%dx%d)\n",
            a.rows, a.cols, b.rows, b.cols);
    exit(1);
  }
  vira_tensor_f32 out = vira_tensor_f32_alloc(a.rows, a.cols);
  int32_t total = a.rows * a.cols;
  int32_t i = 0;

#if defined(__AVX2__)
  for (; i + 7 < total; i += 8) {
    __m256 va = _mm256_loadu_ps(&a.data[i]);
    __m256 vb = _mm256_loadu_ps(&b.data[i]);
    _mm256_storeu_ps(&out.data[i], _mm256_mul_ps(va, vb));
  }
#endif
  for (; i < total; ++i) {
    out.data[i] = a.data[i] * b.data[i];
  }
  return out;
}

static inline vira_tensor_f32 vira_tensor_softmax(vira_tensor_f32 logits) {
  vira_tensor_f32 out = vira_tensor_f32_alloc(logits.rows, logits.cols);
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
            "vex tensor error: embedding index %lld out of range [0, %d)\n",
            (long long)token_id, table.rows);
    exit(1);
  }
  vira_tensor_f32 out = vira_tensor_f32_alloc(1, table.cols);
  memcpy(out.data, &table.data[token_id * table.cols],
         (size_t)table.cols * sizeof(float));
  return out;
}

static inline void vira_report_mem_stats(void) {
  _vira_flush_tls_mem_stats();
  vira_scheduler_shutdown();
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
}

#include "vira_gui.h"

#endif // VIRA_RT_H
