#ifndef VEX_RT_H
#define VEX_RT_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET vex_socket_t;
#define VEX_INVALID_SOCKET INVALID_SOCKET
#define VEX_SOCKET_ERROR SOCKET_ERROR
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
typedef int vex_socket_t;
#define VEX_INVALID_SOCKET -1
#define VEX_SOCKET_ERROR -1
#define closesocket close
#endif

// --- HARDWARE SIMD INTRINSIC HEADERS ---
#if defined(__AVX2__) || defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

// --- THREAD-SAFE TELEMETRY COUNTERS FOR ZERO-LEAK VERIFICATION ---
static volatile int64_t _vex_active_heap_allocs = 0;
static volatile int64_t _vex_total_heap_frees   = 0;
static volatile int64_t _vex_total_tasks_run    = 0;
static volatile int64_t _vex_total_work_steals  = 0;

static inline void _vex_record_alloc(void) {
    __atomic_add_fetch(&_vex_active_heap_allocs, 1, __ATOMIC_RELAXED);
}

static inline void _vex_record_free(void) {
    __atomic_sub_fetch(&_vex_active_heap_allocs, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&_vex_total_heap_frees, 1, __ATOMIC_RELAXED);
}

// ============================================================================
// 1. VEX STRING (16 BYTES: ZERO-ALLOC LITERALS + HEAP STRINGS)
// ============================================================================

typedef struct {
    const uint8_t* data;
    int32_t        len;
    bool           is_lit;
} vex_string;

static inline vex_string vex_str_lit(const char* cstr, int32_t len) {
    return (vex_string){ .data = (const uint8_t*)cstr, .len = len, .is_lit = true };
}

static inline bool vex_str_eq(vex_string a, vex_string b) {
    if (a.len != b.len) return false;
    if (a.len == 0) return true;
    if (a.data == b.data) return true;
    if (a.data == NULL || b.data == NULL) return false;
    return memcmp(a.data, b.data, (size_t)a.len) == 0;
}

static inline vex_string vex_str_format(const char* fmt, ...) {
    va_list args, args_copy;
    va_start(args, fmt);
    va_copy(args_copy, args);
    int32_t needed = vsnprintf(NULL, 0, fmt, args);
    va_end(args);

    if (needed <= 0) {
        va_end(args_copy);
        return vex_str_lit("", 0);
    }

    char* buf = (char*)malloc((size_t)needed + 1);
    if (!buf) {
        fprintf(stderr, "vex panic: out of memory in vex_str_format\n");
        exit(1);
    }
    vsnprintf(buf, (size_t)needed + 1, fmt, args_copy);
    va_end(args_copy);

    _vex_record_alloc();
    return (vex_string){ .data = (const uint8_t*)buf, .len = needed, .is_lit = false };
}

static inline vex_string vex_str_clone(vex_string s) {
    if (s.is_lit || s.data == NULL || s.len == 0) {
        return s;
    }
    uint8_t* copy = (uint8_t*)malloc((size_t)s.len + 1);
    memcpy(copy, s.data, (size_t)s.len);
    copy[s.len] = '\0';
    _vex_record_alloc();
    return (vex_string){ .data = copy, .len = s.len, .is_lit = false };
}

static inline void vex_str_free(vex_string* s) {
    if (s != NULL && !s->is_lit && s->data != NULL) {
        free((void*)s->data);
        s->data   = NULL;
        s->len    = 0;
        s->is_lit = true;
        _vex_record_free();
    }
}

// ============================================================================
// 2. 64-BYTE ALIGNED BUMP-POINTER ARENA ALLOCATOR (`arena { ... }`)
// ============================================================================

typedef struct {
    uint8_t* buf;
    size_t   cap;
    size_t   offset;
} vex_arena;

static inline vex_arena vex_arena_init(size_t capacity) {
    uint8_t* slab = (uint8_t*)malloc(capacity + 64);
    if (!slab) {
        fprintf(stderr, "vex panic: failed to allocate arena slab (%zu bytes)\n", capacity);
        exit(1);
    }
    _vex_record_alloc();
    return (vex_arena){ .buf = slab, .cap = capacity, .offset = 0 };
}

static inline void* vex_arena_alloc(vex_arena* a, size_t size) {
    // 64-byte alignment guarantees AVX-512 / AVX2 / NEON aligned vector loads
    uintptr_t base    = (uintptr_t)a->buf + a->offset;
    uintptr_t aligned = (base + 63) & ~(uintptr_t)63;
    size_t    pad     = (size_t)(aligned - base);
    size_t    total   = pad + ((size + 63) & ~(size_t)63);

    if (a->offset + total > a->cap) {
        fprintf(stderr, "vex panic: arena slab exhausted (%zu / %zu bytes)\n", a->offset + total, a->cap);
        exit(1);
    }
    a->offset += total;
    return (void*)aligned;
}

static inline vex_string vex_arena_str_format(vex_arena* a, const char* fmt, ...) {
    va_list args, args_copy;
    va_start(args, fmt);
    va_copy(args_copy, args);
    int32_t needed = vsnprintf(NULL, 0, fmt, args);
    va_end(args);

    char* buf = (char*)vex_arena_alloc(a, (size_t)needed + 1);
    vsnprintf(buf, (size_t)needed + 1, fmt, args_copy);
    va_end(args_copy);

    return (vex_string){ .data = (const uint8_t*)buf, .len = needed, .is_lit = true };
}

static inline void vex_arena_destroy(vex_arena* a) {
    if (a != NULL && a->buf != NULL) {
        free(a->buf);
        a->buf    = NULL;
        a->cap    = 0;
        a->offset = 0;
        _vex_record_free();
    }
}

// ============================================================================
// 3. AI TENSORS + HARDWARE SIMD (`AVX2` / `SSE2` / `ARM NEON`) `@` MATMUL
// ============================================================================

typedef float    vex_f32;
typedef uint16_t vex_bf16;

static inline const char* vex_simd_backend_name(void) {
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
    vex_f32* data;      // 64-byte cache-line aligned pointer
    void*    raw_alloc; // underlying pointer for free()
    int32_t  rows;
    int32_t  cols;
    bool     is_arena;
} vex_tensor_f32;

static inline vex_tensor_f32 vex_tensor_f32_alloc(int32_t rows, int32_t cols) {
    size_t bytes = (size_t)rows * (size_t)cols * sizeof(vex_f32);
    void*  raw   = malloc(bytes + 64);
    if (!raw) {
        fprintf(stderr, "vex panic: tensor allocation failed (%d x %d)\n", rows, cols);
        exit(1);
    }
    uintptr_t aligned = ((uintptr_t)raw + 63) & ~(uintptr_t)63;
    vex_f32*  data    = (vex_f32*)aligned;
    memset(data, 0, bytes);
    _vex_record_alloc();
    return (vex_tensor_f32){
        .data = data, .raw_alloc = raw, .rows = rows, .cols = cols, .is_arena = false
    };
}

static inline vex_tensor_f32 vex_tensor_f32_arena_alloc(vex_arena* a, int32_t rows, int32_t cols) {
    size_t   bytes = (size_t)rows * (size_t)cols * sizeof(vex_f32);
    vex_f32* data  = (vex_f32*)vex_arena_alloc(a, bytes);
    memset(data, 0, bytes);
    return (vex_tensor_f32){
        .data = data, .raw_alloc = NULL, .rows = rows, .cols = cols, .is_arena = true
    };
}

static inline vex_tensor_f32 vex_tensor_f32_fill(vex_arena* opt_arena, int32_t rows, int32_t cols, vex_f32 val) {
    vex_tensor_f32 t = opt_arena
        ? vex_tensor_f32_arena_alloc(opt_arena, rows, cols)
        : vex_tensor_f32_alloc(rows, cols);
    int32_t total = rows * cols;
    int32_t i     = 0;

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
static inline vex_tensor_f32 vex_tensor_matmul_f32(vex_arena* opt_arena, vex_tensor_f32 a, vex_tensor_f32 b) {
    if (a.cols != b.rows) {
        fprintf(stderr, "vex tensor error: matmul '@' shape mismatch (%dx%d) @ (%dx%d)\n",
                a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }
    vex_tensor_f32 out = opt_arena
        ? vex_tensor_f32_arena_alloc(opt_arena, a.rows, b.cols)
        : vex_tensor_f32_alloc(a.rows, b.cols);

    const int32_t M = a.rows;
    const int32_t K = a.cols;
    const int32_t N = b.cols;

    for (int32_t i = 0; i < M; i++) {
        vex_f32* c_row = &out.data[i * N];
        for (int32_t k = 0; k < K; k++) {
            const vex_f32  a_ik  = a.data[i * K + k];
            const vex_f32* b_row = &b.data[k * N];
            int32_t        j     = 0;

#if defined(__AVX2__)
            __m256 va = _mm256_set1_ps(a_ik);
            for (; j + 8 <= N; j += 8) {
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
            for (; j + 8 <= N; j += 8) {
                float32x4_t vc0 = vld1q_f32(&c_row[j]);
                float32x4_t vc1 = vld1q_f32(&c_row[j + 4]);
                float32x4_t vb0 = vld1q_f32(&b_row[j]);
                float32x4_t vb1 = vld1q_f32(&b_row[j + 4]);
                vc0 = vfmaq_n_f32(vc0, vb0, a_ik);
                vc1 = vfmaq_n_f32(vc1, vb1, a_ik);
                vst1q_f32(&c_row[j], vc0);
                vst1q_f32(&c_row[j + 4], vc1);
            }
            for (; j + 4 <= N; j += 4) {
                float32x4_t vc0 = vld1q_f32(&c_row[j]);
                float32x4_t vb0 = vld1q_f32(&b_row[j]);
                vc0 = vfmaq_n_f32(vc0, vb0, a_ik);
                vst1q_f32(&c_row[j], vc0);
            }
#elif defined(__SSE2__)
            __m128 va = _mm_set1_ps(a_ik);
            for (; j + 4 <= N; j += 4) {
                __m128 vc = _mm_loadu_ps(&c_row[j]);
                __m128 vb = _mm_loadu_ps(&b_row[j]);
                vc = _mm_add_ps(_mm_mul_ps(va, vb), vc);
                _mm_storeu_ps(&c_row[j], vc);
            }
#endif
            for (; j < N; j++) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
    return out;
}

// SIMD-accelerated tensor sum reduction
static inline vex_f32 vex_tensor_sum(vex_tensor_f32 t) {
    int32_t total = t.rows * t.cols;
    int32_t i     = 0;
    vex_f32 sum   = 0.0f;

#if defined(__AVX2__)
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= total; i += 8) {
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(&t.data[i]));
    }
    __m128 lo   = _mm256_castps256_ps128(vsum);
    __m128 hi   = _mm256_extractf128_ps(vsum, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128        = _mm_hadd_ps(s128, s128);
    s128        = _mm_hadd_ps(s128, s128);
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
static inline vex_tensor_f32 vex_tensor_relu(vex_arena* opt_arena, vex_tensor_f32 t) {
    vex_tensor_f32 out = opt_arena
        ? vex_tensor_f32_arena_alloc(opt_arena, t.rows, t.cols)
        : vex_tensor_f32_alloc(t.rows, t.cols);
    int32_t total = t.rows * t.cols;
    int32_t i     = 0;

#if defined(__AVX2__)
    __m256 vzero = _mm256_setzero_ps();
    for (; i + 8 <= total; i += 8) {
        __m256 v  = _mm256_loadu_ps(&t.data[i]);
        __m256 vr = _mm256_max_ps(v, vzero);
        _mm256_storeu_ps(&out.data[i], vr);
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t vzero = vdupq_n_f32(0.0f);
    for (; i + 4 <= total; i += 4) {
        float32x4_t v  = vld1q_f32(&t.data[i]);
        float32x4_t vr = vmaxq_f32(v, vzero);
        vst1q_f32(&out.data[i], vr);
    }
#elif defined(__SSE2__)
    __m128 vzero = _mm_setzero_ps();
    for (; i + 4 <= total; i += 4) {
        __m128 v  = _mm_loadu_ps(&t.data[i]);
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
static inline vex_tensor_f32 vex_tensor_add(vex_arena* opt_arena, vex_tensor_f32 a, vex_tensor_f32 b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "vex tensor error: add shape mismatch (%dx%d) + (%dx%d)\n",
                a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }
    vex_tensor_f32 out = opt_arena
        ? vex_tensor_f32_arena_alloc(opt_arena, a.rows, a.cols)
        : vex_tensor_f32_alloc(a.rows, a.cols);
    int32_t total = a.rows * a.cols;
    int32_t i     = 0;

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
static inline vex_tensor_f32 vex_tensor_sub(vex_arena* opt_arena, vex_tensor_f32 a, vex_tensor_f32 b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "vex tensor error: sub shape mismatch (%dx%d) - (%dx%d)\n",
                a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }
    vex_tensor_f32 out = opt_arena
        ? vex_tensor_f32_arena_alloc(opt_arena, a.rows, a.cols)
        : vex_tensor_f32_alloc(a.rows, a.cols);
    int32_t total = a.rows * a.cols;
    int32_t i     = 0;

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
static inline vex_tensor_f32 vex_tensor_scale(vex_arena* opt_arena, vex_tensor_f32 t, vex_f32 scale) {
    vex_tensor_f32 out = opt_arena
        ? vex_tensor_f32_arena_alloc(opt_arena, t.rows, t.cols)
        : vex_tensor_f32_alloc(t.rows, t.cols);
    int32_t total = t.rows * t.cols;
    int32_t i     = 0;

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

static inline void vex_tensor_f32_free(vex_tensor_f32* t) {
    if (t != NULL && !t->is_arena && t->raw_alloc != NULL) {
        free(t->raw_alloc);
        t->raw_alloc = NULL;
        t->data      = NULL;
        _vex_record_free();
    }
}

// ============================================================================
// 4. MULTI-THREADED LOCK-FREE WORK-STEALING ACTOR POOL & SUPERVISOR
// ============================================================================

typedef enum {
    VEX_ACTOR_RUNNING = 0,
    VEX_ACTOR_FAULTED = 1,
    VEX_ACTOR_STOPPED = 2,
} vex_actor_state;

typedef struct {
    int32_t          id;
    const char*      name;
    vex_actor_state  state;
    int32_t          restarts;
    int32_t          max_restarts;
    vex_arena        isolated_heap;  // Wiped in O(1) if the actor panics
    jmp_buf          fault_boundary; // Thread-safe fault jump target when active
    vex_string       last_error;
    bool             in_actor;
    volatile uint8_t mailbox_lock;   // Ensures single-actor sequential state safety
    int32_t          last_worker_id; // Tracks which worker thread executed the actor
} vex_actor_ctx;

// Thread-Local Storage (TLS): Each worker pthread tracks its own active actor!
static __thread vex_actor_ctx* _vex_tls_actor     = NULL;
static __thread int32_t        _vex_tls_worker_id = -1;

static inline void vex_actor_ctx_init(vex_actor_ctx* ctx, int32_t id, const char* name, int32_t max_restarts) {
    ctx->id             = id;
    ctx->name           = name;
    ctx->state          = VEX_ACTOR_RUNNING;
    ctx->restarts       = 0;
    ctx->max_restarts   = max_restarts;
    ctx->isolated_heap  = vex_arena_init(256 * 1024); // 256 KB SIMD-aligned scratch arena per actor
    ctx->last_error     = vex_str_lit("", 0);
    ctx->in_actor       = false;
    ctx->mailbox_lock   = 0;
    ctx->last_worker_id = -1;
}

static inline void vex_actor_ctx_destroy(vex_actor_ctx* ctx) {
    vex_arena_destroy(&ctx->isolated_heap);
}

static inline void vex_panic(vex_string msg) {
    if (_vex_tls_actor != NULL && _vex_tls_actor->in_actor) {
        _vex_tls_actor->state      = VEX_ACTOR_FAULTED;
        _vex_tls_actor->last_error = msg;
        longjmp(_vex_tls_actor->fault_boundary, 1);
    }
    fprintf(stderr, "vex unrecoverable panic: %.*s\n", msg.len, (const char*)msg.data);
    exit(1);
}

// --- CHASE-LEV STYLE LOCK-FREE WORK-STEALING DEQUE ---
#define VEX_WS_CAP  256
#define VEX_WS_MASK (VEX_WS_CAP - 1)
#define VEX_MAX_WORKERS 8

typedef void (*vex_task_fn)(void* env);

typedef struct {
    vex_task_fn      fn;
    void*            env;
    volatile int32_t done; // 0 = pending, 1 = completed on worker pool
} vex_task;

typedef struct {
    int32_t          worker_id;
    pthread_t        thread;
    volatile int64_t top;    // Modified via CAS by thieves (FIFO steal)
    volatile int64_t bottom; // Modified by owner (LIFO push/pop)
    vex_task*        ring[VEX_WS_CAP];
} vex_ws_deque;

typedef struct {
    int32_t          num_workers;
    bool             initialized;
    volatile int32_t shutdown;
    volatile uint32_t rr_cursor;
    vex_ws_deque     workers[VEX_MAX_WORKERS];
} vex_scheduler;

static vex_scheduler _vex_sched = {0};

// Push task onto a deque's bottom
static inline bool vex_ws_push(vex_ws_deque* q, vex_task* t) {
    int64_t b = __atomic_load_n(&q->bottom, __ATOMIC_RELAXED);
    int64_t top = __atomic_load_n(&q->top, __ATOMIC_ACQUIRE);
    if (b - top >= VEX_WS_CAP - 1) {
        return false; // Queue full
    }
    __atomic_store_n(&q->ring[b & VEX_WS_MASK], t, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&q->bottom, b + 1, __ATOMIC_RELEASE);
    return true;
}

// Pop or steal a task from the top of any deque using lock-free CAS
static inline vex_task* vex_ws_steal(vex_ws_deque* q) {
    int64_t t = __atomic_load_n(&q->top, __ATOMIC_ACQUIRE);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    int64_t b = __atomic_load_n(&q->bottom, __ATOMIC_ACQUIRE);
    if (t >= b) {
        return NULL; // Empty
    }
    vex_task* task = __atomic_load_n(&q->ring[t & VEX_WS_MASK], __ATOMIC_RELAXED);
    if (!__atomic_compare_exchange_n(&q->top, &t, t + 1, false, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {
        return NULL; // Lost race to another thief
    }
    return task;
}

static void* _vex_worker_loop(void* arg) {
    vex_ws_deque* self = (vex_ws_deque*)arg;
    _vex_tls_worker_id = self->worker_id;

    while (!__atomic_load_n(&_vex_sched.shutdown, __ATOMIC_RELAXED)) {
        // 1. Try taking from own deque first
        vex_task* task = vex_ws_steal(self);

        // 2. If own deque is empty, steal from peer worker deques!
        if (task == NULL) {
            for (int32_t i = 1; i < _vex_sched.num_workers; i++) {
                int32_t victim_idx = (self->worker_id + i) % _vex_sched.num_workers;
                task = vex_ws_steal(&_vex_sched.workers[victim_idx]);
                if (task != NULL) {
                    __atomic_add_fetch(&_vex_total_work_steals, 1, __ATOMIC_RELAXED);
                    break;
                }
            }
        }

        if (task != NULL) {
            task->fn(task->env);
            __atomic_add_fetch(&_vex_total_tasks_run, 1, __ATOMIC_RELAXED);
            __atomic_store_n(&task->done, 1, __ATOMIC_RELEASE);
        } else {
            sched_yield();
        }
    }
    return NULL;
}

static inline void vex_scheduler_init(int32_t num_workers) {
    if (_vex_sched.initialized) return;
    if (num_workers <= 0 || num_workers > VEX_MAX_WORKERS) {
        num_workers = 4;
    }
    _vex_sched.num_workers = num_workers;
    _vex_sched.shutdown    = 0;
    _vex_sched.rr_cursor   = 0;
    _vex_sched.initialized = true;

    for (int32_t i = 0; i < num_workers; i++) {
        _vex_sched.workers[i].worker_id = i;
        _vex_sched.workers[i].top       = 0;
        _vex_sched.workers[i].bottom    = 0;
        pthread_create(&_vex_sched.workers[i].thread, NULL, _vex_worker_loop, &_vex_sched.workers[i]);
    }
}

static inline void vex_scheduler_shutdown(void) {
    if (!_vex_sched.initialized) return;
    __atomic_store_n(&_vex_sched.shutdown, 1, __ATOMIC_SEQ_CST);
    for (int32_t i = 0; i < _vex_sched.num_workers; i++) {
        pthread_join(_vex_sched.workers[i].thread, NULL);
    }
    _vex_sched.initialized = false;
}

// --- ASYNCHRONOUS FUTURE[T] PRIMITIVES (FAN-OUT + WORK-HELPING AWAIT) ---

typedef struct {
    vex_task* task;
    void*     env;
} vex_future;

// Pushes a heap-backed actor task onto a worker deque and returns immediately (non-blocking)
static inline void vex_scheduler_submit_async(vex_task* task) {
    if (!_vex_sched.initialized) {
        vex_scheduler_init(4);
    }
    __atomic_store_n(&task->done, 0, __ATOMIC_RELAXED);

    // Cluster bursts on Worker 0's deque first so idle peer threads steal tasks concurrently
    uint32_t seq       = __atomic_fetch_add(&_vex_sched.rr_cursor, 1, __ATOMIC_RELAXED);
    int32_t  queue_idx = (seq % 2 == 0) ? 0 : (int32_t)(seq % (uint32_t)_vex_sched.num_workers);

    while (!vex_ws_push(&_vex_sched.workers[queue_idx], task)) {
        sched_yield();
    }
}

// Waits for a future's task to complete; while waiting, the caller helps steal pending tasks!
static inline void vex_scheduler_await(vex_task* task) {
    if (task == NULL) return;
    while (__atomic_load_n(&task->done, __ATOMIC_ACQUIRE) == 0) {
        bool helped = false;
        for (int32_t i = 0; i < _vex_sched.num_workers; i++) {
            vex_task* stolen = vex_ws_steal(&_vex_sched.workers[i]);
            if (stolen != NULL) {
                __atomic_add_fetch(&_vex_total_work_steals, 1, __ATOMIC_RELAXED);
                stolen->fn(stolen->env);
                __atomic_add_fetch(&_vex_total_tasks_run, 1, __ATOMIC_RELAXED);
                __atomic_store_n(&stolen->done, 1, __ATOMIC_RELEASE);
                helped = true;
                break;
            }
        }
        if (!helped) {
            sched_yield();
        }
    }
}

// Synchronous helper (spawn + immediate await)
static inline void vex_scheduler_submit_and_wait(vex_task_fn fn, void* env) {
    if (!_vex_sched.initialized) {
        vex_scheduler_init(4);
    }
    vex_task task = { .fn = fn, .env = env, .done = 0 };
    vex_scheduler_submit_async(&task);
    vex_scheduler_await(&task);
}

// Idempotent scope-exit cleanup injected by `internal/autofree` for Future[T]
static inline void vex_future_free(vex_future* f) {
    if (f != NULL && f->env != NULL) {
        // If an early return skipped `.await()`, wait for the worker thread to finish before freeing
        if (f->task != NULL) {
            vex_scheduler_await(f->task);
        }
        free(f->env);
        f->env  = NULL;
        f->task = NULL;
        _vex_record_free();
    }
}

// ============================================================================
// 5. BUILTIN I/O & DIAGNOSTICS
// ============================================================================

static inline void vex_println(vex_string s) {
    if (s.data != NULL && s.len > 0) {
        fwrite(s.data, 1, (size_t)s.len, stdout);
    }
    fputc('\n', stdout);
    fflush(stdout);
}

static inline void vex_println_temp(vex_string s) {
    vex_println(s);
    if (!s.is_lit && s.data != NULL) {
        free((void*)s.data);
        _vex_record_free();
    }
}

static inline void vex_eprintln(vex_string s) {
    if (s.data != NULL && s.len > 0) {
        fwrite(s.data, 1, (size_t)s.len, stderr);
    }
    fputc('\n', stderr);
}

static inline void vex_eprintln_temp(vex_string s) {
    vex_eprintln(s);
    if (!s.is_lit && s.data != NULL) {
        free((void*)s.data);
        _vex_record_free();
    }
}

// ============================================================================
// 6. NATIVE FILE I/O & BUFFER OPERATIONS
// ============================================================================

#ifndef _Result_string_DEFINED
#define _Result_string_DEFINED
typedef struct {
    bool       is_err;
    vex_string err;
    vex_string val;
} Result_string;
#endif

#ifndef _Result_bool_DEFINED
#define _Result_bool_DEFINED
typedef struct {
    bool       is_err;
    vex_string err;
    bool       val;
} Result_bool;
#endif

static inline bool vex_file_exists(vex_string path) {
    char path_buf[1024];
    int len = path.len < 1023 ? path.len : 1023;
    if (path.data != NULL && len > 0) {
        memcpy(path_buf, path.data, len);
    }
    path_buf[len] = '\0';
    FILE* f = fopen(path_buf, "rb");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

static inline Result_string vex_read_file(vex_string path) {
    char path_buf[1024];
    int len = path.len < 1023 ? path.len : 1023;
    if (path.data != NULL && len > 0) {
        memcpy(path_buf, path.data, len);
    }
    path_buf[len] = '\0';

    FILE* f = fopen(path_buf, "rb");
    if (!f) {
        return (Result_string){ .is_err = true, .err = vex_str_format("failed to open file '%s'", path_buf) };
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        return (Result_string){ .is_err = true, .err = vex_str_format("failed to seek in '%s'", path_buf) };
    }
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return (Result_string){ .is_err = true, .err = vex_str_lit("out of memory reading file", 26) };
    }
    size_t read_bytes = fread(buf, 1, (size_t)sz, f);
    buf[read_bytes] = '\0';
    fclose(f);
    _vex_record_alloc();
    return (Result_string){
        .is_err = false,
        .val = (vex_string){ .data = (const uint8_t*)buf, .len = (int32_t)read_bytes, .is_lit = false }
    };
}

static inline Result_bool vex_write_file(vex_string path, vex_string content) {
    char path_buf[1024];
    int len = path.len < 1023 ? path.len : 1023;
    if (path.data != NULL && len > 0) {
        memcpy(path_buf, path.data, len);
    }
    path_buf[len] = '\0';

    FILE* f = fopen(path_buf, "wb");
    if (!f) {
        return (Result_bool){ .is_err = true, .err = vex_str_format("failed to open file '%s' for writing", path_buf) };
    }
    if (content.len > 0 && content.data != NULL) {
        size_t written = fwrite(content.data, 1, (size_t)content.len, f);
        if (written != (size_t)content.len) {
            fclose(f);
            return (Result_bool){ .is_err = true, .err = vex_str_format("failed to write complete content to '%s'", path_buf) };
        }
    }
    fclose(f);
    return (Result_bool){ .is_err = false, .val = true };
}

static inline Result_bool vex_append_file(vex_string path, vex_string content) {
    char path_buf[1024];
    int len = path.len < 1023 ? path.len : 1023;
    if (path.data != NULL && len > 0) {
        memcpy(path_buf, path.data, len);
    }
    path_buf[len] = '\0';

    FILE* f = fopen(path_buf, "ab");
    if (!f) {
        return (Result_bool){ .is_err = true, .err = vex_str_format("failed to open file '%s' for appending", path_buf) };
    }
    if (content.len > 0 && content.data != NULL) {
        size_t written = fwrite(content.data, 1, (size_t)content.len, f);
        if (written != (size_t)content.len) {
            fclose(f);
            return (Result_bool){ .is_err = true, .err = vex_str_format("failed to append complete content to '%s'", path_buf) };
        }
    }
    fclose(f);
    return (Result_bool){ .is_err = false, .val = true };
}

static inline Result_bool vex_remove_file(vex_string path) {
    char path_buf[1024];
    int len = path.len < 1023 ? path.len : 1023;
    if (path.data != NULL && len > 0) {
        memcpy(path_buf, path.data, len);
    }
    path_buf[len] = '\0';

    if (remove(path_buf) == 0) {
        return (Result_bool){ .is_err = false, .val = true };
    }
    return (Result_bool){ .is_err = true, .err = vex_str_format("failed to remove file '%s'", path_buf) };
}

// ============================================================================
// 7. PRETRAINED TENSOR BINARY WEIGHT I/O (PYTORCH / SAFETENSORS / GGUF COMPAT)
// ============================================================================

#ifndef _Result_tensor_f32_DEFINED
#define _Result_tensor_f32_DEFINED
typedef struct {
    bool           is_err;
    vex_string     err;
    vex_tensor_f32 val;
} Result_tensor_f32;
#endif

static inline Result_tensor_f32 vex_tensor_load_bin(vex_arena* opt_arena, int32_t rows, int32_t cols, vex_string path) {
    char path_buf[1024];
    int len = path.len < 1023 ? path.len : 1023;
    if (path.data != NULL && len > 0) {
        memcpy(path_buf, path.data, len);
    }
    path_buf[len] = '\0';
    vex_str_free(&path);

    FILE* f = fopen(path_buf, "rb");
    if (!f) {
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("could not open weight file '%s'", path_buf),
            .val = (vex_tensor_f32){0}
        };
    }

    vex_tensor_f32 t = opt_arena
        ? vex_tensor_f32_arena_alloc(opt_arena, rows, cols)
        : vex_tensor_f32_alloc(rows, cols);

    size_t expected = (size_t)rows * (size_t)cols;
    size_t read_count = fread(t.data, sizeof(vex_f32), expected, f);
    fclose(f);

    if (read_count != expected) {
        if (!opt_arena) vex_tensor_f32_free(&t);
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("weight file '%s' size mismatch: expected %zu floats, read %zu", path_buf, expected, read_count),
            .val = (vex_tensor_f32){0}
        };
    }

    return (Result_tensor_f32){
        .is_err = false,
        .err = vex_str_lit("", 0),
        .val = t
    };
}

static inline Result_bool vex_tensor_save_bin(vex_tensor_f32 t, vex_string path) {
    char path_buf[1024];
    int len = path.len < 1023 ? path.len : 1023;
    if (path.data != NULL && len > 0) {
        memcpy(path_buf, path.data, len);
    }
    path_buf[len] = '\0';
    vex_str_free(&path);

    FILE* f = fopen(path_buf, "wb");
    if (!f) {
        return (Result_bool){
            .is_err = true,
            .err = vex_str_format("failed to open file '%s' for writing weights", path_buf),
            .val = false
        };
    }

    size_t total = (size_t)t.rows * (size_t)t.cols;
    size_t written = fwrite(t.data, sizeof(vex_f32), total, f);
    fclose(f);

    if (written != total) {
        return (Result_bool){
            .is_err = true,
            .err = vex_str_format("failed to write complete weights to '%s'", path_buf),
            .val = false
        };
    }

    return (Result_bool){
        .is_err = false,
        .err = vex_str_lit("", 0),
        .val = true
    };
}

// ============================================================================
// 12. HIGH-PERFORMANCE NETWORKING & HTTP 1.1 WEB ENGINE
// ============================================================================

static inline void _vex_net_init(void) {
#ifdef _WIN32
    static int initialized = 0;
    if (!initialized) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        initialized = 1;
    }
#endif
}

static inline int64_t vex_net_listen(int64_t port) {
    _vex_net_init();
    vex_socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == VEX_INVALID_SOCKET) return -1;

    int opt = 1;
#ifdef _WIN32
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) == VEX_SOCKET_ERROR) {
        closesocket(fd);
        return -1;
    }

    if (listen(fd, 128) == VEX_SOCKET_ERROR) {
        closesocket(fd);
        return -1;
    }

    return (int64_t)fd;
}

static inline int64_t vex_net_accept(int64_t server_fd) {
    _vex_net_init();
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    vex_socket_t client = accept((vex_socket_t)server_fd, (struct sockaddr*)&client_addr, &addr_len);
    if (client == VEX_INVALID_SOCKET) return -1;
    return (int64_t)client;
}

static inline vex_string vex_net_read(int64_t client_fd, int64_t max_bytes) {
    if (max_bytes <= 0) max_bytes = 4096;
    char* buf = (char*)malloc(max_bytes + 1);
    if (!buf) return vex_str_lit("", 0);
    _vex_record_alloc();

    int n = recv((vex_socket_t)client_fd, buf, (int)max_bytes, 0);
    if (n <= 0) {
        free(buf);
        _vex_record_free();
        return vex_str_lit("", 0);
    }
    buf[n] = '\0';
    return (vex_string){ .data = (const uint8_t*)buf, .len = n, .is_lit = false };
}

static inline int64_t vex_net_write(int64_t client_fd, vex_string msg) {
    if (msg.len <= 0 || msg.data == NULL) return 0;
    int sent = send((vex_socket_t)client_fd, (const char*)msg.data, msg.len, 0);
    return (int64_t)sent;
}

static inline void vex_net_close(int64_t fd) {
    if (fd >= 0) {
        closesocket((vex_socket_t)fd);
    }
}

static inline vex_string vex_http_response(int64_t status_code, vex_string content_type, vex_string body) {
    const char* status_text = "OK";
    if (status_code == 201) status_text = "Created";
    else if (status_code == 400) status_text = "Bad Request";
    else if (status_code == 404) status_text = "Not Found";
    else if (status_code == 500) status_text = "Internal Server Error";

    char ctype_buf[128];
    int ctype_len = content_type.len < 127 ? content_type.len : 127;
    if (content_type.data && ctype_len > 0) {
        memcpy(ctype_buf, content_type.data, ctype_len);
    }
    ctype_buf[ctype_len] = '\0';

    int header_est = 256 + body.len;
    char* res = (char*)malloc(header_est);
    if (!res) return vex_str_lit("", 0);
    _vex_record_alloc();

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
    return (vex_string){ .data = (const uint8_t*)res, .len = n, .is_lit = false };
}

// ============================================================================
// 13. FAST JSON FIELD PARSER
// ============================================================================

static inline vex_string vex_json_get_str(vex_string json, vex_string key) {
    char pattern[256];
    int klen = key.len < 200 ? key.len : 200;
    if (key.data && klen > 0) {
        memcpy(pattern, key.data, klen);
    }
    pattern[klen] = '\0';

    const char* hay = (const char*)json.data;
    if (!hay) return vex_str_lit("", 0);

    char search_quote[260];
    snprintf(search_quote, sizeof(search_quote), "\"%s\"", pattern);
    
    char* found = strstr(hay, search_quote);
    if (!found) return vex_str_lit("", 0);
    
    char* colon = strchr(found + strlen(search_quote), ':');
    if (!colon) return vex_str_lit("", 0);
    
    char* val_start = strchr(colon, '"');
    if (!val_start) return vex_str_lit("", 0);
    val_start++;
    
    char* val_end = strchr(val_start, '"');
    if (!val_end) return vex_str_lit("", 0);
    
    int val_len = (int)(val_end - val_start);
    char* out = (char*)malloc(val_len + 1);
    if (!out) return vex_str_lit("", 0);
    _vex_record_alloc();
    memcpy(out, val_start, val_len);
    out[val_len] = '\0';
    return (vex_string){ .data = (const uint8_t*)out, .len = val_len, .is_lit = false };
}

static inline int64_t vex_json_get_int(vex_string json, vex_string key) {
    char search_quote[260];
    int klen = key.len < 200 ? key.len : 200;
    snprintf(search_quote, sizeof(search_quote), "\"%.*s\"", klen, (const char*)key.data);

    const char* hay = (const char*)json.data;
    if (!hay) return 0;
    char* found = strstr(hay, search_quote);
    if (!found) return 0;

    char* colon = strchr(found + strlen(search_quote), ':');
    if (!colon) return 0;
    colon++;
    while (*colon == ' ' || *colon == '\t') colon++;

    return (int64_t)strtoll(colon, NULL, 10);
}

static inline float vex_json_get_f32(vex_string json, vex_string key) {
    char search_quote[260];
    int klen = key.len < 200 ? key.len : 200;
    snprintf(search_quote, sizeof(search_quote), "\"%.*s\"", klen, (const char*)key.data);

    const char* hay = (const char*)json.data;
    if (!hay) return 0.0f;
    char* found = strstr(hay, search_quote);
    if (!found) return 0.0f;

    char* colon = strchr(found + strlen(search_quote), ':');
    if (!colon) return 0.0f;
    colon++;
    while (*colon == ' ' || *colon == '\t') colon++;

    return (float)strtod(colon, NULL);
}

// ============================================================================
// 14. NATIVE SAFETENSORS ZERO-COPY PARSER & LOADER
// ============================================================================

static inline Result_tensor_f32 vex_safetensors_load(vex_string path, vex_string tensor_name) {
    char path_buf[1024];
    int len = path.len < 1023 ? path.len : 1023;
    if (path.data != NULL && len > 0) {
        memcpy(path_buf, path.data, len);
    }
    path_buf[len] = '\0';
    vex_str_free(&path);

    char name_buf[256];
    int nlen = tensor_name.len < 255 ? tensor_name.len : 255;
    if (tensor_name.data != NULL && nlen > 0) {
        memcpy(name_buf, tensor_name.data, nlen);
    }
    name_buf[nlen] = '\0';
    vex_str_free(&tensor_name);

    FILE* f = fopen(path_buf, "rb");
    if (!f) {
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("failed to open safetensors file '%s'", path_buf),
            .val = (vex_tensor_f32){0}
        };
    }

    uint64_t header_len = 0;
    if (fread(&header_len, sizeof(uint64_t), 1, f) != 1 || header_len > 100 * 1024 * 1024) {
        fclose(f);
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("corrupt safetensors header in '%s'", path_buf),
            .val = (vex_tensor_f32){0}
        };
    }

    char* header_json = (char*)malloc(header_len + 1);
    if (!header_json) {
        fclose(f);
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("failed to allocate memory for header (%llu bytes)", (unsigned long long)header_len),
            .val = (vex_tensor_f32){0}
        };
    }

    if (fread(header_json, 1, header_len, f) != header_len) {
        free(header_json);
        fclose(f);
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("incomplete read of safetensors header in '%s'", path_buf),
            .val = (vex_tensor_f32){0}
        };
    }
    header_json[header_len] = '\0';

    char tensor_key[300];
    snprintf(tensor_key, sizeof(tensor_key), "\"%s\"", name_buf);
    char* t_entry = strstr(header_json, tensor_key);
    if (!t_entry) {
        free(header_json);
        fclose(f);
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("tensor '%s' not found in '%s'", name_buf, path_buf),
            .val = (vex_tensor_f32){0}
        };
    }

    int rows = 1, cols = 1;
    char* shape_ptr = strstr(t_entry, "\"shape\"");
    if (shape_ptr) {
        char* lbracket = strchr(shape_ptr, '[');
        if (lbracket) {
            rows = (int)strtol(lbracket + 1, &lbracket, 10);
            while (*lbracket == ' ' || *lbracket == ',') lbracket++;
            if (*lbracket != ']' && *lbracket != '\0') {
                cols = (int)strtol(lbracket, NULL, 10);
            }
        }
    }
    if (rows <= 0) rows = 1;
    if (cols <= 0) cols = 1;

    uint64_t begin_off = 0;
    char* off_ptr = strstr(t_entry, "\"data_offsets\"");
    if (off_ptr) {
        char* lbracket = strchr(off_ptr, '[');
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

    vex_tensor_f32 t = vex_tensor_f32_alloc(rows, cols);
    size_t expected_bytes = (size_t)rows * (size_t)cols * sizeof(float);
    size_t read_bytes = fread(t.data, 1, expected_bytes, f);
    fclose(f);

    if (read_bytes != expected_bytes) {
        vex_tensor_f32_free(&t);
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("tensor '%s' data truncated: expected %zu bytes, read %zu", name_buf, expected_bytes, read_bytes),
            .val = (vex_tensor_f32){0}
        };
    }

    return (Result_tensor_f32){
        .is_err = false,
        .err = vex_str_lit("", 0),
        .val = t
    };
}

// ============================================================================
// 15. HARDWARE SIMD AUTOGRAD BACKWARD OPS & OPTIMIZERS
// ============================================================================

static inline vex_tensor_f32 vex_tensor_transpose(vex_tensor_f32 a) {
    vex_tensor_f32 out = vex_tensor_f32_alloc(a.cols, a.rows);
    for (int r = 0; r < a.rows; ++r) {
        for (int c = 0; c < a.cols; ++c) {
            out.data[c * out.cols + r] = a.data[r * a.cols + c];
        }
    }
    return out;
}

static inline vex_tensor_f32 vex_tensor_matmul_grad_a(vex_tensor_f32 grad_out, vex_tensor_f32 b) {
    vex_tensor_f32 bt = vex_tensor_transpose(b);
    vex_tensor_f32 grad_a = vex_tensor_matmul_f32(NULL, grad_out, bt);
    vex_tensor_f32_free(&bt);
    return grad_a;
}

static inline vex_tensor_f32 vex_tensor_matmul_grad_b(vex_tensor_f32 a, vex_tensor_f32 grad_out) {
    vex_tensor_f32 at = vex_tensor_transpose(a);
    vex_tensor_f32 grad_b = vex_tensor_matmul_f32(NULL, at, grad_out);
    vex_tensor_f32_free(&at);
    return grad_b;
}

static inline vex_tensor_f32 vex_tensor_relu_backward(vex_tensor_f32 input, vex_tensor_f32 grad_out) {
    vex_tensor_f32 out = vex_tensor_f32_alloc(input.rows, input.cols);
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

static inline void vex_tensor_sgd_step(vex_tensor_f32* param, vex_tensor_f32 grad, float lr) {
    if (!param || !param->data || !grad.data) return;
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

static inline float _vex_fp16_to_fp32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp  = ((uint32_t)h & 0x7C00) >> 10;
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

static inline void _vex_gguf_skip_val(FILE* f, uint32_t val_type) {
    switch (val_type) {
        case 0: case 1: case 7:
            fseek(f, 1, SEEK_CUR); break;
        case 2: case 3:
            fseek(f, 2, SEEK_CUR); break;
        case 4: case 5: case 6:
            fseek(f, 4, SEEK_CUR); break;
        case 10: case 11: case 12:
            fseek(f, 8, SEEK_CUR); break;
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
                    _vex_gguf_skip_val(f, elem_type);
                }
            }
            break;
        }
    }
}

static inline Result_tensor_f32 vex_gguf_load_tensor(vex_string path, vex_string tensor_name) {
    char path_buf[1024];
    int len = path.len < 1023 ? path.len : 1023;
    if (path.data && len > 0) memcpy(path_buf, path.data, len);
    path_buf[len] = '\0';
    vex_str_free(&path);

    char name_buf[256];
    int nlen = tensor_name.len < 255 ? tensor_name.len : 255;
    if (tensor_name.data && nlen > 0) memcpy(name_buf, tensor_name.data, nlen);
    name_buf[nlen] = '\0';
    vex_str_free(&tensor_name);

    FILE* f = fopen(path_buf, "rb");
    if (!f) {
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("failed to open GGUF file '%s'", path_buf),
            .val = (vex_tensor_f32){0}
        };
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;

    if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) {
        fclose(f);
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("invalid GGUF magic in '%s'", path_buf),
            .val = (vex_tensor_f32){0}
        };
    }
    if (fread(&version, 4, 1, f) != 1 || (version != 2 && version != 3)) {
        fclose(f);
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("unsupported GGUF version %u in '%s'", version, path_buf),
            .val = (vex_tensor_f32){0}
        };
    }
    if (fread(&tensor_count, 8, 1, f) != 1 || fread(&kv_count, 8, 1, f) != 1) {
        fclose(f);
        return (Result_tensor_f32){
            .is_err = true,
            .err = vex_str_format("corrupt GGUF header in '%s'", path_buf),
            .val = (vex_tensor_f32){0}
        };
    }

    uint32_t alignment = 32;

    for (uint64_t k = 0; k < kv_count; ++k) {
        uint64_t klen = 0;
        if (fread(&klen, 8, 1, f) != 1) break;
        char key_buf[256];
        size_t to_read = klen < 255 ? klen : 255;
        if (fread(key_buf, 1, to_read, f) != to_read) break;
        key_buf[to_read] = '\0';
        if (klen > to_read) fseek(f, (long)(klen - to_read), SEEK_CUR);

        uint32_t val_type = 0;
        if (fread(&val_type, 4, 1, f) != 1) break;

        if (strcmp(key_buf, "general.alignment") == 0 && val_type == 4) {
            uint32_t a = 32;
            if (fread(&a, 4, 1, f) == 1 && a > 0) alignment = a;
        } else {
            _vex_gguf_skip_val(f, val_type);
        }
    }

    bool found = false;
    uint32_t matched_type = 0;
    uint64_t matched_offset = 0;
    int32_t t_rows = 1, t_cols = 1;

    for (uint64_t t = 0; t < tensor_count; ++t) {
        uint64_t n_len = 0;
        if (fread(&n_len, 8, 1, f) != 1) break;
        char cur_name[256];
        size_t to_read = n_len < 255 ? n_len : 255;
        if (fread(cur_name, 1, to_read, f) != to_read) break;
        cur_name[to_read] = '\0';
        if (n_len > to_read) fseek(f, (long)(n_len - to_read), SEEK_CUR);

        uint32_t n_dims = 0;
        if (fread(&n_dims, 4, 1, f) != 1) break;

        uint64_t dims[4] = {1, 1, 1, 1};
        for (uint32_t d = 0; d < n_dims && d < 4; ++d) {
            if (fread(&dims[d], 8, 1, f) != 1) break;
        }
        if (n_dims > 4) {
            fseek(f, (long)(8 * (n_dims - 4)), SEEK_CUR);
        }

        uint32_t dtype = 0;
        uint64_t offset = 0;
        if (fread(&dtype, 4, 1, f) != 1) break;
        if (fread(&offset, 8, 1, f) != 1) break;

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
            .err = vex_str_format("tensor '%s' not found in GGUF file '%s'", name_buf, path_buf),
            .val = (vex_tensor_f32){0}
        };
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

    vex_tensor_f32 t = vex_tensor_f32_alloc(t_rows, t_cols);
    int32_t total = t_rows * t_cols;

    if (matched_type == 0) {
        fread(t.data, sizeof(float), total, f);
    } else if (matched_type == 1) {
        uint16_t* hbuf = (uint16_t*)malloc(total * sizeof(uint16_t));
        if (hbuf) {
            fread(hbuf, sizeof(uint16_t), total, f);
            for (int32_t i = 0; i < total; ++i) {
                t.data[i] = _vex_fp16_to_fp32(hbuf[i]);
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
            float scale = _vex_fp16_to_fp32(scale_h);
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
            float scale = _vex_fp16_to_fp32(scale_h);
            int32_t base = b * 32;
            for (int32_t j = 0; j < 16; ++j) {
                uint8_t byte = nibbles[j];
                int8_t v0 = (int8_t)(byte & 0x0F) - 8;
                int8_t v1 = (int8_t)(byte >> 4) - 8;
                t.data[base + 2 * j]     = scale * (float)v0;
                t.data[base + 2 * j + 1] = scale * (float)v1;
            }
        }
    } else {
        fread(t.data, sizeof(float), total, f);
    }

    fclose(f);
    return (Result_tensor_f32){
        .is_err = false,
        .err = vex_str_lit("", 0),
        .val = t
    };
}

// ============================================================================
// 17. LLM TRANSFORMER OPERATORS (RMSNORM, ROPE, SILU, EMBEDDING, SAMPLING)
// ============================================================================

static inline vex_tensor_f32 vex_tensor_rmsnorm(vex_tensor_f32 x, vex_tensor_f32 weight, float eps) {
    vex_tensor_f32 out = vex_tensor_f32_alloc(x.rows, x.cols);
    for (int32_t r = 0; r < x.rows; ++r) {
        float* x_row = &x.data[r * x.cols];
        float* out_row = &out.data[r * x.cols];
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

static inline vex_tensor_f32 vex_tensor_rope(vex_tensor_f32 q, int64_t pos, float freq_base) {
    vex_tensor_f32 out = vex_tensor_f32_alloc(q.rows, q.cols);
    int32_t half_dim = q.cols / 2;

    for (int32_t r = 0; r < q.rows; ++r) {
        float* q_row = &q.data[r * q.cols];
        float* out_row = &out.data[r * q.cols];

        for (int32_t i = 0; i < half_dim; ++i) {
            float exponent = -(2.0f * (float)i) / (float)q.cols;
            float freq = powf(freq_base, exponent);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float q0 = q_row[2 * i];
            float q1 = q_row[2 * i + 1];

            out_row[2 * i]     = q0 * cos_a - q1 * sin_a;
            out_row[2 * i + 1] = q0 * sin_a + q1 * cos_a;
        }
    }
    return out;
}

static inline vex_tensor_f32 vex_tensor_silu(vex_tensor_f32 x) {
    vex_tensor_f32 out = vex_tensor_f32_alloc(x.rows, x.cols);
    int32_t total = x.rows * x.cols;
    for (int32_t i = 0; i < total; ++i) {
        float v = x.data[i];
        out.data[i] = v / (1.0f + expf(-v));
    }
    return out;
}

static inline vex_tensor_f32 vex_tensor_elemwise_mul(vex_tensor_f32 a, vex_tensor_f32 b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "vex tensor error: elemwise_mul shape mismatch (%dx%d) * (%dx%d)\n",
                a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }
    vex_tensor_f32 out = vex_tensor_f32_alloc(a.rows, a.cols);
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

static inline vex_tensor_f32 vex_tensor_softmax(vex_tensor_f32 logits) {
    vex_tensor_f32 out = vex_tensor_f32_alloc(logits.rows, logits.cols);
    for (int32_t r = 0; r < logits.rows; ++r) {
        float* in_row = &logits.data[r * logits.cols];
        float* out_row = &out.data[r * logits.cols];

        float max_val = in_row[0];
        for (int32_t c = 1; c < logits.cols; ++c) {
            if (in_row[c] > max_val) max_val = in_row[c];
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

static inline int64_t vex_sample_token(vex_tensor_f32 logits, float temperature) {
    int32_t total = logits.rows * logits.cols;
    if (total == 0) return 0;

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

    vex_tensor_f32 scaled = vex_tensor_f32_alloc(logits.rows, logits.cols);
    for (int32_t i = 0; i < total; ++i) {
        scaled.data[i] = logits.data[i] / temperature;
    }
    vex_tensor_f32 probs = vex_tensor_softmax(scaled);
    vex_tensor_f32_free(&scaled);

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
    vex_tensor_f32_free(&probs);
    return chosen;
}

static inline vex_tensor_f32 vex_tensor_embedding(vex_tensor_f32 table, int64_t token_id) {
    if (token_id < 0 || token_id >= table.rows) {
        fprintf(stderr, "vex tensor error: embedding index %lld out of range [0, %d)\n",
                (long long)token_id, table.rows);
        exit(1);
    }
    vex_tensor_f32 out = vex_tensor_f32_alloc(1, table.cols);
    memcpy(out.data, &table.data[token_id * table.cols], (size_t)table.cols * sizeof(float));
    return out;
}

static inline void vex_report_mem_stats(void) {
    vex_scheduler_shutdown();
    printf("[vex_rt] simd=%s\n", vex_simd_backend_name());
    printf("[vex_rt] pool_workers=%d | actor_tasks_run=%lld | work_steals=%lld\n",
           _vex_sched.num_workers,
           (long long)__atomic_load_n(&_vex_total_tasks_run, __ATOMIC_RELAXED),
           (long long)__atomic_load_n(&_vex_total_work_steals, __ATOMIC_RELAXED));
    printf("[vex_rt] active_leaks=%lld | total_freed=%lld\n",
           (long long)__atomic_load_n(&_vex_active_heap_allocs, __ATOMIC_RELAXED),
           (long long)__atomic_load_n(&_vex_total_heap_frees, __ATOMIC_RELAXED));
    fflush(stdout);
}

#endif // VEX_RT_H
