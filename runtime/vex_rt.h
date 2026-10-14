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
