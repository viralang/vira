#ifndef VIRA_RT_SCHEDULER_H
#define VIRA_RT_SCHEDULER_H

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

/**
 * Checks whether the current thread is actively executing within an actor message handler.
 * @return True if currently in an actor context, false if running in the main thread or general worker.
 * Usage: Disables nested multithreading inside actors to prevent thread oversubscription.
 */
static inline bool _vira_in_active_actor(void) {
  return _vira_tls_actor != NULL && _vira_tls_actor->in_actor;
}

/**
 * Records the current depth of the active actor's defer cleanup stack.
 * @return Current defer stack count.
 * Usage: Scopes deferred cleanups to specific sub-blocks or handler iterations.
 */
static inline int32_t _vira_actor_defer_mark(void) {
  if (_vira_tls_actor != NULL && _vira_tls_actor->in_actor) {
    return _vira_tls_actor->defer_top;
  }
  return 0;
}

/**
 * Unwinds and pops deferred cleanup actions back down to the specified base mark.
 * @param base Target defer stack depth.
 * Usage: Executes block-scoped defers upon normal block exit.
 */
static inline void _vira_actor_defer_restore(int32_t base) {
  if (_vira_tls_actor != NULL && _vira_tls_actor->in_actor) {
    if (base >= 0 && base <= _vira_tls_actor->defer_top) {
      _vira_tls_actor->defer_top = base;
    }
  }
}

/**
 * Pushes a cleanup function and pointer to the active actor's defer stack with arena tracking.
 * @param fn Destructor function callback.
 * @param ptr Pointer to resource to clean up.
 * @param is_arena True if resource was allocated in an arena.
 * Usage: Tracks dynamic resources for automatic unwinding on actor fault or completion.
 */
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

/**
 * Convenience helper pushing a cleanup function and pointer to the active actor's defer stack.
 * @param fn Destructor function callback.
 * @param ptr Pointer to resource.
 * Usage: Default defer registration for actor tasks.
 */
static inline void _vira_actor_defer_push(vira_defer_fn fn, void *ptr) {
  _vira_actor_defer_push_ex(fn, ptr, false);
}

/**
 * Removes a deferred cleanup entry matching the specified resource pointer without executing it.
 * @param ptr Pointer to resource previously registered with defer.
 * Usage: Invoked when a resource is explicitly freed before defer unwinds.
 */
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

/**
 * Checks if a pointer was allocated inside any arena registered in the actor's defer stack.
 * @param ctx Pointer to actor context.
 * @param ptr Address to test.
 * @return True if pointer is arena-managed, false otherwise.
 * Usage: Prevents double-freeing arena memory during actor failure unwinding.
 */
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

/**
 * Executes all registered defer cleanups for an actor during failure recovery or termination.
 * @param ctx Pointer to actor context.
 * Usage: Ensures zero leaks when an actor catches a panic or undergoes supervised restart.
 */
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

/**
 * Initializes an actor context with mailbox, name, restart limit, and supervisor settings.
 * @param ctx Pointer to actor context struct.
 * @param id Unique integer actor identifier.
 * @param name Human-readable actor name for diagnostics.
 * @param max_restarts Maximum restart attempts under supervisor supervision.
 * Usage: Prepares actor context before registering with scheduler (`Actor.spawn(...)`).
 */
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

/**
 * Releases all resources, mailboxes, and pending messages belonging to an actor context.
 * @param ctx Pointer to actor context.
 * Usage: Invoked when an actor terminates or is shut down by its supervisor.
 */
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

/**
 * Traps a runtime panic, unwinds the active actor context if isolated, or aborts process with diagnostics.
 * @param msg Error message describing the panic reason.
 * Usage: Invoked by assertions, failed bounds checks, and explicit user panic calls (`panic("...")`).
 */
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

/**
 * Pushes a task to the global scheduler overflow queue when local deques are full.
 * @param t Pointer to task.
 * Usage: Fallback queue management in work-stealing scheduler.
 */
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

/**
 * Pops a task from the global scheduler overflow queue.
 * @return Task pointer or NULL if overflow queue is empty.
 * Usage: Work discovery by idle worker threads.
 */
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
/**
 * Pushes a task to the bottom of the local Chase-Lev work-stealing deque.
 * @param q Pointer to work-stealing deque.
 * @param t Pointer to task.
 * @return True on success, false if deque is saturated.
 * Usage: Lock-free task submission by the owning worker thread.
 */
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
/**
 * Thread-safe multi-producer push to a work-stealing deque.
 * @param q Pointer to work-stealing deque.
 * @param t Pointer to task.
 * @return True on success, false if full.
 * Usage: Enqueuing tasks from external or non-owning threads.
 */
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
/**
 * Pops a task from the bottom of the local work-stealing deque (LIFO order).
 * @param q Pointer to deque.
 * @return Task pointer, or NULL if empty or lost race.
 * Usage: High-cache-locality task execution by the owning worker thread.
 */
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
/**
 * Steals a task from the top of a remote worker's work-stealing deque (FIFO order).
 * @param q Pointer to target deque.
 * @return Stolen task pointer, or NULL if empty.
 * Usage: Work-stealing load balancing across threads in the pool.
 */
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

/**
 * Inspects all worker deques and the overflow queue to determine if any pending tasks exist.
 * @return True if pending work exists in the scheduler, false if completely idle.
 * Usage: Fast check before parking idle worker threads.
 */
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

/**
 * Wakes up one parked worker thread in the scheduler pool when new work is submitted.
 * Usage: Low-overhead thread signaling on task submission.
 */
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

/**
 * Parks the current worker thread on a condition variable until new work is submitted or shutdown begins.
 * Usage: Eliminates CPU busy-spinning when the system is idle.
 */
static void _vira_park(void) {
  pthread_mutex_lock(&_vira_parker.mu);
  __atomic_add_fetch(&_vira_parker.sleepers, 1, __ATOMIC_SEQ_CST);
  if (!_vira_any_work() && !__atomic_load_n(&_vira_sched.shutdown, __ATOMIC_ACQUIRE)) {
    // `_vira_parker.cv` uses `PTHREAD_COND_INITIALIZER`, which measures timeouts
    // against `CLOCK_REALTIME`. Using `CLOCK_MONOTONIC` here caused
    // `pthread_cond_timedwait` to return `ETIMEDOUT` immediately (0 ns) and
    // busy-spin all worker threads at 100% CPU while idle.
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += 50000000L; // 50 ms fallback poll (producers wake via cond_signal immediately)
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

/**
 * Fast pseudo-random number generator for selecting random victim deques during work stealing.
 * @param s Pointer to RNG seed state.
 * @return 32-bit pseudo-random integer.
 * Usage: Unbiased random victim selection in work-stealing loops.
 */
static inline uint32_t _vira_xorshift(uint32_t *s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return *s = x;
}

// Own deque first, then sweep victims starting at a random offset, then overflow queue
/**
 * Searches local deque, attempts stealing from peer workers, or checks overflow queue for tasks.
 * @param self Pointer to caller's deque.
 * @param rng Pointer to thread RNG state.
 * @param stolen Output flag set to true if task was stolen from a peer.
 * @return Discovered task pointer, or NULL if none found.
 * Usage: Core task discovery logic in worker thread event loops.
 */
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

/**
 * Thread entry point and main event loop for scheduler pool worker threads.
 * @param arg Pointer to integer worker ID.
 * @return Always NULL on thread exit.
 * Usage: Runs continuously processing tasks, work-stealing, and handling actor mailboxes.
 */
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

/**
 * Initializes the global work-stealing thread pool scheduler with the specified worker count.
 * @param num_workers Number of background worker threads to spawn (0 for CPU core count).
 * Usage: Bootstraps the Vira concurrency and actor runtime.
 */
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

/**
 * Signals all worker threads to exit, drains remaining queues, and shuts down the thread pool.
 * Usage: Clean teardown of scheduler resources at application termination.
 */
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
/**
 * Idempotent lazy initializer ensuring the work-stealing scheduler is running.
 * Usage: Called automatically by `async`, actors, and parallel tensor operations before submitting tasks.
 */
static inline void vira_scheduler_ensure_init(void) {
  if (__atomic_load_n(&_vira_sched.initialized, __ATOMIC_ACQUIRE)) {
    return;
  }
  pthread_mutex_lock(&_vira_sched_init_mu);
  if (!_vira_sched.initialized) {
    vira_scheduler_init(8);
  }
  pthread_mutex_unlock(&_vira_sched_init_mu);
}

/**
 * Returns the active number of background worker threads in the scheduler pool.
 * @return Integer worker count.
 * Usage: Sizing parallel partitions in tensor and matrix multiplication routines.
 */
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
/**
 * Submits an asynchronous task to the work-stealing scheduler for background execution.
 * @param task Pointer to initialized task structure.
 * Usage: Asynchronous task dispatch (`async { ... }`).
 */
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

/**
 * Retrieves and removes the next message task from an actor's lock-free mailbox.
 * @param ctx Pointer to actor context.
 * @return Message task pointer, or NULL if mailbox is empty.
 * Usage: Mailbox processing by actor drain tasks.
 */
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

/**
 * Scheduler task function that continuously drains and processes messages in an actor's mailbox.
 * @param env Pointer to actor context.
 * Usage: Dispatched automatically to scheduler threads when messages are sent to an actor.
 */
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

/**
 * Enqueues a message task into an actor's mailbox and schedules mailbox draining.
 * @param ctx Pointer to destination actor context.
 * @param task Pointer to message task.
 * Usage: Implements actor message passing in Vira (`actor.send(msg)`).
 */
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
/**
 * Awaits task completion with work-helping, executing available tasks on current thread rather than blocking.
 * @param task Pointer to task to await.
 * Usage: Implements the `await` keyword on asynchronous tasks and futures.
 */
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
/**
 * Submits a task to the scheduler and blocks the caller until execution finishes.
 * @param fn Task function pointer.
 * @param env Environment data pointer.
 * Usage: Synchronous task dispatch across worker threads.
 */
static inline void vira_scheduler_submit_and_wait(vira_task_fn fn, void *env) {
  vira_scheduler_ensure_init();
  vira_task task = {.fn = fn, .env = env, .done = 0};
  vira_scheduler_submit_async(&task);
  vira_scheduler_await(&task);
}

// Idempotent scope-exit cleanup injected by `internal/autofree` for Future[T]
/**
 * Releases memory allocated for an asynchronous future structure.
 * @param f Pointer to future to release.
 * Usage: Clean up completed futures.
 */
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

#endif // VIRA_RT_SCHEDULER_H
