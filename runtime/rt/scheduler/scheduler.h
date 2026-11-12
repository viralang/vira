#ifndef VIRA_RT_SCHEDULER_H
#define VIRA_RT_SCHEDULER_H

// ============================================================================
// 4. MULTI-THREADED LOCK-FREE WORK-STEALING ACTOR POOL & SUPERVISOR
// ============================================================================

#define VIRA_WS_CAP 4096
#define VIRA_WS_MASK (VIRA_WS_CAP - 1)
#define VIRA_MAX_WORKERS 32
#define VIRA_ACTOR_INITIAL_DEFERS 16

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

// `done` sentinels.
//  VIRA_TASK_DETACHED:     fire-and-forget actor message; the drain loop frees the envelope.
//  VIRA_TASK_SELF_MANAGED: the task owns its own lifetime (actor drain tasks). The scheduler must
//                          NOT write `done` after running it, because the owning actor may already
//                          have been destroyed by the time the task function returns.
//  VIRA_TASK_RUNNING:      an actor message whose handler has started. The drain loop moves a
//                          message from 0 to RUNNING with a CAS, so a waiter that gives up on a
//                          message (deadlock between actors) can CAS 0 -> DETACHED and know
//                          whether it won: only a message that has not started can be abandoned.
#define VIRA_TASK_DETACHED (-1)
#define VIRA_TASK_SELF_MANAGED (-2)
#define VIRA_TASK_RUNNING (-3)

// Actor scheduling states (`vira_actor_ctx.scheduled`).
#define VIRA_ACTOR_IDLE 0      // no drain task queued or running
#define VIRA_ACTOR_SCHEDULED 1 // a drain task is queued or running
#define VIRA_ACTOR_NOTIFIED 2  // a drain task is running and a sender enqueued more work

typedef void (*vira_defer_fn)(void *ptr);

typedef struct {
  vira_defer_fn fn;
  void *ptr;
  bool is_arena;
} vira_defer_entry;

static void _vira_actor_drain_task(void *env);
static inline void vira_scheduler_await(vira_task *task);
static inline bool _vira_sched_help_one(uint32_t *rng);

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
  vira_defer_entry *defer_stack; // Lazily allocated, grows on demand (never silently drops entries)
  int32_t defer_cap;
  int32_t defer_top;
  uint32_t wf_slot; // this actor's entry in the wait-for graph (VIRA_WF_NONE if unavailable)
};

// Thread-Local Storage (TLS): Each worker pthread tracks its own active actor!
static __thread vira_actor_ctx *_vira_tls_actor = NULL;
static __thread int32_t _vira_tls_worker_id = -1;

// ----------------------------------------------------------------------------
// WAIT-FOR GRAPH: detects actors that await each other (A waits on B, B waits on A).
//
// An actor handler that awaits a message to another actor publishes an EDGE "I wait on that
// actor" in a global slot table. The table lives in chunks that are never freed, so a check
// can follow edges without touching actor memory that may already be gone. An edge packs the
// waiter's wait sequence number (upper 32 bits) with the target's slot + 1 (lower 32 bits, 0 =
// not waiting); the sequence number changes on every new wait.
//
// A thread also records the actors suspended in an await on its own stack (work-helping runs
// other handlers on top of them); such an actor cannot resume before the handlers above it
// return, so reaching it counts as reaching the waiter itself.
//
// While a waiter spins it periodically walks the edges from its target. A cycle back to itself
// (or to an actor suspended beneath it) is reported only if two walks, taken apart, see exactly
// the same edges: each edge then existed continuously between the walks, so at one instant
// they all held at once, and a cycle of blocked waits never resolves. A transient chain seen
// while waits are starting and finishing does not repeat.
// ----------------------------------------------------------------------------
#define VIRA_WF_NONE 0xFFFFFFFFu
#define VIRA_WF_CHUNK_BITS 12
#define VIRA_WF_CHUNK (1u << VIRA_WF_CHUNK_BITS)
#define VIRA_WF_MAX_CHUNKS 1024 // 4M live actors
#define VIRA_WF_MAX_HOPS 64
#define VIRA_WF_STACK 40 // > VIRA_MAX_HELP_DEPTH: one entry per await nested on a thread

typedef struct {
  volatile uint64_t edge;
  uint32_t next_free; // free-list link (slot + 1), guarded by _vira_wf_lock
} _vira_wf_slot;

static _vira_wf_slot *volatile _vira_wf_chunks[VIRA_WF_MAX_CHUNKS];
static volatile int32_t _vira_wf_lock = 0;
static uint32_t _vira_wf_free_head = 0; // slot + 1; 0 = empty
static uint32_t _vira_wf_next = 0;      // next never-used slot

static __thread uint32_t _vira_tls_wf_stack[VIRA_WF_STACK];
static __thread int32_t _vira_tls_wf_depth = 0;

static inline _vira_wf_slot *_vira_wf_at(uint32_t s) {
  _vira_wf_slot *chunk = __atomic_load_n(&_vira_wf_chunks[s >> VIRA_WF_CHUNK_BITS], __ATOMIC_ACQUIRE);
  return &chunk[s & (VIRA_WF_CHUNK - 1)];
}

static inline void _vira_wf_lock_acquire(void) {
  while (__atomic_exchange_n(&_vira_wf_lock, 1, __ATOMIC_ACQUIRE) != 0) {
    while (__atomic_load_n(&_vira_wf_lock, __ATOMIC_RELAXED) != 0) {
    }
  }
}

static inline void _vira_wf_lock_release(void) { __atomic_store_n(&_vira_wf_lock, 0, __ATOMIC_RELEASE); }

/** Assigns a wait-for slot to a new actor; VIRA_WF_NONE if the table is full (detection off). */
static inline uint32_t _vira_wf_alloc(void) {
  _vira_wf_lock_acquire();
  uint32_t s;
  if (_vira_wf_free_head != 0) {
    s = _vira_wf_free_head - 1;
    _vira_wf_free_head = _vira_wf_at(s)->next_free;
  } else {
    s = _vira_wf_next;
    uint32_t ci = s >> VIRA_WF_CHUNK_BITS;
    if (ci >= VIRA_WF_MAX_CHUNKS) {
      _vira_wf_lock_release();
      return VIRA_WF_NONE;
    }
    if ((s & (VIRA_WF_CHUNK - 1)) == 0) {
      _vira_wf_slot *chunk = (_vira_wf_slot *)calloc(VIRA_WF_CHUNK, sizeof(_vira_wf_slot));
      if (chunk == NULL) {
        _vira_wf_lock_release();
        return VIRA_WF_NONE;
      }
      __atomic_store_n(&_vira_wf_chunks[ci], chunk, __ATOMIC_RELEASE);
    }
    _vira_wf_next++;
  }
  _vira_wf_lock_release();
  // Keep the sequence number of a reused slot so an old edge value never repeats.
  _vira_wf_slot *slot = _vira_wf_at(s);
  uint64_t e = __atomic_load_n(&slot->edge, __ATOMIC_RELAXED);
  __atomic_store_n(&slot->edge, e & 0xFFFFFFFF00000000ull, __ATOMIC_SEQ_CST);
  return s;
}

static inline void _vira_wf_free(uint32_t s) {
  if (s == VIRA_WF_NONE)
    return;
  _vira_wf_slot *slot = _vira_wf_at(s);
  uint64_t e = __atomic_load_n(&slot->edge, __ATOMIC_RELAXED);
  __atomic_store_n(&slot->edge, e & 0xFFFFFFFF00000000ull, __ATOMIC_SEQ_CST);
  _vira_wf_lock_acquire();
  slot->next_free = _vira_wf_free_head;
  _vira_wf_free_head = s + 1;
  _vira_wf_lock_release();
}

typedef struct {
  uint32_t slot;
  uint64_t edge;
} _vira_wf_hop;

/**
 * Follows wait-for edges from `target`.
 * @return True if the chain reaches an actor suspended on this thread (the waiter included).
 * `hops`/`n` receive the edges read, for comparison with a later walk.
 */
static inline bool _vira_wf_walk(uint32_t target, _vira_wf_hop *hops, int32_t *n) {
  uint32_t cur = target;
  *n = 0;
  for (int32_t i = 0; i < VIRA_WF_MAX_HOPS; i++) {
    for (int32_t j = 0; j < _vira_tls_wf_depth; j++) {
      if (_vira_tls_wf_stack[j] == cur)
        return true;
    }
    uint64_t e = __atomic_load_n(&_vira_wf_at(cur)->edge, __ATOMIC_SEQ_CST);
    hops[i].slot = cur;
    hops[i].edge = e;
    *n = i + 1;
    uint32_t next = (uint32_t)(e & 0xFFFFFFFFu);
    if (next == 0)
      return false;
    cur = next - 1;
  }
  return false;
}

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
    vira_actor_ctx *ctx = _vira_tls_actor;
    if (ctx->defer_top >= ctx->defer_cap) {
      int32_t new_cap = ctx->defer_cap > 0 ? ctx->defer_cap * 2 : VIRA_ACTOR_INITIAL_DEFERS;
      vira_defer_entry *grown = (vira_defer_entry *)realloc(
          ctx->defer_stack, (size_t)new_cap * sizeof(vira_defer_entry));
      if (grown == NULL) {
        fprintf(stderr, "vira panic: out of memory growing actor defer stack\n");
        exit(1);
      }
      ctx->defer_stack = grown;
      ctx->defer_cap = new_cap;
    }
    ctx->defer_stack[ctx->defer_top++] =
        (vira_defer_entry){.fn = fn, .ptr = ptr, .is_arena = is_arena};
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
  ctx->drain_task.done = VIRA_TASK_SELF_MANAGED;
  ctx->defer_stack = NULL;
  ctx->defer_cap = 0;
  ctx->defer_top = 0;
  ctx->wf_slot = _vira_wf_alloc();
}

/**
 * Blocks until an actor has no queued or running work, helping the scheduler while it waits.
 * After this returns (and until the next send) no worker thread holds a pointer into the actor,
 * so the actor value may be moved or destroyed.
 * @param ctx Pointer to actor context.
 * Usage: Called before an actor is destroyed or relocated (e.g. when a slice of actors grows).
 */
static inline void vira_actor_quiesce(vira_actor_ctx *ctx) {
  if (ctx == NULL)
    return;
  if (_vira_tls_actor == ctx && ctx->in_actor)
    return; // An actor cannot wait for its own handler to finish
  uint32_t rng = 0xC2B2AE35u;
  int32_t idle = 0;
  while (__atomic_load_n(&ctx->scheduled, __ATOMIC_SEQ_CST) != VIRA_ACTOR_IDLE) {
    if (_vira_sched_help_one(&rng)) {
      idle = 0;
      continue;
    }
    if (idle < 32) {
      vira_cpu_relax();
    } else {
      sched_yield();
    }
    idle++;
  }
}

/**
 * Releases all resources, mailboxes, and pending messages belonging to an actor context.
 * @param ctx Pointer to actor context.
 * Usage: Invoked when an actor terminates or is shut down by its supervisor.
 */
static inline void vira_actor_ctx_destroy(vira_actor_ctx *ctx) {
  vira_actor_quiesce(ctx);
  if (!ctx->last_error.is_lit && ctx->last_error.data != NULL) {
    vira_str_free(&ctx->last_error);
  }
  ctx->defer_top = 0;
  ctx->defer_cap = 0;
  free(ctx->defer_stack);
  ctx->defer_stack = NULL;
  vira_arena_destroy(&ctx->isolated_heap);
  _vira_wf_free(ctx->wf_slot);
  ctx->wf_slot = VIRA_WF_NONE;
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
    } else if (msg.is_lit && !msg.is_borrowed && !in_arena) {
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

/**
 * Raises a runtime fault (bounds error, division by zero, pop from empty slice, ...).
 * Inside an actor handler the fault is routed through `vira_panic`, so the supervisor traps it
 * and restarts the actor exactly like an explicit `panic(...)`. Outside an actor it prints the
 * message and terminates the process.
 */
static void _vira_rt_fault(const char *fmt, ...) {
  char buf[512];
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  if (n < 0) {
    n = 0;
    buf[0] = '\0';
  }
  if (n >= (int)sizeof(buf)) {
    n = (int)sizeof(buf) - 1;
  }
  if (_vira_tls_actor != NULL && _vira_tls_actor->in_actor) {
    // Non-literal: vira_panic copies the bytes before unwinding this stack frame.
    vira_panic((vira_string){.data = (const uint8_t *)buf, .len = (int32_t)n, .is_lit = false});
  }
  fprintf(stderr, "%s\n", buf);
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

// Clock the parker condition variable measures timeouts against. Set once in
// `vira_scheduler_init`; `_vira_park` must compute its deadline with the SAME clock.
static clockid_t _vira_parker_clock = CLOCK_REALTIME;
static bool _vira_parker_cv_ready = false;

/**
 * Wakes up one parked worker thread in the scheduler pool when new work is submitted.
 * Usage: Low-overhead thread signaling on task submission.
 */
static inline void vira_sched_wake_one(void) {
  // Full barrier between the producer's queue publish and this load of `sleepers`.
  // Together with the barrier in `_vira_park` (sleepers++ then re-check queues) this closes the
  // store-buffering window in which a producer skips the signal while a worker goes to sleep.
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (__atomic_load_n(&_vira_parker.sleepers, __ATOMIC_SEQ_CST) <= 0) {
    return;
  }
  pthread_mutex_lock(&_vira_parker.mu);
  pthread_cond_signal(&_vira_parker.cv);
  pthread_mutex_unlock(&_vira_parker.mu);
}

/**
 * Parks the current worker thread on a condition variable until new work is submitted or shutdown begins.
 * Usage: Eliminates CPU busy-spinning when the system is idle.
 */
static void _vira_park(void) {
  pthread_mutex_lock(&_vira_parker.mu);
  __atomic_add_fetch(&_vira_parker.sleepers, 1, __ATOMIC_SEQ_CST);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (!_vira_any_work() && !__atomic_load_n(&_vira_sched.shutdown, __ATOMIC_SEQ_CST)) {
    // 50 ms fallback poll. The deadline MUST use the condvar's own clock: mixing
    // CLOCK_REALTIME deadlines with a CLOCK_MONOTONIC condvar (or the reverse) makes the wait
    // either return immediately or effectively never time out.
    struct timespec ts;
    clock_gettime(_vira_parker_clock, &ts);
    ts.tv_nsec += 50000000L;
    if (ts.tv_nsec >= 1000000000L) {
      ts.tv_sec++;
      ts.tv_nsec -= 1000000000L;
    }
    pthread_cond_timedwait(&_vira_parker.cv, &_vira_parker.mu, &ts);
  }
  // Exactly one decrement per park, whether signalled or timed out.
  __atomic_sub_fetch(&_vira_parker.sleepers, 1, __ATOMIC_SEQ_CST);
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
      bool self_managed =
          (__atomic_load_n(&task->done, __ATOMIC_RELAXED) == VIRA_TASK_SELF_MANAGED);
      fn(env);
      local_run++;
      if (!self_managed) {
        __atomic_store_n(&task->done, 1, __ATOMIC_RELEASE); // task may be freed after this
      }
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
 * Default worker-pool size: `VIRA_WORKERS` if set, otherwise the number of logical cores,
 * clamped to [2, 16]. (At least 2 so one blocking handler cannot stall the whole pool.)
 */
static inline int32_t _vira_default_worker_count(void) {
  const char *env = getenv("VIRA_WORKERS");
  if (env != NULL && env[0] != '\0') {
    int32_t v = atoi(env);
    if (v > 0) {
      return v > VIRA_MAX_WORKERS ? VIRA_MAX_WORKERS : v;
    }
  }
  int32_t n = _vira_detect_logical_cores();
  if (n < 2) n = 2;
  if (n > 16) n = 16;
  return n;
}

/**
 * Initializes the global work-stealing thread pool scheduler with the specified worker count.
 * @param num_workers Number of background worker threads to spawn (0 for the default, see
 *                    `_vira_default_worker_count`).
 * Usage: Bootstraps the Vira concurrency and actor runtime.
 */
static inline void vira_scheduler_init(int32_t num_workers) {
  if (_vira_sched.initialized)
    return;
  if (num_workers <= 0) {
    num_workers = _vira_default_worker_count();
  }
  if (num_workers > VIRA_MAX_WORKERS) {
    num_workers = VIRA_MAX_WORKERS;
  }
  _vira_sched.num_workers = num_workers;
  _vira_sched.shutdown = 0;
  _vira_sched.rr_cursor = 0;

  _vira_sched.overflow.head = NULL;
  _vira_sched.overflow.tail = NULL;
  pthread_mutex_init(&_vira_sched.overflow.mu, NULL);

  if (!_vira_parker_cv_ready) {
    _vira_parker_cv_ready = true; // configure once; survives shutdown / re-init
#if defined(__linux__) && defined(CLOCK_MONOTONIC)
    // On Linux, move the condvar to CLOCK_MONOTONIC so wall-clock adjustments cannot stretch
    // or shrink the park timeout. `_vira_parker_clock` records the clock actually in use.
    // Other platforms keep the statically initialised CLOCK_REALTIME condvar: not every
    // pthread implementation honours pthread_condattr_setclock (some accept it and still
    // time out against the wall clock), and a mismatch there is exactly the bug this guards
    // against.
    pthread_condattr_t cattr;
    if (pthread_condattr_init(&cattr) == 0) {
      if (pthread_condattr_setclock(&cattr, CLOCK_MONOTONIC) == 0 &&
          pthread_cond_init(&_vira_parker.cv, &cattr) == 0) {
        _vira_parker_clock = CLOCK_MONOTONIC;
      }
      pthread_condattr_destroy(&cattr);
    }
#endif
  }

  // Initialise EVERY deque before starting any thread: a worker begins stealing from its
  // peers immediately, so their indices must already be in place.
  for (int32_t i = 0; i < num_workers; i++) {
    _vira_sched.workers[i].worker_id = i;
    __atomic_store_n(&_vira_sched.workers[i].top, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&_vira_sched.workers[i].bottom, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&_vira_sched.workers[i].push_lock, 0, __ATOMIC_RELEASE);
  }
  for (int32_t i = 0; i < num_workers; i++) {
    pthread_create(&_vira_sched.workers[i].thread, NULL, _vira_worker_loop,
                   &_vira_sched.workers[i]);
  }
  __atomic_store_n(&_vira_sched.initialized, true, __ATOMIC_RELEASE);
}

// ============================================================================
// PERSISTENT LOW-BARRIER COMPUTE WORKER TEAM (SPIN-THEN-SLEEP FOR LINEAR/GQA)
// ============================================================================

#define VIRA_TEAM_MAX_THREADS 16
#define VIRA_TEAM_SPIN_ITERS 4096

typedef struct {
  pthread_mutex_t call_mu;
  pthread_mutex_t wake_mu;
  pthread_cond_t wake_cv;
  VIRA_CACHELINE_ALIGNED volatile uint64_t gen;
  VIRA_CACHELINE_ALIGNED volatile int32_t remaining;
  VIRA_CACHELINE_ALIGNED volatile int32_t parked;
  volatile int32_t shutdown;
  int32_t active_workers;
  int32_t configured_threads;
  int32_t call_threads;
  _vira_team_fn fn;
  void *ctx;
  pthread_t threads[VIRA_TEAM_MAX_THREADS];
  int32_t tids[VIRA_TEAM_MAX_THREADS];
} _vira_compute_team_state;

static _vira_compute_team_state _vira_team = {
    .call_mu = PTHREAD_MUTEX_INITIALIZER,
    .wake_mu = PTHREAD_MUTEX_INITIALIZER,
    .wake_cv = PTHREAD_COND_INITIALIZER,
    .gen = 0,
    .remaining = 0,
    .parked = 0,
    .shutdown = 0,
    .active_workers = 0,
    .configured_threads = 0,
    .call_threads = 0,
    .fn = NULL,
    .ctx = NULL,
};

static __thread bool _vira_tls_in_team = false;

static inline int32_t vira_sched_get_threads(void) {
  int32_t cfg = __atomic_load_n(&_vira_team.configured_threads, __ATOMIC_ACQUIRE);
  if (cfg > 0) {
    return cfg > VIRA_TEAM_MAX_THREADS ? VIRA_TEAM_MAX_THREADS : cfg;
  }
  const char *env = getenv("VIRA_THREADS");
  if (env != NULL && env[0] != '\0') {
    int32_t v = atoi(env);
    if (v > 0) {
      return v > VIRA_TEAM_MAX_THREADS ? VIRA_TEAM_MAX_THREADS : v;
    }
  }
  int32_t n = _vira_detect_logical_cores();
  if (n < 1) n = 1;
  if (n > VIRA_TEAM_MAX_THREADS) n = VIRA_TEAM_MAX_THREADS;
  return n;
}

static void *_vira_team_worker_main(void *arg) {
  int32_t tid = *(const int32_t *)arg;
  _vira_compute_team_state *t = &_vira_team;
  uint64_t seen_gen = 0;

  while (true) {
    uint64_t cur_gen = 0;
    bool timed_out_spin = true;
    for (int32_t spin = 0; spin < VIRA_TEAM_SPIN_ITERS; ++spin) {
      cur_gen = __atomic_load_n(&t->gen, __ATOMIC_ACQUIRE);
      if (cur_gen != seen_gen) {
        timed_out_spin = false;
        break;
      }
      if (__atomic_load_n(&t->shutdown, __ATOMIC_ACQUIRE)) {
        _vira_flush_tls_mem_stats();
        return NULL;
      }
      vira_cpu_relax();
    }

    if (timed_out_spin) {
      pthread_mutex_lock(&t->wake_mu);
      __atomic_add_fetch(&t->parked, 1, __ATOMIC_SEQ_CST);
      while ((cur_gen = __atomic_load_n(&t->gen, __ATOMIC_SEQ_CST)) == seen_gen &&
             !__atomic_load_n(&t->shutdown, __ATOMIC_SEQ_CST)) {
        pthread_cond_wait(&t->wake_cv, &t->wake_mu);
      }
      __atomic_sub_fetch(&t->parked, 1, __ATOMIC_SEQ_CST);
      pthread_mutex_unlock(&t->wake_mu);
    }

    if (cur_gen == seen_gen) {
      if (__atomic_load_n(&t->shutdown, __ATOMIC_ACQUIRE)) {
        break;
      }
      continue;
    }

    seen_gen = cur_gen;
    _vira_team_fn fn = t->fn;
    void *ctx = t->ctx;
    int32_t nt = t->call_threads;
    if (fn != NULL && tid < nt) {
      _vira_tls_in_team = true;
      fn(ctx, tid, nt);
      _vira_tls_in_team = false;
    }
    __atomic_sub_fetch(&t->remaining, 1, __ATOMIC_ACQ_REL);
  }

  _vira_flush_tls_mem_stats();
  return NULL;
}

static inline void _vira_team_shutdown_locked(void) {
  _vira_compute_team_state *t = &_vira_team;
  if (t->active_workers > 0) {
    __atomic_store_n(&t->shutdown, 1, __ATOMIC_SEQ_CST);
    pthread_mutex_lock(&t->wake_mu);
    pthread_cond_broadcast(&t->wake_cv);
    pthread_mutex_unlock(&t->wake_mu);
    for (int32_t i = 0; i < t->active_workers; ++i) {
      pthread_join(t->threads[i], NULL);
    }
    t->active_workers = 0;
    __atomic_store_n(&t->gen, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&t->shutdown, 0, __ATOMIC_RELEASE);
  }
}

static inline void _vira_team_shutdown(void) {
  pthread_mutex_lock(&_vira_team.call_mu);
  _vira_team_shutdown_locked();
  pthread_mutex_unlock(&_vira_team.call_mu);
}

static inline void _vira_team_ensure_init_locked(int32_t target_threads) {
  _vira_compute_team_state *t = &_vira_team;
  if (target_threads < 1) target_threads = 1;
  if (target_threads > VIRA_TEAM_MAX_THREADS) target_threads = VIRA_TEAM_MAX_THREADS;
  int32_t want_workers = target_threads - 1;
  if (t->active_workers == want_workers) {
    return;
  }
  if (t->active_workers > 0) {
    _vira_team_shutdown_locked();
  }
  if (want_workers > 0) {
    __atomic_store_n(&t->shutdown, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&t->gen, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&t->remaining, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&t->parked, 0, __ATOMIC_RELEASE);
    for (int32_t i = 0; i < want_workers; ++i) {
      t->tids[i] = i + 1;
      pthread_create(&t->threads[i], NULL, _vira_team_worker_main, &t->tids[i]);
    }
    t->active_workers = want_workers;
  }
}

static inline void _vira_team_run(_vira_team_fn fn, void *ctx, int32_t max_threads) {
  if (fn == NULL) return;
  if (_vira_tls_in_team || _vira_in_active_actor() || max_threads <= 1) {
    fn(ctx, 0, 1);
    return;
  }
  _vira_compute_team_state *t = &_vira_team;
  pthread_mutex_lock(&t->call_mu);
  int32_t target = vira_sched_get_threads();
  int32_t eff = target;
  if (max_threads > 0 && max_threads < eff) {
    eff = max_threads;
  }
  if (eff <= 1) {
    pthread_mutex_unlock(&t->call_mu);
    fn(ctx, 0, 1);
    return;
  }
  _vira_team_ensure_init_locked(target);
  if (t->active_workers <= 0) {
    pthread_mutex_unlock(&t->call_mu);
    fn(ctx, 0, 1);
    return;
  }
  if (eff > t->active_workers + 1) {
    eff = t->active_workers + 1;
  }

  t->fn = fn;
  t->ctx = ctx;
  t->call_threads = eff;
  __atomic_store_n(&t->remaining, t->active_workers, __ATOMIC_RELEASE);
  __atomic_add_fetch(&t->gen, 1, __ATOMIC_SEQ_CST);

  if (__atomic_load_n(&t->parked, __ATOMIC_SEQ_CST) > 0) {
    pthread_mutex_lock(&t->wake_mu);
    pthread_cond_broadcast(&t->wake_cv);
    pthread_mutex_unlock(&t->wake_mu);
  }

  _vira_tls_in_team = true;
  fn(ctx, 0, eff);
  _vira_tls_in_team = false;

  while (__atomic_load_n(&t->remaining, __ATOMIC_ACQUIRE) > 0) {
    vira_cpu_relax();
  }
  pthread_mutex_unlock(&t->call_mu);
}

static inline void vira_sched_set_threads(int32_t n) {
  if (n <= 0) {
    pthread_mutex_lock(&_vira_team.call_mu);
    __atomic_store_n(&_vira_team.configured_threads, 0, __ATOMIC_RELEASE);
    int32_t target = vira_sched_get_threads();
    if (_vira_team.active_workers > 0 && _vira_team.active_workers != target - 1) {
      _vira_team_shutdown_locked();
    }
    pthread_mutex_unlock(&_vira_team.call_mu);
    return;
  }
  if (n > VIRA_TEAM_MAX_THREADS) {
    n = VIRA_TEAM_MAX_THREADS;
  }
  pthread_mutex_lock(&_vira_team.call_mu);
  if (_vira_team.active_workers > 0 && _vira_team.active_workers != n - 1) {
    _vira_team_shutdown_locked();
  }
  __atomic_store_n(&_vira_team.configured_threads, n, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&_vira_team.call_mu);
}

/**
 * Signals all worker threads to exit, drains remaining queues, and shuts down the thread pool.
 * Usage: Clean teardown of scheduler resources at application termination.
 */
static inline void vira_scheduler_shutdown(void) {
  _vira_team_shutdown();
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
    vira_scheduler_init(0); // sized from the machine's logical core count
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
  vira_actor_ctx *actor; // target actor for actor messages (NULL for plain async tasks)
} vira_future;

// Pushes a task onto a worker deque and returns immediately (non-blocking)
/**
 * Submits an asynchronous task to the work-stealing scheduler for background execution.
 * @param task Pointer to initialized task structure.
 * Usage: Asynchronous task dispatch (`async { ... }`).
 */
static inline void vira_scheduler_submit_async(vira_task *task) {
  vira_scheduler_ensure_init();
  if (__atomic_load_n(&task->done, __ATOMIC_RELAXED) != VIRA_TASK_SELF_MANAGED) {
    __atomic_store_n(&task->done, 0, __ATOMIC_RELAXED);
  }

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
  vira_task *head = __atomic_load_n(&ctx->mbox_head, __ATOMIC_ACQUIRE);
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
    __atomic_store_n(&ctx->mbox_head, next, __ATOMIC_RELEASE);
    return head;
  }

  // `head` is the last linked node. Clear `mbox_head` BEFORE publishing "empty" via the CAS on
  // `mbox_tail`: once the tail is NULL a producer may immediately store a new head, and clearing
  // afterwards would overwrite it (lost message, permanently wedged mailbox).
  vira_task *expected = head;
  __atomic_store_n(&ctx->mbox_head, NULL, __ATOMIC_RELEASE);
  if (__atomic_compare_exchange_n(&ctx->mbox_tail, &expected, NULL, false,
                                  __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    return head;
  }

  while ((next = __atomic_load_n(&head->next_actor_msg, __ATOMIC_ACQUIRE)) == NULL) {
    vira_cpu_relax();
  }
  __atomic_store_n(&ctx->mbox_head, next, __ATOMIC_RELEASE);
  return head;
}

/**
 * Queues an actor's drain task on the scheduler. The caller must own the SCHEDULED state.
 */
static inline void _vira_actor_submit_drain(vira_actor_ctx *ctx) {
  ctx->drain_task.fn = _vira_actor_drain_task;
  ctx->drain_task.env = ctx;
  ctx->drain_task.next_actor_msg = NULL;
  ctx->drain_task.done = VIRA_TASK_SELF_MANAGED;
  vira_scheduler_submit_async(&ctx->drain_task);
}

/**
 * Scheduler task function that continuously drains and processes messages in an actor's mailbox.
 * @param env Pointer to actor context.
 * Usage: Dispatched automatically to scheduler threads when messages are sent to an actor.
 *
 * Lifetime rule: the LAST access this function makes to `ctx` is either the CAS that releases
 * the actor (SCHEDULED -> IDLE) or the re-submission of its drain task. `vira_actor_quiesce`
 * relies on this: once it observes IDLE, no worker thread touches the actor again until the
 * next send.
 */
static void _vira_actor_drain_task(void *env) {
  vira_actor_ctx *ctx = (vira_actor_ctx *)env;
  int32_t quota = 64;
  int32_t count = 0;

  for (;;) {
    bool drained = false;
    while (quota > 0) {
      vira_task *msg = vira_actor_pop_msg(ctx);
      if (!msg) {
        drained = true;
        break;
      }
      quota--;
      vira_task_fn fn = msg->fn;
      void *menv = msg->env;
      // Claim the message: 0 -> RUNNING. If that fails it is DETACHED, either sent that way or
      // abandoned by a waiter that broke a deadlock; then this loop frees the envelope.
      int32_t expected_pending = 0;
      bool is_detached = !__atomic_compare_exchange_n(&msg->done, &expected_pending, VIRA_TASK_RUNNING,
                                                      false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
      fn(menv);
      count++;
      if (is_detached) {
        free(menv);
        _vira_record_free();
      } else {
        __atomic_store_n(&msg->done, 1, __ATOMIC_RELEASE);
      }
    }

    if (!drained) {
      // Quota exhausted: yield the worker and continue later. We still own SCHEDULED.
      if (count > 1) {
        __atomic_add_fetch(&_vira_total_tasks_run, count - 1, __ATOMIC_RELAXED);
      }
      __atomic_store_n(&ctx->scheduled, VIRA_ACTOR_SCHEDULED, __ATOMIC_SEQ_CST);
      _vira_actor_submit_drain(ctx); // last access to ctx
      return;
    }

    // Mailbox looked empty. Flush stats BEFORE releasing: nothing may touch ctx afterwards.
    if (count > 1) {
      __atomic_add_fetch(&_vira_total_tasks_run, count - 1, __ATOMIC_RELAXED);
      count = 1;
    }
    int32_t expected = VIRA_ACTOR_SCHEDULED;
    if (__atomic_compare_exchange_n(&ctx->scheduled, &expected, VIRA_ACTOR_IDLE, false,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
      return; // released: last access to ctx
    }
    // A sender enqueued while we were draining (NOTIFIED). Take the work ourselves.
    __atomic_store_n(&ctx->scheduled, VIRA_ACTOR_SCHEDULED, __ATOMIC_SEQ_CST);
    if (quota <= 0) {
      _vira_actor_submit_drain(ctx); // last access to ctx
      return;
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
  if (__atomic_load_n(&task->done, __ATOMIC_RELAXED) != VIRA_TASK_DETACHED) {
    __atomic_store_n(&task->done, 0, __ATOMIC_RELAXED);
  }

  vira_task *prev = __atomic_exchange_n(&ctx->mbox_tail, task, __ATOMIC_ACQ_REL);
  if (prev != NULL) {
    __atomic_store_n(&prev->next_actor_msg, task, __ATOMIC_RELEASE);
  } else {
    __atomic_store_n(&ctx->mbox_head, task, __ATOMIC_RELEASE);
  }

  // IDLE -> SCHEDULED (we submit the drain task), or SCHEDULED -> NOTIFIED (the running drain
  // task will see the flag when it tries to release the actor and keep draining).
  int32_t st = __atomic_load_n(&ctx->scheduled, __ATOMIC_SEQ_CST);
  for (;;) {
    if (st == VIRA_ACTOR_IDLE) {
      if (__atomic_compare_exchange_n(&ctx->scheduled, &st, VIRA_ACTOR_SCHEDULED, false,
                                      __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
        _vira_actor_submit_drain(ctx);
        return;
      }
    } else if (st == VIRA_ACTOR_NOTIFIED) {
      return;
    } else if (__atomic_compare_exchange_n(&ctx->scheduled, &st, VIRA_ACTOR_NOTIFIED, false,
                                           __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
      return;
    }
  }
}

// Work-helping can nest (a helped task may itself await). Bound the depth so a long chain of
// awaits cannot overflow the C stack; past the limit the waiter just yields.
#define VIRA_MAX_HELP_DEPTH 32
static __thread int32_t _vira_tls_help_depth = 0;

/**
 * Steals and runs at most one pending task on the calling thread.
 * @param rng Pointer to the caller's victim-selection RNG state.
 * @return True if a task was executed.
 * Usage: Shared by `vira_scheduler_await` and `vira_actor_quiesce`.
 */
static inline bool _vira_sched_help_one(uint32_t *rng) {
  int32_t n = _vira_sched.num_workers;
  if (n <= 0 || _vira_tls_help_depth >= VIRA_MAX_HELP_DEPTH) {
    return false;
  }
  int32_t start = (int32_t)(_vira_xorshift(rng) % (uint32_t)n);
  vira_task *stolen = NULL;
  for (int32_t i = 0; i < n && stolen == NULL; i++) {
    stolen = vira_ws_steal(&_vira_sched.workers[(start + i) % n]);
  }
  if (stolen == NULL) {
    stolen = _vira_overflow_pop();
  }
  if (stolen == NULL) {
    return false;
  }
  __atomic_add_fetch(&_vira_total_work_steals, 1, __ATOMIC_RELAXED);
  vira_task_fn fn = stolen->fn;
  void *env = stolen->env;
  bool self_managed =
      (__atomic_load_n(&stolen->done, __ATOMIC_RELAXED) == VIRA_TASK_SELF_MANAGED);
  _vira_tls_help_depth++;
  fn(env);
  _vira_tls_help_depth--;
  __atomic_add_fetch(&_vira_total_tasks_run, 1, __ATOMIC_RELAXED);
  if (!self_managed) {
    __atomic_store_n(&stolen->done, 1, __ATOMIC_RELEASE);
  }
  return true;
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
  for (;;) {
    int32_t d = __atomic_load_n(&task->done, __ATOMIC_ACQUIRE);
    if (d != 0 && d != VIRA_TASK_RUNNING)
      break;
    if (_vira_sched_help_one(&rng)) {
      idle = 0;
      continue;
    }
    if (idle < 32) {
      vira_cpu_relax();
    } else {
      sched_yield();
    }
    idle++;
  }
}

/**
 * True when the calling thread is running a handler of the given actor.
 * Usage: Generated await/sync-call code uses this to reject an actor waiting on a message it
 * sent to itself, which can never complete (the message is queued behind the running handler).
 */
static inline bool vira_actor_is_current(const vira_actor_ctx *ctx) {
  return ctx != NULL && _vira_tls_actor == ctx && ctx->in_actor;
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

/**
 * Handles a future whose message was sent by an actor to ITSELF and is still pending.
 * Such a message is queued behind the running handler, so waiting for it would never return.
 * The message is converted into a detached send (it still runs after the current handler, and
 * the drain loop frees its envelope) and the future is emptied.
 * @return True if the future was detached; the caller must not wait on it or free its envelope.
 */
static inline bool _vira_future_detach_if_self(vira_future *f) {
  if (f == NULL || f->env == NULL || f->task == NULL)
    return false;
  if (!vira_actor_is_current(f->actor))
    return false;
  int32_t expected = 0;
  if (!__atomic_compare_exchange_n(&f->task->done, &expected, VIRA_TASK_DETACHED, false,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    return false; // already running, completed or detached
  f->env = NULL;
  f->task = NULL;
  f->drop_fn = NULL;
  return true;
}

/**
 * Waits for an actor message like `vira_scheduler_await`, but from inside an actor handler it
 * also watches the wait-for graph (see the comment at VIRA_WF_NONE). If waiting would deadlock,
 * and the message has not started yet, the message is converted into a detached send (it still
 * runs later and the drain loop frees its envelope) and the wait is abandoned.
 * @param task The message being waited on.
 * @param target The actor the message was sent to.
 * @return True if the wait was abandoned because of a deadlock; the caller must then treat the
 * envelope as no longer its own.
 */
static inline bool vira_actor_await_msg(vira_task *task, const vira_actor_ctx *target) {
  vira_actor_ctx *me = (_vira_tls_actor != NULL && _vira_tls_actor->in_actor) ? _vira_tls_actor : NULL;
  if (task == NULL || me == NULL || target == NULL || me->wf_slot == VIRA_WF_NONE ||
      target->wf_slot == VIRA_WF_NONE || _vira_tls_wf_depth >= VIRA_WF_STACK) {
    vira_scheduler_await(task);
    return false;
  }
  _vira_wf_slot *my_slot = _vira_wf_at(me->wf_slot);
  uint64_t seq = (__atomic_load_n(&my_slot->edge, __ATOMIC_RELAXED) >> 32) + 1;
  __atomic_store_n(&my_slot->edge, (seq << 32) | ((uint64_t)target->wf_slot + 1), __ATOMIC_SEQ_CST);
  _vira_tls_wf_stack[_vira_tls_wf_depth++] = me->wf_slot;

  _vira_wf_hop prev[VIRA_WF_MAX_HOPS];
  _vira_wf_hop cur[VIRA_WF_MAX_HOPS];
  int32_t prev_n = -1;
  bool abandoned = false;
  uint32_t rng = 0x27D4EB2Fu;
  int32_t idle = 0;
  uint32_t iter = 0;
  for (;;) {
    int32_t d = __atomic_load_n(&task->done, __ATOMIC_ACQUIRE);
    if (d != 0 && d != VIRA_TASK_RUNNING)
      break;
    if ((iter++ & 255u) == 0) {
      int32_t cur_n = 0;
      if (_vira_wf_walk(target->wf_slot, cur, &cur_n)) {
        if (prev_n == cur_n && memcmp(prev, cur, sizeof(_vira_wf_hop) * (size_t)cur_n) == 0) {
          int32_t expected = 0;
          if (__atomic_compare_exchange_n(&task->done, &expected, VIRA_TASK_DETACHED, false,
                                          __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            abandoned = true;
            break;
          }
          // The message is already running: another actor in the cycle is waiting on a message
          // that has not started, and it will break the cycle instead.
        }
        memcpy(prev, cur, sizeof(_vira_wf_hop) * (size_t)cur_n);
        prev_n = cur_n;
      } else {
        prev_n = -1;
      }
    }
    if (_vira_sched_help_one(&rng)) {
      idle = 0;
      continue;
    }
    if (idle < 32) {
      vira_cpu_relax();
    } else {
      sched_yield();
    }
    idle++;
  }

  _vira_tls_wf_depth--;
  __atomic_store_n(&my_slot->edge, seq << 32, __ATOMIC_SEQ_CST);
  return abandoned;
}

/**
 * Waits for a future's message (deadlock-aware, see vira_actor_await_msg).
 * @return True if the wait was abandoned; the future has then been emptied and its envelope
 * belongs to the drain loop.
 */
static inline bool _vira_future_wait_or_detach(vira_future *f) {
  if (vira_actor_await_msg(f->task, f->actor)) {
    f->env = NULL;
    f->task = NULL;
    f->drop_fn = NULL;
    return true;
  }
  return false;
}

// Idempotent scope-exit cleanup injected by `internal/autofree` for Future[T]
/**
 * Releases memory allocated for an asynchronous future structure.
 * @param f Pointer to future to release.
 * Usage: Clean up completed futures.
 */
static inline void vira_future_free(vira_future *f) {
  if (f != NULL && f->env != NULL) {
    if (_vira_future_detach_if_self(f)) {
      return;
    }
    // If an early return skipped `.await()`, wait for the worker thread to
    // finish before freeing unconsumed result and envelope
    if (f->task != NULL && _vira_future_wait_or_detach(f)) {
      return;
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
