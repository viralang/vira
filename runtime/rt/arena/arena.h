#ifndef VIRA_RT_ARENA_H
#define VIRA_RT_ARENA_H

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

/**
 * Initializes a 64-byte aligned bump-pointer memory arena with the given initial capacity.
 * @param capacity Initial capacity in bytes.
 * @return Freshly initialized `vira_arena`.
 * Usage: Creates arenas for batch O(1) allocation with bulk deallocation (`arena { ... }`).
 */
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

/**
 * Performs fast O(1) bump-pointer allocation from an arena with 64-byte cache-line alignment.
 * @param a Pointer to arena.
 * @param size Allocation size in bytes.
 * @return 64-byte aligned memory address.
 * Usage: High-throughput tensor and struct allocation inside arena scopes.
 */
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

/**
 * Checks whether a given pointer falls within any memory chunk owned by the arena.
 * @param a Pointer to arena.
 * @param ptr Memory address to test.
 * @return True if address is within arena blocks, false otherwise.
 * Usage: Memory safety validation and actor error isolation.
 */
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

static inline void _vira_gpu_invalidate_activations(const void *arena_ptr);

/**
 * Resets an arena's allocation offset back to zero, reusing allocated memory without system frees.
 * @param a Pointer to arena.
 * @Usage: Clears intermediate allocations between training steps or frame iterations.
 */
static inline void vira_arena_reset(vira_arena *a) {
  if (!a)
    return;
  _vira_gpu_invalidate_activations(a);
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

/**
 * Formats a `printf`-style string directly into arena-managed memory.
 * @param a Pointer to arena.
 * @param fmt Format string.
 * @return Formatted `vira_string` residing in arena memory.
 * Usage: Zero-cost string formatting inside arena blocks with guaranteed bulk cleanup.
 */
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

  // BORROWED from the arena slab: cloned to the heap if it is stored outside the arena.
  return (vira_string){
      .data = (const uint8_t *)buf, .len = needed, .is_lit = true, .is_borrowed = true};
}

/**
 * Releases all memory chunks and system allocations belonging to an arena.
 * @param a Pointer to arena.
 * Usage: Invoked automatically at the end of an `arena` block scope.
 */
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

#endif // VIRA_RT_ARENA_H
