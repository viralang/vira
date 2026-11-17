#ifndef VIRA_RT_CLOSURE_H
#define VIRA_RT_CLOSURE_H

// ============================================================================
// 1.7. VIRA CLOSURES (`fn (A) R` values)
// ============================================================================
//
// A closure is a function pointer plus an optional environment holding the
// values it captured. Captures are immutable copies, so copies of a closure can
// share one environment: copying retains it, releasing the last copy frees the
// captured values (via the generated `drop`) and the environment. The count is
// atomic because closures can be sent to actors on other threads.
//
// `fn` always has the signature `R fn(vira_closure_env* env, A...)`; the
// compiler casts it at each call. Named functions used as values get a small
// generated thunk with that signature and a NULL environment.

typedef struct vira_closure_env {
  int32_t refs;
  void (*drop)(struct vira_closure_env *env); // releases the captured values
} vira_closure_env;

typedef struct {
  void *fn;
  vira_closure_env *env;
} vira_closure;

static inline void *vira_closure_env_new(size_t size, void (*drop)(vira_closure_env *)) {
  vira_closure_env *env = (vira_closure_env *)calloc(1, size);
  if (!env) {
    fprintf(stderr, "vira panic: out of memory allocating a closure\n");
    exit(1);
  }
  env->refs = 1;
  env->drop = drop;
  _vira_record_alloc();
  return env;
}

static inline vira_closure vira_closure_retain(vira_closure c) {
  if (c.env)
    __atomic_add_fetch(&c.env->refs, 1, __ATOMIC_RELAXED);
  return c;
}

static inline void vira_closure_release(vira_closure *c) {
  if (c == NULL)
    return;
  if (c->env && __atomic_sub_fetch(&c->env->refs, 1, __ATOMIC_ACQ_REL) == 0) {
    if (c->env->drop)
      c->env->drop(c->env);
    free(c->env);
    _vira_record_free();
  }
  c->env = NULL;
}

static inline void _vira_closure_null_call(void) {
  _vira_rt_fault("vira panic: called an uninitialised function value");
}

#endif // VIRA_RT_CLOSURE_H
