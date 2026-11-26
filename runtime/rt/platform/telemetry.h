#ifndef VIRA_RT_PLATFORM_TELEMETRY_H
#define VIRA_RT_PLATFORM_TELEMETRY_H

// ============================================================================
// PROCESS EXIT TELEMETRY & RUNTIME SHUTDOWN
// ============================================================================

#ifndef VIRA_NO_GPU
static inline bool vira_gpu_was_initialized(void);
static inline void _vira_gpu_stats_peak(void);
static inline void vira_gpu_stats_print(vira_string label, int64_t per);
#endif

/**
 * Prints the runtime diagnostics banner reporting memory leak verification, SIMD backend, and actor pool metrics.
 * Usage: Invoked automatically at process exit; verifies that `active_leaks == 0`.
 */
static inline void vira_report_mem_stats(void) {
  _vira_flush_tls_mem_stats();
#if defined(VIRA_QUIET_RT)
#ifndef VIRA_NO_GPU
  vira_gpu_shutdown();
#endif
  vira_scheduler_shutdown();
#ifndef VIRA_NO_GPU
  _vira_llm_shutdown();
  _vira_rt_quant_shutdown();
#endif
  _vira_flush_tls_mem_stats();
  return;
#else
  if (_vira_quiet_rt) {
#ifndef VIRA_NO_GPU
    vira_gpu_shutdown();
#endif
    vira_scheduler_shutdown();
#ifndef VIRA_NO_GPU
    _vira_llm_shutdown();
    _vira_rt_quant_shutdown();
#endif
    _vira_flush_tls_mem_stats();
    return;
  }
  const char *env_stats = getenv("VIRA_RT_STATS");
  if (env_stats != NULL && strcmp(env_stats, "0") == 0) {
#ifndef VIRA_NO_GPU
    vira_gpu_shutdown();
#endif
    vira_scheduler_shutdown();
#ifndef VIRA_NO_GPU
    _vira_llm_shutdown();
    _vira_rt_quant_shutdown();
#endif
    _vira_flush_tls_mem_stats();
    return;
  }
#ifndef VIRA_NO_GPU
  if (vira_gpu_was_initialized()) {
    vira_string _dev_n = vira_gpu_device_name();
    vira_string _be_n = vira_gpu_backend_name();
    // Peak, not the amount at exit: by now the program has freed its tensors, and
    // freeing a tensor also frees its device copy.
    _vira_gpu_stats_peak();
    printf("[vira_rt] gpu=%.*s (%.*s) | vram_peak=%lld MiB / %lld MiB | oom_fallbacks=%lld\n",
           _dev_n.len, (const char *)_dev_n.data,
           _be_n.len, (const char *)_be_n.data,
           (long long)(_vira_gpu_stats.peak_bytes >> 20),
           (long long)(vira_gpu_vram_bytes() >> 20),
           (long long)vira_gpu_oom_fallbacks());
    const char *env_gs = getenv("VIRA_GPU_STATS");
    if (env_gs != NULL && env_gs[0] != '\0' && env_gs[0] != '0') {
      vira_gpu_stats_print(vira_str_lit("since last reset", 16), 0);
    }
    vira_str_free(&_dev_n);
    vira_str_free(&_be_n);
  }
  vira_gpu_shutdown();
#endif
  vira_scheduler_shutdown();
#ifndef VIRA_NO_GPU
  _vira_llm_shutdown();
  _vira_rt_quant_shutdown();
#endif
  _vira_flush_tls_mem_stats();
#ifndef VIRA_NO_GPU
  printf("[vira_rt] simd=%s\n", vira_simd_backend_name());
#endif
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

#endif // VIRA_RT_PLATFORM_TELEMETRY_H
