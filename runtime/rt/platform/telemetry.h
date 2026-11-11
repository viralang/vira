#ifndef VIRA_RT_PLATFORM_TELEMETRY_H
#define VIRA_RT_PLATFORM_TELEMETRY_H

// ============================================================================
// PROCESS EXIT TELEMETRY & RUNTIME SHUTDOWN
// ============================================================================

/**
 * Prints the runtime diagnostics banner reporting memory leak verification, SIMD backend, and actor pool metrics.
 * Usage: Invoked automatically at process exit; verifies that `active_leaks == 0`.
 */
static inline void vira_report_mem_stats(void) {
  _vira_flush_tls_mem_stats();
#if defined(VIRA_QUIET_RT)
  vira_gpu_shutdown();
  vira_scheduler_shutdown();
  _vira_llm_shutdown();
  _vira_rt_quant_shutdown();
  _vira_flush_tls_mem_stats();
  return;
#else
  if (_vira_quiet_rt) {
    vira_gpu_shutdown();
    vira_scheduler_shutdown();
    _vira_llm_shutdown();
    _vira_rt_quant_shutdown();
    _vira_flush_tls_mem_stats();
    return;
  }
  const char *env_stats = getenv("VIRA_RT_STATS");
  if (env_stats != NULL && strcmp(env_stats, "0") == 0) {
    vira_gpu_shutdown();
    vira_scheduler_shutdown();
    _vira_llm_shutdown();
    _vira_rt_quant_shutdown();
    _vira_flush_tls_mem_stats();
    return;
  }
  if (vira_gpu_is_available()) {
    vira_string _dev_n = vira_gpu_device_name();
    vira_string _be_n = vira_gpu_backend_name();
    printf("[vira_rt] gpu=%.*s (%.*s) | vram_used=%lld MiB / %lld MiB | oom_fallbacks=%lld\n",
           _dev_n.len, (const char *)_dev_n.data,
           _be_n.len, (const char *)_be_n.data,
           (long long)(vira_gpu_allocated_bytes() >> 20),
           (long long)(vira_gpu_vram_bytes() >> 20),
           (long long)vira_gpu_oom_fallbacks());
    vira_str_free(&_dev_n);
    vira_str_free(&_be_n);
  }
  vira_gpu_shutdown();
  vira_scheduler_shutdown();
  _vira_llm_shutdown();
  _vira_rt_quant_shutdown();
  _vira_flush_tls_mem_stats();
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

#endif // VIRA_RT_PLATFORM_TELEMETRY_H
