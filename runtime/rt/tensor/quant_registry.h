#ifndef VIRA_RT_TENSOR_QUANT_REGISTRY_H
#define VIRA_RT_TENSOR_QUANT_REGISTRY_H

// ============================================================================
// 3.5 HOST QUANTIZED WEIGHT REGISTRY & LAZY FP32 MATERIALIZATION (PHASE 4.1)
// ============================================================================

/**
 * Converts a 16-bit IEEE 754 half-precision float into a 32-bit single-precision float.
 * @param h Raw 16-bit half precision float bits.
 * @return Converted 32-bit float value.
 * Usage: Dequantization of FP16 model weights during GGUF parsing and lazy CPU fallbacks.
 */
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

/**
 * Decodes a 6-bit sub-block scale and min pair from a 12-byte GGUF K-quant (`Q4_K` / `Q5_K`) scale table.
 * @param sub Sub-block index in `[0, 7]`.
 * @param scales Pointer to the 12-byte packed GGUF scale array.
 * @param out_sc Output pointer receiving the 6-bit unsigned scale factor.
 * @param out_m Output pointer receiving the 6-bit unsigned minimum offset.
 * Usage: Internal helper used for `Q4_K` and `Q5_K` super-block dequantization and dot products.
 */
static inline void _vira_gguf_k4_scale_min(int32_t sub, const uint8_t *scales,
                                           uint8_t *out_sc, uint8_t *out_m) {
  if (sub < 4) {
    *out_sc = (uint8_t)(scales[sub] & 63u);
    *out_m = (uint8_t)(scales[sub + 4] & 63u);
  } else {
    *out_sc = (uint8_t)((scales[sub + 4] & 0x0Fu) |
                        ((scales[sub - 4] >> 6) << 4));
    *out_m = (uint8_t)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
  }
}

static inline bool _vira_lazy_host_fp32_enabled(void) {
  const char *env = getenv("VIRA_LAZY_HOST_FP32");
  return !(env != NULL && strcmp(env, "0") == 0);
}

static inline size_t _vira_gguf_dtype_qk(uint32_t gguf_dtype) {
  switch (gguf_dtype) {
  case 1: return 1;
  case 2: case 3: case 6: case 7: case 8: return 32;
  case 12: case 13: case 14: return 256;
  default: return 0;
  }
}

static inline size_t _vira_gguf_dtype_bs(uint32_t gguf_dtype) {
  switch (gguf_dtype) {
  case 1: return 2;
  case 2: return 18;
  case 3: return 20;
  case 6: return 22;
  case 7: return 24;
  case 8: return 34;
  case 12: return 144;
  case 13: return 176;
  case 14: return 210;
  default: return 0;
  }
}

static inline void _vira_rt_dequant_blocks_to_fp32(uint32_t gguf_dtype,
                                                   const uint8_t *raw_q,
                                                   size_t start_block,
                                                   size_t n_blocks,
                                                   float *dst_out) {
  if (!raw_q || !dst_out || n_blocks == 0)
    return;
  if (gguf_dtype == 1) {
    for (size_t b = 0; b < n_blocks; ++b) {
      uint16_t h = 0;
      memcpy(&h, raw_q + (start_block + b) * 2u, 2);
      dst_out[b] = _vira_fp16_to_fp32(h);
    }
  } else if (gguf_dtype == 2) {
    for (size_t b = 0; b < n_blocks; ++b) {
      const uint8_t *blk = raw_q + (start_block + b) * 18u;
      uint16_t scale_h = 0;
      memcpy(&scale_h, blk, 2);
      float scale = _vira_fp16_to_fp32(scale_h);
      const uint8_t *nibbles = blk + 2;
      float *dst = dst_out + b * 32u;
      for (int32_t j = 0; j < 16; ++j) {
        uint8_t byte = nibbles[j];
        int8_t v0 = (int8_t)(byte & 0x0Fu) - 8;
        int8_t v1 = (int8_t)(byte >> 4) - 8;
        dst[j] = scale * (float)v0;
        dst[j + 16] = scale * (float)v1;
      }
    }
  } else if (gguf_dtype == 3) {
    for (size_t b = 0; b < n_blocks; ++b) {
      const uint8_t *blk = raw_q + (start_block + b) * 20u;
      uint16_t d_h = 0, m_h = 0;
      memcpy(&d_h, blk, 2);
      memcpy(&m_h, blk + 2, 2);
      float d = _vira_fp16_to_fp32(d_h);
      float m = _vira_fp16_to_fp32(m_h);
      const uint8_t *qs = blk + 4;
      float *dst = dst_out + b * 32u;
      for (int32_t j = 0; j < 16; ++j) {
        uint8_t byte = qs[j];
        dst[j] = d * (float)(byte & 0x0Fu) + m;
        dst[j + 16] = d * (float)(byte >> 4) + m;
      }
    }
  } else if (gguf_dtype == 6) {
    for (size_t b = 0; b < n_blocks; ++b) {
      const uint8_t *blk = raw_q + (start_block + b) * 22u;
      uint16_t d_h = 0;
      memcpy(&d_h, blk, 2);
      float d = _vira_fp16_to_fp32(d_h);
      uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8) |
                    ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
      const uint8_t *qs = blk + 6;
      float *dst = dst_out + b * 32u;
      for (int32_t j = 0; j < 16; ++j) {
        uint8_t byte = qs[j];
        int32_t v0 =
            (int32_t)((uint32_t)(byte & 0x0Fu) | (((qh >> j) & 1u) << 4)) - 16;
        int32_t v1 =
            (int32_t)((uint32_t)(byte >> 4) | (((qh >> (j + 16)) & 1u) << 4)) -
            16;
        dst[j] = d * (float)v0;
        dst[j + 16] = d * (float)v1;
      }
    }
  } else if (gguf_dtype == 7) {
    for (size_t b = 0; b < n_blocks; ++b) {
      const uint8_t *blk = raw_q + (start_block + b) * 24u;
      uint16_t d_h = 0, m_h = 0;
      memcpy(&d_h, blk, 2);
      memcpy(&m_h, blk + 2, 2);
      float d = _vira_fp16_to_fp32(d_h);
      float m = _vira_fp16_to_fp32(m_h);
      uint32_t qh = (uint32_t)blk[4] | ((uint32_t)blk[5] << 8) |
                    ((uint32_t)blk[6] << 16) | ((uint32_t)blk[7] << 24);
      const uint8_t *qs = blk + 8;
      float *dst = dst_out + b * 32u;
      for (int32_t j = 0; j < 16; ++j) {
        uint8_t byte = qs[j];
        uint32_t v0 = (uint32_t)(byte & 0x0Fu) | (((qh >> j) & 1u) << 4);
        uint32_t v1 = (uint32_t)(byte >> 4) | (((qh >> (j + 16)) & 1u) << 4);
        dst[j] = d * (float)v0 + m;
        dst[j + 16] = d * (float)v1 + m;
      }
    }
  } else if (gguf_dtype == 8) {
    for (size_t b = 0; b < n_blocks; ++b) {
      const uint8_t *blk = raw_q + (start_block + b) * 34u;
      uint16_t scale_h = 0;
      memcpy(&scale_h, blk, 2);
      float scale = _vira_fp16_to_fp32(scale_h);
      const int8_t *qs = (const int8_t *)(blk + 2);
      float *dst = dst_out + b * 32u;
      for (int32_t j = 0; j < 32; ++j) {
        dst[j] = scale * (float)qs[j];
      }
    }
  } else if (gguf_dtype == 12) {
    for (size_t b = 0; b < n_blocks; ++b) {
      const uint8_t *blk = raw_q + (start_block + b) * 144u;
      uint16_t d_h = 0, dmin_h = 0;
      memcpy(&d_h, blk, 2);
      memcpy(&dmin_h, blk + 2, 2);
      float d = _vira_fp16_to_fp32(d_h);
      float dmin = _vira_fp16_to_fp32(dmin_h);
      const uint8_t *scales = blk + 4;
      const uint8_t *qs = blk + 16;
      float *dst = dst_out + b * 256u;
      for (int32_t ib = 0; ib < 8; ++ib) {
        uint8_t sc = 0, m_sc = 0;
        _vira_gguf_k4_scale_min(ib, scales, &sc, &m_sc);
        float d_sub = d * (float)sc;
        float min_sub = dmin * (float)m_sc;
        const uint8_t *qs_sub = qs + ((ib >> 1) * 32);
        int32_t shift = (ib & 1) * 4;
        float *dst_sub = dst + ib * 32;
        for (int32_t l = 0; l < 32; ++l) {
          dst_sub[l] =
              d_sub * (float)((qs_sub[l] >> shift) & 0x0Fu) - min_sub;
        }
      }
    }
  } else if (gguf_dtype == 13) {
    for (size_t b = 0; b < n_blocks; ++b) {
      const uint8_t *blk = raw_q + (start_block + b) * 176u;
      uint16_t d_h = 0, dmin_h = 0;
      memcpy(&d_h, blk, 2);
      memcpy(&dmin_h, blk + 2, 2);
      float d = _vira_fp16_to_fp32(d_h);
      float dmin = _vira_fp16_to_fp32(dmin_h);
      const uint8_t *scales = blk + 4;
      const uint8_t *qh = blk + 16;
      const uint8_t *qs = blk + 48;
      float *dst = dst_out + b * 256u;
      for (int32_t ib = 0; ib < 8; ++ib) {
        uint8_t sc = 0, m_sc = 0;
        _vira_gguf_k4_scale_min(ib, scales, &sc, &m_sc);
        float d_sub = d * (float)sc;
        float min_sub = dmin * (float)m_sc;
        const uint8_t *qs_sub = qs + ((ib >> 1) * 32);
        int32_t shift = (ib & 1) * 4;
        uint8_t u_mask = (uint8_t)(1u << ib);
        float *dst_sub = dst + ib * 32;
        for (int32_t l = 0; l < 32; ++l) {
          uint32_t q = (uint32_t)((qs_sub[l] >> shift) & 0x0Fu) |
                       ((qh[l] & u_mask) ? 16u : 0u);
          dst_sub[l] = d_sub * (float)q - min_sub;
        }
      }
    }
  } else if (gguf_dtype == 14) {
    for (size_t b = 0; b < n_blocks; ++b) {
      const uint8_t *blk = raw_q + (start_block + b) * 210u;
      uint16_t d_h = 0;
      memcpy(&d_h, blk + 208, 2);
      float d = _vira_fp16_to_fp32(d_h);
      for (int32_t h = 0; h < 2; ++h) {
        const uint8_t *ql_h = blk + h * 64;
        const uint8_t *qh_h = blk + 128 + h * 32;
        const int8_t *sc_h = (const int8_t *)(blk + 192) + h * 8;
        float *dst_h = dst_out + b * 256u + h * 128;
        for (int32_t l = 0; l < 32; ++l) {
          int32_t is = l >> 4;
          int32_t q1 =
              (int32_t)((ql_h[l + 0] & 0x0Fu) | (((qh_h[l] >> 0) & 3u) << 4)) -
              32;
          int32_t q2 = (int32_t)((ql_h[l + 32] & 0x0Fu) |
                                 (((qh_h[l] >> 2) & 3u) << 4)) -
                       32;
          int32_t q3 =
              (int32_t)((ql_h[l + 0] >> 4) | (((qh_h[l] >> 4) & 3u) << 4)) -
              32;
          int32_t q4 =
              (int32_t)((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3u) << 4)) -
              32;
          dst_h[l + 0] = d * (float)sc_h[is + 0] * (float)q1;
          dst_h[l + 32] = d * (float)sc_h[is + 2] * (float)q2;
          dst_h[l + 64] = d * (float)sc_h[is + 4] * (float)q3;
          dst_h[l + 96] = d * (float)sc_h[is + 6] * (float)q4;
        }
      }
    }
  }
}

typedef struct {
  const vira_f32 *host_ptr;
  void *raw_q;
  size_t raw_bytes;
  size_t elems;
  int32_t rows;
  int32_t cols;
  uint32_t gguf_dtype;
  uint8_t fp32_state; // 0 = deferred, 1 = dequantizing, 2 = ready
  uint8_t occupied;   // 0 = empty, 1 = active, 2 = tombstone
} _vira_rt_quant_entry;

static pthread_mutex_t _vira_rt_quant_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t _vira_rt_quant_cond = PTHREAD_COND_INITIALIZER;
static struct {
  _vira_rt_quant_entry *entries;
  size_t cap;
  size_t count;
  size_t tombstones;
} _vira_rt_quant_table = {NULL, 0, 0, 0};
static size_t _vira_rt_quant_active_count = 0;
static uint8_t _vira_rt_quant_bloom[4096] = {0};

static inline size_t _vira_rt_quant_hash(const vira_f32 *ptr) {
  uint64_t x = (uint64_t)(uintptr_t)ptr;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return (size_t)(x ^ (x >> 31));
}

static inline bool _vira_rt_quant_may_exist(const vira_f32 *ptr) {
  if (!ptr)
    return false;
  if (__atomic_load_n(&_vira_rt_quant_active_count, __ATOMIC_ACQUIRE) == 0)
    return false;
  size_t h = _vira_rt_quant_hash(ptr);
  return __atomic_load_n(&_vira_rt_quant_bloom[h & 4095u], __ATOMIC_ACQUIRE) !=
             0 &&
         __atomic_load_n(&_vira_rt_quant_bloom[(h >> 12) & 4095u],
                           __ATOMIC_ACQUIRE) != 0;
}

static inline void _vira_rt_quant_rehash_locked(size_t new_cap) {
  _vira_rt_quant_entry *old = _vira_rt_quant_table.entries;
  size_t old_cap = _vira_rt_quant_table.cap;
  _vira_rt_quant_entry *nw =
      (_vira_rt_quant_entry *)calloc(new_cap, sizeof(_vira_rt_quant_entry));
  if (!nw) {
    fprintf(stderr, "vira panic: out of memory in _vira_rt_quant_table\n");
    exit(1);
  }
  size_t mask = new_cap - 1u;
  size_t cnt = 0;
  if (old != NULL) {
    for (size_t i = 0; i < old_cap; ++i) {
      if (old[i].occupied == 1) {
        size_t idx = _vira_rt_quant_hash(old[i].host_ptr) & mask;
        while (nw[idx].occupied == 1) {
          idx = (idx + 1u) & mask;
        }
        nw[idx] = old[i];
        cnt++;
      }
    }
    free(old);
  }
  _vira_rt_quant_table.entries = nw;
  _vira_rt_quant_table.cap = new_cap;
  _vira_rt_quant_table.count = cnt;
  _vira_rt_quant_table.tombstones = 0;
}

static inline _vira_rt_quant_entry *_vira_rt_quant_find_locked(
    const vira_f32 *ptr) {
  if (!ptr || !_vira_rt_quant_table.entries || _vira_rt_quant_table.cap == 0)
    return NULL;
  size_t mask = _vira_rt_quant_table.cap - 1u;
  size_t idx = _vira_rt_quant_hash(ptr) & mask;
  for (size_t probe = 0; probe < _vira_rt_quant_table.cap; ++probe) {
    _vira_rt_quant_entry *e = &_vira_rt_quant_table.entries[idx];
    if (e->occupied == 0)
      return NULL;
    if (e->occupied == 1 && e->host_ptr == ptr)
      return e;
    idx = (idx + 1u) & mask;
  }
  return NULL;
}

// Takes ownership of `raw_q` (previously malloc'd) and records it in `_vira_active_heap_allocs`.
static inline void _vira_rt_quant_register(const vira_f32 *host_ptr,
                                           uint32_t gguf_dtype, void *raw_q,
                                           size_t raw_bytes, size_t elems,
                                           int32_t rows, int32_t cols,
                                           bool fp32_ready) {
  if (!host_ptr || !raw_q || raw_bytes == 0) {
    free(raw_q);
    return;
  }
  void *old_raw = NULL;
  pthread_mutex_lock(&_vira_rt_quant_lock);
  if (_vira_rt_quant_table.cap == 0) {
    _vira_rt_quant_rehash_locked(1024);
  } else if ((_vira_rt_quant_table.count + _vira_rt_quant_table.tombstones + 1u) * 10u >=
             _vira_rt_quant_table.cap * 7u) {
    size_t target_cap = (_vira_rt_quant_table.count * 2u >= _vira_rt_quant_table.cap)
                            ? (_vira_rt_quant_table.cap * 2u)
                            : _vira_rt_quant_table.cap;
    _vira_rt_quant_rehash_locked(target_cap);
  }
  size_t mask = _vira_rt_quant_table.cap - 1u;
  size_t idx = _vira_rt_quant_hash(host_ptr) & mask;
  ssize_t first_tomb = -1;
  size_t target_idx = idx;
  bool updated = false;
  for (size_t probe = 0; probe < _vira_rt_quant_table.cap; ++probe) {
    _vira_rt_quant_entry *e = &_vira_rt_quant_table.entries[idx];
    if (e->occupied == 0) {
      target_idx = (first_tomb >= 0) ? (size_t)first_tomb : idx;
      break;
    }
    if (e->occupied == 2) {
      if (first_tomb < 0)
        first_tomb = (ssize_t)idx;
    } else if (e->host_ptr == host_ptr) {
      while (e->fp32_state == 1) {
        pthread_cond_wait(&_vira_rt_quant_cond, &_vira_rt_quant_lock);
        e = _vira_rt_quant_find_locked(host_ptr);
        if (!e)
          break;
      }
      if (e) {
        old_raw = e->raw_q;
        e->raw_q = raw_q;
        e->raw_bytes = raw_bytes;
        e->elems = elems;
        e->rows = rows;
        e->cols = cols;
        e->gguf_dtype = gguf_dtype;
        e->fp32_state = fp32_ready ? 2 : 0;
        updated = true;
      }
      break;
    }
    idx = (idx + 1u) & mask;
  }
  if (!updated) {
    _vira_rt_quant_entry *slot = &_vira_rt_quant_table.entries[target_idx];
    if (slot->occupied == 2 && _vira_rt_quant_table.tombstones > 0) {
      _vira_rt_quant_table.tombstones--;
    }
    slot->host_ptr = host_ptr;
    slot->raw_q = raw_q;
    slot->raw_bytes = raw_bytes;
    slot->elems = elems;
    slot->rows = rows;
    slot->cols = cols;
    slot->gguf_dtype = gguf_dtype;
    slot->fp32_state = fp32_ready ? 2 : 0;
    slot->occupied = 1;
    _vira_rt_quant_table.count++;
  }
  size_t h = _vira_rt_quant_hash(host_ptr);
  __atomic_store_n(&_vira_rt_quant_bloom[h & 4095u], 1, __ATOMIC_RELEASE);
  __atomic_store_n(&_vira_rt_quant_bloom[(h >> 12) & 4095u], 1,
                   __ATOMIC_RELEASE);
  __atomic_store_n(&_vira_rt_quant_active_count, _vira_rt_quant_table.count,
                   __ATOMIC_RELEASE);
  _vira_record_alloc();
  pthread_mutex_unlock(&_vira_rt_quant_lock);
  if (old_raw) {
    free(old_raw);
    _vira_record_free();
  }
}

static inline bool _vira_rt_quant_try_get(const vira_f32 *host_ptr,
                                          uint32_t *out_dtype,
                                          const void **out_raw_q,
                                          size_t *out_raw_bytes,
                                          size_t *out_elems) {
  if (!_vira_rt_quant_may_exist(host_ptr))
    return false;
  bool found = false;
  pthread_mutex_lock(&_vira_rt_quant_lock);
  _vira_rt_quant_entry *e = _vira_rt_quant_find_locked(host_ptr);
  if (e && e->raw_q != NULL) {
    if (out_dtype) *out_dtype = e->gguf_dtype;
    if (out_raw_q) *out_raw_q = e->raw_q;
    if (out_raw_bytes) *out_raw_bytes = e->raw_bytes;
    if (out_elems) *out_elems = e->elems;
    found = true;
  }
  pthread_mutex_unlock(&_vira_rt_quant_lock);
  return found;
}

static inline void _vira_rt_quant_free(const vira_f32 *host_ptr) {
  if (!_vira_rt_quant_may_exist(host_ptr))
    return;
  void *to_free = NULL;
  pthread_mutex_lock(&_vira_rt_quant_lock);
  _vira_rt_quant_entry *e = _vira_rt_quant_find_locked(host_ptr);
  while (e && e->fp32_state == 1) {
    pthread_cond_wait(&_vira_rt_quant_cond, &_vira_rt_quant_lock);
    e = _vira_rt_quant_find_locked(host_ptr);
  }
  if (e) {
    to_free = e->raw_q;
    memset(e, 0, sizeof(*e));
    e->occupied = 2;
    if (_vira_rt_quant_table.count > 0)
      _vira_rt_quant_table.count--;
    _vira_rt_quant_table.tombstones++;
    __atomic_store_n(&_vira_rt_quant_active_count, _vira_rt_quant_table.count,
                     __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&_vira_rt_quant_lock);
  if (to_free != NULL) {
    free(to_free);
    _vira_record_free();
  }
}

static inline void _vira_rt_quant_remove(const vira_f32 *host_ptr) {
  _vira_rt_quant_free(host_ptr);
}

static inline void _vira_rt_quant_shutdown(void) {
  pthread_mutex_lock(&_vira_rt_quant_lock);
  if (_vira_rt_quant_table.count == 0 && _vira_rt_quant_table.entries != NULL) {
    free(_vira_rt_quant_table.entries);
    _vira_rt_quant_table.entries = NULL;
    _vira_rt_quant_table.cap = 0;
    _vira_rt_quant_table.tombstones = 0;
    memset(_vira_rt_quant_bloom, 0, sizeof(_vira_rt_quant_bloom));
    __atomic_store_n(&_vira_rt_quant_active_count, 0, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&_vira_rt_quant_lock);
}

static inline void _vira_rt_quant_clone_entry(const vira_f32 *dst_ptr,
                                              const vira_f32 *src_ptr) {
  if (!dst_ptr || !src_ptr || dst_ptr == src_ptr ||
      !_vira_rt_quant_may_exist(src_ptr))
    return;
  pthread_mutex_lock(&_vira_rt_quant_lock);
  _vira_rt_quant_entry *e = _vira_rt_quant_find_locked(src_ptr);
  if (!e || !e->raw_q || e->raw_bytes == 0) {
    pthread_mutex_unlock(&_vira_rt_quant_lock);
    return;
  }
  uint32_t dtype = e->gguf_dtype;
  size_t raw_bytes = e->raw_bytes;
  size_t elems = e->elems;
  int32_t rows = e->rows;
  int32_t cols = e->cols;
  bool ready = (e->fp32_state == 2);
  void *copy_buf = malloc(raw_bytes);
  if (!copy_buf) {
    pthread_mutex_unlock(&_vira_rt_quant_lock);
    return;
  }
  memcpy(copy_buf, e->raw_q, raw_bytes);
  pthread_mutex_unlock(&_vira_rt_quant_lock);
  _vira_rt_quant_register(dst_ptr, dtype, copy_buf, raw_bytes, elems, rows, cols,
                          ready);
}

static inline bool _vira_rt_quant_try_clone_lazy(vira_tensor_f32 src,
                                                 vira_tensor_f32 *out_copy) {
  if (!out_copy || !_vira_rt_quant_may_exist(src.data))
    return false;
  pthread_mutex_lock(&_vira_rt_quant_lock);
  _vira_rt_quant_entry *e = _vira_rt_quant_find_locked(src.data);
  if (!e || e->fp32_state != 0 || !e->raw_q || e->raw_bytes == 0) {
    pthread_mutex_unlock(&_vira_rt_quant_lock);
    return false;
  }
  uint32_t dtype = e->gguf_dtype;
  size_t raw_bytes = e->raw_bytes;
  size_t elems = e->elems;
  int32_t rows = e->rows;
  int32_t cols = e->cols;
  void *copy_buf = malloc(raw_bytes);
  if (!copy_buf) {
    pthread_mutex_unlock(&_vira_rt_quant_lock);
    return false;
  }
  memcpy(copy_buf, e->raw_q, raw_bytes);
  pthread_mutex_unlock(&_vira_rt_quant_lock);

  vira_tensor_f32 copy = _vira_tensor_f32_alloc_uninit_nd(src.n_dims, src.dims);
  _vira_rt_quant_register(copy.data, dtype, copy_buf, raw_bytes, elems, rows,
                          cols, false);
  *out_copy = copy;
  return true;
}

static inline void _vira_rt_ensure_fp32_host_ptr(const vira_f32 *ptr) {
  if (!_vira_rt_quant_may_exist(ptr))
    return;
  pthread_mutex_lock(&_vira_rt_quant_lock);
  _vira_rt_quant_entry *e = _vira_rt_quant_find_locked(ptr);
  if (!e || e->fp32_state == 2) {
    pthread_mutex_unlock(&_vira_rt_quant_lock);
    return;
  }
  while (e && e->fp32_state == 1) {
    pthread_cond_wait(&_vira_rt_quant_cond, &_vira_rt_quant_lock);
    e = _vira_rt_quant_find_locked(ptr);
  }
  if (!e || e->fp32_state == 2) {
    pthread_mutex_unlock(&_vira_rt_quant_lock);
    return;
  }
  e->fp32_state = 1;
  uint32_t dtype = e->gguf_dtype;
  const uint8_t *raw_q = (const uint8_t *)e->raw_q;
  size_t elems = e->elems;
  pthread_mutex_unlock(&_vira_rt_quant_lock);

  size_t qk = _vira_gguf_dtype_qk(dtype);
  if (raw_q && qk > 0 && elems > 0 && (elems % qk) == 0) {
    _vira_rt_dequant_blocks_to_fp32(dtype, raw_q, 0, elems / qk,
                                    (float *)(uintptr_t)ptr);
  }

  pthread_mutex_lock(&_vira_rt_quant_lock);
  e = _vira_rt_quant_find_locked(ptr);
  if (e) {
    e->fp32_state = 2;
  }
  pthread_cond_broadcast(&_vira_rt_quant_cond);
  pthread_mutex_unlock(&_vira_rt_quant_lock);
}

static inline void _vira_tensor_sync_host_f32(const vira_f32 *ptr) {
  if (!ptr)
    return;
  _vira_gpu_sync_host(ptr);
  _vira_rt_ensure_fp32_host_ptr(ptr);
}

#endif // VIRA_RT_TENSOR_QUANT_REGISTRY_H
