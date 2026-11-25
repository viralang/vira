#ifndef VIRA_RT_WEIGHTS_GGUF_H
#define VIRA_RT_WEIGHTS_GGUF_H

#include <sys/stat.h>

// ============================================================================
// GGUF v2/v3 PARSER & STREAMING QUANTIZED TENSOR LOADER (HARDENED, T-012)
// ============================================================================

static inline int _vira_gguf_seek(FILE *f, int64_t off, int whence) {
#ifdef _WIN32
  return _fseeki64(f, off, whence);
#else
  return fseeko(f, (off_t)off, whence);
#endif
}

static inline int64_t _vira_gguf_tell(FILE *f) {
#ifdef _WIN32
  return (int64_t)_ftelli64(f);
#else
  return (int64_t)ftello(f);
#endif
}

static inline bool _vira_gguf_file_stat(const char *path, int64_t *out_size,
                                        uint64_t *out_mtime,
                                        uint64_t *out_ino) {
#ifdef _WIN32
  struct __stat64 st;
  if (_stat64(path, &st) != 0 || st.st_size < 0) return false;
  *out_size = (int64_t)st.st_size;
  *out_mtime = (uint64_t)st.st_mtime;
  *out_ino = (uint64_t)st.st_ino;
  return true;
#else
  struct stat st;
  if (stat(path, &st) != 0 || st.st_size < 0) return false;
  *out_size = (int64_t)st.st_size;
  *out_mtime = (uint64_t)st.st_mtime;
  *out_ino = ((uint64_t)st.st_dev << 32) ^ (uint64_t)st.st_ino;
  return true;
#endif
}

static inline bool _vira_gguf_skip_bytes(FILE *f, uint64_t n,
                                         int64_t file_size) {
  int64_t cur = _vira_gguf_tell(f);
  if (cur < 0 || cur > file_size) return false;
  if (n > (uint64_t)(file_size - cur)) return false;
  if (n == 0) return true;
  return _vira_gguf_seek(f, cur + (int64_t)n, SEEK_SET) == 0;
}

/**
 * Advances the file cursor past a GGUF metadata value with full bounds and EOF checks.
 */
static inline bool _vira_gguf_skip_val(FILE *f, uint32_t val_type,
                                       int64_t file_size, int depth) {
  if (depth > 16) return false;
  uint8_t scratch[16];
  switch (val_type) {
  case 0:
  case 1:
  case 7:
    return fread(scratch, 1, 1, f) == 1;
  case 2:
  case 3:
    return fread(scratch, 1, 2, f) == 2;
  case 4:
  case 5:
  case 6:
    return fread(scratch, 1, 4, f) == 4;
  case 10:
  case 11:
  case 12:
    return fread(scratch, 1, 8, f) == 8;
  case 8: {
    uint64_t slen = 0;
    if (fread(&slen, sizeof(uint64_t), 1, f) != 1) return false;
    return _vira_gguf_skip_bytes(f, slen, file_size);
  }
  case 9: {
    uint32_t elem_type = 0;
    uint64_t elem_count = 0;
    if (fread(&elem_type, sizeof(uint32_t), 1, f) != 1 ||
        fread(&elem_count, sizeof(uint64_t), 1, f) != 1) {
      return false;
    }
    size_t fixed_sz = 0;
    if (elem_type == 0 || elem_type == 1 || elem_type == 7)
      fixed_sz = 1;
    else if (elem_type == 2 || elem_type == 3)
      fixed_sz = 2;
    else if (elem_type == 4 || elem_type == 5 || elem_type == 6)
      fixed_sz = 4;
    else if (elem_type == 10 || elem_type == 11 || elem_type == 12)
      fixed_sz = 8;
    if (fixed_sz > 0) {
      if (elem_count > (uint64_t)file_size / (uint64_t)fixed_sz) return false;
      return _vira_gguf_skip_bytes(f, elem_count * (uint64_t)fixed_sz,
                                   file_size);
    }
    if (elem_type != 8 && elem_type != 9) return false;
    // Each nested string/array occupies at least 8 bytes on disk.
    if (elem_count > (uint64_t)file_size / 8u) return false;
    for (uint64_t i = 0; i < elem_count; ++i) {
      if (!_vira_gguf_skip_val(f, elem_type, file_size, depth + 1)) {
        return false;
      }
    }
    return true;
  }
  default:
    return false;
  }
}

static inline uint64_t _vira_gguf_fnv1a(const char *s, size_t len) {
  uint64_t h = 14695981039346656037ULL;
  for (size_t i = 0; i < len; ++i) {
    h ^= (uint8_t)s[i];
    h *= 1099511628211ULL;
  }
  return h;
}

typedef struct {
  char *name;
  size_t name_len;
  uint64_t name_hash;
  uint32_t dtype;
  int32_t ndims;
  int32_t dims[4];
  int64_t file_pos;
} _vira_gguf_tensor_entry;

#define VIRA_GGUF_DIR_CACHE_SLOTS 4

typedef struct {
  char *path;
  int64_t file_size;
  uint64_t file_mtime;
  uint64_t file_ino;
  FILE *fp;
  int32_t count;
  _vira_gguf_tensor_entry *entries;
  int32_t *hash_slots; // 1-based index into entries, 0 = empty
  size_t hash_cap;     // power of 2
  uint64_t last_used;
} _vira_gguf_dir_slot;

static pthread_mutex_t _vira_gguf_dir_mu = PTHREAD_MUTEX_INITIALIZER;
static _vira_gguf_dir_slot _vira_gguf_dir_slots[VIRA_GGUF_DIR_CACHE_SLOTS];
static uint64_t _vira_gguf_dir_clock = 0;

static inline void _vira_gguf_dir_slot_clear_locked(_vira_gguf_dir_slot *s) {
  if (s->fp) {
    fclose(s->fp);
    s->fp = NULL;
  }
  if (s->entries) {
    for (int32_t i = 0; i < s->count; ++i) {
      free(s->entries[i].name);
    }
    free(s->entries);
    s->entries = NULL;
  }
  free(s->hash_slots);
  s->hash_slots = NULL;
  s->hash_cap = 0;
  free(s->path);
  s->path = NULL;
  s->count = 0;
  s->file_size = 0;
  s->file_mtime = 0;
  s->file_ino = 0;
  s->last_used = 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((destructor))
#endif
static void _vira_gguf_dir_shutdown(void) {
  pthread_mutex_lock(&_vira_gguf_dir_mu);
  for (int i = 0; i < VIRA_GGUF_DIR_CACHE_SLOTS; ++i) {
    _vira_gguf_dir_slot_clear_locked(&_vira_gguf_dir_slots[i]);
  }
  pthread_mutex_unlock(&_vira_gguf_dir_mu);
}

static inline _vira_gguf_tensor_entry *_vira_gguf_dir_lookup_locked(
    const _vira_gguf_dir_slot *s, const char *name, size_t name_len) {
  if (!s->hash_slots || s->hash_cap == 0) {
    return NULL;
  }
  uint64_t h = _vira_gguf_fnv1a(name, name_len);
  size_t mask = s->hash_cap - 1u;
  size_t idx = (size_t)h & mask;
  for (size_t probe = 0; probe < s->hash_cap; ++probe) {
    int32_t slot = s->hash_slots[idx];
    if (slot == 0) return NULL;
    _vira_gguf_tensor_entry *ent = &s->entries[slot - 1];
    if (ent->name_hash == h && ent->name_len == name_len &&
        ent->name != NULL && memcmp(ent->name, name, name_len) == 0) {
      return ent;
    }
    idx = (idx + 1u) & mask;
  }
  return NULL;
}

/**
 * Parses a GGUF v2/v3 binary file, locates the requested tensor, and streams it to GPU VRAM or dequantizes to CPU SIMD.
 */
static inline Result_tensor_f32 vira_gguf_load_tensor(vira_string path,
                                                      vira_string tensor_name) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  char stack_name[256];
  vira_cstr_buf nbuf =
      vira_str_to_cstr_buf(tensor_name, stack_name, sizeof(stack_name));
  size_t nlen = strlen(nbuf.ptr);

  int64_t file_size = 0;
  uint64_t file_mtime = 0, file_ino = 0;
  if (!_vira_gguf_file_stat(pbuf.ptr, &file_size, &file_mtime, &file_ino)) {
    vira_string err =
        vira_str_format("failed to open GGUF file '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  pthread_mutex_lock(&_vira_gguf_dir_mu);

  _vira_gguf_dir_slot *dir_slot = NULL;
  for (int i = 0; i < VIRA_GGUF_DIR_CACHE_SLOTS; ++i) {
    _vira_gguf_dir_slot *cand = &_vira_gguf_dir_slots[i];
    if (cand->fp != NULL && cand->path != NULL && cand->entries != NULL &&
        strcmp(cand->path, pbuf.ptr) == 0 &&
        cand->file_size == file_size &&
        cand->file_mtime == file_mtime &&
        cand->file_ino == file_ino) {
      dir_slot = cand;
      dir_slot->last_used = ++_vira_gguf_dir_clock;
      break;
    }
  }

  if (!dir_slot) {
    _vira_gguf_dir_slot *victim = &_vira_gguf_dir_slots[0];
    for (int i = 0; i < VIRA_GGUF_DIR_CACHE_SLOTS; ++i) {
      if (_vira_gguf_dir_slots[i].path == NULL) {
        victim = &_vira_gguf_dir_slots[i];
        break;
      }
      if (_vira_gguf_dir_slots[i].last_used < victim->last_used) {
        victim = &_vira_gguf_dir_slots[i];
      }
    }
    _vira_gguf_dir_slot_clear_locked(victim);

    FILE *f = fopen(pbuf.ptr, "rb");
    if (!f) {
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_string err =
          vira_str_format("failed to open GGUF file '%s'", pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    setvbuf(f, NULL, _IOFBF, 65536);

    uint32_t magic = 0, version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) {
      fclose(f);
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_string err = vira_str_format("invalid GGUF magic in '%s'", pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    if (fread(&version, 4, 1, f) != 1 || (version != 2 && version != 3)) {
      fclose(f);
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_string err = vira_str_format("unsupported GGUF version %u in '%s'",
                                        version, pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    if (fread(&tensor_count, 8, 1, f) != 1 || fread(&kv_count, 8, 1, f) != 1) {
      fclose(f);
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_string err =
          vira_str_format("corrupt GGUF header in '%s'", pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    // Minimum entry sizes: KV entry >= 12 bytes, tensor info entry >= 24 bytes.
    if (kv_count > (uint64_t)file_size / 12u ||
        tensor_count > (uint64_t)file_size / 24u ||
        tensor_count > (uint64_t)INT32_MAX / 2u) {
      fclose(f);
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_string err = vira_str_format(
          "corrupt GGUF header counts (tensors=%llu, kvs=%llu) in '%s'",
          (unsigned long long)tensor_count, (unsigned long long)kv_count,
          pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }

    uint32_t alignment = 32;
    bool hdr_ok = true;
    for (uint64_t k = 0; k < kv_count; ++k) {
      uint64_t klen = 0;
      if (fread(&klen, 8, 1, f) != 1) { hdr_ok = false; break; }
      int64_t kpos = _vira_gguf_tell(f);
      if (kpos < 0 || klen > (uint64_t)(file_size - kpos)) { hdr_ok = false; break; }
      char key_buf[64];
      size_t to_read = klen < sizeof(key_buf) - 1u ? (size_t)klen : sizeof(key_buf) - 1u;
      if (to_read > 0 && fread(key_buf, 1, to_read, f) != to_read) { hdr_ok = false; break; }
      key_buf[to_read] = '\0';
      if (klen > to_read && !_vira_gguf_skip_bytes(f, klen - to_read, file_size)) {
        hdr_ok = false; break;
      }

      uint32_t val_type = 0;
      if (fread(&val_type, 4, 1, f) != 1) { hdr_ok = false; break; }

      if (klen == 17 && strcmp(key_buf, "general.alignment") == 0 && val_type == 4) {
        uint32_t a = 32;
        if (fread(&a, 4, 1, f) != 1) { hdr_ok = false; break; }
        if (a > 0) alignment = a;
      } else {
        if (!_vira_gguf_skip_val(f, val_type, file_size, 0)) { hdr_ok = false; break; }
      }
    }

    if (!hdr_ok) {
      fclose(f);
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_string err =
          vira_str_format("corrupt GGUF metadata in '%s'", pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }

    size_t cap = 16u;
    while (cap < (size_t)tensor_count * 2u + 8u) cap <<= 1;
    _vira_gguf_tensor_entry *entries = (_vira_gguf_tensor_entry *)calloc(
        (size_t)(tensor_count > 0 ? tensor_count : 1u),
        sizeof(_vira_gguf_tensor_entry));
    int32_t *hash_slots = (int32_t *)calloc(cap, sizeof(int32_t));
    char *path_copy = strdup(pbuf.ptr);
    if (!entries || !hash_slots || !path_copy) {
      free(entries);
      free(hash_slots);
      free(path_copy);
      fclose(f);
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_string err = vira_str_format(
          "out of memory parsing GGUF directory in '%s'", pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }

    int32_t parsed = 0;
    for (uint64_t t = 0; t < tensor_count; ++t) {
      uint64_t n_len = 0;
      if (fread(&n_len, 8, 1, f) != 1) { hdr_ok = false; break; }
      int64_t npos = _vira_gguf_tell(f);
      if (npos < 0 || n_len > (uint64_t)(file_size - npos) || n_len > 65536u) {
        hdr_ok = false; break;
      }
      char *name_buf = (char *)malloc((size_t)n_len + 1u);
      if (!name_buf) { hdr_ok = false; break; }
      if (n_len > 0 && fread(name_buf, 1, (size_t)n_len, f) != (size_t)n_len) {
        free(name_buf);
        hdr_ok = false; break;
      }
      name_buf[n_len] = '\0';

      uint32_t n_dims = 0;
      if (fread(&n_dims, 4, 1, f) != 1 || n_dims > 16u) {
        free(name_buf);
        hdr_ok = false; break;
      }

      uint64_t dims[4] = {1, 1, 1, 1};
      for (uint32_t d = 0; d < n_dims && d < 4; ++d) {
        if (fread(&dims[d], 8, 1, f) != 1) { hdr_ok = false; break; }
      }
      if (!hdr_ok) { free(name_buf); break; }
      if (n_dims > 4 && !_vira_gguf_skip_bytes(f, 8u * (uint64_t)(n_dims - 4u), file_size)) {
        free(name_buf);
        hdr_ok = false; break;
      }

      uint32_t dtype = 0;
      uint64_t offset = 0;
      if (fread(&dtype, 4, 1, f) != 1 || fread(&offset, 8, 1, f) != 1) {
        free(name_buf);
        hdr_ok = false; break;
      }

      _vira_gguf_tensor_entry *ent = &entries[t];
      ent->name = name_buf;
      ent->name_len = (size_t)n_len;
      ent->name_hash = _vira_gguf_fnv1a(name_buf, (size_t)n_len);
      ent->dtype = dtype;
      uint32_t clamped_nd = (n_dims == 0) ? 1 : ((n_dims > 4) ? 4 : n_dims);
      ent->ndims = (int32_t)clamped_nd;
      for (uint32_t d = 0; d < clamped_nd; ++d) {
        uint64_t dv = dims[clamped_nd - 1 - d];
        ent->dims[d] = (dv == 0) ? 1 : (dv > (uint64_t)INT32_MAX ? -1 : (int32_t)dv);
      }
      if (offset > (uint64_t)file_size) {
        ent->file_pos = -1;
      } else {
        ent->file_pos = (int64_t)offset;
      }

      size_t mask = cap - 1u;
      size_t idx = (size_t)ent->name_hash & mask;
      while (hash_slots[idx] != 0) idx = (idx + 1u) & mask;
      hash_slots[idx] = (int32_t)t + 1;
      parsed = (int32_t)t + 1;
    }

    int64_t cur_pos = hdr_ok ? _vira_gguf_tell(f) : -1;
    if (!hdr_ok || cur_pos < 0 || cur_pos > file_size) {
      for (int32_t i = 0; i < parsed; ++i) free(entries[i].name);
      free(entries);
      free(hash_slots);
      free(path_copy);
      fclose(f);
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_string err =
          vira_str_format("corrupt GGUF tensor directory in '%s'", pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }

    int64_t align64 = (int64_t)alignment;
    int64_t pad = (align64 - (cur_pos % align64)) % align64;
    int64_t data_start = cur_pos + pad;
    if (data_start < cur_pos) {
      for (int32_t i = 0; i < parsed; ++i) free(entries[i].name);
      free(entries);
      free(hash_slots);
      free(path_copy);
      fclose(f);
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_string err =
          vira_str_format("corrupt GGUF data offset in '%s'", pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    for (int32_t t = 0; t < parsed; ++t) {
      if (entries[t].file_pos < 0 || data_start > file_size ||
          entries[t].file_pos > file_size - data_start) {
        entries[t].file_pos = -1;
      } else {
        entries[t].file_pos += data_start;
      }
    }

    victim->path = path_copy;
    victim->file_size = file_size;
    victim->file_mtime = file_mtime;
    victim->file_ino = file_ino;
    victim->fp = f;
    victim->count = parsed;
    victim->entries = entries;
    victim->hash_slots = hash_slots;
    victim->hash_cap = cap;
    victim->last_used = ++_vira_gguf_dir_clock;
    dir_slot = victim;
  }

  _vira_gguf_tensor_entry *ent = _vira_gguf_dir_lookup_locked(dir_slot, nbuf.ptr, nlen);
  if (!ent) {
    pthread_mutex_unlock(&_vira_gguf_dir_mu);
    vira_string err = vira_str_format(
        "tensor '%s' not found in GGUF file '%s'", nbuf.ptr, pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  uint32_t matched_type = ent->dtype;
  int64_t tensor_file_pos = ent->file_pos;
  int32_t matched_ndims = ent->ndims;
  int32_t matched_dims[4] = {ent->dims[0], ent->dims[1], ent->dims[2], ent->dims[3]};

  for (int32_t d = 0; d < matched_ndims; ++d) {
    if (matched_dims[d] < 0) {
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_string err = vira_str_format(
          "tensor '%s' in GGUF file '%s' has an axis longer than 2^31-1 elements",
          nbuf.ptr, pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
  }

  // Overflow-safe total element count and byte extent check BEFORE any allocation.
  size_t total = 1;
  bool dim_overflow = false;
  for (int32_t d = 0; d < matched_ndims; ++d) {
    size_t dv = (size_t)matched_dims[d];
    if (dv > 0 && total > SIZE_MAX / dv) { dim_overflow = true; break; }
    total *= dv;
  }
  size_t qk = _vira_gguf_dtype_qk(matched_type);
  size_t bs = _vira_gguf_dtype_bs(matched_type);
  size_t need_bytes = 0;
  bool supported = false;
  if (!dim_overflow) {
    if (matched_type == 0) {
      supported = (total <= SIZE_MAX / sizeof(float));
      need_bytes = total * sizeof(float);
    } else if (matched_type == 26) {
      supported = (total <= SIZE_MAX / sizeof(int32_t));
      need_bytes = total * sizeof(int32_t);
    } else if (qk > 0 && bs > 0 && (qk == 1 || (total % qk) == 0)) {
      size_t blocks = (qk == 1) ? total : (total / qk);
      supported = (blocks <= SIZE_MAX / bs);
      need_bytes = blocks * bs;
    }
  }

  if (dim_overflow || !supported) {
    pthread_mutex_unlock(&_vira_gguf_dir_mu);
    vira_string err = vira_str_format(
        "unsupported GGUF quantization dtype %u (total=%zu) for '%s'",
        matched_type, total, nbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  if (tensor_file_pos < 0 || tensor_file_pos > dir_slot->file_size ||
      (uint64_t)need_bytes > (uint64_t)(dir_slot->file_size - tensor_file_pos)) {
    pthread_mutex_unlock(&_vira_gguf_dir_mu);
    vira_string err = vira_str_format(
        "GGUF tensor '%s' extends past end of file (offset=%lld, bytes=%zu, file=%lld)",
        nbuf.ptr, (long long)tensor_file_pos, need_bytes,
        (long long)dir_slot->file_size);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  FILE *f = dir_slot->fp;
  if (_vira_gguf_seek(f, tensor_file_pos, SEEK_SET) != 0) {
    pthread_mutex_unlock(&_vira_gguf_dir_mu);
    vira_string err = vira_str_format(
        "failed to seek to GGUF tensor '%s' at %lld", nbuf.ptr,
        (long long)tensor_file_pos);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  vira_cstr_buf_free(&nbuf);
  vira_cstr_buf_free(&pbuf);

  bool deferred = (matched_type != 0 && matched_type != 26 && qk > 0 && bs > 0 &&
                   (qk == 1 || (total % qk) == 0) &&
                   matched_ndims >= 2 && total >= 65536u &&
                   _vira_lazy_host_fp32_enabled());

  vira_tensor_f32 t = deferred
      ? _vira_tensor_f32_alloc_reserved_nd(matched_ndims, matched_dims)
      : _vira_tensor_f32_alloc_uninit_nd(matched_ndims, matched_dims);

  if (matched_type == 0) {
    size_t nr = (total > 0) ? fread(t.data, sizeof(float), total, f) : 0;
    pthread_mutex_unlock(&_vira_gguf_dir_mu);
    if (nr != total) {
      vira_tensor_f32_free(&t);
      vira_string err = vira_str_format(
          "truncated GGUF F32 tensor (expected %zu floats, got %zu)", total,
          nr);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
  } else if (matched_type == 26) {
    int32_t *iv = (int32_t *)malloc(need_bytes > 0 ? need_bytes : 1u);
    size_t nr = (iv && total > 0) ? fread(iv, sizeof(int32_t), total, f) : 0;
    pthread_mutex_unlock(&_vira_gguf_dir_mu);
    if (!iv || nr != total) {
      free(iv);
      vira_tensor_f32_free(&t);
      vira_string err = vira_str_format(
          "truncated GGUF I32 tensor (expected %zu values, got %zu)", total, nr);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    for (size_t i = 0; i < total; ++i) t.data[i] = (float)iv[i];
    free(iv);
  } else {
    size_t blocks = (qk == 1) ? total : (total / qk);
    uint8_t *raw_q = (uint8_t *)malloc(need_bytes > 0 ? need_bytes : 1u);
    if (!raw_q) {
      pthread_mutex_unlock(&_vira_gguf_dir_mu);
      vira_tensor_f32_free(&t);
      vira_string err =
          vira_str_format("out of memory loading GGUF tensor (%zu bytes)",
                          need_bytes);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    size_t nr = (blocks > 0) ? fread(raw_q, bs, blocks, f) : 0;
    pthread_mutex_unlock(&_vira_gguf_dir_mu);
    if (nr != blocks) {
      free(raw_q);
      vira_tensor_f32_free(&t);
      vira_string err = vira_str_format(
          "truncated GGUF dtype %u tensor (expected %zu blocks, got %zu)",
          matched_type, blocks, nr);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    if (!deferred) {
      _vira_rt_dequant_blocks_to_fp32(matched_type, raw_q, 0, blocks, t.data);
    }
    _vira_rt_quant_register(t.data, matched_type, raw_q, need_bytes, total,
                            t.rows, t.cols, !deferred);
    _vira_gpu_register_quant(t.data, matched_type, raw_q, need_bytes, total);
  }

  return (Result_tensor_f32){
      .is_err = false, .err = vira_str_lit("", 0), .val = t};
}

#endif // VIRA_RT_WEIGHTS_GGUF_H
