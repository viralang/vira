#ifndef VIRA_RT_WEIGHTS_GGUF_H
#define VIRA_RT_WEIGHTS_GGUF_H

// ============================================================================
// GGUF v2/v3 PARSER & STREAMING QUANTIZED TENSOR LOADER
// ============================================================================

/**
 * Advances the file cursor past a GGUF metadata value according to its type descriptor.
 * @param f Open file stream.
 * @param val_type GGUF metadata value type tag.
 * Usage: Skipping metadata fields during GGUF header parsing.
 */
static inline void _vira_gguf_skip_val(FILE *f, uint32_t val_type) {
  uint8_t scratch[256];
  size_t _nr = 0;
  switch (val_type) {
  case 0:
  case 1:
  case 7:
    _nr = fread(scratch, 1, 1, f);
    break;
  case 2:
  case 3:
    _nr = fread(scratch, 1, 2, f);
    break;
  case 4:
  case 5:
  case 6:
    _nr = fread(scratch, 1, 4, f);
    break;
  case 10:
  case 11:
  case 12:
    _nr = fread(scratch, 1, 8, f);
    break;
  case 8: {
    uint64_t slen = 0;
    if (fread(&slen, sizeof(uint64_t), 1, f) == 1) {
      if (slen <= sizeof(scratch)) {
        if (slen > 0)
          _nr = fread(scratch, 1, (size_t)slen, f);
      } else {
        fseek(f, (long)slen, SEEK_CUR);
      }
    }
    break;
  }
  case 9: {
    uint32_t elem_type = 0;
    uint64_t elem_count = 0;
    if (fread(&elem_type, sizeof(uint32_t), 1, f) == 1 &&
        fread(&elem_count, sizeof(uint64_t), 1, f) == 1) {
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
#ifdef _WIN32
        _fseeki64(f, (int64_t)(elem_count * fixed_sz), SEEK_CUR);
#else
        fseeko(f, (off_t)(elem_count * fixed_sz), SEEK_CUR);
#endif
      } else {
        for (uint64_t i = 0; i < elem_count; ++i) {
          _vira_gguf_skip_val(f, elem_type);
        }
      }
    }
    break;
  }
  }
  (void)_nr;
}

typedef struct {
  char name[128];
  uint32_t dtype;
  int32_t ndims;
  int32_t dims[4];
  int64_t file_pos;
} _vira_gguf_tensor_entry;

static struct {
  char path[1024];
  int32_t count;
  _vira_gguf_tensor_entry *entries;
} _vira_gguf_dir_cache = {"", 0, NULL};

 

/**
 * Parses a GGUF v2/v3 binary file, locates the requested tensor, and streams it to GPU VRAM or dequantizes to CPU SIMD.
 * @param path Path to `.gguf` model file.
 * @param tensor_name Name of the tensor in the GGUF directory (e.g. "blk.0.attn_q.weight").
 * @return Result containing loaded tensor or error message.
 * Usage: High-speed streaming LLM model loading with automatic Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K GPU acceleration (`gguf_load_tensor(...)`).
 */
static inline Result_tensor_f32 vira_gguf_load_tensor(vira_string path,
                                                      vira_string tensor_name) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  char stack_name[256];
  vira_cstr_buf nbuf =
      vira_str_to_cstr_buf(tensor_name, stack_name, sizeof(stack_name));

  FILE *f = fopen(pbuf.ptr, "rb");
  if (!f) {
    vira_string err =
        vira_str_format("failed to open GGUF file '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }
  setvbuf(f, NULL, _IOFBF, 65536);

  if (strcmp(_vira_gguf_dir_cache.path, pbuf.ptr) != 0 ||
      _vira_gguf_dir_cache.entries == NULL) {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;

    if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) {
      fclose(f);
      vira_string err = vira_str_format("invalid GGUF magic in '%s'", pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    if (fread(&version, 4, 1, f) != 1 || (version != 2 && version != 3)) {
      fclose(f);
      vira_string err = vira_str_format("unsupported GGUF version %u in '%s'",
                                        version, pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    if (fread(&tensor_count, 8, 1, f) != 1 || fread(&kv_count, 8, 1, f) != 1) {
      fclose(f);
      vira_string err =
          vira_str_format("corrupt GGUF header in '%s'", pbuf.ptr);
      vira_cstr_buf_free(&nbuf);
      vira_cstr_buf_free(&pbuf);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }

    uint32_t alignment = 32;

    for (uint64_t k = 0; k < kv_count; ++k) {
      uint64_t klen = 0;
      if (fread(&klen, 8, 1, f) != 1)
        break;
      char key_buf[256];
      size_t to_read = klen < 255 ? (size_t)klen : 255;
      if (fread(key_buf, 1, to_read, f) != to_read)
        break;
      key_buf[to_read] = '\0';
      if (klen > to_read)
        fseek(f, (long)(klen - to_read), SEEK_CUR);

      uint32_t val_type = 0;
      if (fread(&val_type, 4, 1, f) != 1)
        break;

      if (strcmp(key_buf, "general.alignment") == 0 && val_type == 4) {
        uint32_t a = 32;
        if (fread(&a, 4, 1, f) == 1 && a > 0)
          alignment = a;
      } else {
        _vira_gguf_skip_val(f, val_type);
      }
    }

    if (_vira_gguf_dir_cache.entries) {
      free(_vira_gguf_dir_cache.entries);
      _vira_gguf_dir_cache.entries = NULL;
    }
    _vira_gguf_dir_cache.count = (int32_t)tensor_count;
    _vira_gguf_dir_cache.entries = (_vira_gguf_tensor_entry *)calloc(
        (size_t)(tensor_count > 0 ? tensor_count : 1),
        sizeof(_vira_gguf_tensor_entry));
    strncpy(_vira_gguf_dir_cache.path, pbuf.ptr,
            sizeof(_vira_gguf_dir_cache.path) - 1);
    _vira_gguf_dir_cache.path[sizeof(_vira_gguf_dir_cache.path) - 1] = '\0';

    for (uint64_t t = 0; t < tensor_count; ++t) {
      uint64_t n_len = 0;
      if (fread(&n_len, 8, 1, f) != 1)
        break;
      char cur_name[256];
      size_t to_read = n_len < 255 ? (size_t)n_len : 255;
      if (fread(cur_name, 1, to_read, f) != to_read)
        break;
      cur_name[to_read] = '\0';
      if (n_len > to_read)
        fseek(f, (long)(n_len - to_read), SEEK_CUR);

      uint32_t n_dims = 0;
      if (fread(&n_dims, 4, 1, f) != 1)
        break;

      uint64_t dims[4] = {1, 1, 1, 1};
      for (uint32_t d = 0; d < n_dims && d < 4; ++d) {
        if (fread(&dims[d], 8, 1, f) != 1)
          break;
      }
      if (n_dims > 4) {
        fseek(f, (long)(8 * (n_dims - 4)), SEEK_CUR);
      }

      uint32_t dtype = 0;
      uint64_t offset = 0;
      if (fread(&dtype, 4, 1, f) != 1)
        break;
      if (fread(&offset, 8, 1, f) != 1)
        break;

      _vira_gguf_tensor_entry *ent = &_vira_gguf_dir_cache.entries[t];
      strncpy(ent->name, cur_name, sizeof(ent->name) - 1);
      ent->name[sizeof(ent->name) - 1] = '\0';
      ent->dtype = dtype;
      uint32_t clamped_nd = (n_dims == 0) ? 1 : ((n_dims > 4) ? 4 : n_dims);
      ent->ndims = (int32_t)clamped_nd;
      for (uint32_t d = 0; d < clamped_nd; ++d) {
        int32_t dv = (int32_t)dims[clamped_nd - 1 - d];
        ent->dims[d] = (dv > 0) ? dv : 1;
      }
      ent->file_pos = (int64_t)offset;
    }

#ifdef _WIN32
    int64_t cur_pos = _ftelli64(f);
#else
    int64_t cur_pos = (int64_t)ftello(f);
#endif
    int64_t pad = (alignment - (cur_pos % alignment)) % alignment;
    int64_t data_start = cur_pos + pad;
    for (int32_t t = 0; t < _vira_gguf_dir_cache.count; ++t) {
      _vira_gguf_dir_cache.entries[t].file_pos += data_start;
    }
  }

  bool found = false;
  uint32_t matched_type = 0;
  int64_t tensor_file_pos = 0;
  int32_t matched_ndims = 2;
  int32_t matched_dims[4] = {1, 1, 1, 1};

  for (int32_t t = 0; t < _vira_gguf_dir_cache.count; ++t) {
    _vira_gguf_tensor_entry *ent = &_vira_gguf_dir_cache.entries[t];
    if (strcmp(ent->name, nbuf.ptr) == 0) {
      found = true;
      matched_type = ent->dtype;
      tensor_file_pos = ent->file_pos;
      matched_ndims = ent->ndims;
      for (int32_t d = 0; d < 4; ++d)
        matched_dims[d] = ent->dims[d];
      break;
    }
  }

  if (!found) {
    fclose(f);
    vira_string err = vira_str_format(
        "tensor '%s' not found in GGUF file '%s'", nbuf.ptr, pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  vira_cstr_buf_free(&nbuf);
  vira_cstr_buf_free(&pbuf);

#ifdef _WIN32
  _fseeki64(f, tensor_file_pos, SEEK_SET);
#else
  fseeko(f, tensor_file_pos, SEEK_SET);
#endif

  vira_tensor_f32 t =
      _vira_tensor_f32_alloc_uninit_nd(matched_ndims, matched_dims);
  size_t total = (size_t)t.rows * (size_t)t.cols;
  size_t qk = _vira_gguf_dtype_qk(matched_type);
  size_t bs = _vira_gguf_dtype_bs(matched_type);

  if (matched_type == 0) {
    size_t nr = fread(t.data, sizeof(float), total, f);
    if (nr != total) {
      fclose(f);
      vira_tensor_f32_free(&t);
      vira_string err = vira_str_format(
          "truncated GGUF F32 tensor (expected %zu floats, got %zu)", total,
          nr);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
  } else if (qk > 0 && bs > 0 && (qk == 1 || (total % qk) == 0)) {
    size_t blocks = (qk == 1) ? total : (total / qk);
    size_t raw_bytes = blocks * bs;
    uint8_t *raw_q = (uint8_t *)malloc(raw_bytes > 0 ? raw_bytes : 1u);
    if (!raw_q) {
      fclose(f);
      vira_tensor_f32_free(&t);
      vira_string err =
          vira_str_format("out of memory loading GGUF tensor (%zu bytes)",
                          raw_bytes);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    size_t nr = (blocks > 0) ? fread(raw_q, bs, blocks, f) : 0;
    if (nr != blocks) {
      free(raw_q);
      fclose(f);
      vira_tensor_f32_free(&t);
      vira_string err = vira_str_format(
          "truncated GGUF dtype %u tensor (expected %zu blocks, got %zu)",
          matched_type, blocks, nr);
      return (Result_tensor_f32){
          .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
    }
    bool deferred = (matched_ndims >= 2 && total >= 65536u &&
                     _vira_lazy_host_fp32_enabled());
    if (!deferred) {
      _vira_rt_dequant_blocks_to_fp32(matched_type, raw_q, 0, blocks, t.data);
    }
    _vira_rt_quant_register(t.data, matched_type, raw_q, raw_bytes, total,
                            t.rows, t.cols, !deferred);
    _vira_gpu_register_quant(t.data, matched_type, raw_q, raw_bytes, total);
  } else {
    fclose(f);
    vira_tensor_f32_free(&t);
    vira_string err = vira_str_format(
        "unsupported GGUF quantization dtype %u (total=%zu)", matched_type,
        total);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  fclose(f);
  return (Result_tensor_f32){
      .is_err = false, .err = vira_str_lit("", 0), .val = t};
}

#endif // VIRA_RT_WEIGHTS_GGUF_H
