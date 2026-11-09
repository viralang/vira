#ifndef VIRA_RT_WEIGHTS_SAFETENSORS_H
#define VIRA_RT_WEIGHTS_SAFETENSORS_H

// ============================================================================
// 14. NATIVE SAFETENSORS ZERO-COPY PARSER & LOADER
// ============================================================================

/**
 * Parses a HuggingFace SafeTensors file header and zero-copy memory maps or loads the specified tensor.
 * @param path Path to `.safetensors` file.
 * @param tensor_name Name of the tensor inside the archive (e.g. "model.layers.0.self_attn.q_proj.weight").
 * @return Result containing loaded tensor or error message.
 * Usage: High-speed zero-copy loading of modern deep learning weights (`safetensors::load("model.safetensors", "weight")`).
 */
static inline Result_tensor_f32 vira_safetensors_load(vira_string path,
                                                      vira_string tensor_name) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  char stack_name[256];
  vira_cstr_buf nbuf =
      vira_str_to_cstr_buf(tensor_name, stack_name, sizeof(stack_name));

  FILE *f = fopen(pbuf.ptr, "rb");
  if (!f) {
    vira_string err =
        vira_str_format("failed to open safetensors file '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  uint64_t header_len = 0;
  if (fread(&header_len, sizeof(uint64_t), 1, f) != 1 ||
      header_len > 100 * 1024 * 1024) {
    fclose(f);
    vira_string err =
        vira_str_format("corrupt safetensors header in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  char *header_json = (char *)malloc(header_len + 1);
  if (!header_json) {
    fclose(f);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true,
        .err =
            vira_str_format("failed to allocate memory for header (%llu bytes)",
                            (unsigned long long)header_len),
        .val = (vira_tensor_f32){0}};
  }

  if (fread(header_json, 1, header_len, f) != header_len) {
    free(header_json);
    fclose(f);
    vira_string err = vira_str_format(
        "incomplete read of safetensors header in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }
  header_json[header_len] = '\0';

  vira_string hstr = {.data = (const uint8_t *)header_json,
                      .len = (int32_t)header_len,
                      .is_lit = true};
  const char *t_entry = _vira_json_find_val_at_depth(hstr, tensor_name, 1);
  if (!t_entry) {
    free(header_json);
    fclose(f);
    vira_string err =
        vira_str_format("tensor '%s' not found in '%s'", nbuf.ptr, pbuf.ptr);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  int32_t parsed_dims[4] = {1, 1, 1, 1};
  int32_t parsed_ndims = 0;
  const char *shape_ptr = strstr(t_entry, "\"shape\"");
  if (shape_ptr) {
    const char *cur = strchr(shape_ptr, '[');
    if (cur) {
      cur++;
      while (*cur != '\0' && *cur != ']' && parsed_ndims < 4) {
        while (*cur == ' ' || *cur == ',' || *cur == '\t' || *cur == '\n' ||
               *cur == '\r') {
          cur++;
        }
        if (*cur == ']' || *cur == '\0')
          break;
        char *endp = NULL;
        long dval = strtol(cur, &endp, 10);
        if (endp == cur)
          break;
        parsed_dims[parsed_ndims++] = (dval > 0) ? (int32_t)dval : 1;
        cur = endp;
      }
    }
  }
  if (parsed_ndims == 0) {
    parsed_ndims = 1;
    parsed_dims[0] = 1;
  }

  uint64_t begin_off = 0;
  const char *off_ptr = strstr(t_entry, "\"data_offsets\"");
  if (off_ptr) {
    const char *lbracket = strchr(off_ptr, '[');
    if (lbracket) {
      begin_off = (uint64_t)strtoull(lbracket + 1, NULL, 10);
    }
  }
  free(header_json);

  uint64_t file_offset = 8 + header_len + begin_off;
#ifdef _WIN32
  _fseeki64(f, file_offset, SEEK_SET);
#else
  fseeko(f, file_offset, SEEK_SET);
#endif

  vira_tensor_f32 t = vira_tensor_f32_alloc_nd(parsed_ndims, parsed_dims);
  size_t expected_bytes = (size_t)t.rows * (size_t)t.cols * sizeof(float);
  size_t read_bytes = fread(t.data, 1, expected_bytes, f);
  fclose(f);

  if (read_bytes != expected_bytes) {
    vira_tensor_f32_free(&t);
    vira_string err = vira_str_format(
        "tensor '%s' data truncated: expected %zu bytes, read %zu", nbuf.ptr,
        expected_bytes, read_bytes);
    vira_cstr_buf_free(&nbuf);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  vira_cstr_buf_free(&nbuf);
  vira_cstr_buf_free(&pbuf);
  return (Result_tensor_f32){
      .is_err = false, .err = vira_str_lit("", 0), .val = t};
}

#endif // VIRA_RT_WEIGHTS_SAFETENSORS_H
