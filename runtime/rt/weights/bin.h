#ifndef VIRA_RT_WEIGHTS_BIN_H
#define VIRA_RT_WEIGHTS_BIN_H

// ============================================================================
// 7. PRETRAINED TENSOR BINARY WEIGHT I/O (PYTORCH / SAFETENSORS / GGUF COMPAT)
// ============================================================================

#ifndef _Result_tensor_f32_DEFINED
#define _Result_tensor_f32_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  vira_tensor_f32 val;
} Result_tensor_f32;
#endif

/**
 * Reads raw binary float tensor weights from disk into memory.
 * @param opt_arena Optional memory arena for allocation.
 * @param rows Matrix row count.
 * @param cols Matrix column count.
 * @param path File path to binary weight file.
 * @return Result containing loaded tensor or error message.
 * Usage: Loading binary pretrained weight matrices.
 */
static inline Result_tensor_f32 vira_tensor_load_bin(vira_arena *opt_arena,
                                                     int32_t rows, int32_t cols,
                                                     vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "rb");
  if (!f) {
    vira_string err =
        vira_str_format("could not open weight file '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  vira_tensor_f32 t = opt_arena
                          ? vira_tensor_f32_arena_alloc(opt_arena, rows, cols)
                          : vira_tensor_f32_alloc(rows, cols);

  size_t expected = (size_t)rows * (size_t)cols;
  size_t read_count = fread(t.data, sizeof(vira_f32), expected, f);
  fclose(f);

  if (read_count != expected) {
    if (!opt_arena)
      vira_tensor_f32_free(&t);
    vira_string err = vira_str_format(
        "weight file '%s' size mismatch: expected %zu floats, read %zu",
        pbuf.ptr, expected, read_count);
    vira_cstr_buf_free(&pbuf);
    return (Result_tensor_f32){
        .is_err = true, .err = err, .val = (vira_tensor_f32){0}};
  }

  vira_cstr_buf_free(&pbuf);
  return (Result_tensor_f32){
      .is_err = false, .err = vira_str_lit("", 0), .val = t};
}

/**
 * Writes tensor float data directly to disk as a raw binary file.
 * @param t Tensor to write.
 * @param path Target file path.
 * @return Result with true on success or error message.
 * Usage: Saving raw tensor weights to disk.
 */
static inline Result_bool vira_tensor_save_bin(vira_tensor_f32 t,
                                               vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "wb");
  if (!f) {
    vira_string err = vira_str_format(
        "failed to open file '%s' for writing weights", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_bool){.is_err = true, .err = err, .val = false};
  }

  _vira_tensor_sync_host_f32(t.data);
  size_t total = (size_t)t.rows * (size_t)t.cols;
  size_t written = fwrite(t.data, sizeof(vira_f32), total, f);
  fclose(f);

  if (written != total) {
    vira_string err =
        vira_str_format("failed to write complete weights to '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_bool){.is_err = true, .err = err, .val = false};
  }

  vira_cstr_buf_free(&pbuf);
  return (Result_bool){
      .is_err = false, .err = vira_str_lit("", 0), .val = true};
}

#endif // VIRA_RT_WEIGHTS_BIN_H
