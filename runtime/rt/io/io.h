#ifndef VIRA_RT_IO_H
#define VIRA_RT_IO_H

// ============================================================================
// 5. BUILTIN I/O & DIAGNOSTICS
// ============================================================================

/**
 * Prints a `vira_string` to standard output without appending a trailing newline.
 * @param s String to print.
 * Usage: Basic standard output display (`print("...")`).
 */
static inline void vira_print(vira_string s) {
  if (s.data != NULL && s.len > 0) {
    fwrite(s.data, 1, (size_t)s.len, stdout);
    fflush(stdout);
  }
}

/**
 * Prints a temporary `vira_string` to standard output and automatically frees its memory.
 * @param s Temporary string to print and free.
 * Usage: Injected by compiler for temporary string print expressions with guaranteed zero leaks.
 */
static inline void vira_print_temp(vira_string s) {
  vira_print(s);
  if (!s.is_lit && s.data != NULL) {
    free((void *)s.data);
    _vira_record_free();
  }
}

/**
 * Prints a `vira_string` to standard output followed by a trailing newline.
 * @param s String to print.
 * Usage: Standard line output display (`println("...")`).
 */
static inline void vira_println(vira_string s) {
  if (s.data != NULL && s.len > 0) {
    fwrite(s.data, 1, (size_t)s.len, stdout);
  }
  fputc('\n', stdout);
  fflush(stdout);
}

/**
 * Prints a temporary `vira_string` to standard output with trailing newline and frees its memory.
 * @param s Temporary string to print and free.
 * Usage: Compiler-generated wrapper for temporary formatted strings.
 */
static inline void vira_println_temp(vira_string s) {
  vira_println(s);
  if (!s.is_lit && s.data != NULL) {
    free((void *)s.data);
    _vira_record_free();
  }
}

/**
 * Prints a `vira_string` to standard error without appending a trailing newline.
 * @param s String to print.
 * Usage: Error stream output (`eprint("...")`).
 */
static inline void vira_eprint(vira_string s) {
  if (s.data != NULL && s.len > 0) {
    fwrite(s.data, 1, (size_t)s.len, stderr);
    fflush(stderr);
  }
}

/**
 * Prints a temporary `vira_string` to standard error and frees its memory.
 * @param s Temporary string to print and free.
 * Usage: Compiler-generated wrapper for temporary error print expressions.
 */
static inline void vira_eprint_temp(vira_string s) {
  vira_eprint(s);
  if (!s.is_lit && s.data != NULL) {
    free((void *)s.data);
    _vira_record_free();
  }
}

/**
 * Prints a `vira_string` to standard error followed by a trailing newline.
 * @param s String to print.
 * Usage: Error line output (`eprintln("...")`).
 */
static inline void vira_eprintln(vira_string s) {
  if (s.data != NULL && s.len > 0) {
    fwrite(s.data, 1, (size_t)s.len, stderr);
  }
  fputc('\n', stderr);
  fflush(stderr);
}

/**
 * Prints a temporary `vira_string` to standard error with trailing newline and frees its memory.
 * @param s Temporary string to print and free.
 * Usage: Compiler-generated wrapper for temporary error line expressions.
 */
static inline void vira_eprintln_temp(vira_string s) {
  vira_eprintln(s);
  if (!s.is_lit && s.data != NULL) {
    free((void *)s.data);
    _vira_record_free();
  }
}

/**
 * Suspends thread execution for the specified number of milliseconds.
 * @param ms Duration to sleep in milliseconds.
 * Usage: Delays, throttling, and polling loops (`time::sleep_ms(ms)`).
 */
static inline void vira_time_sleep_ms(int64_t ms) {
  if (ms <= 0) return;
#ifdef _WIN32
  Sleep((DWORD)ms);
#else
  struct timespec ts;
  ts.tv_sec = ms / 1000;
  ts.tv_nsec = (ms % 1000) * 1000000L;
  nanosleep(&ts, NULL);
#endif
}

/**
 * Suspends thread execution for the specified number of microseconds.
 * @param us Duration to sleep in microseconds.
 * Usage: High-precision delays (`time::sleep_us(us)`).
 */
static inline void vira_time_sleep_us(int64_t us) {
  if (us <= 0) return;
#ifdef _WIN32
  Sleep((DWORD)((us + 999) / 1000));
#else
  struct timespec ts;
  ts.tv_sec = us / 1000000;
  ts.tv_nsec = (us % 1000000) * 1000L;
  nanosleep(&ts, NULL);
#endif
}

/**
 * Returns current system epoch timestamp in microseconds.
 * @return Microseconds since UNIX epoch.
 * Usage: High-resolution benchmarking and timers (`time::now_us()`).
 */
static inline int64_t vira_time_now_us(void) {
#if defined(_WIN32)
  static LARGE_INTEGER freq;
  static int init = 0;
  if (!init) {
    QueryPerformanceFrequency(&freq);
    init = 1;
  }
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  return (int64_t)((counter.QuadPart * 1000000LL) / freq.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000LL + (int64_t)(ts.tv_nsec / 1000LL);
#endif
}

/**
 * Returns current system epoch timestamp in milliseconds.
 * @return Milliseconds since UNIX epoch.
 * Usage: General timestamps and timeouts (`time::now_ms()`).
 */
static inline int64_t vira_time_now_ms(void) {
  return vira_time_now_us() / 1000LL;
}

// ============================================================================
// 6. NATIVE FILE I/O & BUFFER OPERATIONS
// ============================================================================

#ifndef _Result_string_DEFINED
#define _Result_string_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  vira_string val;
} Result_string;
#endif

#ifndef _Result_bool_DEFINED
#define _Result_bool_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  bool val;
} Result_bool;
#endif

/**
 * Checks whether a filesystem entry exists at the specified path and is a regular file.
 * @param path File path string.
 * @return True if file exists, false otherwise.
 * Usage: File existence verification (`file::exists("path")`).
 */
static inline bool vira_file_exists(vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));
  FILE *f = fopen(pbuf.ptr, "rb");
  vira_cstr_buf_free(&pbuf);
  if (f) {
    fclose(f);
    return true;
  }
  return false;
}

/**
 * Checks whether a filesystem directory exists at the specified path.
 * @param path Directory path string.
 * @return True if directory exists, false otherwise.
 * Usage: Directory existence verification (`dir::exists("path")`).
 */
static inline bool vira_dir_exists(vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));
#ifdef _WIN32
  DWORD attr = GetFileAttributesA(pbuf.ptr);
  vira_cstr_buf_free(&pbuf);
  return (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
  struct stat st;
  int res = stat(pbuf.ptr, &st);
  vira_cstr_buf_free(&pbuf);
  return (res == 0 && S_ISDIR(st.st_mode));
#endif
}

static inline int _vira_fseek64(FILE *f, int64_t offset, int origin) {
#ifdef _WIN32
  return _fseeki64(f, offset, origin);
#else
  return fseeko(f, (off_t)offset, origin);
#endif
}

static inline int64_t _vira_ftell64(FILE *f) {
#ifdef _WIN32
  return (int64_t)_ftelli64(f);
#else
  return (int64_t)ftello(f);
#endif
}

/**
 * Reads the entire contents of a file into an owned `vira_string`.
 * @param path Path to file.
 * @return Result containing file contents string or error message.
 * Usage: Complete file reading (`file::read("path")`).
 */
static inline Result_string vira_read_file(vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "rb");
  if (!f) {
    vira_string err = vira_str_format("failed to open file '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_string){.is_err = true, .err = err};
  }
  if (_vira_fseek64(f, 0, SEEK_END) != 0) {
    fclose(f);
    vira_string err = vira_str_format("failed to seek in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_string){.is_err = true, .err = err};
  }
  int64_t sz = _vira_ftell64(f);
  if (sz < 0 || _vira_fseek64(f, 0, SEEK_SET) != 0) {
    fclose(f);
    vira_string err = vira_str_format("failed to seek in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_string){.is_err = true, .err = err};
  }
  if (sz > (int64_t)INT32_MAX) {
    fclose(f);
    vira_string err = vira_str_format(
        "file '%s' exceeds maximum string size (%lld bytes)", pbuf.ptr,
        (long long)sz);
    vira_cstr_buf_free(&pbuf);
    return (Result_string){.is_err = true, .err = err};
  }
  vira_cstr_buf_free(&pbuf);

  char *buf = (char *)malloc((size_t)sz + 1);
  if (!buf) {
    fclose(f);
    return (Result_string){
        .is_err = true, .err = vira_str_lit("out of memory reading file", 26)};
  }
  size_t read_bytes = fread(buf, 1, (size_t)sz, f);
  buf[read_bytes] = '\0';
  fclose(f);
  _vira_record_alloc();
  return (Result_string){.is_err = false,
                         .val = (vira_string){.data = (const uint8_t *)buf,
                                              .len = (int32_t)read_bytes,
                                              .is_lit = false}};
}

/**
 * Reads the initial portion of a file up to `max_bytes`.
 * @param path Path to file.
 * @param max_bytes Maximum bytes to read.
 * @return Result containing partial string or error message.
 * Usage: Inspecting file headers or large log files without loading into memory.
 */
static inline Result_string vira_read_file_head(vira_string path,
                                                int32_t max_bytes) {
  char stack_path[1024];
  vira_cstr_buf pbuf =
      vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "rb");
  if (!f) {
    vira_string err = vira_str_format("failed to open file '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_string){.is_err = true, .err = err};
  }
  if (_vira_fseek64(f, 0, SEEK_END) != 0) {
    fclose(f);
    vira_string err = vira_str_format("failed to seek in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_string){.is_err = true, .err = err};
  }
  int64_t sz = _vira_ftell64(f);
  if (sz < 0 || _vira_fseek64(f, 0, SEEK_SET) != 0) {
    fclose(f);
    vira_string err = vira_str_format("failed to seek in '%s'", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_string){.is_err = true, .err = err};
  }
  vira_cstr_buf_free(&pbuf);

  int64_t capped = sz;
  if (max_bytes > 0 && capped > (int64_t)max_bytes) {
    capped = (int64_t)max_bytes;
  }
  if (capped > (int64_t)INT32_MAX) {
    capped = (int64_t)INT32_MAX;
  }
  size_t to_read = (size_t)capped;
  char *buf = (char *)malloc(to_read + 1);
  if (!buf) {
    fclose(f);
    return (Result_string){
        .is_err = true, .err = vira_str_lit("out of memory reading file", 26)};
  }
  size_t read_bytes = fread(buf, 1, to_read, f);
  buf[read_bytes] = '\0';
  fclose(f);
  _vira_record_alloc();
  return (Result_string){.is_err = false,
                         .val = (vira_string){.data = (const uint8_t *)buf,
                                              .len = (int32_t)read_bytes,
                                              .is_lit = false}};
}

/**
 * Writes string content to a file, creating it or overwriting existing contents.
 * @param path Path to target file.
 * @param content String data to write.
 * @return Result with true on success or error message.
 * Usage: File writing (`file::write("path", text)`).
 */
static inline Result_bool vira_write_file(vira_string path,
                                          vira_string content) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "wb");
  if (!f) {
    vira_string err =
        vira_str_format("failed to open file '%s' for writing", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_bool){.is_err = true, .err = err};
  }
  if (content.len > 0 && content.data != NULL) {
    size_t written = fwrite(content.data, 1, (size_t)content.len, f);
    if (written != (size_t)content.len) {
      fclose(f);
      vira_string err = vira_str_format(
          "failed to write complete content to '%s'", pbuf.ptr);
      vira_cstr_buf_free(&pbuf);
      return (Result_bool){.is_err = true, .err = err};
    }
  }
  fclose(f);
  vira_cstr_buf_free(&pbuf);
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Appends string content to the end of a file, creating it if it does not exist.
 * @param path Path to target file.
 * @param content String data to append.
 * @return Result with true on success or error message.
 * Usage: Appending log entries or records (`file::append("path", text)`).
 */
static inline Result_bool vira_append_file(vira_string path,
                                           vira_string content) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  FILE *f = fopen(pbuf.ptr, "ab");
  if (!f) {
    vira_string err =
        vira_str_format("failed to open file '%s' for appending", pbuf.ptr);
    vira_cstr_buf_free(&pbuf);
    return (Result_bool){.is_err = true, .err = err};
  }
  if (content.len > 0 && content.data != NULL) {
    size_t written = fwrite(content.data, 1, (size_t)content.len, f);
    if (written != (size_t)content.len) {
      fclose(f);
      vira_string err = vira_str_format(
          "failed to append complete content to '%s'", pbuf.ptr);
      vira_cstr_buf_free(&pbuf);
      return (Result_bool){.is_err = true, .err = err};
    }
  }
  fclose(f);
  vira_cstr_buf_free(&pbuf);
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Deletes a file from the filesystem.
 * @param path Path to file.
 * @return Result with true on success or error message.
 * Usage: File deletion (`file::remove("path")`).
 */
static inline Result_bool vira_remove_file(vira_string path) {
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  if (remove(pbuf.ptr) == 0) {
    vira_cstr_buf_free(&pbuf);
    return (Result_bool){.is_err = false, .val = true};
  }
  vira_string err = vira_str_format("failed to remove file '%s'", pbuf.ptr);
  vira_cstr_buf_free(&pbuf);
  return (Result_bool){.is_err = true, .err = err};
}

/**
 * Opens a file with specified mode string ("r", "w", "a", etc.), returning an integer file descriptor handle.
 * @param path Path to file.
 * @param mode Mode string.
 * @return File descriptor handle, or negative on error.
 * Usage: Low-level file handle acquisition (`io::open("path", "rb")`).
 */
static inline int64_t vira_io_file_open(vira_string path, vira_string mode) {
  if (path.len == 0 || path.data == NULL || mode.len == 0 ||
      mode.data == NULL) {
    return 0;
  }
  char stack_path[1024];
  vira_cstr_buf pbuf = vira_str_to_cstr_buf(path, stack_path, sizeof(stack_path));

  char stack_mode[16];
  vira_cstr_buf mbuf = vira_str_to_cstr_buf(mode, stack_mode, sizeof(stack_mode));

  FILE *f = fopen(pbuf.ptr, mbuf.ptr);
  vira_cstr_buf_free(&mbuf);
  vira_cstr_buf_free(&pbuf);
  return (int64_t)(uintptr_t)f;
}

/**
 * Reads up to `max_len` bytes from an open file handle into a buffer slice.
 * @param handle File descriptor handle.
 * @param buf Pointer to destination byte slice.
 * @param max_len Maximum bytes to read.
 * @return Number of bytes actually read, or negative on error.
 * Usage: Chunked file streaming.
 */
static inline int32_t vira_io_file_read(int64_t handle, vira_slice_u8 *buf,
                                        int32_t max_len) {
  if (handle == 0 || buf == NULL || buf->data == NULL || max_len <= 0) {
    return 0;
  }
  FILE *f = (FILE *)(uintptr_t)handle;
  int32_t can_read = buf->cap < max_len ? buf->cap : max_len;
  if (can_read <= 0)
    return 0;
  size_t n = fread(buf->data, 1, (size_t)can_read, f);
  buf->len = (int32_t)n;
  return (int32_t)n;
}

/**
 * Writes `len` bytes from a buffer slice to an open file handle.
 * @param handle File descriptor handle.
 * @param buf Source byte slice.
 * @param len Number of bytes to write.
 * @return Number of bytes written, or negative on error.
 * Usage: Chunked file writing.
 */
static inline int32_t vira_io_file_write(int64_t handle, vira_slice_u8 buf,
                                         int32_t len) {
  if (handle == 0 || buf.data == NULL || len <= 0) {
    return 0;
  }
  FILE *f = (FILE *)(uintptr_t)handle;
  int32_t to_write = buf.len < len ? buf.len : len;
  if (to_write <= 0)
    return 0;
  size_t n = fwrite(buf.data, 1, (size_t)to_write, f);
  return (int32_t)n;
}

/**
 * Repositions file read/write offset according to offset and whence (SEEK_SET, SEEK_CUR, SEEK_END).
 * @param handle File descriptor handle.
 * @param offset Byte offset.
 * @param whence Seek origin mode.
 * @return Resulting absolute file offset in bytes.
 * Usage: Random-access file navigation (`io::seek(handle, offset, whence)`).
 */
static inline int64_t vira_io_file_seek(int64_t handle, int64_t offset,
                                        int32_t whence) {
  if (handle == 0)
    return -1;
  FILE *f = (FILE *)(uintptr_t)handle;
  int c_whence = SEEK_SET;
  if (whence == 1)
    c_whence = SEEK_CUR;
  else if (whence == 2)
    c_whence = SEEK_END;
  if (_vira_fseek64(f, offset, c_whence) != 0) {
    return -1;
  }
  return _vira_ftell64(f);
}

/**
 * Flushes all pending buffered writes for an open file handle to physical storage.
 * @param handle File descriptor handle.
 * @return 0 on success, negative on error.
 * Usage: Ensuring write durability (`io::flush(handle)`).
 */
static inline int32_t vira_io_file_flush(int64_t handle) {
  if (handle == 0)
    return -1;
  FILE *f = (FILE *)(uintptr_t)handle;
  return fflush(f) == 0 ? 0 : -1;
}

/**
 * Closes an open file descriptor handle and releases OS file locks.
 * @param handle File descriptor handle.
 * @return 0 on success, negative on error.
 * Usage: File handle deallocation (`io::close(handle)`).
 */
static inline int32_t vira_io_file_close(int64_t handle) {
  if (handle == 0)
    return 0;
  FILE *f = (FILE *)(uintptr_t)handle;
  return fclose(f) == 0 ? 0 : -1;
}

/**
 * Queries the total size in bytes of an open file descriptor.
 * @param handle File descriptor handle.
 * @return Total size in bytes.
 * Usage: File size discovery on open descriptors (`io::size(handle)`).
 */
static inline int64_t vira_io_file_size(int64_t handle) {
  if (handle == 0)
    return -1;
  FILE *f = (FILE *)(uintptr_t)handle;
  int64_t cur = _vira_ftell64(f);
  if (cur < 0 || _vira_fseek64(f, 0, SEEK_END) != 0)
    return -1;
  int64_t sz = _vira_ftell64(f);
  _vira_fseek64(f, cur, SEEK_SET);
  return sz;
}

#endif // VIRA_RT_IO_H
