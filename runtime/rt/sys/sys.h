#ifndef VIRA_RT_SYS_H
#define VIRA_RT_SYS_H

// ============================================================================
// 9. SYSTEMS, PROCESS, SIGNALS & CONTAINER RUNTIME SHIMS
// ============================================================================

#define _VIRA_SIG_QUEUE_CAP 16
static volatile sig_atomic_t _vira_sig_queue[_VIRA_SIG_QUEUE_CAP];
static volatile sig_atomic_t _vira_sig_head = 0;
static volatile sig_atomic_t _vira_sig_tail = 0;

/**
 * POSIX signal handler recording received signal numbers into the runtime signal queue.
 * @param sig Signal number received.
 * Usage: Asynchronous signal interception.
 */
static void _vira_sig_handler(int sig) {
  int next = (_vira_sig_head + 1) % _VIRA_SIG_QUEUE_CAP;
  if (next != _vira_sig_tail) {
    _vira_sig_queue[_vira_sig_head] = sig;
    _vira_sig_head = next;
  }
}

/**
 * Registers interest in specified POSIX signals, routing them to the runtime signal queue.
 * @param sigs Slice of signal numbers (e.g. SIGINT, SIGTERM).
 * @return Result with true on success or error.
 * Usage: Signal interception and graceful shutdown setup (`sys::signal_notify([2, 15])`).
 */
static inline Result_bool vira_sys_signal_notify(vira_slice_int sigs) {
  for (int32_t i = 0; i < sigs.len; i++) {
    signal(sigs.data[i], _vira_sig_handler);
  }
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Non-blocking check for received signals in the signal queue.
 * @return Signal number if one was caught, or 0 if none.
 * Usage: Polling for termination signals in event loops (`sys::signal_poll()`).
 */
static inline int32_t vira_sys_signal_poll(void) {
  if (_vira_sig_tail == _vira_sig_head) {
    return 0;
  }
  int sig = _vira_sig_queue[_vira_sig_tail];
  _vira_sig_tail = (_vira_sig_tail + 1) % _VIRA_SIG_QUEUE_CAP;
  return sig;
}

/**
 * Blocks the current thread until one of the monitored signals is caught.
 * @return Received signal number.
 * Usage: Waiting for Ctrl+C or kill signals in daemon processes (`sys::signal_wait()`).
 */
static inline int32_t vira_sys_signal_wait(void) {
  while (1) {
    int sig = vira_sys_signal_poll();
    if (sig != 0) return sig;
    vira_time_sleep_ms(10);
  }
}

/**
 * Returns the last system error code (`errno`).
 * @return Integer error code.
 * Usage: Inspecting OS system call failures (`sys::errno()`).
 */
static inline int32_t vira_sys_errno(void) {
  return (int32_t)errno;
}

/**
 * Returns the human-readable description string corresponding to a system error code.
 * @param err System error code.
 * @return Error message string.
 * Usage: Translating errno codes to user messages (`sys::strerror(err)`).
 */
static inline vira_string vira_sys_strerror(int32_t err) {
  return vira_str_from_cstr(strerror(err));
}

#define VIRA_O_RDONLY   0
#define VIRA_O_WRONLY   1
#define VIRA_O_RDWR     2
#define VIRA_O_CREAT    0x40
#define VIRA_O_EXCL     0x80
#define VIRA_O_TRUNC    0x200
#define VIRA_O_APPEND   0x400

/**
 * Issues low-level OS `open` system call with path, access flags, and permission mode.
 * @param path Filesystem path.
 * @param flags Bitmask of POSIX open flags (O_RDONLY, O_CREAT, etc.).
 * @param mode Permission mode bits (e.g. 0644).
 * @return Result containing file descriptor or error message.
 * Usage: Direct POSIX file descriptor acquisition (`sys::open(...)`).
 */
static inline Result_int vira_sys_open(vira_string path, int32_t flags, int32_t mode) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
  int native_flags = 0;
  int acc = flags & 3;
#ifdef _WIN32
  if (acc == VIRA_O_WRONLY) native_flags |= _O_WRONLY;
  else if (acc == VIRA_O_RDWR) native_flags |= _O_RDWR;
  else native_flags |= _O_RDONLY;
  if (flags & VIRA_O_CREAT) native_flags |= _O_CREAT;
  if (flags & VIRA_O_EXCL) native_flags |= _O_EXCL;
  if (flags & VIRA_O_TRUNC) native_flags |= _O_TRUNC;
  if (flags & VIRA_O_APPEND) native_flags |= _O_APPEND;
  native_flags |= _O_BINARY;
  int fd = _open(p.ptr, native_flags, mode == 0 ? 0666 : mode);
#else
  if (acc == VIRA_O_WRONLY) native_flags |= O_WRONLY;
  else if (acc == VIRA_O_RDWR) native_flags |= O_RDWR;
  else native_flags |= O_RDONLY;
  if (flags & VIRA_O_CREAT) native_flags |= O_CREAT;
  if (flags & VIRA_O_EXCL) native_flags |= O_EXCL;
  if (flags & VIRA_O_TRUNC) native_flags |= O_TRUNC;
  if (flags & VIRA_O_APPEND) native_flags |= O_APPEND;
  int fd = open(p.ptr, native_flags, mode == 0 ? 0666 : (mode_t)mode);
#endif
  vira_cstr_buf_free(&p);
  if (fd < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("open error: %s", strerror(errno))};
  }
  return (Result_int){.is_err = false, .val = fd};
}

/**
 * Issues low-level OS `read` system call, reading up to `max_bytes` into a buffer slice.
 * @param fd File descriptor.
 * @param buf Pointer to destination byte slice.
 * @param max_bytes Maximum bytes to read.
 * @return Result containing count of bytes read or error message.
 * Usage: Direct POSIX descriptor reading.
 */
static inline Result_int vira_sys_read(int32_t fd, vira_slice_u8 *buf, int32_t max_bytes) {
  if (buf == NULL || max_bytes <= 0) return (Result_int){.is_err = false, .val = 0};
  if (buf->cap < max_bytes) {
    while (buf->cap < max_bytes) {
      _vira_slice_grow((void**)&buf->data, &buf->cap, sizeof(uint8_t), buf->is_arena);
    }
  }
#ifdef _WIN32
  int n = _read(fd, buf->data, (unsigned int)max_bytes);
#else
  ssize_t n = read(fd, buf->data, (size_t)max_bytes);
#endif
  if (n < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("read error: %s", strerror(errno))};
  }
  buf->len = (int32_t)n;
  return (Result_int){.is_err = false, .val = (int32_t)n};
}

/**
 * Issues low-level OS `write` system call, writing `count` bytes from a buffer slice.
 * @param fd File descriptor.
 * @param buf Source byte slice.
 * @param count Number of bytes to write.
 * @return Result containing count of bytes written or error message.
 * Usage: Direct POSIX descriptor writing.
 */
static inline Result_int vira_sys_write(int32_t fd, vira_slice_u8 buf, int32_t count) {
  if (buf.data == NULL || count <= 0) return (Result_int){.is_err = false, .val = 0};
  if (count > buf.len) count = buf.len;
#ifdef _WIN32
  int n = _write(fd, buf.data, (unsigned int)count);
#else
  ssize_t n = write(fd, buf.data, (size_t)count);
#endif
  if (n < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("write error: %s", strerror(errno))};
  }
  return (Result_int){.is_err = false, .val = (int32_t)n};
}

/**
 * Issues low-level OS `close` system call to close a file descriptor.
 * @param fd File descriptor.
 * @return Result with true on success or error message.
 * Usage: Direct POSIX descriptor closing.
 */
static inline Result_bool vira_sys_close(int32_t fd) {
#ifdef _WIN32
  int res = _close(fd);
#else
  int res = close(fd);
#endif
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("close error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Issues low-level OS `lseek` system call to reposition read/write offset.
 * @param fd File descriptor.
 * @param offset Byte offset.
 * @param whence Origin position mode.
 * @return Result containing new offset or error message.
 * Usage: Direct descriptor seeking.
 */
static inline Result_i64 vira_sys_lseek(int32_t fd, int64_t offset, int32_t whence) {
#ifdef _WIN32
  __int64 res = _lseeki64(fd, offset, whence);
#else
  off_t res = lseek(fd, (off_t)offset, whence);
#endif
  if (res < 0) {
    return (Result_i64){.is_err = true, .err = vira_str_format("lseek error: %s", strerror(errno))};
  }
  return (Result_i64){.is_err = false, .val = (int64_t)res};
}

/**
 * Creates an unidirectional IPC data channel, returning a 2-element slice of file descriptors [read_fd, write_fd].
 * @param fds Pointer to integer slice receiving [read_fd, write_fd].
 * @return Result with true on success or error.
 * Usage: Inter-process communication and piping (`sys::pipe(&fds)`).
 */
static inline Result_bool vira_sys_pipe(vira_slice_int *fds) {
  if (fds == NULL) return (Result_bool){.is_err = true, .err = vira_str_lit("null slice", 10)};
  int p[2];
#ifdef _WIN32
  if (_pipe(p, 4096, _O_BINARY) != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("pipe error: %s", strerror(errno))};
  }
#else
  if (pipe(p) != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("pipe error: %s", strerror(errno))};
  }
#endif
  vira_slice_int_clear(fds);
  vira_slice_int_push(fds, p[0]);
  vira_slice_int_push(fds, p[1]);
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Duplicates an open file descriptor using OS `dup`.
 * @param fd Existing file descriptor.
 * @return Result containing new duplicated file descriptor or error.
 * Usage: Descriptor redirection (`sys::dup(fd)`).
 */
static inline Result_int vira_sys_dup(int32_t fd) {
#ifdef _WIN32
  int res = _dup(fd);
#else
  int res = dup(fd);
#endif
  if (res < 0) return (Result_int){.is_err = true, .err = vira_str_format("dup error: %s", strerror(errno))};
  return (Result_int){.is_err = false, .val = res};
}

/**
 * Duplicates an open file descriptor onto a specified target descriptor using OS `dup2`.
 * @param oldfd Source descriptor.
 * @param newfd Target descriptor number.
 * @return Result containing newfd or error.
 * Usage: Standard stream redirection (stdin/stdout/stderr).
 */
static inline Result_int vira_sys_dup2(int32_t oldfd, int32_t newfd) {
#ifdef _WIN32
  int res = _dup2(oldfd, newfd);
#else
  int res = dup2(oldfd, newfd);
#endif
  if (res < 0) return (Result_int){.is_err = true, .err = vira_str_format("dup2 error: %s", strerror(errno))};
  return (Result_int){.is_err = false, .val = res};
}

/**
 * Tests whether a file descriptor refers to an interactive terminal device.
 * @param fd File descriptor.
 * @return True if descriptor is a TTY, false otherwise.
 * Usage: Determining whether to enable ANSI color escapes (`sys::isatty(1)`).
 */
static inline bool vira_sys_isatty(int32_t fd) {
#ifdef _WIN32
  return _isatty(fd) != 0;
#else
  return isatty(fd) != 0;
#endif
}

/**
 * Returns the process ID (PID) of the current calling process.
 * @return Integer process ID.
 * Usage: Process tracking and PID file writing (`sys::getpid()`).
 */
static inline int32_t vira_sys_getpid(void) {
#ifdef _WIN32
  return (int32_t)GetCurrentProcessId();
#else
  return (int32_t)getpid();
#endif
}

/**
 * Returns the parent process ID (PPID) of the current calling process.
 * @return Integer parent process ID.
 * Usage: Inspecting parent process identity (`sys::getppid()`).
 */
static inline int32_t vira_sys_getppid(void) {
#ifdef _WIN32
  return 0;
#else
  return (int32_t)getppid();
#endif
}

/**
 * Returns the real user ID (UID) of the calling process.
 * @return Integer user ID.
 * Usage: Privilege checking (`sys::getuid()`).
 */
static inline int32_t vira_sys_getuid(void) {
#ifdef _WIN32
  return 0;
#else
  return (int32_t)getuid();
#endif
}

/**
 * Returns the real group ID (GID) of the calling process.
 * @return Integer group ID.
 * Usage: Privilege checking (`sys::getgid()`).
 */
static inline int32_t vira_sys_getgid(void) {
#ifdef _WIN32
  return 0;
#else
  return (int32_t)getgid();
#endif
}

/**
 * Sends a signal to a process or process group identified by PID.
 * @param pid Target process ID.
 * @param sig Signal number (SIGTERM, SIGKILL, etc.).
 * @return Result with true on success or error.
 * Usage: Process management and termination (`sys::kill(pid, 9)`).
 */
static inline Result_bool vira_sys_kill(int32_t pid, int32_t sig) {
#ifdef _WIN32
  HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD)pid);
  if (!h) return (Result_bool){.is_err = true, .err = vira_str_lit("failed to open process", 22)};
  BOOL ok = TerminateProcess(h, (UINT)sig);
  CloseHandle(h);
  if (!ok) return (Result_bool){.is_err = true, .err = vira_str_lit("failed to terminate process", 27)};
  return (Result_bool){.is_err = false, .val = true};
#else
  if (kill(pid, sig) != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("kill error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#endif
}

/**
 * Executes a shell command string using the host system's command processor.
 * @param cmd Command string.
 * @return Process exit status code.
 * Usage: Shell command execution (`sys::system("ls -la")`).
 */
static inline int32_t vira_sys_system(vira_string cmd) {
  char stack_buf[1024];
  vira_cstr_buf c = vira_str_to_cstr_buf(cmd, stack_buf, sizeof(stack_buf));
  int res = system(c.ptr);
  vira_cstr_buf_free(&c);
  return res;
}

/**
 * Retrieves the value of an environment variable by name.
 * @param name Environment variable name string.
 * @return Result containing variable value or error if not set.
 * Usage: Configuration from environment (`sys::getenv("PATH")`).
 */
static inline Result_string vira_sys_getenv(vira_string name) {
  char stack_buf[256];
  vira_cstr_buf n = vira_str_to_cstr_buf(name, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  char win_buf[32767];
  DWORD win_len = GetEnvironmentVariableA(n.ptr, win_buf, sizeof(win_buf));
  if (win_len > 0) {
    vira_cstr_buf_free(&n);
    return (Result_string){.is_err = false, .val = vira_str_from_cstr(win_buf)};
  }
#endif
  const char *val = getenv(n.ptr);
  vira_cstr_buf_free(&n);
  if (!val) {
    return (Result_string){.is_err = true, .err = vira_str_lit("environment variable not found", 30)};
  }
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(val)};
}

/**
 * Sets or updates the value of an environment variable.
 * @param name Variable name string.
 * @param val Variable value string.
 * @return Result with true on success or error.
 * Usage: Environment variable mutation (`sys::setenv("KEY", "VALUE")`).
 */
static inline Result_bool vira_sys_setenv(vira_string name, vira_string val) {
  char stack_n[256], stack_v[1024];
  vira_cstr_buf n = vira_str_to_cstr_buf(name, stack_n, sizeof(stack_n));
  vira_cstr_buf v = vira_str_to_cstr_buf(val, stack_v, sizeof(stack_v));
#ifdef _WIN32
  _putenv_s(n.ptr, v.ptr);
  BOOL ok = SetEnvironmentVariableA(n.ptr, v.ptr);
  vira_cstr_buf_free(&n);
  vira_cstr_buf_free(&v);
  if (!ok) return (Result_bool){.is_err = true, .err = vira_str_lit("SetEnvironmentVariable failed", 29)};
  return (Result_bool){.is_err = false, .val = true};
#else
  int res = setenv(n.ptr, v.ptr, 1);
  vira_cstr_buf_free(&n);
  vira_cstr_buf_free(&v);
  if (res != 0) return (Result_bool){.is_err = true, .err = vira_str_format("setenv error: %s", strerror(errno))};
  return (Result_bool){.is_err = false, .val = true};
#endif
}

/**
 * Deletes an environment variable from the process environment.
 * @param name Variable name string.
 * @return Result with true on success or error.
 * Usage: Environment variable removal (`sys::unsetenv("KEY")`).
 */
static inline Result_bool vira_sys_unsetenv(vira_string name) {
  char stack_n[256];
  vira_cstr_buf n = vira_str_to_cstr_buf(name, stack_n, sizeof(stack_n));
#ifdef _WIN32
  _putenv_s(n.ptr, "");
  SetEnvironmentVariableA(n.ptr, NULL);
  vira_cstr_buf_free(&n);
  return (Result_bool){.is_err = false, .val = true};
#else
  int res = unsetenv(n.ptr);
  vira_cstr_buf_free(&n);
  if (res != 0) return (Result_bool){.is_err = true, .err = vira_str_format("unsetenv error: %s", strerror(errno))};
  return (Result_bool){.is_err = false, .val = true};
#endif
}

/**
 * Returns all environment variables as a slice of "KEY=VALUE" strings.
 * @return Slice of environment strings (`[]string`).
 * Usage: Environment dumping and child process inheritance (`sys::environ()`).
 */
static inline vira_slice_string vira_sys_environ(void) {
  vira_slice_string res = vira_slice_string_new(16);
#ifdef _WIN32
  char *env_block = GetEnvironmentStringsA();
  if (env_block) {
    char *p = env_block;
    while (*p) {
      vira_slice_string_push(&res, vira_str_from_cstr(p));
      p += strlen(p) + 1;
    }
    FreeEnvironmentStringsA(env_block);
  }
#else
  extern char **environ;
  if (environ) {
    for (char **p = environ; *p != NULL; p++) {
      vira_slice_string_push(&res, vira_str_from_cstr(*p));
    }
  }
#endif
  return res;
}

/**
 * Retrieves the standard network hostname of the local machine.
 * @return Result containing hostname string or error.
 * Usage: Machine identification and networking (`sys::hostname()`).
 */
static inline Result_string vira_sys_hostname(void) {
  char buf[256];
  if (gethostname(buf, sizeof(buf)) != 0) {
    return (Result_string){.is_err = true, .err = vira_str_lit("failed to get hostname", 22)};
  }
  buf[sizeof(buf) - 1] = '\0';
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(buf)};
}

/**
 * Returns the absolute filesystem path of the currently executing binary.
 * @return Result containing executable path string or error.
 * Usage: Locating sibling assets, configurations, and self-restart (`sys::executable_path()`).
 */
static inline Result_string vira_sys_executable_path(void) {
#ifdef _WIN32
  char buf[MAX_PATH];
  DWORD len = GetModuleFileNameA(NULL, buf, MAX_PATH);
  if (len == 0 || len >= MAX_PATH) {
    return (Result_string){.is_err = true, .err = vira_str_lit("GetModuleFileName failed", 24)};
  }
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(buf)};
#elif defined(__linux__)
  char buf[4096];
  ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len <= 0) {
    return (Result_string){.is_err = true, .err = vira_str_lit("readlink /proc/self/exe failed", 30)};
  }
  buf[len] = '\0';
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(buf)};
#else
  return (Result_string){.is_err = true, .err = vira_str_lit("executable_path not supported on this platform", 46)};
#endif
}

/**
 * Creates a directory with the specified POSIX permissions.
 * @param path Directory path string.
 * @param perm Permission mode bits (e.g. 0755).
 * @return Result with true on success or error.
 * Usage: Directory creation (`sys::mkdir("build", 0755)`).
 */
static inline Result_bool vira_sys_mkdir(vira_string path, int32_t perm) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  (void)perm;
  int res = _mkdir(p.ptr);
#else
  int res = mkdir(p.ptr, (mode_t)perm);
#endif
  vira_cstr_buf_free(&p);
  if (res != 0 && errno != EEXIST) {
    return (Result_bool){.is_err = true, .err = vira_str_format("mkdir error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Removes an empty directory from the filesystem.
 * @param path Directory path string.
 * @return Result with true on success or error.
 * Usage: Directory cleanup (`sys::rmdir("empty_dir")`).
 */
static inline Result_bool vira_sys_rmdir(vira_string path) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  int res = _rmdir(p.ptr);
#else
  int res = rmdir(p.ptr);
#endif
  vira_cstr_buf_free(&p);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("rmdir error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Deletes a name from the filesystem using OS `unlink`.
 * @param path Path to file or symbolic link.
 * @return Result with true on success or error.
 * Usage: Low-level file removal (`sys::unlink(path)`).
 */
static inline Result_bool vira_sys_unlink(vira_string path) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  int res = _unlink(p.ptr);
#else
  int res = unlink(p.ptr);
#endif
  vira_cstr_buf_free(&p);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("unlink error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Changes the current working directory of the process.
 * @param path Target directory path.
 * @return Result with true on success or error.
 * Usage: Working directory navigation (`sys::chdir("/tmp")`).
 */
static inline Result_bool vira_sys_chdir(vira_string path) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  int res = _chdir(p.ptr);
#else
  int res = chdir(p.ptr);
#endif
  vira_cstr_buf_free(&p);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("chdir error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Retrieves the current working directory path of the calling process.
 * @return Result containing working directory path or error.
 * Usage: Current directory query (`sys::getcwd()`).
 */
static inline Result_string vira_sys_getcwd(void) {
  char buf[4096];
#ifdef _WIN32
  char *res = _getcwd(buf, sizeof(buf));
#else
  char *res = getcwd(buf, sizeof(buf));
#endif
  if (!res) {
    return (Result_string){.is_err = true, .err = vira_str_format("getcwd error: %s", strerror(errno))};
  }
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(buf)};
}

/**
 * Changes the filesystem permissions mode of a target path.
 * @param path Filesystem path.
 * @param perm Permission mode bits.
 * @return Result with true on success or error.
 * Usage: Permission management (`sys::chmod("script.sh", 0755)`).
 */
static inline Result_bool vira_sys_chmod(vira_string path, int32_t perm) {
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  int res = _chmod(p.ptr, perm);
#else
  int res = chmod(p.ptr, (mode_t)perm);
#endif
  vira_cstr_buf_free(&p);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("chmod error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Reads all directory entry names in the specified path, excluding "." and "..".
 * @param path Directory path.
 * @return Slice of entry name strings (`[]string`).
 * Usage: Directory scanning and file listing (`sys::readdir(".")`).
 */
static inline vira_slice_string vira_sys_readdir(vira_string path) {
  vira_slice_string res = vira_slice_string_new(16);
  char stack_buf[1024];
  vira_cstr_buf p = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
#ifdef _WIN32
  char search_path[MAX_PATH];
  snprintf(search_path, sizeof(search_path), "%s\\*", p.ptr);
  WIN32_FIND_DATAA fd;
  HANDLE hFind = FindFirstFileA(search_path, &fd);
  if (hFind != INVALID_HANDLE_VALUE) {
    do {
      if (strcmp(fd.cFileName, ".") != 0 && strcmp(fd.cFileName, "..") != 0) {
        vira_slice_string_push(&res, vira_str_from_cstr(fd.cFileName));
      }
    } while (FindNextFileA(hFind, &fd));
    FindClose(hFind);
  }
#else
  DIR *d = opendir(p.ptr);
  if (d) {
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
      if (strcmp(de->d_name, ".") != 0 && strcmp(de->d_name, "..") != 0) {
        vira_slice_string_push(&res, vira_str_from_cstr(de->d_name));
      }
    }
    closedir(d);
  }
#endif
  vira_cstr_buf_free(&p);
  return res;
}

/**
 * Attaches a filesystem to a mount point with filesystem type, flags, and data options (Linux container shim).
 * @param source Device or source path.
 * @param target Target directory mount point.
 * @param fstype Filesystem type name (e.g. "proc", "tmpfs").
 * @param flags Mount flag bitmask (MS_NOSUID, MS_NODEV, etc.).
 * @param data Optional filesystem data options string.
 * @return Result with true on success or error.
 * Usage: Linux sandbox and container runtime creation.
 */
static inline Result_bool vira_sys_mount(vira_string source, vira_string target, vira_string fstype, int64_t flags, vira_string data) {
#if defined(__linux__)
  char s_src[1024], s_tgt[1024], s_fst[64], s_dat[1024];
  vira_cstr_buf b_src = vira_str_to_cstr_buf(source, s_src, sizeof(s_src));
  vira_cstr_buf b_tgt = vira_str_to_cstr_buf(target, s_tgt, sizeof(s_tgt));
  vira_cstr_buf b_fst = vira_str_to_cstr_buf(fstype, s_fst, sizeof(s_fst));
  vira_cstr_buf b_dat = vira_str_to_cstr_buf(data, s_dat, sizeof(s_dat));
  const char *p_src = (source.len > 0) ? b_src.ptr : NULL;
  const char *p_fst = (fstype.len > 0) ? b_fst.ptr : NULL;
  const void *p_dat = (data.len > 0) ? b_dat.ptr : NULL;
  int res = mount(p_src, b_tgt.ptr, p_fst, (unsigned long)flags, p_dat);
  vira_cstr_buf_free(&b_src);
  vira_cstr_buf_free(&b_tgt);
  vira_cstr_buf_free(&b_fst);
  vira_cstr_buf_free(&b_dat);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("mount error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)source; (void)target; (void)fstype; (void)flags; (void)data;
  return (Result_bool){.is_err = true, .err = vira_str_lit("mount: only supported on Linux", 30)};
#endif
}

/**
 * Detaches a previously mounted filesystem from a target directory.
 * @param target Mount point path.
 * @param flags Detach flags (e.g. MNT_DETACH).
 * @return Result with true on success or error.
 * Usage: Container teardown.
 */
static inline Result_bool vira_sys_unmount(vira_string target, int32_t flags) {
#if defined(__linux__)
  char stack_buf[1024];
  vira_cstr_buf b = vira_str_to_cstr_buf(target, stack_buf, sizeof(stack_buf));
  int res = umount2(b.ptr, flags);
  vira_cstr_buf_free(&b);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("umount2 error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)target; (void)flags;
  return (Result_bool){.is_err = true, .err = vira_str_lit("umount2: only supported on Linux", 32)};
#endif
}

/**
 * Changes the root directory of the calling process to the specified path (jail/container boundary).
 * @param path New root directory path.
 * @return Result with true on success or error.
 * Usage: Sandboxing and container isolation.
 */
static inline Result_bool vira_sys_chroot(vira_string path) {
#if defined(__linux__)
  char stack_buf[1024];
  vira_cstr_buf b = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
  int res = chroot(b.ptr);
  vira_cstr_buf_free(&b);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("chroot error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)path;
  return (Result_bool){.is_err = true, .err = vira_str_lit("chroot: only supported on Linux", 31)};
#endif
}

/**
 * Creates a filesystem node (special file, FIFO, or device).
 * @param path Node path.
 * @param mode File mode and type flags (S_IFCHR, S_IFBLK, etc.).
 * @param dev Device number created with `makedev`.
 * @return Result with true on success or error.
 * Usage: Provisioning device nodes inside container filesystems.
 */
static inline Result_bool vira_sys_mknod(vira_string path, int32_t mode, int64_t dev) {
#if defined(__linux__)
  char stack_buf[1024];
  vira_cstr_buf b = vira_str_to_cstr_buf(path, stack_buf, sizeof(stack_buf));
  int res = mknod(b.ptr, (mode_t)mode, (dev_t)dev);
  vira_cstr_buf_free(&b);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("mknod error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)path; (void)mode; (void)dev;
  return (Result_bool){.is_err = true, .err = vira_str_lit("mknod: only supported on Linux", 30)};
#endif
}

/**
 * Combines major and minor device numbers into a 64-bit device ID.
 * @param major Major device number.
 * @param minor Minor device number.
 * @return Device ID.
 * Usage: Parameter construction for `sys::mknod`.
 */
static inline int64_t vira_sys_makedev(int32_t major, int32_t minor) {
#if defined(__linux__)
  return (int64_t)gnu_dev_makedev((unsigned int)major, (unsigned int)minor);
#else
  return ((int64_t)(major) << 8) | (int64_t)(minor);
#endif
}

/**
 * Disassociates parts of the process execution context into new Linux namespaces (PID, Mount, Net, UTS, IPC).
 * @param flags Namespace flag bitmask (CLONE_NEWPID, CLONE_NEWNS, CLONE_NEWNET, etc.).
 * @return Result with true on success or error.
 * Usage: Creating unprivileged Linux containers.
 */
static inline Result_bool vira_sys_unshare(int32_t flags) {
#if defined(__linux__)
  if (unshare(flags) != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("unshare error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)flags;
  return (Result_bool){.is_err = true, .err = vira_str_lit("unshare: only supported on Linux", 32)};
#endif
}

/**
 * Updates the system or container UTS namespace hostname.
 * @param name New hostname string.
 * @return Result with true on success or error.
 * Usage: Configuring container network identity.
 */
static inline Result_bool vira_sys_sethostname(vira_string name) {
#if defined(__linux__)
  char stack_buf[256];
  vira_cstr_buf b = vira_str_to_cstr_buf(name, stack_buf, sizeof(stack_buf));
  int res = sethostname(b.ptr, strlen(b.ptr));
  vira_cstr_buf_free(&b);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("sethostname error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
#else
  (void)name;
  return (Result_bool){.is_err = true, .err = vira_str_lit("sethostname: only supported on Linux", 36)};
#endif
}

/**
 * Spawns an external command with arguments, working directory, and wait, returning its exit code.
 * @param args Slice of command and argument strings.
 * @param dir Optional working directory (empty string for current dir).
 * @param inherit_stdio True to inherit standard input/output streams.
 * @return Result containing integer exit code or error message.
 * Usage: Child process execution (`sys::exec_run(["git", "status"], "", true)`).
 */
static inline Result_int vira_sys_exec_run(vira_slice_string args, vira_string dir, bool inherit_stdio) {
  if (args.data == NULL || args.len <= 0) {
    return (Result_int){.is_err = true, .err = vira_str_lit("empty command", 13)};
  }
#ifdef _WIN32
  size_t cmd_len = 0;
  for (int32_t i = 0; i < args.len; i++) {
    cmd_len += (size_t)args.data[i].len + 3;
  }
  size_t cmd_cap = (cmd_len < 4096) ? 4096 : (cmd_len + 1024);
  char *cmd_line = (char *)calloc(1, cmd_cap);
  for (int32_t i = 0; i < args.len; i++) {
    char sbuf[256];
    vira_cstr_buf b = vira_str_to_cstr_buf(args.data[i], sbuf, sizeof(sbuf));
    if (i > 0) strcat(cmd_line, " ");
    bool need_quotes = (strchr(b.ptr, ' ') != NULL || b.ptr[0] == '\0');
    if (need_quotes) strcat(cmd_line, "\"");
    strcat(cmd_line, b.ptr);
    if (need_quotes) strcat(cmd_line, "\"");
    vira_cstr_buf_free(&b);
  }

  STARTUPINFOA si;
  PROCESS_INFORMATION pi;
  memset(&si, 0, sizeof(si));
  memset(&pi, 0, sizeof(pi));
  si.cb = sizeof(si);

  char dstack[1024];
  vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
  const char *cdir = (dir.len > 0) ? db.ptr : NULL;

  BOOL ok = CreateProcessA(NULL, cmd_line, NULL, NULL, TRUE, 0, NULL, cdir, &si, &pi);
  free(cmd_line);
  vira_cstr_buf_free(&db);
  if (!ok) {
    return (Result_int){.is_err = true, .err = vira_str_lit("CreateProcess failed", 20)};
  }
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD exit_code = 0;
  GetExitCodeProcess(pi.hProcess, &exit_code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  return (Result_int){.is_err = false, .val = (int32_t)exit_code};
#else
  (void)inherit_stdio;
  pid_t pid = fork();
  if (pid == 0) {
    if (dir.len > 0) {
      char dstack[1024];
      vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
      if (chdir(db.ptr) != 0) { /* continue */ }
      vira_cstr_buf_free(&db);
    }
    char **argv = (char **)malloc((size_t)(args.len + 1) * sizeof(char *));
    for (int32_t i = 0; i < args.len; i++) {
      argv[i] = vira_str_cstr(args.data[i]);
    }
    argv[args.len] = NULL;
    execvp(argv[0], argv);
    _exit(127);
  }
  if (pid < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("fork failed: %s", strerror(errno))};
  }
  int status = 0;
  waitpid(pid, &status, 0);
  int code = WIFEXITED(status) ? WEXITSTATUS(status) : (WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1);
  return (Result_int){.is_err = false, .val = code};
#endif
}

/**
 * Spawns an external command, passes optional standard input, captures its standard output/error, and waits for completion.
 * @param args Slice of command and argument strings.
 * @param dir Working directory (or empty string).
 * @param stdin_text Text to pipe into standard input.
 * @param combine_err True to combine standard error into output string.
 * @return Result containing captured stdout string or error message.
 * Usage: Running commands and capturing output (`sys::exec_output(["whoami"], "", "", true)`).
 */
static inline Result_string vira_sys_exec_output(vira_slice_string args, vira_string dir, vira_string stdin_text, bool combine_err) {
  if (args.data == NULL || args.len <= 0) {
    return (Result_string){.is_err = true, .err = vira_str_lit("empty command", 13)};
  }
#ifdef _WIN32
  (void)stdin_text;
  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(SECURITY_ATTRIBUTES);
  sa.bInheritHandle = TRUE;
  sa.lpSecurityDescriptor = NULL;

  HANDLE hRead, hWrite;
  if (!CreatePipe(&hRead, &hWrite, &sa, 0)) {
    return (Result_string){.is_err = true, .err = vira_str_lit("CreatePipe failed", 17)};
  }
  SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

  size_t cmd_len = 0;
  for (int32_t i = 0; i < args.len; i++) {
    cmd_len += (size_t)args.data[i].len + 3;
  }
  size_t cmd_cap = (cmd_len < 4096) ? 4096 : (cmd_len + 1024);
  char *cmd_line = (char *)calloc(1, cmd_cap);
  for (int32_t i = 0; i < args.len; i++) {
    char sbuf[256];
    vira_cstr_buf b = vira_str_to_cstr_buf(args.data[i], sbuf, sizeof(sbuf));
    if (i > 0) strcat(cmd_line, " ");
    bool need_quotes = (strchr(b.ptr, ' ') != NULL || b.ptr[0] == '\0');
    if (need_quotes) strcat(cmd_line, "\"");
    strcat(cmd_line, b.ptr);
    if (need_quotes) strcat(cmd_line, "\"");
    vira_cstr_buf_free(&b);
  }

  STARTUPINFOA si;
  PROCESS_INFORMATION pi;
  memset(&si, 0, sizeof(si));
  memset(&pi, 0, sizeof(pi));
  si.cb = sizeof(si);
  si.dwFlags |= STARTF_USESTDHANDLES;
  si.hStdOutput = hWrite;
  si.hStdError = combine_err ? hWrite : GetStdHandle(STD_ERROR_HANDLE);
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

  char dstack[1024];
  vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
  const char *cdir = (dir.len > 0) ? db.ptr : NULL;

  BOOL ok = CreateProcessA(NULL, cmd_line, NULL, NULL, TRUE, 0, NULL, cdir, &si, &pi);
  free(cmd_line);
  vira_cstr_buf_free(&db);
  CloseHandle(hWrite);

  if (!ok) {
    CloseHandle(hRead);
    return (Result_string){.is_err = true, .err = vira_str_lit("CreateProcess failed", 20)};
  }

  size_t cap = 1024;
  size_t len = 0;
  char *out = (char *)malloc(cap);
  DWORD bytes_read = 0;
  char chunk[4096];
  while (ReadFile(hRead, chunk, sizeof(chunk), &bytes_read, NULL) && bytes_read > 0) {
    if (len + bytes_read + 1 > cap) {
      while (len + bytes_read + 1 > cap) cap *= 2;
      char *nb = (char *)realloc(out, cap);
      if (!nb) break;
      out = nb;
    }
    memcpy(out + len, chunk, bytes_read);
    len += bytes_read;
  }
  out[len] = '\0';
  CloseHandle(hRead);
  WaitForSingleObject(pi.hProcess, INFINITE);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  _vira_record_alloc();
  return (Result_string){.is_err = false, .val = (vira_string){.data = (const uint8_t *)out, .len = (int32_t)len, .is_lit = false}};
#else
  (void)stdin_text;
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    return (Result_string){.is_err = true, .err = vira_str_format("pipe error: %s", strerror(errno))};
  }
  pid_t pid = fork();
  if (pid == 0) {
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    if (combine_err) {
      dup2(pipefd[1], STDERR_FILENO);
    }
    close(pipefd[1]);
    if (dir.len > 0) {
      char dstack[1024];
      vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
      if (chdir(db.ptr) != 0) { /* continue */ }
      vira_cstr_buf_free(&db);
    }
    char **argv = (char **)malloc((size_t)(args.len + 1) * sizeof(char *));
    for (int32_t i = 0; i < args.len; i++) {
      argv[i] = vira_str_cstr(args.data[i]);
    }
    argv[args.len] = NULL;
    execvp(argv[0], argv);
    _exit(127);
  }
  close(pipefd[1]);
  if (pid < 0) {
    close(pipefd[0]);
    return (Result_string){.is_err = true, .err = vira_str_format("fork failed: %s", strerror(errno))};
  }
  size_t cap = 1024;
  size_t len = 0;
  char *out = (char *)malloc(cap);
  char chunk[4096];
  ssize_t n;
  while ((n = read(pipefd[0], chunk, sizeof(chunk))) > 0) {
    if (len + (size_t)n + 1 > cap) {
      while (len + (size_t)n + 1 > cap) cap *= 2;
      char *nb = (char *)realloc(out, cap);
      if (!nb) break;
      out = nb;
    }
    memcpy(out + len, chunk, (size_t)n);
    len += (size_t)n;
  }
  out[len] = '\0';
  close(pipefd[0]);
  waitpid(pid, NULL, 0);
  _vira_record_alloc();
  return (Result_string){.is_err = false, .val = (vira_string){.data = (const uint8_t *)out, .len = (int32_t)len, .is_lit = false}};
#endif
}

/**
 * Spawns an external process asynchronously in the background, optionally inside new Linux namespaces.
 * @param args Command and argument strings.
 * @param dir Working directory.
 * @param clone_flags Linux clone/namespace flags (or 0 for normal spawn).
 * @param inherit_stdio True to inherit standard streams.
 * @return Result containing spawned PID or error message.
 * Usage: Asynchronous process spawning and daemon launch (`sys::proc_start(...)`).
 */
static inline Result_int vira_sys_proc_start(vira_slice_string args, vira_string dir, int32_t clone_flags, bool inherit_stdio) {
  if (args.data == NULL || args.len <= 0) {
    return (Result_int){.is_err = true, .err = vira_str_lit("empty command", 13)};
  }
#ifdef _WIN32
  (void)clone_flags;
  (void)inherit_stdio;
  size_t cmd_len = 0;
  for (int32_t i = 0; i < args.len; i++) {
    cmd_len += (size_t)args.data[i].len + 3;
  }
  size_t cmd_cap = (cmd_len < 4096) ? 4096 : (cmd_len + 1024);
  char *cmd_line = (char *)calloc(1, cmd_cap);
  for (int32_t i = 0; i < args.len; i++) {
    char sbuf[256];
    vira_cstr_buf b = vira_str_to_cstr_buf(args.data[i], sbuf, sizeof(sbuf));
    if (i > 0) strcat(cmd_line, " ");
    bool need_quotes = (strchr(b.ptr, ' ') != NULL || b.ptr[0] == '\0');
    if (need_quotes) strcat(cmd_line, "\"");
    strcat(cmd_line, b.ptr);
    if (need_quotes) strcat(cmd_line, "\"");
    vira_cstr_buf_free(&b);
  }

  STARTUPINFOA si;
  PROCESS_INFORMATION pi;
  memset(&si, 0, sizeof(si));
  memset(&pi, 0, sizeof(pi));
  si.cb = sizeof(si);

  char dstack[1024];
  vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
  const char *cdir = (dir.len > 0) ? db.ptr : NULL;

  BOOL ok = CreateProcessA(NULL, cmd_line, NULL, NULL, TRUE, 0, NULL, cdir, &si, &pi);
  free(cmd_line);
  vira_cstr_buf_free(&db);
  if (!ok) {
    return (Result_int){.is_err = true, .err = vira_str_lit("CreateProcess failed", 20)};
  }
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  return (Result_int){.is_err = false, .val = (int32_t)pi.dwProcessId};
#else
  pid_t pid;
  if (clone_flags != 0) {
#if defined(__linux__) && defined(SYS_clone)
    pid = (pid_t)syscall(SYS_clone, clone_flags | SIGCHLD, NULL, NULL, NULL, 0);
#else
    pid = fork();
#endif
  } else {
    pid = fork();
  }
  if (pid == 0) {
    if (dir.len > 0) {
      char dstack[1024];
      vira_cstr_buf db = vira_str_to_cstr_buf(dir, dstack, sizeof(dstack));
      if (chdir(db.ptr) != 0) { /* continue */ }
      vira_cstr_buf_free(&db);
    }
    char **argv = (char **)malloc((size_t)(args.len + 1) * sizeof(char *));
    for (int32_t i = 0; i < args.len; i++) {
      argv[i] = vira_str_cstr(args.data[i]);
    }
    argv[args.len] = NULL;
    execvp(argv[0], argv);
    _exit(127);
  }
  if (pid < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("proc_start failed: %s", strerror(errno))};
  }
  return (Result_int){.is_err = false, .val = (int32_t)pid};
#endif
}

/**
 * Waits for a spawned child process PID to exit and retrieves its exit code.
 * @param pid Child process ID.
 * @return Result containing integer exit code or error message.
 * Usage: Awaiting background child processes (`sys::proc_wait(pid)`).
 */
static inline Result_int vira_sys_proc_wait(int32_t pid) {
#ifdef _WIN32
  HANDLE h = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_INFORMATION, FALSE, (DWORD)pid);
  if (!h) {
    return (Result_int){.is_err = true, .err = vira_str_lit("failed to open process handle for wait", 39)};
  }
  WaitForSingleObject(h, INFINITE);
  DWORD exit_code = 0;
  GetExitCodeProcess(h, &exit_code);
  CloseHandle(h);
  return (Result_int){.is_err = false, .val = (int32_t)exit_code};
#else
  int status = 0;
  pid_t w = waitpid((pid_t)pid, &status, 0);
  if (w < 0) {
    return (Result_int){.is_err = true, .err = vira_str_format("waitpid error: %s", strerror(errno))};
  }
  int code = WIFEXITED(status) ? WEXITSTATUS(status) : (WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1);
  return (Result_int){.is_err = false, .val = code};
#endif
}

#endif // VIRA_RT_SYS_H
