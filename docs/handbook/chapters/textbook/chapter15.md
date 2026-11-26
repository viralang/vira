# Chapter 15: I/O, Filesystem, CLI Applications & Subprocesses

Beyond memory safety and actors, systems programs spend most of their time interacting with the operating system: reading configuration and `/proc` virtual files, inspecting paths and environment variables, parsing command-line flags, emitting structured logs, spawning subprocesses, and driving low-level file descriptors. This chapter tours Vira's systems standard library: `io`, `os`, `flag`, `log`, `exec`, and `syscall`.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch15_io_cli_and_processes.vir`](../code/ch15_io_cli_and_processes.vir)

---

## 15.1. File I/O Builtins and Buffered Streams (`import io`)

For quick one-shot file operations, Vira provides five built-in functions available in every module without an import:

- `file_exists(path string) bool`
- `read_file(path string) !string`
- `write_file(path string, content string) !bool`
- `append_file(path string, content string) !bool`
- `remove_file(path string) !bool`

### Reading `/proc` and Virtual Filesystems

Notice an important systems detail in the runtime (`vira_read_file` in `runtime/rt/io/io.h`): when reading regular files, Vira seeks to `SEEK_END` to size the buffer in a single read. However, virtual filesystems on Linux such as `/proc/version` or `/sys/...` (and non-seekable pipes) report `ftell == 0` or fail `fseek`. Vira automatically falls back to `_vira_read_file_stream`, which reads in chunks until EOF—so `read_file('/proc/version')` works seamlessly without special-casing.

### Streaming & Buffered I/O (`import io`)

When you need incremental streaming, seeking, line-by-line parsing, or interactive terminal input, `import io` provides:

| Type / Function | Key Methods & Purpose |
| :--- | :--- |
| **`io.File`** | `File.open(path, mode) !File`, `.read(mut buf []u8) !int`, `.write(buf []u8) !int`, `.write_string(s) !int`, `.seek(offset, whence) !i64`, `.flush() !bool`, `.close() !bool`, `.size() i64`. |
| **`io.BufReader`** | `BufReader.new(f File)`, `BufReader.from_string(s)`, `BufReader.from_bytes(b)`, `.read_byte() !u8`, `.read_line() !string` (strips `\r\n` or `\n`), `.read_exact(count) !string`, `.read_all() string`, `.close() !bool`. |
| **Structural Interfaces** | `io.Reader`, `io.Writer`, `io.Closer`, `io.ReadCloser`, `io.WriteCloser` — all structurally satisfied by `io.File`. |
| **Stdio & File Helpers** | `io.read_line() !string`, `io.read_all_stdin() !string`, `io.prompt(msg) !string`, `io.rename(old, new) !bool`, `io.remove(path) !bool`, `io.temp_file() !string`. |

Pair `remove_file` with `defer` right after declaring a temporary path so cleanup runs on every exit path:

```v
demo_path := '/tmp/vira_ch15_demo.txt'
defer remove_file(demo_path)

_ := write_file(demo_path, 'alpha=10\n') or { return }
_ := append_file(demo_path, 'beta=20\n') or { return }

mut f := io.File.open(demo_path, 'rb') or { return }
mut br := io.BufReader.new(f)
line1 := br.read_line() or { '' }
line2 := br.read_line() or { '' }
br.close() or { false }
```

---

## 15.2. Environment, Arguments, and Paths (`import os`)

The `os` package wraps process identity, environment variables, directory operations, and cross-platform path utilities:

- **Arguments & Platform:** `os.args() []string`, `os.platform() string`, `os.is_windows() bool`, `os.exit(code int)`, `os.set_quiet_rt(quiet bool)`.
- **Process Identity & Environment:** `os.getpid()`, `os.getppid()`, `os.getuid()`, `os.getgid()`, `os.hostname() !string`, `os.executable() !string`, `os.getenv(name) string`, `os.setenv(name, val) !bool`, `os.unsetenv(name) !bool`, `os.environ() []string`.
- **Filesystem & Directories:** `os.exists`, `os.is_file`, `os.is_dir`, `os.stat`, `os.mkdir(path, perm)`, `os.mkdir_all(path, perm)`, `os.read_dir(path) ![]string`, `os.remove(path)`, `os.remove_all(path)`, `os.read_bytes`, `os.read_bytes_head`, `os.write_bytes`.
- **Paths:** `os.join_path(p1, p2)`, `os.base_name(path)`, `os.dir_name(path)`, `os.ext(path)`, and `os.clean_path(path)` (normalizes `\` to `/`).

```v
_ := os.setenv('VIRA_CH15_MODE', 'cookbook') or { false }
mode := os.getenv('VIRA_CH15_MODE')
_ := os.unsetenv('VIRA_CH15_MODE') or { false }

joined := os.join_path('/tmp', 'builds/app.vir')
cleaned := os.clean_path('assets\\models\\weights.bin')
```

---

## 15.3. Declarative Command-Line Flag Parsing (`import flag`) and Structured Logging (`import log`)

### `flag.FlagSet`

Instead of hand-parsing `os.args()`, `flag.FlagSet` lets you declare typed flags with both long names and short aliases:

```v
mut fs := flag.FlagSet.new('vira_tool')
fs.bool_flag('verbose', 'v', false, 'Enable debug logs')
fs.int_flag('workers', 'w', 4, 'Worker thread count')
fs.string_flag('target', 't', 'local', 'Target environment')

sample_argv := ['--verbose', '-w', '8', '--target=staging', 'job_a', 'job_b']
_ := fs.parse(sample_argv) or { return }
```

`FlagSet.parse` supports `--name=value`, `--name value`, `-short value`, `--` pass-through markers, and `FlagSet.new` also supports `i64_flag` and `f64_flag`. Retrieve typed values with `fs.get_bool`, `fs.get_int`, `fs.get_i64`, `fs.get_f64`, and `fs.get_string`, and read remaining non-flag arguments via `fs.positional()` (or `fs.args`). `fs.help_string()` / `fs.print_help()` formats usage text automatically, and `flag` also exports ANSI helpers (`ansi_bold`, `ansi_green`, `ansi_yellow`, `ansi_red`, `ansi_blue`, `ansi_cyan`).

> **Note:** When passing `os.args()` from a real CLI binary to `fs.parse(...)`, slice off `args[0]` (the program name) first with `args.slice(1, args.len)`.

### `log.Logger`

The `log` package emits leveled diagnostics to `stderr` (`LEVEL_DEBUG`, `LEVEL_INFO`, `LEVEL_WARN`, `LEVEL_ERROR`, `LEVEL_FATAL`):

```v
mut logger := log.Logger.new(log.LEVEL_INFO)
logger.set_prefix('ch15')
logger.set_timestamp(false)
if fs.get_bool('verbose') {
	logger.set_level(log.LEVEL_DEBUG)
}
logger.debug('Parsed workers=${fs.get_int('workers')}, target=${fs.get_string('target')}')
logger.info('Positional args count=${positionals.len}')
```

Package-level functions (`log.debug`, `log.info`, `log.warn`, `log.error`, `log.fatal`) are also available when you do not need a custom prefix or level threshold.

---

## 15.4. Spawning Subprocesses (`import exec`) and Raw Kernel Syscalls (`import syscall`)

### `exec.Command`

Use `exec.Command.new(name, extra_args)` to spawn external processes safely without going through a shell:
- Configure options with `.set_dir(dir)`, `.set_stdin(text)`, `.set_inherit_stdio(bool)`, and `.set_clone_flags(flags)` (for Linux namespace flags).
- Run synchronously with `.run() !int` (returns exit code), `.output() !string` (captures `stdout`), or `.combined_output() !string` (captures `stdout + stderr`).
- Or manage process lifetimes asynchronously with `.start() !int` (returns child PID) followed by `.wait() !int`.

### `import syscall`

When you need raw POSIX file descriptors, kernel pipes, signals, or Linux container primitives, `import syscall` exposes direct intrinsics:
- **File Descriptors & IPC:** `open`, `read`, `write`, `close`, `lseek`, `pipe`, `dup`, `dup2`, `isatty`, `get_errno`, `strerror`.
- **Processes & Signals:** `getpid`, `getppid`, `getuid`, `getgid`, `kill`, `signal_notify`, `signal_poll`, `signal_wait` (`SIGHUP`, `SIGINT`, `SIGTERM`, `SIGKILL`, `SIGQUIT`).
- **Linux Isolation Constants & Calls:** `unshare`, `mount`, `unmount`, `chroot`, `mknod`, `makedev`, `sethostname`, and `CLONE_NEWNS` / `CLONE_NEWPID` / `CLONE_NEWNET` / `CLONE_NEWUTS` / `CLONE_NEWIPC` / `CLONE_NEWUSER` / `CLONE_NEWCGROUP`.

```v
mut fds := []int{}
syscall.pipe(mut fds) or { return }
r_fd := fds[0]
w_fd := fds[1]
defer syscall.close(r_fd)
defer syscall.close(w_fd)

payload := 'PIPE_OK'
wrote := syscall.write(w_fd, payload.bytes(), payload.len) or { 0 }
mut rx := []u8{}
read_n := syscall.read(r_fd, mut rx, 64) or { 0 }
```

---

## 15.5. Running the Complete Chapter 15 Example

Run [`docs/handbook/chapters/code/ch15_io_cli_and_processes.vir`](../code/ch15_io_cli_and_processes.vir):

```bash
./vira run docs/handbook/chapters/code/ch15_io_cli_and_processes.vir
```

Output:
```text
=== [1] File Builtins, io.File, io.BufReader & /proc ===
Builtins read (17 bytes), exists=true
BufReader lines (size=17): "alpha=10", "beta=20"
BufReader.from_string roundtrip len=17
/proc/version chunked read ok: non_empty=true

=== [2] Environment, Process Info & Paths (import os) ===
Platform=linux, pid=true, env="cookbook" -> unset=""
Paths: join="/tmp/builds/app.vir", base="app.vir", dir="/tmp/builds", ext=".vir", clean="assets/models/weights.bin"

=== [3] CLI Flag Parsing (import flag) & Logging (import log) ===
[ch15] [DEBUG] Parsed workers=8, target=staging
[ch15] [INFO] Positional args count=2
Flags: verbose=true, workers=8, target=staging, pos=job_a,job_b

=== [4] Subprocesses (import exec) & Kernel Pipes (import syscall) ===
exec.Command output: "vira-subprocess-ok"
syscall.pipe IPC: wrote=7, read=7, msg="PIPE_OK"

Cleanup check: /tmp/vira_ch15_demo.txt exists=false
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=62
```

---

## Summary & Exercises

1. **Real CLI Arguments:** Update `demo_flag_and_log` to parse `os.args().slice(1, os.args().len)` when `os.args().len > 1`, and run `./vira run docs/handbook/chapters/code/ch15_io_cli_and_processes.vir -- --workers=16 --target=prod task1`.
2. **Binary File Seek & Read:** Use `io.File.open` in `'wb'` / `'rb'` mode with `.seek(6, 0)` (from start of file) and `io.BufReader` to verify random-access positioning inside `/tmp/vira_ch15_demo.txt`.
3. **Subprocess Pipeline:** Use `exec.Command.new('cat', [])` with `.set_stdin('hello from stdin\n')` and `.output()` to feed input into a child process and read back its `stdout` with `active_leaks=0`.
