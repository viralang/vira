# Vira Standard Library Manual (`packages/`)

> **Version:** Vira `v0.5.13`+  
> **Package Root:** [`packages/`](file:///mnt/flutter_drive/vira/packages)  
> **Guarantees:** Zero Garbage Collection, Compile-Time Autofree, 100% Deterministic Cleanup (`active_leaks=0`), Strict Result Monad Error Handling (`!T`).

---

## 1. Overview & Automatic Package Resolution

Vira provides 14 first-party standard library packages implemented in pure Vira code backed by lean C99 intrinsics ([runtime/vira_rt.h](file:///mnt/flutter_drive/vira/runtime/vira_rt.h)).

Standard packages are stored directly in the `packages/` directory of the Vira distribution:
```
packages/
├── os/          # Operating system, process environment, paths, files & directories
├── exec/        # Child process execution, stdio pipes, clone flags, async start/wait
├── syscall/     # Low-level POSIX/Linux syscalls, file descriptors, mounts, namespaces
├── flag/        # Declarative POSIX/GNU command-line flag parsing
├── log/         # Leveled timestamped stderr logging
├── strings/     # String utilities (fields, join, split, replace, trim, StringBuilder)
├── math/        # C99 math intrinsics (trigonometry, exp, log, pow, sqrt, constants)
├── time/        # Wall-clock, monotonic timer, microsecond sleep, strftime, Duration
├── ctype/       # Fast ASCII character classification and case conversion
├── io/          # Unified Reader/Writer/Closer streams, File handles, BufReader, prompt
├── net/         # Cross-platform TCP listeners/streams, UDP sockets, non-blocking Poller
├── http/        # HTTP/1.1 request/response parser, HeaderMap, SSE writer, server loop
├── json/        # Pure-Vira RFC 8259 parser, validator, and O(N) JsonBuilder
└── viraserve/   # Actor-driven asynchronous HTTP microframework and router
```

### Zero-Configuration Imports
You do not need to download or declare standard packages in `vira.mod`. Simply import them by their module name:

```v
import os
import math
import strings
import time
```

The Vira compiler automatically resolves standard library packages from the bundled `packages/` directory during module resolution (Tier 0).

---

## 2. Package Catalog & Quick Reference

### 2.1 `packages/os` — System & Process Primitives
The [`os`](file:///mnt/flutter_drive/vira/packages/os) package gives direct, cross-platform access to process metadata, command-line arguments, environment variables, and the host filesystem:

- **CLI Arguments**: `os.args() []string` (returns program invocation and parameters).
- **Process Info**: `os.getpid() int`, `os.getppid() int`, `os.getuid() int`, `os.getgid() int`, `os.hostname() !string`.
- **Environment**: `os.getenv(name string) string`, `os.setenv(name, val)!`, `os.unsetenv(name)!`, `os.environ() []string`.
- **Filesystem**: `os.exists(path) bool`, `os.is_file(path) bool`, `os.is_dir(path) bool`, `os.read_file(path) !string`, `os.write_file(path, content) !bool`, `os.mkdir(path, perm)!`, `os.mkdir_all(path, perm)!`, `os.remove(path)!`, `os.remove_all(path)!`, `os.read_dir(path) ![]string`.
- **Paths**: `os.join_path(p1, p2) string`, `os.base_name(path) string`, `os.dir_name(path) string`, `os.ext(path) string`, `os.clean_path(path) string`.
- **Process Lifecycle**: `os.exit(code int)`, `os.set_quiet_rt(quiet bool)`.

```v
import os

fn main() {
    args := os.args()
    println('Program: ${args[0]}')

    user := os.getenv('USER')
    println('Current user: ${user}')

    os.mkdir_all('build/logs', 0o755) or { panic(err) }
    defer os.remove_all('build/logs') or { false }

    os.write_file('build/logs/app.log', 'Ready\n') or { panic(err) }
}
```

---

### 2.2 `packages/exec` — Subprocess Execution & Supervision
The [`exec`](file:///mnt/flutter_drive/vira/packages/exec) package manages child processes with support for pipe redirection, clone flags, and asynchronous tracking:

- **Command Builder**: `exec.Command.new(prog string, args []string) Command`
- **Configuration**: `cmd.set_dir(dir)`, `cmd.set_stdin(text)`, `cmd.set_inherit_stdio(bool)`, `cmd.set_clone_flags(flags)`.
- **One-Shot Execution**:
  - `cmd.run() !int`: Executes command and returns process exit code.
  - `cmd.output() !string`: Captures standard output.
  - `cmd.combined_output() !string`: Merges standard output and standard error.
  - `exec.combined_output(prog, args) !string`: Helper shorthand.
- **Async Execution**:
  - `pid := cmd.start() !int`: Starts subprocess in background and returns PID immediately.
  - `exit_code := cmd.wait() !int`: Awaits child termination and captures exit code.

```v
import exec

fn main() {
    // One-shot execution
    out := exec.combined_output('git', ['status', '--short']) or { 'error' }
    println('Git status:\n${out}')

    // Asynchronous lifecycle (ideal for container runtimes)
    mut cmd := exec.Command.new('sleep', ['1'])
    pid := cmd.start() or { panic(err) }
    println('Background child PID: ${pid}')
    code := cmd.wait() or { panic(err) }
    println('Exited with code: ${code}')
}
```

---

### 2.3 `packages/syscall` — Kernel System Calls & Linux Containers
The [`syscall`](file:///mnt/flutter_drive/vira/packages/syscall) package provides direct access to operating system kernel calls, low-level file descriptors, and Linux isolation primitives:

- **File Descriptors**: `open(path, flags, mode) !int`, `read(fd, mut buf, count) !int`, `write(fd, buf, count) !int`, `close(fd) !bool`, `lseek(fd, offset, whence) !i64`.
- **Pipes & Redirection**: `pipe(mut fds []int) !bool`, `dup(fd) !int`, `dup2(old_fd, new_fd) !int`, `isatty(fd) bool`.
- **Linux Containers & Namespaces**:
  - `mount(source, target, fstype, flags, data) !bool`
  - `unmount(target, flags) !bool` (`umount2`)
  - `chroot(path) !bool`
  - `mknod(path, mode, dev) !bool`, `makedev(major, minor) i64`
  - `unshare(flags) !bool`, `sethostname(name) !bool`
  - Namespace flags: `CLONE_NEWPID`, `CLONE_NEWUTS`, `CLONE_NEWNS`, `CLONE_NEWNET`, `CLONE_NEWIPC`.
- **Signal Handling**: `signal_notify(sigs []int) !bool`, `signal_wait() int`, `signal_poll() int`.

```v
import syscall

fn main() {
    // Pipe IPC example
    mut fds := [0, 0]
    syscall.pipe(mut fds) or { panic(err) }
    defer {
        syscall.close(fds[0]) or { false }
        syscall.close(fds[1]) or { false }
    }

    msg := 'kernel-pipe'
    syscall.write(fds[1], msg.bytes(), msg.len) or { panic(err) }

    mut buf := []u8{}
    for mut i := 0; i < 32; i += 1 { buf.push(u8(0)) }
    n := syscall.read(fds[0], mut buf, 32) or { panic(err) }
    println('Read ${n} bytes from raw kernel pipe')
}
```

---

### 2.4 `packages/flag` — POSIX & GNU CLI Argument Parsing
The [`flag`](file:///mnt/flutter_drive/vira/packages/flag) package allows you to declare and parse flags with type safety, defaults, and auto-generated manuals:

```v
import flag
import os

fn main() {
    mut fs := flag.FlagSet.new('server')
    fs.bool_flag('verbose', 'v', false, 'Enable verbose logging')
    fs.int_flag('port', 'p', 8080, 'Listening port')
    fs.string_flag('host', 'h', '127.0.0.1', 'Bind host address')

    raw := os.args()
    mut args := []string{}
    if raw.len > 1 {
        args = raw.slice(1, raw.len)
    }

    fs.parse(args) or {
        eprintln('Argument error: ${err}')
        eprintln(fs.help_string())
        return
    }

    port := fs.get_int('port')
    println('Running on port ${port}')
}
```

---

### 2.5 `packages/log` — Leveled Structured Logging
The [`log`](file:///mnt/flutter_drive/vira/packages/log) package formats diagnostic logs to `stderr` with timestamps, severity levels, and optional prefixes:

- **Levels**: `LEVEL_DEBUG (0)`, `LEVEL_INFO (1)`, `LEVEL_WARN (2)`, `LEVEL_ERROR (3)`, `LEVEL_FATAL (4)`.
- **Package Helpers**: `log.debug(msg)`, `log.info(msg)`, `log.warn(msg)`, `log.error(msg)`, `log.fatal(msg)` (terminates process with exit code 1).
- **Custom Logger**: `log.Logger.new(level)` with `.set_prefix(prefix)`, `.set_timestamp(bool)`.

```v
import log

fn main() {
    log.info('System initializing')
    log.warn('Cache miss rate above threshold')

    mut logger := log.Logger.new(log.LEVEL_DEBUG)
    logger.set_prefix('worker-0')
    logger.debug('Executing batch task')
}
```

---

### 2.6 `packages/strings` — String Utilities
The [`strings`](file:///mnt/flutter_drive/vira/packages/strings) package provides functions for slice manipulation, whitespace splitting, joining, and dynamic string building:

- **Splitting & Joining**:
  - `strings.fields(s string) []string`: Splits by whitespace, matching Go `strings.Fields`.
  - `strings.join(parts []string, sep string) string`: Joins strings with separator.
  - `strings.split(s string, delim string) []string`: Splits by delimiter.
- **Prefix & Suffix**: `strings.has_prefix`, `strings.has_suffix`, `strings.trim_prefix`, `strings.trim_suffix`.
- **Padding & Counting**: `strings.pad_left`, `strings.pad_right`, `strings.count`.
- **Base Conversion**: `strings.parse_i64_base(s, base) !i64`, `strings.format_i64_base(val, base) string`.
- **Sorting**: `strings.sort(mut s []string)`.
- **`StringBuilder`**: High-performance multi-part string construction (`write`, `write_line`, `to_string`, `clear`).

```v
import strings

fn main() {
    tokens := strings.fields('   container   start   --name=app   ')
    println('Tokens count: ${tokens.len}') // 3

    csv := strings.join(tokens, ',')
    println('CSV: ${csv}')

    mut sb := strings.StringBuilder.new()
    sb.write('Hello')
    sb.write(' ')
    sb.write('Vira!')
    println(sb.to_string())
}
```

---

### 2.7 `packages/math` — High-Performance Math
The [`math`](file:///mnt/flutter_drive/vira/packages/math) package exposes C99 floating-point operations:

- **Constants**: `math.PI`, `math.TAU`, `math.E`, `math.SQRT2`.
- **Trigonometry**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`.
- **Hyperbolics**: `sinh`, `cosh`, `tanh`.
- **Exponential & Logarithmic**: `exp`, `exp2`, `log`, `log2`, `log10`.
- **Powers & Roots**: `pow`, `sqrt`, `cbrt`, `hypot`.
- **Rounding & Modulo**: `ceil`, `floor`, `trunc`, `round`, `fmod`.
- **Helpers**: `abs`, `abs_int`, `abs_i64`, `min`, `max`, `clamp`, `is_nan`, `is_inf`.
- **Random Number Generation**: `srand(seed)`, `rand_int(min, max)`, `rand_f64()`.

```v
import math

fn main() {
    hyp := math.hypot(3.0, 4.0)
    println('Hypotenuse: ${hyp}') // 5.0

    rad := 45.0 * (math.PI / 180.0)
    println('Sin(45 deg): ${math.sin(rad)}')

    clamped := math.clamp(120.0, 0.0, 100.0)
    println('Clamped: ${clamped}') // 100.0
}
```

---

### 2.8 `packages/time` — High-Precision Clocks & Durations
The [`time`](file:///mnt/flutter_drive/vira/packages/time) package provides nanosecond wall-clock, monotonic timers, microsecond sleep, and `strftime` formatting:

- **Current Time**: `time.now_unix() i64`, `time.now_ms() i64`, `time.now_ns() i64`, `time.monotonic_ns() i64`.
- **Sleep**: `time.sleep_ms(ms int)`, `time.sleep_us(us int)`.
- **Formatting**: `time.format(fmt string, unix_sec i64, utc bool) !string`.
- **Duration**: `time.Duration.from_millis(ms)`, `time.Duration.from_secs(secs)`, `.millis()`, `.secs()`.

```v
import time

fn main() {
    start := time.monotonic_ns()
    time.sleep_ms(10)
    elapsed_ns := time.monotonic_ns() - start
    println('Slept for ${elapsed_ns / 1000000} ms')

    now := time.now_unix()
    formatted := time.format('%Y-%m-%d %H:%M:%S', now, false) or { 'unknown' }
    println('Local Time: ${formatted}')
}
```

---

### 2.9 `packages/ctype` — Fast ASCII Character Classification
The [`ctype`](file:///mnt/flutter_drive/vira/packages/ctype) package provides zero-overhead character inspection:

- **Classifiers**: `is_alpha(c u8) bool`, `is_digit(c u8) bool`, `is_alnum(c u8) bool`, `is_space(c u8) bool`, `is_upper(c u8) bool`, `is_lower(c u8) bool`, `is_hex_digit(c u8) bool`, `is_print(c u8) bool`, `is_punct(c u8) bool`, `is_cntrl(c u8) bool`.
- **Converters**: `to_lower(c u8) u8`, `to_upper(c u8) u8`.

```v
import ctype

fn main() {
    ch := u8(65) // 'A'
    println('Is Alpha: ${ctype.is_alpha(ch)}')
    println('Lower: ${ctype.to_lower(ch)}') // 97 ('a')
}
```

---

### 2.10 `packages/io` — Unified I/O & Streams
The [`io`](file:///mnt/flutter_drive/vira/packages/io) package provides stream abstractions, buffered I/O, and interactive console readers:

- **Traits**: `Reader`, `Writer`, `Closer`.
- **File Handles**: `io.File.open(path, mode) !File`, `io.File.create(path) !File`, `f.read(mut buf)`, `f.write(buf)`, `f.write_string(s)`, `f.close()`.
- **Buffered Reader**: `io.BufReader.new(file)`, `r.read_line() !string`, `r.read_all() ![]u8`.
- **Interactive Stdin**: `io.prompt(msg string) !string`, `io.read_line() !string`, `io.read_all_stdin() !string`.
- **Filesystem Utilities**: `io.rename(old, new) !bool`, `io.remove(path) !bool`, `io.temp_file() !string`.

```v
import io

fn main() {
    name := io.prompt('Enter your name: ') or { 'Guest' }
    println('Hello, ${name}!')

    mut f := io.File.create('greeting.txt') or { panic(err) }
    f.write_string('Welcome to Vira\n') or { panic(err) }
    f.close() or { false }
    defer io.remove('greeting.txt') or { false }
}
```

---

### 2.11 `packages/net`, `packages/http`, `packages/json`, & `packages/viraserve`
- **`net`**: Cross-platform TCP listeners/streams, UDP sockets, non-blocking `Poller`.
- **`http`**: HTTP/1.1 request/response parser, `HeaderMap`, server loops, and Server-Sent Events (`SseStream`).
- **`json`**: Pure-Vira RFC 8259 parser, validator (`json.is_valid`), depth-aware field extractors (`json.get_str`, `json.get_int`, `json.get_bool`), and $O(N)$ `json.JsonBuilder`.
- **`viraserve`**: Actor-driven HTTP microframework for serving high-throughput REST APIs and AI inference pipelines with zero memory leaks.

---

## 3. Testing Standard Packages

You can execute test suites across any package using `vira test`:

```bash
# Test an individual package
vira test packages/os
vira test packages/exec
vira test packages/syscall
vira test packages/strings
vira test packages/math
vira test packages/json

# Test all packages
vira test packages/*
```

All standard library tests verify `active_leaks=0` upon completion, ensuring production-grade safety.
