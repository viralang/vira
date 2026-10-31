# Vira Standard Library (`packages/`)

Welcome to the **Vira Standard Library**! This directory houses the 14 first-party standard library packages distributed with Vira.

Every package is implemented in idiomatic, pure Vira code backed by lightweight C99 shims in [`runtime/vira_rt.h`](file:///mnt/flutter_drive/vira/runtime/vira_rt.h), providing zero-GC execution, compile-time Autofree safety, and zero memory leaks.

---

## 📦 Package Directory & Module Index

| Directory | Module | Description | Guide & Docs |
| :--- | :--- | :--- | :--- |
| [`os/`](./os) | `module os` | CLI args (`os.args()`), PID, env vars, files, directories, paths (`os.join_path`), platform queries. | [README](./os/README.md) |
| [`exec/`](./exec) | `module exec` | Child processes (`exec.Command`), stdio redirection, `CLONE_*` flags, async start/wait, `combined_output`. | [README](./exec/README.md) |
| [`syscall/`](./syscall) | `module syscall` | Low-level kernel calls: file descriptors, pipes, Linux namespaces, mounts, `chroot`, `mknod`, signals. | [README](./syscall/README.md) |
| [`flag/`](./flag) | `module flag` | POSIX/GNU command-line flag parsing (`FlagSet`, typed flags, automated `--help` table). | [README](./flag/README.md) |
| [`log/`](./log) | `module log` | Leveled structured stderr logger (`DEBUG`, `INFO`, `WARN`, `ERROR`, `FATAL`) with timestamps and prefixes. | [README](./log/README.md) |
| [`strings/`](./strings) | `module strings` | String processing: `fields` (Go-compatible), `join`, `split`, `replace`, `trim`, `pad`, `sort`, `StringBuilder`. | [README](./strings/README.md) |
| [`math/`](./math) | `module math` | High-performance C99 math intrinsics (trigonometry, hyperbolic, exp, log, pow, sqrt, clamp, `PI`, `E`). | [README](./math/README.md) |
| [`time/`](./time) | `module time` | Nanosecond wall-clock, monotonic timers, microsecond sleep, `strftime` formatting, and `Duration`. | [README](./time/README.md) |
| [`ctype/`](./ctype) | `module ctype` | Fast ASCII character classification (`is_alpha`, `is_digit`, `is_space`, etc.) and case conversion. | [README](./ctype/README.md) |
| [`io/`](./io) | `module io` | Stream traits (`Reader`, `Writer`, `Closer`), `File` handles, `BufReader`, `prompt`, `read_line`. | [README](./io/README.md) |
| [`net/`](./net) | `module net` | Cross-platform networking: `TcpListener`, `TcpStream`, `UdpSocket`, and non-blocking `Poller`. | [README](./net/README.md) |
| [`http/`](./http) | `module http` | HTTP/1.1 request/response parser, `HeaderMap`, server connection loops, Server-Sent Events (`SseStream`). | [README](./http/README.md) |
| [`json/`](./json) | `module json` | Pure-Vira RFC 8259 parser, validator (`json.is_valid`), depth-aware field extractors, and $O(N)$ `JsonBuilder`. | [README](./json/README.md) |
| [`viraserve/`](./viraserve) | `module viraserve` | Actor-driven asynchronous HTTP microframework and router with zero memory leaks. | [README](./viraserve/README.md) |

---

## ⚡ How to Import Standard Packages

Standard packages are **resolved automatically by the compiler** (Tier 0 resolution). You do not need to configure `vira.mod` or download any external modules:

```v
module main

import os
import math
import strings
import time

fn main() {
    args := os.args()
    println('Running: ${args[0]}')

    val := math.sqrt(64.0)
    println('Sqrt: ${val}')

    joined := strings.join(['vira', 'standard', 'library'], '/')
    println('Path: ${joined}')

    now := time.now_unix()
    println('Current epoch: ${now}')
}
```

---

## 🧪 Testing Packages

All packages have automated unit test suites under their respective `tests/` directories. Run tests using `vira test`:

```bash
# Run unit tests for a specific package
vira test packages/os
vira test packages/exec
vira test packages/syscall
vira test packages/flag
vira test packages/log
vira test packages/strings
vira test packages/math
vira test packages/time
vira test packages/ctype
vira test packages/io
vira test packages/net
vira test packages/http
vira test packages/json
vira test packages/viraserve

# Run all standard library test suites
vira test packages/*
```

Every test verifies `active_leaks=0` upon completion, ensuring 100% deterministic memory management.

---

## 📚 In-Depth Documentation

For architectural tutorials and full API guides:
- [**Standard Library Manual**](../docs/STANDARD_LIBRARY.md)
- [**Builtins & API Reference**](../docs/BUILTINS_AND_API_REFERENCE.md)
- [**Tutorial 31: Basic CLI Applications**](../docs/tutorials/31_basic_cli_application_development.md)
- [**Tutorial 32: Advanced CLI Flags, Subprocesses & Containers**](../docs/tutorials/32_advanced_cli_flags_syscall_and_exec.md)
- [**Tutorial 33: ND Tensors & JSON**](../docs/tutorials/33_nd_tensors_and_stdlib_json.md)
