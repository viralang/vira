# `packages/syscall`: Low-Level Kernel Syscalls & Container Primitives

The `syscall` package provides direct access to operating system kernel calls, low-level file descriptors, Linux container isolation, and async-signal-safe primitives in the **Vira Programming Language**.

---

## 🚀 Key Features

- **File Descriptors**: `syscall.open`, `syscall.read`, `syscall.write`, `syscall.close`, `syscall.lseek`.
- **Pipes & Duplication**: `syscall.pipe(mut fds []int)`, `syscall.dup(fd)`, `syscall.dup2(old_fd, new_fd)`, `syscall.isatty(fd)`.
- **Linux Container Primitives**:
  - `syscall.mount(src, target, fs, flags, data) !bool`
  - `syscall.unmount(target, flags) !bool` (`umount2`)
  - `syscall.chroot(path) !bool`
  - `syscall.mknod(path, mode, dev) !bool`, `syscall.makedev(major, minor) i64`
  - `syscall.unshare(flags) !bool`, `syscall.sethostname(name) !bool`
  - Namespace flags: `CLONE_NEWPID`, `CLONE_NEWUTS`, `CLONE_NEWNS`, `CLONE_NEWNET`, `CLONE_NEWIPC`.
- **Signal Notification**: Async-signal-safe self-pipe (`syscall.signal_notify`, `syscall.signal_wait`, `syscall.signal_poll`).
- **Error Diagnostics**: `syscall.get_errno()` and `syscall.strerror(err)`.

---

## 🛠️ Usage Example

```v
module main

import syscall

fn main() {
    // 1. Raw UNIX pipe IPC
    mut fds := [0, 0]
    syscall.pipe(mut fds) or { panic(err) }
    defer {
        syscall.close(fds[0]) or { false }
        syscall.close(fds[1]) or { false }
    }

    // Write message into write-end of pipe
    msg := 'vira-syscall-kernel-pipe'
    syscall.write(fds[1], msg.bytes(), msg.len) or { panic(err) }

    // Read message from read-end of pipe
    mut buf := []u8{}
    for mut i := 0; i < 64; i += 1 { buf.push(u8(0)) }
    n := syscall.read(fds[0], mut buf, 64) or { panic(err) }
    println('Successfully transferred ${n} bytes through raw kernel pipe')
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/syscall
```
