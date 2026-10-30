# `packages/exec`: Process Spawning & Supervision

The `exec` package manages child process execution, stdio pipe redirection, `CLONE_*` namespace flags, and asynchronous process lifecycles in the **Vira Programming Language**.

---

## 🚀 Key Features

- **Fluent Command Builder**: `exec.Command.new(prog string, args []string) Command`.
- **One-Shot Execution**:
  - `cmd.run() !int`: Runs process synchronously and returns exit status.
  - `cmd.output() !string`: Captures standard output as a string.
  - `cmd.combined_output() !string`: Merges standard output and standard error.
  - `exec.combined_output(prog, args) !string`: Shorthand helper.
- **Async Process Lifecycle**:
  - `pid := cmd.start() !int`: Starts subprocess in background, immediately returning child PID.
  - `exit_code := cmd.wait() !int`: Waits for child process to exit.
- **Container Isolation**: `cmd.set_clone_flags(flags int)` activates `SYS_clone` with Linux namespaces (`CLONE_NEWPID`, `CLONE_NEWUTS`, `CLONE_NEWNS`, `CLONE_NEWNET`, `CLONE_NEWIPC`).
- **Input / Output Redirection**: `cmd.set_stdin(text string)` and `cmd.set_inherit_stdio(bool)`.

---

## 🛠️ Usage Example

```v
module main

import exec

fn main() {
    // 1. One-shot command execution
    out := exec.combined_output('echo', ['Hello from subprocess!']) or { 'failed' }
    println('Output: ${out}')

    // 2. Asynchronous process management (start -> background setup -> wait)
    mut cmd := exec.Command.new('sleep', ['1'])
    pid := cmd.start() or { panic(err) }
    println('Started background child with PID: ${pid}')

    code := cmd.wait() or { panic(err) }
    println('Child process completed with status code: ${code}')
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/exec
```
