# `packages/os`: Operating System & Process Primitives

The `os` package provides cross-platform access to command-line arguments, process attributes, environment variables, filesystem operations, and path manipulation in the **Vira Programming Language**.

---

## 🚀 Key Features

- **CLI Argument Inspection**: Direct access to program invocation and arguments via `os.args() []string`.
- **Process Information**: `os.getpid()`, `os.getppid()`, `os.getuid()`, `os.getgid()`, `os.hostname()`.
- **Environment Manipulation**: `os.getenv()`, `os.setenv()`, `os.unsetenv()`, `os.environ()`.
- **Filesystem Operations**: `os.read_file()`, `os.write_file()`, `os.mkdir()`, `os.mkdir_all()`, `os.remove()`, `os.remove_all()`, `os.read_dir()`, `os.exists()`, `os.is_file()`, `os.is_dir()`.
- **Path Helpers**: `os.join_path()`, `os.base_name()`, `os.dir_name()`, `os.ext()`, `os.clean_path()`.
- **Process Lifecycle**: `os.exit(code int)`, `os.set_quiet_rt(quiet bool)`.

---

## 🛠️ Usage Example

```v
module main

import os

fn main() {
    // 1. Inspect command-line arguments
    args := os.args()
    println('Program name: ${args[0]}')
    println('Argument count: ${args.len}')

    // 2. Read environment variable
    user := os.getenv('USER')
    println('Current user: ${user}')

    // 3. Create directories and files
    os.mkdir_all('dist/data', 0o755) or { panic(err) }
    defer os.remove_all('dist/data') or { false }

    path := os.join_path('dist/data', 'hello.txt')
    os.write_file(path, 'Hello from Vira OS package!\n') or { panic(err) }

    // 4. Verify file existence and read content
    if os.exists(path) {
        content := os.read_file(path) or { '' }
        println('File contents:\n${content}')
    }
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/os
```
