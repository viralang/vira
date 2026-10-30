# `packages/flag`: Declarative POSIX/GNU CLI Flag Parser

The `flag` package provides declarative command-line argument and flag parsing inspired by POSIX and GNU conventions in the **Vira Programming Language**.

---

## 🚀 Key Features

- **Typed Flag Declarations**:
  - `fs.bool_flag(name, short, def, usage)`
  - `fs.int_flag(name, short, def, usage)`
  - `fs.i64_flag(name, short, def, usage)`
  - `fs.f64_flag(name, short, def, usage)`
  - `fs.string_flag(name, short, def, usage)`
- **POSIX & GNU Syntax**: Supports both short (`-v`, `-p 8080`, `-p=8080`) and long (`--verbose`, `--port=8080`) flags.
- **Auto-Generated Help**: `fs.help_string()` generates aligned tables of options and descriptions.
- **Positional Arguments**: Preserves remaining non-flag arguments after `--` or positional boundaries in `fs.args`.

---

## 🛠️ Usage Example

```v
module main

import flag
import os

fn main() {
    mut fs := flag.FlagSet.new('app')
    fs.bool_flag('verbose', 'v', false, 'Enable verbose debug output')
    fs.int_flag('port', 'p', 8080, 'Server listening port')
    fs.string_flag('config', 'c', 'config.json', 'Path to config file')

    // Slice arguments (skipping program name in argv[0])
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

    if fs.get_bool('verbose') {
        println('Verbose mode enabled')
    }
    port := fs.get_int('port')
    println('Server port: ${port}')
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/flag
```
