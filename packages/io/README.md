# `packages/io`: Stream Traits, File Handles & Console I/O

The `io` package provides unified stream traits (`Reader`, `Writer`, `Closer`), buffered reading, interactive standard input readers, and file manipulation in the **Vira Programming Language**.

---

## 🚀 Key Features

- **Structural Traits**:
  - `Reader`: `read(mut buf []u8) !int`
  - `Writer`: `write(buf []u8) !int`
  - `Closer`: `close() !bool`
- **File Abstraction (`io.File`)**:
  - `io.File.open(path, mode) !File`
  - `io.File.create(path) !File`
  - `f.read(mut buf)`, `f.write(buf)`, `f.write_string(s)`, `f.seek(offset, whence)`, `f.close()`
- **Buffered Reader (`io.BufReader`)**:
  - `io.BufReader.new(file)`
  - `r.read_line() !string`, `r.read_byte() !u8`, `r.read_all() ![]u8`
- **Console Standard Input**:
  - `io.prompt(msg string) !string`: Prints message and reads input line.
  - `io.read_line() !string`: Reads single line from standard input.
  - `io.read_all_stdin() !string`: Slurps all standard input until EOF.
- **Filesystem Utilities**: `io.rename(old, new) !bool`, `io.remove(path) !bool`, `io.temp_file() !string`.

---

## 🛠️ Usage Example

```v
module main

import io

fn main() {
    // 1. Interactive prompt
    name := io.prompt('Please enter your username: ') or { 'Guest' }
    println('Welcome, ${name}!')

    // 2. File creation and buffered writing
    mut f := io.File.create('notes.txt') or { panic(err) }
    f.write_string('Line 1: Vira standard I/O\n') or { panic(err) }
    f.close() or { false }
    defer io.remove('notes.txt') or { false }

    // 3. Buffered reading
    mut rf := io.File.open('notes.txt', 'r') or { panic(err) }
    mut reader := io.BufReader.new(rf)
    line := reader.read_line() or { '' }
    println('Read: ${line}')
    reader.close() or { false }
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/io
```
