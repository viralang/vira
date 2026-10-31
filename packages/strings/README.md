# `packages/strings`: Pure-Vira String Utilities

The `strings` package provides high-performance string manipulation, whitespace tokenization, delimiter splitting, padding, arbitrary-base parsing, sorting, and dynamic string building in the **Vira Programming Language**.

---

## 🚀 Key Features

- **Tokenization & Joining**:
  - `strings.fields(s string) []string`: Splits string by arbitrary whitespace (Go `strings.Fields` equivalent).
  - `strings.join(parts []string, sep string) string`: Concatenates slices with delimiter.
  - `strings.split(s string, delim string) []string`: Splits by exact substring.
- **Prefixes & Suffixes**:
  - `strings.has_prefix(s, prefix)`, `strings.has_suffix(s, suffix)`.
  - `strings.trim_prefix(s, prefix)`, `strings.trim_suffix(s, suffix)`.
- **Formatting & Padding**: `strings.pad_left`, `strings.pad_right`, `strings.count`, `strings.compare`.
- **Arbitrary Base Conversions**:
  - `strings.parse_i64_base(s, base) !i64`: Parses binary (`base 2`), octal (`base 8`), hex (`base 16`), or custom bases.
  - `strings.format_i64_base(val, base) string`: Formats integers in arbitrary radix.
- **Alphabetical Sorting**: `strings.sort(mut s []string)`.
- **`StringBuilder`**: Mutable buffer for assembling strings with zero intermediate reallocations (`write`, `write_line`, `to_string`, `clear`, `len`).

---

## 🛠️ Usage Example

```v
module main

import strings

fn main() {
    // 1. Whitespace fields splitting
    line := '  container   run   --detach   --name=app  '
    tokens := strings.fields(line)
    println('Parsed ${tokens.len} tokens:')
    for t in tokens {
        println('  token: ${t}')
    }

    // 2. Joining
    csv := strings.join(tokens, ',')
    println('Joined CSV: ${csv}')

    // 3. Number formatting and parsing across radices
    hex_str := strings.format_i64_base(255i64, 16)
    println('255 in hex: 0x${hex_str}') // 0xff

    val := strings.parse_i64_base('ff', 16) or { 0i64 }
    println('Parsed hex: ${val}') // 255

    // 4. StringBuilder
    mut sb := strings.StringBuilder.new()
    sb.write('HTTP/1.1 200 OK')
    sb.write_line('')
    sb.write('Content-Length: 0')
    println(sb.to_string())
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/strings
```
