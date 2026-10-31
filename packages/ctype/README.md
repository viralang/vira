# `packages/ctype`: Fast ASCII Character Classification

The `ctype` package provides zero-overhead character classification and case conversion functions in the **Vira Programming Language**.

---

## 🚀 Key Features

- **Classifiers**:
  - `ctype.is_alpha(c u8) bool`
  - `ctype.is_digit(c u8) bool`
  - `ctype.is_alnum(c u8) bool`
  - `ctype.is_space(c u8) bool`
  - `ctype.is_upper(c u8) bool`
  - `ctype.is_lower(c u8) bool`
  - `ctype.is_hex_digit(c u8) bool`
  - `ctype.is_print(c u8) bool`
  - `ctype.is_punct(c u8) bool`
  - `ctype.is_cntrl(c u8) bool`
- **Converters**:
  - `ctype.to_lower(c u8) u8`
  - `ctype.to_upper(c u8) u8`

---

## 🛠️ Usage Example

```v
module main

import ctype

fn main() {
    c := u8(65) // ASCII 'A'

    if ctype.is_alpha(c) {
        println('Character is alphabetic')
    }

    if ctype.is_upper(c) {
        lower := ctype.to_lower(c)
        println('Converted to lowercase: ${int(lower)}') // 97 ('a')
    }
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/ctype
```
