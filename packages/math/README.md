# `packages/math`: High-Performance Math Intrinsics

The `math` package provides C99 floating-point operations, mathematical constants, rounding, root-finding, and pseudorandom number generation in the **Vira Programming Language**.

---

## 🚀 Key Features

- **Constants**: `math.PI`, `math.TAU`, `math.E`, `math.SQRT2`.
- **Trigonometric Functions**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`.
- **Hyperbolics**: `sinh`, `cosh`, `tanh`.
- **Exponential & Logarithmic**: `exp`, `exp2`, `log`, `log2`, `log10`.
- **Powers & Roots**: `pow`, `sqrt`, `cbrt`, `hypot`.
- **Rounding & Modulo**: `ceil`, `floor`, `trunc`, `round`, `fmod`.
- **Utility Helpers**: `abs`, `abs_int`, `abs_i64`, `min`, `max`, `clamp`, `is_nan`, `is_inf`.
- **RNG**: `srand(seed)`, `rand_int(min, max)`, `rand_f64()`.

---

## 🛠️ Usage Example

```v
module main

import math

fn main() {
    // 1. Hypotenuse calculation
    h := math.hypot(3.0, 4.0)
    println('Hypotenuse (3, 4): ${h}') // 5.0

    // 2. Trigonometry
    angle_rad := 45.0 * (math.PI / 180.0)
    println('sin(45 deg): ${math.sin(angle_rad)}')

    // 3. Logarithms and powers
    println('2^10: ${math.pow(2.0, 10.0)}') // 1024.0
    println('log10(1000): ${math.log10(1000.0)}') // 3.0

    // 4. Clamping
    val := math.clamp(150.0, 0.0, 100.0)
    println('Clamped: ${val}') // 100.0
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/math
```
