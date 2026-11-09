#ifndef VIRA_RT_STDLIB_H
#define VIRA_RT_STDLIB_H

// ============================================================================
// 8. C STANDARD LIBRARY SHIMS (STDIO, CTYPE, MATH, TIME)
// ============================================================================

/**
 * Reads a single line of text from standard input up to a newline character.
 * @return Result containing string read from stdin or EOF error.
 * Usage: Interactive CLI input (`io::read_line()`).
 */
static inline Result_string vira_io_read_line(void) {
  size_t cap = 128;
  size_t len = 0;
  char *buf = (char *)malloc(cap);
  if (!buf) {
    return (Result_string){.is_err = true, .err = vira_str_lit("out of memory reading stdin", 27)};
  }
  int c;
  while ((c = fgetc(stdin)) != EOF) {
    if (c == '\r') continue;
    if (c == '\n') break;
    if (len + 1 >= cap) {
      cap *= 2;
      char *nb = (char *)realloc(buf, cap);
      if (!nb) {
        free(buf);
        return (Result_string){.is_err = true, .err = vira_str_lit("out of memory reading line", 26)};
      }
      buf = nb;
    }
    buf[len++] = (char)c;
  }
  if (len == 0 && c == EOF) {
    free(buf);
    return (Result_string){.is_err = true, .err = vira_str_lit("EOF", 3)};
  }
  buf[len] = '\0';
  _vira_record_alloc();
  return (Result_string){.is_err = false, .val = (vira_string){.data = (const uint8_t *)buf, .len = (int32_t)len, .is_lit = false}};
}

/**
 * Reads all available text from standard input until EOF into a single `vira_string`.
 * @return Result containing complete stdin text or error message.
 * Usage: Unix pipeline filter reading (`io::read_all_stdin()`).
 */
static inline Result_string vira_io_read_all_stdin(void) {
  size_t cap = 1024;
  size_t len = 0;
  char *buf = (char *)malloc(cap);
  if (!buf) {
    return (Result_string){.is_err = true, .err = vira_str_lit("out of memory", 13)};
  }
  char chunk[4096];
  size_t n;
  while ((n = fread(chunk, 1, sizeof(chunk), stdin)) > 0) {
    if (len + n + 1 > cap) {
      while (len + n + 1 > cap) cap *= 2;
      char *nb = (char *)realloc(buf, cap);
      if (!nb) {
        free(buf);
        return (Result_string){.is_err = true, .err = vira_str_lit("out of memory", 13)};
      }
      buf = nb;
    }
    memcpy(buf + len, chunk, n);
    len += n;
  }
  buf[len] = '\0';
  _vira_record_alloc();
  return (Result_string){.is_err = false, .val = (vira_string){.data = (const uint8_t *)buf, .len = (int32_t)len, .is_lit = false}};
}

/**
 * Renames or moves a file or directory from `old_p` to `new_p`.
 * @param old_p Existing filesystem path.
 * @param new_p New destination path.
 * @return Result with true on success or error message.
 * Usage: Atomic file replacement and directory moving (`io::rename(old, new)`).
 */
static inline Result_bool vira_io_rename(vira_string old_p, vira_string new_p) {
  char s_old[1024], s_new[1024];
  vira_cstr_buf b_old = vira_str_to_cstr_buf(old_p, s_old, sizeof(s_old));
  vira_cstr_buf b_new = vira_str_to_cstr_buf(new_p, s_new, sizeof(s_new));
  int res = rename(b_old.ptr, b_new.ptr);
  vira_cstr_buf_free(&b_old);
  vira_cstr_buf_free(&b_new);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("rename error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Deletes a file or directory path from the filesystem.
 * @param path Path to entry to delete.
 * @return Result with true on success or error message.
 * Usage: Filesystem cleanup (`io::remove(path)`).
 */
static inline Result_bool vira_io_remove(vira_string path) {
  char s_p[1024];
  vira_cstr_buf b = vira_str_to_cstr_buf(path, s_p, sizeof(s_p));
  int res = remove(b.ptr);
  vira_cstr_buf_free(&b);
  if (res != 0) {
    return (Result_bool){.is_err = true, .err = vira_str_format("remove error: %s", strerror(errno))};
  }
  return (Result_bool){.is_err = false, .val = true};
}

/**
 * Generates and creates a unique temporary file path on disk.
 * @return Result containing temporary file path string or error message.
 * Usage: Scratch file allocation for compilation or testing (`io::temp_file()`).
 */
static inline Result_string vira_io_temp_file(void) {
#ifdef _WIN32
  char tmp_dir[MAX_PATH];
  GetTempPathA(MAX_PATH, tmp_dir);
  char tmp_file[MAX_PATH];
  if (GetTempFileNameA(tmp_dir, "vir", 0, tmp_file) == 0) {
    return (Result_string){.is_err = true, .err = vira_str_lit("failed to create temp file", 26)};
  }
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(tmp_file)};
#else
  char pattern[] = "/tmp/vira_tmp_XXXXXX";
  int fd = mkstemp(pattern);
  if (fd < 0) {
    return (Result_string){.is_err = true, .err = vira_str_format("mkstemp error: %s", strerror(errno))};
  }
  close(fd);
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(pattern)};
#endif
}

/**
 * Tests whether a character is an alphabetic letter (A-Z or a-z).
 * @param c Character code.
 * @return True if alphabetic, false otherwise.
 * Usage: Text classification (`ctype::is_alpha(c)`).
 */
static inline bool vira_ctype_is_alpha(int32_t c) { return isalpha((unsigned char)c) != 0; }
/**
 * Tests whether a character is a numeric decimal digit (0-9).
 * @param c Character code.
 * @return True if digit, false otherwise.
 * Usage: Text classification (`ctype::is_digit(c)`).
 */
static inline bool vira_ctype_is_digit(int32_t c) { return isdigit((unsigned char)c) != 0; }
/**
 * Tests whether a character is alphanumeric (letter or digit).
 * @param c Character code.
 * @return True if alphanumeric, false otherwise.
 * Usage: Identifier and token validation (`ctype::is_alnum(c)`).
 */
static inline bool vira_ctype_is_alnum(int32_t c) { return isalnum((unsigned char)c) != 0; }
/**
 * Tests whether a character is a whitespace character (space, tab, newline, return).
 * @param c Character code.
 * @return True if whitespace, false otherwise.
 * Usage: Token delimiter detection (`ctype::is_space(c)`).
 */
static inline bool vira_ctype_is_space(int32_t c) { return isspace((unsigned char)c) != 0; }
/**
 * Tests whether a character is an uppercase ASCII letter.
 * @param c Character code.
 * @return True if uppercase, false otherwise.
 * Usage: Case inspection (`ctype::is_upper(c)`).
 */
static inline bool vira_ctype_is_upper(int32_t c) { return isupper((unsigned char)c) != 0; }
/**
 * Tests whether a character is a lowercase ASCII letter.
 * @param c Character code.
 * @return True if lowercase, false otherwise.
 * Usage: Case inspection (`ctype::is_lower(c)`).
 */
static inline bool vira_ctype_is_lower(int32_t c) { return islower((unsigned char)c) != 0; }
/**
 * Tests whether a character is a valid hexadecimal digit (0-9, a-f, A-F).
 * @param c Character code.
 * @return True if hex digit, false otherwise.
 * Usage: Hex parsing and validation (`ctype::is_xdigit(c)`).
 */
static inline bool vira_ctype_is_xdigit(int32_t c) { return isxdigit((unsigned char)c) != 0; }
/**
 * Tests whether a character is a printable ASCII character (including space).
 * @param c Character code.
 * @return True if printable, false otherwise.
 * Usage: Text sanitization (`ctype::is_print(c)`).
 */
static inline bool vira_ctype_is_print(int32_t c) { return isprint((unsigned char)c) != 0; }
/**
 * Tests whether a character is a punctuation symbol.
 * @param c Character code.
 * @return True if punctuation, false otherwise.
 * Usage: Text tokenization (`ctype::is_punct(c)`).
 */
static inline bool vira_ctype_is_punct(int32_t c) { return ispunct((unsigned char)c) != 0; }
/**
 * Tests whether a character is an ASCII control character.
 * @param c Character code.
 * @return True if control character, false otherwise.
 * Usage: Protocol parsing and sanitization (`ctype::is_cntrl(c)`).
 */
static inline bool vira_ctype_is_cntrl(int32_t c) { return iscntrl((unsigned char)c) != 0; }
/**
 * Converts an ASCII character code to its lowercase equivalent.
 * @param c Character code.
 * @return Lowercase character code.
 * Usage: Case folding (`ctype::to_lower(c)`).
 */
static inline int32_t vira_ctype_to_lower(int32_t c) { return tolower((unsigned char)c); }
/**
 * Converts an ASCII character code to its uppercase equivalent.
 * @param c Character code.
 * @return Uppercase character code.
 * Usage: Case folding (`ctype::to_upper(c)`).
 */
static inline int32_t vira_ctype_to_upper(int32_t c) { return toupper((unsigned char)c); }

/**
 * Computes the sine of an angle in radians.
 * @param x Angle in radians.
 * @return Sine value.
 * Usage: Trigonometric calculation (`math::sin(x)`).
 */
static inline double vira_math_sin(double x) { return sin(x); }
/**
 * Computes the cosine of an angle in radians.
 * @param x Angle in radians.
 * @return Cosine value.
 * Usage: Trigonometric calculation (`math::cos(x)`).
 */
static inline double vira_math_cos(double x) { return cos(x); }
/**
 * Computes the tangent of an angle in radians.
 * @param x Angle in radians.
 * @return Tangent value.
 * Usage: Trigonometric calculation (`math::tan(x)`).
 */
static inline double vira_math_tan(double x) { return tan(x); }
/**
 * Computes the arc sine (inverse sine) in radians.
 * @param x Value between -1.0 and 1.0.
 * @return Angle in radians.
 * Usage: Inverse trigonometry (`math::asin(x)`).
 */
static inline double vira_math_asin(double x) { return asin(x); }
/**
 * Computes the arc cosine (inverse cosine) in radians.
 * @param x Value between -1.0 and 1.0.
 * @return Angle in radians.
 * Usage: Inverse trigonometry (`math::acos(x)`).
 */
static inline double vira_math_acos(double x) { return acos(x); }
/**
 * Computes the arc tangent (inverse tangent) in radians.
 * @param x Value.
 * @return Angle in radians.
 * Usage: Inverse trigonometry (`math::atan(x)`).
 */
static inline double vira_math_atan(double x) { return atan(x); }
/**
 * Computes the four-quadrant arc tangent of y and x in radians.
 * @param y Y coordinate.
 * @param x X coordinate.
 * @return Angle in radians in [-pi, pi].
 * Usage: Direction vector angle calculation (`math::atan2(y, x)`).
 */
static inline double vira_math_atan2(double y, double x) { return atan2(y, x); }
/**
 * Computes the hyperbolic sine of x.
 * @param x Float value.
 * @return Hyperbolic sine.
 * Usage: Hyperbolic mathematics (`math::sinh(x)`).
 */
static inline double vira_math_sinh(double x) { return sinh(x); }
/**
 * Computes the hyperbolic cosine of x.
 * @param x Float value.
 * @return Hyperbolic cosine.
 * Usage: Hyperbolic mathematics (`math::cosh(x)`).
 */
static inline double vira_math_cosh(double x) { return cosh(x); }
/**
 * Computes the hyperbolic tangent of x.
 * @param x Float value.
 * @return Hyperbolic tangent in [-1.0, 1.0].
 * Usage: Hyperbolic activations and math (`math::tanh(x)`).
 */
static inline double vira_math_tanh(double x) { return tanh(x); }
/**
 * Computes the natural exponential `e^x`.
 * @param x Exponent.
 * @return Value of e raised to power x.
 * Usage: Exponential growth and probability calculations (`math::exp(x)`).
 */
static inline double vira_math_exp(double x) { return exp(x); }
/**
 * Computes base-2 exponential `2^x`.
 * @param x Exponent.
 * @return Value of 2 raised to power x.
 * Usage: Binary scale calculation (`math::exp2(x)`).
 */
static inline double vira_math_exp2(double x) { return exp2(x); }
/**
 * Computes the natural logarithm `ln(x)`.
 * @param x Value > 0.
 * @return Natural log.
 * Usage: Logarithmic calculation and loss functions (`math::log(x)`).
 */
static inline double vira_math_log(double x) { return log(x); }
/**
 * Computes the base-2 logarithm of x.
 * @param x Value > 0.
 * @return Base-2 logarithm.
 * Usage: Information theory and entropy calculations (`math::log2(x)`).
 */
static inline double vira_math_log2(double x) { return log2(x); }
/**
 * Computes the base-10 common logarithm of x.
 * @param x Value > 0.
 * @return Base-10 logarithm.
 * Usage: Decibels and magnitude calculations (`math::log10(x)`).
 */
static inline double vira_math_log10(double x) { return log10(x); }
/**
 * Computes `base` raised to the power of `exp` (`base^exp`).
 * @param x Base value.
 * @param y Exponent value.
 * @return Result of power operation.
 * Usage: General exponentiation (`math::pow(x, y)`).
 */
static inline double vira_math_pow(double x, double y) { return pow(x, y); }
/**
 * Computes the square root of x.
 * @param x Non-negative value.
 * @return Square root.
 * Usage: Euclidean distance and standard deviation (`math::sqrt(x)`).
 */
static inline double vira_math_sqrt(double x) { return sqrt(x); }
/**
 * Computes the cube root of x.
 * @param x Float value.
 * @return Cube root.
 * Usage: Geometric calculation (`math::cbrt(x)`).
 */
static inline double vira_math_cbrt(double x) { return cbrt(x); }
/**
 * Computes Euclidean hypotenuse `sqrt(x^2 + y^2)` avoiding intermediate overflow.
 * @param x First coordinate.
 * @param y Second coordinate.
 * @return Hypotenuse length.
 * Usage: 2D vector length calculation (`math::hypot(x, y)`).
 */
static inline double vira_math_hypot(double x, double y) { return hypot(x, y); }
/**
 * Computes the absolute value of a floating-point number.
 * @param x Float value.
 * @return Absolute value.
 * Usage: Float magnitude (`math::abs(x)`).
 */
static inline double vira_math_abs(double x) { return fabs(x); }
/**
 * Computes the absolute value of a 32-bit signed integer.
 * @param x 32-bit integer.
 * @return Absolute value.
 * Usage: Integer magnitude (`math::abs_int(x)`).
 */
static inline int32_t vira_math_abs_int(int32_t x) { return x < 0 ? -x : x; }
/**
 * Computes the absolute value of a 64-bit signed integer.
 * @param x 64-bit integer.
 * @return Absolute value.
 * Usage: Integer magnitude (`math::abs_i64(x)`).
 */
static inline int64_t vira_math_abs_i64(int64_t x) { return x < 0 ? -x : x; }
/**
 * Computes the largest integer less than or equal to x.
 * @param x Float value.
 * @return Floor value.
 * Usage: Rounding down (`math::floor(x)`).
 */
static inline double vira_math_floor(double x) { return floor(x); }
/**
 * Computes the smallest integer greater than or equal to x.
 * @param x Float value.
 * @return Ceiling value.
 * Usage: Rounding up (`math::ceil(x)`).
 */
static inline double vira_math_ceil(double x) { return ceil(x); }
/**
 * Rounds a floating-point number to the nearest integer.
 * @param x Float value.
 * @return Nearest integer float.
 * Usage: Nearest integer rounding (`math::round(x)`).
 */
static inline double vira_math_round(double x) { return round(x); }
/**
 * Truncates the fractional part of a floating-point number toward zero.
 * @param x Float value.
 * @return Truncated integer float.
 * Usage: Fractional stripping (`math::trunc(x)`).
 */
static inline double vira_math_trunc(double x) { return trunc(x); }
/**
 * Computes floating-point remainder of division `x / y`.
 * @param x Numerator.
 * @param y Denominator.
 * @return Remainder.
 * Usage: Modulo arithmetic on floats (`math::fmod(x, y)`).
 */
static inline double vira_math_fmod(double x, double y) { return fmod(x, y); }
/**
 * Checks whether a floating-point value is Not-a-Number (NaN).
 * @param x Float value.
 * @return True if NaN, false otherwise.
 * Usage: Numerical sanity checking (`math::is_nan(x)`).
 */
static inline bool vira_math_is_nan(double x) { return isnan(x) != 0; }
/**
 * Checks whether a floating-point value is positive or negative infinity.
 * @param x Float value.
 * @return True if infinite, false otherwise.
 * Usage: Overflow checking (`math::is_inf(x)`).
 */
static inline bool vira_math_is_inf(double x) { return isinf(x) != 0; }
/**
 * Seeds the pseudo-random number generator with the specified integer seed.
 * @param seed Unsigned integer seed.
 * Usage: Reproducible random sequence initialization (`math::srand(seed)`).
 */
static inline void vira_math_srand(uint32_t seed) { srand((unsigned int)seed); }
/**
 * Generates a pseudo-random integer uniformly distributed in range [min_val, max_val].
 * @param min_val Minimum bound (inclusive).
 * @param max_val Maximum bound (inclusive).
 * @return Random 32-bit integer.
 * Usage: Random integer generation (`math::rand_int(1, 100)`).
 */
static inline int32_t vira_math_rand_int(int32_t min_v, int32_t max_v) {
  if (min_v >= max_v) return min_v;
  return min_v + (rand() % (max_v - min_v + 1));
}
/**
 * Generates a pseudo-random 64-bit float uniformly distributed in range [0.0, 1.0).
 * @return Random float.
 * Usage: Random float generation (`math::rand_f64()`).
 */
static inline double vira_math_rand_f64(void) {
  return (double)rand() / (double)RAND_MAX;
}

/**
 * Internal comparator callback for sorting 32-bit integers with `qsort`.
 * @param a Pointer to first int32.
 * @param b Pointer to second int32.
 * @return Comparison result.
 * Usage: Internal sort helper.
 */
static int _cmp_vira_int(const void *a, const void *b) {
  int32_t ia = *(const int32_t *)a;
  int32_t ib = *(const int32_t *)b;
  return (ia > ib) - (ia < ib);
}
/**
 * Sorts a slice of 32-bit integers in ascending order in-place.
 * @param s Pointer to integer slice.
 * Usage: In-place integer slice sorting (`math::sort_ints(&slice)`).
 */
static inline void vira_math_sort_ints(vira_slice_int *s) {
  if (s != NULL && s->data != NULL && s->len > 1) {
    qsort(s->data, (size_t)s->len, sizeof(int32_t), _cmp_vira_int);
  }
}

/**
 * Internal comparator callback for sorting 64-bit floats with `qsort`.
 * @param a Pointer to first double.
 * @param b Pointer to second double.
 * @return Comparison result.
 * Usage: Internal sort helper.
 */
static int _cmp_vira_f64(const void *a, const void *b) {
  double fa = *(const double *)a;
  double fb = *(const double *)b;
  return (fa > fb) - (fa < fb);
}
/**
 * Sorts a slice of 64-bit floats in ascending order in-place.
 * @param s Pointer to float slice.
 * Usage: In-place float slice sorting (`math::sort_f64s(&slice)`).
 */
static inline void vira_math_sort_f64s(vira_slice_f64 *s) {
  if (s != NULL && s->data != NULL && s->len > 1) {
    qsort(s->data, (size_t)s->len, sizeof(double), _cmp_vira_f64);
  }
}

/**
 * Internal comparator callback for sorting `vira_string` elements with `qsort`.
 * @param a Pointer to first vira_string.
 * @param b Pointer to second vira_string.
 * @return Comparison result.
 * Usage: Internal sort helper.
 */
static int _cmp_vira_str(const void *a, const void *b) {
  const vira_string *sa = (const vira_string *)a;
  const vira_string *sb = (const vira_string *)b;
  int32_t min_len = sa->len < sb->len ? sa->len : sb->len;
  if (min_len > 0 && sa->data != NULL && sb->data != NULL) {
    int c = memcmp(sa->data, sb->data, (size_t)min_len);
    if (c != 0) return c;
  }
  return sa->len - sb->len;
}
/**
 * Sorts a slice of `vira_string` elements in lexicographical ascending order in-place.
 * @param s Pointer to string slice.
 * Usage: In-place string slice sorting (`strings::sort(&slice)`).
 */
static inline void vira_strings_sort(vira_slice_string *s) {
  if (s != NULL && s->data != NULL && s->len > 1) {
    qsort(s->data, (size_t)s->len, sizeof(vira_string), _cmp_vira_str);
  }
}

/**
 * Returns current timestamp in seconds since the UNIX epoch (1970-01-01 00:00:00 UTC).
 * @return Epoch seconds.
 * Usage: Standard calendar timestamps (`time::now_unix()`).
 */
static inline int64_t vira_time_now_unix(void) {
  return (int64_t)time(NULL);
}

/**
 * Returns current real-time clock timestamp in nanoseconds since the UNIX epoch.
 * @return Nanoseconds since epoch.
 * Usage: Ultra high-resolution timestamps (`time::now_ns()`).
 */
static inline int64_t vira_time_now_ns(void) {
#if defined(_WIN32)
  static LARGE_INTEGER freq;
  static int init = 0;
  if (!init) {
    QueryPerformanceFrequency(&freq);
    init = 1;
  }
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  return (int64_t)((counter.QuadPart * 1000000000LL) / freq.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
#endif
}

/**
 * Returns nanoseconds elapsed on a high-precision monotonic clock unaffected by system clock adjustments.
 * @return Monotonic nanoseconds.
 * Usage: High-precision execution profiling and elapsed interval measurement (`time::monotonic_ns()`).
 */
static inline int64_t vira_time_monotonic_ns(void) {
#if defined(_WIN32)
  static LARGE_INTEGER freq;
  static int init = 0;
  if (!init) {
    QueryPerformanceFrequency(&freq);
    init = 1;
  }
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  return (int64_t)((counter.QuadPart * 1000000000LL) / freq.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
#endif
}

/**
 * Formats a UNIX timestamp into a human-readable date/time string according to a `strftime` format pattern.
 * @param fmt Format string (e.g. "%Y-%m-%d %H:%M:%S").
 * @param unix_sec Seconds since epoch.
 * @param utc True for UTC time, false for local timezone.
 * @return Result containing formatted date string or error.
 * Usage: Date formatting (`time::strftime("%Y-%m-%d", ts, false)`).
 */
static inline Result_string vira_time_strftime(vira_string fmt, int64_t unix_sec, bool utc) {
  char stack_fmt[256];
  vira_cstr_buf fb = vira_str_to_cstr_buf(fmt, stack_fmt, sizeof(stack_fmt));
  time_t t = (time_t)unix_sec;
  struct tm tm_buf;
#ifdef _WIN32
  if (utc) gmtime_s(&tm_buf, &t);
  else localtime_s(&tm_buf, &t);
#else
  if (utc) gmtime_r(&t, &tm_buf);
  else localtime_r(&t, &tm_buf);
#endif
  char out_buf[256];
  size_t n = strftime(out_buf, sizeof(out_buf), fb.ptr, &tm_buf);
  vira_cstr_buf_free(&fb);
  if (n == 0) {
    return (Result_string){.is_err = true, .err = vira_str_lit("strftime formatting failed", 26)};
  }
  return (Result_string){.is_err = false, .val = vira_str_from_cstr(out_buf)};
}

#endif // VIRA_RT_STDLIB_H
