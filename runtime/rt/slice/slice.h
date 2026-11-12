#ifndef VIRA_RT_SLICE_H
#define VIRA_RT_SLICE_H

// ============================================================================
// 1.5. VIRA DYNAMIC SLICE RUNTIME (`[]T`) & FIXED ARRAYS (`[N]T`)
// ============================================================================

/**
 * Validates that an index falls within [0, len), panicking on out-of-bounds access.
 * @param index Requested element index.
 * @param len Total length of slice or array.
 * @param file Source file name for error reporting.
 * @param line Source line number for error reporting.
 * Usage: Injected by the compiler before array/slice indexing when bounds checking is enabled.
 */
static inline void vira_bounds_check(int32_t index, int32_t len,
                                     const char *file, int line) {
  if (index < 0 || index >= len) {
    // Supervised inside actors (trapped like `panic(...)`); fatal elsewhere.
    _vira_rt_fault("%s:%d: vira panic: index out of bounds: index %d with len %d",
                   file, line, index, len);
  }
}

/**
 * Reallocates a slice backing buffer using geometric 1.5x capacity growth.
 * @param data_ptr Pointer to slice buffer pointer.
 * @param cap_ptr Pointer to slice capacity counter.
 * @param elem_size Size of single element in bytes.
 * @param is_arena True if slice memory resides in a bump arena (reallocation disallowed).
 * Usage: Internal helper invoked during `push` operations when slice is full.
 */
static inline void _vira_slice_grow(void **data_ptr, int32_t *cap_ptr,
                                    size_t elem_size, bool is_arena) {
  int32_t old_cap = *cap_ptr;
  int32_t new_cap = (old_cap <= 0) ? 4 : (old_cap * 2);
  void *new_buf;
  if (old_cap == 0 || *data_ptr == NULL) {
    new_buf = malloc((size_t)new_cap * elem_size);
    if (new_buf)
      _vira_record_alloc();
  } else {
    new_buf = realloc(*data_ptr, (size_t)new_cap * elem_size);
  }
  if (!new_buf) {
    fprintf(stderr, "vira panic: out of memory growing slice buffer\n");
    exit(1);
  }
  *data_ptr = new_buf;
  *cap_ptr = new_cap;
}

/**
 * Releases the backing memory of a slice and resets its length and capacity to 0.
 * @param data_ptr Pointer to buffer pointer.
 * @param len_ptr Pointer to length counter.
 * @param cap_ptr Pointer to capacity counter.
 * @param is_arena True if slice memory is arena-managed (no-op free).
 * Usage: Internal helper shared by all typed slice deallocation functions.
 */
static inline void _vira_slice_free_raw(void **data_ptr, int32_t *len_ptr,
                                        int32_t *cap_ptr, bool is_arena) {
  if (data_ptr != NULL && *data_ptr != NULL && !is_arena) {
    free(*data_ptr);
    *data_ptr = NULL;
    _vira_record_free();
  }
  if (len_ptr)
    *len_ptr = 0;
  if (cap_ptr)
    *cap_ptr = 0;
}

/**
 * Instantiates the typed dynamic slice structure and standard operations for type `TYPE_NAME`.
 * Generated methods for `vira_slice_##TYPE_NAME`:
 *  - `vira_slice_##TYPE_NAME##_new(int32_t initial_cap)`: Allocates slice with initial capacity.
 *  - `vira_slice_##TYPE_NAME##_push(s, val)`: Appends an element, growing with 1.5x geometric factor.
 *  - `vira_slice_##TYPE_NAME##_pop(s)`: Removes and returns the trailing element.
 *  - `vira_slice_##TYPE_NAME##_set(s, idx, val)`: Assigns element at index with bounds checking.
 *  - `vira_slice_##TYPE_NAME##_clear(s)`: Sets length to 0 while keeping capacity allocated.
 *  - `vira_slice_##TYPE_NAME##_is_empty(s)`: Returns true if length == 0.
 *  - `vira_slice_##TYPE_NAME##_clone(s)`: Creates an independent copy of the slice and elements.
 *  - `vira_slice_##TYPE_NAME##_subslice(s, start, end)`: Extracts subslice range into a new slice.
 *  - `vira_slice_##TYPE_NAME##_contains(s, val)`: Linear search for matching element value.
 *  - `vira_slice_##TYPE_NAME##_free(s)`: Deallocates heap backing memory and resets fields to 0.
 */
#define VIRA_DEFINE_SLICE_TYPE(TYPE_NAME, C_TYPE)                              \
  typedef struct {                                                             \
    C_TYPE *data;                                                              \
    int32_t len;                                                               \
    int32_t cap;                                                               \
    bool is_arena;                                                             \
  } vira_slice_##TYPE_NAME;                                                    \
                                                                               \
  static inline vira_slice_##TYPE_NAME vira_slice_##TYPE_NAME##_new(           \
      int32_t initial_cap) {                                                   \
    vira_slice_##TYPE_NAME s = {                                               \
        .data = NULL, .len = 0, .cap = 0, .is_arena = false};                  \
    if (initial_cap > 0) {                                                     \
      s.data = (C_TYPE *)malloc((size_t)initial_cap * sizeof(C_TYPE));         \
      if (s.data) {                                                            \
        s.cap = initial_cap;                                                   \
        _vira_record_alloc();                                                  \
      }                                                                        \
    }                                                                          \
    return s;                                                                  \
  }                                                                            \
                                                                               \
  static inline void vira_slice_##TYPE_NAME##_push(vira_slice_##TYPE_NAME *s,  \
                                                   C_TYPE val) {               \
    if (s->len >= s->cap) {                                                    \
      _vira_slice_grow((void **)&s->data, &s->cap, sizeof(C_TYPE),             \
                       s->is_arena);                                           \
    }                                                                          \
    s->data[s->len++] = val;                                                   \
  }                                                                            \
                                                                               \
  static inline C_TYPE vira_slice_##TYPE_NAME##_pop(                           \
      vira_slice_##TYPE_NAME *s) {                                             \
    if (s->len <= 0) {                                                         \
      _vira_rt_fault("vira panic: pop from empty slice");                      \
    }                                                                          \
    return s->data[--s->len];                                                  \
  }                                                                            \
                                                                               \
  static inline void vira_slice_##TYPE_NAME##_set(vira_slice_##TYPE_NAME *s,   \
                                                  int32_t idx, C_TYPE val) {   \
    vira_bounds_check(idx, s->len, __FILE__, __LINE__);                        \
    s->data[idx] = val;                                                        \
  }                                                                            \
                                                                               \
  static inline void vira_slice_##TYPE_NAME##_clear(                           \
      vira_slice_##TYPE_NAME *s) {                                             \
    s->len = 0;                                                                \
  }                                                                            \
                                                                               \
  static inline bool vira_slice_##TYPE_NAME##_is_empty(                        \
      const vira_slice_##TYPE_NAME *s) {                                       \
    return s->len == 0;                                                        \
  }                                                                            \
                                                                               \
  static inline vira_slice_##TYPE_NAME vira_slice_##TYPE_NAME##_clone(         \
      const vira_slice_##TYPE_NAME *s) {                                       \
    vira_slice_##TYPE_NAME res = vira_slice_##TYPE_NAME##_new(s->len);         \
    if (s->len > 0 && s->data != NULL) {                                       \
      memcpy(res.data, s->data, (size_t)s->len * sizeof(C_TYPE));              \
      res.len = s->len;                                                        \
    }                                                                          \
    return res;                                                                \
  }                                                                            \
                                                                               \
  static inline vira_slice_##TYPE_NAME vira_slice_##TYPE_NAME##_subslice(      \
      const vira_slice_##TYPE_NAME *s, int32_t start, int32_t end) {          \
    if (start < 0) start = 0;                                                  \
    if (end > s->len) end = s->len;                                            \
    if (start > end) start = end;                                              \
    int32_t count = end - start;                                               \
    vira_slice_##TYPE_NAME res = vira_slice_##TYPE_NAME##_new(count);          \
    if (count > 0 && s->data != NULL) {                                        \
      memcpy(res.data, s->data + start, (size_t)count * sizeof(C_TYPE));       \
      res.len = count;                                                         \
    }                                                                          \
    return res;                                                                \
  }                                                                            \
                                                                               \
  static inline bool vira_slice_##TYPE_NAME##_contains(                        \
      const vira_slice_##TYPE_NAME *s, C_TYPE val) {                           \
    for (int32_t i = 0; i < s->len; i++) {                                     \
      if (s->data[i] == val)                                                   \
        return true;                                                           \
    }                                                                          \
    return false;                                                              \
  }                                                                            \
                                                                               \
  static inline void vira_slice_##TYPE_NAME##_free(                            \
      vira_slice_##TYPE_NAME *s) {                                             \
    _vira_slice_free_raw((void **)&s->data, &s->len, &s->cap, s->is_arena);    \
  }

#ifndef _vira_slice_u8_DEFINED
#define _vira_slice_u8_DEFINED
VIRA_DEFINE_SLICE_TYPE(u8, uint8_t)
#endif
#ifndef _vira_slice_int_DEFINED
#define _vira_slice_int_DEFINED
VIRA_DEFINE_SLICE_TYPE(int, int32_t)
#endif
#ifndef _vira_slice_i64_DEFINED
#define _vira_slice_i64_DEFINED
VIRA_DEFINE_SLICE_TYPE(i64, int64_t)
#endif
#ifndef _vira_slice_f32_DEFINED
#define _vira_slice_f32_DEFINED
VIRA_DEFINE_SLICE_TYPE(f32, vira_f32)
#endif
#ifndef _vira_slice_f64_DEFINED
#define _vira_slice_f64_DEFINED
VIRA_DEFINE_SLICE_TYPE(f64, double)
#endif
#ifndef _vira_slice_bool_DEFINED
#define _vira_slice_bool_DEFINED
VIRA_DEFINE_SLICE_TYPE(bool, bool)
#endif

/**
 * Returns a 1-character substring at the specified byte index.
 * @param s Source string.
 * @param idx Zero-based byte index.
 * @return Single character `vira_string`.
 * Usage: Used for character-by-character string inspection.
 */
static inline vira_string vira_str_char_at(vira_string s, int32_t idx) {
  vira_bounds_check(idx, s.len, __FILE__, __LINE__);
  // A view into `s`: borrowed unless `s` itself is a static literal.
  return (vira_string){.data = &s.data[idx], .len = 1, .is_lit = true,
                       .is_borrowed = !s.is_lit || s.is_borrowed};
}

#ifndef _vira_slice_string_DEFINED
#define _vira_slice_string_DEFINED
// Slice for vira_string
typedef struct {
  vira_string *data;
  int32_t len;
  int32_t cap;
  bool is_arena;
} vira_slice_string;
#endif

/**
 * Allocates a new dynamic slice of `vira_string` elements with initial capacity.
 * @param initial_cap Initial element capacity.
 * @return Freshly initialized `vira_slice_string`.
 * Usage: Constructs empty or pre-sized string slices (`[]string`).
 */
static inline vira_slice_string vira_slice_string_new(int32_t initial_cap) {
  vira_slice_string s = {.data = NULL, .len = 0, .cap = 0, .is_arena = false};
  if (initial_cap > 0) {
    s.data = (vira_string *)malloc((size_t)initial_cap * sizeof(vira_string));
    if (s.data) {
      s.cap = initial_cap;
      _vira_record_alloc();
    }
  }
  return s;
}

/**
 * Appends a string to the end of a string slice, expanding capacity if needed.
 * @param s Pointer to destination slice.
 * @param val String value to append.
 * Usage: Implements `slice.push(val)` in Vira.
 */
static inline void vira_slice_string_push(vira_slice_string *s,
                                          vira_string val) {
  if (s->len >= s->cap) {
    _vira_slice_grow((void **)&s->data, &s->cap, sizeof(vira_string),
                     s->is_arena);
  }
  s->data[s->len++] = val;
}

/**
 * Removes and returns the last string from a string slice, panicking if empty.
 * @param s Pointer to slice.
 * @return Removed string value.
 * Usage: Implements `slice.pop()` in Vira.
 */
static inline vira_string vira_slice_string_pop(vira_slice_string *s) {
  if (s->len <= 0) {
    _vira_rt_fault("vira panic: pop from empty slice");
  }
  return s->data[--s->len];
}

/**
 * Sets the string value at the specified index with bounds checking.
 * @param s Pointer to slice.
 * @param idx Zero-based index.
 * @param val New string value.
 * Usage: Implements index assignment `slice[idx] = val`.
 */
static inline void vira_slice_string_set(vira_slice_string *s, int32_t idx,
                                         vira_string val) {
  vira_bounds_check(idx, s->len, __FILE__, __LINE__);
  vira_str_free(&s->data[idx]);
  s->data[idx] = val;
}

/**
 * Resets the slice length to zero without freeing its backing buffer.
 * @param s Pointer to slice.
 * Usage: Fast reuse of allocated slice capacity (`slice.clear()`).
 */
static inline void vira_slice_string_clear(vira_slice_string *s) {
  for (int32_t i = 0; i < s->len; i++) {
    vira_str_free(&s->data[i]);
  }
  s->len = 0;
}

/**
 * Checks whether the string slice contains zero elements.
 * @param s Pointer to slice.
 * @return True if length is 0, false otherwise.
 * Usage: Slice emptiness query (`slice.is_empty()`).
 */
static inline bool vira_slice_string_is_empty(const vira_slice_string *s) {
  return s->len == 0;
}

/**
 * Performs linear search to determine if a string is present in the slice.
 * @param s Pointer to slice.
 * @param val String to search for.
 * @return True if match found, false otherwise.
 * Usage: Implements membership test (`slice.contains(val)`).
 */
static inline bool vira_slice_string_contains(const vira_slice_string *s,
                                              vira_string val) {
  for (int32_t i = 0; i < s->len; i++) {
    if (vira_str_eq(s->data[i], val))
      return true;
  }
  return false;
}

/**
 * Creates a deep clone of a string slice, cloning each contained dynamic string.
 * @param s Pointer to source slice.
 * @return Newly allocated independent copy of the slice.
 * Usage: Deep copy operations (`slice.clone()`).
 */
static inline vira_slice_string
vira_slice_string_clone(const vira_slice_string *s) {
  vira_slice_string res = vira_slice_string_new(s->len);
  for (int32_t i = 0; i < s->len; i++) {
    vira_slice_string_push(&res, vira_str_clone(s->data[i]));
  }
  return res;
}

/**
 * Extracts a subslice of strings from start to end index into a new slice.
 * @param s Pointer to source slice.
 * @param start Starting index (inclusive).
 * @param end Ending index (exclusive).
 * @return New string slice containing the range.
 * Usage: Implements slice range extraction (`slice[start..end]`).
 */
static inline vira_slice_string
vira_slice_string_subslice(const vira_slice_string *s, int32_t start, int32_t end) {
  if (start < 0) start = 0;
  if (end > s->len) end = s->len;
  if (start > end) start = end;
  int32_t count = end - start;
  vira_slice_string res = vira_slice_string_new(count);
  for (int32_t i = 0; i < count; i++) {
    vira_slice_string_push(&res, vira_str_clone(s->data[start + i]));
  }
  return res;
}

/**
 * Frees all dynamic strings contained in the slice and releases its backing buffer.
 * @param s Pointer to string slice to release.
 * Usage: Complete deep cleanup of string slices to guarantee zero memory leaks.
 */
static inline void vira_slice_string_free(vira_slice_string *s) {
  if (s->data != NULL && !s->is_arena) {
    for (int32_t i = 0; i < s->len; i++) {
      vira_str_free(&s->data[i]);
    }
    free(s->data);
    s->data = NULL;
    _vira_record_free();
  }
  s->len = 0;
  s->cap = 0;
}

#ifndef _vira_slice_vira_slice_string_DEFINED
#define _vira_slice_vira_slice_string_DEFINED
typedef struct {
  vira_slice_string *data;
  int32_t len;
  int32_t cap;
  bool is_arena;
} vira_slice_vira_slice_string;
#endif

/**
 * Allocates a 2D slice of string slices with initial capacity.
 * @param initial_cap Initial capacity.
 * @return Fresh 2D string slice (`[][]string`).
 * Usage: Used for matrices, tabular data, and CSV row collections.
 */
static inline vira_slice_vira_slice_string vira_slice_vira_slice_string_new(int32_t initial_cap) {
  vira_slice_vira_slice_string s = {.data = NULL, .len = 0, .cap = 0, .is_arena = false};
  if (initial_cap > 0) {
    s.data = (vira_slice_string *)malloc((size_t)initial_cap * sizeof(vira_slice_string));
    if (s.data) {
      s.cap = initial_cap;
      _vira_record_alloc();
    }
  }
  return s;
}

/**
 * Appends a string slice row to a 2D string slice.
 * @param s Pointer to 2D slice.
 * @param val String slice to append.
 * Usage: Appends rows to tables and multi-dimensional string lists.
 */
static inline void vira_slice_vira_slice_string_push(vira_slice_vira_slice_string *s, vira_slice_string val) {
  if (s->len >= s->cap) {
    _vira_slice_grow((void **)&s->data, &s->cap, sizeof(vira_slice_string), s->is_arena);
  }
  s->data[s->len++] = val;
}

/**
 * Removes and returns the last row from a 2D string slice.
 * @param s Pointer to 2D slice.
 * @return Removed string slice.
 * Usage: Pops the last row from a 2D table.
 */
static inline vira_slice_string vira_slice_vira_slice_string_pop(vira_slice_vira_slice_string *s) {
  if (s->len <= 0) {
    _vira_rt_fault("vira panic: pop from empty slice");
  }
  return s->data[--s->len];
}

/**
 * Sets the string slice row at the specified index with bounds checking.
 * @param s Pointer to 2D slice.
 * @param idx Row index.
 * @param val New string slice row.
 * Usage: Replaces a row in a 2D string slice.
 */
static inline void vira_slice_vira_slice_string_set(vira_slice_vira_slice_string *s, int32_t idx, vira_slice_string val) {
  vira_bounds_check(idx, s->len, __FILE__, __LINE__);
  vira_slice_string_free(&s->data[idx]);
  s->data[idx] = val;
}

/**
 * Resets 2D slice length to zero without freeing backing memory.
 * @param s Pointer to 2D slice.
 * Usage: Fast reuse of 2D slice capacity.
 */
static inline void vira_slice_vira_slice_string_clear(vira_slice_vira_slice_string *s) {
  for (int32_t i = 0; i < s->len; i++) {
    vira_slice_string_free(&s->data[i]);
  }
  s->len = 0;
}

/**
 * Checks whether the 2D string slice contains zero rows.
 * @param s Pointer to 2D slice.
 * @return True if length is 0, false otherwise.
 * Usage: Emptiness check on 2D table structures.
 */
static inline bool vira_slice_vira_slice_string_is_empty(const vira_slice_vira_slice_string *s) {
  return s->len == 0;
}

/**
 * Deep clones a 2D string slice and all its nested rows.
 * @param s Pointer to source 2D slice.
 * @return Deep cloned 2D string slice.
 * Usage: Independent copy of tables and 2D arrays.
 */
static inline vira_slice_vira_slice_string vira_slice_vira_slice_string_clone(const vira_slice_vira_slice_string *s) {
  vira_slice_vira_slice_string res = vira_slice_vira_slice_string_new(s->len);
  for (int32_t i = 0; i < s->len; i++) {
    vira_slice_string clone_item = vira_slice_string_clone(&s->data[i]);
    vira_slice_vira_slice_string_push(&res, clone_item);
  }
  return res;
}

/**
 * Extracts a subset of rows from a 2D string slice.
 * @param s Pointer to source 2D slice.
 * @param start Start row index.
 * @param end End row index.
 * @return Subslice of rows.
 * Usage: Row range extraction in tables.
 */
static inline vira_slice_vira_slice_string vira_slice_vira_slice_string_subslice(const vira_slice_vira_slice_string *s, int32_t start, int32_t end) {
  if (start < 0) start = 0;
  if (end > s->len) end = s->len;
  if (start > end) start = end;
  int32_t count = end - start;
  vira_slice_vira_slice_string res = vira_slice_vira_slice_string_new(count);
  for (int32_t i = 0; i < count; i++) {
    vira_slice_string clone_item = vira_slice_string_clone(&s->data[start + i]);
    vira_slice_vira_slice_string_push(&res, clone_item);
  }
  return res;
}

/**
 * Deep frees all contained string slices and releases the 2D slice buffer.
 * @param s Pointer to 2D slice.
 * Usage: Destructor ensuring zero leaks for multi-dimensional string structures.
 */
static inline void vira_slice_vira_slice_string_free(vira_slice_vira_slice_string *s) {
  if (s->data != NULL && !s->is_arena) {
    for (int32_t i = 0; i < s->len; i++) {
      vira_slice_string_free(&s->data[i]);
    }
    free(s->data);
    s->data = NULL;
    _vira_record_free();
  }
  s->len = 0;
  s->cap = 0;
}

/**
 * Returns command-line arguments passed to the program as a slice of strings.
 * @return Slice of argument strings (`[]string`).
 * Usage: Command-line parsing and flags processing (`os::args()`).
 */
static inline vira_slice_string vira_os_args(void) {
  vira_slice_string s = vira_slice_string_new(_vira_argc);
  for (int i = 0; i < _vira_argc; i++) {
    vira_slice_string_push(&s, vira_str_from_cstr(_vira_argv[i]));
  }
  return s;
}

/**
 * Returns the host operating system platform identifier ("windows", "darwin", "linux").
 * @return Platform name string.
 * Usage: Cross-platform conditional branching (`os::platform()`).
 */
static inline vira_string vira_os_platform(void) {
#if defined(_WIN32)
  return vira_str_lit("windows", 7);
#elif defined(__APPLE__)
  return vira_str_lit("darwin", 6);
#elif defined(__linux__)
  return vira_str_lit("linux", 5);
#else
  return vira_str_lit("unknown", 7);
#endif
}

#ifndef _Result_u8_DEFINED
#define _Result_u8_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  uint8_t val;
} Result_u8;
#endif

#ifndef _Result_f32_DEFINED
#define _Result_f32_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  vira_f32 val;
} Result_f32;
#endif

#ifndef _Result_f64_DEFINED
#define _Result_f64_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  double val;
} Result_f64;
#endif

#ifndef _Result_int_DEFINED
#define _Result_int_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  int32_t val;
} Result_int;
#endif

#ifndef _Result_i64_DEFINED
#define _Result_i64_DEFINED
typedef struct {
  bool is_err;
  vira_string err;
  int64_t val;
} Result_i64;
#endif

/**
 * Checks if a string contains a specified substring.
 * @param s Main string.
 * @param sub Substring to search for.
 * @return True if found, false otherwise.
 * Usage: Substring test (`str.contains("abc")`).
 */
static inline bool vira_str_contains(vira_string s, vira_string sub) {
  if (sub.len == 0)
    return true;
  if (s.len < sub.len || s.data == NULL || sub.data == NULL)
    return false;
  for (int32_t i = 0; i <= s.len - sub.len; i++) {
    if (memcmp(&s.data[i], sub.data, (size_t)sub.len) == 0) {
      return true;
    }
  }
  return false;
}

/**
 * Returns the zero-based index of the first occurrence of a substring, or -1 if not found.
 * @param s Main string.
 * @param sub Substring to search for.
 * @return First match index, or -1.
 * Usage: Substring location search (`str.index_of("abc")`).
 */
static inline int32_t vira_str_index_of(vira_string s, vira_string sub) {
  if (sub.len == 0)
    return 0;
  if (s.len < sub.len || s.data == NULL || sub.data == NULL)
    return -1;
  for (int32_t i = 0; i <= s.len - sub.len; i++) {
    if (memcmp(&s.data[i], sub.data, (size_t)sub.len) == 0) {
      return i;
    }
  }
  return -1;
}

/**
 * Checks if a string begins with the specified prefix.
 * @param s Main string.
 * @param prefix Prefix string.
 * @return True if string begins with prefix, false otherwise.
 * Usage: Prefix testing (`str.starts_with("https://")`).
 */
static inline bool vira_str_starts_with(vira_string s, vira_string prefix) {
  if (prefix.len == 0)
    return true;
  if (s.len < prefix.len || s.data == NULL || prefix.data == NULL)
    return false;
  return memcmp(s.data, prefix.data, (size_t)prefix.len) == 0;
}

/**
 * Checks if a string ends with the specified suffix.
 * @param s Main string.
 * @param suffix Suffix string.
 * @return True if string ends with suffix, false otherwise.
 * Usage: Extension and suffix testing (`str.ends_with(".vir")`).
 */
static inline bool vira_str_ends_with(vira_string s, vira_string suffix) {
  if (suffix.len == 0)
    return true;
  if (s.len < suffix.len || s.data == NULL || suffix.data == NULL)
    return false;
  return memcmp(&s.data[s.len - suffix.len], suffix.data, (size_t)suffix.len) ==
         0;
}

/**
 * Extracts a substring slice between start and end indices.
 * @param s Source string.
 * @param start Start byte offset (inclusive).
 * @param end End byte offset (exclusive).
 * @return Substring slice.
 * Usage: String slicing operations (`str.substr(0, 5)`).
 */
static inline vira_string vira_str_substr(vira_string s, int32_t start,
                                          int32_t end) {
  if (start < 0)
    start = 0;
  if (end > s.len)
    end = s.len;
  if (start >= end || s.data == NULL) {
    return vira_str_lit("", 0);
  }
  int32_t len = end - start;
  uint8_t *buf = (uint8_t *)malloc((size_t)len + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in vira_str_substr\n");
    exit(1);
  }
  memcpy(buf, &s.data[start], (size_t)len);
  buf[len] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = buf, .len = len, .is_lit = false};
}

/**
 * Trims leading and trailing whitespace characters from a string.
 * @param s Source string.
 * @return Trimmed string view.
 * Usage: Sanitizes user input and file lines (`str.trim()`).
 */
static inline vira_string vira_str_trim(vira_string s) {
  if (s.len == 0 || s.data == NULL)
    return vira_str_lit("", 0);
  int32_t start = 0;
  while (start < s.len && (s.data[start] == ' ' || s.data[start] == '\t' ||
                           s.data[start] == '\r' || s.data[start] == '\n')) {
    start++;
  }
  int32_t end = s.len;
  while (end > start && (s.data[end - 1] == ' ' || s.data[end - 1] == '\t' ||
                         s.data[end - 1] == '\r' || s.data[end - 1] == '\n')) {
    end--;
  }
  return vira_str_substr(s, start, end);
}

/**
 * Converts all ASCII letters in a string to lowercase.
 * @param s Source string.
 * @return Newly allocated lowercase string.
 * Usage: Case-insensitive comparisons and normalization (`str.to_lower()`).
 */
static inline vira_string vira_str_to_lower(vira_string s) {
  if (s.len <= 0 || s.data == NULL)
    return vira_str_lit("", 0);
  size_t sz = (size_t)s.len;
  uint8_t *buf = (uint8_t *)malloc(sz + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in vira_str_to_lower\n");
    exit(1);
  }
  for (size_t i = 0; i < sz; i++) {
    uint8_t c = s.data[i];
    if (c >= 'A' && c <= 'Z')
      c = (uint8_t)(c + ('a' - 'A'));
    buf[i] = c;
  }
  buf[sz] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = buf, .len = (int32_t)sz, .is_lit = false};
}

/**
 * Converts all ASCII letters in a string to uppercase.
 * @param s Source string.
 * @return Newly allocated uppercase string.
 * Usage: String normalization (`str.to_upper()`).
 */
static inline vira_string vira_str_to_upper(vira_string s) {
  if (s.len <= 0 || s.data == NULL)
    return vira_str_lit("", 0);
  size_t sz = (size_t)s.len;
  uint8_t *buf = (uint8_t *)malloc(sz + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in vira_str_to_upper\n");
    exit(1);
  }
  for (size_t i = 0; i < sz; i++) {
    uint8_t c = s.data[i];
    if (c >= 'a' && c <= 'z')
      c = (uint8_t)(c - ('a' - 'A'));
    buf[i] = c;
  }
  buf[sz] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = buf, .len = (int32_t)sz, .is_lit = false};
}

/**
 * Parses a decimal string into a signed 32-bit integer result.
 * @param s Source string.
 * @return Result containing parsed int32 or error message.
 * Usage: Numeric input parsing (`str.to_int()`).
 */
static inline Result_int vira_str_to_int(vira_string s) {
  if (s.len == 0 || s.data == NULL) {
    return (Result_int){.is_err = true,
                        .err = vira_str_lit("empty string", 12)};
  }
  char stack_buf[128];
  vira_cstr_buf b = vira_str_to_cstr_buf(s, stack_buf, sizeof(stack_buf));
  char *endptr = NULL;
  long val = strtol(b.ptr, &endptr, 10);
  bool valid = (endptr != b.ptr && *endptr == '\0');
  vira_cstr_buf_free(&b);
  if (!valid) {
    return (Result_int){.is_err = true,
                        .err = vira_str_lit("invalid integer format", 22)};
  }
  return (Result_int){.is_err = false, .val = (int32_t)val};
}

/**
 * Parses a decimal string into a signed 64-bit integer result.
 * @param s Source string.
 * @return Result containing parsed int64 or error message.
 * Usage: Large integer parsing (`str.to_i64()`).
 */
static inline Result_i64 vira_str_to_i64(vira_string s) {
  if (s.len == 0 || s.data == NULL) {
    return (Result_i64){.is_err = true,
                        .err = vira_str_lit("empty string", 12)};
  }
  char stack_buf[128];
  vira_cstr_buf b = vira_str_to_cstr_buf(s, stack_buf, sizeof(stack_buf));
  char *endptr = NULL;
  long long val = strtoll(b.ptr, &endptr, 10);
  bool valid = (endptr != b.ptr && *endptr == '\0');
  vira_cstr_buf_free(&b);
  if (!valid) {
    return (Result_i64){.is_err = true,
                        .err = vira_str_lit("invalid integer format", 22)};
  }
  return (Result_i64){.is_err = false, .val = (int64_t)val};
}

/**
 * Parses a floating-point string into a 64-bit float result.
 * @param s Source string.
 * @return Result containing parsed double or error message.
 * Usage: Float input parsing (`str.to_f64()`).
 */
static inline Result_f64 vira_str_to_f64(vira_string s) {
  if (s.len == 0 || s.data == NULL) {
    return (Result_f64){.is_err = true, .err = vira_str_lit("empty string", 12)};
  }
  char stack_buf[128];
  vira_cstr_buf b = vira_str_to_cstr_buf(s, stack_buf, sizeof(stack_buf));
  char *endptr = NULL;
  double val = strtod(b.ptr, &endptr);
  bool valid = (endptr != b.ptr && *endptr == '\0');
  vira_cstr_buf_free(&b);
  if (!valid) {
    return (Result_f64){.is_err = true, .err = vira_str_lit("invalid float format", 20)};
  }
  return (Result_f64){.is_err = false, .val = val};
}

/**
 * Parses a floating-point string into a 32-bit float result.
 * @param s Source string.
 * @return Result containing parsed float or error message.
 * Usage: Float input parsing (`str.to_f32()`).
 */
static inline Result_f32 vira_str_to_f32(vira_string s) {
  Result_f64 r = vira_str_to_f64(s);
  if (r.is_err) {
    return (Result_f32){.is_err = true, .err = r.err};
  }
  return (Result_f32){.is_err = false, .val = (vira_f32)r.val};
}

/**
 * Parses an integer string in the specified radix base (2, 8, 10, or 16).
 * @param s Source string.
 * @param base Radix base between 2 and 36.
 * @return Result containing parsed int64 or error message.
 * Usage: Hex, octal, and binary string parsing (`str::parse_i64_base(hex_str, 16)`).
 */
static inline Result_i64 vira_str_parse_i64_base(vira_string s, int32_t base) {
  if (s.len == 0 || s.data == NULL || base < 2 || base > 36) {
    return (Result_i64){.is_err = true, .err = vira_str_lit("invalid base or empty string", 28)};
  }
  char stack_buf[128];
  vira_cstr_buf b = vira_str_to_cstr_buf(s, stack_buf, sizeof(stack_buf));
  char *endptr = NULL;
  errno = 0;
  long long val = strtoll(b.ptr, &endptr, base);
  bool valid = (endptr != b.ptr && *endptr == '\0' && errno == 0);
  vira_cstr_buf_free(&b);
  if (!valid) {
    return (Result_i64){.is_err = true, .err = vira_str_lit("invalid integer format for base", 31)};
  }
  return (Result_i64){.is_err = false, .val = (int64_t)val};
}

/**
 * Formats a 64-bit integer into a string representation in the specified radix base.
 * @param v Value to format.
 * @param base Radix base between 2 and 36.
 * @return Formatted string.
 * Usage: Hex and binary string generation (`str::format_i64_base(val, 16)`).
 */
static inline vira_string vira_str_format_i64_base(int64_t v, int32_t base) {
  if (base < 2 || base > 36) base = 10;
  if (base == 10) {
    return vira_str_format("%lld", (long long)v);
  }
  if (base == 16) {
    return vira_str_format("%llx", (unsigned long long)v);
  }
  if (base == 8) {
    return vira_str_format("%llo", (unsigned long long)v);
  }
  char buf[70];
  int idx = 69;
  buf[idx--] = '\0';
  bool neg = (v < 0 && base == 10);
  uint64_t uv = neg ? (uint64_t)(-v) : (uint64_t)v;
  if (uv == 0) {
    buf[idx--] = '0';
  } else {
    const char *digits = "0123456789abcdefghijklmnopqrstuvwxyz";
    while (uv > 0) {
      buf[idx--] = digits[uv % (uint64_t)base];
      uv /= (uint64_t)base;
    }
    if (neg) {
      buf[idx--] = '-';
    }
  }
  return vira_str_from_cstr(&buf[idx + 1]);
}

/**
 * Retrieves the raw unsigned byte value at the specified index as a result.
 * @param s Source string.
 * @param idx Zero-based byte index.
 * @return Result containing byte or out-of-bounds error.
 * Usage: Low-level byte inspection of strings (`str.byte_at(idx)`).
 */
static inline Result_u8 vira_str_byte_at(vira_string s, int32_t idx) {
  if (idx < 0 || idx >= s.len || s.data == NULL) {
    return (Result_u8){.is_err = true, .err = vira_str_lit("index out of bounds", 19)};
  }
  return (Result_u8){.is_err = false, .val = (uint8_t)s.data[idx]};
}

/**
 * Returns the zero-based index of the last occurrence of a substring, or -1 if not found.
 * @param s Main string.
 * @param sub Substring to locate.
 * @return Last match index, or -1.
 * Usage: Locates file extensions or trailing delimiters (`str.last_index_of(".")`).
 */
static inline int32_t vira_str_last_index_of(vira_string s, vira_string sub) {
  if (sub.len == 0) return s.len;
  if (s.len < sub.len || s.data == NULL || sub.data == NULL) return -1;
  for (int32_t i = s.len - sub.len; i >= 0; i--) {
    if (memcmp(s.data + i, sub.data, (size_t)sub.len) == 0) {
      return i;
    }
  }
  return -1;
}

/**
 * Trims leading whitespace characters from a string.
 * @param s Source string.
 * @return Left-trimmed string view.
 * Usage: Strips indentation or leading spaces (`str.trim_left()`).
 */
static inline vira_string vira_str_trim_left(vira_string s) {
  if (s.len == 0 || s.data == NULL) return s;
  int32_t start = 0;
  while (start < s.len && (s.data[start] == ' ' || s.data[start] == '\t' || s.data[start] == '\n' || s.data[start] == '\r')) {
    start++;
  }
  return (vira_string){.data = s.data + start, .len = s.len - start, .is_lit = true,
                       .is_borrowed = !s.is_lit || s.is_borrowed};
}

/**
 * Trims trailing whitespace characters from a string.
 * @param s Source string.
 * @return Right-trimmed string view.
 * Usage: Strips trailing spaces and newlines (`str.trim_right()`).
 */
static inline vira_string vira_str_trim_right(vira_string s) {
  if (s.len == 0 || s.data == NULL) return s;
  int32_t end = s.len;
  while (end > 0 && (s.data[end - 1] == ' ' || s.data[end - 1] == '\t' || s.data[end - 1] == '\n' || s.data[end - 1] == '\r')) {
    end--;
  }
  return (vira_string){.data = s.data, .len = end, .is_lit = true,
                       .is_borrowed = !s.is_lit || s.is_borrowed};
}

/**
 * Constructs a new string by repeating a source string `count` times.
 * @param s Source string.
 * @param count Number of repetitions.
 * @return Newly allocated repeated string.
 * Usage: Generating banners, padding, or indentation (`str.repeat("=", 40)`).
 */
static inline vira_string vira_str_repeat(vira_string s, int32_t count) {
  if (count <= 0 || s.len == 0 || s.data == NULL) {
    return vira_str_lit("", 0);
  }
  size_t total = (size_t)s.len * (size_t)count;
  char *buf = (char *)malloc(total + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in str.repeat\n");
    exit(1);
  }
  for (int32_t i = 0; i < count; i++) {
    memcpy(buf + ((size_t)i * (size_t)s.len), s.data, (size_t)s.len);
  }
  buf[total] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = (const uint8_t *)buf, .len = (int32_t)total, .is_lit = false};
}

/**
 * Returns a new string with all occurrences of `old_sub` replaced by `new_sub`.
 * @param s Source string.
 * @param old_sub Substring to replace.
 * @param new_sub Replacement string.
 * @return Newly allocated modified string.
 * Usage: Text transformation and sanitization (`str.replace("foo", "bar")`).
 */
static inline vira_string vira_str_replace(vira_string s, vira_string old_sub, vira_string new_sub) {
  if (s.len == 0 || s.data == NULL || old_sub.len == 0 || old_sub.data == NULL) {
    return vira_str_clone(s);
  }
  int32_t count = 0;
  int32_t idx = 0;
  while (idx <= s.len - old_sub.len) {
    if (memcmp(s.data + idx, old_sub.data, (size_t)old_sub.len) == 0) {
      count++;
      idx += old_sub.len;
    } else {
      idx++;
    }
  }
  if (count == 0) {
    return vira_str_clone(s);
  }
  size_t new_len = (size_t)s.len + (size_t)count * (size_t)(new_sub.len - old_sub.len);
  char *buf = (char *)malloc(new_len + 1);
  if (!buf) {
    fprintf(stderr, "vira panic: out of memory in str.replace\n");
    exit(1);
  }
  size_t dst = 0;
  idx = 0;
  while (idx < s.len) {
    if (idx <= s.len - old_sub.len && memcmp(s.data + idx, old_sub.data, (size_t)old_sub.len) == 0) {
      if (new_sub.len > 0) {
        memcpy(buf + dst, new_sub.data, (size_t)new_sub.len);
        dst += (size_t)new_sub.len;
      }
      idx += old_sub.len;
    } else {
      buf[dst++] = s.data[idx++];
    }
  }
  buf[new_len] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = (const uint8_t *)buf, .len = (int32_t)new_len, .is_lit = false};
}

/**
 * Splits a string into lines by newline delimiters (`
` or `
`).
 * @param s Source string.
 * @return Slice of line strings (`[]string`).
 * Usage: Processing line-oriented text files (`str.lines()`).
 */
static inline vira_slice_string vira_str_lines(vira_string s) {
  vira_slice_string res = vira_slice_string_new(8);
  if (s.len == 0 || s.data == NULL) {
    return res;
  }
  int32_t start = 0;
  for (int32_t i = 0; i < s.len; i++) {
    if (s.data[i] == '\n') {
      int32_t line_end = i;
      if (line_end > start && s.data[line_end - 1] == '\r') {
        line_end--;
      }
      vira_slice_string_push(&res, vira_str_substr(s, start, line_end));
      start = i + 1;
    }
  }
  if (start <= s.len) {
    int32_t line_end = s.len;
    if (line_end > start && s.data[line_end - 1] == '\r') {
      line_end--;
    }
    vira_slice_string_push(&res, vira_str_substr(s, start, line_end));
  }
  return res;
}

/**
 * Splits a string into parts delimited by a separator substring.
 * @param s Source string.
 * @param sep Delimiter substring.
 * @return Slice of string tokens.
 * Usage: Tokenizing CSV, path segments, and query parameters (`str.split(",")`).
 */
static inline vira_slice_string vira_str_split(vira_string s, vira_string sep) {
  vira_slice_string res = vira_slice_string_new(4);
  if (s.len == 0 || s.data == NULL) {
    vira_slice_string_push(&res, vira_str_lit("", 0));
    return res;
  }
  if (sep.len == 0) {
    for (int32_t i = 0; i < s.len; i++) {
      vira_slice_string_push(&res, vira_str_substr(s, i, i + 1));
    }
    return res;
  }
  int32_t cur = 0;
  while (cur <= s.len) {
    int32_t next = -1;
    if (cur <= s.len - sep.len) {
      for (int32_t i = cur; i <= s.len - sep.len; i++) {
        if (memcmp(&s.data[i], sep.data, (size_t)sep.len) == 0) {
          next = i;
          break;
        }
      }
    }
    if (next == -1) {
      vira_slice_string_push(&res, vira_str_substr(s, cur, s.len));
      break;
    } else {
      vira_slice_string_push(&res, vira_str_substr(s, cur, next));
      cur = next + sep.len;
    }
  }
  return res;
}

/**
 * Converts a string into a dynamic slice of unsigned bytes.
 * @param s Source string.
 * @return Byte slice (`[]u8`).
 * Usage: Binary encoding and serialization (`str.to_bytes()`).
 */
static inline vira_slice_u8 vira_str_to_bytes(vira_string s) {
  vira_slice_u8 res = vira_slice_u8_new(s.len);
  if (s.len > 0 && s.data != NULL) {
    memcpy(res.data, s.data, (size_t)s.len);
    res.len = s.len;
  }
  return res;
}

/**
 * Constructs an owned `vira_string` from a slice of unsigned bytes.
 * @param s Pointer to byte slice.
 * @return Owned dynamic `vira_string`.
 * Usage: Deserialization and network decoding (`bytes.to_string()`).
 */
static inline vira_string vira_bytes_to_string(const vira_slice_u8 *s) {
  if (!s || s->len <= 0 || s->data == NULL) {
    return vira_str_lit("", 0);
  }
  uint8_t *copy = (uint8_t *)malloc((size_t)s->len + 1);
  if (!copy) {
    fprintf(stderr, "vira panic: out of memory in vira_bytes_to_string\n");
    exit(1);
  }
  memcpy(copy, s->data, (size_t)s->len);
  copy[s->len] = '\0';
  _vira_record_alloc();
  return (vira_string){.data = copy, .len = s->len, .is_lit = false};
}

#endif // VIRA_RT_SLICE_H
