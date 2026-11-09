#ifndef VIRA_RT_JSON_H
#define VIRA_RT_JSON_H

// ============================================================================
// 13. DEPTH- & ESCAPE-AWARE JSON FIELD HELPERS (SEE ALSO `import json`)
// ============================================================================

/**
 * Fast scanner that advances past a JSON string literal respecting escape characters.
 * @param s Character buffer.
 * @param len Total buffer length.
 * @param open_quote_idx Index of opening double quote.
 * @return Index of closing quote, or len if unterminated.
 * Usage: Internal helper for high-throughput zero-copy JSON parsing.
 */
static inline size_t _vira_json_skip_string(const char *s, size_t len,
                                            size_t open_quote_idx) {
  size_t i = open_quote_idx + 1;
  while (i < len) {
    if (s[i] == '\\') {
      i += 2;
      continue;
    }
    if (s[i] == '"') {
      return i + 1;
    }
    i++;
  }
  return len;
}

/**
 * Checks whether the JSON key starting at open_quote_idx matches the target key string.
 * @param s JSON character buffer.
 * @param len Total buffer length.
 * @param open_quote_idx Offset of key's opening quote.
 * @param key Target key to compare against.
 * @return True if key matches exactly, false otherwise.
 * Usage: Fast key matching in zero-copy JSON scanner.
 */
static inline bool _vira_json_key_matches(const char *s, size_t len,
                                          size_t open_quote_idx,
                                          vira_string key) {
  size_t i = open_quote_idx + 1;
  int32_t ki = 0;
  while (i < len && s[i] != '"') {
    char c = s[i];
    if (c == '\\') {
      i++;
      if (i >= len)
        return false;
      switch (s[i]) {
      case '"':
        c = '"';
        break;
      case '\\':
        c = '\\';
        break;
      case '/':
        c = '/';
        break;
      case 'n':
        c = '\n';
        break;
      case 'r':
        c = '\r';
        break;
      case 't':
        c = '\t';
        break;
      default:
        c = s[i];
        break;
      }
    }
    if (ki >= key.len || (uint8_t)c != key.data[ki]) {
      return false;
    }
    ki++;
    i++;
  }
  return (i < len && s[i] == '"' && ki == key.len);
}

// Finds the value start offset for `key`. Prefers top-level object (`target_depth == 1`);
// falls back to nested object match when `target_depth == 0`.
/**
 * Scans a JSON string and locates the value pointer corresponding to key at the specified nesting depth.
 * @param json Input JSON string.
 * @param key Target property key.
 * @param target_depth Target object nesting depth (1 for root object).
 * @return Pointer to start of value token, or NULL if not found.
 * Usage: Nested JSON property search.
 */
static inline const char *_vira_json_find_val_at_depth(vira_string json,
                                                       vira_string key,
                                                       int32_t target_depth) {
  if (json.data == NULL || json.len <= 0 || key.data == NULL || key.len <= 0) {
    return NULL;
  }
  const char *s = (const char *)json.data;
  size_t len = (size_t)json.len;
  int32_t obj_depth = 0;
  int32_t arr_depth = 0;
  size_t i = 0;

  while (i < len) {
    char c = s[i];
    if (c == '{') {
      obj_depth++;
      i++;
    } else if (c == '}') {
      if (obj_depth > 0)
        obj_depth--;
      i++;
    } else if (c == '[') {
      arr_depth++;
      i++;
    } else if (c == ']') {
      if (arr_depth > 0)
        arr_depth--;
      i++;
    } else if (c == '"') {
      size_t after_str = _vira_json_skip_string(s, len, i);
      size_t j = after_str;
      while (j < len &&
             (s[j] == ' ' || s[j] == '\t' || s[j] == '\r' || s[j] == '\n')) {
        j++;
      }
      bool is_obj_key = (j < len && s[j] == ':' && obj_depth >= 1);
      bool depth_ok = (target_depth == 0) ||
                      (obj_depth == target_depth && arr_depth == 0);
      if (is_obj_key && depth_ok && _vira_json_key_matches(s, len, i, key)) {
        j++; // skip ':'
        while (j < len &&
               (s[j] == ' ' || s[j] == '\t' || s[j] == '\r' || s[j] == '\n')) {
          j++;
        }
        return (j < len) ? (s + j) : NULL;
      }
      i = after_str;
    } else {
      i++;
    }
  }
  return NULL;
}

/**
 * Locates the value token pointer for a top-level property key in a JSON string.
 * @param json Input JSON string.
 * @param key Property key string.
 * @return Pointer to value token, or NULL if not found.
 * Usage: Top-level JSON property lookup.
 */
static inline const char *_vira_json_find_val(vira_string json,
                                              vira_string key) {
  const char *v = _vira_json_find_val_at_depth(json, key, 1);
  if (v != NULL)
    return v;
  return _vira_json_find_val_at_depth(json, key, 0);
}

/**
 * Extracts and unescapes a string property value from a JSON string by key.
 * @param json Input JSON string.
 * @param key Target property key.
 * @return Extracted string, or empty string if not found.
 * Usage: JSON string property reading (`json::get_str(json, "name")`).
 */
static inline vira_string vira_json_get_str(vira_string json, vira_string key) {
  const char *val = _vira_json_find_val(json, key);
  if (!val || *val != '"')
    return vira_str_lit("", 0);
  const char *end = (const char *)json.data + json.len;
  const char *p = val + 1;

  size_t max_len = (size_t)(end - p);
  char *out = (char *)malloc(max_len + 1);
  if (!out)
    return vira_str_lit("", 0);

  int32_t out_len = 0;
  bool closed = false;
  while (p < end) {
    char c = *p++;
    if (c == '"') {
      closed = true;
      break;
    }
    if (c == '\\' && p < end) {
      char esc = *p++;
      switch (esc) {
      case '"':
        out[out_len++] = '"';
        break;
      case '\\':
        out[out_len++] = '\\';
        break;
      case '/':
        out[out_len++] = '/';
        break;
      case 'b':
        out[out_len++] = '\b';
        break;
      case 'f':
        out[out_len++] = '\f';
        break;
      case 'n':
        out[out_len++] = '\n';
        break;
      case 'r':
        out[out_len++] = '\r';
        break;
      case 't':
        out[out_len++] = '\t';
        break;
      default:
        out[out_len++] = esc;
        break;
      }
    } else {
      out[out_len++] = c;
    }
  }
  if (!closed) {
    free(out);
    return vira_str_lit("", 0);
  }
  out[out_len] = '\0';
  _vira_record_alloc();
  return (vira_string){
      .data = (const uint8_t *)out, .len = out_len, .is_lit = false};
}

/**
 * Extracts an integer property value from a JSON string by key.
 * @param json Input JSON string.
 * @param key Target property key.
 * @return Parsed 64-bit integer, or 0 if not found.
 * Usage: JSON integer property reading (`json::get_int(json, "count")`).
 */
static inline int64_t vira_json_get_int(vira_string json, vira_string key) {
  const char *val = _vira_json_find_val(json, key);
  if (!val)
    return 0;
  const char *end = (const char *)json.data + json.len;
  char num_buf[64];
  int nlen = 0;
  while (val + nlen < end && nlen < 63) {
    char c = val[nlen];
    if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
      num_buf[nlen++] = c;
    } else {
      break;
    }
  }
  if (nlen == 0)
    return 0;
  num_buf[nlen] = '\0';
  return (int64_t)strtoll(num_buf, NULL, 10);
}

/**
 * Extracts a floating-point property value from a JSON string by key.
 * @param json Input JSON string.
 * @param key Target property key.
 * @return Parsed float value, or 0.0f if not found.
 * Usage: JSON float property reading (`json::get_f32(json, "rate")`).
 */
static inline float vira_json_get_f32(vira_string json, vira_string key) {
  const char *val = _vira_json_find_val(json, key);
  if (!val)
    return 0.0f;
  const char *end = (const char *)json.data + json.len;
  char num_buf[64];
  int nlen = 0;
  while (val + nlen < end && nlen < 63) {
    char c = val[nlen];
    if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' ||
        c == 'e' || c == 'E') {
      num_buf[nlen++] = c;
    } else {
      break;
    }
  }
  if (nlen == 0)
    return 0.0f;
  num_buf[nlen] = '\0';
  return (float)strtod(num_buf, NULL);
}

#endif // VIRA_RT_JSON_H
