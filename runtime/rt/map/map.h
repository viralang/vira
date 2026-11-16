#ifndef VIRA_RT_MAP_H
#define VIRA_RT_MAP_H

// ============================================================================
// 1.6. VIRA HASH MAP RUNTIME (`map[K]V`)
// ============================================================================
//
// One untyped container serves every map type. Keys are either strings or
// integers (int, i64, u8, bool and enums are widened to int64_t). Values are
// stored inline as raw bytes; the compiler generates a `vira_map_ops` table per
// value type with its size and its clone / free functions, so a map owns its
// keys and values exactly like a slice owns its elements.
//
// Entries are kept in insertion order (iteration is deterministic) in a dense
// array; an open-addressing index of entry positions gives O(1) lookups.
// Deleting marks an entry dead; dead entries are compacted when the index is
// rebuilt. A zero-initialised `vira_map` is a valid empty map.

typedef struct {
  size_t val_size;
  bool key_is_str;
  void (*clone_val)(void *dst, const void *src); // NULL: plain memcpy
  void (*free_val)(void *val);                   // NULL: nothing to free
} vira_map_ops;

typedef struct {
  uint64_t hash;
  int64_t ikey;
  vira_string skey;
  bool live;
} vira_map_entry_hdr;

typedef struct vira_map {
  const vira_map_ops *ops;
  char *entries;     // n_entries * stride bytes
  int32_t *index;    // index_cap slots: entry position + 1, or 0 for empty
  int32_t len;       // live entries
  int32_t n_entries; // used entry slots (live + dead)
  int32_t cap_entries;
  int32_t index_cap;
} vira_map;

static inline size_t _vira_map_hdr_size(void) {
  return (sizeof(vira_map_entry_hdr) + 15) & ~(size_t)15;
}

static inline size_t _vira_map_stride(const vira_map_ops *ops) {
  return _vira_map_hdr_size() + ((ops->val_size + 15) & ~(size_t)15);
}

static inline vira_map_entry_hdr *_vira_map_hdr(const vira_map *m, int32_t i) {
  return (vira_map_entry_hdr *)(m->entries + (size_t)i * _vira_map_stride(m->ops));
}

static inline void *_vira_map_val(const vira_map *m, int32_t i) {
  return (char *)_vira_map_hdr(m, i) + _vira_map_hdr_size();
}

static inline uint64_t _vira_map_hash(bool is_str, int64_t ikey, vira_string skey) {
  uint64_t h = 1469598103934665603ULL; // FNV-1a
  if (is_str) {
    for (int32_t i = 0; i < skey.len; i++) {
      h ^= (uint64_t)skey.data[i];
      h *= 1099511628211ULL;
    }
  } else {
    uint64_t x = (uint64_t)ikey;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    h = x;
  }
  return h ? h : 1;
}

static inline bool _vira_map_key_eq(const vira_map *m, const vira_map_entry_hdr *e, uint64_t h,
                                    int64_t ikey, vira_string skey) {
  if (!e->live || e->hash != h)
    return false;
  return m->ops->key_is_str ? vira_str_eq(e->skey, skey) : e->ikey == ikey;
}

// _vira_map_find returns the entry position of a key, or -1.
static inline int32_t _vira_map_find(const vira_map *m, int64_t ikey, vira_string skey) {
  if (m == NULL || m->len == 0 || m->index_cap == 0)
    return -1;
  uint64_t h = _vira_map_hash(m->ops->key_is_str, ikey, skey);
  uint32_t mask = (uint32_t)m->index_cap - 1;
  for (uint32_t probe = (uint32_t)h & mask, n = 0; n < (uint32_t)m->index_cap; probe = (probe + 1) & mask, n++) {
    int32_t slot = m->index[probe];
    if (slot == 0)
      return -1;
    if (_vira_map_key_eq(m, _vira_map_hdr(m, slot - 1), h, ikey, skey))
      return slot - 1;
  }
  return -1;
}

// _vira_map_rebuild compacts dead entries and rebuilds the index with room to grow.
static inline void _vira_map_rebuild(vira_map *m, int32_t want_entries) {
  size_t stride = _vira_map_stride(m->ops);
  int32_t w = 0;
  for (int32_t r = 0; r < m->n_entries; r++) {
    vira_map_entry_hdr *e = _vira_map_hdr(m, r);
    if (!e->live)
      continue;
    if (w != r)
      memmove(m->entries + (size_t)w * stride, e, stride);
    w++;
  }
  m->n_entries = w;
  if (want_entries < w + 1)
    want_entries = w + 1;
  if (want_entries > m->cap_entries) {
    int32_t cap = m->cap_entries ? m->cap_entries : 8;
    while (cap < want_entries)
      cap *= 2;
    char *buf = (char *)realloc(m->entries, (size_t)cap * stride);
    if (!buf) {
      fprintf(stderr, "vira panic: out of memory growing map\n");
      exit(1);
    }
    if (m->entries == NULL)
      _vira_record_alloc();
    m->entries = buf;
    m->cap_entries = cap;
  }
  int32_t icap = 16;
  while (icap < m->cap_entries * 2)
    icap *= 2;
  if (icap != m->index_cap) {
    if (m->index == NULL)
      _vira_record_alloc();
    free(m->index);
    m->index = (int32_t *)malloc((size_t)icap * sizeof(int32_t));
    if (!m->index) {
      fprintf(stderr, "vira panic: out of memory growing map index\n");
      exit(1);
    }
    m->index_cap = icap;
  }
  memset(m->index, 0, (size_t)m->index_cap * sizeof(int32_t));
  uint32_t mask = (uint32_t)m->index_cap - 1;
  for (int32_t i = 0; i < m->n_entries; i++) {
    uint32_t probe = (uint32_t)_vira_map_hdr(m, i)->hash & mask;
    while (m->index[probe] != 0)
      probe = (probe + 1) & mask;
    m->index[probe] = i + 1;
  }
}

/** Returns a pointer to the value stored under a key, or NULL. */
static inline void *vira_map_get(const vira_map *m, int64_t ikey, vira_string skey) {
  int32_t i = _vira_map_find(m, ikey, skey);
  return i < 0 ? NULL : _vira_map_val(m, i);
}

/** `m[k]` read: the stored value, or `zero` when the key is missing. */
static inline void *vira_map_get_or(const vira_map *m, int64_t ikey, vira_string skey, void *zero) {
  void *p = vira_map_get(m, ikey, skey);
  return p ? p : zero;
}

static inline bool vira_map_has(const vira_map *m, int64_t ikey, vira_string skey) {
  return _vira_map_find(m, ikey, skey) >= 0;
}

/**
 * `m[k] = v` target: returns the value slot for a key, inserting a zeroed value (and an owned
 * copy of a string key) when the key is new.
 */
static inline void *vira_map_slot(vira_map *m, const vira_map_ops *ops, int64_t ikey, vira_string skey) {
  if (m->ops == NULL)
    m->ops = ops;
  int32_t found = _vira_map_find(m, ikey, skey);
  if (found >= 0)
    return _vira_map_val(m, found);
  if (m->n_entries + 1 > m->cap_entries || (m->n_entries + 1) * 2 > m->index_cap)
    _vira_map_rebuild(m, m->n_entries + 1);
  int32_t i = m->n_entries++;
  vira_map_entry_hdr *e = _vira_map_hdr(m, i);
  memset(e, 0, _vira_map_stride(m->ops));
  e->hash = _vira_map_hash(m->ops->key_is_str, ikey, skey);
  e->live = true;
  if (m->ops->key_is_str)
    e->skey = vira_str_clone(skey);
  else
    e->ikey = ikey;
  uint32_t mask = (uint32_t)m->index_cap - 1;
  uint32_t probe = (uint32_t)e->hash & mask;
  while (m->index[probe] != 0)
    probe = (probe + 1) & mask;
  m->index[probe] = i + 1;
  m->len++;
  return _vira_map_val(m, i);
}

static inline void _vira_map_release_entry(vira_map *m, vira_map_entry_hdr *e) {
  if (m->ops->key_is_str)
    vira_str_free(&e->skey);
  if (m->ops->free_val)
    m->ops->free_val((char *)e + _vira_map_hdr_size());
  e->live = false;
}

/** Removes a key; returns whether it was present. */
static inline bool vira_map_delete(vira_map *m, int64_t ikey, vira_string skey) {
  int32_t i = _vira_map_find(m, ikey, skey);
  if (i < 0)
    return false;
  _vira_map_release_entry(m, _vira_map_hdr(m, i));
  m->len--;
  // The index keeps pointing at the dead entry; lookups skip it until the next rebuild.
  return true;
}

/** Iteration: the next live entry position >= i, or -1. */
static inline int32_t vira_map_next(const vira_map *m, int32_t i) {
  for (; i < m->n_entries; i++) {
    if (_vira_map_hdr(m, i)->live)
      return i;
  }
  return -1;
}

static inline int64_t vira_map_entry_ikey(const vira_map *m, int32_t i) {
  return _vira_map_hdr(m, i)->ikey;
}

static inline vira_string vira_map_entry_skey(const vira_map *m, int32_t i) {
  vira_string k = _vira_map_hdr(m, i)->skey;
  k.is_lit = true; // a borrowed view: the map keeps owning the key
  k.is_borrowed = true;
  return k;
}

static inline void *vira_map_entry_val(const vira_map *m, int32_t i) {
  return _vira_map_val(m, i);
}

/** Removes every entry, keeping the allocated storage. */
static inline void vira_map_clear(vira_map *m) {
  if (m->ops == NULL)
    return;
  for (int32_t i = 0; i < m->n_entries; i++) {
    vira_map_entry_hdr *e = _vira_map_hdr(m, i);
    if (e->live)
      _vira_map_release_entry(m, e);
  }
  m->n_entries = 0;
  m->len = 0;
  if (m->index)
    memset(m->index, 0, (size_t)m->index_cap * sizeof(int32_t));
}

static inline void vira_map_free(vira_map *m) {
  if (m == NULL || m->ops == NULL)
    return;
  vira_map_clear(m);
  if (m->entries) {
    free(m->entries);
    _vira_record_free();
  }
  if (m->index) {
    free(m->index);
    _vira_record_free();
  }
  memset(m, 0, sizeof(*m));
}

/** Deep copy: keys and values are cloned, insertion order is kept. */
static inline vira_map vira_map_clone(const vira_map *src) {
  vira_map m;
  memset(&m, 0, sizeof(m));
  if (src == NULL || src->ops == NULL || src->len == 0) {
    if (src)
      m.ops = src->ops;
    return m;
  }
  m.ops = src->ops;
  for (int32_t i = vira_map_next(src, 0); i >= 0; i = vira_map_next(src, i + 1)) {
    vira_map_entry_hdr *e = _vira_map_hdr(src, i);
    void *dst = vira_map_slot(&m, src->ops, e->ikey, e->skey);
    void *val = _vira_map_val(src, i);
    if (src->ops->clone_val)
      src->ops->clone_val(dst, val);
    else
      memcpy(dst, val, src->ops->val_size);
  }
  return m;
}

#endif // VIRA_RT_MAP_H
