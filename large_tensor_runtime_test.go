package main_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// T-010 regression: tensors with more than 2^31 elements (full DeepSeek-V3 / Kimi-K2 /
// DeepSeek-V4-Pro expert tensors). Checks numel64, the shape guard, a full-size sum and
// argmax, and that the GGUF loader reports an axis above 2^31-1 instead of loading it
// as size 1. The 8.6 GB of data is an uncommitted MAP_NORESERVE mapping that is only
// read, so peak memory stays small. POSIX only.
func TestRuntimeLargeTensorElementCounts(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses mmap(MAP_NORESERVE); POSIX only")
	}
	if strings.HasSuffix(runtime.GOARCH, "32") || runtime.GOARCH == "386" || runtime.GOARCH == "arm" {
		t.Skip("needs a 64-bit address space")
	}
	var cCompiler string
	for _, cand := range []string{"gcc", "clang", "cc"} {
		if _, err := exec.LookPath(cand); err == nil {
			cCompiler = cand
			break
		}
	}
	if cCompiler == "" {
		t.Skip("Host C compiler not found in PATH; skipping large tensor test")
	}
	runtimeDir, err := filepath.Abs("runtime")
	if err != nil {
		t.Fatalf("failed to resolve runtime dir: %v", err)
	}

	cCode := `/* T-010 regression: element counts above 2^31 (full DeepSeek-V3 / Kimi-K2 expert tensors).
   A real 2^31-element tensor needs 8.6 GB, so the data lives in an uncommitted
   MAP_NORESERVE mapping and the test only reads it (untouched pages are the shared
   zero page); only two pages near the end are written. Peak RSS stays small. */
#include "vira_rt.h"
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

static void put_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }

int main(void) {
  /* 2 x 32768 x 32769 = 2,147,549,184 elements: just over 2^31 */
  const int32_t dims[3] = {2, 32768, 32769};
  const int64_t want = 2LL * 32768 * 32769;

  /* 1. shape metadata and numel */
  vira_tensor_f32 t = {0};
  _vira_tensor_init_shape(&t, 3, dims);
  CHECK(t.rows == 65536 && t.cols == 32769, "rows/cols %d x %d", t.rows, t.cols);
  CHECK(vira_tensor_numel64(t) == want, "numel64 %lld, want %lld", (long long)vira_tensor_numel64(t), (long long)want);
  CHECK(t.strides[0] == 32768 * 32769 && t.strides[1] == 32769 && t.strides[2] == 1, "strides");

  /* 2. a shape whose flattened row count does not fit must be refused, not wrapped */
  pid_t pid = fork();
  if (pid == 0) {
    if (!freopen("/dev/null", "w", stderr)) _exit(3);
    vira_tensor_f32 bad = {0};
    const int32_t big[3] = {65536, 65536, 2}; /* rows = 2^32 */
    _vira_tensor_init_shape(&bad, 3, big);
    _exit(0);
  }
  int st = 0;
  waitpid(pid, &st, 0);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 1, "oversized row count was not refused (status %d)", st);

  /* 3. full-size reductions over > 2^31 elements */
  size_t bytes = (size_t)want * sizeof(vira_f32);
  void *mem = mmap(NULL, bytes + 64, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (mem == MAP_FAILED) {
    printf("SKIP: cannot reserve %zu bytes of address space\n", bytes);
    return fails != 0;
  }
  t.data = (vira_f32 *)mem;
  t.raw_alloc = NULL;
  t.is_view = true;
  const int64_t far = want - 100; /* past 2^31 */
  t.data[far] = 3.0f;
  t.data[7] = 2.0f;

  vira_f32 s = vira_tensor_sum(t);
  CHECK(s == 5.0f, "sum %g, want 5 (the element past 2^31 was skipped)", s);

  int64_t am = vira_sample_token(t, 0.0f); /* greedy: argmax over all elements */
  CHECK(am == far, "argmax %lld, want %lld", (long long)am, (long long)far);

  /* 4. a reshape of the big view keeps the 64-bit count check */
  const int32_t d2[2] = {65536, 32769};
  vira_tensor_f32 v = vira_tensor_view_nd(t, 2, d2);
  CHECK(v.data == t.data && v.rows == 65536, "view_nd");
  munmap(mem, bytes + 64);

  /* 5. GGUF: an axis above 2^31-1 is reported, not silently loaded as size 1 */
  const char *path = "large_axis_test.gguf";
  FILE *f = fopen(path, "wb");
  put_u32(f, 0x46554747); put_u32(f, 3); put_u64(f, 1); put_u64(f, 0);
  const char *name = "blk.0.ffn_down_exps.weight";
  put_u64(f, strlen(name)); fwrite(name, 1, strlen(name), f);
  put_u32(f, 2); put_u64(f, 3000000000ULL); put_u64(f, 2); /* ggml order: ne0 = 3e9 */
  put_u32(f, 0); put_u64(f, 0);                            /* F32, offset 0 */
  fclose(f);
  Result_tensor_f32 r = vira_gguf_load_tensor(vira_str_lit(path, strlen(path)), vira_str_lit(name, strlen(name)));
  CHECK(r.is_err, "loader accepted an axis of 3e9 elements");
  if (r.is_err) {
    CHECK(r.err.len > 0 && strstr((const char *)r.err.data, "2^31") != NULL, "unexpected error: %.*s", (int)r.err.len, (const char *)r.err.data);
    vira_str_free(&r.err);
  } else {
    vira_tensor_f32_free(&r.val);
  }
  remove(path);

  printf(fails ? "large tensor test: %d failure(s)\n" : "large tensor test ok\n", fails);
  return fails != 0;
}
`
	tmpDir := t.TempDir()
	cFile := filepath.Join(tmpDir, "large_tensor_test.c")
	binFile := filepath.Join(tmpDir, "large_tensor_test")
	if err := os.WriteFile(cFile, []byte(cCode), 0644); err != nil {
		t.Fatalf("write C file: %v", err)
	}
	// Same semantics as generated code (-fwrapv); UBSan would also catch a remaining
	// signed overflow, but it is not available everywhere, so it is not required here.
	cFlags := []string{"-std=gnu99", "-O2", "-fwrapv", "-pthread", "-I" + runtimeDir, cFile, "-lm", "-o", binFile}
	buildCmd := exec.Command(cCompiler, cFlags...)
	if out, err := buildCmd.CombinedOutput(); err != nil {
		t.Fatalf("large tensor test failed to compile (%v):\n%s", err, out)
	}
	runCmd := exec.Command(binFile)
	runCmd.Dir = tmpDir // the test writes and removes a small .gguf file
	out, err := runCmd.CombinedOutput()
	if strings.Contains(string(out), "SKIP:") {
		t.Skip(strings.TrimSpace(string(out)))
	}
	if err != nil || !strings.Contains(string(out), "large tensor test ok") {
		t.Fatalf("large tensor element counts are wrong (%v):\n%s", err, out)
	}
}
