package main_test

import (
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// V1 regression (T-004): _vira_fp16_to_fp32 must agree with the C compiler's own
// IEEE 754 binary16 definition (a Go reference decoder; no _Float16 needed, which gcc < 12 and
// clang < 15 lack on x86) for all 65,536 half values. The scalar path (no F16C:
// every ARM build, and x86 builds without -mf16c) used to decode the 2,046 subnormals
// as values around +-1e36. Both paths are checked when the host can run F16C.
func TestRuntimeFP16DecodeExhaustive(t *testing.T) {
	var cCompiler string
	for _, cand := range []string{"gcc", "clang", "cc"} {
		if _, err := exec.LookPath(cand); err == nil {
			cCompiler = cand
			break
		}
	}
	if cCompiler == "" {
		t.Skip("Host C compiler not found in PATH; skipping fp16 decode test")
	}
	runtimeDir, err := filepath.Abs("runtime")
	if err != nil {
		t.Fatalf("failed to resolve runtime dir: %v", err)
	}

	cCode := `
#include "vira_rt.h"
#include <math.h>

/* argv[1]: file with 65536 little-endian float32 bit patterns, the expected decode of
   every half value (computed by the Go test from the IEEE 754 binary16 definition). */
int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    static uint32_t ref[65536];
    if (fread(ref, 4, 65536, f) != 65536) return 2;
    fclose(f);
    int bad = 0, sub_bad = 0;
    for (uint32_t i = 0; i < 65536u; i++) {
        uint16_t h = (uint16_t)i;
        float want, got = _vira_fp16_to_fp32(h);
        memcpy(&want, &ref[i], 4);
        uint32_t gb;
        memcpy(&gb, &got, 4);
        /* NaN payloads may be quietened differently; compare NaN-ness and sign only */
        int ok = isnan(want) ? (isnan(got) && signbit(got) == signbit(want)) : (ref[i] == gb);
        if (!ok) {
            if (bad < 5) printf("mismatch 0x%04x: got %.9g want %.9g\n", (unsigned)h, got, want);
            bad++;
            if ((h & 0x7C00u) == 0) sub_bad++;
        }
    }
    printf("fp16 exhaustive: %d mismatches (%d subnormal) of 65536\n", bad, sub_bad);
    return bad != 0;
}
`
	refFile := filepath.Join(t.TempDir(), "fp16_ref.bin")
	refBytes := make([]byte, 0, 65536*4)
	for i := 0; i < 65536; i++ {
		refBytes = binary.LittleEndian.AppendUint32(refBytes, math.Float32bits(halfToFloat32(uint16(i))))
	}
	if err := os.WriteFile(refFile, refBytes, 0644); err != nil {
		t.Fatalf("write reference table: %v", err)
	}
	tmpDir := t.TempDir()
	cFile := filepath.Join(tmpDir, "fp16_exhaustive.c")
	if err := os.WriteFile(cFile, []byte(cCode), 0644); err != nil {
		t.Fatalf("write C file: %v", err)
	}

	variants := []struct {
		name  string
		flags []string
	}{
				{"scalar", []string{"-mno-f16c"}},
		{"f16c", []string{"-mf16c"}},
	}
	if runtime.GOARCH != "amd64" {
		variants = []struct {
			name  string
			flags []string
		}{{"scalar", nil}}
	}

	for _, v := range variants {
		v := v
		t.Run(v.name, func(t *testing.T) {
			if v.name == "f16c" && !hostHasF16C() {
				t.Skip("host CPU has no F16C")
			}
			binFile := filepath.Join(tmpDir, "fp16_"+v.name)
			if os.PathSeparator == '\\' {
				binFile += ".exe"
			}
			cFlags := []string{"-std=gnu99", "-O2"}
			cFlags = append(cFlags, v.flags...)
			cFlags = append(cFlags, "-pthread", "-I"+runtimeDir, cFile, "-lm", "-o", binFile)
			buildCmd := exec.Command(cCompiler, cFlags...)
			buildCmd.Env = append(os.Environ(), "PATH="+filepath.Dir(cCompiler)+string(os.PathListSeparator)+os.Getenv("PATH"))
			if out, err := buildCmd.CombinedOutput(); err != nil {
				t.Fatalf("fp16 test failed to compile (%v):\n%s", err, out)
			}
			out, err := exec.Command(binFile, refFile).CombinedOutput()
			if v.name == "f16c" && err != nil && !strings.Contains(string(out), "fp16 exhaustive:") {
				// e.g. Sandy Bridge on Windows/macOS: the F16C binary dies with an illegal instruction
				t.Skipf("F16C build did not run on this CPU (%v)", err)
			}
			if err != nil || !strings.Contains(string(out), "fp16 exhaustive: 0 mismatches") {
				t.Fatalf("fp16 decode (%s path) is wrong (%v):\n%s", v.name, err, out)
			}
		})
	}
}

func hostHasF16C() bool {
	if runtime.GOARCH != "amd64" {
		return false
	}
	if runtime.GOOS == "linux" {
		b, err := os.ReadFile("/proc/cpuinfo")
		return err == nil && strings.Contains(string(b), " f16c")
	}
	return true // macOS / Windows: try it; a CPU without F16C (Sandy Bridge) is skipped after the run
}

// halfToFloat32 decodes an IEEE 754 binary16 value exactly, straight from the definition.
func halfToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h & 0x3FF)
	switch exp {
	case 0:
		// zero or subnormal: frac * 2^-24, exact in float32
		v := float32(frac) * (1.0 / (1 << 24))
		return math.Float32frombits(math.Float32bits(v) | sign)
	case 31:
		return math.Float32frombits(sign | 0xFF<<23 | frac<<13) // inf, or NaN with payload
	}
	return math.Float32frombits(sign | (exp+112)<<23 | frac<<13)
}
