package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// V17: a bare call inside a package must resolve to that package's own
// function even when another imported module (strings) exports the same name.
func TestBareCallPrefersCurrentModule(t *testing.T) {
	if _, err := exec.LookPath("cc"); err != nil {
		t.Skip("no C compiler")
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "vira-bin")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if out, err := exec.Command("go", "build", "-o", bin, "./cmd/vira").CombinedOutput(); err != nil {
		t.Fatalf("build vira: %v\n%s", err, out)
	}
	proj := filepath.Join(dir, "app")
	files := map[string]string{
		"vira.mod": "module app\nversion 0.1.0\n\ndeps\n\tmylib 0.1.0\n",
		"src/main.vir": "module main\n\nimport strings\nimport mylib\n\nfn main() {\n\tprintln(strings.to_lower('A') + mylib.label('ab'))\n}\n",
		".vira/pkg/mylib/src/mylib.vir": "module mylib\n\nfn pad_right(s string, width int) string {\n\tmut out := s.clone()\n\tfor out.len < width {\n\t\tout += '.'\n\t}\n\treturn out\n}\n\npub fn label(s string) string {\n\treturn pad_right(s, 6)\n}\n",
	}
	for name, src := range files {
		p := filepath.Join(proj, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cmd := exec.Command(bin, "run", "src/main.vir")
	cmd.Dir = proj
	out, err := cmd.CombinedOutput()
	if err != nil || !strings.Contains(string(out), "aab....") {
		t.Fatalf("want output containing %q, err=%v\n%s", "aab....", err, out)
	}
}
