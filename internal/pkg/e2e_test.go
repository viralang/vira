package pkg_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestPackageManagerEndToEnd drives the real vira binary through add,
// install, run and vendor on a project with a transitive chain, a cycle, a
// sub-package import in the unquoted form, and a git package with a
// dependency inside itself.
func TestPackageManagerEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the vira binary")
	}
	haveCC := false
	for _, cc := range []string{"cc", "gcc", "clang"} {
		if _, err := exec.LookPath(cc); err == nil {
			haveCC = true
			break
		}
	}
	if !haveCC {
		t.Skip("no C compiler")
	}
	needGit(t)

	tmp := t.TempDir()
	bin := filepath.Join(tmp, "vira")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if out, err := exec.Command("go", "build", "-o", bin, "../../cmd/vira").CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}

	// A git package whose vira.mod refers to a folder inside itself.
	tools := filepath.Join(tmp, "tools")
	writeFiles(t, tools, map[string]string{
		"vira.mod":        "module tools\nversion 0.1.0\n\ndeps\n\t./inner\n",
		"tools.vir":       "module tools\n\nimport inner\n\npub fn twice() int {\n\treturn inner.one() * 2\n}\n",
		"inner/vira.mod":  "module inner\nversion 0.1.0\n",
		"inner/inner.vir": "module inner\n\npub fn one() int {\n\treturn 1\n}\n",
	})
	toolsURL := gitRepo(t, tools)

	app := filepath.Join(tmp, "app")
	writeFiles(t, app, map[string]string{
		"vira.mod": "module app\nversion 0.1.0\n",
		// mid → ../leaf → ../mid: a transitive chain with a cycle.
		"libs/mid/vira.mod":     "module mid\nversion 0.1.0\n\ndeps\n\t../leaf\n",
		"libs/mid/mid.vir":      "module mid\n\nimport leaf\n\npub fn run() int {\n\treturn leaf.val() + 1\n}\n",
		"libs/leaf/vira.mod":    "module leaf\nversion 0.1.0\n\ndeps\n\t../mid\n",
		"libs/leaf/leaf.vir":    "module leaf\n\npub fn val() int {\n\treturn 41\n}\n",
		"libs/leaf/sub/sub.vir": "module sub\n\npub fn hi() string {\n\treturn \"hi\"\n}\n",
		"main.vir": "module main\n\nimport mid\nimport leaf/sub\nimport tools\n\nfn main() {\n" +
			"\tprintln(\"mid=${mid.run()}\")\n\tprintln(sub.hi())\n\tprintln(\"tools=${tools.twice()}\")\n}\n",
	})

	vira := func(args ...string) string {
		t.Helper()
		cmd := exec.Command(bin, args...)
		cmd.Dir = app
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("vira %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}
	checkRun := func(stage string) {
		t.Helper()
		out := vira("run", "main.vir")
		for _, want := range []string{"mid=42", "hi", "tools=2"} {
			if !strings.Contains(out, want) {
				t.Fatalf("%s: output lacks %q:\n%s", stage, want, out)
			}
		}
	}
	cache := filepath.Join(app, ".vira")

	vira("add", "./libs/mid")
	vira("add", toolsURL)
	lock, _ := os.ReadFile(filepath.Join(app, "vira.lock"))
	for _, want := range []string{`"./libs/mid"`, `"./libs/leaf"`, `"module": "inner"`} {
		if !strings.Contains(string(lock), want) {
			t.Errorf("vira.lock lacks %s:\n%s", want, lock)
		}
	}
	checkRun("after add")

	if err := os.RemoveAll(cache); err != nil {
		t.Fatal(err)
	}
	if out := vira("install"); !strings.Contains(out, "installed 4 package") {
		t.Errorf("install output: %s", out)
	}
	if after, _ := os.ReadFile(filepath.Join(app, "vira.lock")); string(after) != string(lock) {
		t.Errorf("install rewrote vira.lock:\n%s\nwas:\n%s", after, lock)
	}
	checkRun("after install")

	if out := vira("vendor"); !strings.Contains(out, "Vendored 4 package") {
		t.Errorf("vendor output: %s", out)
	}
	if err := os.RemoveAll(cache); err != nil {
		t.Fatal(err)
	}
	checkRun("from vendor/ only")
}
