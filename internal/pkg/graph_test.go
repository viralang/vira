package pkg_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/viralang/vira/internal/mod"
	"github.com/viralang/vira/internal/pkg"
)

func writeFiles(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for rel, content := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}
}

func needGit(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
}

// gitRepo turns dir (already holding files) into a git repository with one
// commit and returns its file:// URL.
func gitRepo(t *testing.T, dir string) string {
	t.Helper()
	for _, args := range [][]string{
		{"init", "-q", "-b", "main"},
		{"config", "uploadpack.allowAnySHA1InWant", "true"},
		{"add", "-A"},
		{"commit", "-q", "-m", "init"},
	} {
		cmd := exec.Command("git", append([]string{"-C", dir, "-c", "user.name=t", "-c", "user.email=t@example.com"}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return "file://" + filepath.ToSlash(dir)
}

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// A1: a local dependency inside the project, referring to a sibling with
// ../leaf, must stay local (it used to become https://libs/leaf/).
func TestLocalDependencyInsideProject(t *testing.T) {
	app := t.TempDir()
	writeFiles(t, app, map[string]string{
		"vira.mod":           "module app\nversion 0.1.0\n",
		"libs/mid/vira.mod":  "module mid\nversion 0.1.0\n\ndeps\n\t../leaf\n",
		"libs/mid/mid.vir":   "module mid\n",
		"libs/leaf/vira.mod": "module leaf\nversion 0.1.0\n",
		"libs/leaf/leaf.vir": "module leaf\n",
	})
	mgr := pkg.NewManager()
	locked, err := mgr.Add(app, "./libs/../libs/mid", "")
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if locked.Path != "./libs/mid" {
		t.Errorf("Add recorded %q, want ./libs/mid", locked.Path)
	}
	lf, _ := mod.ReadLockfile(filepath.Join(app, "vira.lock"))
	if lf.Find("./libs/leaf") == nil {
		t.Fatalf("expected ./libs/leaf in vira.lock, got %+v", lf.Packages)
	}

	_ = os.RemoveAll(pkg.ProjectCacheDir(app))
	n, err := mgr.Install(app)
	if err != nil {
		t.Fatalf("Install failed: %v", err)
	}
	if n != 2 {
		t.Errorf("Install restored %d packages, want 2", n)
	}
	if !exists(filepath.Join(pkg.ProjectCacheDir(app), "leaf", "leaf.vir")) {
		t.Errorf("leaf not restored")
	}
}

// A2: a git package's ./inner resolves inside the package; ../.. and
// absolute paths are refused.
func TestGitPackageRelativeDependencies(t *testing.T) {
	needGit(t)
	tmp := t.TempDir()

	good := filepath.Join(tmp, "good")
	writeFiles(t, good, map[string]string{
		"vira.mod":        "module good\nversion 0.1.0\n\ndeps\n\t./inner\n",
		"good.vir":        "module good\n",
		"inner/vira.mod":  "module inner\nversion 0.1.0\n",
		"inner/inner.vir": "module inner\n",
	})
	goodURL := gitRepo(t, good)

	outside := filepath.Join(tmp, "outside")
	writeFiles(t, outside, map[string]string{"secret.txt": "do not copy\n", "vira.mod": "module outside\n"})

	app := filepath.Join(tmp, "app")
	writeFiles(t, app, map[string]string{"vira.mod": "module app\nversion 0.1.0\n"})
	mgr := pkg.NewManager()
	if _, err := mgr.Add(app, goodURL, ""); err != nil {
		t.Fatalf("Add of a git package with ./inner failed: %v", err)
	}
	if !exists(filepath.Join(pkg.ProjectCacheDir(app), "inner", "inner.vir")) {
		t.Errorf("inner package not installed")
	}
	_ = os.RemoveAll(pkg.ProjectCacheDir(app))
	if _, err := mgr.Install(app); err != nil {
		t.Fatalf("Install failed: %v", err)
	}
	if !exists(filepath.Join(pkg.ProjectCacheDir(app), "inner", "inner.vir")) {
		t.Errorf("inner package not restored by Install")
	}

	for name, dep := range map[string]string{
		"escape":   "../../outside",
		"absolute": outside,
	} {
		bad := filepath.Join(tmp, "bad_"+name)
		writeFiles(t, bad, map[string]string{
			"vira.mod": "module bad_" + name + "\nversion 0.1.0\n\ndeps\n\t" + dep + "\n",
			"bad.vir":  "module bad\n",
		})
		badURL := gitRepo(t, bad)
		badApp := filepath.Join(tmp, "app_"+name)
		writeFiles(t, badApp, map[string]string{"vira.mod": "module app\nversion 0.1.0\n"})
		_, err := mgr.Add(badApp, badURL, "")
		if err == nil || !strings.Contains(err.Error(), "may only use folders inside itself") {
			t.Errorf("%s: expected the dependency %q to be refused, got %v", name, dep, err)
		}
		if exists(filepath.Join(pkg.ProjectCacheDir(badApp), "outside")) {
			t.Errorf("%s: a folder outside the package was copied", name)
		}
		if b, _ := os.ReadFile(filepath.Join(badApp, "vira.mod")); strings.Contains(string(b), "deps") {
			t.Errorf("%s: vira.mod changed although Add failed:\n%s", name, b)
		}
	}
}

// transitiveGitSetup: app → ../mid (local) → leaf (git, locked at a commit).
func transitiveGitSetup(t *testing.T) (app string, leafURL string) {
	t.Helper()
	needGit(t)
	tmp := t.TempDir()
	leaf := filepath.Join(tmp, "leaf")
	writeFiles(t, leaf, map[string]string{
		"vira.mod": "module leaf\nversion 0.1.0\n",
		"leaf.vir": "module leaf\n\npub fn val() int {\n\treturn 41\n}\n",
	})
	leafURL = gitRepo(t, leaf)
	writeFiles(t, filepath.Join(tmp, "mid"), map[string]string{
		"vira.mod": "module mid\nversion 0.1.0\n\ndeps\n\t" + leafURL + "\n",
		"mid.vir":  "module mid\n",
	})
	app = filepath.Join(tmp, "app")
	writeFiles(t, app, map[string]string{"vira.mod": "module app\nversion 0.1.0\n"})
	if _, err := pkg.NewManager().Add(app, "../mid", ""); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	return app, leafURL
}

// A3: transitive packages are checksum-verified and vira.lock is not rewritten.
func TestTransitiveChecksumVerified(t *testing.T) {
	app, leafURL := transitiveGitSetup(t)
	lockPath := filepath.Join(app, "vira.lock")
	lf, _ := mod.ReadLockfile(lockPath)
	entry := lf.Find(leafURL)
	if entry == nil || entry.Commit == "" || entry.Commit == "local" {
		t.Fatalf("expected leaf locked at a commit, got %+v", entry)
	}

	// Unchanged lock: install verifies and leaves vira.lock byte-identical.
	before, _ := os.ReadFile(lockPath)
	_ = os.RemoveAll(pkg.ProjectCacheDir(app))
	if _, err := pkg.NewManager().Install(app); err != nil {
		t.Fatalf("Install failed: %v", err)
	}
	if after, _ := os.ReadFile(lockPath); string(after) != string(before) {
		t.Errorf("vira.lock rewritten by install:\n%s\nwas:\n%s", after, before)
	}

	// Tampered checksum: install fails and leaves the lock alone.
	entry.Checksum = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
	if err := lf.Save(lockPath); err != nil {
		t.Fatal(err)
	}
	tampered, _ := os.ReadFile(lockPath)
	_ = os.RemoveAll(pkg.ProjectCacheDir(app))
	_, err := pkg.NewManager().Install(app)
	if err == nil || !strings.Contains(err.Error(), "vira.lock expects") {
		t.Fatalf("expected a checksum failure, got %v", err)
	}
	if after, _ := os.ReadFile(lockPath); string(after) != string(tampered) {
		t.Errorf("vira.lock changed after a failed install")
	}
	if exists(pkg.PackageDestDir(pkg.ProjectCacheDir(app), leafURL)) {
		t.Errorf("content that failed verification was left in the cache")
	}
}

// A4: vendor restores a transitive package missing from the cache.
func TestVendorRestoresMissingTransitive(t *testing.T) {
	app, leafURL := transitiveGitSetup(t)
	leafCache := pkg.PackageDestDir(pkg.ProjectCacheDir(app), leafURL)
	if err := os.RemoveAll(leafCache); err != nil {
		t.Fatal(err)
	}
	n, err := pkg.NewManager().Vendor(app)
	if err != nil {
		t.Fatalf("Vendor failed: %v", err)
	}
	if n != 2 {
		t.Errorf("vendored %d packages, want 2", n)
	}
	if !exists(filepath.Join(leafCache, "leaf.vir")) {
		t.Errorf("missing transitive package not restored to the cache")
	}
	if !exists(filepath.Join(pkg.PackageDestDir(pkg.VendorDir(app), leafURL), "leaf.vir")) {
		t.Errorf("transitive package not vendored")
	}

	// A transitive package that cannot be fetched makes vendor fail.
	lf, _ := mod.ReadLockfile(filepath.Join(app, "vira.lock"))
	lf.Find(leafURL).Commit = "0123456789abcdef0123456789abcdef01234567"
	_ = lf.Save(filepath.Join(app, "vira.lock"))
	_ = os.RemoveAll(leafCache)
	if _, err := pkg.NewManager().Vendor(app); err == nil {
		t.Errorf("Vendor succeeded although a transitive package could not be fetched")
	}
}

// A5: two packages wanting different versions of the same dependency is an
// error, unless the root vira.mod pins one.
func TestVersionConflict(t *testing.T) {
	tmp := t.TempDir()
	writeFiles(t, tmp, map[string]string{
		"c/vira.mod":   "module c\nversion 0.1.0\n",
		"c/c.vir":      "module c\n",
		"a/vira.mod":   "module a\nversion 0.1.0\n\ndeps\n\t../c v1\n",
		"a/a.vir":      "module a\n",
		"b/vira.mod":   "module b\nversion 0.1.0\n\ndeps\n\t../c v2\n",
		"b/b.vir":      "module b\n",
		"app/vira.mod": "module app\nversion 0.1.0\n",
	})
	app := filepath.Join(tmp, "app")
	mgr := pkg.NewManager()
	if _, err := mgr.Add(app, "../a", ""); err != nil {
		t.Fatalf("Add a: %v", err)
	}
	modBefore, _ := os.ReadFile(filepath.Join(app, "vira.mod"))
	_, err := mgr.Add(app, "../b", "")
	if err == nil {
		t.Fatalf("expected a version conflict")
	}
	for _, want := range []string{"'../c'", "'a' requires v1", "'b' requires v2"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("conflict error %q does not mention %s", err, want)
		}
	}
	if modAfter, _ := os.ReadFile(filepath.Join(app, "vira.mod")); string(modAfter) != string(modBefore) {
		t.Errorf("vira.mod changed by a failed Add:\n%s", modAfter)
	}

	if _, err := mgr.Add(app, "../c", "v2"); err != nil {
		t.Fatalf("pinning c in the root failed: %v", err)
	}
	if _, err := mgr.Add(app, "../b", ""); err != nil {
		t.Fatalf("Add b with c pinned in the root failed: %v", err)
	}
	if _, err := mgr.Install(app); err != nil {
		t.Fatalf("Install with c pinned failed: %v", err)
	}
}

// A6: a package is found by the module name in its vira.mod, not only by the
// last part of its URL.
func TestImportByModuleName(t *testing.T) {
	needGit(t)
	tmp := t.TempDir()
	ra := filepath.Join(tmp, "ra")
	writeFiles(t, ra, map[string]string{
		"vira.mod": "module a\nversion 0.1.0\n",
		"a.vir":    "module a\n",
	})
	url := gitRepo(t, ra)
	app := filepath.Join(tmp, "app")
	writeFiles(t, app, map[string]string{"vira.mod": "module app\nversion 0.1.0\n"})
	locked, err := pkg.NewManager().Add(app, url, "")
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if locked.Module != "a" {
		t.Errorf("lock module = %q, want a", locked.Module)
	}
	cacheDir := pkg.PackageDestDir(pkg.ProjectCacheDir(app), url)
	for _, name := range []string{"a", "ra"} {
		found := false
		for _, c := range pkg.ResolveImportCandidates(app, app, name) {
			if filepath.Clean(c) == filepath.Clean(cacheDir) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("import %s does not resolve to %s", name, cacheDir)
		}
	}
}
