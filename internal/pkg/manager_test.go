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

func TestPackageManager_LocalPackageLifecycle(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Create a dummy external local package "my_math"
	mathPkgDir := filepath.Join(tempDir, "my_math")
	if err := os.MkdirAll(mathPkgDir, 0755); err != nil {
		t.Fatal(err)
	}
	mathSrc := `module my_math

pub fn add(a int, b int) int {
	return a + b
}
`
	if err := os.WriteFile(filepath.Join(mathPkgDir, "math.vir"), []byte(mathSrc), 0644); err != nil {
		t.Fatal(err)
	}

	// 2. Create consumer project with vira.mod
	appDir := filepath.Join(tempDir, "my_app")
	if err := os.MkdirAll(appDir, 0755); err != nil {
		t.Fatal(err)
	}
	manifestSrc := `module my_app
version 0.1.0
license MIT
`
	if err := os.WriteFile(filepath.Join(appDir, "vira.mod"), []byte(manifestSrc), 0644); err != nil {
		t.Fatal(err)
	}

	mgr := pkg.NewManager()

	// 3. Test `vira add <local-path>`
	relMathPath, _ := filepath.Rel(appDir, mathPkgDir)
	locked, err := mgr.Add(appDir, relMathPath, "v1.0.0")
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if locked.Checksum == "" || !strings.HasPrefix(locked.Checksum, "sha256:") {
		t.Errorf("expected valid sha256 checksum, got: %s", locked.Checksum)
	}

	// Check that vira.mod was updated with deps
	modBytes, _ := os.ReadFile(filepath.Join(appDir, "vira.mod"))
	if !strings.Contains(string(modBytes), "deps") || !strings.Contains(string(modBytes), relMathPath) {
		t.Errorf("expected vira.mod to contain dependency: %s", string(modBytes))
	}

	// Check that vira.lock was generated
	lockBytes, _ := os.ReadFile(filepath.Join(appDir, "vira.lock"))
	if !strings.Contains(string(lockBytes), locked.Checksum) {
		t.Errorf("expected vira.lock to contain checksum: %s", string(lockBytes))
	}

	// Check that package was copied to .vira/pkg/
	cachedFile := filepath.Join(pkg.PackageDestDir(pkg.ProjectCacheDir(appDir), relMathPath), "math.vir")
	if _, err := os.Stat(cachedFile); os.IsNotExist(err) {
		t.Errorf("expected cached file at %s", cachedFile)
	}

	// 4. Test `vira vendor`
	vendorCount, err := mgr.Vendor(appDir)
	if err != nil {
		t.Fatalf("Vendor failed: %v", err)
	}
	if vendorCount != 1 {
		t.Errorf("expected 1 vendored package, got %d", vendorCount)
	}
	vendoredFile := filepath.Join(pkg.PackageDestDir(pkg.VendorDir(appDir), relMathPath), "math.vir")
	if _, err := os.Stat(vendoredFile); os.IsNotExist(err) {
		t.Errorf("expected vendored file at %s", vendoredFile)
	}

	// 5. Test `vira install`
	// Delete .vira/pkg and re-run install
	_ = os.RemoveAll(pkg.ProjectCacheDir(appDir))
	installed, err := mgr.Install(appDir)
	if err != nil {
		t.Fatalf("Install failed: %v", err)
	}
	if installed != 1 {
		t.Errorf("expected 1 installed package, got %d", installed)
	}
	if _, err := os.Stat(cachedFile); os.IsNotExist(err) {
		t.Errorf("expected re-installed cached file at %s", cachedFile)
	}

	// 6. Test `vira remove`
	if err := mgr.Remove(appDir, relMathPath); err != nil {
		t.Fatalf("Remove failed: %v", err)
	}

	modAfterRemove, _ := os.ReadFile(filepath.Join(appDir, "vira.mod"))
	if strings.Contains(string(modAfterRemove), relMathPath) {
		t.Errorf("expected dependency removed from vira.mod, got: %s", string(modAfterRemove))
	}

	lockAfterRemove, _ := mod.ReadLockfile(filepath.Join(appDir, "vira.lock"))
	if lockAfterRemove.Find(relMathPath) != nil {
		t.Errorf("expected dependency removed from vira.lock")
	}
}

// TestPackageManager_InstallHonoursLock: `vira install` skips packages that
// are bundled with the compiler and reproduces the commit in vira.lock, even
// after the remote branch has moved on. It used to try `git clone https://net/`
// for a "net 0.1.0" dependency, and it always fetched the newest commit and
// rewrote the lock.
func TestPackageManager_InstallHonoursLock(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	tempDir := t.TempDir()
	git := func(dir string, args ...string) string {
		cmd := exec.Command("git", append([]string{"-C", dir, "-c", "user.name=t", "-c", "user.email=t@example.com"}, args...)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}

	// A git "remote" with one commit.
	remote := filepath.Join(tempDir, "remote")
	if err := os.MkdirAll(remote, 0755); err != nil {
		t.Fatal(err)
	}
	git(remote, "init", "-q", "-b", "main")
	// Allow fetching a commit that is no longer a branch tip, as GitHub does.
	git(remote, "config", "uploadpack.allowAnySHA1InWant", "true")
	libFile := filepath.Join(remote, "lib.vir")
	if err := os.WriteFile(libFile, []byte("module lib\n\npub fn v() int {\n\treturn 1\n}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	git(remote, "add", "-A")
	git(remote, "commit", "-q", "-m", "one")
	first := git(remote, "rev-parse", "HEAD")

	appDir := filepath.Join(tempDir, "app")
	if err := os.MkdirAll(appDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(appDir, "vira.mod"), []byte("module app\nversion 0.1.0\n\ndeps\n\tnet 0.1.0\n"), 0644); err != nil {
		t.Fatal(err)
	}

	mgr := pkg.NewManager()
	depPath := "file://" + filepath.ToSlash(remote)
	locked, err := mgr.Add(appDir, depPath, "")
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if locked.Commit != first {
		t.Fatalf("Add locked %s, want %s", locked.Commit, first)
	}

	// The remote moves on; a fresh clone of the app has no package cache.
	if err := os.WriteFile(libFile, []byte("module lib\n\npub fn v() int {\n\treturn 2\n}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	git(remote, "commit", "-q", "-am", "two")
	if err := os.RemoveAll(pkg.ProjectCacheDir(appDir)); err != nil {
		t.Fatal(err)
	}

	count, err := mgr.Install(appDir)
	if err != nil {
		t.Fatalf("Install failed (a bundled package such as net must be skipped): %v", err)
	}
	if count != 1 {
		t.Errorf("expected 1 installed package, got %d", count)
	}
	got, err := os.ReadFile(filepath.Join(pkg.PackageDestDir(pkg.ProjectCacheDir(appDir), depPath), "lib.vir"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(got), "return 1") {
		t.Errorf("Install fetched the newest commit instead of the locked one:\n%s", got)
	}
	lf, err := mod.ReadLockfile(filepath.Join(appDir, "vira.lock"))
	if err != nil {
		t.Fatal(err)
	}
	if l := lf.Find(depPath); l == nil || l.Commit != first || l.Checksum != locked.Checksum {
		t.Errorf("vira.lock changed during install: %+v", l)
	}

	if !pkg.IsStandardPackage("net") || pkg.IsStandardPackage("github.com/viralang/odbc") || pkg.IsStandardPackage("../net") {
		t.Errorf("IsStandardPackage misclassifies a name")
	}
}
