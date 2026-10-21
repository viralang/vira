package mod_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/viralang/vira/internal/mod"
)

func TestParseManifest(t *testing.T) {
	manifestContent := `
# Vira Module Configuration
module my_ai_engine
version 1.2.0
license Apache-2.0
author Alice
description High-performance tensor graph
`
	m, err := mod.ParseManifest(manifestContent)
	if err != nil {
		t.Fatalf("ParseManifest failed: %v", err)
	}

	if m.Module != "my_ai_engine" {
		t.Errorf("expected module 'my_ai_engine', got %s", m.Module)
	}
	if m.Version != "1.2.0" {
		t.Errorf("expected version '1.2.0', got %s", m.Version)
	}
	if m.License != "Apache-2.0" {
		t.Errorf("expected license 'Apache-2.0', got %s", m.License)
	}
}

func TestFindProjectRoot(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "vira_mod_test_*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	subDir := filepath.Join(tmpDir, "src", "deep")
	if err := os.MkdirAll(subDir, 0755); err != nil {
		t.Fatal(err)
	}

	modPath := filepath.Join(tmpDir, "vex.mod")
	if err := os.WriteFile(modPath, []byte("module test_project\n"), 0644); err != nil {
		t.Fatal(err)
	}

	root, m, err := mod.FindProjectRoot(subDir)
	if err != nil {
		t.Fatalf("FindProjectRoot failed: %v", err)
	}

	if root != tmpDir {
		t.Errorf("expected root %s, got %s", tmpDir, root)
	}
	if m.Module != "test_project" {
		t.Errorf("expected module 'test_project', got %s", m.Module)
	}
}

func TestManifestWithColonsAndName(t *testing.T) {
	content := `name: "http"
version: "0.1.0"
description: "Vira Standard HTTP Engine"
deps
	net 0.1.0
	io 0.1.0
`
	m, err := mod.ParseManifest(content)
	if err != nil {
		t.Fatalf("ParseManifest with colons failed: %v", err)
	}
	if m.Module != "http" {
		t.Errorf("expected module 'http', got %s", m.Module)
	}
	if len(m.Dependencies) != 2 {
		t.Errorf("expected 2 dependencies, got %d", len(m.Dependencies))
	}
}

func TestGeneratePackageLockfiles(t *testing.T) {
	pkgsRoot := filepath.Join("..", "..", "packages")
	ioCS, err := mod.ComputeDirChecksum(filepath.Join(pkgsRoot, "io"))
	if err != nil {
		t.Fatalf("failed to compute io checksum: %v", err)
	}
	netCS, err := mod.ComputeDirChecksum(filepath.Join(pkgsRoot, "net"))
	if err != nil {
		t.Fatalf("failed to compute net checksum: %v", err)
	}
	httpCS, err := mod.ComputeDirChecksum(filepath.Join(pkgsRoot, "http"))
	if err != nil {
		t.Fatalf("failed to compute http checksum: %v", err)
	}

	// 1. packages/net/vira.lock
	netLock := mod.NewLockfile()
	netLock.Set(mod.LockedPackage{
		Path:     "io",
		Version:  "0.1.0",
		Checksum: ioCS,
	})
	if err := netLock.Save(filepath.Join(pkgsRoot, "net", "vira.lock")); err != nil {
		t.Fatalf("failed to save net/vira.lock: %v", err)
	}

	// 2. packages/http/vira.lock
	httpLock := mod.NewLockfile()
	httpLock.Set(mod.LockedPackage{
		Path:     "io",
		Version:  "0.1.0",
		Checksum: ioCS,
	})
	httpLock.Set(mod.LockedPackage{
		Path:     "net",
		Version:  "0.1.0",
		Checksum: netCS,
	})
	if err := httpLock.Save(filepath.Join(pkgsRoot, "http", "vira.lock")); err != nil {
		t.Fatalf("failed to save http/vira.lock: %v", err)
	}

	// 3. packages/viraserve/vira.lock
	vsLock := mod.NewLockfile()
	vsLock.Set(mod.LockedPackage{
		Path:     "io",
		Version:  "0.1.0",
		Checksum: ioCS,
	})
	vsLock.Set(mod.LockedPackage{
		Path:     "net",
		Version:  "0.1.0",
		Checksum: netCS,
	})
	vsLock.Set(mod.LockedPackage{
		Path:     "http",
		Version:  "0.1.0",
		Checksum: httpCS,
	})
	if err := vsLock.Save(filepath.Join(pkgsRoot, "viraserve", "vira.lock")); err != nil {
		t.Fatalf("failed to save viraserve/vira.lock: %v", err)
	}
}
