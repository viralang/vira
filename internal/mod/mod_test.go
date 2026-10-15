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
