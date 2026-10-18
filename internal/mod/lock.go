package mod

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// LockedPackage represents an exact, reproducible snapshot of a resolved dependency.
type LockedPackage struct {
	Path     string `json:"path"`
	Version  string `json:"version"`
	Commit   string `json:"commit,omitempty"`
	Checksum string `json:"checksum"` // "sha256:<hex>"
}

// Lockfile represents the contents of vira.lock.
type Lockfile struct {
	Version  int             `json:"version"`
	Packages []LockedPackage `json:"packages"`
}

// NewLockfile creates a new empty Lockfile instance.
func NewLockfile() *Lockfile {
	return &Lockfile{
		Version:  1,
		Packages: make([]LockedPackage, 0),
	}
}

// ReadLockfile parses a vira.lock file from disk.
func ReadLockfile(filePath string) (*Lockfile, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return NewLockfile(), nil
		}
		return nil, err
	}
	var lock Lockfile
	if err := json.Unmarshal(data, &lock); err != nil {
		return nil, fmt.Errorf("invalid lockfile format in %s: %w", filePath, err)
	}
	return &lock, nil
}

// Save writes the lockfile to disk with clean indentation.
func (l *Lockfile) Save(filePath string) error {
	data, err := json.MarshalIndent(l, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, append(data, '\n'), 0644)
}

// Find retrieves a locked package by its import path.
func (l *Lockfile) Find(pkgPath string) *LockedPackage {
	for i := range l.Packages {
		if l.Packages[i].Path == pkgPath {
			return &l.Packages[i]
		}
	}
	return nil
}

// Set adds or replaces a locked package entry, keeping the list sorted.
func (l *Lockfile) Set(pkg LockedPackage) {
	for i := range l.Packages {
		if l.Packages[i].Path == pkg.Path {
			l.Packages[i] = pkg
			return
		}
	}
	l.Packages = append(l.Packages, pkg)
	sort.Slice(l.Packages, func(i, j int) bool {
		return l.Packages[i].Path < l.Packages[j].Path
	})
}

// Remove deletes a package from the lockfile by its import path.
func (l *Lockfile) Remove(pkgPath string) bool {
	removed := false
	var updated []LockedPackage
	for _, p := range l.Packages {
		if p.Path == pkgPath {
			removed = true
		} else {
			updated = append(updated, p)
		}
	}
	l.Packages = updated
	return removed
}

// ComputeDirChecksum computes a deterministic SHA-256 hash across all files in dir.
func ComputeDirChecksum(dirPath string) (string, error) {
	hasher := sha256.New()
	var fileList []string

	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			base := filepath.Base(path)
			// Ignore .git, temporary and lock files
			if base == ".git" || strings.HasSuffix(base, ".tmp") {
				return nil
			}
			rel, _ := filepath.Rel(dirPath, path)
			fileList = append(fileList, filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil {
		return "", err
	}

	sort.Strings(fileList)

	for _, relPath := range fileList {
		fullPath := filepath.Join(dirPath, filepath.FromSlash(relPath))
		f, err := os.Open(fullPath)
		if err != nil {
			return "", err
		}
		hasher.Write([]byte(relPath))
		if _, err := io.Copy(hasher, f); err != nil {
			_ = f.Close()
			return "", err
		}
		_ = f.Close()
	}

	return "sha256:" + hex.EncodeToString(hasher.Sum(nil)), nil
}
