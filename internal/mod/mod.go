package mod

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Manifest represents a parsed vex.mod configuration.
type Manifest struct {
	Module      string
	Version     string
	License     string
	Author      string
	Description string
	Deps        []string
}

// ParseManifest parses the contents of a vex.mod file.
func ParseManifest(content string) (*Manifest, error) {
	m := &Manifest{
		Version: "0.1.0",
		License: "MIT",
	}

	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "//") {
			continue
		}

		parts := strings.SplitN(line, " ", 2)
		if len(parts) < 2 {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(parts[0]))
		val := strings.Trim(strings.TrimSpace(parts[1]), `"'`)

		switch key {
		case "module":
			m.Module = val
		case "version":
			m.Version = val
		case "license":
			m.License = val
		case "author":
			m.Author = val
		case "description":
			m.Description = val
		}
	}

	if m.Module == "" {
		return nil, fmt.Errorf("vex.mod missing required 'module' declaration")
	}

	return m, nil
}

// FindProjectRoot searches up the directory hierarchy starting from `startDir`
// to find a `vex.mod` file.
func FindProjectRoot(startDir string) (string, *Manifest, error) {
	curr, err := filepath.Abs(startDir)
	if err != nil {
		return "", nil, err
	}

	for {
		for _, name := range []string{"vira.mod", "vex.mod"} {
			modPath := filepath.Join(curr, name)
			if data, err := os.ReadFile(modPath); err == nil {
				m, err := ParseManifest(string(data))
				if err != nil {
					return curr, nil, fmt.Errorf("syntax error in %s: %w", modPath, err)
				}
				return curr, m, nil
			}
		}

		parent := filepath.Dir(curr)
		if parent == curr {
			break
		}
		curr = parent
	}

	return "", nil, os.ErrNotExist
}
