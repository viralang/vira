package mod

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Dependency represents an external or local package dependency.
type Dependency struct {
	Path    string `json:"path"`
	Version string `json:"version"`
}

// Manifest represents a parsed vira.mod configuration.
type Manifest struct {
	Module       string       `json:"module"`
	Version      string       `json:"version"`
	License      string       `json:"license"`
	Author       string       `json:"author"`
	Description  string       `json:"description"`
	Dependencies []Dependency `json:"dependencies"`
	Deps         []string     `json:"-"` // legacy compatibility
}

// ParseManifest parses the contents of a vira.mod / vex.mod file.
func ParseManifest(content string) (*Manifest, error) {
	m := &Manifest{
		Version: "0.1.0",
		License: "MIT",
	}

	inDeps := false
	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		rawLine := scanner.Text()
		trimmed := strings.TrimSpace(rawLine)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "//") {
			continue
		}

		if strings.ToLower(trimmed) == "deps" {
			inDeps = true
			continue
		}

		if inDeps {
			if strings.HasPrefix(rawLine, " ") || strings.HasPrefix(rawLine, "\t") {
				fields := strings.Fields(trimmed)
				if len(fields) >= 1 {
					dep := Dependency{Path: fields[0], Version: "latest"}
					if len(fields) >= 2 {
						dep.Version = fields[1]
					}
					m.Dependencies = append(m.Dependencies, dep)
					m.Deps = append(m.Deps, dep.Path)
				}
				continue
			} else {
				inDeps = false
			}
		}

		parts := strings.SplitN(trimmed, " ", 2)
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
		case "dep":
			fields := strings.Fields(val)
			if len(fields) >= 1 {
				dep := Dependency{Path: fields[0], Version: "latest"}
				if len(fields) >= 2 {
					dep.Version = fields[1]
				}
				m.Dependencies = append(m.Dependencies, dep)
				m.Deps = append(m.Deps, dep.Path)
			}
		}
	}

	if m.Module == "" {
		return nil, fmt.Errorf("vira.mod missing required 'module' declaration")
	}

	return m, nil
}

// Format formats the manifest into canonical vira.mod syntax.
func (m *Manifest) Format() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("module %s\n", m.Module))
	if m.Version != "" {
		sb.WriteString(fmt.Sprintf("version %s\n", m.Version))
	}
	if m.License != "" {
		sb.WriteString(fmt.Sprintf("license %s\n", m.License))
	}
	if m.Author != "" {
		sb.WriteString(fmt.Sprintf("author %s\n", m.Author))
	}
	if m.Description != "" {
		sb.WriteString(fmt.Sprintf("description %s\n", m.Description))
	}
	if len(m.Dependencies) > 0 {
		sb.WriteString("\ndeps\n")
		for _, dep := range m.Dependencies {
			if dep.Version != "" && dep.Version != "latest" {
				sb.WriteString(fmt.Sprintf("\t%s %s\n", dep.Path, dep.Version))
			} else {
				sb.WriteString(fmt.Sprintf("\t%s\n", dep.Path))
			}
		}
	}
	return sb.String()
}

// Save writes the manifest formatted string to filePath.
func (m *Manifest) Save(filePath string) error {
	return os.WriteFile(filePath, []byte(m.Format()), 0644)
}

// AddDependency adds or updates a package dependency.
func (m *Manifest) AddDependency(pkgPath, version string) {
	if version == "" {
		version = "latest"
	}
	for i, d := range m.Dependencies {
		if d.Path == pkgPath {
			m.Dependencies[i].Version = version
			return
		}
	}
	m.Dependencies = append(m.Dependencies, Dependency{Path: pkgPath, Version: version})
	m.Deps = append(m.Deps, pkgPath)
}

// RemoveDependency removes a package dependency by path.
func (m *Manifest) RemoveDependency(pkgPath string) bool {
	removed := false
	var updated []Dependency
	for _, d := range m.Dependencies {
		if d.Path == pkgPath {
			removed = true
		} else {
			updated = append(updated, d)
		}
	}
	m.Dependencies = updated
	var updatedDeps []string
	for _, d := range m.Deps {
		if d != pkgPath {
			updatedDeps = append(updatedDeps, d)
		}
	}
	m.Deps = updatedDeps
	return removed
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
