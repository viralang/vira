package pkg

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/viralang/vira/internal/mod"
)

// Manager coordinates package resolution, fetching, locking, and vendoring.
type Manager struct {
	GlobalCacheDir string
}

// NewManager initializes the package manager with the default global cache.
func NewManager() *Manager {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	cacheDir := filepath.Join(home, ".vir", "pkg")
	_ = os.MkdirAll(cacheDir, 0755)
	return &Manager{GlobalCacheDir: cacheDir}
}

// ProjectCacheDir returns <projectRoot>/.vira/pkg
func ProjectCacheDir(projectRoot string) string {
	return filepath.Join(projectRoot, ".vira", "pkg")
}

// VendorDir returns <projectRoot>/vendor
func VendorDir(projectRoot string) string {
	return filepath.Join(projectRoot, "vendor")
}

// PackageDestDir returns the destination directory for a package within baseDir.
// For local directory dependencies (e.g. C:\path\to\pkg or ../pkg), it maps to the base folder name (e.g. "pkg").
// For remote Git dependencies (e.g. github.com/user/repo), it preserves the full repository path.
func PackageDestDir(baseDir string, pkgPath string) string {
	isLocal := strings.HasPrefix(pkgPath, ".") || strings.HasPrefix(pkgPath, "/") || (len(pkgPath) >= 2 && pkgPath[1] == ':')
	if isLocal {
		name := filepath.Base(filepath.Clean(pkgPath))
		if name != "" && name != "." && name != string(filepath.Separator) {
			return filepath.Join(baseDir, name)
		}
	}

	cleaned := filepath.Clean(filepath.FromSlash(pkgPath))
	for strings.HasPrefix(cleaned, "..") {
		cleaned = strings.TrimPrefix(cleaned, "..")
		cleaned = strings.TrimPrefix(cleaned, string(filepath.Separator))
		cleaned = strings.TrimPrefix(cleaned, "/")
	}
	cleaned = strings.TrimPrefix(cleaned, string(filepath.Separator))
	cleaned = strings.TrimPrefix(cleaned, "/")

	if len(cleaned) >= 2 && cleaned[1] == ':' {
		cleaned = cleaned[2:]
		cleaned = strings.TrimPrefix(cleaned, string(filepath.Separator))
		cleaned = strings.TrimPrefix(cleaned, "/")
	}
	if cleaned == "" || cleaned == "." {
		cleaned = filepath.Base(pkgPath)
	}
	return filepath.Join(baseDir, cleaned)
}

// ResolveImportCandidates returns candidate file and directory paths for an import target.
// It checks in priority order:
// 1. Relative to importing file's directory
// 2. Project root's src/ directory
// 3. Project root's root directory
// 4. Project vendor/ directory
// 5. Project .vira/pkg/ local cache
// 6. Global ~/.vir/pkg/ cache
func ResolveImportCandidates(fileDir string, projectRoot string, impTarget string) []string {
	var candidates []string

	// 1. Relative to importing file
	if fileDir != "" {
		candidates = append(candidates,
			filepath.Join(fileDir, impTarget),
			filepath.Join(fileDir, impTarget+".vir"),
			filepath.Join(fileDir, impTarget+".vex"),
		)
	}

	// 2. Project root relative paths
	checkRoot := projectRoot
	if checkRoot == "" {
		if cwd, err := os.Getwd(); err == nil {
			checkRoot = cwd
		}
	}
	if checkRoot != "" {
		candidates = append(candidates,
			filepath.Join(checkRoot, "src", impTarget),
			filepath.Join(checkRoot, "src", impTarget+".vir"),
			filepath.Join(checkRoot, "src", impTarget+".vex"),
			filepath.Join(checkRoot, impTarget),
			filepath.Join(checkRoot, impTarget+".vir"),
			filepath.Join(checkRoot, impTarget+".vex"),
			filepath.Join(checkRoot, "vendor", impTarget),
			filepath.Join(checkRoot, "vendor", impTarget+".vir"),
			filepath.Join(checkRoot, "vendor", impTarget+".vex"),
			filepath.Join(checkRoot, ".vira", "pkg", impTarget),
			filepath.Join(checkRoot, ".vira", "pkg", impTarget+".vir"),
			filepath.Join(checkRoot, ".vira", "pkg", impTarget+".vex"),
		)

		// Check manifest dependencies for matches by base name or subpath
		manifestPath := filepath.Join(checkRoot, "vira.mod")
		if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
			manifestPath = filepath.Join(checkRoot, "vex.mod")
		}
		if mBytes, err := os.ReadFile(manifestPath); err == nil {
			if m, err := mod.ParseManifest(string(mBytes)); err == nil {
				for _, dep := range m.Dependencies {
					depDir := PackageDestDir(filepath.Join(checkRoot, ".vira", "pkg"), dep.Path)
					vendorDir := PackageDestDir(filepath.Join(checkRoot, "vendor"), dep.Path)
					if filepath.Base(dep.Path) == impTarget || strings.HasSuffix(filepath.ToSlash(dep.Path), "/"+impTarget) {
						candidates = append(candidates, vendorDir, depDir)
					}
				}
			}
		}
	}

	// 3. Global cache (~/.vir/pkg)
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		candidates = append(candidates,
			filepath.Join(home, ".vir", "pkg", impTarget),
			filepath.Join(home, ".vir", "pkg", impTarget+".vir"),
			filepath.Join(home, ".vir", "pkg", impTarget+".vex"),
		)
	}

	return candidates
}

// Add fetches a package, updates vira.mod and vira.lock.
func (m *Manager) Add(projectRoot string, pkgPath string, version string) (*mod.LockedPackage, error) {
	manifestPath := filepath.Join(projectRoot, "vira.mod")
	if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
		manifestPath = filepath.Join(projectRoot, "vex.mod")
	}

	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("could not find vira.mod in %s: %w", projectRoot, err)
	}

	manifest, err := mod.ParseManifest(string(manifestData))
	if err != nil {
		return nil, err
	}

	lockPath := filepath.Join(projectRoot, "vira.lock")
	lockfile, err := mod.ReadLockfile(lockPath)
	if err != nil {
		return nil, err
	}

	destDir := PackageDestDir(ProjectCacheDir(projectRoot), pkgPath)
	commit, checksum, err := m.fetch(projectRoot, pkgPath, version, destDir)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch package '%s': %w", pkgPath, err)
	}

	if version == "" {
		version = "latest"
	}

	manifest.AddDependency(pkgPath, version)
	if err := manifest.Save(manifestPath); err != nil {
		return nil, fmt.Errorf("failed to save manifest: %w", err)
	}

	locked := mod.LockedPackage{
		Path:     pkgPath,
		Version:  version,
		Commit:   commit,
		Checksum: checksum,
	}
	lockfile.Set(locked)
	if err := lockfile.Save(lockPath); err != nil {
		return nil, fmt.Errorf("failed to save lockfile: %w", err)
	}

	return &locked, nil
}

// Install reads vira.mod and vira.lock, fetching any missing dependencies.
func (m *Manager) Install(projectRoot string) (int, error) {
	manifestPath := filepath.Join(projectRoot, "vira.mod")
	if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
		manifestPath = filepath.Join(projectRoot, "vex.mod")
	}

	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return 0, fmt.Errorf("could not find vira.mod in %s: %w", projectRoot, err)
	}

	manifest, err := mod.ParseManifest(string(manifestData))
	if err != nil {
		return 0, err
	}

	lockPath := filepath.Join(projectRoot, "vira.lock")
	lockfile, err := mod.ReadLockfile(lockPath)
	if err != nil {
		return 0, err
	}

	installedCount := 0
	for _, dep := range manifest.Dependencies {
		destDir := PackageDestDir(ProjectCacheDir(projectRoot), dep.Path)
		
		// If already present, verify checksum if locked
		if _, err := os.Stat(destDir); err == nil {
			if locked := lockfile.Find(dep.Path); locked != nil && locked.Checksum != "" {
				currentSum, cErr := mod.ComputeDirChecksum(destDir)
				if cErr == nil && currentSum == locked.Checksum {
					continue // Already installed and verified!
				}
			}
		}

		commit, checksum, err := m.fetch(projectRoot, dep.Path, dep.Version, destDir)
		if err != nil {
			return installedCount, fmt.Errorf("failed to install '%s': %w", dep.Path, err)
		}

		lockfile.Set(mod.LockedPackage{
			Path:     dep.Path,
			Version:  dep.Version,
			Commit:   commit,
			Checksum: checksum,
		})
		installedCount++
	}

	_ = lockfile.Save(lockPath)
	return installedCount, nil
}

// Remove deletes a dependency from manifest, lockfile, and local project cache.
func (m *Manager) Remove(projectRoot string, pkgPath string) error {
	manifestPath := filepath.Join(projectRoot, "vira.mod")
	if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
		manifestPath = filepath.Join(projectRoot, "vex.mod")
	}

	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return fmt.Errorf("could not find vira.mod in %s: %w", projectRoot, err)
	}

	manifest, err := mod.ParseManifest(string(manifestData))
	if err != nil {
		return err
	}

	if !manifest.RemoveDependency(pkgPath) {
		return fmt.Errorf("package '%s' is not declared in vira.mod", pkgPath)
	}
	_ = manifest.Save(manifestPath)

	lockPath := filepath.Join(projectRoot, "vira.lock")
	if lockfile, err := mod.ReadLockfile(lockPath); err == nil {
		lockfile.Remove(pkgPath)
		_ = lockfile.Save(lockPath)
	}

	pkgDir := PackageDestDir(ProjectCacheDir(projectRoot), pkgPath)
	_ = os.RemoveAll(pkgDir)

	return nil
}

// Vendor copies all project dependencies into <projectRoot>/vendor.
func (m *Manager) Vendor(projectRoot string) (int, error) {
	manifestPath := filepath.Join(projectRoot, "vira.mod")
	if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
		manifestPath = filepath.Join(projectRoot, "vex.mod")
	}

	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return 0, fmt.Errorf("could not find vira.mod in %s: %w", projectRoot, err)
	}

	manifest, err := mod.ParseManifest(string(manifestData))
	if err != nil {
		return 0, err
	}

	vendorBase := VendorDir(projectRoot)
	count := 0

	for _, dep := range manifest.Dependencies {
		srcDir := PackageDestDir(ProjectCacheDir(projectRoot), dep.Path)
		if _, err := os.Stat(srcDir); os.IsNotExist(err) {
			// Fetch it first if missing
			if _, _, fErr := m.fetch(projectRoot, dep.Path, dep.Version, srcDir); fErr != nil {
				return count, fmt.Errorf("failed to fetch package '%s' for vendoring: %w", dep.Path, fErr)
			}
		}

		targetVendor := PackageDestDir(vendorBase, dep.Path)
		_ = os.RemoveAll(targetVendor)
		if err := copyDir(srcDir, targetVendor); err != nil {
			return count, fmt.Errorf("failed to vendor '%s': %w", dep.Path, err)
		}
		count++
	}

	return count, nil
}

// fetch downloads or links a package (from local directory or Git repo) to destDir.
func (m *Manager) fetch(projectRoot string, pkgPath string, version string, destDir string) (commit string, checksum string, err error) {
	// Clean destination directory
	_ = os.RemoveAll(destDir)
	if err := os.MkdirAll(filepath.Dir(destDir), 0755); err != nil {
		return "", "", err
	}

	// 1. Check if local directory dependency (starts with ., /, or Windows drive letter)
	isLocal := strings.HasPrefix(pkgPath, ".") || strings.HasPrefix(pkgPath, "/") || (len(pkgPath) >= 2 && pkgPath[1] == ':')
	if isLocal {
		srcPath := pkgPath
		if !filepath.IsAbs(srcPath) {
			srcPath = filepath.Join(projectRoot, pkgPath)
		}
		srcPath, sErr := filepath.Abs(srcPath)
		if sErr != nil {
			return "", "", sErr
		}
		fi, sErr := os.Stat(srcPath)
		if sErr != nil || !fi.IsDir() {
			return "", "", fmt.Errorf("local package path '%s' does not exist or is not a directory", pkgPath)
		}
		if err := copyDir(srcPath, destDir); err != nil {
			return "", "", err
		}
		sum, _ := mod.ComputeDirChecksum(destDir)
		return "local", sum, nil
	}

	// 2. Git repository dependency (e.g. github.com/user/repo)
	gitURL := pkgPath
	if !strings.HasPrefix(gitURL, "https://") && !strings.HasPrefix(gitURL, "http://") && !strings.HasPrefix(gitURL, "git@") {
		gitURL = "https://" + gitURL
	}

	cloneArgs := []string{"clone", "--depth", "1"}
	if version != "" && version != "latest" {
		cloneArgs = append(cloneArgs, "-b", version)
	}
	cloneArgs = append(cloneArgs, gitURL, destDir)

	cmd := exec.Command("git", cloneArgs...)
	_, err = cmd.CombinedOutput()
	if err != nil {
		// Fallback: try clone without depth if tag checkout failed
		fallbackCmd := exec.Command("git", "clone", gitURL, destDir)
		fOut, fErr := fallbackCmd.CombinedOutput()
		if fErr != nil {
			return "", "", fmt.Errorf("git clone failed (%s): %s", fErr, string(fOut))
		}
		if version != "" && version != "latest" {
			coCmd := exec.Command("git", "-C", destDir, "checkout", version)
			_ = coCmd.Run()
		}
	}

	// Capture exact commit hash
	commitCmd := exec.Command("git", "-C", destDir, "rev-parse", "HEAD")
	if commitBytes, cErr := commitCmd.Output(); cErr == nil {
		commit = strings.TrimSpace(string(commitBytes))
	}

	// Remove .git directory to keep cache clean
	_ = os.RemoveAll(filepath.Join(destDir, ".git"))

	// Compute content checksum
	sum, err := mod.ComputeDirChecksum(destDir)
	if err != nil {
		return commit, "", err
	}

	return commit, sum, nil
}

// copyDir recursively copies all files from src to dst.
func copyDir(src string, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			base := info.Name()
			if base == ".git" {
				return filepath.SkipDir
			}
			return os.MkdirAll(target, 0755)
		}
		return copyFile(path, target)
	})
}

func copyFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
}
