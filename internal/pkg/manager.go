package pkg

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/viralang/vira/internal/mod"
	stdpkg "github.com/viralang/vira/packages"
)

// IsStandardPackage reports whether name is a package bundled with the
// compiler (net, http, json, ...). vira.mod may list those under deps with a
// version; they are never fetched.
func IsStandardPackage(name string) bool {
	if name == "" || strings.ContainsAny(name, "/\\.:@") {
		return false
	}
	fi, err := fs.Stat(stdpkg.StandardFS, name)
	return err == nil && fi.IsDir()
}

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

func isLocalPath(pkgPath string) bool {
	return strings.HasPrefix(pkgPath, ".") || strings.HasPrefix(pkgPath, "/") || (len(pkgPath) >= 2 && pkgPath[1] == ':')
}

// PackageDestDir returns the destination directory for a package within baseDir.
// For local directory dependencies (e.g. C:\path\to\pkg or ../pkg), it maps to the base folder name (e.g. "pkg").
// For remote Git dependencies (e.g. github.com/user/repo), it preserves the full repository path.
func PackageDestDir(baseDir string, pkgPath string) string {
	if isLocalPath(pkgPath) {
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

// appendDepCandidates adds candidate directories/files for a single dependency path
// (including sub-package paths like `import mypkg/sub`).
func appendDepCandidates(candidates []string, checkRoot string, depPath string, impTarget string) []string {
	depDir := PackageDestDir(filepath.Join(checkRoot, ".vira", "pkg"), depPath)
	vendorDir := PackageDestDir(filepath.Join(checkRoot, "vendor"), depPath)
	depBase := filepath.Base(filepath.Clean(filepath.FromSlash(depPath)))

	// Direct package match (e.g. `import odbc` matching `https://github.com/viralang/odbc`)
	if depBase == impTarget || strings.HasSuffix(filepath.ToSlash(depPath), "/"+impTarget) {
		candidates = append(candidates,
			filepath.Join(vendorDir, "src"),
			vendorDir,
			filepath.Join(depDir, "src"),
			depDir,
		)
	}

	// Sub-package match inside a dependency (e.g. `import mypkg/sub` where `mypkg` is a dependency)
	slashIdx := strings.IndexByte(impTarget, '/')
	if slashIdx > 0 {
		head := impTarget[:slashIdx]
		sub := impTarget[slashIdx+1:]
		if sub != "" && (depBase == head || strings.HasSuffix(filepath.ToSlash(depPath), "/"+head)) {
			for _, root := range []string{vendorDir, depDir} {
				candidates = append(candidates,
					filepath.Join(root, sub, "src"),
					filepath.Join(root, sub),
					filepath.Join(root, "src", sub),
					filepath.Join(root, sub+".vir"),
					filepath.Join(root, sub+".vex"),
					filepath.Join(root, "src", sub+".vir"),
					filepath.Join(root, "src", sub+".vex"),
				)
			}
		}
	}

	return candidates
}

// ResolveImportCandidates returns candidate file and directory paths for an import target.
// It checks in priority order:
// 1. Relative to importing file's directory
// 2. Project root's src/ directory
// 3. Project root's root directory
// 4. Project vendor/ directory
// 5. Project .vira/pkg/ local cache (including direct, transitive locked, and sub-packages)
// 6. Standard library packages
// 7. Global ~/.vir/pkg/ cache
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

		// Check root manifest dependencies for matches by base name or subpath
		manifestPath := filepath.Join(checkRoot, "vira.mod")
		if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
			manifestPath = filepath.Join(checkRoot, "vex.mod")
		}
		if mBytes, err := os.ReadFile(manifestPath); err == nil {
			if m, err := mod.ParseManifest(string(mBytes)); err == nil {
				if m.Module == impTarget || filepath.Base(m.Module) == impTarget {
					candidates = append(candidates, filepath.Join(checkRoot, "src"), checkRoot)
				}
				for _, dep := range m.Dependencies {
					candidates = appendDepCandidates(candidates, checkRoot, dep.Path, impTarget)
				}
			}
		}

		// Also check vira.lock so transitive dependencies installed in .vira/pkg are resolved
		lockPath := filepath.Join(checkRoot, "vira.lock")
		if lf, err := mod.ReadLockfile(lockPath); err == nil && lf != nil {
			for _, pkgEntry := range lf.Packages {
				candidates = appendDepCandidates(candidates, checkRoot, pkgEntry.Path, impTarget)
			}
		}

		// If the importing file itself lives inside a cached/vendored package with its own vira.mod,
		// also inspect that package's vira.mod dependencies.
		if fileDir != "" {
			if subRoot, subMod, err := mod.FindProjectRoot(fileDir); err == nil && subMod != nil && subRoot != checkRoot {
				for _, dep := range subMod.Dependencies {
					candidates = appendDepCandidates(candidates, checkRoot, dep.Path, impTarget)
				}
			}
		}
	}

	// 3. Standard library packages (packages/<impTarget>/src, packages/<impTarget>)
	stdRoots := []string{}
	for d := fileDir; d != "" && d != "." && d != "/" && d != "\\" && d != string(filepath.Separator); {
		pkgCandidate := filepath.Join(d, "packages")
		if info, err := os.Stat(pkgCandidate); err == nil && info.IsDir() {
			stdRoots = append(stdRoots, pkgCandidate)
			break
		}
		parent := filepath.Dir(d)
		if parent == d {
			break
		}
		d = parent
	}
	if checkRoot != "" {
		stdRoots = append(stdRoots, filepath.Join(checkRoot, "packages"))
	}
	if vRoot := os.Getenv("VIRA_ROOT"); vRoot != "" {
		stdRoots = append(stdRoots, filepath.Join(vRoot, "packages"))
	}
	if exe, err := os.Executable(); err == nil {
		stdRoots = append(stdRoots, filepath.Join(filepath.Dir(exe), "packages"))
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		stdRoots = append(stdRoots,
			filepath.Join(home, ".vira", "packages"),
			filepath.Join(home, ".vir", "packages"),
		)
	}

	for _, sRoot := range stdRoots {
		candidates = append(candidates,
			filepath.Join(sRoot, impTarget, "src"),
			filepath.Join(sRoot, impTarget),
			filepath.Join(sRoot, impTarget+".vir"),
			filepath.Join(sRoot, impTarget+".vex"),
		)
	}

	// 4. Global cache (~/.vir/pkg)
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		candidates = append(candidates,
			filepath.Join(home, ".vir", "pkg", impTarget),
			filepath.Join(home, ".vir", "pkg", impTarget+".vir"),
			filepath.Join(home, ".vir", "pkg", impTarget+".vex"),
		)
	}

	return candidates
}

// Add fetches a package (and recursively all of its transitive dependencies),
// updating vira.mod and vira.lock.
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

	// Recursively resolve and install transitive dependencies from the added package's vira.mod
	visited := map[string]bool{
		pkgPath:                   true,
		filepath.Clean(destDir):   true,
	}
	parentOrigin := projectRoot
	if isLocalPath(pkgPath) {
		p := pkgPath
		if !filepath.IsAbs(p) {
			p = filepath.Join(projectRoot, p)
		}
		if abs, aErr := filepath.Abs(p); aErr == nil {
			parentOrigin = abs
		}
	}
	if _, err := m.installTransitive(projectRoot, parentOrigin, destDir, lockfile, visited); err != nil {
		return nil, err
	}

	if err := lockfile.Save(lockPath); err != nil {
		return nil, fmt.Errorf("failed to save lockfile: %w", err)
	}

	return &locked, nil
}

// Install reads vira.mod and vira.lock, fetching any missing direct and transitive dependencies.
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
	visited := make(map[string]bool)

	for _, dep := range manifest.Dependencies {
		if IsStandardPackage(dep.Path) {
			continue // bundled with the compiler, nothing to fetch
		}
		visited[dep.Path] = true
		destDir := PackageDestDir(ProjectCacheDir(projectRoot), dep.Path)
		visited[filepath.Clean(destDir)] = true
		locked := lockfile.Find(dep.Path)

		alreadyVerified := false
		if _, err := os.Stat(destDir); err == nil {
			if locked != nil && locked.Checksum != "" {
				currentSum, cErr := mod.ComputeDirChecksum(destDir)
				if cErr == nil && currentSum == locked.Checksum {
					alreadyVerified = true
				}
			}
		}

		if !alreadyVerified {
			// Install reproduces vira.lock: fetch the locked commit, not whatever
			// the branch points at today. `vira add` is what moves a lock forward.
			wantCommit := ""
			if locked != nil && locked.Version == dep.Version && locked.Commit != "local" {
				wantCommit = locked.Commit
			}
			commit, checksum, err := m.fetchAt(projectRoot, dep.Path, dep.Version, destDir, wantCommit)
			if err != nil {
				return installedCount, fmt.Errorf("failed to install '%s': %w", dep.Path, err)
			}
			if wantCommit != "" && locked.Checksum != "" && checksum != locked.Checksum {
				_ = os.RemoveAll(destDir)
				return installedCount, fmt.Errorf("failed to install '%s': checksum of commit %s is %s, but vira.lock expects %s",
					dep.Path, commit, checksum, locked.Checksum)
			}

			lockfile.Set(mod.LockedPackage{
				Path:     dep.Path,
				Version:  dep.Version,
				Commit:   commit,
				Checksum: checksum,
			})
			installedCount++
		}

		parentOrigin := projectRoot
		if isLocalPath(dep.Path) {
			p := dep.Path
			if !filepath.IsAbs(p) {
				p = filepath.Join(projectRoot, p)
			}
			if abs, aErr := filepath.Abs(p); aErr == nil {
				parentOrigin = abs
			}
		}
		transCount, err := m.installTransitive(projectRoot, parentOrigin, destDir, lockfile, visited)
		if err != nil {
			return installedCount, err
		}
		installedCount += transCount
	}

	_ = lockfile.Save(lockPath)
	return installedCount, nil
}

// installTransitive inspects pkgDir/vira.mod (and pkgDir/vira.lock if present) and recursively
// installs any non-standard dependencies into ProjectCacheDir(projectRoot) and records them in lockfile.
func (m *Manager) installTransitive(projectRoot string, parentOrigin string, pkgDir string, lockfile *mod.Lockfile, visited map[string]bool) (int, error) {
	subModPath := filepath.Join(pkgDir, "vira.mod")
	if _, err := os.Stat(subModPath); os.IsNotExist(err) {
		subModPath = filepath.Join(pkgDir, "vex.mod")
	}
	subModBytes, err := os.ReadFile(subModPath)
	if err != nil {
		return 0, nil // package has no manifest of its own
	}
	subManifest, err := mod.ParseManifest(string(subModBytes))
	if err != nil {
		return 0, nil
	}

	subLock, _ := mod.ReadLockfile(filepath.Join(pkgDir, "vira.lock"))
	installedCount := 0

	for _, subDep := range subManifest.Dependencies {
		if IsStandardPackage(subDep.Path) {
			continue
		}

		resolvedPath := subDep.Path
		nextOrigin := projectRoot
		if isLocalPath(subDep.Path) {
			candidate := subDep.Path
			if !filepath.IsAbs(candidate) {
				candidate = filepath.Join(parentOrigin, subDep.Path)
			}
			if abs, aErr := filepath.Abs(candidate); aErr == nil {
				nextOrigin = abs
				if rel, rErr := filepath.Rel(projectRoot, abs); rErr == nil {
					resolvedPath = rel
				} else {
					resolvedPath = abs
				}
			}
		}

		destDir := PackageDestDir(ProjectCacheDir(projectRoot), resolvedPath)
		cleanDest := filepath.Clean(destDir)
		if visited[resolvedPath] || visited[cleanDest] {
			continue
		}
		visited[resolvedPath] = true
		visited[cleanDest] = true

		locked := lockfile.Find(resolvedPath)
		if locked == nil && subLock != nil {
			locked = subLock.Find(subDep.Path)
		}

		alreadyVerified := false
		if _, sErr := os.Stat(destDir); sErr == nil {
			if locked != nil && locked.Checksum != "" {
				if currentSum, cErr := mod.ComputeDirChecksum(destDir); cErr == nil && currentSum == locked.Checksum {
					alreadyVerified = true
				}
			}
		}

		if !alreadyVerified {
			wantCommit := ""
			if locked != nil && locked.Version == subDep.Version && locked.Commit != "local" {
				wantCommit = locked.Commit
			}
			commit, checksum, fErr := m.fetchAt(projectRoot, resolvedPath, subDep.Version, destDir, wantCommit)
			if fErr != nil {
				return installedCount, fmt.Errorf("failed to install transitive dependency '%s' (required by '%s'): %w", subDep.Path, subManifest.Module, fErr)
			}
			lockfile.Set(mod.LockedPackage{
				Path:     resolvedPath,
				Version:  subDep.Version,
				Commit:   commit,
				Checksum: checksum,
			})
			installedCount++
		} else if lockfile.Find(resolvedPath) == nil && locked != nil {
			lockfile.Set(mod.LockedPackage{
				Path:     resolvedPath,
				Version:  subDep.Version,
				Commit:   locked.Commit,
				Checksum: locked.Checksum,
			})
		}

		childCount, cErr := m.installTransitive(projectRoot, nextOrigin, destDir, lockfile, visited)
		if cErr != nil {
			return installedCount, cErr
		}
		installedCount += childCount
	}

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
	vendored := make(map[string]bool)

	for _, dep := range manifest.Dependencies {
		if IsStandardPackage(dep.Path) {
			continue
		}
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
		vendored[dep.Path] = true
		count++
	}

	// Also vendor any transitive packages recorded in vira.lock
	if lf, err := mod.ReadLockfile(filepath.Join(projectRoot, "vira.lock")); err == nil && lf != nil {
		for _, locked := range lf.Packages {
			if vendored[locked.Path] || IsStandardPackage(locked.Path) {
				continue
			}
			srcDir := PackageDestDir(ProjectCacheDir(projectRoot), locked.Path)
			if _, err := os.Stat(srcDir); err == nil {
				targetVendor := PackageDestDir(vendorBase, locked.Path)
				_ = os.RemoveAll(targetVendor)
				if err := copyDir(srcDir, targetVendor); err != nil {
					return count, fmt.Errorf("failed to vendor transitive package '%s': %w", locked.Path, err)
				}
				vendored[locked.Path] = true
				count++
			}
		}
	}

	return count, nil
}

// fetch downloads or links a package (from local directory or Git repo) to destDir.
func (m *Manager) fetch(projectRoot string, pkgPath string, version string, destDir string) (commit string, checksum string, err error) {
	return m.fetchAt(projectRoot, pkgPath, version, destDir, "")
}

// splitGitRepoAndSubpath splits a path like "github.com/org/repo/sub/pkg" into
// ("https://github.com/org/repo", "sub/pkg", true) when it contains a subpath
// inside a 3-segment host/owner/repo path.
func splitGitRepoAndSubpath(pkgPath string) (repoURL string, subPath string, ok bool) {
	trimmed := pkgPath
	scheme := "https://"
	for _, pfx := range []string{"https://", "http://"} {
		if strings.HasPrefix(trimmed, pfx) {
			scheme = pfx
			trimmed = strings.TrimPrefix(trimmed, pfx)
			break
		}
	}
	parts := strings.Split(strings.Trim(trimmed, "/"), "/")
	if len(parts) <= 3 {
		return "", "", false
	}
	repoURL = scheme + strings.Join(parts[:3], "/")
	subPath = filepath.FromSlash(strings.Join(parts[3:], "/"))
	return repoURL, subPath, true
}

// fetchAt is fetch with an optional exact commit: when wantCommit is set and
// the clone is not already at it, that commit is fetched and checked out, and
// a failure to get it is an error.
func (m *Manager) fetchAt(projectRoot string, pkgPath string, version string, destDir string, wantCommit string) (commit string, checksum string, err error) {
	// Clean destination directory
	_ = os.RemoveAll(destDir)
	if err := os.MkdirAll(filepath.Dir(destDir), 0755); err != nil {
		return "", "", err
	}

	// 1. Check if local directory dependency (starts with ., /, or Windows drive letter)
	if isLocalPath(pkgPath) {
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

	// 2. Git repository dependency (e.g. github.com/user/repo or github.com/user/repo/subpkg)
	gitURL := pkgPath
	if !strings.HasPrefix(gitURL, "https://") && !strings.HasPrefix(gitURL, "http://") && !strings.HasPrefix(gitURL, "git@") &&
		!strings.HasPrefix(gitURL, "ssh://") && !strings.HasPrefix(gitURL, "file://") {
		gitURL = "https://" + gitURL
	}

	cloneInto := destDir
	extractedSubpath := ""
	tempRepoDir := ""

	cloneArgs := []string{"clone", "--depth", "1"}
	if version != "" && version != "latest" {
		cloneArgs = append(cloneArgs, "-b", version)
	}
	cloneArgs = append(cloneArgs, gitURL, cloneInto)

	cmd := exec.Command("git", cloneArgs...)
	_, err = cmd.CombinedOutput()
	if err != nil {
		// Fallback 1: try clone without depth if tag checkout failed
		fallbackCmd := exec.Command("git", "clone", gitURL, cloneInto)
		fOut, fErr := fallbackCmd.CombinedOutput()
		if fErr == nil {
			if version != "" && version != "latest" {
				coCmd := exec.Command("git", "-C", cloneInto, "checkout", version)
				_ = coCmd.Run()
			}
		} else if repoRootURL, subRel, hasSub := splitGitRepoAndSubpath(pkgPath); hasSub {
			// Fallback 2: pkgPath points to a sub-package directory inside a parent Git repo
			tmpDir, tErr := os.MkdirTemp("", "vira_subpkg_*")
			if tErr != nil {
				return "", "", fmt.Errorf("git clone failed (%s): %s", fErr, string(fOut))
			}
			defer os.RemoveAll(tmpDir)
			tempRepoDir = tmpDir
			cloneInto = tmpDir
			extractedSubpath = subRel

			subCloneArgs := []string{"clone", "--depth", "1"}
			if version != "" && version != "latest" {
				subCloneArgs = append(subCloneArgs, "-b", version)
			}
			subCloneArgs = append(subCloneArgs, repoRootURL, cloneInto)
			if sOut, sErr := exec.Command("git", subCloneArgs...).CombinedOutput(); sErr != nil {
				return "", "", fmt.Errorf("git clone failed (%s): %s", sErr, string(sOut))
			}
		} else {
			return "", "", fmt.Errorf("git clone failed (%s): %s", fErr, string(fOut))
		}
	}

	// Capture exact commit hash
	revParse := func() string {
		out, cErr := exec.Command("git", "-C", cloneInto, "rev-parse", "HEAD").Output()
		if cErr != nil {
			return ""
		}
		return strings.TrimSpace(string(out))
	}
	commit = revParse()

	if wantCommit != "" && commit != wantCommit {
		fOut, fErr := exec.Command("git", "-C", cloneInto, "fetch", "--depth", "1", "origin", wantCommit).CombinedOutput()
		if fErr == nil {
			fOut, fErr = exec.Command("git", "-C", cloneInto, "checkout", "-q", "--detach", "FETCH_HEAD").CombinedOutput()
		}
		if fErr != nil || revParse() != wantCommit {
			_ = os.RemoveAll(destDir)
			return "", "", fmt.Errorf("could not fetch locked commit %s (%v): %s", wantCommit, fErr, strings.TrimSpace(string(fOut)))
		}
		commit = wantCommit
	}

	// If extracting a sub-package directory from a parent git repository
	if tempRepoDir != "" && extractedSubpath != "" {
		subSrc := filepath.Join(tempRepoDir, extractedSubpath)
		fi, sErr := os.Stat(subSrc)
		if sErr != nil || !fi.IsDir() {
			_ = os.RemoveAll(destDir)
			return "", "", fmt.Errorf("sub-package path '%s' not found in repository", extractedSubpath)
		}
		if cErr := copyDir(subSrc, destDir); cErr != nil {
			return "", "", cErr
		}
	} else {
		// Remove .git directory to keep cache clean
		_ = os.RemoveAll(filepath.Join(destDir, ".git"))
	}

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
			if base == ".git" || base == ".vira" {
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
