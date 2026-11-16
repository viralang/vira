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

// localDepPath writes an absolute local directory as a dependency path
// relative to the project root. The result always starts with "./" or "../"
// (or is absolute when no relative form exists), so it can never be mistaken
// for a git host such as "libs/leaf" → https://libs/leaf/.
func localDepPath(projectRoot string, abs string) string {
	rel, err := filepath.Rel(projectRoot, abs)
	if err != nil {
		return abs
	}
	rel = filepath.ToSlash(rel)
	if rel == "." {
		return "."
	}
	if rel == ".." || strings.HasPrefix(rel, "../") {
		return rel
	}
	return "./" + rel
}

// absLocalPath resolves a local dependency path against base.
func absLocalPath(base string, pkgPath string) string {
	p := filepath.FromSlash(pkgPath)
	if !filepath.IsAbs(p) {
		p = filepath.Join(base, p)
	}
	if abs, err := filepath.Abs(p); err == nil {
		return abs
	}
	return filepath.Clean(p)
}

// normalizeDepPath returns the canonical form of a dependency path as written
// in vira.mod / vira.lock: relative local paths are rewritten with
// localDepPath, absolute local paths and remote paths are kept as given.
func normalizeDepPath(projectRoot string, pkgPath string) string {
	if !isLocalPath(pkgPath) || filepath.IsAbs(filepath.FromSlash(pkgPath)) || (len(pkgPath) >= 2 && pkgPath[1] == ':') {
		return pkgPath
	}
	return localDepPath(projectRoot, absLocalPath(projectRoot, pkgPath))
}

// isWithin reports whether path is root or lies inside root.
func isWithin(root string, path string) bool {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel))
}

// readModuleName returns the `module` declared in dir/vira.mod (or vex.mod), or "".
func readModuleName(dir string) string {
	for _, name := range []string{"vira.mod", "vex.mod"} {
		if data, err := os.ReadFile(filepath.Join(dir, name)); err == nil {
			if m, err := mod.ParseManifest(string(data)); err == nil {
				return m.Module
			}
			return ""
		}
	}
	return ""
}

// readManifest reads <dir>/vira.mod (or vex.mod) and returns it with its path.
func readManifest(dir string) (*mod.Manifest, string, error) {
	manifestPath := filepath.Join(dir, "vira.mod")
	if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
		manifestPath = filepath.Join(dir, "vex.mod")
	}
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, manifestPath, fmt.Errorf("could not find vira.mod in %s: %w", dir, err)
	}
	m, err := mod.ParseManifest(string(data))
	if err != nil {
		return nil, manifestPath, err
	}
	return m, manifestPath, nil
}

// depRef is one dependency known to import resolution.
type depRef struct {
	path   string
	module string
}

// depNameMatch reports whether name refers to the dependency: by its declared
// module name, or (fallback) by the last part of its path/URL.
func depNameMatch(d depRef, name string) (byModule bool, byPath bool) {
	if d.module != "" && (d.module == name || filepath.Base(filepath.FromSlash(d.module)) == name) {
		byModule = true
	}
	depBase := filepath.Base(filepath.Clean(filepath.FromSlash(d.path)))
	if depBase == name || strings.HasSuffix(filepath.ToSlash(d.path), "/"+name) {
		byPath = true
	}
	return
}

// appendDepCandidates adds candidate directories/files for a single dependency
// (including sub-package paths like `import mypkg/sub`). When wantModule is
// true only module-name matches are added, otherwise only path matches.
func appendDepCandidates(candidates []string, checkRoot string, d depRef, impTarget string, wantModule bool) []string {
	depDir := PackageDestDir(filepath.Join(checkRoot, ".vira", "pkg"), d.path)
	vendorDir := PackageDestDir(filepath.Join(checkRoot, "vendor"), d.path)

	matches := func(name string) bool {
		byModule, byPath := depNameMatch(d, name)
		if wantModule {
			return byModule
		}
		return byPath && !byModule
	}

	// Direct package match (e.g. `import odbc` matching `https://github.com/viralang/odbc`)
	if matches(impTarget) {
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
		if sub != "" && matches(head) {
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
//  1. Relative to importing file's directory
//  2. Project root's src/ directory
//  3. Project root's root directory
//  4. Project vendor/ directory
//  5. Project .vira/pkg/ local cache (direct, transitive locked, and sub-packages),
//     matching a dependency's declared module name first and its path/URL second
//  6. Standard library packages
//  7. Global ~/.vir/pkg/ cache
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

		var deps []depRef
		seen := map[string]int{}
		addDep := func(path string, module string) {
			if i, ok := seen[path]; ok {
				if deps[i].module == "" {
					deps[i].module = module
				}
				return
			}
			seen[path] = len(deps)
			deps = append(deps, depRef{path: path, module: module})
		}

		// Root manifest dependencies
		if m, _, err := readManifest(checkRoot); err == nil {
			if m.Module == impTarget || filepath.Base(m.Module) == impTarget {
				candidates = append(candidates, filepath.Join(checkRoot, "src"), checkRoot)
			}
			for _, dep := range m.Dependencies {
				addDep(dep.Path, "")
			}
		}

		// vira.lock, so transitive dependencies installed in .vira/pkg are resolved
		if lf, err := mod.ReadLockfile(filepath.Join(checkRoot, "vira.lock")); err == nil && lf != nil {
			for _, pkgEntry := range lf.Packages {
				addDep(pkgEntry.Path, pkgEntry.Module)
			}
		}

		// If the importing file itself lives inside a cached/vendored package with its own vira.mod,
		// also inspect that package's vira.mod dependencies.
		if fileDir != "" {
			if subRoot, subMod, err := mod.FindProjectRoot(fileDir); err == nil && subMod != nil && subRoot != checkRoot {
				for _, dep := range subMod.Dependencies {
					addDep(dep.Path, "")
				}
			}
		}

		// Older lock files have no module name: read it from the package itself.
		for i := range deps {
			if deps[i].module == "" && !IsStandardPackage(deps[i].path) {
				for _, base := range []string{"vendor", filepath.Join(".vira", "pkg")} {
					if name := readModuleName(PackageDestDir(filepath.Join(checkRoot, base), deps[i].path)); name != "" {
						deps[i].module = name
						break
					}
				}
			}
		}

		for _, wantModule := range []bool{true, false} {
			for _, d := range deps {
				candidates = appendDepCandidates(candidates, checkRoot, d, impTarget, wantModule)
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

// requirement records who first asked for a dependency, and at which version.
type requirement struct {
	version string
	by      string
}

// resolver walks a project's dependency graph for Add, Install and Vendor.
// It fetches what is missing, verifies what vira.lock pins, and refuses
// version conflicts between packages.
type resolver struct {
	m           *Manager
	projectRoot string
	lockfile    *mod.Lockfile
	pins        map[string]string      // dependency key → version pinned in the root vira.mod
	required    map[string]requirement // dependency key → first requirement seen
	destOwner   map[string]string      // cache folder → dependency key installed there
	installed   int
}

const rootRequester = "the root vira.mod"

func newResolver(m *Manager, projectRoot string, manifest *mod.Manifest, lockfile *mod.Lockfile) *resolver {
	r := &resolver{
		m:           m,
		projectRoot: projectRoot,
		lockfile:    lockfile,
		pins:        map[string]string{},
		required:    map[string]requirement{},
		destOwner:   map[string]string{},
	}
	for _, dep := range manifest.Dependencies {
		if IsStandardPackage(dep.Path) {
			continue
		}
		r.pins[r.key(dep.Path)] = dep.Version
	}
	return r
}

// key identifies a dependency independently of how it was spelled: the
// absolute directory for a local package, the path/URL for a remote one.
func (r *resolver) key(pkgPath string) string {
	if isLocalPath(pkgPath) {
		return "local:" + absLocalPath(r.projectRoot, pkgPath)
	}
	return pkgPath
}

func displayVersion(v string) string {
	if v == "" {
		return "latest"
	}
	return v
}

// claimDest makes sure no two different packages are installed into the same cache folder.
func (r *resolver) claimDest(key string, path string, destDir string) error {
	clean := filepath.Clean(destDir)
	if owner, ok := r.destOwner[clean]; ok && owner != key {
		return fmt.Errorf("two different packages would both be installed in %s: '%s' and '%s'; rename one of them",
			clean, strings.TrimPrefix(owner, "local:"), path)
	}
	r.destOwner[clean] = key
	return nil
}

// installRoot installs every non-standard dependency of the root manifest and,
// recursively, all of their dependencies.
func (r *resolver) installRoot(manifest *mod.Manifest) error {
	type rootDep struct {
		path, version, dest string
	}
	var roots []rootDep
	for _, dep := range manifest.Dependencies {
		if IsStandardPackage(dep.Path) {
			continue
		}
		k := r.key(dep.Path)
		r.required[k] = requirement{version: dep.Version, by: rootRequester}
		dest := PackageDestDir(ProjectCacheDir(r.projectRoot), dep.Path)
		if err := r.claimDest(k, dep.Path, dest); err != nil {
			return err
		}
		roots = append(roots, rootDep{dep.Path, dep.Version, dest})
	}
	for _, d := range roots {
		if err := r.ensure(d.path, d.version, d.dest, nil, ""); err != nil {
			return fmt.Errorf("failed to install '%s': %w", d.path, err)
		}
	}
	for _, d := range roots {
		origin, confine := r.origins(d.path, d.dest, "")
		if err := r.walk(d.path, origin, confine, d.dest); err != nil {
			return err
		}
	}
	return nil
}

// origins returns the directory a package's relative dependencies resolve
// against, and the folder they must stay inside ("" = unrestricted).
// A package from the user's own disk resolves against its source folder.
// A git package resolves against its own copy in the cache and may not
// reach outside it.
func (r *resolver) origins(pkgPath string, destDir string, parentConfine string) (origin string, confine string) {
	if isLocalPath(pkgPath) {
		return absLocalPath(r.projectRoot, pkgPath), parentConfine
	}
	abs := absLocalPath(r.projectRoot, destDir)
	return abs, abs
}

// ensure makes destDir hold pkgPath at the version vira.lock pins (fetching it
// if needed) and records it in the lockfile. A package fetched at a locked
// commit must match the locked checksum; its lock entry is never rewritten.
func (r *resolver) ensure(pkgPath string, version string, destDir string, subLock *mod.Lockfile, subLockPath string) error {
	locked := r.lockfile.Find(pkgPath)
	inRootLock := locked != nil
	if locked == nil && subLock != nil {
		if l := subLock.Find(subLockPath); l != nil {
			cp := *l
			cp.Path = pkgPath
			locked = &cp
		}
	}

	if _, err := os.Stat(destDir); err == nil && locked != nil && locked.Checksum != "" {
		if sum, cErr := mod.ComputeDirChecksum(destDir); cErr == nil && sum == locked.Checksum {
			if !inRootLock {
				entry := *locked
				entry.Version = version
				if entry.Module == "" {
					entry.Module = readModuleName(destDir)
				}
				r.lockfile.Set(entry)
			}
			return nil
		}
	}

	// Install reproduces vira.lock: fetch the locked commit, not whatever
	// the branch points at today. `vira add` is what moves a lock forward.
	wantCommit := ""
	if locked != nil && locked.Version == version && locked.Commit != "" && locked.Commit != "local" {
		wantCommit = locked.Commit
	}
	commit, checksum, err := r.m.fetchAt(r.projectRoot, pkgPath, version, destDir, wantCommit)
	if err != nil {
		return err
	}
	if wantCommit != "" && locked.Checksum != "" && checksum != locked.Checksum {
		_ = os.RemoveAll(destDir)
		return fmt.Errorf("checksum of commit %s is %s, but vira.lock expects %s", commit, checksum, locked.Checksum)
	}
	r.installed++
	if wantCommit != "" && inRootLock {
		return nil // fetched exactly what the lock says; leave the entry alone
	}
	r.lockfile.Set(mod.LockedPackage{
		Path:     pkgPath,
		Version:  version,
		Commit:   commit,
		Checksum: checksum,
		Module:   readModuleName(destDir),
	})
	return nil
}

// walk inspects pkgDir/vira.mod and installs its dependencies recursively.
// origin is where its relative dependencies resolve; confine, when set, is
// the folder they must stay inside.
func (r *resolver) walk(pkgPath string, origin string, confine string, pkgDir string) error {
	subManifest, _, err := readManifest(pkgDir)
	if err != nil {
		return nil // package has no (valid) manifest of its own
	}
	requester := subManifest.Module
	if requester == "" {
		requester = pkgPath
	}
	subLock, _ := mod.ReadLockfile(filepath.Join(pkgDir, "vira.lock"))
	projectAbs := absLocalPath(r.projectRoot, ".")

	for _, subDep := range subManifest.Dependencies {
		if IsStandardPackage(subDep.Path) {
			continue
		}

		resolvedPath := subDep.Path
		if isLocalPath(subDep.Path) {
			if confine != "" && (filepath.IsAbs(filepath.FromSlash(subDep.Path)) || strings.HasPrefix(subDep.Path, "/") ||
				(len(subDep.Path) >= 2 && subDep.Path[1] == ':')) {
				return fmt.Errorf("package '%s' depends on the absolute path '%s'; a downloaded package may only use folders inside itself",
					requester, subDep.Path)
			}
			abs := absLocalPath(origin, subDep.Path)
			if confine != "" && !isWithin(confine, abs) {
				return fmt.Errorf("package '%s' depends on '%s', which is outside the package; a downloaded package may only use folders inside itself",
					requester, subDep.Path)
			}
			if abs == projectAbs {
				continue // a dependency back onto the project itself
			}
			resolvedPath = localDepPath(r.projectRoot, abs)
		}

		k := r.key(resolvedPath)
		if _, pinned := r.pins[k]; pinned {
			continue // the root vira.mod decides; it is installed from there
		}
		if prev, ok := r.required[k]; ok {
			if displayVersion(prev.version) != displayVersion(subDep.Version) {
				return fmt.Errorf("version conflict for '%s': '%s' requires %s but '%s' requires %s; add the version you want to the deps of the root vira.mod",
					resolvedPath, prev.by, displayVersion(prev.version), requester, displayVersion(subDep.Version))
			}
			continue // already installed (also breaks cycles)
		}
		r.required[k] = requirement{version: subDep.Version, by: requester}

		destDir := PackageDestDir(ProjectCacheDir(r.projectRoot), resolvedPath)
		if err := r.claimDest(k, resolvedPath, destDir); err != nil {
			return err
		}
		if err := r.ensure(resolvedPath, subDep.Version, destDir, subLock, subDep.Path); err != nil {
			return fmt.Errorf("failed to install transitive dependency '%s' (required by '%s'): %w", subDep.Path, requester, err)
		}

		var childOrigin, childConfine string
		if isLocalPath(resolvedPath) {
			childOrigin, childConfine = absLocalPath(r.projectRoot, resolvedPath), confine
		} else {
			childOrigin, childConfine = r.origins(resolvedPath, destDir, confine)
		}
		if err := r.walk(resolvedPath, childOrigin, childConfine, destDir); err != nil {
			return err
		}
	}
	return nil
}

// Add fetches a package (and recursively all of its transitive dependencies),
// updating vira.mod and vira.lock.
func (m *Manager) Add(projectRoot string, pkgPath string, version string) (*mod.LockedPackage, error) {
	manifest, manifestPath, err := readManifest(projectRoot)
	if err != nil {
		return nil, err
	}

	lockPath := filepath.Join(projectRoot, "vira.lock")
	lockfile, err := mod.ReadLockfile(lockPath)
	if err != nil {
		return nil, err
	}

	pkgPath = normalizeDepPath(projectRoot, pkgPath)
	destDir := PackageDestDir(ProjectCacheDir(projectRoot), pkgPath)
	commit, checksum, err := m.fetch(projectRoot, pkgPath, version, destDir)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch package '%s': %w", pkgPath, err)
	}

	if version == "" {
		version = "latest"
	}

	manifest.AddDependency(pkgPath, version)
	locked := mod.LockedPackage{
		Path:     pkgPath,
		Version:  version,
		Commit:   commit,
		Checksum: checksum,
		Module:   readModuleName(destDir),
	}
	lockfile.Set(locked)

	// Resolve the whole graph again so a conflict between the new package and
	// an existing one is caught before anything is saved.
	if err := newResolver(m, projectRoot, manifest, lockfile).installRoot(manifest); err != nil {
		return nil, err
	}

	if err := manifest.Save(manifestPath); err != nil {
		return nil, fmt.Errorf("failed to save manifest: %w", err)
	}
	if err := lockfile.Save(lockPath); err != nil {
		return nil, fmt.Errorf("failed to save lockfile: %w", err)
	}

	return &locked, nil
}

// Install reads vira.mod and vira.lock, fetching any missing direct and transitive dependencies.
func (m *Manager) Install(projectRoot string) (int, error) {
	manifest, _, err := readManifest(projectRoot)
	if err != nil {
		return 0, err
	}

	lockPath := filepath.Join(projectRoot, "vira.lock")
	lockfile, err := mod.ReadLockfile(lockPath)
	if err != nil {
		return 0, err
	}

	r := newResolver(m, projectRoot, manifest, lockfile)
	if err := r.installRoot(manifest); err != nil {
		return r.installed, err // vira.lock is left untouched
	}

	if err := lockfile.Save(lockPath); err != nil {
		return r.installed, fmt.Errorf("failed to save lockfile: %w", err)
	}
	return r.installed, nil
}

// Remove deletes a dependency from manifest, lockfile, and local project cache.
func (m *Manager) Remove(projectRoot string, pkgPath string) error {
	manifest, manifestPath, err := readManifest(projectRoot)
	if err != nil {
		return err
	}

	if !manifest.RemoveDependency(pkgPath) {
		normalized := normalizeDepPath(projectRoot, pkgPath)
		if normalized == pkgPath || !manifest.RemoveDependency(normalized) {
			return fmt.Errorf("package '%s' is not declared in vira.mod", pkgPath)
		}
		pkgPath = normalized
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

// Vendor copies all project dependencies, direct and transitive, into
// <projectRoot>/vendor. Anything missing from the cache is first fetched at
// the commit vira.lock pins; if that fails, so does Vendor.
func (m *Manager) Vendor(projectRoot string) (int, error) {
	manifest, _, err := readManifest(projectRoot)
	if err != nil {
		return 0, err
	}

	if _, err := m.Install(projectRoot); err != nil {
		return 0, fmt.Errorf("could not restore dependencies before vendoring: %w", err)
	}

	lf, err := mod.ReadLockfile(filepath.Join(projectRoot, "vira.lock"))
	if err != nil {
		return 0, err
	}

	var paths []string
	for _, dep := range manifest.Dependencies {
		if !IsStandardPackage(dep.Path) {
			paths = append(paths, dep.Path)
		}
	}
	for _, locked := range lf.Packages {
		if !IsStandardPackage(locked.Path) {
			paths = append(paths, locked.Path)
		}
	}

	vendorBase := VendorDir(projectRoot)
	count := 0
	vendored := make(map[string]bool)
	for _, p := range paths {
		if vendored[p] {
			continue
		}
		srcDir := PackageDestDir(ProjectCacheDir(projectRoot), p)
		if fi, err := os.Stat(srcDir); err != nil || !fi.IsDir() {
			return count, fmt.Errorf("failed to vendor '%s': not in the package cache after install (%s)", p, srcDir)
		}
		targetVendor := PackageDestDir(vendorBase, p)
		_ = os.RemoveAll(targetVendor)
		if err := copyDir(srcDir, targetVendor); err != nil {
			return count, fmt.Errorf("failed to vendor '%s': %w", p, err)
		}
		vendored[p] = true
		count++
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
