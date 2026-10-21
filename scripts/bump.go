//go:build ignore

package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/viralang/vira/internal/version"
)

var (
	pkgFlag    bool
	buildFlag  bool
	dryRunFlag bool
)

func init() {
	flag.BoolVar(&pkgFlag, "package", false, "Compile binary and package all release archives in dist/")
	flag.BoolVar(&pkgFlag, "p", false, "Alias for -package")
	flag.BoolVar(&buildFlag, "build", false, "Compile local vira binary only")
	flag.BoolVar(&buildFlag, "b", false, "Alias for -build")
	flag.BoolVar(&dryRunFlag, "dry-run", false, "Preview changes without modifying files")
	flag.BoolVar(&dryRunFlag, "n", false, "Alias for -dry-run")
}

func main() {
	flag.Parse()

	args := flag.Args()
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "Usage: go run ./scripts/bump.go [flags] <new-version>\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  go run ./scripts/bump.go v0.5.10\n")
		fmt.Fprintf(os.Stderr, "  go run ./scripts/bump.go -p v0.5.10    # bump and build release packages\n")
		fmt.Fprintf(os.Stderr, "  go run ./scripts/bump.go --dry-run 0.5.10\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	rawNewVer := strings.TrimSpace(args[0])
	newVer := normalizeVersion(rawNewVer)
	newVerBare := strings.TrimPrefix(newVer, "v")

	oldVer := version.Version

	if oldVer == newVer {
		fmt.Printf("Version is already %s. No bump needed.\n", newVer)
		os.Exit(0)
	}

	fmt.Printf("====================================================\n")
	fmt.Printf("  Vira Release Bumper: %s  ->  %s\n", oldVer, newVer)
	fmt.Printf("====================================================\n\n")

	if dryRunFlag {
		fmt.Printf("[DRY-RUN MODE: No files will be changed]\n\n")
	}

	changes := []struct {
		File    string
		Pattern *regexp.Regexp
		Replace string
		Desc    string
	}{
		{
			File:    filepath.Join("internal", "version", "version.go"),
			Pattern: regexp.MustCompile(`var Version = "[^"]+"`),
			Replace: fmt.Sprintf(`var Version = "%s"`, newVer),
			Desc:    "Single Source of Truth Go version constant",
		},
		{
			File:    filepath.Join("scripts", "install.ps1"),
			Pattern: regexp.MustCompile(`\$DefaultVersion = "[^"]+"`),
			Replace: fmt.Sprintf(`$DefaultVersion = "%s"`, newVer),
			Desc:    "PowerShell installer default version",
		},
		{
			File:    filepath.Join("scripts", "install.sh"),
			Pattern: regexp.MustCompile(`DEFAULT_VERSION="[^"]+"`),
			Replace: fmt.Sprintf(`DEFAULT_VERSION="%s"`, newVer),
			Desc:    "Shell installer default version",
		},
		{
			File:    "README.md",
			Pattern: regexp.MustCompile(`badge/version-[^"-]+-blue\.svg`),
			Replace: fmt.Sprintf(`badge/version-%s-blue.svg`, newVer),
			Desc:    "README version badge",
		},
		{
			File:    "README.md",
			Pattern: regexp.MustCompile(fmt.Sprintf(`vira-%s-`, regexp.QuoteMeta(oldVer))),
			Replace: fmt.Sprintf(`vira-%s-`, newVer),
			Desc:    "README release package archive filenames",
		},
		{
			File:    "LANGUAGE_GUIDE.md",
			Pattern: regexp.MustCompile(`>\s*\*\*Version:\*\*\s*Production Toolchain \([^)]+\)`),
			Replace: fmt.Sprintf(`> **Version:** Production Toolchain (%s)`, newVer),
			Desc:    "LANGUAGE_GUIDE.md toolchain version header",
		},
		{
			File:    filepath.Join("docs", "LANGUAGE_GUIDE.md"),
			Pattern: regexp.MustCompile(`>\s*\*\*Version:\*\*\s*Production Toolchain \([^)]+\)`),
			Replace: fmt.Sprintf(`> **Version:** Production Toolchain (%s)`, newVer),
			Desc:    "docs/LANGUAGE_GUIDE.md toolchain version header",
		},
	}

	updatedCount := 0
	for _, c := range changes {
		if _, err := os.Stat(c.File); os.IsNotExist(err) {
			continue
		}

		contentBytes, err := os.ReadFile(c.File)
		if err != nil {
			fatal("Failed to read %s: %v", c.File, err)
		}
		content := string(contentBytes)

		if !c.Pattern.MatchString(content) {
			fmt.Printf("  - [SKIP] %s: pattern not matched for %s\n", c.File, c.Desc)
			continue
		}

		newContent := c.Pattern.ReplaceAllString(content, c.Replace)
		if newContent == content {
			fmt.Printf("  = [NOOP] %s (%s)\n", c.File, c.Desc)
			continue
		}

		if !dryRunFlag {
			if err := os.WriteFile(c.File, []byte(newContent), 0644); err != nil {
				fatal("Failed to write %s: %v", c.File, err)
			}
		}

		fmt.Printf("  ✓ [UPDATED] %s (%s)\n", c.File, c.Desc)
		updatedCount++
	}

	// Also check if root VERSION file exists or create it
	versionFilePath := "VERSION"
	if !dryRunFlag {
		_ = os.WriteFile(versionFilePath, []byte(newVerBare+"\n"), 0644)
		fmt.Printf("  ✓ [UPDATED] %s (plain-text version file)\n", versionFilePath)
		updatedCount++
	}

	fmt.Printf("\nDone! Updated %d location(s) across the project.\n\n", updatedCount)

	// Build local binary if requested
	if (buildFlag || pkgFlag) && !dryRunFlag {
		fmt.Printf("--- Building local vira binary [%s] ---\n", newVer)
		cmd := exec.Command("go", "build", "-ldflags=-s -w", "-o", "vira.exe", "./cmd/vira")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fatal("failed to build vira binary: %v", err)
		}
		fmt.Printf("  ✓ Compiled vira.exe successfully\n\n")
	}

	// Run full packaging if requested
	if pkgFlag && !dryRunFlag {
		fmt.Printf("--- Packaging cross-platform releases in dist/ [%s] ---\n", newVer)
		cmd := exec.Command("go", "run", "./scripts/package.go", "-version", newVer)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fatal("failed to package releases: %v", err)
		}
		fmt.Printf("\n")
	}

	// Print Git helper instructions
	fmt.Printf("====================================================\n")
	fmt.Printf("  Ready to publish %s! Next Git steps:\n", newVer)
	fmt.Printf("====================================================\n")
	fmt.Printf("  git add -A\n")
	fmt.Printf("  git commit -m \"chore(release): bump version to %s\"\n", newVer)
	fmt.Printf("  git tag %s\n", newVer)
	fmt.Printf("  git push origin main --tags\n\n")
}

func normalizeVersion(raw string) string {
	raw = strings.TrimSpace(raw)
	if !strings.HasPrefix(raw, "v") {
		return "v" + raw
	}
	return raw
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "Error: "+format+"\n", args...)
	os.Exit(1)
}
