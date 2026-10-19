//go:build ignore

package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const version = "v0.5.9"

type Target struct {
	OS      string
	Arch    string
	ExeName string
	Archive string // "zip" or "tar.gz"
}

var targets = []Target{
	{OS: "windows", Arch: "amd64", ExeName: "vira.exe", Archive: "zip"},
	{OS: "linux", Arch: "amd64", ExeName: "vira", Archive: "tar.gz"},
	{OS: "darwin", Arch: "arm64", ExeName: "vira", Archive: "tar.gz"},
	{OS: "darwin", Arch: "amd64", ExeName: "vira", Archive: "tar.gz"},
}

func main() {
	start := time.Now()
	distDir := "dist"
	if err := os.MkdirAll(distDir, 0755); err != nil {
		fatal("failed to create dist directory: %v", err)
	}

	fmt.Printf("=== Packaging Vira Programming Language [%s] ===\n\n", version)

	for _, tgt := range targets {
		targetName := fmt.Sprintf("vira-%s-%s-%s", version, tgt.OS, tgt.Arch)
		fmt.Printf("Building %s (%s/%s)...\n", targetName, tgt.OS, tgt.Arch)

		tempDir, err := os.MkdirTemp("", targetName+"-*")
		if err != nil {
			fatal("temp dir failed: %v", err)
		}

		binDir := filepath.Join(tempDir, targetName, "bin")
		if err := os.MkdirAll(binDir, 0755); err != nil {
			fatal("mkdir bin failed: %v", err)
		}

		outBinary := filepath.Join(binDir, tgt.ExeName)

		// 1. Cross-compile Vira binary
		cmd := exec.Command("go", "build", "-ldflags=-s -w", "-o", outBinary, "./cmd/vira")
		cmd.Env = append(os.Environ(),
			"GOOS="+tgt.OS,
			"GOARCH="+tgt.Arch,
			"CGO_ENABLED=0",
		)
		if out, err := cmd.CombinedOutput(); err != nil {
			fatal("compilation failed for %s/%s:\n%s", tgt.OS, tgt.Arch, string(out))
		}

		// 2. Copy Runtime directory
		copyDir("runtime", filepath.Join(tempDir, targetName, "runtime"))

		// 3. Copy Examples
		copyDir("examples", filepath.Join(tempDir, targetName, "examples"))

		// 4. Copy Documentation & Installers
		copyFile("LANGUAGE_GUIDE.md", filepath.Join(tempDir, targetName, "LANGUAGE_GUIDE.md"))
		if _, err := os.Stat("scripts/install.ps1"); err == nil {
			copyFile("scripts/install.ps1", filepath.Join(tempDir, targetName, "install.ps1"))
		}
		if _, err := os.Stat("scripts/install.sh"); err == nil {
			copyFile("scripts/install.sh", filepath.Join(tempDir, targetName, "install.sh"))
		}

		// 5. Create Archive
		var archiveFile string
		if tgt.Archive == "zip" {
			archiveFile = filepath.Join(distDir, targetName+".zip")
			createZip(filepath.Join(tempDir, targetName), archiveFile)
		} else {
			archiveFile = filepath.Join(distDir, targetName+".tar.gz")
			createTarGz(filepath.Join(tempDir, targetName), archiveFile)
		}

		fi, _ := os.Stat(archiveFile)
		sizeMB := float64(fi.Size()) / (1024 * 1024)
		fmt.Printf("  ✓ Created %s (%.2f MB)\n", archiveFile, sizeMB)

		_ = os.RemoveAll(tempDir)
	}

	fmt.Printf("\n✓ All packages built successfully in %s (output: %s/)\n",
		time.Since(start).Round(time.Millisecond), distDir)
}

func createZip(srcDir, zipPath string) {
	outFile, err := os.Create(zipPath)
	if err != nil {
		fatal("create zip error: %v", err)
	}
	defer outFile.Close()

	w := zip.NewWriter(outFile)
	defer w.Close()

	baseDir := filepath.Base(srcDir)
	_ = filepath.Walk(srcDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(srcDir, path)
		if err != nil {
			return err
		}
		zipEntryPath := filepath.ToSlash(filepath.Join(baseDir, rel))

		header, err := zip.FileInfoHeader(info)
		if err != nil {
			return err
		}
		header.Name = zipEntryPath
		header.Method = zip.Deflate

		writer, err := w.CreateHeader(header)
		if err != nil {
			return err
		}
		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()
		_, err = io.Copy(writer, file)
		return err
	})
}

func createTarGz(srcDir, tarPath string) {
	outFile, err := os.Create(tarPath)
	if err != nil {
		fatal("create tar.gz error: %v", err)
	}
	defer outFile.Close()

	gw := gzip.NewWriter(outFile)
	defer gw.Close()

	tw := tar.NewWriter(gw)
	defer tw.Close()

	baseDir := filepath.Base(srcDir)
	_ = filepath.Walk(srcDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		rel, err := filepath.Rel(srcDir, path)
		if err != nil {
			return err
		}
		tarEntryPath := filepath.ToSlash(filepath.Join(baseDir, rel))

		header, err := tar.FileInfoHeader(info, info.Name())
		if err != nil {
			return err
		}
		header.Name = tarEntryPath

		// Mark executable files with 0755
		if !info.IsDir() && (strings.HasSuffix(tarEntryPath, "/vira") || strings.HasSuffix(tarEntryPath, ".sh")) {
			header.Mode = 0755
		}

		if err := tw.WriteHeader(header); err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()
		_, err = io.Copy(tw, file)
		return err
	})
}

func copyFile(src, dst string) {
	in, err := os.Open(src)
	if err != nil {
		return
	}
	defer in.Close()

	_ = os.MkdirAll(filepath.Dir(dst), 0755)
	out, err := os.Create(dst)
	if err != nil {
		return
	}
	defer out.Close()

	_, _ = io.Copy(out, in)
}

func copyDir(src, dst string) {
	_ = filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return nil
		}
		targetPath := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(targetPath, 0755)
		}
		copyFile(path, targetPath)
		return nil
	})
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "packager error: "+format+"\n", args...)
	os.Exit(1)
}
