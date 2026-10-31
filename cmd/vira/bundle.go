package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	goRuntime "runtime"
	"strings"
	"time"

	"github.com/viralang/vira/internal/mod"
)

func bundleCmd(args []string) {
	fs := flag.NewFlagSet("bundle", flag.ExitOnError)
	nameFlag := fs.String("name", "", "Application display name (e.g. 'My Studio')")
	idFlag := fs.String("id", "", "Application bundle identifier (e.g. 'com.company.app')")
	iconFlag := fs.String("icon", "", "Application icon path (.png, .ico, .icns)")
	assetsFlag := fs.String("assets", "", "Assets directory to bundle (default: assets/)")
	outDirFlag := fs.String("o", "dist", "Output distribution directory")
	targetFlag := fs.String("target", goRuntime.GOOS, "Target operating system (darwin, linux, windows)")
	prodFlag := fs.Bool("prod", true, "Compile with full production optimizations")
	_ = fs.Parse(args)

	start := time.Now()

	// 1. Discover project root and manifest
	projectRoot, m, _ := mod.FindProjectRoot(".")
	if projectRoot == "" {
		projectRoot, _ = os.Getwd()
	}

	// 2. Discover entrypoint
	var entrypoint string
	if fs.NArg() > 0 {
		entrypoint = fs.Arg(0)
	} else if fileExists(filepath.Join(projectRoot, "src", "main.vir")) {
		entrypoint = filepath.Join(projectRoot, "src", "main.vir")
	} else if fileExists(filepath.Join(projectRoot, "main.vir")) {
		entrypoint = filepath.Join(projectRoot, "main.vir")
	} else {
		fmt.Fprintf(os.Stderr, "error: cannot find entrypoint (expected 'src/main.vir' or 'main.vir')\n")
		fmt.Fprintf(os.Stderr, "usage: vira bundle [-name AppName] [-icon icon.png] [path/to/main.vir]\n")
		os.Exit(1)
	}

	// 3. Resolve metadata from manifest or flags
	modName := filepath.Base(projectRoot)
	if m != nil && m.Module != "" {
		modName = m.Module
	}

	appName := *nameFlag
	if appName == "" && m != nil && m.AppName != "" {
		appName = m.AppName
	}
	if appName == "" {
		appName = toTitleCase(modName)
	}

	appID := *idFlag
	if appID == "" && m != nil && m.AppID != "" {
		appID = m.AppID
	}
	if appID == "" {
		cleanID := strings.ToLower(strings.ReplaceAll(modName, "_", ""))
		appID = fmt.Sprintf("com.viralang.%s", cleanID)
	}

	appVersion := "0.1.0"
	if m != nil && m.Version != "" {
		appVersion = m.Version
	}

	iconPath := *iconFlag
	if iconPath == "" && m != nil && m.Icon != "" {
		iconPath = m.Icon
	}
	if iconPath == "" {
		for _, cand := range []string{
			filepath.Join(projectRoot, "assets", "icon.png"),
			filepath.Join(projectRoot, "assets", "icon.ico"),
			filepath.Join(projectRoot, "assets", "icon.icns"),
			filepath.Join(projectRoot, "icon.png"),
		} {
			if fileExists(cand) {
				iconPath = cand
				break
			}
		}
	}
	if iconPath != "" && !filepath.IsAbs(iconPath) {
		iconPath = filepath.Join(projectRoot, iconPath)
	}

	assetsDir := *assetsFlag
	if assetsDir == "" && m != nil && m.Assets != "" {
		assetsDir = m.Assets
	}
	if assetsDir == "" {
		cand := filepath.Join(projectRoot, "assets")
		if dirExists(cand) {
			assetsDir = cand
		}
	}
	if assetsDir != "" && !filepath.IsAbs(assetsDir) {
		assetsDir = filepath.Join(projectRoot, assetsDir)
	}

	targetOS := strings.ToLower(*targetFlag)
	outDir := *outDirFlag
	if !filepath.IsAbs(outDir) {
		outDir = filepath.Join(projectRoot, outDir)
	}
	_ = os.MkdirAll(outDir, 0755)

	fmt.Printf("📦 Bundling Native GUI Application: %s (v%s)\n", appName, appVersion)
	fmt.Printf("   • Identifier : %s\n", appID)
	fmt.Printf("   • Target OS  : %s (%s)\n", targetOS, goRuntime.GOARCH)
	fmt.Printf("   • Entrypoint : %s\n", entrypoint)
	if iconPath != "" && fileExists(iconPath) {
		fmt.Printf("   • App Icon   : %s\n", iconPath)
	}
	if assetsDir != "" && dirExists(assetsDir) {
		fmt.Printf("   • Assets Dir : %s\n", assetsDir)
	}

	// 4. Compile application
	tempBuildDir := filepath.Join(projectRoot, "build")
	_ = os.MkdirAll(tempBuildDir, 0755)

	execName := strings.ReplaceAll(appName, " ", "")
	if targetOS == "windows" {
		execName += ".exe"
	}
	tempBin := filepath.Join(tempBuildDir, execName)

	cCode, _, _ := compileVexToC99(entrypoint, false)
	cFile := filepath.Join(tempBuildDir, strings.TrimSuffix(filepath.Base(entrypoint), ".vir")+".c")
	if err := os.WriteFile(cFile, []byte(cCode), 0644); err != nil {
		fatalIO(err)
	}

	opts := compileOptions{
		prod:    *prodFlag,
		outPath: tempBin,
		guiApp:  true,
		quietRT: true,
	}

	compiler, modeDesc := invokeCBackend(cFile, tempBin, opts)
	_ = os.Remove(cFile)

	fmt.Printf("✓ Compiled standalone binary via %s [%s]\n", compiler, modeDesc)

	// 5. Package per target OS
	switch targetOS {
	case "darwin", "macos", "mac":
		bundleDarwin(appName, appID, appVersion, tempBin, iconPath, assetsDir, outDir)
	case "windows", "win":
		bundleWindows(appName, appID, appVersion, tempBin, iconPath, assetsDir, outDir)
	case "linux":
		bundleLinux(appName, appID, appVersion, tempBin, iconPath, assetsDir, outDir)
	default:
		bundleLinux(appName, appID, appVersion, tempBin, iconPath, assetsDir, outDir)
	}

	_ = os.Remove(tempBin)
	elapsed := time.Since(start).Round(time.Millisecond)
	fmt.Printf("\n✨ Successfully bundled %s in %s! Output directory: %s\n", appName, elapsed, outDir)
}

func bundleDarwin(appName, appID, version, binPath, iconPath, assetsDir, outDir string) {
	appBundle := filepath.Join(outDir, appName+".app")
	_ = os.RemoveAll(appBundle)

	macosDir := filepath.Join(appBundle, "Contents", "MacOS")
	resDir := filepath.Join(appBundle, "Contents", "Resources")
	_ = os.MkdirAll(macosDir, 0755)
	_ = os.MkdirAll(resDir, 0755)

	// 1. Copy executable
	binDest := filepath.Join(macosDir, appName)
	if err := copyFile(binPath, binDest); err != nil {
		fatalIO(err)
	}
	_ = os.Chmod(binDest, 0755)

	// 2. Generate Info.plist
	plistContent := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>%s</string>
    <key>CFBundleIdentifier</key>
    <string>%s</string>
    <key>CFBundleName</key>
    <string>%s</string>
    <key>CFBundleDisplayName</key>
    <string>%s</string>
    <key>CFBundleVersion</key>
    <string>%s</string>
    <key>CFBundleShortVersionString</key>
    <string>%s</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
</dict>
</plist>
`, appName, appID, appName, appName, version, version)
	_ = os.WriteFile(filepath.Join(appBundle, "Contents", "Info.plist"), []byte(plistContent), 0644)

	// 3. Copy Icon
	if iconPath != "" && fileExists(iconPath) {
		ext := strings.ToLower(filepath.Ext(iconPath))
		if ext == ".icns" {
			_ = copyFile(iconPath, filepath.Join(resDir, "AppIcon.icns"))
		} else {
			_ = copyFile(iconPath, filepath.Join(resDir, "AppIcon.png"))
			_ = copyFile(iconPath, filepath.Join(resDir, "icon.png"))
		}
	}

	// 4. Copy Assets
	if assetsDir != "" && dirExists(assetsDir) {
		_ = copyDir(assetsDir, filepath.Join(resDir, "assets"))
	}

	// 5. Create .zip archive
	zipArchive := filepath.Join(outDir, fmt.Sprintf("%s-%s-macos.zip", appName, version))
	if err := zipDirectory(appBundle, zipArchive); err == nil {
		fmt.Printf("✓ Created macOS standalone archive: %s\n", filepath.Base(zipArchive))
	}
	fmt.Printf("✓ Created macOS Application Bundle: %s\n", appBundle)
}

func bundleLinux(appName, appID, version, binPath, iconPath, assetsDir, outDir string) {
	pkgDir := filepath.Join(outDir, appName)
	_ = os.RemoveAll(pkgDir)
	_ = os.MkdirAll(pkgDir, 0755)

	// 1. Copy executable
	binDest := filepath.Join(pkgDir, appName)
	if err := copyFile(binPath, binDest); err != nil {
		fatalIO(err)
	}
	_ = os.Chmod(binDest, 0755)

	// 2. Copy Assets
	if assetsDir != "" && dirExists(assetsDir) {
		_ = copyDir(assetsDir, filepath.Join(pkgDir, "assets"))
	}

	// 3. Copy Icon
	iconName := "icon.png"
	if iconPath != "" && fileExists(iconPath) {
		_ = copyFile(iconPath, filepath.Join(pkgDir, iconName))
	}

	// 4. Generate .desktop file
	desktopContent := fmt.Sprintf(`[Desktop Entry]
Type=Application
Name=%s
Comment=Native Vira GUI Application
Exec=./%s
Icon=%s
Terminal=false
Categories=Utility;Graphics;Development;
StartupNotify=true
`, appName, appName, iconName)
	_ = os.WriteFile(filepath.Join(pkgDir, appName+".desktop"), []byte(desktopContent), 0644)

	// 5. Generate run.sh launcher
	runScript := fmt.Sprintf(`#!/bin/sh
APPDIR="$(dirname "$(readlink -f "$0")")"
cd "$APPDIR"
exec "./%s" "$@"
`, appName)
	runPath := filepath.Join(pkgDir, "run.sh")
	_ = os.WriteFile(runPath, []byte(runScript), 0755)
	_ = os.Chmod(runPath, 0755)

	// 6. Create .tar.gz archive
	tarArchive := filepath.Join(outDir, fmt.Sprintf("%s-%s-linux.tar.gz", appName, version))
	if err := tarGzDirectory(pkgDir, tarArchive); err == nil {
		fmt.Printf("✓ Created Linux standalone archive: %s\n", filepath.Base(tarArchive))
	}
	fmt.Printf("✓ Created Linux Portable Package: %s\n", pkgDir)
}

func bundleWindows(appName, appID, version, binPath, iconPath, assetsDir, outDir string) {
	pkgDir := filepath.Join(outDir, appName)
	_ = os.RemoveAll(pkgDir)
	_ = os.MkdirAll(pkgDir, 0755)

	// 1. Copy executable
	execName := appName
	if !strings.HasSuffix(strings.ToLower(execName), ".exe") {
		execName += ".exe"
	}
	binDest := filepath.Join(pkgDir, execName)
	if err := copyFile(binPath, binDest); err != nil {
		fatalIO(err)
	}

	// 2. Copy Assets
	if assetsDir != "" && dirExists(assetsDir) {
		_ = copyDir(assetsDir, filepath.Join(pkgDir, "assets"))
	}

	// 3. Copy Icon
	if iconPath != "" && fileExists(iconPath) {
		_ = copyFile(iconPath, filepath.Join(pkgDir, filepath.Base(iconPath)))
	}

	// 4. Create .zip archive
	zipArchive := filepath.Join(outDir, fmt.Sprintf("%s-%s-windows.zip", appName, version))
	if err := zipDirectory(pkgDir, zipArchive); err == nil {
		fmt.Printf("✓ Created Windows standalone archive: %s\n", filepath.Base(zipArchive))
	}
	fmt.Printf("✓ Created Windows Portable Package: %s\n", pkgDir)
}

// --- ARCHIVE & FILE UTILITIES ---

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	_ = os.MkdirAll(filepath.Dir(dst), 0755)
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
}

func copyDir(src, dst string) error {
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
			return os.MkdirAll(target, info.Mode())
		}
		return copyFile(path, target)
	})
}

func zipDirectory(srcDir, zipPath string) error {
	_ = os.Remove(zipPath)
	zipFile, err := os.Create(zipPath)
	if err != nil {
		return err
	}
	defer zipFile.Close()

	archive := zip.NewWriter(zipFile)
	defer archive.Close()

	baseDir := filepath.Dir(srcDir)

	return filepath.Walk(srcDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(baseDir, path)
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		header, err := zip.FileInfoHeader(info)
		if err != nil {
			return err
		}
		header.Name = filepath.ToSlash(rel)
		header.Method = zip.Deflate

		writer, err := archive.CreateHeader(header)
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

func tarGzDirectory(srcDir, tarPath string) error {
	_ = os.Remove(tarPath)
	tarFile, err := os.Create(tarPath)
	if err != nil {
		return err
	}
	defer tarFile.Close()

	gz := gzip.NewWriter(tarFile)
	defer gz.Close()

	tw := tar.NewWriter(gz)
	defer tw.Close()

	baseDir := filepath.Dir(srcDir)

	return filepath.Walk(srcDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(baseDir, path)
		if err != nil {
			return err
		}

		header, err := tar.FileInfoHeader(info, info.Name())
		if err != nil {
			return err
		}
		header.Name = filepath.ToSlash(rel)

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

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir()
}

func dirExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}
