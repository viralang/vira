param(
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"

$DefaultVersion = "v0.5.12"
$ViraVersion = if ($Version) { $Version } elseif ($env:VIRA_VERSION) { $env:VIRA_VERSION } else { $DefaultVersion }
if (-not $ViraVersion.StartsWith("v")) {
    $ViraVersion = "v$ViraVersion"
}
$InstallDir  = "$env:USERPROFILE\.vir"
$BinDir      = "$InstallDir\bin"
$RuntimeDir  = "$InstallDir\runtime"
$LibDir      = "$InstallDir\lib"
$ExamplesDir = "$InstallDir\examples"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Vira Programming Language - Windows Installer ($ViraVersion)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Create target directories
Write-Host "[1/5] Creating installation directories in $InstallDir..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $LibDir | Out-Null
New-Item -ItemType Directory -Force -Path $ExamplesDir | Out-Null

# 2. Locate, Extract or Download vira.exe
Write-Host "[2/5] Installing Vira compiler binary..." -ForegroundColor Yellow

# Terminate any background instances (e.g., active LSP server) holding a file lock on $BinDir\vira.exe
cmd.exe /c "taskkill /F /IM vira.exe >nul 2>&1"
cmd.exe /c "taskkill /F /IM vex.exe >nul 2>&1"
Start-Sleep -Milliseconds 300

$SourceExe = $null

# Check local compiled binaries
$LocalCandidates = @(
    ".\vira.exe",
    ".\bin\vira.exe",
    "..\vira.exe",
    ".\vex.exe",
    ".\bin\vex.exe",
    "..\vex.exe"
)
foreach ($cand in $LocalCandidates) {
    if (Test-Path $cand) {
        $SourceExe = (Resolve-Path $cand).Path
        break
    }
}

# Check local release packages
if (-not $SourceExe) {
    $LocalPackages = @(
        ".\dist\vira-$ViraVersion-windows-amd64.zip",
        "..\dist\vira-$ViraVersion-windows-amd64.zip",
        ".\dist\vex-$ViraVersion-windows-amd64.zip",
        "..\dist\vex-$ViraVersion-windows-amd64.zip"
    )
    foreach ($pkg in $LocalPackages) {
        if (Test-Path $pkg) {
            $TempDist = "$env:TEMP\vira-dist-$(Get-Random)"
            Expand-Archive -Path $pkg -DestinationPath $TempDist -Force
            $found = Get-ChildItem -Path $TempDist -Filter "vira.exe" -Recurse | Select-Object -First 1
            if (-not $found) {
                $found = Get-ChildItem -Path $TempDist -Filter "vex.exe" -Recurse | Select-Object -First 1
            }
            if ($found) {
                $SourceExe = $found.FullName
                break
            }
        }
    }
}

function Install-BinaryFile($src, $dst) {
    if (Test-Path $dst) {
        try {
            Copy-Item -Path $src -Destination $dst -Force -ErrorAction Stop
            return
        } catch {
            # File is locked by a running process (e.g. IDE LSP client).
            # On NTFS, open executables can still be renamed, freeing up the target path.
            $oldDst = "$dst.old"
            if (Test-Path $oldDst) {
                Remove-Item -Force $oldDst -ErrorAction SilentlyContinue
            }
            try {
                Move-Item -Path $dst -Destination $oldDst -Force -ErrorAction Stop
                Copy-Item -Path $src -Destination $dst -Force -ErrorAction Stop
            } catch {
                cmd.exe /c "taskkill /F /IM $(Split-Path $dst -Leaf) >nul 2>&1"
                Start-Sleep -Milliseconds 250
                Copy-Item -Path $src -Destination $dst -Force
            }
            Remove-Item -Force $oldDst -ErrorAction SilentlyContinue
        }
    } else {
        Copy-Item -Path $src -Destination $dst -Force
    }
}

if ($SourceExe) {
    Install-BinaryFile -src $SourceExe -dst "$BinDir\vira.exe"
    Install-BinaryFile -src $SourceExe -dst "$BinDir\vex.exe"
    Write-Host "  ✓ Installed local binary from $SourceExe" -ForegroundColor Green
} else {
    Write-Host "  = Downloading vira.exe ($ViraVersion)..." -ForegroundColor Gray
    $Downloaded = $false
    $DownloadUrls = @(
        "https://github.com/viralang/vira/releases/download/$ViraVersion/vira-$ViraVersion-windows-amd64.zip",
        "https://github.com/viralang/vira/releases/download/$ViraVersion/vex-$ViraVersion-windows-amd64.zip"
    )
    $ZipPath = "$env:TEMP\vira-install.zip"
    $ExtractPath = "$env:TEMP\vira-extract-$(Get-Random)"

    foreach ($url in $DownloadUrls) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $ZipPath -UseBasicParsing -TimeoutSec 30
            if (Test-Path $ZipPath) {
                Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force
                $found = Get-ChildItem -Path $ExtractPath -Filter "vira.exe" -Recurse | Select-Object -First 1
                if (-not $found) {
                    $found = Get-ChildItem -Path $ExtractPath -Filter "vex.exe" -Recurse | Select-Object -First 1
                }
                if ($found) {
                    Install-BinaryFile -src $found.FullName -dst "$BinDir\vira.exe"
                    Install-BinaryFile -src $found.FullName -dst "$BinDir\vex.exe"
                    Write-Host "  ✓ Downloaded and installed vira.exe" -ForegroundColor Green
                    $Downloaded = $true
                    break
                }
            }
        } catch {
            # Try next URL
        } finally {
            if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
            if (Test-Path $ExtractPath) { Remove-Item -Recurse -Force $ExtractPath }
        }
    }

    if (-not $Downloaded) {
        Write-Warning "Could not download pre-built release package."
        if (Get-Command go -ErrorAction SilentlyContinue) {
            Write-Host "  = Building vira.exe from source via Go..." -ForegroundColor Cyan
            go build -ldflags="-s -w" -o "$BinDir\vira.exe" .\cmd\vira
            Copy-Item -Path "$BinDir\vira.exe" -Destination "$BinDir\vex.exe" -Force
            Write-Host "  ✓ Built vira.exe from source via Go" -ForegroundColor Green
        } else {
            Write-Host "  ! Please build from source using: go build -o $BinDir\vira.exe .\cmd\vira" -ForegroundColor Yellow
        }
    }
}

# 3. Copy Runtime Headers & Examples
Write-Host "[3/5] Deploying runtime headers and examples..." -ForegroundColor Yellow
if (Test-Path ".\runtime\vira_rt.h") {
    Copy-Item -Path ".\runtime\*.h" -Destination $RuntimeDir -Force
    Write-Host "  ✓ Copied runtime headers to $RuntimeDir" -ForegroundColor Green
} elseif (Test-Path "..\runtime\vira_rt.h") {
    Copy-Item -Path "..\runtime\*.h" -Destination $RuntimeDir -Force
    Write-Host "  ✓ Copied runtime headers to $RuntimeDir" -ForegroundColor Green
} else {
    Write-Host "  ✓ Embedded runtime headers will auto-extract on first run" -ForegroundColor Green
}

if (Test-Path ".\examples") {
    Copy-Item -Path ".\examples\*" -Destination $ExamplesDir -Recurse -Force
    Write-Host "  ✓ Copied runnable examples to $ExamplesDir" -ForegroundColor Green
} elseif (Test-Path "..\examples") {
    Copy-Item -Path "..\examples\*" -Destination $ExamplesDir -Recurse -Force
    Write-Host "  ✓ Copied runnable examples to $ExamplesDir" -ForegroundColor Green
}

if (Test-Path ".\LANGUAGE_GUIDE.md") {
    Copy-Item -Path ".\LANGUAGE_GUIDE.md" -Destination "$InstallDir\LANGUAGE_GUIDE.md" -Force
}
if (Test-Path ".\README.md") {
    Copy-Item -Path ".\README.md" -Destination "$InstallDir\README.md" -Force
}

# 4. Dependency Resolution (C Compiler & System Graphics)
Write-Host "[4/5] Resolving system dependencies..." -ForegroundColor Yellow
$CompilerFound = $false
$Compilers = @("gcc", "clang", "cc", "tcc")

foreach ($cc in $Compilers) {
    if (Get-Command $cc -ErrorAction SilentlyContinue) {
        $CompilerPath = (Get-Command $cc).Source
        Write-Host "  ✓ Found C backend compiler: $cc ($CompilerPath)" -ForegroundColor Green
        $CompilerFound = $true
        break
    }
}

if (-not $CompilerFound) {
    $KnownPaths = @(
        "$env:USERPROFILE\w64devkit\bin\gcc.exe",
        "C:\w64devkit\bin\gcc.exe",
        "C:\msys64\ucrt64\bin\gcc.exe",
        "C:\msys64\mingw64\bin\gcc.exe",
        "C:\Program Files\LLVM\bin\clang.exe"
    )
    foreach ($cand in $KnownPaths) {
        if (Test-Path $cand) {
            Write-Host "  ✓ Discovered local C compiler at: $cand" -ForegroundColor Green
            $CompilerFound = $true
            break
        }
    }
}

if (-not $CompilerFound) {
    Write-Host "  ! No C99 compiler found in PATH." -ForegroundColor Yellow
    Write-Host "    Vira compiles to optimized native C99/SIMD code and requires GCC or Clang." -ForegroundColor Gray
    Write-Host "    Quick Fix Options:" -ForegroundColor Cyan
    Write-Host "      1. winget install -e --id GnuWin32.GCC" -ForegroundColor White
    Write-Host "      2. winget install -e --id LLVM.LLVM" -ForegroundColor White
    Write-Host "      3. Download w64devkit from: https://github.com/skeeto/w64devkit/releases" -ForegroundColor White
} else {
    Write-Host "  ✓ Graphics libraries: GDI32, User32, WinMM, OpenGL32 (native Windows)" -ForegroundColor Green
}

# 5. Configure User Environment PATH
Write-Host "[5/5] Configuring environment PATH..." -ForegroundColor Yellow
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$BinDir;$UserPath", "User")
    $env:PATH = "$BinDir;$env:PATH"
    Write-Host "  ✓ Added $BinDir to User PATH" -ForegroundColor Green
} else {
    Write-Host "  ✓ $BinDir is already in User PATH" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  ✓ Vira was successfully installed to: $InstallDir" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open a new terminal window" -ForegroundColor White
Write-Host "  2. Verify:  vira -h" -ForegroundColor White
Write-Host "  3. Run GUI: vira run examples\gui_demo.vir" -ForegroundColor White
Write-Host ""
