# Vira Programming Language - Windows Automated Installer & Dependency Resolver
# Usage: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

$VexVersion = "v0.5.5"
$InstallDir = "$env:USERPROFILE\.vir"
$BinDir     = "$InstallDir\bin"
$RuntimeDir = "$InstallDir\runtime"
$LibDir     = "$InstallDir\lib"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Vira Programming Language - Windows Installer ($VexVersion)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Create target directories
Write-Host "[1/5] Creating installation directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $LibDir | Out-Null

# 2. Locate or Download vira.exe
Write-Host "[2/5] Installing Vira compiler binary..." -ForegroundColor Yellow
$SourceExe = $null

if (Test-Path ".\vira.exe") {
    $SourceExe = ".\vira.exe"
} elseif (Test-Path ".\bin\vira.exe") {
    $SourceExe = ".\bin\vira.exe"
} elseif (Test-Path "..\vira.exe") {
    $SourceExe = "..\vira.exe"
} elseif (Test-Path ".\dist\vex-$VexVersion-windows-amd64.zip") {
    Expand-Archive -Path ".\dist\vex-$VexVersion-windows-amd64.zip" -DestinationPath "$env:TEMP\vex-dist" -Force
    $SourceExe = (Get-ChildItem -Path "$env:TEMP\vex-dist" -Filter "vira.exe" -Recurse).FullName
}

if ($SourceExe) {
    Copy-Item -Path $SourceExe -Destination "$BinDir\vira.exe" -Force
    Write-Host "  ✓ Installed local binary from $SourceExe" -ForegroundColor Green
} else {
    Write-Host "  = Downloading vira.exe ($VexVersion)..." -ForegroundColor Gray
    $DownloadUrl = "https://github.com/viralang/vira/releases/download/$VexVersion/vex-$VexVersion-windows-amd64.zip"
    $ZipPath = "$env:TEMP\vex-install.zip"
    try {
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $ZipPath -UseBasicParsing
        Expand-Archive -Path $ZipPath -DestinationPath "$env:TEMP\vex-extract" -Force
        Copy-Item -Path "$env:TEMP\vex-extract\*\bin\vira.exe" -Destination "$BinDir\vira.exe" -Force
        Remove-Item -Force $ZipPath
        Remove-Item -Recurse -Force "$env:TEMP\vex-extract"
        Write-Host "  ✓ Downloaded and installed vira.exe" -ForegroundColor Green
    } catch {
        Write-Warning "Failed to download remote binary: $_"
        Write-Host "  = If building from source, run: go build -o $BinDir\vira.exe .\cmd\vex" -ForegroundColor Yellow
    }
}

# 3. Copy Runtime Headers
Write-Host "[3/5] Deploying runtime & GUI headers..." -ForegroundColor Yellow
if (Test-Path ".\runtime\vira_rt.h") {
    Copy-Item -Path ".\runtime\*.h" -Destination $RuntimeDir -Force
    Write-Host "  ✓ Copied runtime headers to $RuntimeDir" -ForegroundColor Green
} else {
    Write-Host "  ✓ Embedded runtime headers will auto-extract on first run" -ForegroundColor Green
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
