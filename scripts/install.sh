#!/usr/bin/env bash
# Vira Programming Language - Linux & macOS Automated Installer & Dependency Resolver
# Usage: curl -fsSL https://raw.githubusercontent.com/viralang/vira/main/scripts/install.sh | bash
#    or: ./scripts/install.sh

DEFAULT_VERSION="v0.5.10"
VIRA_VERSION="${1:-${VIRA_VERSION:-$DEFAULT_VERSION}}"
if [[ "$VIRA_VERSION" != v* ]]; then
    VIRA_VERSION="v$VIRA_VERSION"
fi
INSTALL_DIR="$HOME/.vir"
BIN_DIR="$INSTALL_DIR/bin"
RUNTIME_DIR="$INSTALL_DIR/runtime"
LIB_DIR="$INSTALL_DIR/lib"
INCLUDE_DIR="$INSTALL_DIR/include"
EXAMPLES_DIR="$INSTALL_DIR/examples"

BOLD="$(tput bold 2>/dev/null || echo '')"
GREEN="$(tput setaf 2 2>/dev/null || echo '')"
YELLOW="$(tput setaf 3 2>/dev/null || echo '')"
CYAN="$(tput setaf 6 2>/dev/null || echo '')"
RESET="$(tput sgr0 2>/dev/null || echo '')"

echo "${CYAN}${BOLD}==========================================================${RESET}"
echo "${CYAN}${BOLD}   Vira Programming Language - Installer (${VIRA_VERSION})${RESET}"
echo "${CYAN}${BOLD}==========================================================${RESET}"
echo ""

# 1. Detect Platform & Architecture
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$ARCH" in
    x86_64|amd64)
        TARGET_ARCH="amd64"
        ;;
    aarch64|arm64)
        TARGET_ARCH="arm64"
        ;;
    *)
        echo "${YELLOW}Warning: Unknown architecture $ARCH. Defaulting to amd64.${RESET}"
        TARGET_ARCH="amd64"
        ;;
esac

echo "${YELLOW}[1/5] Platform detected:${RESET} $OS ($TARGET_ARCH)"

# 2. Create Target Directories
echo "${YELLOW}[2/5] Creating installation directories in $INSTALL_DIR...${RESET}"
mkdir -p "$BIN_DIR" "$RUNTIME_DIR" "$LIB_DIR" "$INCLUDE_DIR" "$EXAMPLES_DIR"
mkdir -p "$HOME/.vira/bin" "$HOME/.vira/runtime" "$HOME/.vira/lib" "$HOME/.vira/include" 2>/dev/null || true

# 3. Install Compiler Binary
echo "${YELLOW}[3/5] Installing Vira compiler binary...${RESET}"
INSTALLED=0

# Check local compiled binaries
for cand in "./vira" "./bin/vira" "../vira" "./vex" "./bin/vex" "../vex"; do
    if [ -f "$cand" ]; then
        cp "$cand" "$BIN_DIR/vira"
        ln -sf "$BIN_DIR/vira" "$BIN_DIR/vex" 2>/dev/null || cp "$BIN_DIR/vira" "$BIN_DIR/vex"
        chmod +x "$BIN_DIR/vira" "$BIN_DIR/vex" 2>/dev/null || true
        echo "  ${GREEN}✓ Installed local binary from $cand${RESET}"
        INSTALLED=1
        break
    fi
done

# Check local release archives
if [ $INSTALLED -eq 0 ]; then
    for pkg in \
        "./dist/vira-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz" \
        "../dist/vira-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz" \
        "./dist/vex-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz" \
        "../dist/vex-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz"; do
        if [ -f "$pkg" ]; then
            TMP_DIST="$(mktemp -d)"
            tar -xzf "$pkg" -C "$TMP_DIST"
            found_bin="$(find "$TMP_DIST" -type f \( -name "vira" -o -name "vex" \) | head -n 1)"
            if [ -n "$found_bin" ]; then
                cp "$found_bin" "$BIN_DIR/vira"
                ln -sf "$BIN_DIR/vira" "$BIN_DIR/vex" 2>/dev/null || cp "$BIN_DIR/vira" "$BIN_DIR/vex"
                chmod +x "$BIN_DIR/vira" "$BIN_DIR/vex" 2>/dev/null || true
                echo "  ${GREEN}✓ Installed from local package: $pkg${RESET}"
                INSTALLED=1
            fi
            rm -rf "$TMP_DIST"
            [ $INSTALLED -eq 1 ] && break
        fi
    done
fi

# Download from GitHub Releases
if [ $INSTALLED -eq 0 ]; then
    DOWNLOAD_SUCCESS=0
    TMP_DIR="$(mktemp -d)"

    for TAR_NAME in "vira-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz" "vex-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz"; do
        DOWNLOAD_URL="https://github.com/viralang/vira/releases/download/${VIRA_VERSION}/${TAR_NAME}"
        echo "  = Attempting download: $DOWNLOAD_URL"

        if command -v curl >/dev/null 2>&1; then
            curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/vira.tar.gz" 2>/dev/null || true
        elif command -v wget >/dev/null 2>&1; then
            wget -q "$DOWNLOAD_URL" -O "$TMP_DIR/vira.tar.gz" 2>/dev/null || true
        fi

        if [ -s "$TMP_DIR/vira.tar.gz" ]; then
            tar -xzf "$TMP_DIR/vira.tar.gz" -C "$TMP_DIR"
            found_bin="$(find "$TMP_DIR" -type f \( -name "vira" -o -name "vex" \) | head -n 1)"
            if [ -n "$found_bin" ]; then
                cp "$found_bin" "$BIN_DIR/vira"
                ln -sf "$BIN_DIR/vira" "$BIN_DIR/vex" 2>/dev/null || cp "$BIN_DIR/vira" "$BIN_DIR/vex"
                chmod +x "$BIN_DIR/vira" "$BIN_DIR/vex" 2>/dev/null || true
                echo "  ${GREEN}✓ Downloaded and installed vira binary (${TAR_NAME})${RESET}"
                DOWNLOAD_SUCCESS=1
                INSTALLED=1
                break
            fi
        fi
    done
    rm -rf "$TMP_DIR"

    # Build from source via Go if binary download not available
    if [ $DOWNLOAD_SUCCESS -eq 0 ]; then
        echo "  ${YELLOW}! Remote download not found. Attempting to build from source via Go...${RESET}"
        if command -v go >/dev/null 2>&1; then
            if [ -d "./cmd/vira" ]; then
                go build -ldflags="-s -w" -o "$BIN_DIR/vira" ./cmd/vira
            elif [ -d "../cmd/vira" ]; then
                go build -ldflags="-s -w" -o "$BIN_DIR/vira" ../cmd/vira
            else
                go install github.com/viralang/vira/cmd/vira@latest || true
                if command -v vira >/dev/null 2>&1; then
                    cp "$(command -v vira)" "$BIN_DIR/vira"
                fi
            fi

            if [ -f "$BIN_DIR/vira" ]; then
                ln -sf "$BIN_DIR/vira" "$BIN_DIR/vex" 2>/dev/null || cp "$BIN_DIR/vira" "$BIN_DIR/vex"
                chmod +x "$BIN_DIR/vira" "$BIN_DIR/vex" 2>/dev/null || true
                echo "  ${GREEN}✓ Built and installed vira binary from source via Go${RESET}"
                INSTALLED=1
            fi
        fi
        if [ $INSTALLED -eq 0 ]; then
            echo "  ${YELLOW}! Please ensure 'vira' binary is placed in $BIN_DIR${RESET}"
        fi
    fi
fi

# 4. Copy Runtime Headers, Examples & Documentation
echo "${YELLOW}[4/5] Deploying runtime headers and examples...${RESET}"
if [ -d "./runtime" ]; then
    cp -r ./runtime/*.h "$RUNTIME_DIR/" 2>/dev/null || true
    echo "  ${GREEN}✓ Copied runtime headers to $RUNTIME_DIR${RESET}"
elif [ -d "../runtime" ]; then
    cp -r ../runtime/*.h "$RUNTIME_DIR/" 2>/dev/null || true
    echo "  ${GREEN}✓ Copied runtime headers to $RUNTIME_DIR${RESET}"
else
    echo "  ${GREEN}✓ Embedded runtime headers will auto-extract on first run${RESET}"
fi

if [ -d "./examples" ]; then
    cp -r ./examples/* "$EXAMPLES_DIR/" 2>/dev/null || true
    echo "  ${GREEN}✓ Copied runnable examples to $EXAMPLES_DIR${RESET}"
elif [ -d "../examples" ]; then
    cp -r ../examples/* "$EXAMPLES_DIR/" 2>/dev/null || true
    echo "  ${GREEN}✓ Copied runnable examples to $EXAMPLES_DIR${RESET}"
fi

[ -f "./LANGUAGE_GUIDE.md" ] && cp "./LANGUAGE_GUIDE.md" "$INSTALL_DIR/" 2>/dev/null || true
[ -f "./README.md" ] && cp "./README.md" "$INSTALL_DIR/" 2>/dev/null || true

# Synchronize with ~/.vira for VS Code and language server tools
if [ "$INSTALL_DIR" != "$HOME/.vira" ]; then
    mkdir -p "$HOME/.vira/bin" "$HOME/.vira/runtime" 2>/dev/null || true
    cp -f "$BIN_DIR/vira" "$HOME/.vira/bin/vira" 2>/dev/null || true
    cp -f "$BIN_DIR/vex" "$HOME/.vira/bin/vex" 2>/dev/null || true
    cp -rf "$RUNTIME_DIR"/*.h "$HOME/.vira/runtime/" 2>/dev/null || true
fi

# 5. Dependency Resolution (C Compiler & System Libraries)
echo "${YELLOW}[5/5] Checking system dependencies & shell PATH...${RESET}"
CC_FOUND=0
for cc in gcc clang cc tcc; do
    if command -v "$cc" >/dev/null 2>&1; then
        echo "  ${GREEN}✓ Found C backend compiler:${RESET} $(command -v $cc)"
        CC_FOUND=1
        break
    fi
done

if [ $CC_FOUND -eq 0 ]; then
    echo "  ${YELLOW}! No C99 backend compiler found. Attempting automated dependency resolution...${RESET}"
    
    SUDO_CMD=""
    if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
        if sudo -n true 2>/dev/null; then
            SUDO_CMD="sudo"
        fi
    fi

    if [ "$(id -u)" -eq 0 ] || [ -n "$SUDO_CMD" ]; then
        if [ -f /etc/debian_version ] && command -v apt-get >/dev/null 2>&1; then
            echo "  ${CYAN}= Installing gcc, make, X11 & OpenGL development packages via apt-get...${RESET}"
            $SUDO_CMD apt-get update -qq && $SUDO_CMD apt-get install -y --no-install-recommends \
                gcc libc6-dev make libx11-dev libgl1-mesa-dev libxcursor-dev libxinerama-dev libxi-dev libxrandr-dev >/dev/null 2>&1 || true
        elif [ -f /etc/alpine-release ] && command -v apk >/dev/null 2>&1; then
            echo "  ${CYAN}= Installing gcc, musl, X11 & mesa dev packages via apk...${RESET}"
            $SUDO_CMD apk add --no-cache gcc musl-dev make libx11-dev mesa-dev >/dev/null 2>&1 || true
        elif command -v dnf >/dev/null 2>&1; then
            echo "  ${CYAN}= Installing gcc, make, X11 & OpenGL dev packages via dnf...${RESET}"
            $SUDO_CMD dnf install -y gcc glibc-devel make libX11-devel mesa-libGL-devel libXcursor-devel libXinerama-devel libXi-devel libXrandr-devel >/dev/null 2>&1 || true
        fi

        for cc in gcc clang cc tcc; do
            if command -v "$cc" >/dev/null 2>&1; then
                echo "  ${GREEN}✓ Successfully installed and verified C compiler:${RESET} $(command -v $cc)"
                CC_FOUND=1
                break
            fi
        done
    fi

    if [ $CC_FOUND -eq 0 ]; then
        echo "  ${YELLOW}! Please install a C compiler using your system package manager:${RESET}"
        if [ "$OS" = "darwin" ]; then
            echo "    ${CYAN}Fix for macOS:${RESET} Run: xcode-select --install"
        elif [ -f /etc/debian_version ]; then
            echo "    ${CYAN}Fix for Debian/Ubuntu:${RESET} sudo apt-get update && sudo apt-get install -y build-essential libx11-dev libgl1-mesa-dev"
        elif [ -f /etc/fedora-release ] || [ -f /etc/redhat-release ]; then
            echo "    ${CYAN}Fix for Fedora/RHEL:${RESET} sudo dnf groupinstall \"Development Tools\" && sudo dnf install -y libX11-devel mesa-libGL-devel"
        elif [ -f /etc/arch-release ]; then
            echo "    ${CYAN}Fix for Arch Linux:${RESET} sudo pacman -S --needed base-devel libx11 mesa raylib"
        elif [ -f /etc/alpine-release ]; then
            echo "    ${CYAN}Fix for Alpine Linux:${RESET} sudo apk add build-base libx11-dev mesa-dev"
        fi
    fi
fi

# Configure GUI & Graphics Libraries (Raylib & OpenGL)
echo "  ${YELLOW}= Configuring GUI & Graphics Libraries (Raylib)...${RESET}"
HAS_RAYLIB=0
for cand in \
    "$LIB_DIR/libraylib.a" \
    "$HOME/.vira/lib/libraylib.a" \
    "/usr/lib/libraylib.a" \
    "/usr/lib/libraylib.so" \
    "/usr/local/lib/libraylib.a" \
    "/usr/lib/x86_64-linux-gnu/libraylib.a" \
    "/usr/lib/x86_64-linux-gnu/libraylib.so" \
    "./lib/libraylib.a"; do
    if [ -f "$cand" ]; then
        echo "  ${GREEN}✓ Found Raylib graphics library:${RESET} $cand"
        HAS_RAYLIB=1
        break
    fi
done

if [ $HAS_RAYLIB -eq 0 ]; then
    echo "  ${CYAN}= Downloading precompiled Raylib 5.0 for $OS ($TARGET_ARCH)...${RESET}"
    RAYLIB_PKG=""
    if [ "$OS" = "linux" ]; then
        case "$TARGET_ARCH" in
            amd64) RAYLIB_PKG="raylib-5.0_linux_amd64.tar.gz" ;;
            arm64) RAYLIB_PKG="raylib-5.0_linux_arm64.tar.gz" ;;
            *)     RAYLIB_PKG="raylib-5.0_linux_amd64.tar.gz" ;;
        esac
    elif [ "$OS" = "darwin" ]; then
        RAYLIB_PKG="raylib-5.0_macos.tar.gz"
    fi

    if [ -n "$RAYLIB_PKG" ]; then
        RAYLIB_URL="https://github.com/raysan5/raylib/releases/download/5.0/${RAYLIB_PKG}"
        TMP_RAYLIB="$(mktemp -d)"
        echo "  = Fetching: $RAYLIB_URL"
        if curl -fsSL "$RAYLIB_URL" -o "$TMP_RAYLIB/raylib.tar.gz" 2>/dev/null || wget -q "$RAYLIB_URL" -O "$TMP_RAYLIB/raylib.tar.gz" 2>/dev/null; then
            tar -xzf "$TMP_RAYLIB/raylib.tar.gz" -C "$TMP_RAYLIB"
            EXTRACTED_DIR="$(find "$TMP_RAYLIB" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
            if [ -n "$EXTRACTED_DIR" ] && [ -d "$EXTRACTED_DIR/lib" ]; then
                cp -r "$EXTRACTED_DIR/lib/"* "$LIB_DIR/" 2>/dev/null || true
                cp -r "$EXTRACTED_DIR/include/"* "$INCLUDE_DIR/" 2>/dev/null || true
                mkdir -p "$HOME/.vira/lib" "$HOME/.vira/include" 2>/dev/null || true
                cp -r "$EXTRACTED_DIR/lib/"* "$HOME/.vira/lib/" 2>/dev/null || true
                cp -r "$EXTRACTED_DIR/include/"* "$HOME/.vira/include/" 2>/dev/null || true
                echo "  ${GREEN}✓ Successfully installed Raylib to $LIB_DIR and $INCLUDE_DIR${RESET}"
                HAS_RAYLIB=1
            fi
        fi
        rm -rf "$TMP_RAYLIB"
    fi
fi

# Configure Shell PATH
PATH_LINE="export PATH=\"\$HOME/.vir/bin:\$HOME/.vira/bin:\$PATH\""

SHELL_PROFILES=()
[ -f "$HOME/.bashrc" ] && SHELL_PROFILES+=("$HOME/.bashrc")
[ -f "$HOME/.zshrc" ] && SHELL_PROFILES+=("$HOME/.zshrc")
[ -f "$HOME/.profile" ] && SHELL_PROFILES+=("$HOME/.profile")

ADDED_TO_PROFILE=0
for profile in "${SHELL_PROFILES[@]}"; do
    if ! grep -q ".vir/bin" "$profile" 2>/dev/null; then
        echo "" >> "$profile"
        echo "# Vira Programming Language" >> "$profile"
        echo "$PATH_LINE" >> "$profile"
        echo "  ${GREEN}✓ Added ~/.vir/bin to $profile${RESET}"
        ADDED_TO_PROFILE=1
    fi
done

if [ $ADDED_TO_PROFILE -eq 0 ]; then
    echo "  ${GREEN}✓ ~/.vir/bin is already in shell profile${RESET}"
fi

echo ""
echo "${GREEN}${BOLD}==========================================================${RESET}"
echo "${GREEN}${BOLD}  ✓ Vira was successfully installed to: $INSTALL_DIR${RESET}"
echo "${GREEN}${BOLD}==========================================================${RESET}"
echo ""
echo "${CYAN}Next steps:${RESET}"
echo "  1. Reload shell:  ${BOLD}source ~/.bashrc${RESET} (or ${BOLD}source ~/.zshrc${RESET})"
echo "  2. Verify:        ${BOLD}vira -h${RESET}"
echo "  3. Run demo:      ${BOLD}vira run examples/gui_demo.vir${RESET}"
echo ""
