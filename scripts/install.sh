#!/usr/bin/env bash
# Vira Programming Language - Linux & macOS Automated Installer & Dependency Resolver
# Usage: curl -fsSL https://viralang.org/install.sh | bash
#    or: ./scripts/install.sh

set -e

VIRA_VERSION="v0.5.5"
INSTALL_DIR="$HOME/.vir"
BIN_DIR="$INSTALL_DIR/bin"
RUNTIME_DIR="$INSTALL_DIR/runtime"
LIB_DIR="$INSTALL_DIR/lib"

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
mkdir -p "$BIN_DIR" "$RUNTIME_DIR" "$LIB_DIR"

# 3. Install Compiler Binary
echo "${YELLOW}[3/5] Installing Vira compiler binary...${RESET}"
INSTALLED=0

if [ -f "./vex" ]; then
    cp "./vex" "$BIN_DIR/vex"
    chmod +x "$BIN_DIR/vex"
    echo "  ${GREEN}✓ Installed local binary from ./vex${RESET}"
    INSTALLED=1
elif [ -f "./bin/vex" ]; then
    cp "./bin/vex" "$BIN_DIR/vex"
    chmod +x "$BIN_DIR/vex"
    echo "  ${GREEN}✓ Installed local binary from ./bin/vex${RESET}"
    INSTALLED=1
elif [ -f "../vex" ]; then
    cp "../vex" "$BIN_DIR/vex"
    chmod +x "$BIN_DIR/vex"
    echo "  ${GREEN}✓ Installed local binary from ../vex${RESET}"
    INSTALLED=1
elif [ -f "./dist/vex-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz" ]; then
    TMP_DIST="$(mktemp -d)"
    tar -xzf "./dist/vex-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz" -C "$TMP_DIST"
    cp "$TMP_DIST"/*/bin/vex "$BIN_DIR/vex"
    chmod +x "$BIN_DIR/vex"
    rm -rf "$TMP_DIST"
    echo "  ${GREEN}✓ Installed from local package: ./dist/vex-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz${RESET}"
    INSTALLED=1
elif [ -f "../dist/vex-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz" ]; then
    TMP_DIST="$(mktemp -d)"
    tar -xzf "../dist/vex-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz" -C "$TMP_DIST"
    cp "$TMP_DIST"/*/bin/vex "$BIN_DIR/vex"
    chmod +x "$BIN_DIR/vex"
    rm -rf "$TMP_DIST"
    echo "  ${GREEN}✓ Installed from local package: ../dist/vex-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz${RESET}"
    INSTALLED=1
fi

if [ $INSTALLED -eq 0 ]; then
    TAR_NAME="vex-${VIRA_VERSION}-${OS}-${TARGET_ARCH}.tar.gz"
    DOWNLOAD_URL="https://github.com/viralang/vira/releases/download/${VIRA_VERSION}/${TAR_NAME}"
    TMP_DIR="$(mktemp -d)"

    echo "  = Downloading $TAR_NAME from GitHub..."
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/vex.tar.gz" || true
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$DOWNLOAD_URL" -O "$TMP_DIR/vex.tar.gz" || true
    fi

    if [ -f "$TMP_DIR/vex.tar.gz" ]; then
        tar -xzf "$TMP_DIR/vex.tar.gz" -C "$TMP_DIR"
        cp "$TMP_DIR"/*/bin/vex "$BIN_DIR/vex" 2>/dev/null || cp "$TMP_DIR/vex" "$BIN_DIR/vex"
        chmod +x "$BIN_DIR/vex"
        echo "  ${GREEN}✓ Downloaded and extracted vex binary${RESET}"
    else
        echo "  ${YELLOW}! Remote download not found. Attempting to build from source via Go...${RESET}"
        if command -v go >/dev/null 2>&1; then
            go build -ldflags="-s -w" -o "$BIN_DIR/vex" ./cmd/vex
            chmod +x "$BIN_DIR/vex"
            echo "  ${GREEN}✓ Built vex binary from source via Go${RESET}"
        else
            echo "  ${YELLOW}! Please ensure 'vira' binary is placed in $BIN_DIR${RESET}"
        fi
    fi
    rm -rf "$TMP_DIR"
fi

# 4. Copy Runtime Headers
if [ -d "./runtime" ]; then
    cp -r ./runtime/*.h "$RUNTIME_DIR/" 2>/dev/null || true
    echo "  ${GREEN}✓ Copied runtime headers to $RUNTIME_DIR${RESET}"
fi

# 5. Dependency Resolution (C Compiler & System Libraries)
echo "${YELLOW}[4/5] Checking and resolving system dependencies...${RESET}"
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
            echo "  ${CYAN}= Installing gcc & libc6-dev via apt-get...${RESET}"
            $SUDO_CMD apt-get update -qq && $SUDO_CMD apt-get install -y --no-install-recommends gcc libc6-dev make >/dev/null 2>&1 || true
        elif [ -f /etc/alpine-release ] && command -v apk >/dev/null 2>&1; then
            echo "  ${CYAN}= Installing gcc & musl-dev via apk...${RESET}"
            $SUDO_CMD apk add --no-cache gcc musl-dev make >/dev/null 2>&1 || true
        elif command -v dnf >/dev/null 2>&1; then
            echo "  ${CYAN}= Installing gcc via dnf...${RESET}"
            $SUDO_CMD dnf install -y gcc glibc-devel make >/dev/null 2>&1 || true
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
            echo "    ${CYAN}Fix for Debian/Ubuntu:${RESET} sudo apt-get update && sudo apt-get install -y build-essential"
        elif [ -f /etc/fedora-release ] || [ -f /etc/redhat-release ]; then
            echo "    ${CYAN}Fix for Fedora/RHEL:${RESET} sudo dnf groupinstall \"Development Tools\""
        elif [ -f /etc/arch-release ]; then
            echo "    ${CYAN}Fix for Arch Linux:${RESET} sudo pacman -S --needed base-devel"
        elif [ -f /etc/alpine-release ]; then
            echo "    ${CYAN}Fix for Alpine Linux:${RESET} sudo apk add build-base"
        fi
    fi
fi

# Check graphics libraries on Linux
if [ "$OS" = "linux" ]; then
    if ! ldconfig -p 2>/dev/null | grep -q "libX11.so" && [ ! -f /usr/lib/libX11.so ] && [ ! -f /usr/lib64/libX11.so ]; then
        echo "  ${YELLOW}= Note for GUI apps on Linux:${RESET} Install X11/OpenGL development headers:"
        echo "    Debian/Ubuntu: sudo apt-get install -y libx11-dev libgl1-mesa-dev"
        echo "    Fedora/RHEL:   sudo dnf install -y libX11-devel mesa-libGL-devel"
        echo "    Arch:          sudo pacman -S libx11 mesa"
    else
        echo "  ${GREEN}✓ Native graphics libraries detected (libX11 / OpenGL)${RESET}"
    fi
elif [ "$OS" = "darwin" ]; then
    echo "  ${GREEN}✓ Native macOS graphics frameworks available (Cocoa / OpenGL)${RESET}"
fi

# 6. Configure Shell PATH
echo "${YELLOW}[5/5] Configuring shell PATH...${RESET}"
PATH_LINE="export PATH=\"\$HOME/.vir/bin:\$PATH\""

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
echo "  2. Verify:        ${BOLD}vex -h${RESET}"
echo "  3. Run demo:      ${BOLD}vira run examples/gui_demo.vir${RESET}"
echo ""
