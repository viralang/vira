#!/usr/bin/env bash
# Vira Release Packager for Linux and macOS
# Runs the automated Go packager to build releases for Windows, Linux, and macOS.

set -e

echo "Running Vira Cross-Platform Release Packager..."
go run ./scripts/package.go "$@"
