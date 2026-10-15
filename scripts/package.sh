#!/usr/bin/env bash
# Vex Release Packager for Linux and macOS
# Runs the automated Go packager to build releases for Windows, Linux, and macOS.

set -e

echo "Running Vex Cross-Platform Release Packager..."
go run ./scripts/package.go
