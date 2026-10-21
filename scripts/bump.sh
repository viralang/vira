#!/usr/bin/env bash
# Vira Release Bumper for Linux/macOS
# Usage:
#   ./scripts/bump.sh v0.5.10
#   ./scripts/bump.sh -p v0.5.10
#   ./scripts/bump.sh -n v0.5.10

set -e
go run ./scripts/bump.go "$@"
