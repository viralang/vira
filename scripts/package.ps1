# Vex Release Packager for Windows
# Runs the automated Go packager to build releases for Windows, Linux, and macOS.

Write-Host "Running Vex Cross-Platform Release Packager..." -ForegroundColor Cyan
go run .\scripts\package.go
