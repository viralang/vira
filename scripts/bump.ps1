# Vira Release Bumper for PowerShell
# Usage:
#   .\scripts\bump.ps1 v0.5.10
#   .\scripts\bump.ps1 -Package v0.5.10   (or -p)
#   .\scripts\bump.ps1 -DryRun v0.5.10    (or -n)

param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$NewVersion,
    [Alias("p")][switch]$Package,
    [Alias("b")][switch]$Build,
    [Alias("n")][switch]$DryRun
)

$argsList = @()
if ($Package) { $argsList += "-package" }
if ($Build)   { $argsList += "-build" }
if ($DryRun)  { $argsList += "-dry-run" }
$argsList += $NewVersion

go run .\scripts\bump.go @argsList
