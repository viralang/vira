package vexrt

import "embed"

// RuntimeFS embeds all Vex runtime headers (.h) directly into the compiled binary.
// This guarantees vex.exe is 100% self-contained and portable.
//
//go:embed *.h
var RuntimeFS embed.FS
