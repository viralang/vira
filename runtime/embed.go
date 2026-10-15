package virart

import "embed"

// RuntimeFS embeds all Vira runtime headers (.h) directly into the compiled binary.
// This guarantees vira.exe is 100% self-contained and portable.
//
//go:embed *.h
var RuntimeFS embed.FS
