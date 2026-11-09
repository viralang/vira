package virart

import "embed"

// RuntimeFS embeds all Vira runtime headers (.h), including modular headers under rt/,
// directly into the compiled binary. This guarantees vira.exe is 100% self-contained and portable.
//
//go:embed *.h rt
var RuntimeFS embed.FS
