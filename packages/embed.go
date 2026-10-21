package stdpkg

import "embed"

// StandardFS embeds all standard packages into the compiler binary.
//
//go:embed all:*
var StandardFS embed.FS
