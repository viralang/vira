package version

// Version specifies the current semantic version of the Vira compiler toolchain.
// This is the single source of truth across the entire repository.
// It can also be overridden at link-time with:
//
//	-ldflags="-X github.com/viralang/vira/internal/version.Version=vX.Y.Z"
var Version = "v0.5.12"
