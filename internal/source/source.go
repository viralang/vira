package source

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vexlang/vex/internal/token"
)

// ANSI color codes for terminal diagnostics
const (
	colorReset  = "\033[0m"
	colorBold   = "\033[1m"
	colorRed    = "\033[31m"
	colorCyan   = "\033[36m"
	colorYellow = "\033[33m"
)

// File indexes a .vex source file's newline offsets for O(log N) span-to-line resolution.
type File struct {
	Filename   string
	Content    string
	lineStarts []uint32
}

// New builds a line-start index over `content` in a single pass.
func New(filename, content string) *File {
	starts := []uint32{0}
	for i := 0; i < len(content); i++ {
		if content[i] == '\n' && i+1 <= len(content) {
			starts = append(starts, uint32(i+1))
		}
	}
	return &File{
		Filename:   filename,
		Content:    content,
		lineStarts: starts,
	}
}

// Location resolves a 0-based byte offset into 1-based (line, col) coordinates.
func (f *File) Location(offset uint32) (line int, col int) {
	if int(offset) > len(f.Content) {
		offset = uint32(len(f.Content))
	}
	// Binary search for the largest lineStart <= offset
	idx := sort.Search(len(f.lineStarts), func(i int) bool {
		return f.lineStarts[i] > offset
	}) - 1
	if idx < 0 {
		idx = 0
	}
	line = idx + 1
	col = int(offset-f.lineStarts[idx]) + 1
	return line, col
}

// LineSlice returns the raw text of 1-based `lineNum` (without trailing \r or \n).
func (f *File) LineSlice(lineNum int) string {
	if lineNum < 1 || lineNum > len(f.lineStarts) {
		return ""
	}
	start := f.lineStarts[lineNum-1]
	end := uint32(len(f.Content))
	if lineNum < len(f.lineStarts) {
		end = f.lineStarts[lineNum]
	}
	raw := f.Content[start:end]
	return strings.TrimRight(raw, "\r\n")
}

// FormatDiagnostic renders a Rust/V-style source-span snippet with caret underlines and help hints.
func (f *File) FormatDiagnostic(stage string, rawMsg string, span token.Span) string {
	mainMsg, helpHint := splitHelpHint(rawMsg)
	line, col := f.Location(span.Start)
	rawLine := f.LineSlice(line)

	// Expand tabs to 4 spaces so the caret `^^^^` aligns under the exact visual column
	var expandedLine strings.Builder
	visualCol := 0
	targetVisualCol := 0

	for i := 0; i < len(rawLine); i++ {
		if i == col-1 {
			targetVisualCol = visualCol
		}
		if rawLine[i] == '\t' {
			expandedLine.WriteString("    ")
			visualCol += 4
		} else {
			expandedLine.WriteByte(rawLine[i])
			visualCol++
		}
	}
	if col-1 >= len(rawLine) {
		targetVisualCol = visualCol
	}

	// Compute caret width (minimum 1 `^`, capped to the current line length)
	spanLen := int(span.End - span.Start)
	if spanLen <= 0 {
		spanLen = 1
	}
	remainingInLine := len(rawLine) - (col - 1)
	if remainingInLine > 0 && spanLen > remainingInLine {
		spanLen = remainingInLine
	}
	if spanLen <= 0 {
		spanLen = 1
	}

	gutterWidth := len(fmt.Sprintf("%d", line))
	if gutterWidth < 2 {
		gutterWidth = 2
	}
	emptyGutter := strings.Repeat(" ", gutterWidth)
	lineGutter := fmt.Sprintf("%*d", gutterWidth, line)
	caretPad := strings.Repeat(" ", targetVisualCol)
	carets := strings.Repeat("^", spanLen)

	var out strings.Builder
	out.WriteString(fmt.Sprintf("%s%serror[%s]%s%s: %s%s\n",
		colorBold, colorRed, stage, colorReset, colorBold, mainMsg, colorReset))
	out.WriteString(fmt.Sprintf("%s%s-->%s %s:%d:%d\n",
		emptyGutter, colorCyan, colorReset, f.Filename, line, col))
	out.WriteString(fmt.Sprintf("%s %s|%s\n", emptyGutter, colorCyan, colorReset))
	out.WriteString(fmt.Sprintf("%s%s |%s %s\n",
		colorCyan, lineGutter, colorReset, expandedLine.String()))
	out.WriteString(fmt.Sprintf("%s %s|%s %s%s%s%s\n",
		emptyGutter, colorCyan, colorReset, caretPad, colorBold, colorRed, carets+colorReset))

	if helpHint != "" {
		out.WriteString(fmt.Sprintf("%s %s=%s %shelp:%s %s\n",
			emptyGutter, colorCyan, colorReset, colorYellow, colorReset, helpHint))
	}
	return out.String()
}

// splitHelpHint extracts trailing parenthetical suggestions into a `help:` line.
func splitHelpHint(msg string) (mainMsg string, help string) {
	if strings.HasSuffix(msg, ")") {
		openIdx := strings.LastIndex(msg, " (")
		if openIdx != -1 {
			return msg[:openIdx], msg[openIdx+2 : len(msg)-1]
		}
	}
	return msg, ""
}
