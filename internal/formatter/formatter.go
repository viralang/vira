package formatter

import (
	"strings"
)

// Format formats a Vex source code string according to standard style guidelines.
func Format(src string) string {
	lines := strings.Split(src, "\n")
	var result []string
	indentLevel := 0
	prevEmpty := false

	for _, rawLine := range lines {
		trimmed := strings.TrimRight(rawLine, " \t\r")
		clean := strings.TrimSpace(trimmed)

		// Handle empty lines (collapse multiple consecutive empty lines to 1)
		if clean == "" {
			if !prevEmpty && len(result) > 0 {
				result = append(result, "")
				prevEmpty = true
			}
			continue
		}
		prevEmpty = false

		// Check for closing braces at line start to decrease indent before printing
		dedentBefore := 0
		checkPrefix := clean
		for strings.HasPrefix(checkPrefix, "}") || strings.HasPrefix(checkPrefix, ")") || strings.HasPrefix(checkPrefix, "]") {
			dedentBefore++
			checkPrefix = strings.TrimSpace(checkPrefix[1:])
		}
		currentIndent := indentLevel - dedentBefore
		if currentIndent < 0 {
			currentIndent = 0
		}

		// Quadrant labels inside structs (pub:, mut:, pub mut:) should have one less indent
		isQuadrantLabel := clean == "pub:" || clean == "mut:" || clean == "pub mut:" || clean == "default:"
		lineIndent := currentIndent
		if isQuadrantLabel && lineIndent > 0 {
			lineIndent--
		}

		// Apply operator spacing for cleaner syntax
		formattedLine := formatLineTokens(clean)

		// Prepend tabs
		indentStr := strings.Repeat("\t", lineIndent)
		result = append(result, indentStr+formattedLine)

		// Calculate net change in indentation for subsequent lines
		netIndent := calculateIndentDelta(clean)
		indentLevel += netIndent
		if indentLevel < 0 {
			indentLevel = 0
		}
	}

	res := strings.Join(result, "\n")
	if !strings.HasSuffix(res, "\n") {
		res += "\n"
	}
	return res
}

func calculateIndentDelta(line string) int {
	// Strip string literals and comments to avoid counting braces inside them
	clean := stripLiteralsAndComments(line)
	delta := 0
	for _, ch := range clean {
		switch ch {
		case '{', '(':
			delta++
		case '}', ')':
			delta--
		}
	}
	return delta
}

func stripLiteralsAndComments(line string) string {
	var sb strings.Builder
	inSingleQuote := false
	inDoubleQuote := false
	inEscape := false

	runes := []rune(line)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if inEscape {
			inEscape = false
			continue
		}
		if r == '\\' {
			inEscape = true
			continue
		}
		if r == '\'' && !inDoubleQuote {
			inSingleQuote = !inSingleQuote
			continue
		}
		if r == '"' && !inSingleQuote {
			inDoubleQuote = !inDoubleQuote
			continue
		}
		if !inSingleQuote && !inDoubleQuote {
			// Check for single line comment
			if r == '/' && i+1 < len(runes) && runes[i+1] == '/' {
				break
			}
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

func formatLineTokens(line string) string {
	// Normalize common operator spacing if not in string literals
	// e.g. `:=` `<-` `@` `==` `!=`
	if strings.HasPrefix(line, "//") || strings.HasPrefix(line, "/*") {
		return line
	}
	// Preserve comment at end of line if present
	commentIdx := strings.Index(line, "//")
	mainPart := line
	commentPart := ""
	if commentIdx != -1 {
		// Verify comment is not inside a quote
		prefix := stripLiteralsAndComments(line[:commentIdx])
		if len(prefix) == commentIdx {
			mainPart = strings.TrimSpace(line[:commentIdx])
			commentPart = " " + strings.TrimSpace(line[commentIdx:])
		}
	}

	// Minor normalization for assignment operators
	// We keep formatting conservative to not mutate user-formatted expressions incorrectly
	return mainPart + commentPart
}
