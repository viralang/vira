package formatter_test

import (
	"strings"
	"testing"

	"github.com/viralang/vira/internal/formatter"
)

func TestFormatBasicIndentation(t *testing.T) {
	input := `
module main

pub struct Account {
pub:
id string
pub mut:
balance i64
}

pub fn calculate(x int) int {
if x > 10 {
return x * 2
} else {
return x + 1
}
}
`

	formatted := formatter.Format(input)

	expectedSnippet := "pub struct Account {\npub:\n\tid string\npub mut:\n\tbalance i64\n}"
	if !strings.Contains(formatted, expectedSnippet) {
		t.Errorf("expected formatted struct with quadrant indentation, got:\n%s", formatted)
	}

	if !strings.Contains(formatted, "\t\treturn x * 2") {
		t.Errorf("expected indented body in if block, got:\n%s", formatted)
	}
}

func TestFormatIdempotency(t *testing.T) {
	input := `
module math

pub fn dot(a f32, b f32) f32 {
	return a * b
}
`
	firstPass := formatter.Format(input)
	secondPass := formatter.Format(firstPass)

	if firstPass != secondPass {
		t.Errorf("formatter should be idempotent.\nFirst pass:\n%s\nSecond pass:\n%s", firstPass, secondPass)
	}
}
