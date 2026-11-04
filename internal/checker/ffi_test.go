package checker_test

import (
	"strings"
	"testing"

	"github.com/viralang/vira/internal/checker"
	"github.com/viralang/vira/internal/parser"
)

func checkSource(src string) []string {
	p := parser.New(src)
	file := p.ParseFile()
	chk := checker.New()
	chk.CheckFile(file)
	var errs []string
	for _, e := range chk.Errors() {
		errs = append(errs, e.Msg)
	}
	return errs
}

func TestFFI_ExternParsingAndChecking(t *testing.T) {
	src := `
#include <stdio.h>
#include <string.h>

extern "C" {
	fn puts(s &c_char) c_int
	fn strlen(s &c_char) c_size
}

fn test_valid() {
	unsafe {
		puts("Hello from Vira C FFI!")
		len := strlen("test")
	}
}
`
	errs := checkSource(src)
	if len(errs) > 0 {
		t.Fatalf("unexpected checker errors: %v", errs)
	}
}

func TestFFI_UnsafeCallEnforcement(t *testing.T) {
	src := `
extern "C" {
	fn puts(s &c_char) c_int
}

fn safe_caller() {
	puts("this should fail outside unsafe block")
}
`
	errs := checkSource(src)
	if len(errs) == 0 {
		t.Fatalf("expected error when calling extern C function outside unsafe block, got none")
	}
	found := false
	for _, e := range errs {
		if strings.Contains(e, "is unsafe") && strings.Contains(e, "unsafe") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected unsafe error diagnostic, got: %v", errs)
	}
}

func TestFFI_UnsafeFunctionAllowed(t *testing.T) {
	src := `
extern "C" {
	fn puts(s &c_char) c_int
}

@[unsafe]
fn unsafe_caller() {
	puts("allowed because enclosing function is @[unsafe]")
}
`
	errs := checkSource(src)
	if len(errs) > 0 {
		t.Fatalf("unexpected checker errors: %v", errs)
	}
}

func TestFFI_PointersAndAddressOf(t *testing.T) {
	src := `
fn test_pointers() {
	mut x := 42
	ptr := &x
	mut_ptr := &mut x
	val := *ptr
}
`
	errs := checkSource(src)
	if len(errs) > 0 {
		t.Fatalf("unexpected checker errors: %v", errs)
	}
}

func TestFFI_OpaqueAndCLayoutStructs(t *testing.T) {
	src := `
@[opaque]
struct Sqlite3 {}

@[c_layout]
struct Point2D {
	pub:
	x c_int
	y c_int
}

@[export: "compute_point"]
fn compute_point(p Point2D) c_int {
	return p.x + p.y
}
`
	errs := checkSource(src)
	if len(errs) > 0 {
		t.Fatalf("unexpected checker errors: %v", errs)
	}
}
