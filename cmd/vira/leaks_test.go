package main

import (
	"flag"
	"reflect"
	"testing"
)

func TestActiveLeaks(t *testing.T) {
	cases := []struct {
		out  string
		n    int
		seen bool
	}{
		{"[vira_rt] active_leaks=0 | total_freed=9\n", 0, true},
		{"[vira_rt] active_leaks=5 | total_freed=9\n", 5, true},
		{"ok\n", 0, false},
		{"active_leaks=\n", 0, false},
		{"active_leaks=0\n... active_leaks=3 | x", 3, true},
	}
	for _, c := range cases {
		n, seen := activeLeaks(c.out)
		if n != c.n || seen != c.seen {
			t.Errorf("activeLeaks(%q) = %d, %v; want %d, %v", c.out, n, seen, c.n, c.seen)
		}
	}
}

func TestReorderFlagsBeforeArgs(t *testing.T) {
	fs := flag.NewFlagSet("build", flag.ContinueOnError)
	fs.Bool("prod", false, "")
	fs.String("o", "", "")

	got := reorderFlagsBeforeArgs(fs, []string{"main.vir", "-prod", "-o", "out/app"})
	want := []string{"-prod", "-o", "out/app", "main.vir"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reorderFlagsBeforeArgs = %v; want %v", got, want)
	}
}
