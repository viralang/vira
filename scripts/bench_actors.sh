#!/usr/bin/env bash
# Actor benchmark baseline: builds scripts/bench/actor_baseline.vir with -prod, runs it
# RUNS times with a fixed worker count, and prints the median, min and max of each metric
# plus the runtime's allocation counters. Run it before and after a runtime change on the
# same machine, with nothing else heavy running.
#
#   scripts/bench_actors.sh                 # 15 runs, VIRA_WORKERS=4, ./vira
#   RUNS=25 WORKERS=8 VIRA=./vira-old scripts/bench_actors.sh
#
# Output is also appended to bench/results.tsv (date, commit, workers, metric, median, min, max)
# so runs can be compared later.
set -euo pipefail

cd "$(dirname "$0")/.."
RUNS=${RUNS:-15}
WORKERS=${WORKERS:-4}
VIRA=${VIRA:-./vira}
SRC=scripts/bench/actor_baseline.vir
BIN=$(mktemp -d)/actor_baseline
OUT=$(mktemp)

"$VIRA" build -prod -o "$BIN" "$SRC" >/dev/null

for i in $(seq 1 "$RUNS"); do
  VIRA_WORKERS=$WORKERS "$BIN" >>"$OUT" 2>&1
done

if grep -q BENCH_ERROR "$OUT"; then
  echo "benchmark self-check failed:" >&2
  grep BENCH_ERROR "$OUT" >&2
  exit 1
fi

commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
if ! git diff --quiet 2>/dev/null; then commit="$commit+dirty"; fi
mkdir -p bench

echo "runs=$RUNS workers=$WORKERS cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo ?) commit=$commit"
printf '%-10s %12s %12s %12s %8s\n' metric median_us min_us max_us spread
for m in spawn_us fanout_us chaos_us ring_us pool_us; do
  vals=$(grep '^BENCH ' "$OUT" | tr ' ' '\n' | grep "^$m=" | cut -d= -f2 | sort -n)
  n=$(echo "$vals" | wc -l)
  med=$(echo "$vals" | sed -n "$(( (n + 1) / 2 ))p")
  min=$(echo "$vals" | head -1)
  max=$(echo "$vals" | tail -1)
  spread=$(awk -v a="$max" -v b="$min" -v m="$med" 'BEGIN { if (m > 0) printf "%.0f%%", (a - b) * 100 / m; else print "-" }')
  printf '%-10s %12s %12s %12s %8s\n' "$m" "$med" "$min" "$max" "$spread"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F)" "$commit" "$WORKERS" "$m" "$med" "$min" "$max" >>bench/results.tsv
done

echo
echo "allocation counters (identical every run unless the runtime changed):"
grep '^\[vira_rt\]' "$OUT" | grep -E 'active_leaks|total_freed' | sort | uniq -c
rm -f "$OUT"
