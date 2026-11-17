#!/usr/bin/env bash
# Actor benchmarks comparable with figures published for Erlang/OTP and Akka:
# one-way message throughput, a 1,000-actor ring, spawn cost, and resident memory per
# idle actor. Prints the median, min and max over RUNS runs and appends to
# bench/compare.tsv. Linux only (memory is read from /proc).
#
#   scripts/bench_actor_compare.sh             # 9 runs, VIRA_WORKERS=4, ./vira
#   RUNS=15 WORKERS=8 VIRA=./vira scripts/bench_actor_compare.sh
set -euo pipefail

cd "$(dirname "$0")/.."
RUNS=${RUNS:-9}
WORKERS=${WORKERS:-4}
VIRA=${VIRA:-./vira}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

"$VIRA" build -prod -o "$TMP/compare" scripts/bench/actor_compare.vir >/dev/null
"$VIRA" build -prod -o "$TMP/memory" scripts/bench/actor_memory.vir >/dev/null

for i in $(seq 1 "$RUNS"); do
  VIRA_WORKERS=$WORKERS "$TMP/compare" 2>/dev/null | grep '^BENCH' >>"$TMP/out"
done
if grep -q BENCH_ERROR "$TMP/out"; then
  echo "benchmark self-check failed:" >&2
  grep BENCH_ERROR "$TMP/out" >&2
  exit 1
fi

rss_kb() { # $1 = number of actors
  ACTORS=$1 VIRA_WORKERS=$WORKERS "$TMP/memory" >"$TMP/mem.out" 2>&1 &
  local pid=$!
  for _ in $(seq 1 400); do
    grep -q READY "$TMP/mem.out" 2>/dev/null && break
    sleep 0.05
  done
  awk '/VmRSS/ {print $2}' "/proc/$pid/status"
  wait "$pid"
}
base=$(rss_kb 0)
full=$(rss_kb 1000000)
bytes_per_actor=$(( (full - base) * 1024 / 1000000 ))

commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
mkdir -p bench
echo "Vira actor comparison benchmark: $RUNS runs, VIRA_WORKERS=$WORKERS, $(nproc) cores, commit $commit"
for metric in send1_mps sendN_mps ring_hops spawn_ns; do
  grep -o "$metric=[0-9]*" "$TMP/out" | cut -d= -f2 | sort -n >"$TMP/vals"
  n=$(wc -l <"$TMP/vals")
  med=$(sed -n "$(( (n + 1) / 2 ))p" "$TMP/vals")
  min=$(head -1 "$TMP/vals")
  max=$(tail -1 "$TMP/vals")
  printf '  %-10s median %12s  min %12s  max %12s\n' "$metric" "$med" "$min" "$max"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$commit" "$WORKERS" "$metric" "$med" "$min" "$max" >>bench/compare.tsv
done
printf '  %-10s %s bytes (1,000,000 idle actors, VmRSS difference)\n' "memory" "$bytes_per_actor"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$commit" "$WORKERS" "bytes_per_actor" "$bytes_per_actor" "$bytes_per_actor" "$bytes_per_actor" >>bench/compare.tsv
