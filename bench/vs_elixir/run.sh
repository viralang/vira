#!/usr/bin/env bash
# Vira vs Elixir: actors and concurrency, side by side on this machine.
#
#   bench/vs_elixir/run.sh                      # 5 runs, workers = cores (max 16)
#   RUNS=9 WORKERS=4 VIRA=./vira bench/vs_elixir/run.sh
#
# Both sides get the same number of OS threads: VIRA_WORKERS=W for Vira and
# +S W:W (W schedulers) for the BEAM. PAR=W CPU-bound actors in the parallel test.
# Each metric is the median of RUNS runs (time measured inside the program, so VM
# start-up and compile time are excluded). Writes a markdown report and a TSV next
# to this script. Linux only (memory is read from /proc).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
RUNS=${RUNS:-5}
CORES=$(nproc)
W=${WORKERS:-$(( CORES > 16 ? 16 : CORES ))}
VIRA=${VIRA:-./vira}
ERL_FLAGS="+P 4000000 +S $W:$W"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

command -v elixir >/dev/null || { echo "elixir not found in PATH" >&2; exit 1; }
echo "Building Vira benchmark ($("$VIRA" version 2>/dev/null | head -1))..."
"$VIRA" build -prod -o "$TMP/vira_actors" "$HERE/actors.vir" >/dev/null
"$VIRA" build -prod -o "$TMP/vira_memory" scripts/bench/actor_memory.vir >/dev/null
EXV=$(elixir --version 2>/dev/null | tail -1)
echo "Using $EXV"

for i in $(seq 1 "$RUNS"); do
  printf '  run %d/%d: vira' "$i" "$RUNS"
  VIRA_WORKERS=$W PAR=$W "$TMP/vira_actors" 2>/dev/null | grep '^BENCH' >>"$TMP/vira.out"
  printf ', elixir\n'
  PAR=$W elixir --erl "$ERL_FLAGS" "$HERE/actors.exs" 2>/dev/null | grep '^BENCH' >>"$TMP/elixir.out"
done
if grep -h BENCH_ERROR "$TMP"/*.out; then
  echo "self-check failed, results not valid" >&2
  exit 1
fi

median() { # $1 = file, $2 = metric
  grep -o "\b$2=[0-9]*" "$1" | cut -d= -f2 | sort -n >"$TMP/vals" || true
  local n; n=$(wc -l <"$TMP/vals")
  if [ "$n" -eq 0 ]; then echo "-"; else sed -n "$(( (n + 1) / 2 ))p" "$TMP/vals"; fi
}

rss_kb() { # $1 = number of actors, $2.. = command
  local n=$1; shift
  ACTORS=$n VIRA_WORKERS=$W "$@" >"$TMP/mem.out" 2>&1 &
  local pid=$!
  for _ in $(seq 1 600); do
    grep -q READY "$TMP/mem.out" 2>/dev/null && break
    sleep 0.05
  done
  # elixir is a shell wrapper: read the BEAM child if there is one
  local beam; beam=$(pgrep -P "$pid" beam 2>/dev/null | head -1 || true)
  awk '/VmRSS/ {print $2}' "/proc/${beam:-$pid}/status"
  wait "$pid"
}
echo "Measuring memory per idle actor (1,000,000 actors each)..."
vb=$(rss_kb 0 "$TMP/vira_memory"); vf=$(rss_kb 1000000 "$TMP/vira_memory")
eb=$(rss_kb 0 elixir --erl "$ERL_FLAGS" "$HERE/memory.exs")
ef=$(rss_kb 1000000 elixir --erl "$ERL_FLAGS" "$HERE/memory.exs")
vira_mem=$(( (vf - vb) * 1024 / 1000000 ))
elixir_mem=$(( (ef - eb) * 1024 / 1000000 ))

commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
stamp=$(date -u +%FT%TZ)
REPORT="$HERE/results-$(date +%F).md"
TSV="$HERE/results.tsv"

# metric | higher-is-better? | description
METRICS="send1_mps:1:1M fire-and-forget msgs to 1 actor (msg/s)
sendN_mps:1:1M msgs over 64 actors (msg/s)
reqrep_ns:0:sync request/reply round trip (ns)
fanout_rps:1:4 parallel requests, await all (replies/s)
ring_hops:1:1000-actor ring, main-driven hops (hops/s)
ring_native_hops:1:1000-process ring, process-to-process (hops/s)
spawn_ns:0:spawn + 1 msg + drop, per actor (ns)
mass_ms:0:100k live actors x 10 msgs + drain (ms)
chaos_us:0:5000 jobs, 1/7 crash + supervised restart (us)
par_seq_us:0:$W CPU jobs one at a time (us)
par_us:0:$W CPU jobs in parallel (us)
par_x100:1:parallel speedup x100 (ideal = $((W * 100)))
bytes_per_actor:0:RSS per idle actor, 1M actors (bytes)"

{
  echo "# Vira vs Elixir — actors & concurrency"
  echo
  echo "- Date: $stamp, commit \`$commit\`"
  echo "- Machine: $CORES cores ($(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs))"
  echo "- Threads: VIRA_WORKERS=$W vs BEAM \`+S $W:$W\`; median of $RUNS runs"
  echo "- Vira: $("$VIRA" version 2>/dev/null | head -1); Elixir: $EXV"
  echo
  echo "| Test | Vira | Elixir | Result |"
  echo "|---|---:|---:|---|"
} >"$REPORT"

printf '\n%-18s %14s %14s   %s\n' "metric" "vira" "elixir" "result"
printf '%s\n' "--------------------------------------------------------------------------"
while IFS=: read -r m hib desc; do
  if [ "$m" = bytes_per_actor ]; then v=$vira_mem; e=$elixir_mem
  else v=$(median "$TMP/vira.out" "$m"); e=$(median "$TMP/elixir.out" "$m"); fi
  if [ "$v" = "-" ] || [ "$e" = "-" ] || [ "$v" = 0 ] || [ "$e" = 0 ]; then
    verdict="n/a (Vira actors cannot hold each other)"
    [ "$m" != ring_native_hops ] && verdict="n/a"
  else
    verdict=$(awk -v v="$v" -v e="$e" -v h="$hib" 'BEGIN {
      r = (h == 1) ? v / e : e / v
      if (r >= 1) printf "Vira %.2fx better", r; else printf "Elixir %.2fx better", 1 / r }')
  fi
  printf '%-18s %14s %14s   %s\n' "$m" "$v" "$e" "$verdict"
  echo "| $desc | $v | $e | $verdict |" >>"$REPORT"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$stamp" "$commit" "$W" "$m" "$v" "$e" >>"$TSV"
done <<<"$METRICS"

echo
echo "Report: $REPORT"
echo "History: $TSV"
