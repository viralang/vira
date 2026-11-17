# Vira vs. Elixir (Erlang/OTP 27 JIT) — Actor & Concurrency Benchmark

This document records the methodology, reproducible scripts, and observed results comparing **Vira**'s native C-ABI actor runtime against **Elixir 1.17.3 / Erlang/OTP 27 (BEAM JIT)** on the same 4-core Linux host.

---

## 1. Test Environment

| Component | Configuration |
| :--- | :--- |
| **Host OS / CPU** | Linux, 4 CPU cores |
| **Vira** | Commit `f279e84`, compiled with `vira build -prod`, `VIRA_WORKERS=4` |
| **Elixir / Erlang** | Elixir `1.17.3`, Erlang/OTP `27` (`erts-15.2.7.11`, `[smp:4:4]`, `[jit:ns]`) |
| **BEAM Flags** | `+S 4:4` (4 scheduler threads); `+P 2000000` for the 1,000,000-process memory test |
| **Iterations** | 5 runs per suite (median, min, and max reported) |

---

## 2. Results Summary

| Benchmark | Vira (median) | Elixir 1.17 / OTP 27 JIT (median) | Ratio |
| :--- | ---: | ---: | :--- |
| **`send1_mps`** *(1 → 1, 1M fire-and-forget msgs)* | **5,767,112 msgs/s**<br>*(min 5,319,120 / max 5,959,582)* | **4,129,501 msgs/s**<br>*(min 3,330,790 / max 4,910,698)* | **Vira 1.40x faster** |
| **`sendN_mps`** *(1 → 64 actors, 1M fire-and-forget msgs)* | **4,800,122 msgs/s**<br>*(min 4,393,731 / max 5,325,409)* | **2,003,594 msgs/s**<br>*(min 1,823,775 / max 2,150,755)* | **Vira 2.40x faster** |
| **`ring_hops`** *(1,000-actor ring, 1,000,000 total hops)* | **5,198,234 hops/s**<br>*(min 4,889,760 / max 5,385,290)* | **729,858 hops/s** *(awaited hop)*<br>**1,536,353 hops/s** *(direct forwarding)* | **Vira 7.12x faster** *(awaited)*<br>**Vira 3.38x faster** *(direct)* |
| **`spawn_ns`** *(spawn + 1 msg + drop, 100,000 actors)* | **271 ns/actor**<br>*(min 227 / max 425)* | **1,773 ns/actor**<br>*(min 1,634 / max 3,032)* | **Vira 6.54x faster** |
| **`bytes_per_actor`** *(1,000,000 idle actors, `VmRSS` delta)* | **408 bytes / actor** | **2,639 bytes / process** | **Vira 6.47x smaller** |

---

## 3. Benchmark Methodology

All five sub-benchmarks are implemented 1:1 between Vira ([`scripts/bench/actor_compare.vir`](../scripts/bench/actor_compare.vir), [`scripts/bench/actor_memory.vir`](../scripts/bench/actor_memory.vir)) and Elixir:

1. **`send1_mps` (1-to-1 Message Throughput)**:
   - Spawns 1 `Sink` actor/process.
   - Sends 1,000,000 fire-and-forget messages (`s <- hit(1)` in Vira; `send(pid, {:hit, 1})` in Elixir) and awaits a final `total` reply to ensure all 1,000,000 messages were processed.
   - Reports `1,000,000 / elapsed_seconds`.

2. **`sendN_mps` (1-to-64 Fan-Out Throughput)**:
   - Spawns 64 `Sink` actors/processes.
   - Sends 1,000,000 fire-and-forget messages round-robin (`i % 64`) and awaits the sum of `total` across all 64 actors.
   - Reports `1,000,000 / elapsed_seconds`.

3. **`ring_hops` (1,000-Actor Token Ring)**:
   - Creates 1,000 actors and passes a token `1,000` laps around the ring (`1,000,000` hops total).
   - **Awaited-hop variant** (exact match to Vira's single-owner model): the driver sends `pass(left)` and awaits the decremented count from each actor in turn.
   - **Direct-forwarding variant** (idiomatic BEAM): each process holds the PID of its neighbor and forwards `{:token, left - 1}` directly until `0`.

4. **`spawn_ns` (Spawn + Message + Teardown Latency)**:
   - Creates 100,000 actors/processes in a loop, sends 1 message to each, and drops/exits them.
   - Reports `elapsed_ns / 100,000`.

5. **`bytes_per_actor` (Idle Resident Memory Footprint)**:
   - Launches the process with `ACTORS=0` and measures `VmRSS` from `/proc/<pid>/status` once `READY 0` is printed.
   - Launches the process with `ACTORS=1000000`, spawns 1,000,000 idle actors/processes, sends each 1 message, and measures `VmRSS` once `READY 1000000` is printed.
   - Computes `(rss_1M_kb - rss_0_kb) * 1024 / 1,000,000`.

---

## 4. Reproducing the Benchmarks

### Run Vira

```bash
RUNS=5 WORKERS=4 ./scripts/bench_actor_compare.sh
```

### Run Elixir

Save the two Elixir scripts below and run them with 4 schedulers (`+S 4:4`):

#### `elixir_compare.exs`

```elixir
defmodule Bench do
  def sink_loop(count) do
    receive do
      {:hit, v} ->
        sink_loop(count + v)

      {:total, from} ->
        send(from, {:total_reply, count})
        :ok
    end
  end

  def sink_once do
    receive do
      {:hit, _v} -> :ok
    end
  end

  def hop_loop(seen) do
    receive do
      {:pass, left, from} ->
        send(from, {:passed, left - 1})
        hop_loop(seen + 1)

      :stop ->
        :ok
    end
  end

  def ring_node_loop(next_pid, owner) do
    receive do
      {:token, 0} ->
        send(owner, :ring_done)
        ring_node_loop(next_pid, owner)

      {:token, left} ->
        send(next_pid, {:token, left - 1})
        ring_node_loop(next_pid, owner)

      {:set_next, pid} ->
        ring_node_loop(pid, owner)

      :stop ->
        :ok
    end
  end

  def bench_send1(n) do
    pid = spawn(fn -> sink_loop(0) end)
    t0 = System.monotonic_time(:microsecond)
    send1_loop(pid, n)
    send(pid, {:total, self()})
    total =
      receive do
        {:total_reply, v} -> v
      end
    elapsed = System.monotonic_time(:microsecond) - t0
    if total != n, do: IO.puts(:stderr, "BENCH_ERROR send1 total=#{total}")
    elapsed
  end

  defp send1_loop(_pid, 0), do: :ok
  defp send1_loop(pid, i) do
    send(pid, {:hit, 1})
    send1_loop(pid, i - 1)
  end

  def bench_send_n(n, actors) do
    sinks = List.to_tuple(for _ <- 0..(actors - 1), do: spawn(fn -> sink_loop(0) end))
    t0 = System.monotonic_time(:microsecond)
    send_n_loop(sinks, actors, 0, n)
    for i <- 0..(actors - 1) do
      send(elem(sinks, i), {:total, self()})
    end
    total =
      Enum.reduce(0..(actors - 1), 0, fn _, acc ->
        receive do
          {:total_reply, v} -> acc + v
        end
      end)
    elapsed = System.monotonic_time(:microsecond) - t0
    if total != n, do: IO.puts(:stderr, "BENCH_ERROR sendN total=#{total}")
    elapsed
  end

  defp send_n_loop(_sinks, _actors, n, n), do: :ok
  defp send_n_loop(sinks, actors, i, n) do
    send(elem(sinks, rem(i, actors)), {:hit, 1})
    send_n_loop(sinks, actors, i + 1, n)
  end

  def bench_ring_sync(size, laps) do
    ring = List.to_tuple(for _ <- 0..(size - 1), do: spawn(fn -> hop_loop(0) end))
    total_hops = size * laps
    t0 = System.monotonic_time(:microsecond)
    ring_sync_loop(ring, size, 0, total_hops)
    elapsed = System.monotonic_time(:microsecond) - t0
    for i <- 0..(size - 1), do: send(elem(ring, i), :stop)
    elapsed
  end

  defp ring_sync_loop(_ring, _size, _idx, 0), do: :ok
  defp ring_sync_loop(ring, size, idx, left) do
    send(elem(ring, idx), {:pass, left, self()})
    next_left =
      receive do
        {:passed, v} -> v
      end
    next_idx = if idx + 1 == size, do: 0, else: idx + 1
    ring_sync_loop(ring, size, next_idx, next_left)
  end

  def bench_ring_direct(size, laps) do
    owner = self()
    pids = for _ <- 0..(size - 1), do: spawn(fn -> ring_node_loop(nil, owner) end)
    shifted = tl(pids) ++ [hd(pids)]
    Enum.zip(pids, shifted)
    |> Enum.each(fn {p, next_p} -> send(p, {:set_next, next_p}) end)

    first = hd(pids)
    t0 = System.monotonic_time(:microsecond)
    send(first, {:token, size * laps})
    receive do
      :ring_done -> :ok
    end
    elapsed = System.monotonic_time(:microsecond) - t0
    Enum.each(pids, &send(&1, :stop))
    elapsed
  end

  def bench_spawn(n) do
    t0 = System.monotonic_time(:microsecond)
    spawn_loop(n)
    System.monotonic_time(:microsecond) - t0
  end

  defp spawn_loop(0), do: :ok
  defp spawn_loop(i) do
    p = spawn(fn -> sink_once() end)
    send(p, {:hit, 1})
    spawn_loop(i - 1)
  end

  def run do
    n = 1_000_000
    s1 = bench_send1(n)
    sn = bench_send_n(n, 64)
    ring_size = 1000
    laps = 1000
    rs = bench_ring_sync(ring_size, laps)
    rd = bench_ring_direct(ring_size, laps)
    spawn_n = 100_000
    sp = bench_spawn(spawn_n)

    send1_mps = div(n * 1_000_000, s1)
    sendn_mps = div(n * 1_000_000, sn)
    ring_sync_hops = div(ring_size * laps * 1_000_000, rs)
    ring_direct_hops = div(ring_size * laps * 1_000_000, rd)
    spawn_ns = div(sp * 1000, spawn_n)

    IO.puts("BENCH send1_mps=#{send1_mps} sendN_mps=#{sendn_mps} ring_sync_hops=#{ring_sync_hops} ring_direct_hops=#{ring_direct_hops} spawn_ns=#{spawn_ns}")
  end
end

Bench.run()
```

#### `elixir_memory.exs`

```elixir
defmodule MemBench do
  def cell_loop(v) do
    receive do
      {:set, x} -> cell_loop(x)
      {:ping, from} ->
        send(from, :pong)
        cell_loop(v)
    end
  end

  def run do
    n = String.to_integer(System.get_env("ACTORS") || "1000000")
    last =
      if n > 0 do
        Enum.reduce(0..(n - 1), nil, fn i, _acc ->
          p = spawn(fn -> cell_loop(0) end)
          send(p, {:set, i})
          p
        end)
      else
        nil
      end

    if last do
      send(last, {:ping, self()})
      receive do
        :pong -> :ok
      end
    end

    IO.puts("READY #{n}")
    Process.sleep(1500)
  end
end

MemBench.run()
```

#### Execution Harness

```bash
for i in $(seq 1 5); do
  elixir --erl "+S 4:4" elixir_compare.exs
done

ACTORS=0 elixir --erl "+S 4:4 +P 2000000" elixir_memory.exs &
# read VmRSS from /proc/$pid/status
ACTORS=1000000 elixir --erl "+S 4:4 +P 2000000" elixir_memory.exs &
# read VmRSS from /proc/$pid/status
```

