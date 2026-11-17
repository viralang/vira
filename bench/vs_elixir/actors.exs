# Elixir side of the Vira-vs-Elixir actor benchmark (bench/vs_elixir/run.sh).
# Each test mirrors the same-named test in actors.vir: same message pattern, counts
# and self-check. Raw processes with send/receive are used (the fastest BEAM path);
# the chaos test uses a GenServer under a Supervisor, the OTP way to restart.
#   ring_native_hops  each process forwards the token to its neighbour itself (Vira: Ref[T])
#   pubsub_mps        Registry.dispatch (the local pub/sub Phoenix.PubSub is built on) to 16
#                     subscribers, 100,000 publishes (Vira: eventbus.Topic)
#
#   elixir --erl "+P 4000000 +S 4:4" actors.exs

:logger.set_primary_config(:level, :none)

defmodule B do
  def now, do: System.monotonic_time(:microsecond)

  # --- Sink: count / total / echo -------------------------------------------------
  def sink(count) do
    receive do
      {:hit, v} -> sink(count + v)
      {:total, from, ref} -> send(from, {ref, count}); sink(count)
      {:echo, from, ref, v} -> send(from, {ref, v + 1}); sink(count)
      :stop -> :ok
    end
  end

  def call(pid, msg_fun) do
    ref = make_ref()
    send(pid, msg_fun.(self(), ref))
    receive do
      {^ref, v} -> v
    end
  end

  def total(pid), do: call(pid, fn from, ref -> {:total, from, ref} end)

  def send_loop(_pid, 0), do: :ok
  def send_loop(pid, n), do: (send(pid, {:hit, 1}); send_loop(pid, n - 1))

  def bench_send1(n) do
    s = spawn(fn -> sink(0) end)
    t0 = now()
    send_loop(s, n)
    total = total(s)
    el = now() - t0
    if total != n, do: IO.puts("BENCH_ERROR send1 total=#{total}")
    send(s, :stop)
    el
  end

  def bench_send_n(n, actors) do
    sinks = for _ <- 1..actors, do: spawn(fn -> sink(0) end)
    tup = List.to_tuple(sinks)
    t0 = now()
    send_rr(tup, actors, 0, n)
    total = Enum.reduce(sinks, 0, fn p, acc -> acc + total(p) end)
    el = now() - t0
    if total != n, do: IO.puts("BENCH_ERROR sendN total=#{total}")
    Enum.each(sinks, &send(&1, :stop))
    el
  end

  defp send_rr(_t, _a, i, n) when i == n, do: :ok
  defp send_rr(t, a, i, n) do
    send(elem(t, rem(i, a)), {:hit, 1})
    send_rr(t, a, i + 1, n)
  end

  def bench_reqrep(n) do
    s = spawn(fn -> sink(0) end)
    t0 = now()
    acc = reqrep_loop(s, 0, n)
    el = now() - t0
    if acc != n, do: IO.puts("BENCH_ERROR reqrep acc=#{acc}")
    send(s, :stop)
    el
  end

  defp reqrep_loop(_s, acc, 0), do: acc
  defp reqrep_loop(s, acc, n) do
    v = call(s, fn from, ref -> {:echo, from, ref, acc} end)
    reqrep_loop(s, v, n - 1)
  end

  # --- Node: ping / job ------------------------------------------------------------
  def worker(id) do
    receive do
      {:ping, from, ref, v} -> send(from, {ref, v + 1}); worker(id)
      :stop -> :ok
    end
  end

  def bench_fanout(rounds) do
    ws = for i <- 0..3, do: spawn(fn -> worker(i) end)
    [w0, w1, w2, w3] = ws
    t0 = now()
    ok = fanout_loop(w0, w1, w2, w3, 0, rounds, 0)
    el = now() - t0
    if ok != rounds * 4, do: IO.puts("BENCH_ERROR fanout ok=#{ok}")
    Enum.each(ws, &send(&1, :stop))
    el
  end

  defp fanout_loop(_, _, _, _, b, rounds, ok) when b == rounds, do: ok
  defp fanout_loop(w0, w1, w2, w3, b, rounds, ok) do
    me = self()
    r0 = make_ref(); send(w0, {:ping, me, r0, b})
    r1 = make_ref(); send(w1, {:ping, me, r1, b})
    r2 = make_ref(); send(w2, {:ping, me, r2, b})
    r3 = make_ref(); send(w3, {:ping, me, r3, b})
    v0 = receive do {^r0, v} -> v end
    v1 = receive do {^r1, v} -> v end
    v2 = receive do {^r2, v} -> v end
    v3 = receive do {^r3, v} -> v end
    ok = if v0 > 0 and v1 > 0 and v2 > 0 and v3 > 0, do: ok + 4, else: ok
    fanout_loop(w0, w1, w2, w3, b + 1, rounds, ok)
  end

  # --- Ring driven by main (same shape as Vira) ------------------------------------
  def hop(seen) do
    receive do
      {:pass, from, ref, left} -> send(from, {ref, left - 1}); hop(seen + 1)
      :stop -> :ok
    end
  end

  def bench_ring(size, laps) do
    ring = List.to_tuple(for _ <- 1..size, do: spawn(fn -> hop(0) end))
    t0 = now()
    ring_loop(ring, size, 0, size * laps)
    el = now() - t0
    for i <- 0..(size - 1), do: send(elem(ring, i), :stop)
    el
  end

  defp ring_loop(_r, _s, _i, 0), do: :ok
  defp ring_loop(r, s, i, left) do
    ref = make_ref()
    send(elem(r, i), {:pass, self(), ref, left})
    left = receive do {^ref, v} -> v end
    i = if i + 1 == s, do: 0, else: i + 1
    ring_loop(r, s, i, left)
  end

  # --- Native ring: processes forward to each other (Elixir only) ------------------
  def ring_proc do
    receive do
      {:next, nxt} -> ring_fwd(nxt)
    end
  end

  defp ring_fwd(nxt) do
    receive do
      {:token, 0, main} -> send(main, :ring_done); ring_fwd(nxt)
      {:token, left, main} -> send(nxt, {:token, left - 1, main}); ring_fwd(nxt)
      :stop -> :ok
    end
  end

  def bench_ring_native(size, laps) do
    procs = for _ <- 1..size, do: spawn(&ring_proc/0)
    nexts = tl(procs) ++ [hd(procs)]
    Enum.zip(procs, nexts) |> Enum.each(fn {p, n} -> send(p, {:next, n}) end)
    t0 = now()
    send(hd(procs), {:token, size * laps - 1, self()})
    receive do :ring_done -> :ok end
    el = now() - t0
    Enum.each(procs, &send(&1, :stop))
    el
  end

  # --- Pub/sub: Registry with duplicate keys, dispatch in the publisher ------------
  def wait_total(pid, want) do
    if total(pid) < want, do: (Process.sleep(1); wait_total(pid, want)), else: :ok
  end

  def subscriber(parent) do
    Registry.register(BenchBus, "bench", nil)
    send(parent, :subscribed)
    sink(0)
  end

  def bench_pubsub(consumers, msgs) do
    {:ok, reg} = Registry.start_link(keys: :duplicate, name: BenchBus,
                                     partitions: System.schedulers_online())
    me = self()
    subs = for _ <- 1..consumers, do: spawn(fn -> subscriber(me) end)
    for _ <- subs, do: (receive do :subscribed -> :ok end)
    t0 = now()
    pub_loop(msgs)
    Enum.each(subs, &wait_total(&1, msgs))
    el = now() - t0
    Enum.each(subs, &send(&1, :stop))
    Process.unlink(reg)
    Process.exit(reg, :shutdown)
    el
  end

  defp pub_loop(0), do: :ok
  defp pub_loop(n) do
    Registry.dispatch(BenchBus, "bench", fn entries ->
      for {pid, _} <- entries, do: send(pid, {:hit, 1})
    end)
    pub_loop(n - 1)
  end

  # --- Spawn: create, one message, process exits -----------------------------------
  def bench_spawn(n) do
    t0 = now()
    spawn_loop(n)
    el = now() - t0
    el
  end

  defp spawn_loop(0), do: :ok
  defp spawn_loop(n) do
    pid = spawn(fn -> receive do {:hit, _} -> :ok end end)
    send(pid, {:hit, 1})
    spawn_loop(n - 1)
  end

  # --- Mass: 100k live processes x 10 messages, then a sync drain each -------------
  def bench_mass(actors, per) do
    t0 = now()
    cells = List.to_tuple(for _ <- 1..actors, do: spawn(fn -> sink(0) end))
    for _ <- 1..per, i <- 0..(actors - 1), do: send(elem(cells, i), {:hit, 1})
    total = Enum.reduce(0..(actors - 1), 0, fn i, acc -> acc + total(elem(cells, i)) end)
    el = now() - t0
    if total != actors * per, do: IO.puts("BENCH_ERROR mass total=#{total}")
    for i <- 0..(actors - 1), do: send(elem(cells, i), :stop)
    el
  end

  # --- Chaos: supervised GenServer, 1 in 7 jobs crash it --------------------------
  def bench_chaos(n) do
    {:ok, sup} =
      Supervisor.start_link([%{id: ChaosNode, start: {ChaosNode, :start_link, []}}],
        strategy: :one_for_one, max_restarts: 1_000_000, max_seconds: 1)
    t0 = now()
    trapped = chaos_loop(0, n, 0)
    el = now() - t0
    if trapped != div(n + 6, 7), do: IO.puts("BENCH_ERROR chaos trapped=#{trapped}")
    Process.unlink(sup)
    Process.exit(sup, :shutdown)
    el
  end

  defp chaos_loop(c, n, trapped) when c == n, do: trapped
  defp chaos_loop(c, n, trapped) do
    t =
      try do
        GenServer.call(ChaosNode, {:job, c, rem(c, 7) == 0})
        trapped
      catch
        :exit, {:noproc, _} -> throw(:retry)
        :exit, _ -> trapped + 1
      end
    chaos_loop(c + 1, n, t)
  catch
    :retry -> wait_restart(); chaos_loop(c, n, trapped)
  end

  defp wait_restart do
    if Process.whereis(ChaosNode) == nil, do: (:erlang.yield(); wait_restart())
  end

  # --- Parallel CPU: same integer loop as Cruncher.crunch --------------------------
  def crunch(acc, i, iters) when i == iters, do: acc
  def crunch(acc, i, iters), do: crunch(inner(acc, i, 0), i + 1, iters)
  defp inner(acc, _i, 5000), do: acc
  defp inner(acc, i, j), do: inner(rem(acc + (i * 3 + j * 7), 1_000_007), i, j + 1)

  def cruncher(id, result) do
    receive do
      {:crunch, iters} -> cruncher(id, crunch(id, 0, iters))
      {:get, from, ref} -> send(from, {ref, result}); cruncher(id, result)
      :stop -> :ok
    end
  end

  def get(pid), do: call(pid, fn from, ref -> {:get, from, ref} end)

  def run_crunch(pool, iters, serial) do
    if serial do
      Enum.reduce(pool, 0, fn p, acc -> send(p, {:crunch, iters}); acc + get(p) end)
    else
      Enum.each(pool, &send(&1, {:crunch, iters}))
      Enum.reduce(pool, 0, fn p, acc -> acc + get(p) end)
    end
  end
end

defmodule ChaosNode do
  use GenServer
  def start_link, do: GenServer.start_link(__MODULE__, 0, name: __MODULE__)
  @impl true
  def init(ops), do: {:ok, ops}
  @impl true
  def handle_call({:job, _v, true}, _from, _ops), do: raise("injected fault")
  def handle_call({:job, v, false}, _from, ops), do: {:reply, v * 2, ops + 1}
end

env_int = fn name, default ->
  case System.get_env(name) do
    nil -> default
    s -> String.to_integer(s)
  end
end

par = env_int.("PAR", 4)
crunch_iters = env_int.("CRUNCH", 600)

n = 1_000_000
s1 = B.bench_send1(n)
sn = B.bench_send_n(n, 64)
rr_n = 200_000
rr = B.bench_reqrep(rr_n)
fan_rounds = 20_000
fo = B.bench_fanout(fan_rounds)
ring_size = 1000
laps = 1000
r = B.bench_ring(ring_size, laps)
rn = B.bench_ring_native(ring_size, laps)
ps_consumers = 16
ps_msgs = 100_000
ps = B.bench_pubsub(ps_consumers, ps_msgs)
spawn_n = 100_000
sp = B.bench_spawn(spawn_n)
ms = B.bench_mass(100_000, 10)
ch = B.bench_chaos(5000)

pool = for i <- 0..(par - 1), do: spawn(fn -> B.cruncher(i, 0) end)
t = B.now()
sum_seq = B.run_crunch(pool, crunch_iters, true)
seq_us = B.now() - t
t = B.now()
sum_par = B.run_crunch(pool, crunch_iters, false)
par_us = B.now() - t
if sum_seq != sum_par, do: IO.puts("BENCH_ERROR crunch seq=#{sum_seq} par=#{sum_par}")

IO.puts(
  "BENCH send1_mps=#{div(n * 1_000_000, s1)} sendN_mps=#{div(n * 1_000_000, sn)}" <>
    " reqrep_ns=#{div(rr * 1000, rr_n)} fanout_rps=#{div(fan_rounds * 4 * 1_000_000, fo)}" <>
    " ring_hops=#{div(ring_size * laps * 1_000_000, r)} ring_native_hops=#{div(ring_size * laps * 1_000_000, rn)}" <>
    " pubsub_mps=#{div(ps_consumers * ps_msgs * 1_000_000, ps)}" <>
    " spawn_ns=#{div(sp * 1000, spawn_n)} mass_ms=#{div(ms, 1000)} chaos_us=#{ch}" <>
    " par_seq_us=#{seq_us} par_us=#{par_us} par_x100=#{div(seq_us * 100, par_us)} crunch_sum=#{sum_par}"
)
