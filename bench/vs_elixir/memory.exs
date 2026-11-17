# Resident memory per idle process, the Elixir counterpart of
# scripts/bench/actor_memory.vir: ACTORS processes, one message each, then sleep so
# run.sh can read VmRSS. Run with ACTORS=0 and ACTORS=1000000 and divide the difference.

defmodule M do
  def cell(v) do
    receive do
      {:set, x} -> cell(x)
      {:get, from, ref} -> send(from, {ref, v}); cell(v)
    end
  end
end

n = String.to_integer(System.get_env("ACTORS") || "1000000")
cells = for i <- 0..(n - 1)//1, do: (p = spawn(fn -> M.cell(0) end); send(p, {:set, i}); p)

if n > 0 do
  ref = make_ref()
  send(List.last(cells), {:get, self(), ref})
  receive do {^ref, _} -> :ok end
end

IO.puts("READY #{n}")
Process.sleep(1500)
