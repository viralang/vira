# Tutorial 44: Actor References (`Ref[T]`) & the Event Bus

Until v0.7.9 an actor could only be reached through the one variable, field or slice element
that owns it. Actors could not message each other unless their owner relayed every message.
This tutorial covers the two features that lift that limit:

- **`Ref[T]`**: a copyable address of an actor that does not own it.
- **`import eventbus`**: Vert.x-style typed topics (publish to all, send to one, request/reply)
  built on `Ref[T]`.

Runnable examples:

| Example | Use case |
| :-- | :-- |
| [`49_actor_references.vir`](../examples/49_actor_references.vir) | Workers reporting to their manager, a pipeline, a ring, dead references, restarts |
| [`50_eventbus_order_pipeline.vir`](../examples/50_eventbus_order_pipeline.vir) | An order backend: billing, inventory, WhatsApp notifications, audit, a courier work queue and a price service |
| [`51_eventbus_chat_rooms.vir`](../examples/51_eventbus_chat_rooms.vir) | Chat rooms: join/leave, @mention filters, a bot that replies, users who disconnect |

| Section | In one line |
| :-- | :-- |
| [1. Why references](#1-why-references) | One owner per actor, any number of addresses. |
| [2. `Ref[T]` basics](#2-reft-basics) | `a.ref()`, the three send forms, `alive()`, the null Ref. |
| [3. When the actor is gone](#3-when-the-actor-is-gone) | Sends fail with `actor is gone`; restarts keep the Ref. |
| [4. Patterns with Refs](#4-patterns-with-refs) | Child → parent, pipelines, rings, waiting from `main`. |
| [5. Topics: publish and send](#5-topics-publish-and-send) | `consumer`, `publish`, `send`, `unregister`, `stats`. |
| [6. Services: request/reply](#6-services-requestreply) | `eventbus.request(svc, v)` with round-robin handlers. |
| [7. Use cases](#7-use-cases) | Order pipeline, work queue, chat, and more. |
| [8. Guarantees](#8-what-the-bus-guarantees-and-what-it-does-not) | Ordering, copies, delivery, back-pressure. |
| [9. Common mistakes](#9-common-mistakes) | Short-lived consumers, heavy handlers, self-waits. |
| [10. Performance](#10-performance) | Numbers against Elixir. |

---

## 1. Why references

Every actor has exactly one **owner**. When the owner goes out of scope the actor finishes
its queued messages and is destroyed. Ownership is what makes actors leak-free, but it means
an actor cannot be copied, so before `Ref[T]` two actors could not hold each other:

```vira
// Not possible before Ref[T]: a ring of actors, each holding its neighbour.
// The owner (main) had to relay every hop:
left = ring[i] <- pass(left) or { 0 }   // a round trip through main per hop
```

A **reference** separates *who owns the actor* from *who can talk to it*. The owner still
decides when the actor dies. Anyone with a Ref can send to it until then, and gets a clean
error afterwards.

---

## 2. `Ref[T]` basics

```vira
pub actor Counter {
mut:
    n int = 0
}

pub fn (mut c Counter) add(v int) { c.n = c.n + v }
pub fn (mut c Counter) get() int { return c.n }

fn main() {
    mut counter := Counter{}        // main owns the actor
    r := counter.ref()              // Ref[Counter]: 8 bytes, copyable
    r2 := r                         // copies are fine: both address the same actor

    r <- add(1)                     // fire-and-forget
    f := r2 <- add(2)               // future
    _ := f.await() or { false }
    v := r <- get() or { 0 }        // synchronous call
    println('${v} alive=${r.alive()}')
}
```

| You can | Example |
| :-- | :-- |
| Take a Ref from the owner or from inside the actor's own handler | `w.ref()`, `return h.ref()` |
| Send in all three forms | `r <- m()`, `f := r <- m()`, `r <- m() or { ... }` |
| Send from an **immutable** binding | `r := a.ref()` (no `mut` needed: a Ref never changes) |
| Store it in actor fields, structs, slices and maps | `mut next Ref[Hop]` |
| Capture it in closures and pass it in messages | `fn (o Order) bool { r <- charge(o) ... }` |
| Check the actor | `r.alive()` is false once the actor is destroyed or stopped |
| Write the null Ref | `Ref[Counter]{}` (also the zero value of a `Ref` field) |

| You cannot | Why |
| :-- | :-- |
| Call a method directly: `r.add(1)` | Like the actor itself, a Ref is reached only by messages (compile error). |
| Write `Ref[SomeStruct]` | Compile error: `Ref[T] needs an actor type`. |
| Read mutable fields through it | State is private to the actor; ask with a message. |
| Keep an actor alive with it | A Ref does not own anything. See [mistake 1](#mistake-1-the-actor-dies-at-the-end-of-the-function). |

---

## 3. When the actor is gone

```vira
fn make_temp() Ref[Counter] {
    mut tmp := Counter{}
    return tmp.ref()            // tmp is destroyed when the function returns
}

stale := make_temp()
println('${stale.alive()}')    // false
stale <- add(1)                 // fire-and-forget: silently dropped
_ := stale <- get() or {
    println(err)                // actor is gone
    0
}
```

The runtime guarantees:

- **No dangling pointers.** Every Ref carries a *generation*. Destroying the actor bumps it,
  so an old Ref can never reach freed memory, or a new actor that reuses the slot.
- **Nothing that got in is lost.** Destroying an actor waits for senders that are halfway
  through enqueueing, then runs every queued message. It never waits for a *reply*.
- **Restarts keep the Ref.** A supervised crash (`panic`, a bounds error, division by zero)
  resets the actor's mutable fields, and every Ref keeps pointing at it:

```vira
a := pr <- parse('41') or { 0 }
_ := pr <- parse('oops') or {        // crash + supervised restart
    println('parse failed: ${err}')
    0
}
b := pr <- parse('1') or { 0 }       // the same Ref still works
```

After `max_restarts` consecutive crashes the actor stops: sends fail with
`Actor has stopped due to unhandled faults` and `alive()` turns false.

---

## 4. Patterns with Refs

All four are in [`49_actor_references.vir`](../examples/49_actor_references.vir).

### Child → parent

The owner hands each child a Ref to itself, so children can report back:

```vira
pub actor Worker {
pub:
    id int
    boss Ref[Manager]
}

pub fn (mut w Worker) run(job int) {
    w.boss <- report(w.id, job * job)
}

mut manager := Manager{}
mut workers := []Worker{}
for mut i := 0; i < 4; i = i + 1 {
    workers.push(Worker{ id: i, boss: manager.ref() })
}
```

### Pipeline

Each stage holds a Ref to the next stage. A null Ref marks the end:

```vira
pub fn (mut s Stage) push(text string) {
    stamped := '${text}>${s.name}'
    if s.next.alive() {
        s.next <- push(stamped)
    } else {
        println('pipeline output: ${stamped}')
    }
}
```

### Ring

```vira
pub fn (mut h Hop) pass(left int, done Ref[Latch]) {
    if left == 0 {
        done <- tick()
        return
    }
    h.next <- pass(left - 1, done)    // straight to the neighbour, no relay through main
}
```

### Waiting from `main`: the latch

`main` is not an actor, so nothing can send *to* it. To wait for work happening between
actors, give the actors a Ref to a counter and poll it:

```vira
pub actor Latch {
mut:
    ticks int = 0
}
pub fn (mut l Latch) tick() { l.ticks = l.ticks + 1 }
pub fn (mut l Latch) count() int { return l.ticks }

fn wait_for(mut l Latch, want int) {
    mut got := 0
    for got < want {
        got = l <- count() or { 0 }
        if got < want {
            time_sleep_ms(1)
        }
    }
}
```

Inside actors you never need this: an actor just handles the next message when it arrives.

---

## 5. Topics: publish and send

A `Topic[T]` is an **address** for messages of type `T`. It is itself an actor, so registering
and delivering are serialized without locks. Hand its Ref to anyone who publishes:

```vira
import eventbus

mut orders_topic := eventbus.Topic[Order]{ name: 'orders.placed' }
orders := orders_topic.ref()            // Ref[eventbus.Topic[Order]]
```

### Consumers

A consumer is a closure `fn (T) bool`. It forwards the message to an actor and returns whether
that actor is still there; returning `false` removes the consumer:

```vira
br := billing.ref()
id := orders <- consumer(fn (o Order) bool {
    br <- charge(o)          // runs later, on the worker pool, one message at a time
    return br.alive()
}) or { 0 }
```

A consumer keeps receiving **every message that arrives, for as long as it is registered**.
Each message is a separate asynchronous run of the actor's handler. Handlers of one actor never
overlap, and different consumers run in parallel on different cores.

The closure can also **filter**, as the chat example does for @mentions:

```vira
_ := room <- consumer(fn (m ChatMsg) bool {
    if !m.text.contains('@karim') {
        return karim.alive()     // skip this one, stay subscribed
    }
    karim <- receive(m)
    return karim.alive()
}) or { 0 }
```

### The messages a topic understands

| Message | Effect |
| :-- | :-- |
| `consumer(h fn (T) bool) int` | Register; returns an id for `unregister`. |
| `publish(v T) int` | Deliver to **every** consumer; returns how many were reached. |
| `send(v T) bool` | Deliver to **one** consumer, rotating between them (a work queue). |
| `unregister(id int) bool` | Remove a consumer. |
| `consumers() int` | How many are registered (dead ones are removed at their next delivery). |
| `stats() TopicStats` | `consumers`, `published`, `delivered`, `removed`. |

Publish with a fire-and-forget send and the publisher never waits:

```vira
orders <- publish(Order{ id: 1, customer: 'Mona', sku: 'shirt', qty: 2, total: 500 })
n := orders <- publish(order) or { 0 }   // or wait, to learn how many consumers got it
```

**Actors publish too.** Give an actor a Ref to a topic and it can emit the next event itself.
In the order example, `Inventory` reserves stock and then sends a `ShipJob`:

```vira
pub actor Inventory {
pub:
    shipping Ref[eventbus.Topic[ShipJob]]
...
pub fn (mut inv Inventory) reserve(o Order) {
    inv.stock[o.sku] = inv.stock[o.sku] - o.qty
    inv.shipping <- send(ShipJob{ order_id: o.id, city: 'Cairo' })
}
```

---

## 6. Services: request/reply

A `Service[T, R]` answers a request of type `T` with an `R`. Each handler comes with an `alive`
check, so dead handlers are skipped and removed:

```vira
mut price_service := eventbus.Service[string, int]{ name: 'catalog.price' }
prices := price_service.ref()

r1 := p1.ref()
_ := prices <- handler(fn (sku string) int {
    return r1 <- price(sku) or { 0 }
}, fn () bool {
    return r1.alive()
}) or { 0 }

unit := eventbus.request(prices, 'shirt') or { 0 }    // round-robin over live handlers
```

`request` asks the service for the next handler and **runs it in the caller's own context**
(main, or the actor calling `request`). The service actor never sits waiting for a reply, so
one slow request cannot hold up everyone else's. With no live handler it fails with
`no handler registered for this service`.

---

## 7. Use cases

### Decoupled order processing ([example 50](../examples/50_eventbus_order_pipeline.vir))

`checkout` only knows the `orders.placed` address. Billing, inventory, WhatsApp
notifications, audit and a live dashboard subscribe on their own. Adding a fraud check or a
loyalty program is one more `consumer` call, with no change to the checkout code.

```
checkout ──publish──► orders.placed ──► Billing
                                   ├──► Inventory ──send──► shipping.jobs ──► courier 1 / 2 / 3
                                   ├──► Notifier (WhatsApp)
                                   ├──► Audit
                                   └──► Dashboard (closed → removed automatically)
checkout ──request──► catalog.price ──► price-1 / price-2 (round-robin)
```

### Work queues

`send` hands each job to one consumer in turn: couriers, email senders, image resizers,
LLM inference workers. Each worker processes one job at a time, and workers run in parallel.

### Chat, presence and live updates ([example 51](../examples/51_eventbus_chat_rooms.vir))

One topic per room. `unregister` when a user leaves. A user whose connection actor is
destroyed disappears from every room on its own. Filters in the handler give
"mentions only" or "only orders above 1000" subscriptions.

### More places the pattern fits

| Use case | Shape |
| :-- | :-- |
| **Cache invalidation** | `Topic[string]` of keys; every cache actor subscribes and evicts. |
| **Metrics & telemetry fan-out** | Handlers publish events; a Prometheus exporter and a log shipper both consume. |
| **Streaming LLM tokens** | A generation actor publishes tokens to a `Topic[string]`; each SSE client connection is a consumer (ViraServe / ViraLama). |
| **Plugin hooks** | Plugins register consumers on `app.started`, `user.created`, … without the core knowing them. |
| **Config reload** | Publish the new config once; every service actor applies it. |
| **Webhook / WhatsApp dispatch** | Business events published once, delivered by a pool of sender actors via `send`. |

---

## 8. What the bus guarantees, and what it does not

| Property | Behaviour |
| :-- | :-- |
| **Order** | Messages from one sender through one topic reach each consumer in the order sent. Across different topics or senders there is no order: in example 51, Ali sees an `#ops` message before an earlier `#general` one. |
| **Isolation** | Each consumer gets its own **copy** of the message (the same deep copy as any actor message), so nothing is shared. |
| **Delivery** | In-memory, at most once. A message to a consumer whose actor is gone is dropped. Nothing is persisted: use a database or queue if a message must survive a restart of the process. |
| **Back-pressure** | Mailboxes are **unbounded**. A publisher that outruns slow consumers makes their queues grow. For high-volume producers, have consumers acknowledge (e.g. tick a counter) and pace the producer. |
| **Failures** | A crash in a consumer's handler is supervised like any actor fault: the actor restarts, stays subscribed, and only the message that crashed is lost. |

---

## 9. Common mistakes

### Mistake 1: the actor dies at the end of the function

```vira
fn subscribe_dashboard(orders Ref[eventbus.Topic[Order]]) {
    mut d := Dashboard{}
    dr := d.ref()
    _ := orders <- consumer(fn (o Order) bool {
        dr <- show(o)
        return dr.alive()
    }) or { 0 }
}   // d is destroyed HERE: the consumer is removed at the next publish
```

A Ref does not keep an actor alive. Keep the actor where it should live: in `main`, in a
slice, or in a field of a longer-lived actor. Example 50 does this on purpose to show the
automatic clean-up.

### Mistake 2: doing real work inside the consumer closure

The consumer closure runs **inside the topic actor**, one delivery after another. Keep it to
forwarding (`r <- handle(v)`) plus an optional cheap filter. Slow work in the closure (a
synchronous call, I/O, a big computation) delays every other consumer of that topic.

### Mistake 3: waiting on the topic from its own consumer

A consumer closure that does `topic <- publish(v) or { ... }` makes the topic wait for itself.
Vira reports `deadlock avoided: an actor cannot wait on a message it sent to itself` instead
of hanging. Publish fire-and-forget (`topic <- publish(v)`), or publish from the consumer's
actor as the bot in example 51 does.

### Mistake 4: returning `true` forever

If the closure always returns `true`, a consumer whose actor is gone stays registered and
every message to it is dropped. Return `r.alive()`.

### Mistake 5: assuming `main` gets told when work finishes

Nothing can message `main`. Use a [latch](#waiting-from-main-the-latch) or ask an actor for
its state with a synchronous call.

---

## 10. Performance

From `bench/vs_elixir/run.sh` (2 cores, median of 5 runs; Elixir 1.14 / OTP 24):

| Test | Vira | Elixir |
| :-- | --: | --: |
| 1,000-actor ring, each actor forwards to its neighbour | 2.45M hops/s | 2.38M hops/s |
| Pub/sub, 16 consumers × 100k publishes | 2.59M deliveries/s | 0.77M (`Registry.dispatch`) |
| Memory per idle actor | 409 B | 2,751 B |

A send through a Ref costs two atomic operations on top of a direct send (pin and unpin of
the reference slot). Run the suite on your own machine:

```bash
bash bench/vs_elixir/run.sh
```

---

**Next:** the full API is in [`packages/eventbus/README.md`](../../packages/eventbus/README.md),
and the language rules in [LANGUAGE_GUIDE § Actor References & the Event Bus](../../LANGUAGE_GUIDE.md#actor-references-reft--the-event-bus).
