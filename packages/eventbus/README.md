# eventbus

Vert.x-style messaging between actors, built on `Ref[T]` (a copyable actor address that does
not own the actor).

```v
import eventbus

pub struct Order { pub: id int  amount int }

pub actor Billing { mut: total int = 0 }
pub fn (mut b Billing) charge(o Order) { b.total += o.amount }

fn main() {
    mut orders := eventbus.Topic[Order]{ name: 'orders.new' }
    bus := orders.ref()                 // Ref[Topic[Order]]: hand it to any actor or closure

    mut billing := Billing{}
    br := billing.ref()
    id := bus <- consumer(fn (o Order) bool {
        br <- charge(o)                 // runs on the worker pool, one message at a time
        return br.alive()               // false removes this consumer
    }) or { 0 }

    bus <- publish(Order{ id: 1, amount: 10 })   // every consumer; returns immediately
    bus <- send(Order{ id: 2, amount: 5 })       // one consumer, round-robin
    _ := bus <- unregister(id) or { false }
}
```

## Topic[T]

| Message | Effect |
| :--- | :--- |
| `consumer(h fn (T) bool) int` | Registers a handler, returns its id. The handler returns `false` to unsubscribe (normally `ref.alive()`). |
| `unregister(id int) bool` | Removes a consumer. |
| `publish(v T) int` | Delivers to every consumer; returns how many were reached. |
| `send(v T) bool` | Delivers to one consumer, rotating; `false` if none is left. |
| `consumers() int`, `stats() TopicStats` | Counters: consumers, published, delivered, removed. |

The topic is an actor, so registration and delivery are serialized without locks. Send with
`bus <- publish(v)` (fire-and-forget) and the publisher never waits. Messages from one sender
are delivered in the order they were sent.

## Service[T, R] (request/reply)

```v
mut svc := eventbus.Service[int, int]{ name: 'pricing' }
sr := svc.ref()
pr := pricer.ref()
_ := sr <- handler(fn (q int) int { return pr <- price(q) or { 0 } },
                   fn () bool { return pr.alive() }) or { 0 }

total := eventbus.request(sr, 3) or { 0 }   // round-robin over live handlers
```

`request` picks a handler from the service and runs it in the caller's own context, so the
service actor is never blocked waiting for a reply. It fails with
`no handler registered for this service` when no live handler is left.

## Notes

- Values are copied for each consumer (the same deep copy as any actor message): no shared memory.
- A consumer whose actor was destroyed is removed on the next delivery to it.
- Mailboxes are unbounded: a publisher that outruns its consumers grows their queues.
