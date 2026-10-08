# LogiEdge — MQTT Topic Tree and QoS Design

Task C1/C2 supporting document. Report Section 2 reproduces the tree as a figure.

## Why a broker on the truck at all

An obvious simplification would be to have the sensor reader call the model
directly in-process and skip MQTT entirely. The broker earns its place for
three reasons specific to this deployment:

1. **Process isolation under a safety requirement.** The inference container
   can crash, be restarted by Ansible, or be swapped between M1/M2/M3 without
   the sensor readers losing a sample. The broker buffers across the gap.
2. **Multiple independent consumers.** Inference, the PSI drift monitor, the
   driver-alert handler and the store-and-forward uplink all consume the same
   streams at different rates. Point-to-point wiring would need four couplings
   where publish/subscribe needs none.
3. **The deployment boundary is already a message boundary.** When the pilot
   scales to 265 vehicles and gains a second model (the load-integrity vision
   model the Hailo-8L has headroom for), a new subscriber is added, not a new
   integration.

The broker is bound to `localhost`. It is not reachable from outside the
vehicle, which is part of the privacy argument in Task A1(d).

## Topic tree

```
logibridge/
└── trucks/
    └── {truck_id}/                      e.g. TRK-001
        ├── sensors/
        │   ├── temperature              QoS 0   1 Hz
        │   └── vibration                QoS 0   0.5 Hz
        ├── events/
        │   └── door                     QoS 1   discrete, retained
        ├── inference                    QoS 1   0.1 Hz  (one per 10 s window)
        ├── alerts                       QoS 2   on Warning or Critical
        └── status/
            ├── drift                    QoS 1   one per PSI evaluation
            └── health                   QoS 0   retained, 5 min heartbeat
```

`logibridge/trucks/{truck_id}/inference` is the topic the problem statement
mandates for inference results.

### Why the hierarchy is shaped this way

The truck ID sits high in the tree, above the data category. That ordering is
what makes the two subscriptions the system actually needs cheap:

- an on-truck consumer subscribes to `logibridge/trucks/TRK-001/#` and receives
  everything for its own vehicle and nothing for any other;
- the operations centre subscribes to `logibridge/trucks/+/alerts` and receives
  exactly the alerts across the whole fleet.

Putting the category first (`logibridge/sensors/trucks/+/temperature`) would
make per-truck isolation require one subscription per category, and per-truck
ACLs impossible to express as a single prefix.

## QoS per topic, and the reasoning

MQTT QoS is a per-message delivery guarantee, and each level costs round trips:
QoS 0 is fire-and-forget, QoS 1 guarantees at-least-once with a PUBACK, QoS 2
guarantees exactly-once with a four-step handshake. The right level follows
from what a lost or duplicated message would actually do.

| Topic | QoS | Justification |
|---|---|---|
| `sensors/temperature` | **0** | A lost sample is superseded by the next one 1 second later, and the 5-sample moving average plus a 30-sample window make any single reading nearly irrelevant. At 86,400 messages/day the handshake overhead of QoS 1 would be pure cost for no safety gain. |
| `sensors/vibration` | **0** | Same argument at 0.5 Hz. The window carries 15 samples; losing one perturbs the RMS by well under the sensor noise. |
| `events/door` | **1** | Discrete and not superseded — a missed OPEN is simply absent from history, and the door state is used both for the temperature model and for the chain-of-custody record. Duplicates are harmless because records are idempotent on `(truck_id, ts)`. Retained, so a restarting consumer learns the current door state immediately rather than waiting for the next transition. |
| `inference` | **1** | The PSI monitor's rolling window must not silently lose entries, or the drift statistic is computed over a different population than it reports. Duplicates are tolerable: they are deduplicated by the `seq` field. |
| `alerts` | **2** | The only topic that justifies the four-step handshake. An alert is the trigger for an operational escalation, and a duplicate at QoS 1 could dispatch a second response to the same event, or double-count a spoilage incident in the regulatory record. At roughly 86 alerts per truck per day the handshake cost is negligible. |
| `status/drift` | **1** | Must arrive for the ops centre to know a model is degrading; duplicates are harmless. |
| `status/health` | **0**, retained | A heartbeat superseded every 5 minutes. Retained so the dashboard shows last-known state for a truck currently in a dead zone rather than a blank. |

The pattern: **QoS rises with the cost of losing one message and falls with the
message rate.** High-rate telemetry is self-correcting, so QoS 0. Discrete
events are not, so QoS 1. Actions that cannot be taken twice get QoS 2.

## Payload schema

One schema for all sensor streams, fixed before preprocessing was written so
that the pipeline never had to change to accommodate a new stream:

```json
{
  "truck_id": "TRK-001",
  "ts":       1791190123.482,
  "sim_t":    247.0,
  "stream":   "temperature",
  "value":    4.0312,
  "unit":     "C",
  "mode":     "none"
}
```

`ts` is wall-clock epoch for the operational record. `sim_t` is simulated
seconds since capture start and is what the windowing logic uses, so a capture
replayed at 400× produces byte-identical features to one replayed in real time.
That property is what makes the fast dataset generation legitimate rather than
a shortcut: the features do not depend on the wall clock.

`mode` carries the simulator's fault mode. It is used only to label training
data and is ignored by the inference service — the model never sees it.

Inference results add `class`, `label`, `confidence`, `probabilities`,
`p_normal`, the 6 features, `latency_ms`, `model` and `seq`.

## Offline behaviour

The broker is on the truck, so every topic above continues to flow during a
35–90 minute cellular outage. Only the bridge from `alerts` and `status/*` to
the operations centre is interrupted. Alerts are written to the local JSONL log
with `fsync` *before* the publish is attempted, so the durable record does not
depend on the publish succeeding; the store-and-forward queue replays records
with `synced: false` in timestamp order once coverage returns.
