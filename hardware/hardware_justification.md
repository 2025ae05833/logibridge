# Tasks B1 and B2 — Hardware Selection and Roofline

Evidence pack for Report Section 1. Numbers from `reports/calculations.py`
(`make report`); the argument belongs in the final PDF in your own words.

## B1 — Constraint Triangle inputs

| Option | Hardware | ₹/truck | TDP | Pilot (85) | Fleet (265) |
|---|---|---|---|---|---|
| 1 | Pi 5 8 GB + AI HAT+ (13 TOPS Hailo-8L) | 15,000 | 7.5 W | ₹12.75 L | ₹39.75 L |
| 2 | Jetson Orin Nano Super (67 TOPS) | 45,000 | 15 W | ₹38.25 L | ₹119.25 L |
| 3 | STM32H7 custom MCU | 3,500 | 0.4 W | ₹2.98 L | ₹9.28 L |

Power budget 10 W (12 V truck supply via DC-DC). **Option 2 exceeds it.**

**Dominant vertex: cost, with power as a hard feasibility gate.**

Two arguments the report has to make properly:

- Option 2 fails on power before cost is considered, and spends 67 TOPS on a
  6-feature MLP measured at ~1 µs per inference.
- Option 3 is the interesting one. It would run an 803-parameter INT8 MLP
  perfectly well under TFLite Micro. What it cannot run is Components D and E
  of this project — Docker, the layer-cached OTA path, Ansible, a local MQTT
  broker. The hardware saving is spent several times over on bespoke firmware
  and fleet-management risk across 265 vehicles.

## B2 — Arithmetic Intensity and Roofline

Given: 45 MFLOP and 18 MB per inference; 16 GFLOP/s NEON; 12 GB/s LPDDR4X.

```
AI       = 45e6 / 18e6   = 2.500 FLOP/byte
I_ridge  = 16e9 / 12e9   = 1.333 FLOP/byte
AI > I_ridge             -> COMPUTE-BOUND
attainable = min(16, 2.5 x 12 = 30) = 16 GFLOP/s
compute time 2.812 ms vs memory time 1.500 ms  (1.88x)
```

Implication: optimise arithmetic (INT8, structured pruning, NPU offload). Do
not spend effort on data layout, tiling or cache blocking — those raise the
bandwidth roof, and this model is not on it.

Caveat the report must state: these are the figures the problem statement
supplies. Our actual model is 803 parameters and ~1.6 kFLOP per inference,
four orders of magnitude smaller, with measured latency ~1 µs rather than the
2.8 ms implied. The Task F2 benchmark confirms the consequence — removing 50%
of the parameters changed latency by nothing, because runtime is dominated by
interpreter invocation overhead rather than by arithmetic or memory traffic.
