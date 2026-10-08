# LogiEdge — Intelligent Edge AI Platform for Cold-Chain Logistics

**BITS Pilani WILP · AIML ZG535 Machine Learning on Edge · Mini-Project · Group 18**

An offline-capable Edge AI pipeline for FreightBridge Logistics' 85 refrigerated
trucks. It monitors cargo temperature, refrigeration-unit vibration and door
events, classifies the compartment as **Normal / Warning / Critical** on the
truck itself, and alerts the operations centre without depending on continuous
cellular connectivity.

## A note on the two names

The problem statement uses **LogiEdge** as the system name and **logibridge**
inside concrete identifiers (the MQTT topic `logibridge/trucks/{id}/inference`,
the deploy path `/opt/logibridge`, the playbook `logibridge_deploy.yml`, the
alert string `[LOGIBRIDGE DRIFT ALERT]`). Both refer to this one system. We
follow the statement literally for every identifier it specifies and use
LogiEdge in prose.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# a broker must be running locally
sudo apt install mosquitto mosquitto-clients   # or: brew install mosquitto

make all        # dataset -> train -> M1/M2/M3 -> benchmark -> calculations
```

`make all` takes a few minutes. The simulator's `--speed` multiplier is what
makes this practical: 20 simulated minutes of sensor data generate in about
three seconds. Because every window is keyed on `sim_t` rather than wall-clock
time, an accelerated capture produces byte-identical features to a real-time
one.

## Results

Measured on the 57-window held-out validation split. Re-run `make all` to
reproduce; latency and energy are hardware-dependent and will differ on your
machine.

| Variant | Mean latency | p95 | Size | Accuracy | Energy | Class 2 recall |
|---|---|---|---|---|---|---|
| M1 FP32 baseline | 0.90 µs | 1.2 µs | 5.26 KB | 94.74% | 0.0171 mJ | 100% |
| M2 PTQ INT8 | 1.60 µs | 1.7 µs | 4.55 KB | 96.49% | 0.0282 mJ | 100% |
| M3 Pruned 35% + INT8 | 1.60 µs | 1.5 µs | 3.84 KB | 96.49% | 0.0284 mJ | 100% |

Four findings worth more than the table itself:

- **Optimisation bought size, not speed.** INT8 is *slower* than FP32 here, and
  removing 50% of the parameters changed latency by nothing. At 803 parameters
  runtime is dominated by interpreter invocation overhead, not arithmetic.
  Measured on x86 with AVX-512; on the Pi 5's NEON path, where INT8 has SDOT
  and FP32 throughput is far lower, the ranking would likely invert.
- **The 90 s SLA is not a model problem.** Decomposed, it is 30 s of window
  fill + up to 10 s of step alignment + ~100 ms of alert path, leaving 49.9 s
  for inference. Every variant uses under 2 µs of it.
- **Validation errors are physically meaningful.** All three sit at
  4.3–4.6 °C with temperature rising slowly — two are door-opening windows
  misread as Warning, one is an early fault-onset window misread as Normal. A
  door opening and a nascent refrigeration fault are genuinely indistinguishable
  from these six features.
- **Corrupted normalisation statistics damage the Warning class, not Critical.**
  A +3σ shift leaves Class 2 recall at 100% but collapses Class 1 recall from
  100% to 70.6%, while headline accuracy still reads 91%.

## Repository layout

```
logibridge/
├── scenario_architecture/
│   ├── constraint_analysis.md          Task A1
│   ├── make_architecture.py            renders the diagram from code
│   └── system_architecture.png         Task A2
├── hardware/
│   └── hardware_justification.md       Tasks B1, B2
├── data_pipeline/
│   ├── simulator.py                    Task C1 — 3 streams, --anomaly, --speed
│   ├── preprocessing.py                Task C2 — filter, window, 6 features
│   ├── normalisation_experiment.py     Task C2 — the mandated 3σ experiment
│   ├── training_stats.npy              frozen normalisation statistics
│   └── mqtt_architecture.md            topic tree + QoS justification
├── training/
│   ├── common.py                       shared helpers, TFLite conversion
│   ├── generate_dataset.py             Task D1 — labelled dataset
│   ├── train_model.py                  Task D1 — M1, enforces the 88% gate
│   ├── convert_ptq.py                  Task F1 — M2 full INT8
│   ├── prune_quantise.py               Task F1 — M3 structured prune + INT8
│   └── models/
├── inference/
│   ├── Dockerfile                      Task D2 — pip layers above COPY model
│   ├── requirements.txt                tflite-runtime, not full TensorFlow
│   └── inference_service.py            Task D2 — offline-first alert path
├── monitoring/
│   ├── drift_monitor.py                Task E1 — PSI
│   └── reference_dist.json             300 clean Normal windows
├── deployment/
│   ├── logibridge_deploy.yml           Task E2 — exactly 7 tasks, idempotent
│   └── inventory.ini
├── optimisation/
│   ├── benchmark.py                    Task F2 — five metrics
│   └── results/
│       ├── benchmark_results.csv
│       └── pareto_chart.png
├── demo/
│   ├── drift_demo.sh                   the E1 injection/recovery sequence
│   └── demo_video_link.txt
└── reports/
    ├── calculations.py                 every number in the report
    └── GROUP18_LogiEdge_Final.pdf
```

## Design decisions worth knowing before reading the code

Each is documented in full at the top of the file that implements it.

**Bounded sawtooth drift** (`simulator.py`). The specified +0.08 °C per reading
at 1 Hz is +4.8 °C/min. Run unbounded for the 15-minute Warning capture the
brief requires, the cargo reaches +72 °C and every window after the first
minute satisfies the *Critical* definition rather than *Warning*. The drift is
therefore bounded by a recovery cycle at the mode's ceiling — physically, a
unit that is struggling but not dead. The specified ramp rate is preserved
exactly; measured result is 64.2% of Warning samples in the 1–3 °C band.

**Gradual fault onset** (`simulator.py`). With faults applied as an instant
step, the classes separate perfectly and the model scores 100% — a number that
describes the simulator, not the model. Real bearings degrade. Ramping fault
severity over 90 s produces genuinely ambiguous windows during onset and a
credible 94.74%.

**Structured, not unstructured, pruning** (`prune_quantise.py`). `tfmot`'s
`prune_low_magnitude` zeros individual weights but leaves tensor shapes intact,
so the MAC count, the tensor dimensions and the TFLite file size barely move.
Zeros are not speed. We use the official `PolynomialDecay` schedule to ramp a
whole-unit mask, then physically compact the survivors: 48 units → 31 (35.4%
removed), 803 parameters → 400.

**P(Normal) rather than max-softmax for PSI** (`drift_monitor.py`). Both are
implemented and both were measured against the same injected fault: P(Normal)
crosses 0.25 after 12 inferences and peaks at 13.91, max-softmax after 20 and
peaks at 8.70. P(Normal) is the default as the more sensitive statistic. The
file documents an initial prediction that max-softmax would never fire at all,
and why measurement contradicted it.

**Two validation splits** (`generate_dataset.py`). Windows overlap by 20 of
their 30 seconds, so a random split leaks between train and validation. Both
the mandated random split and a time-blocked split with a guard band are
reported.

## Reproducing each deliverable

| Task | Command |
|---|---|
| C1 simulator | `python data_pipeline/simulator.py --anomaly combined --duration 300 --speed 10` |
| C2 stats + experiment | `make dataset normexp` |
| D1 training, 88% gate | `make train` |
| D2 container | `make docker` |
| D2 OTA layer cache | `make ota-demo` |
| E1 PSI reference | `make reference` |
| E1 drift injection | `make drift` |
| E2 Ansible idempotency | `make deploy` |
| F1 variants | `make variants` |
| F2/F3 benchmark + Pareto | `make benchmark` |
| A1/B2/D2/E3 report numbers | `make report` |

## Known limitations

Stated here rather than discovered by a reader.

- **Energy is an estimate, not a measurement.** `E = P × t` with
  `P = TDP × CPU%` ignores DVFS, uncore and memory power, and runs on
  development hardware rather than a Pi 5. It ranks the variants; it is not an
  absolute per-inference budget. A deployment decision would need a shunt
  measurement on the target board.
- **Latency was measured on x86, not on the target.** See the first finding
  above — the INT8-vs-FP32 ranking is expected to differ on ARM.
- **The validation set is 57 windows.** One sample is 1.75 percentage points,
  so the +1.75 pp "improvement" from quantisation is one window changing its
  mind, not a real effect. We report it as such.
- **Kurtosis is computed over 15 samples.** The 0.5 Hz vibration stream gives
  15 points per 30 s window, which is a noisy basis for a fourth-moment
  statistic.
- **Door state is tracked but unused.** The mandated feature vector is exactly
  six values with no room for it, and the error analysis shows door openings
  are precisely what causes the false Warnings. This is the architectural
  change proposed in Report Section 5.
