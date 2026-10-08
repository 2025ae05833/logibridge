# Final Report — Outline, Evidence and Required Numbers

**Target:** 2,500–3,000 words excluding figures, tables and references.
**Filename:** `GROUP18_LogiEdge_Final.pdf`

## How to use this file

This is an evidence pack and a structure, not a draft. Every number below is
measured or computed by code in this repository and can be regenerated with
`make all`. What is *not* here is the prose, the argument and the judgement —
the problem statement requires those to be your own, and the viva will test
whether they are.

Work section by section. For each one: regenerate the numbers, look at them
yourself, decide what you think they mean, then write that. Where this file
says **[YOU WRITE]**, it means exactly that.

Sources to open alongside this:
- `reports/calculations_output.txt` — every A1/B1/B2/D2/E3 number
- `optimisation/results/benchmark_results.csv` — the 15-cell table
- `training/models/m*_metrics.json` — accuracy and confusion matrices
- `reports/normalisation_experiment.txt` — the 3σ results
- `monitoring/psi_history.json` — PSI trace from your own run

---

# Section 1 — FreightBridge Deployment Context (400–500 words)

Covers Tasks A1, A2, B1, B2. Worth 7 marks with Section 1 of the rubric.

### Required content
Four Edge AI constraints with numbers · hardware selection via Constraint
Triangle · system architecture diagram · Arithmetic Intensity result and
Roofline classification.

### Evidence

**Latency.** Cloud round trip 0.235 s best case, 0.880 s worst case (leg-by-leg
breakdown in `calculations_output.txt`). SLA is 90 s; at 1 °C/min the SLA
permits a 1.5 °C excursion.

> **[YOU WRITE]** The trap here is concluding "cloud is too slow." It isn't —
> 0.88 s fits inside 90 s comfortably. Decide for yourself what the real
> argument is and make it. Hint: what is the round-trip latency during a
> 90-minute outage?

**Bandwidth.** Raw 518.747 MB/truck/day (temperature 0.346, vibration 518.400,
door 0.001). Vibration is 99.93% of it. Edge-processed 0.1335 MB/day.
Reduction 3,885×. At ₹0.10/MB: ₹51.87 vs ₹0.01 per truck per day; pilot
₹16.09 lakh/year vs ₹414/year; full fleet saving ₹50.2 lakh/year.

Note the two vibration rates: A1 says 500 Hz × 3 axes, C1 says 0.5 Hz RMS.
Reconcile them explicitly — raw accelerometer vs on-sensor RMS, a 3,000×
on-sensor reduction before anything reaches the broker. A grader will check
whether you noticed.

**Connectivity.** 7 dead zones, 35–90 min. Envelope breached 4 min into a gap;
cargo asymptotes to ambient (35–42 °C) after ~31 min, it does not rise without
limit. Worst case 10.5 h unmonitored per run.

**Privacy.** Data minimisation, reduced attack surface, no third-party
processor under DPDP Act 2023, tenant isolation, auditable custody log.

> **[YOU WRITE]** State the honest limit: on-device inference alone does not
> discharge a "cannot be accessed" warranty. What else is needed?

**Hardware.** Pi 5 + AI HAT+ ₹15,000/7.5 W; Jetson Orin Nano ₹45,000/15 W;
STM32H7 ₹3,500/0.4 W. Budget 10 W. Full-fleet cost ₹39.75 lakh / ₹119.25 lakh /
₹9.28 lakh. Dominant vertex: **cost**, with **power** as a hard feasibility gate.

> **[YOU WRITE]** Recommend one, argue against two. The non-obvious argument is
> against the STM32 — it would run the model fine. What can't it run?

**Roofline.** AI = 45e6/18e6 = **2.500 FLOP/byte**. Ridge = 16e9/12e9 =
**1.333 FLOP/byte**. AI > ridge → **COMPUTE-BOUND**. Attainable
min(16, 30) = 16 GFLOP/s. Compute time 2.812 ms vs memory time 1.500 ms (1.88×).

> **[YOU WRITE]** What does compute-bound tell you to optimise, and what does it
> tell you *not* to bother with? And: do these figures describe the model you
> actually built? (They do not — 803 params, ~1.6 kFLOP. Say so.)

---

# Section 2 — Sensor Pipeline and MQTT Design (500–600 words)

Covers Tasks C1, C2, C3. Worth 7 marks.

### Required content
Full MQTT topic tree as a figure · QoS justification per topic · preprocessing
design rationale · normalisation experiment results table · data fusion choice
justified against alternatives.

### Evidence

Topic tree and QoS table: copy the figure from `data_pipeline/mqtt_architecture.md`.

Preprocessing: 5-sample causal MA → 30 s window / 10 s step → 6 features
(temp_mean, temp_std, temp_roc °C/min, vib_rms, vib_peak, vib_kurtosis) →
z-score against frozen `training_stats.npy`.

Frozen stats (fitted on 61 clean windows, 10 min):

| feature | mean | std |
|---|---|---|
| temp_mean | 4.0950 | 0.2063 |
| temp_std | 0.1410 | 0.0457 |
| temp_roc | 0.0137 | 0.5118 |
| vib_rms | 0.4513 | 0.0122 |
| vib_peak | 0.4915 | 0.0210 |
| vib_kurtosis | −0.3430 | 1.2618 |

MA sanity check: raw temperature σ = 0.3, post-MA σ = 0.141 ≈ 0.3/√5. Worth one
sentence — it shows the filter does what theory says.

**Normalisation experiment (the mandated 3σ test):**

| condition | accuracy | Δ | recall N | recall W | recall C |
|---|---|---|---|---|---|
| correct | 96.49% | — | 91.3% | 100.0% | 100.0% |
| mean +3σ | 91.23% | −5.26 | 100.0% | **70.6%** | 100.0% |
| mean −3σ | 94.74% | −1.75 | 87.0% | 100.0% | 100.0% |
| std ×3 | 94.74% | −1.75 | 95.7% | 88.2% | 100.0% |

> **[YOU WRITE]** This is the most interesting table in the report. Critical
> recall never moves; Warning recall collapses to 70.6% while accuracy still
> reads 91%. What kind of production failure is that, and why is it harder to
> notice than a loud one?

**Fusion (C3).** Feature-level: extract per sensor, concatenate to one
6-vector, then classify.

> **[YOU WRITE]** Argue against data-level (two streams at 1 Hz and 0.5 Hz —
> what would you have to do first, and what does that cost?) and against
> decision-level (what distinguishes Class 1 from Class 2? Can a per-sensor
> classifier see it?).

**Bounded sawtooth.** Spec drift +0.08 °C/reading at 1 Hz = +4.8 °C/min.
Unbounded over 15 min → +72 °C → every window is Critical by the brief's own
Class definitions. With the bound: 64.2% of Warning samples land in the 1–3 °C
band, 5.8% above 3 °C. Document this as a deliberate decision.

---

# Section 3 — Model Deployment and Pipeline Mapping (400–500 words)

Covers Tasks D1, D2, D3, E3. Worth 6 marks.

### Required content
10-stage pipeline mapped to LogiEdge · Docker layer-cache result · bandwidth
saving for 85 trucks · OTA strategy recommendation with financial justification.

### Evidence

Dataset 291 windows (117 Normal / 87 Warning / 87 Critical). M1: 803 params,
94.74% random split, 100% blocked split, Class 2 recall 100%. **Gate passed.**

Layer cache: full image ~188 MB vs model layer 0.00455 MB. For 85 trucks,
15,980 MB vs 0.387 MB = **₹1,598.00 vs ₹0.04**, a **41,000×** reduction.
At 8.7 cycles/year: ~₹13,900/year on the pilot, ~₹43,300/year at full fleet.

OTA per cycle (280 KB model, 85 trucks, ₹0.10/MB):

| strategy | MB/cycle | ₹/cycle | ₹/year | blast radius |
|---|---|---|---|---|
| full replacement | 23.80 | 2.38 | 20.65 | all 85 |
| canary (10 first) | 23.80 | 2.38 | 20.65 | 10 then 75 |
| shadow mode | 85.18 | 8.52 | 73.82 | 0 (advisory) |

Canary abort case: 2.80 MB = ₹0.28 instead of ₹2.38.

> **[YOU WRITE]** The costs are effectively identical against a ₹28 lakh
> spoilage event. So cost does not decide this. What does? Make the risk
> argument, and be specific about why rollback is hard on this route.

**10-stage mapping (D3).** One to two sentences per stage, specific to *this*
system — not generic. Use the actual file names. Suggested mapping, which you
should check against your course's stage list:

1. Data collection → `simulator.py`, 3 streams to the local broker
2. Data preprocessing → `preprocessing.py`, 5-sample MA
3. Feature engineering → 6 features per 30 s window, feature-level fusion
4. Model selection → 32/16 MLP, chosen for a 6-feature input on a 7.5 W node
5. Training → `train_model.py`, 88% gate enforced in code
6. Evaluation → dual split, Class 2 recall as the safety metric
7. Optimisation → `convert_ptq.py`, `prune_quantise.py` (M2, M3)
8. Conversion/packaging → TFLite INT8 inside a `python:3.11-slim` container
9. Deployment → `logibridge_deploy.yml`, canary rollout, `MODEL_PATH` switch
10. Monitoring → `drift_monitor.py`, PSI on P(Normal), local alert log

---

# Section 4 — Optimisation and Pareto Analysis (400–500 words)

Covers Tasks F1, F2, F3. Worth 5 marks.

### Required content
Benchmarking results table (15 cells) · Pareto chart with annotation · Class 2
recall for recommended variant · deployment recommendation with SLA evidence.

### Evidence — regenerate these on your own hardware

| | mean (ms) | p95 (ms) | size (KB) | accuracy (%) | energy (mJ) |
|---|---|---|---|---|---|
| M1 FP32 | 0.0009 | 0.0012 | 5.26 | 94.74 | 0.0171 |
| M2 INT8 | 0.0016 | 0.0017 | 4.55 | 96.49 | 0.0282 |
| M3 pruned+INT8 | 0.0016 | 0.0015 | 3.84 | 96.49 | 0.0284 |

Class 2 recall: 100% for all three. Bar is 95%. ✓

M3 structure: 48 units → 31 (35.4% removed), dense_0 32→21, dense_1 16→10,
803 params → 400 (50.2% removed), accuracy preserved.

SLA decomposition: 30,000 ms window fill + 10,000 ms step alignment + 100 ms
alert path = 40,100 ms fixed, leaving **49,900 ms** for inference. Slowest
variant uses 0.0016 ms.

> **[YOU WRITE]** Three things that need your own voice:
> 1. INT8 came out *slower* than FP32 and pruning 50% of parameters changed
>    latency by nothing. Explain why, and state the hardware caveat (measured
>    on x86/AVX-512; the Pi 5's NEON path has SDOT and much lower FP32
>    throughput — what would you expect there, and what would you do about it?).
> 2. The +1.75 pp accuracy gain from quantisation is **one window** out of 57.
>    Do not call it an improvement. Say what it is.
> 3. Which variant do you deploy, and on what grounds, given latency is
>    irrelevant here?

---

# Section 5 — MLOps Monitoring and Reflection (300–400 words)

Covers Tasks E1, E2. Worth 3 marks, plus Section 6's 2 marks for report quality.

### Required content
PSI before/during/after injection · drift type classification · Ansible
idempotency result · one genuine technical difficulty · one architectural change.

### Evidence

PSI trace (use your own run from `monitoring/psi_history.json`):

| phase | PSI range | alert |
|---|---|---|
| 1 clean baseline | 0.031 – 0.109 | none |
| 2 combined injected | 0.433 → **13.909** | `[LOGIBRIDGE DRIFT ALERT]` |
| 3 clean restored | 8.568 → 0.019 | recovered below 0.10 |

Crossed 0.25 after 12 fault inferences (120 s simulated truck time). Recovered
after 105 clean inferences — note *why* recovery is slower than detection (the
100-window buffer must flush).

Score comparison: P(Normal) crosses after 12, peaks 13.91. Max-softmax crosses
after 20, peaks 8.70.

Ansible: run 1 `changed>0`, run 2 **`changed=0`**. Mechanism — checksum-gated
copies, skipped stop task, declarative `docker_container`, `changed_when: false`
on the verify.

> **[YOU WRITE] Drift type.** Classify it. Is this covariate shift, concept
> drift, label drift, or a genuine physical change in the monitored process?
> And the consequence: should this trigger retraining, or not? Be careful —
> the obvious answer may be wrong here.

> **[YOU WRITE] One genuine technical difficulty.** Must be yours. Candidates
> you actually have evidence for in this repo:
> - the Warning-class labelling contradiction and the sawtooth fix
> - the first training run scoring 100% and diagnosing it as a simulator artefact
> - predicting max-softmax PSI would never fire, and measurement disproving it
> - energy reading 0.00000 mJ because 200 inferences finish faster than psutil samples
> - a false `[LOGIBRIDGE DRIFT ALERT]` at PSI=0.405 from a 20-sample window
>
> Pick one you can discuss for two minutes under questioning. Say what you
> expected, what happened, how you diagnosed it, what you changed.

> **[YOU WRITE] One architectural change.** The strongest available: feed door
> state into the classifier. Your error analysis shows all three validation
> errors sit at 4.3–4.6 °C with temperature rising, and two are door-opening
> windows misread as Warning. The extractor already tracks door events; the
> mandated 6-feature vector has nowhere to put them. Say how you'd use it — an
> extra feature, or a gating rule that suppresses Warning during and shortly
> after an OPEN?

---

## Before you submit

- [ ] Word count 2,500–3,000, excluding figures/tables/references
- [ ] Filename exactly `GROUP18_LogiEdge_Final.pdf`
- [ ] Every number regenerated on your hardware, not copied from this file
- [ ] GitHub URL in the report, instructor added as collaborator
- [ ] Demo video link in the report **and** `demo/demo_video_link.txt`
- [ ] Video link tested in incognito
- [ ] Every **[YOU WRITE]** replaced with your own analysis
- [ ] You can explain every file in the repo without reading from notes
