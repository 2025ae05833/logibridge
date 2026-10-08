# LogiEdge — Demo Video Run-Sheet (15–20 minutes)

The problem statement names the video as the primary academic-integrity
mechanism, and says you must be able to explain any component under follow-up
questioning. Treat this as a run-sheet, not a script: say what you understand,
not what is written here.

**Before recording:** `make clean && make all` so everything is fresh, and open
the files you will reference in your editor. Record at 1080p with audio.
Set the YouTube link to *unlisted* or Drive to *anyone with the link*, then
**test it in an incognito window** before submitting, and paste it into
`demo/demo_video_link.txt`.

---

## 0 · Opening (0:00–0:45)

Group number, member names, one sentence on what LogiEdge is. State the three
classes and the 90-second SLA. Say up front that the system runs with no cloud
dependency, because that is the claim the rest of the video substantiates.

## 1 · Architecture and the constraint case (0:45–3:00) — *Section 1, 7 marks*

Show `system_architecture.png`. Trace the detection path with your cursor:
sensor → local broker → preprocess → inference → local alert log → driver. Say
the sentence that matters: **no element of that path crosses the cellular
link.**

Then the four constraints, with numbers from `reports/calculations_output.txt`:

- **Latency.** Don't claim cloud is too slow — the round trip is 0.24–0.88 s,
  which fits inside 90 s. The real argument is availability: during a dead zone
  the round trip is not slow, it is *undefined*. The 2–8 °C envelope is breached
  4 minutes into a 90-minute gap.
- **Bandwidth.** 518.7 MB/truck/day raw versus 0.13 MB processed — a 3,885×
  reduction, ₹50.2 lakh/year at full fleet. Note that vibration is 99.93% of
  the raw volume, so this is really a vibration argument.
- **Connectivity.** Seven dead zones, 35–90 min. Show the write-before-publish
  line in `inference_service.py` — that is the offline guarantee in one place.
- **Privacy.** Data minimisation, smaller attack surface, no third-party
  processor under DPDP 2023. Say the honest limit out loud: edge inference is
  necessary but not sufficient, and must be paired with disk encryption, TLS
  client certs and signed models.

## 2 · Hardware and Roofline (3:00–4:30) — *Section 1*

Constraint Triangle across the three options. Recommend the **Pi 5 + AI HAT+**.
Make the two arguments that are not obvious:

- The Jetson fails on *power* (15 W against a 10 W budget) before cost even
  enters, and costs ₹79.5 lakh more at full fleet for compute a 6-feature MLP
  cannot use.
- The STM32 is cheapest and lowest-power and is still wrong — not because it
  can't run the model (it can), but because it can't run Components D and E:
  Docker, the layer-cached OTA path, Ansible, a local broker.

Roofline: AI = 2.5 FLOP/byte, ridge = 1.333 → **compute-bound**. Say what that
implies: quantise and prune, don't bother with cache blocking. Then say the
honest part — our actual model is ~1.6 kFLOP, four orders of magnitude below
the given figures, so Roofline is the wrong instrument for it.

## 3 · Sensor pipeline live (4:30–7:00) — *Section 2, 7 marks*

Terminal 1: `mosquitto_sub -t 'logibridge/#' -v`
Terminal 2: `python data_pipeline/simulator.py --anomaly none --duration 120 --speed 5`

Show messages flowing. Switch to `--anomaly combined` and show vibration step
and temperature climb in the raw stream.

Walk the topic tree from `mqtt_architecture.md` and justify QoS: **0** for
self-correcting high-rate telemetry, **1** for discrete door events, **2** for
alerts because an escalation must not be dispatched twice.

Then explain the **bounded sawtooth** decision — this is the single best thing
in the project to be asked about. Spec rate is +4.8 °C/min; unbounded over 15
minutes it reaches +72 °C and all your "Warning" data is Critical by the brief's
own definition. Show the measured table: 64.2% of Warning samples in the 1–3 °C
band.

Then `make normexp` and read the result: Class 2 recall holds at 100% under a
+3σ corruption, but Class 1 collapses to 70.6% while accuracy still reads 91%.
Corrupted statistics don't blind you to catastrophe — they blind you to the
early warning that would have prevented it.

## 4 · Training and the 88% gate (7:00–9:00) — *Section 3, 6 marks*

`make dataset` then `make train`. Show 117/87/87 windows and the GATE PASSED
line at 94.74%.

Then explain why it isn't 100%: with an instantaneous fault step it *was* 100%,
which said more about the simulator than the model. Gradual onset over 90 s
produces honest ambiguity. Show the three errors — all at 4.3–4.6 °C and
rising, two door openings read as Warning, one early-onset window read as
Normal. Zero Critical misses.

Mention both splits and why the random one leaks (20 s of 30 s overlap).

## 5 · Docker and the OTA layer cache (9:00–11:00) — *Section 3*

`make docker`, then `make ota-demo`.

Point at the CACHED lines. The model change invalidates one layer because every
pip install sits above it. Show `docker history` and the layer sizes. Then the
arithmetic: 188 MB full image vs 0.0046 MB model layer × 85 trucks = **~₹1,598
saved per update cycle, a ~41,000× reduction**.

Show `MODEL_PATH` switching M1→M3 with no rebuild:
`docker run -e MODEL_PATH=/opt/logibridge/model.tflite ...`

Walk the 10-stage pipeline mapping from Report Section 3 against this repo.

## 6 · PSI drift monitoring (11:00–14:00) — *Section 5, 3 marks*

`make reference` then `make drift`. This is the longest continuous segment —
let it run.

- **Phase 1 clean:** PSI stays 0.03–0.11.
- **Phase 2 injection:** PSI crosses 0.25 and `[LOGIBRIDGE DRIFT ALERT]` fires.
  Peak 13.91.
- **Phase 3 recovery:** PSI falls back below 0.10.

Explain the score choice while it runs. You measured both: P(Normal) crosses
after 12 inferences, max-softmax after 20. Say that you initially predicted
max-softmax would never fire and that measurement proved you wrong — examiners
reward that far more than a tidy story.

Also mention the warm-up guard: an early version fired a false alert at
PSI=0.405 on clean data with only 20 samples in the window. Requiring the
window half-full removed it.

Classify the drift type: this is **concept/data drift from a genuine physical
fault**, not covariate shift from sensor degradation. The right response is to
alert operations, not to retrain — the model is behaving correctly.

## 7 · Ansible idempotency (14:00–16:00) — *Section 5*

`make deploy`. Both runs must be on camera.

- Run 1: `changed` > 0.
- Run 2: **`changed=0`** — hold on the recap line.

Explain how: the two `copy` tasks compare checksums, the stop task is gated on
their results so it is *skipped*, `docker_container` is declarative and reports
`ok`, and the verify task is `changed_when: false` because reading is not
changing. Show the seven tasks and count them aloud.

## 8 · Optimisation and Pareto (16:00–18:30) — *Section 4, 5 marks*

`make benchmark`. Show the 15-cell table and `pareto_chart.png`.

Lead with the counter-intuitive result: **INT8 is slower than FP32 here, and
pruning 50% of parameters changed latency by nothing.** At 803 parameters,
interpreter dispatch dominates. Then the caveat that shows you understand it:
measured on x86 with AVX-512; on the Pi 5's NEON path with SDOT, the ranking
would likely invert. You would re-run on target before committing.

Show structured pruning: 48 units → 31, 803 params → 400, accuracy held.
Explain why unstructured pruning would have bought nothing.

**F3 recommendation: deploy M3.** Class 2 recall 100% (bar is 95%), smallest
file at 3.84 KB which matters for the 6-weekly OTA across 85 trucks, and
latency is irrelevant because every variant is ~8 orders of magnitude inside
the SLA. State the SLA decomposition: 30 s window + 10 s alignment + 100 ms
alert path leaves 49.9 s for inference.

## 9 · OTA strategy and reflection (18:30–20:00) — *Section 5*

**Canary**, 10 trucks first. Costs are indistinguishable (₹2.32 vs ₹2.32 vs
₹8.52 per cycle) against a ₹28 lakh spoilage event, so the decision is risk:
canary bounds the blast radius, and rollback is slow when trucks sit in dead
zones. Against shadow: safest in principle, but doubles inference work on a
7.5 W node and delays every rollout by 14 days for a model with six
interpretable features.

**One genuine difficulty** — pick one you actually hit and can discuss. Good
candidates: the Warning-class labelling contradiction and the sawtooth fix; the
first 100% accuracy result and diagnosing it as a simulator artefact; the PSI
score choice and being wrong about max-softmax; the energy measurement reading
0.00000 mJ because 200 inferences finish faster than psutil can sample.

**One architectural change** — feed door state into the classifier as a gating
signal. The error analysis shows door openings cause the false Warnings; the
extractor already tracks door events; the mandated six-feature vector has
nowhere to put it.

Close with the limitations from the README. Saying them yourself is worth more
than having them found.

---

## Checklist before upload

- [ ] Both Ansible runs visible, `changed=0` legible on run 2
- [ ] `[LOGIBRIDGE DRIFT ALERT]` visible on screen
- [ ] PSI recovery below 0.10 visible
- [ ] CACHED layers visible in the OTA rebuild
- [ ] GATE PASSED line visible
- [ ] Pareto chart on screen with the recommendation stated
- [ ] 85-truck bandwidth saving stated aloud
- [ ] Every group member speaks
- [ ] Link tested in incognito
- [ ] Link pasted into `demo/demo_video_link.txt` **and** the report
