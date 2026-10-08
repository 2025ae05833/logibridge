#!/usr/bin/env python3
"""
LogiEdge -- Report Calculations  (Tasks A1, B2, D2, E3)

Every number quoted in the final report is produced here, so each one is
reproducible and its assumptions are visible rather than buried in prose.

    python calculations.py > calculations_output.txt
"""

SEP = "=" * 74


def rule(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ==========================================================================
# Fleet and tariff constants from the problem statement
# ==========================================================================
PILOT_TRUCKS = 85
FULL_FLEET = 265
TARIFF_PER_MB = 0.10  # rupees
SECONDS_PER_DAY = 86_400
DAYS_PER_YEAR = 365

MB = 1_000_000  # decimal megabyte: cellular tariffs are billed in these
KB = 1_000


# ==========================================================================
# TASK A1 -- LATENCY
# ==========================================================================
def task_a1_latency():
    rule("TASK A1 (a) -- LATENCY: is cloud inference feasible?")

    temp_rise_per_min = 1.0  # C/min when the refrigeration unit fails
    sla_s = 90.0
    print(f"Requirement: detect and alert within {sla_s:.0f} s of a fault signature.")
    print(f"Cargo warms at {temp_rise_per_min:.0f} C/min once the unit fails, so the")
    print(f"SLA permits roughly {temp_rise_per_min * sla_s / 60:.1f} C of excursion "
          f"before an alert must exist.\n")

    # Rural India round-trip latency profile, 4G and fallback 2G/3G.
    legs = [
        ("sensor -> edge node (local I2C/SPI)", 0.005, 0.02),
        ("edge node -> cell tower (RAN, rural 4G)", 0.060, 0.250),
        ("tower -> cloud region (backhaul, Mumbai)", 0.030, 0.080),
        ("cloud inference + queueing", 0.050, 0.200),
        ("cloud -> ops centre -> truck ack", 0.090, 0.330),
    ]
    lo = sum(l for _, l, _ in legs)
    hi = sum(h for _, _, h in legs)
    print(f"{'leg':<44}{'best (s)':>12}{'worst (s)':>13}")
    for name, a, b in legs:
        print(f"{name:<44}{a:>12.3f}{b:>13.3f}")
    print(f"{'-' * 69}")
    print(f"{'cloud round trip':<44}{lo:>12.3f}{hi:>13.3f}")

    print(f"\nIn isolation even the worst case ({hi:.2f} s) fits inside {sla_s:.0f} s,")
    print("so a naive reading says cloud inference is feasible. It is not, for two")
    print("reasons that latency arithmetic alone does not capture:\n")
    print("  1. AVAILABILITY, NOT LATENCY, IS THE BINDING CONSTRAINT. The round trip")
    print("     is only defined when a link exists. On the Nashik-Aurangabad route")
    print("     the link is absent for 35-90 minutes at seven locations, during")
    print("     which cloud round-trip latency is not 0.88 s but unbounded.")
    print(f"  2. A 90-minute outage at {temp_rise_per_min:.0f} C/min permits a "
          f"{temp_rise_per_min * 90:.0f} C excursion.")
    print("     That is total cargo loss for vaccines with a 2-8 C envelope.\n")
    print("Edge inference removes the network from the detection path entirely:")
    print("sensor -> edge node -> inference is 5-20 ms and has no dependency on")
    print("cellular coverage. The uplink then carries alerts, not decisions.")
    return {"cloud_rtt_best_s": lo, "cloud_rtt_worst_s": hi, "sla_s": sla_s}


# ==========================================================================
# TASK A1 -- BANDWIDTH
# ==========================================================================
def task_a1_bandwidth():
    rule("TASK A1 (b) -- BANDWIDTH: raw telemetry vs edge-processed alerts")

    bytes_per_sample = 4  # float32, packed binary frames
    temp_hz, vib_hz, vib_axes = 1.0, 500.0, 3
    door_events_per_day, door_bytes = 20, 50

    temp_samples = temp_hz * SECONDS_PER_DAY
    vib_samples = vib_hz * vib_axes * SECONDS_PER_DAY
    temp_b = temp_samples * bytes_per_sample
    vib_b = vib_samples * bytes_per_sample
    door_b = door_events_per_day * door_bytes
    raw_b = temp_b + vib_b + door_b
    raw_mb = raw_b / MB

    print(f"Assumption: {bytes_per_sample}-byte float per sample, packed binary "
          f"frames (not JSON).\n")
    print(f"{'stream':<30}{'samples/day':>16}{'bytes/day':>16}{'MB/day':>11}")
    print(f"{'temperature 1 Hz':<30}{temp_samples:>16,.0f}{temp_b:>16,.0f}"
          f"{temp_b / MB:>11.3f}")
    print(f"{'vibration 500 Hz x 3 axes':<30}{vib_samples:>16,.0f}{vib_b:>16,.0f}"
          f"{vib_b / MB:>11.3f}")
    print(f"{'door events (discrete)':<30}{door_events_per_day:>16,.0f}"
          f"{door_b:>16,.0f}{door_b / MB:>11.3f}")
    print(f"{'-' * 73}")
    print(f"{'RAW TOTAL per truck per day':<30}{'':>16}{raw_b:>16,.0f}{raw_mb:>11.3f}")
    print(f"\nVibration is {100 * vib_b / raw_b:.2f}% of the raw volume. Any "
          f"bandwidth argument\nfor this deployment is a vibration argument.")

    # ---- edge-processed volume
    inferences_per_day = SECONDS_PER_DAY / 10  # one per 10 s window step
    alert_rate = 0.01  # 1% of windows are Warning or Critical
    alerts = inferences_per_day * alert_rate
    alert_bytes = 300  # JSON alert record
    heartbeats = SECONDS_PER_DAY / 300  # one per 5 minutes
    heartbeat_bytes = 200
    summary_bytes = 50 * KB  # one daily health summary

    edge_b = alerts * alert_bytes + heartbeats * heartbeat_bytes + summary_bytes
    edge_mb = edge_b / MB

    print(f"\nEdge-processed uplink (alerts and health only):")
    print(f"{'  inferences/day (10 s step)':<46}{inferences_per_day:>12,.0f}")
    print(f"{'  alerts/day at 1% non-Normal':<46}{alerts:>12,.0f}"
          f"  x {alert_bytes} B")
    print(f"{'  heartbeats/day (5 min)':<46}{heartbeats:>12,.0f}"
          f"  x {heartbeat_bytes} B")
    print(f"{'  daily health summary':<46}{1:>12,.0f}  x {summary_bytes:,} B")
    print(f"{'  EDGE TOTAL per truck per day':<46}{edge_b:>12,.0f} B "
          f"= {edge_mb:.4f} MB")

    reduction = raw_mb / edge_mb
    print(f"\nReduction factor: {reduction:,.0f}x")

    # ---- cost
    print(f"\n{'COST AT Rs ' + format(TARIFF_PER_MB, '.2f') + '/MB':<34}"
          f"{'raw (cloud)':>18}{'edge':>14}{'saved':>14}")
    for label, n in (("per truck/day", 1), (f"pilot {PILOT_TRUCKS} trucks/day", PILOT_TRUCKS)):
        r = raw_mb * n * TARIFF_PER_MB
        e = edge_mb * n * TARIFF_PER_MB
        print(f"{label:<34}{'Rs ' + format(r, ',.2f'):>18}"
              f"{'Rs ' + format(e, ',.2f'):>14}{'Rs ' + format(r - e, ',.2f'):>14}")
    for label, n in ((f"pilot {PILOT_TRUCKS} trucks/YEAR", PILOT_TRUCKS),
                     (f"full {FULL_FLEET} trucks/YEAR", FULL_FLEET)):
        r = raw_mb * n * DAYS_PER_YEAR * TARIFF_PER_MB
        e = edge_mb * n * DAYS_PER_YEAR * TARIFF_PER_MB
        print(f"{label:<34}{'Rs ' + format(r, ',.0f'):>18}"
              f"{'Rs ' + format(e, ',.0f'):>14}{'Rs ' + format(r - e, ',.0f'):>14}")

    annual_saving = (raw_mb - edge_mb) * FULL_FLEET * DAYS_PER_YEAR * TARIFF_PER_MB
    print(f"\nAt full fleet the edge architecture saves Rs {annual_saving / 100000:,.1f} "
          f"lakh per year in\ncellular data alone -- before counting a single prevented "
          f"spoilage event.")

    # Reconcile the two vibration rates in the problem statement.
    print(f"\nNOTE ON THE TWO VIBRATION RATES")
    print(f"Task A1 specifies vibration at 500 Hz on 3 axes; Task C1 specifies a")
    print(f"vibration_rms stream at 0.5 Hz. These describe different points in the")
    print(f"same chain: 500 Hz x 3 is the raw accelerometer, and the 0.5 Hz stream is")
    print(f"the RMS computed from it on the sensor node. The bandwidth case above")
    print(f"uses the raw rate because that is what a cloud architecture would have to")
    print(f"ship; the simulator and model use the RMS stream because that is what")
    print(f"crosses the wire in the edge architecture. Computing RMS at the sensor is")
    print(f"itself the first and largest data reduction in the pipeline:")
    rms_b = 0.5 * SECONDS_PER_DAY * bytes_per_sample
    print(f"  raw vibration  {vib_b / MB:>10.3f} MB/day")
    print(f"  RMS at 0.5 Hz  {rms_b / MB:>10.3f} MB/day   "
          f"({vib_b / rms_b:,.0f}x reduction, on-sensor)")
    return {"raw_mb_day": raw_mb, "edge_mb_day": edge_mb, "reduction": reduction}


# ==========================================================================
# TASK A1 -- CONNECTIVITY
# ==========================================================================
def task_a1_connectivity():
    rule("TASK A1 (c) -- CONNECTIVITY: behaviour in the seven dead zones")

    gaps, gap_lo, gap_hi = 7, 35, 90
    rise = 1.0  # C/min
    print(f"Nashik-Aurangabad: {gaps} documented outages of {gap_lo}-{gap_hi} minutes.\n")
    print(f"{'':<30}{'cloud-only':>22}{'LogiEdge edge':>24}")
    rows = [
        ("sensing", "continues", "continues"),
        ("inference", "STOPS", "continues on-truck"),
        ("detection of a fault", "NONE", "within 90 s"),
        ("alert to driver/buzzer", "NONE", "immediate, local"),
        ("alert to ops centre", "NONE", "queued, synced later"),
        ("chain-of-custody record", "GAP", "complete, local log"),
    ]
    for a, b, c in rows:
        print(f"{a:<30}{b:>22}{c:>24}")

    worst_total = gaps * gap_hi
    ambient_lo, ambient_hi = 35.0, 42.0  # Maharashtra / Andhra summer cabin ambient
    setpoint = 4.0
    time_to_breach = (8.0 - setpoint) / rise  # minutes to leave the 2-8 C envelope
    time_to_ambient = (ambient_lo - setpoint) / rise
    print(f"\nWorst-case exposure on one run: {gaps} x {gap_hi} min = "
          f"{worst_total} min = {worst_total / 60:.1f} h unmonitored.")
    print(f"\nWhat a failed unit does inside a single {gap_hi}-minute gap, at "
          f"{rise:.0f} C/min:")
    print(f"  leaves the 2-8 C vaccine envelope after {time_to_breach:.0f} min")
    print(f"  reaches cabin ambient after about {time_to_ambient:.0f} min, and then stops")
    print(f"  rising -- the cargo asymptotes to ambient ({ambient_lo:.0f}-{ambient_hi:.0f} C in a")
    print(f"  Maharashtra summer), it does not climb without limit")
    print(f"\nSo the honest statement is not a large temperature number but a timing")
    print(f"one: the envelope is breached {gap_hi - time_to_breach:.0f} minutes before the")
    print(f"truck regains signal, and the cargo then sits above it for the rest of the")
    print(f"gap. Under a cloud-only architecture nobody learns this until the link")
    print(f"returns, by which point the consignment is already lost. That is the exact")
    print(f"failure mode of the Rs 28 lakh Nashik-Aurangabad incident.")
    print(f"\nHow LogiEdge handles it:")
    print(f"  - the MQTT broker is on the truck, so publish/subscribe never leaves it")
    print(f"  - inference reads from that local broker; no cloud call exists in the path")
    print(f"  - alerts append to a local JSONL log with fsync before any network attempt")
    print(f"  - each record carries synced=false until the ops centre acknowledges it")
    print(f"  - on reconnection the uplink replays unsynced records in timestamp order")
    print(f"  Losing the uplink costs visibility at the ops centre, never detection.")

    # Sync burst after the longest gap.
    alerts_in_gap = (gap_hi * 60 / 10) * 0.30  # 30% non-Normal during a real fault
    burst_kb = alerts_in_gap * 300 / KB
    print(f"\nReconnection burst after a {gap_hi}-minute gap with an active fault:")
    print(f"  ~{alerts_in_gap:,.0f} queued alerts x 300 B = {burst_kb:,.1f} KB "
          f"(~Rs {burst_kb / 1000 * TARIFF_PER_MB:.3f})")
    print(f"  Trivial to transmit: the queue drains in well under a second.")


# ==========================================================================
# TASK A1 -- PRIVACY
# ==========================================================================
def task_a1_privacy():
    rule("TASK A1 (d) -- PRIVACY: what on-device inference lets FreightBridge promise")

    print("The pharmaceutical clients' requirement is contractual, not merely")
    print("technical: they need to warrant that cargo condition data cannot be")
    print("reached by unauthorised third parties. On-device inference changes what")
    print("can honestly be written into that contract.\n")
    print("  1. DATA MINIMISATION AS AN ARCHITECTURAL FACT. Raw telemetry never")
    print("     leaves the vehicle; only classifications and alerts do. A party who")
    print("     intercepts the uplink obtains 'TRK-014 reported Warning at 14:22',")
    print("     not a reconstructable thermal history of a named consignment.")
    print("  2. A SMALLER ATTACK SURFACE, STATED CONCRETELY. Cloud inference exposes")
    print("     every reading at the RAN, the backhaul, the ingest endpoint, the")
    print("     queue and the datastore. Edge inference reduces that to one hop")
    print("     carrying already-aggregated records.")
    print("  3. NO THIRD-PARTY PROCESSOR IN THE DATA PATH. Under India's DPDP Act")
    print("     2023 a cloud vendor processing client cargo data is a data processor")
    print("     requiring its own contractual chain. On-device inference keeps")
    print("     processing on FreightBridge-owned hardware, so that chain is shorter")
    print("     and auditable.")
    print("  4. TENANT ISOLATION BY CONSTRUCTION. Each truck processes only its own")
    print("     cargo. There is no shared multi-tenant store in which one client's")
    print("     consignment data sits beside a competitor's.")
    print("  5. AN AUDITABLE CUSTODY RECORD. The local append-only log is the")
    print("     chain-of-custody artefact whose absence caused the rejected hospital")
    print("     shipment. It is produced on the truck and survives connectivity loss.")
    print("\n  Honest limit: 'cannot be accessed' is a claim about the whole system,")
    print("  not about inference location. On-device inference is necessary but not")
    print("  sufficient -- it must be paired with disk encryption on the edge node,")
    print("  TLS with client certificates on the uplink, signed model artefacts, and")
    print("  physical tamper-evidence on the enclosure. The report states this rather")
    print("  than implying edge inference alone discharges the obligation.")


# ==========================================================================
# TASK B2 -- ROOFLINE
# ==========================================================================
def task_b2_roofline():
    rule("TASK B2 -- ARITHMETIC INTENSITY AND ROOFLINE (Raspberry Pi 5 CPU)")

    flops = 45e6  # FLOP per inference, as given
    bytes_moved = 18e6  # B per inference, as given
    peak_flops = 16e9  # FLOP/s, NEON SIMD
    peak_bw = 12e9  # B/s, LPDDR4X

    ai = flops / bytes_moved
    ridge = peak_flops / peak_bw
    attainable = min(peak_flops, ai * peak_bw)
    t_compute = flops / peak_flops
    t_memory = bytes_moved / peak_bw

    print(f"Given:")
    print(f"  work per inference        {flops / 1e6:>10.1f} MFLOP")
    print(f"  data moved per inference  {bytes_moved / 1e6:>10.1f} MB")
    print(f"  peak compute              {peak_flops / 1e9:>10.1f} GFLOP/s  (NEON SIMD)")
    print(f"  peak bandwidth            {peak_bw / 1e9:>10.1f} GB/s     (LPDDR4X)\n")

    print(f"Arithmetic Intensity  AI = FLOPs / Bytes")
    print(f"                         = {flops:,.0f} / {bytes_moved:,.0f}")
    print(f"                         = {ai:.3f} FLOP/byte\n")
    print(f"Ridge point           I_ridge = Peak compute / Peak bandwidth")
    print(f"                              = {peak_flops / 1e9:.0f}e9 / {peak_bw / 1e9:.0f}e9")
    print(f"                              = {ridge:.3f} FLOP/byte\n")

    bound = "COMPUTE-BOUND" if ai > ridge else "MEMORY-BANDWIDTH-BOUND"
    print(f"AI ({ai:.3f}) {'>' if ai > ridge else '<'} ridge ({ridge:.3f})  ->  {bound}")
    print(f"The operating point sits on the flat (compute) roof of the model.\n")

    print(f"Attainable performance = min(peak, AI x BW)")
    print(f"                       = min({peak_flops / 1e9:.0f}, "
          f"{ai:.3f} x {peak_bw / 1e9:.0f} = {ai * peak_bw / 1e9:.1f}) "
          f"= {attainable / 1e9:.1f} GFLOP/s\n")
    print(f"Cross-check by time:")
    print(f"  compute time = {flops:,.0f} / {peak_flops:,.0f} = {t_compute * 1000:.3f} ms")
    print(f"  memory time  = {bytes_moved:,.0f} / {peak_bw:,.0f} = {t_memory * 1000:.3f} ms")
    print(f"  compute time is {t_compute / t_memory:.2f}x the memory time, so compute")
    print(f"  dominates and the classification above is confirmed.\n")

    print(f"WHAT THE ROOFLINE SAYS TO OPTIMISE")
    print(f"  On the compute roof, latency falls only by reducing arithmetic or by")
    print(f"  raising effective throughput. Therefore:")
    print(f"    DO    quantise to INT8 -- NEON's SDOT/UDOT execute four int8 MACs")
    print(f"          per lane-slot, lifting the compute roof itself")
    print(f"    DO    prune structurally -- removes MACs outright, unlike")
    print(f"          unstructured sparsity which leaves the MAC count unchanged")
    print(f"    DO    offload to the Hailo-8L on the AI HAT+ (13 TOPS), which")
    print(f"          replaces the 16 GFLOP/s roof with a far higher integer one")
    print(f"    DON'T spend effort on data layout, tiling, prefetch or cache")
    print(f"          blocking -- those raise the slanted bandwidth roof, and this")
    print(f"          model is not on it. They would buy nothing.\n")

    print(f"HONESTY NOTE")
    print(f"  The figures above are the ones the problem statement supplies and the")
    print(f"  analysis uses them as given. They do not describe the model actually")
    print(f"  built here: our MLP has 803 parameters and performs roughly 1.6 kFLOP")
    print(f"  per inference against a few kilobytes of weights -- about four orders")
    print(f"  of magnitude below the stated figures. Measured latency is ~1 us, not")
    print(f"  the 2.8 ms the stated figures imply. For a model this small the")
    print(f"  Roofline is the wrong instrument entirely: runtime is dominated by")
    print(f"  interpreter invocation overhead, not by arithmetic or by memory")
    print(f"  traffic, which is exactly what the benchmark in Task F2 measured when")
    print(f"  removing 50% of the parameters changed latency by nothing.")
    return {"ai": ai, "ridge": ridge, "bound": bound}


# ==========================================================================
# TASK D2 -- DOCKER LAYER CACHE SAVING
# ==========================================================================
def task_d2_layer_cache():
    rule("TASK D2 -- DOCKER LAYER CACHE: bandwidth saved across 85 trucks")

    base_mb = 130.0  # python:3.11-slim, compressed
    deps_mb = 58.0  # tflite-runtime + numpy + paho-mqtt, compressed
    code_kb = 40.0  # application code + training_stats.npy
    model_kb = 4.55  # m2_int8.tflite, measured

    full_mb = base_mb + deps_mb + code_kb / 1000 + model_kb / 1000
    model_layer_mb = model_kb / 1000

    print(f"Image layers (compressed transfer sizes):")
    print(f"  1 base  python:3.11-slim          {base_mb:>10.2f} MB")
    print(f"  2 deps  pip install -r reqs       {deps_mb:>10.2f} MB")
    print(f"  3 code  app + training_stats.npy  {code_kb / 1000:>10.4f} MB")
    print(f"  4 model model.tflite              {model_layer_mb:>10.5f} MB")
    print(f"  {'-' * 44}")
    print(f"  full image                        {full_mb:>10.2f} MB\n")

    print(f"A model-only change invalidates layer 4 alone, because every pip")
    print(f"install sits above it in the Dockerfile. Layers 1-3 are already on")
    print(f"the truck and are not retransmitted.\n")

    for n, label in ((1, "per truck"), (PILOT_TRUCKS, f"pilot {PILOT_TRUCKS} trucks"),
                     (FULL_FLEET, f"full fleet {FULL_FLEET} trucks")):
        naive = full_mb * n
        cached = model_layer_mb * n
        print(f"{label:<28}{'full image: ' + format(naive, ',.1f') + ' MB':>26}"
              f"{'  cached: ' + format(cached, ',.4f') + ' MB':>22}")

    naive_pilot = full_mb * PILOT_TRUCKS
    cached_pilot = model_layer_mb * PILOT_TRUCKS
    print(f"\nFor the {PILOT_TRUCKS}-truck pilot, one model update:")
    print(f"  without layer caching : {naive_pilot:>12,.1f} MB  = "
          f"Rs {naive_pilot * TARIFF_PER_MB:>10,.2f}")
    print(f"  with layer caching    : {cached_pilot:>12,.4f} MB  = "
          f"Rs {cached_pilot * TARIFF_PER_MB:>10,.2f}")
    print(f"  saved                 : {naive_pilot - cached_pilot:>12,.1f} MB  = "
          f"Rs {(naive_pilot - cached_pilot) * TARIFF_PER_MB:>10,.2f}")
    print(f"  reduction factor      : {naive_pilot / cached_pilot:>12,.0f}x")

    cycles = DAYS_PER_YEAR / 42  # every 6 weeks
    annual = (naive_pilot - cached_pilot) * TARIFF_PER_MB * cycles
    print(f"\nAt one update every 6 weeks ({cycles:.1f} cycles/year), layer ordering")
    print(f"alone saves Rs {annual:,.0f}/year on the pilot fleet and "
          f"Rs {annual * FULL_FLEET / PILOT_TRUCKS:,.0f}/year at full scale.")
    print(f"\nThis is the entire reason the problem statement requires pip install")
    print(f"layers before COPY model.tflite. Reversing those two instructions")
    print(f"multiplies every model update by {naive_pilot / cached_pilot:,.0f}.")
    return {"saved_mb_pilot": naive_pilot - cached_pilot}


# ==========================================================================
# TASK E3 -- OTA STRATEGY SELECTION
# ==========================================================================
def task_e3_ota():
    rule("TASK E3 -- OTA STRATEGY: full replacement vs canary vs shadow")

    model_kb = 280.0
    model_mb = model_kb / 1000
    canary_n = 10
    weeks = 6
    cycles_per_year = 52 / weeks
    shadow_days = 14

    print(f"Model {model_kb:.0f} KB = {model_mb:.3f} MB | {PILOT_TRUCKS} trucks | "
          f"Rs {TARIFF_PER_MB:.2f}/MB | every {weeks} weeks\n")

    # --- full replacement
    full_mb = model_mb * PILOT_TRUCKS
    full_cost = full_mb * TARIFF_PER_MB

    # --- canary: same total in the success case, far cheaper on abort
    canary_mb = model_mb * PILOT_TRUCKS
    canary_cost = canary_mb * TARIFF_PER_MB
    abort_mb = model_mb * canary_n
    abort_cost = abort_mb * TARIFF_PER_MB

    # --- shadow: model to every truck, plus comparison telemetry
    disagree_rate = 0.02
    inferences_day = SECONDS_PER_DAY / 10
    record_b = 300
    shadow_tel_mb = (inferences_day * disagree_rate * record_b * shadow_days
                     * PILOT_TRUCKS) / MB
    shadow_mb = model_mb * PILOT_TRUCKS + shadow_tel_mb
    shadow_cost = shadow_mb * TARIFF_PER_MB

    print(f"{'strategy':<22}{'MB/cycle':>12}{'Rs/cycle':>12}{'Rs/year':>12}"
          f"{'blast radius':>16}")
    print("-" * 74)
    for name, mb, radius in (
        ("full replacement", full_mb, f"all {PILOT_TRUCKS}"),
        ("canary (10 first)", canary_mb, f"{canary_n} then {PILOT_TRUCKS - canary_n}"),
        ("shadow mode", shadow_mb, f"0 (advisory)"),
    ):
        c = mb * TARIFF_PER_MB
        print(f"{name:<22}{mb:>12,.2f}{c:>12,.2f}{c * cycles_per_year:>12,.2f}"
              f"{radius:>16}")

    print(f"\nCanary abort case: only {canary_n} trucks have pulled the model, so an")
    print(f"aborted rollout costs {abort_mb:,.2f} MB = Rs {abort_cost:,.2f} instead of "
          f"Rs {full_cost:,.2f}.")
    print(f"Shadow telemetry assumes {disagree_rate:.0%} disagreement uploaded over "
          f"{shadow_days} days:")
    print(f"  {shadow_tel_mb:,.1f} MB on top of the model, i.e. "
          f"{shadow_mb / full_mb:.1f}x the full-replacement volume.")

    print(f"\nRECOMMENDATION: CANARY")
    print(f"  On cost the three are not meaningfully separated -- Rs {full_cost:.2f} "
          f"versus Rs {canary_cost:.2f}")
    print(f"  versus Rs {shadow_cost:.2f} per cycle is noise against a Rs 28 lakh "
          f"spoilage event. The")
    print(f"  decision is therefore about risk, and canary wins on three specifics:\n")
    print(f"  1. SAFETY-CRITICALITY BOUNDS THE BLAST RADIUS. A regression in Class 2")
    print(f"     recall means undetected spoilage. Canary exposes {canary_n} trucks to that")
    print(f"     risk for one validation window instead of {PILOT_TRUCKS} at once.")
    print(f"  2. RURAL CONNECTIVITY MAKES ROLLBACK SLOW. A truck in one of the seven")
    print(f"     dead zones cannot be reached for up to 90 minutes. Under full")
    print(f"     replacement a bad model is already on all {PILOT_TRUCKS} trucks and some of")
    print(f"     them are unreachable; the fleet cannot be recalled on demand.")
    print(f"  3. VALIDATION IS CHEAP AND DIRECT. Ten trucks running the new model for")
    print(f"     one week produce real Class 2 recall on real routes, which is the")
    print(f"     metric Task F3 gates deployment on.\n")
    print(f"  AGAINST FULL REPLACEMENT: no validation stage at all. It deploys an")
    print(f"  unproven model simultaneously to every vehicle carrying temperature-")
    print(f"  sensitive pharmaceuticals, with a slow and partial rollback path.\n")
    print(f"  AGAINST SHADOW MODE: it is the technically safest option and still the")
    print(f"  wrong choice here. It requires both models resident and both executed")
    print(f"  per window, roughly doubling inference work and memory on a 7.5 W node;")
    print(f"  it costs {shadow_mb / full_mb:.1f}x the bandwidth; and it delays every rollout by the")
    print(f"  {shadow_days}-day comparison window. That overhead is justified for a model whose")
    print(f"  failure modes are unknown. For an 803-parameter MLP whose decision")
    print(f"  boundary is six interpretable features, canary validation on real")
    print(f"  routes answers the same question at a fraction of the cost.")
    return {"full": full_cost, "canary": canary_cost, "shadow": shadow_cost}


# ==========================================================================
# TASK B1 -- HARDWARE COST TABLE (supporting the Constraint Triangle)
# ==========================================================================
def task_b1_hardware():
    rule("TASK B1 -- HARDWARE COST AND POWER (Constraint Triangle inputs)")

    power_budget_w = 10.0
    options = [
        ("1  Pi 5 8GB + AI HAT+ (13 TOPS Hailo-8L)", 15_000, 7.5, 13_000),
        ("2  Jetson Orin Nano Super (67 TOPS)", 45_000, 15.0, 67_000),
        ("3  STM32H7 custom MCU + sensor ICs", 3_500, 0.4, 2),
    ]
    print(f"Power budget: {power_budget_w:.0f} W AI budget from a 12 V truck supply "
          f"via DC-DC converter.\n")
    print(f"{'option':<42}{'Rs/truck':>10}{'TDP W':>8}{'GOPS':>8}"
          f"{'pilot Rs':>13}{'fleet Rs':>14}")
    print("-" * 95)
    for name, price, tdp, gops in options:
        print(f"{name:<42}{price:>10,}{tdp:>8.1f}{gops / 1000:>8.1f}"
              f"{price * PILOT_TRUCKS:>13,}{price * FULL_FLEET:>14,}")

    print(f"\nPower budget check against {power_budget_w:.0f} W:")
    for name, price, tdp, _ in options:
        verdict = "FITS" if tdp <= power_budget_w else "EXCEEDS BUDGET"
        print(f"  {name[:40]:<42}{tdp:>6.1f} W   {verdict}")

    print(f"\nFleet-scale cost delta against Option 1:")
    base = options[0][1]
    for name, price, _, _ in options[1:]:
        d = (price - base) * FULL_FLEET
        print(f"  {name[:40]:<42}{'Rs ' + format(d, '+,'):>16}"
              f"  ({abs(d) / 100000:,.1f} lakh {'more' if d > 0 else 'less'})")

    print(f"\nDOMINANT CONSTRAINT VERTEX: COST, with POWER as a hard feasibility gate.")
    print(f"  Option 2 is eliminated before cost is even considered: at 15 W under")
    print(f"  moderate load it exceeds the {power_budget_w:.0f} W budget, and its "
          f"67 TOPS is spent on")
    print(f"  a 6-feature MLP measured at roughly one microsecond per inference. It")
    print(f"  also costs Rs {(45000 - 15000) * FULL_FLEET / 100000:,.1f} lakh more "
          f"across the fleet for capability the")
    print(f"  workload cannot use.")
    print(f"\n  Option 3 is the most tempting on the two vertices the triangle usually")
    print(f"  rewards -- Rs 3,500 and 0.4 W -- and it is still wrong, for a reason")
    print(f"  that is not about inference at all. An 803-parameter INT8 MLP would run")
    print(f"  perfectly well on an STM32H7 via TFLite Micro. What it cannot run is")
    print(f"  Components D and E of this system: Docker containerisation, the")
    print(f"  layer-cached OTA path, the Ansible-managed deployment, and a local MQTT")
    print(f"  broker. Choosing Option 3 means rebuilding the entire MLOps story as")
    print(f"  bespoke firmware with a custom bootloader and a custom update protocol")
    print(f"  -- across 265 vehicles, with no rollback story in a connectivity dead")
    print(f"  zone. The Rs {(15000 - 3500) * FULL_FLEET / 100000:,.1f} lakh saved on "
          f"hardware is spent several times over on")
    print(f"  firmware engineering and fleet-management risk.")
    print(f"\n  RECOMMENDATION: Option 1, Raspberry Pi 5 + AI HAT+.")
    print(f"    - 7.5 W fits the {power_budget_w:.0f} W budget with "
          f"{power_budget_w - 7.5:.1f} W of headroom")
    print(f"    - Rs {15000 * FULL_FLEET / 100000:,.1f} lakh at full fleet, "
          f"Rs {(45000 - 15000) * FULL_FLEET / 100000:,.1f} lakh below Option 2")
    print(f"    - runs a full Linux userspace, so Docker, Ansible and Mosquitto work")
    print(f"      as deployed everywhere else, not as a bespoke embedded variant")
    print(f"    - the Hailo-8L is headroom, not need: it leaves room to add the")
    print(f"      vision-based load-integrity model the operations director will ask")
    print(f"      for once the pilot succeeds, without a second hardware refresh")
    print(f"    - 90 s SLA: measured inference is ~1 us, roughly 8 orders of")
    print(f"      magnitude inside budget. Latency does not discriminate between")
    print(f"      these options, which is precisely why cost and manageability decide.")


# ==========================================================================
if __name__ == "__main__":
    print("LogiEdge -- Report Calculations")
    print("BITS Pilani WILP | AIML ZG535 Machine Learning on Edge | Group 18")
    task_a1_latency()
    task_a1_bandwidth()
    task_a1_connectivity()
    task_a1_privacy()
    task_b1_hardware()
    task_b2_roofline()
    task_d2_layer_cache()
    task_e3_ota()
    print(f"\n{SEP}\nEnd of calculations.\n{SEP}")
