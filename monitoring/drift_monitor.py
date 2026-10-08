#!/usr/bin/env python3
"""
LogiEdge -- PSI Drift Monitoring  (Task E1)

Population Stability Index on the model's output score distribution, computed
over a rolling window of the last 100 inferences, evaluated every 60 seconds.

    PSI = sum over bins of  (actual% - reference%) * ln(actual% / reference%)

    PSI < 0.10   stable
    0.10 - 0.25  moderate shift, watch
    PSI > 0.25   significant shift  ->  [LOGIBRIDGE DRIFT ALERT]

WHICH SCORE TO MONITOR -- MEASURED, NOT ASSUMED
------------------------------------------------
"Output confidence score" has two plausible readings, and which one is chosen
changes how fast drift is caught. Both are implemented and selectable with
--score, and both were measured on the same injected-fault run:

                       crosses PSI>0.25 after    peak PSI
    p_normal  P(class 0)        12 inferences      13.91
    max_conf  max softmax       20 inferences       8.70

P(Normal) is the default because it is the more sensitive of the two: it
collapses from the clean-data range toward zero as the fault develops, so the
histogram migrates all the way to the bottom bin.

The initial expectation while writing this module was that max-softmax would
not fire at all -- the reasoning being that a confident model reports ~0.99
both for Normal before the fault and for Critical after it, leaving the
histogram unmoved. Measurement contradicted that. This model's confidence on
clean Normal data is only moderate (P(Normal) mostly in the 0.25-0.75 bins,
see reference_dist.json), so the shift to confident Critical predictions does
move the distribution and PSI does rise. The original argument would hold for
a better-calibrated model that was confident on both sides; it does not hold
here, and the numbers above are what the report cites.

The transferable point is that the monitored statistic has to be validated
against an actual injected fault rather than reasoned about. The PSI formula
is not what determines whether drift monitoring works.

Usage
-----
  # 1. build the reference distribution from 300 clean Normal windows
  python drift_monitor.py --build-reference --clean-capture clean.jsonl

  # 2. monitor live inferences
  python drift_monitor.py --interval 60
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import pathlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
sys.path.insert(0, os.path.join(ROOT, "training"))

BINS = [0.0, 0.25, 0.50, 0.75, 1.0001]  # 4 bins, upper edge inclusive
BIN_LABELS = ["[0.00,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00]"]
ROLLING_N = 100
ALERT_THRESHOLD = 0.25
RECOVERY_THRESHOLD = 0.10
REFERENCE_WINDOWS = 300
EPSILON = 1e-4  # replaces empty bins; ln(0) is undefined

# PSI is a comparison of two histograms, and a histogram built from a handful
# of samples is mostly noise. An early version evaluated as soon as 10
# inferences had arrived and raised a false [LOGIBRIDGE DRIFT ALERT] at
# PSI=0.405 on clean baseline data, purely from sampling variance in a
# 20-sample window. The monitor now waits until the rolling window is at least
# half full, which removed the false alarm without delaying detection of the
# injected fault.
MIN_SAMPLES = ROLLING_N // 2

REF_PATH = os.path.join(HERE, "reference_dist.json")
TOPIC_ROOT = "logibridge/trucks"


# --------------------------------------------------------------------------
def histogram(scores):
    """Fraction of scores falling in each of the 4 bins."""
    counts, _ = np.histogram(np.clip(scores, 0.0, 1.0), bins=BINS)
    total = counts.sum()
    if total == 0:
        return np.full(len(BIN_LABELS), 1.0 / len(BIN_LABELS))
    return counts / total


def psi(reference, actual):
    """Population Stability Index between two binned distributions.

    Empty bins are floored at EPSILON rather than dropped. Dropping them would
    understate drift precisely when a distribution has moved completely out of
    a bin, which is the strongest drift signal there is.
    """
    ref = np.maximum(np.asarray(reference, dtype=float), EPSILON)
    act = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    ref = ref / ref.sum()
    act = act / act.sum()
    return float(np.sum((act - ref) * np.log(act / ref)))


def model_sha(path, n=12):
    """Short content hash identifying exactly which model artefact this is."""
    h = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    return h[:n]


def band(value):
    if value > ALERT_THRESHOLD:
        return "SIGNIFICANT"
    if value > RECOVERY_THRESHOLD:
        return "moderate"
    return "stable"


# --------------------------------------------------------------------------
def build_reference(clean_capture, model_path, stats_path, score_kind, out_path):
    """Run inference on REFERENCE_WINDOWS clean Normal windows and bin the scores."""
    from preprocessing import load_stats, normalise, windows_from_jsonl

    import tensorflow as tf

    feats, _ = windows_from_jsonl(clean_capture)
    if feats.shape[0] < REFERENCE_WINDOWS:
        raise SystemExit(
            f"capture has {feats.shape[0]} windows, {REFERENCE_WINDOWS} needed. "
            f"Generate at least {REFERENCE_WINDOWS * 10 + 30} s of clean data."
        )
    feats = feats[:REFERENCE_WINDOWS]
    X = normalise(feats, load_stats(stats_path))

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    scores = []
    for i in range(X.shape[0]):
        x = X[i : i + 1].astype(np.float32)
        if inp["dtype"] == np.int8:
            s, z = inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0]
        if out["dtype"] == np.int8:
            s, z = out["quantization"]
            y = (y.astype(np.float32) - z) * s
        scores.append(float(y[0]) if score_kind == "p_normal" else float(y.max()))

    dist = histogram(np.array(scores))
    payload = {
        "score_kind": score_kind,
        "n_windows": int(X.shape[0]),
        "bin_edges": BINS[:-1] + [1.0],
        "bin_labels": BIN_LABELS,
        "distribution": [round(float(v), 6) for v in dist],
        "model": os.path.basename(model_path),
        # A content hash, not just a filename: the deployed model is always
        # called model.tflite regardless of which variant it holds, so only
        # the bytes identify it.
        "model_sha": model_sha(model_path),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"reference built from {X.shape[0]} clean Normal windows "
          f"(score = {score_kind})")
    for lbl, v in zip(BIN_LABELS, dist):
        bar = "#" * int(round(40 * v))
        print(f"  {lbl}  {v:6.3f}  {bar}")
    print(f"saved -> {out_path}")


# --------------------------------------------------------------------------
class DriftMonitor:
    def __init__(self, reference, score_kind, interval, truck_id, ref_model=None):
        self.reference = np.array(reference, dtype=float)
        self.score_kind = score_kind
        self.interval = interval
        self.truck_id = truck_id
        self.ref_model = ref_model
        self.scores = deque(maxlen=ROLLING_N)
        self.history = []
        self.alerting = False
        self.n_seen = 0
        self.model_warned = False
        self.history_path = os.path.join(HERE, "psi_history.json")

    def _check_model(self, payload):
        """Warn if the serving model is not the one the reference was built on.

        PSI compares a live score distribution against a stored one. If the
        container is serving a different model than the reference was built
        from, every bin differs for reasons that have nothing to do with
        drift, and the monitor alerts permanently on clean data.

        This is not hypothetical: an early run of the demo built the reference
        on M2 while the inference container was serving M3 (left in place by
        the OTA layer-cache demonstration), and clean baseline data scored
        PSI = 4.96 instead of ~0.02. The reference and the served model must
        be the same artefact, so the monitor now says so rather than
        reporting a drift that is not there.
        """
        if self.model_warned or not self.ref_model:
            return
        live = payload.get("model_sha") or payload.get("model")
        if live and live != self.ref_model:
            self.model_warned = True
            print(
                f"\n[drift] WARNING: model mismatch.\n"
                f"        reference built on : {self.ref_model}\n"
                f"        container serving  : {live}\n"
                f"        PSI will be meaningless until these match. Rebuild\n"
                f"        the reference with --model pointing at the served\n"
                f"        model, or redeploy the model the reference used.\n",
                flush=True,
            )

    def observe(self, payload):
        self._check_model(payload)
        if self.score_kind == "p_normal":
            probs = payload.get("probabilities")
            score = (
                float(payload["p_normal"])
                if "p_normal" in payload
                else float(probs[0])
            )
        else:
            score = float(payload["confidence"])
        self.scores.append(score)
        self.n_seen += 1

    def evaluate(self):
        """Compute PSI over the rolling window and print one status line."""
        now = datetime.now().strftime("%H:%M:%S")
        if len(self.scores) < MIN_SAMPLES:
            print(f"[{now}] warming up: {len(self.scores)}/{MIN_SAMPLES} inferences "
                  f"before the first PSI evaluation", flush=True)
            return None

        actual = histogram(np.array(self.scores))
        value = psi(self.reference, actual)
        self.history.append({
            "t": now, "psi": round(value, 4), "n": len(self.scores),
            "distribution": [round(float(v), 4) for v in actual],
        })

        self.save_history(self.history_path)

        bars = " ".join(f"{v:.2f}" for v in actual)
        print(f"[{now}] PSI={value:6.3f}  {band(value):<12} "
              f"n={len(self.scores):>3}  bins=[{bars}]", flush=True)

        if value > ALERT_THRESHOLD:
            # Exact alert string required by the problem statement.
            print(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}", flush=True)
            self.alerting = True
        elif self.alerting and value < RECOVERY_THRESHOLD:
            print(f"[LOGIBRIDGE DRIFT RECOVERED] PSI={value:.3f} "
                  f"(below {RECOVERY_THRESHOLD})", flush=True)
            self.alerting = False
        return value

    # WHY THE HISTORY IS WRITTEN AFTER EVERY EVALUATION
    # -------------------------------------------------
    # An earlier version saved psi_history.json only when the monitor shut
    # down cleanly. The demonstration script stops it with SIGINT and then
    # SIGKILLs anything still alive two seconds later, so a slow shutdown --
    # or any crash -- lost the entire trace, which is the evidence Task E1
    # is assessed on. Writing after each evaluation costs a few hundred bytes
    # of I/O per minute and means the file on disk is always current.
    def save_history(self, path):
        with open(path, "w") as fh:
            json.dump({
                "score_kind": self.score_kind,
                "reference": [round(float(v), 6) for v in self.reference],
                "alert_threshold": ALERT_THRESHOLD,
                "recovery_threshold": RECOVERY_THRESHOLD,
                "rolling_window": ROLLING_N,
                "samples": self.history,
            }, fh, indent=2)


# --------------------------------------------------------------------------
def monitor_live(args, reference, ref_model=None):
    import paho.mqtt.client as mqtt

    mon = DriftMonitor(reference, args.score, args.interval, args.truck_id,
                       ref_model=ref_model)

    def on_connect(client, userdata, flags, rc, properties=None):
        topic = f"{TOPIC_ROOT}/{args.truck_id}/inference"
        client.subscribe(topic, qos=1)
        print(f"[drift] subscribed to {topic}", flush=True)

    def on_message(client, userdata, msg):
        try:
            mon.observe(json.loads(msg.payload.decode()))
        except (ValueError, KeyError, TypeError):
            pass

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                             client_id=f"logibridge-drift-{args.truck_id}")
    except AttributeError:
        client = mqtt.Client(client_id=f"logibridge-drift-{args.truck_id}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()

    print(f"[drift] score={args.score}  rolling window={ROLLING_N}  "
          f"evaluating every {args.interval}s  alert at PSI>{ALERT_THRESHOLD}",
          flush=True)
    if mon.ref_model:
        print(f"[drift] reference built on model sha: {mon.ref_model}", flush=True)
    try:
        while True:
            time.sleep(args.interval)
            mon.evaluate()
    except KeyboardInterrupt:
        print("\n[drift] stopping", flush=True)
    finally:
        client.loop_stop()
        client.disconnect()
        out = os.path.join(HERE, "psi_history.json")
        mon.save_history(out)
        print(f"[drift] {mon.n_seen} inferences observed, history -> {out}",
              flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LogiEdge PSI drift monitor")
    ap.add_argument("--build-reference", action="store_true")
    ap.add_argument("--clean-capture", help="JSONL from --anomaly none")
    ap.add_argument("--model",
                    default=os.path.join(ROOT, "training", "models", "m2_int8.tflite"))
    ap.add_argument("--stats",
                    default=os.path.join(ROOT, "data_pipeline", "training_stats.npy"))
    ap.add_argument("--score", choices=("p_normal", "max_conf"), default="p_normal")
    ap.add_argument("--reference", default=REF_PATH)
    ap.add_argument("--interval", type=float, default=60.0,
                    help="seconds between PSI evaluations (default 60)")
    ap.add_argument("--truck-id", default="TRK-001")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    args = ap.parse_args()

    if args.build_reference:
        if not args.clean_capture:
            ap.error("--build-reference requires --clean-capture")
        build_reference(args.clean_capture, args.model, args.stats,
                        args.score, args.reference)
        return

    if not os.path.exists(args.reference):
        raise SystemExit(f"no reference at {args.reference}; run --build-reference")
    with open(args.reference) as fh:
        ref = json.load(fh)
    if ref["score_kind"] != args.score:
        raise SystemExit(
            f"reference was built for score={ref['score_kind']} but "
            f"--score={args.score} was requested; rebuild the reference"
        )
    monitor_live(args, ref["distribution"],
                 ref.get("model_sha") or ref.get("model"))


if __name__ == "__main__":
    main()
