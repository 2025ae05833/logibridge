#!/usr/bin/env python3
"""
LogiEdge -- Five-Metric Benchmark and Pareto Analysis  (Tasks F2, F3)

Measures all three variants on the five metrics required by Task F2:

    1. mean inference latency   ms   200 timed runs, 10 warm-up runs discarded
    2. p95 inference latency    ms
    3. model file size          KB
    4. classification accuracy  %    held-out validation split
    5. energy per inference     mJ   E = P x t, P from psutil CPU% x TDP

Also records Class 2 (Critical) recall, which Task F3 requires to exceed 95%
on the recommended variant.

ON THE ENERGY FIGURE
--------------------
E = P x t with P estimated as TDP x (CPU utilisation / 100) is a coarse proxy.
It attributes the whole package TDP to the busy fraction of one core, ignores
DVFS, uncore and memory power, and is measured on development hardware rather
than on the Raspberry Pi 5 the fleet would actually carry. It is therefore
useful for RANKING the three variants against each other, which is what the
Pareto analysis needs, and not as an absolute per-inference energy budget. A
deployment decision on absolute energy would need a shunt measurement on the
target board. This limitation is stated in the report rather than hidden.

Outputs
-------
    results/benchmark_results.csv
    results/pareto_chart.png
    results/benchmark_results.json
"""

import argparse
import json
import os
import statistics
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.join(os.path.dirname(HERE), "training")
sys.path.insert(0, TRAINING)

from common import (  # noqa: E402
    CLASS_NAMES,
    MODELS,
    confusion_matrix,
    load_dataset,
    per_class_recall,
)

import tensorflow as tf  # noqa: E402

WARMUP_RUNS = 10
TIMED_RUNS = 200  # per the problem statement
REPEATS = 5  # independent 200-run blocks; see note below
ENERGY_SECONDS = 2.0  # sustained-load block for the energy estimate
DEFAULT_TDP = 15.0  # W, development laptop package TDP

# WHY FIVE REPEATS OF THE SPECIFIED 200 RUNS
# ------------------------------------------
# This model executes in roughly one microsecond, which is close enough to the
# timer and scheduler noise floor that a single 200-run block is not
# repeatable: an early version of this script reported M3 as 1.6x slower than
# M2 on one run and identical on the next. Each measurement below is still the
# specified 200 timed runs after 10 discarded warm-up runs; we simply take five
# independent such blocks and report the mean across them, together with the
# spread, so the headline number carries its own evidence of stability.

# Validated categorical palette, slots 1-3 (see reports/ for the validation run)
COLORS = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a"}

VARIANTS = [
    ("M1", "FP32 baseline", "m1_fp32.tflite"),
    ("M2", "PTQ INT8", "m2_int8.tflite"),
    ("M3", "Pruned 35% + INT8", "m3_pruned_int8.tflite"),
]


def make_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    return interp


def quantise_input(x, detail):
    if detail["dtype"] == np.int8:
        scale, zero = detail["quantization"]
        return np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
    return x.astype(np.float32)


def dequantise_output(y, detail):
    if detail["dtype"] == np.int8:
        scale, zero = detail["quantization"]
        return (y.astype(np.float32) - zero) * scale
    return y


def evaluate_accuracy(interp, X, y):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = np.empty(X.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        interp.set_tensor(inp["index"], quantise_input(X[i : i + 1], inp))
        interp.invoke()
        preds[i] = dequantise_output(interp.get_tensor(out["index"])[0], out).argmax()
    cm = confusion_matrix(y, preds)
    return float((preds == y).mean()), cm, per_class_recall(cm)


def measure_latency(interp, sample, tdp, energy_seconds=ENERGY_SECONDS):
    """Return (mean_ms, p95_ms, energy_mJ, cpu_pct) for one variant.

    Latency and energy are measured in two separate blocks on purpose.

    Latency uses the TIMED_RUNS individually-timed invocations the problem
    statement specifies, after WARMUP_RUNS discarded runs.

    Energy cannot use that same block. This model runs in under two
    microseconds, so 200 invocations complete in well under a millisecond --
    far shorter than psutil's CPU sampling resolution, which simply returns a
    meaningless number over an interval that short. The energy block therefore
    runs inference continuously for energy_seconds, samples CPU utilisation
    across that whole interval, and divides the resulting energy by the number
    of inferences actually completed.
    """
    inp = interp.get_input_details()[0]
    x = quantise_input(sample, inp)

    for _ in range(WARMUP_RUNS):
        interp.set_tensor(inp["index"], x)
        interp.invoke()

    # ---- block 1: per-invocation latency, REPEATS independent blocks
    block_means, block_p95s = [], []
    for _ in range(REPEATS):
        times = []
        for _ in range(TIMED_RUNS):
            interp.set_tensor(inp["index"], x)
            t0 = time.perf_counter()
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        block_means.append(statistics.fmean(times))
        block_p95s.append(times[int(0.95 * len(times)) - 1])
    mean_ms = statistics.fmean(block_means)
    p95_ms = statistics.fmean(block_p95s)
    spread_ms = max(block_means) - min(block_means)

    # ---- block 2: sustained load for an energy estimate
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # reset this process's sampling window
    t_start = time.perf_counter()
    count = 0
    while time.perf_counter() - t_start < energy_seconds:
        for _ in range(1000):
            interp.set_tensor(inp["index"], x)
            interp.invoke()
        count += 1000
    elapsed = time.perf_counter() - t_start
    cpu_pct = proc.cpu_percent(interval=None)

    power_w = tdp * (cpu_pct / 100.0)
    energy_mj = power_w * (elapsed / count) * 1000.0
    return mean_ms, p95_ms, energy_mj, cpu_pct, spread_ms


def pareto_front(points):
    """Indices of non-dominated points, minimising x and maximising y."""
    front = []
    for i, (xi, yi) in enumerate(points):
        if not any(
            (xj <= xi and yj >= yi) and (xj < xi or yj > yi)
            for j, (xj, yj) in enumerate(points)
            if j != i
        ):
            front.append(i)
    return sorted(front, key=lambda i: points[i][0])


def _label_offsets(values):
    """Alternate label offsets so close-together points do not collide."""
    order = np.argsort(values)
    offs = {}
    for rank, idx in enumerate(order):
        offs[int(idx)] = (12, 9) if rank % 2 == 0 else (12, -20)
    return offs


def plot_pareto(rows, out_path, inference_budget_ms):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # Latency is plotted in microseconds. These models run in about one
    # microsecond, so a millisecond axis rounds every variant to "0.002 ms"
    # and hides exactly the differences the analysis is about.
    lat_us = [1000.0 * r["mean_latency_ms"] for r in rows]
    acc = [r["accuracy_pct"] for r in rows]
    energy = [r["energy_mj"] for r in rows]

    # ---- panel 1: latency vs accuracy, bubble area = file size
    pts = list(zip(lat_us, acc))
    front = pareto_front(pts)
    if len(front) > 1:
        ax1.plot([pts[i][0] for i in front], [pts[i][1] for i in front],
                 "-", color="#52514e", lw=1.4, zorder=1, label="Pareto frontier")

    offs1 = _label_offsets(lat_us)
    for i, r in enumerate(rows):
        ax1.scatter(lat_us[i], acc[i], s=90 + 34 * r["size_kb"],
                    color=COLORS[r["variant"]], edgecolor="#fcfcfb",
                    linewidth=2, zorder=3)
        ax1.annotate(
            f"{r['variant']}  {r['size_kb']:.2f} KB\n"
            f"{lat_us[i]:.2f} us, {r['accuracy_pct']:.1f}%",
            (lat_us[i], acc[i]), textcoords="offset points",
            xytext=offs1[i], fontsize=8.5, color="#0b0b0b",
        )
    ax1.set_xlabel("mean inference latency (us, lower is better)")
    ax1.set_ylabel("validation accuracy (%, higher is better)")
    ax1.set_title("Latency vs accuracy\n(bubble area = model file size)",
                  fontsize=10)
    ax1.grid(alpha=0.25)
    ax1.margins(x=0.45, y=0.45)
    if len(front) > 1:
        ax1.legend(frameon=False, fontsize=8, loc="lower right")

    # ---- panel 2: energy vs accuracy
    pts2 = list(zip(energy, acc))
    front2 = pareto_front(pts2)
    if len(front2) > 1:
        ax2.plot([pts2[i][0] for i in front2], [pts2[i][1] for i in front2],
                 "-", color="#52514e", lw=1.4, zorder=1, label="Pareto frontier")

    offs2 = _label_offsets(energy)
    for i, r in enumerate(rows):
        ax2.scatter(energy[i], acc[i], s=130, color=COLORS[r["variant"]],
                    edgecolor="#fcfcfb", linewidth=2, zorder=3)
        ax2.annotate(f"{r['variant']}  {r['energy_mj']:.4f} mJ",
                     (energy[i], acc[i]), textcoords="offset points",
                     xytext=offs2[i], fontsize=8.5, color="#0b0b0b")
    ax2.set_xlabel("energy per inference (mJ, lower is better)")
    ax2.set_ylabel("validation accuracy (%)")
    ax2.set_title("Energy vs accuracy", fontsize=10)
    ax2.grid(alpha=0.25)
    ax2.margins(x=0.45, y=0.45)
    if len(front2) > 1:
        ax2.legend(frameon=False, fontsize=8, loc="lower right")

    budget = (
        f"Inference is not the binding term in the 90 s SLA. After 30 s of window fill, "
        f"up to 10 s of step alignment and ~100 ms for the local log and MQTT publish, "
        f"{inference_budget_ms / 1000:,.1f} s remain for inference; the slowest variant uses "
        f"{max(lat_us):.2f} us of it.\nOptimisation here buys file size and integer-only "
        f"hardware compatibility, not latency."
    )
    fig.suptitle("LogiEdge model optimisation: Pareto analysis", fontsize=12.5)
    fig.text(0.5, -0.10, budget, ha="center", fontsize=8.6, color="#52514e")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Benchmark the three LogiEdge variants")
    ap.add_argument("--tdp", type=float, default=DEFAULT_TDP,
                    help="package TDP in watts used for the energy estimate")
    ap.add_argument("--split", default="random", choices=("random", "blocked"))
    args = ap.parse_args()

    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)

    _, _, Xva, yva, _ = load_dataset(split=args.split)
    sample = Xva[:1].astype(np.float32)

    sla_ms = 90_000.0

    rows = []
    print(f"benchmarking on the {args.split} split "
          f"({Xva.shape[0]} validation windows), TDP {args.tdp:.0f} W\n")

    for tag, desc, fname in VARIANTS:
        path = os.path.join(MODELS, fname)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run the training scripts first")
        size_kb = os.path.getsize(path) / 1024.0

        interp = make_interpreter(path)
        acc, cm, rec = evaluate_accuracy(interp, Xva, yva)
        mean_ms, p95_ms, energy_mj, cpu_pct, spread_ms = measure_latency(
            interp, sample, args.tdp
        )

        rows.append({
            "variant": tag,
            "description": desc,
            "mean_latency_ms": round(mean_ms, 4),
            "p95_latency_ms": round(p95_ms, 4),
            "size_kb": round(size_kb, 2),
            "accuracy_pct": round(100 * acc, 2),
            "energy_mj": round(energy_mj, 7),
            "class2_recall_pct": round(100 * rec[2], 2),
            "confusion_matrix": cm.tolist(),
            "cpu_pct_during_block": round(cpu_pct, 1),
            "latency_spread_ms": round(spread_ms, 6),
        })
        print(f"{tag} {desc:<20} {mean_ms:7.4f} ms mean  {p95_ms:7.4f} ms p95  "
              f"{size_kb:6.2f} KB  {100 * acc:6.2f}%  {energy_mj:.6f} mJ  "
              f"C2 recall {100 * rec[2]:5.1f}%  (+/-{spread_ms * 1000:.3f} us)")

    # ------------------------------------------------------- the 15-cell table
    print(f"\n{'':<6}{'mean ms':>10}{'p95 ms':>10}{'size KB':>10}"
          f"{'acc %':>9}{'energy mJ':>13}")
    print("-" * 58)
    for r in rows:
        print(f"{r['variant']:<6}{r['mean_latency_ms']:>10.4f}"
              f"{r['p95_latency_ms']:>10.4f}{r['size_kb']:>10.2f}"
              f"{r['accuracy_pct']:>9.2f}{r['energy_mj']:>13.6f}")

    # ------------------------------------------------------------- relative
    base = rows[0]
    print(f"\nrelative to M1:")
    for r in rows[1:]:
        ratio = base["mean_latency_ms"] / r["mean_latency_ms"]
        word = "faster" if ratio >= 1.0 else "SLOWER"
        shown = ratio if ratio >= 1.0 else 1.0 / ratio
        print(f"  {r['variant']}: {shown:.2f}x {word}, "
              f"{base['size_kb'] / r['size_kb']:.2f}x smaller, "
              f"{r['accuracy_pct'] - base['accuracy_pct']:+.2f} pp accuracy, "
              f"{base['energy_mj'] / r['energy_mj']:.2f}x energy ratio")

    # ---------------------------------------------------------------- outputs
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    with open(csv_path, "w") as fh:
        fh.write("variant,description,mean_latency_ms,p95_latency_ms,size_kb,"
                 "accuracy_pct,energy_mj,class2_recall_pct\n")
        for r in rows:
            fh.write(f"{r['variant']},{r['description']},{r['mean_latency_ms']},"
                     f"{r['p95_latency_ms']},{r['size_kb']},{r['accuracy_pct']},"
                     f"{r['energy_mj']},{r['class2_recall_pct']}\n")

    png_path = os.path.join(results_dir, "pareto_chart.png")

    with open(os.path.join(results_dir, "benchmark_results.json"), "w") as fh:
        json.dump({
            "config": {
                "warmup_runs": WARMUP_RUNS, "timed_runs": TIMED_RUNS,
                "tdp_w": args.tdp, "split": args.split,
                "n_val_windows": int(Xva.shape[0]),
            },
            "results": rows,
            "class_names": CLASS_NAMES,
        }, fh, indent=2)

    # ----------------------------------------------- F3 recommendation inputs
    #
    # Decomposing the 90 s SLA matters more than quoting a headroom multiple.
    # A fault appearing at time t_f is not visible to the classifier until a
    # window containing enough faulty samples closes. Windows are 30 s long
    # and close every 10 s, so in the worst case the pipeline waits one full
    # window plus one step alignment before the fault dominates the features.
    # That windowing term, not the model, is what consumes the SLA.
    window_fill_ms = 30_000.0  # window length: fault must fill the window
    step_align_ms = 10_000.0  # worst-case wait for the next window to close
    alert_path_ms = 100.0  # local log write + MQTT publish, measured budget
    fixed_ms = window_fill_ms + step_align_ms + alert_path_ms
    inference_budget_ms = sla_ms - fixed_ms

    print(f"\nTask F3 -- 90 s SLA decomposition")
    print(f"  window fill (30 s window)            {window_fill_ms:>10,.0f} ms")
    print(f"  step alignment (worst case 10 s)     {step_align_ms:>10,.0f} ms")
    print(f"  local log + MQTT alert publish       {alert_path_ms:>10,.0f} ms")
    print(f"  {'-' * 52}")
    print(f"  remaining budget for inference       {inference_budget_ms:>10,.0f} ms")
    print(f"\n  The binding term is the 40 s windowing latency, not the model.")
    for r in rows:
        pct = 100.0 * r["mean_latency_ms"] / sla_ms
        ok = "PASS" if r["class2_recall_pct"] > 95.0 else "FAIL"
        print(f"  {r['variant']}: inference uses {r['mean_latency_ms']:.4f} ms "
              f"= {pct:.7f}% of the 90 s SLA | "
              f"Class 2 recall {r['class2_recall_pct']:.1f}% ({ok} the 95% bar)")

    plot_pareto(rows, png_path, inference_budget_ms)

    print(f"\nsaved -> {os.path.relpath(csv_path, HERE)}, "
          f"{os.path.relpath(png_path, HERE)}")


if __name__ == "__main__":
    main()
