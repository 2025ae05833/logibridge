#!/usr/bin/env python3
"""
LogiEdge -- Labelled Dataset Generation  (Task D1)

Runs the simulator in each anomaly mode for the durations mandated by the
problem statement, extracts 30-second/10-second-step feature windows, fits the
frozen normalisation statistics from the clean capture, and writes a single
dataset.npz.

    Class 0  Normal     --anomaly none        20 min   ~120 windows
    Class 1  Warning    --anomaly temp_drift  15 min    ~90 windows
    Class 2  Critical   --anomaly combined    15 min    ~90 windows

TWO VALIDATION SPLITS
---------------------
The problem statement asks for a 20% held-out validation split. A naive random
split is optimistic here: consecutive windows overlap by 20 of their 30 seconds,
so a window in the validation set shares two thirds of its raw samples with its
neighbours in the training set. That is information leakage, and it inflates the
reported accuracy.

We therefore produce both:

  random    stratified random 20% -- the split the problem statement specifies,
            and the number reported against the 88% gate.
  blocked   the final 20% of each capture by time, with a 30-second guard band
            dropped at the boundary so no validation window shares raw samples
            with any training window. This is the honest generalisation estimate.

Both are reported. Where they disagree, the blocked number is the one that
describes how the model will behave on a truck.

Usage
-----
    python generate_dataset.py                 # full spec durations
    python generate_dataset.py --speed 200     # faster wall clock
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(os.path.dirname(HERE), "data_pipeline")
sys.path.insert(0, PIPELINE)

from preprocessing import (  # noqa: E402
    FEATURE_NAMES,
    STEP_SECONDS,
    WINDOW_SECONDS,
    fit_stats,
    normalise,
    save_stats,
    windows_from_jsonl,
)

CLASS_NAMES = ["Normal", "Warning", "Critical"]

# (label, simulator mode, simulated seconds, RNG seed)
CAPTURES = [
    (0, "none", 1200, 1001),
    (1, "temp_drift", 900, 1002),
    (2, "combined", 900, 1003),
]

STATS_MINUTES = 10.0  # clean data used to fit normalisation statistics
GUARD_SECONDS = WINDOW_SECONDS  # dropped at the blocked-split boundary


def run_capture(mode, duration, seed, speed, out_path):
    """Invoke simulator.py as a subprocess, writing a JSONL capture."""
    cmd = [
        sys.executable,
        os.path.join(PIPELINE, "simulator.py"),
        "--anomaly", mode,
        "--duration", str(duration),
        "--speed", str(speed),
        "--sink", "file",
        "--file", out_path,
        "--seed", str(seed),
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


def blocked_split_mask(ends, frac=0.2):
    """True where a window belongs to the time-blocked validation tail.

    Windows within GUARD_SECONDS before the cut are dropped from both sides so
    that no validation window shares raw samples with a training window.
    """
    cut = ends.min() + (1.0 - frac) * (ends.max() - ends.min())
    val = ends > cut
    guard = (ends <= cut) & (ends > cut - GUARD_SECONDS)
    return val, guard


def main():
    ap = argparse.ArgumentParser(description="Build the LogiEdge training dataset")
    ap.add_argument("--speed", type=float, default=400.0,
                    help="simulator wall-clock acceleration (default 400x)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "data"))
    ap.add_argument("--seed", type=int, default=20250518,
                    help="RNG seed for the random split")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------------------------------------------- capture
    feats_by_class, ends_by_class = {}, {}
    print(f"{'class':<10}{'mode':<12}{'sim min':>9}{'windows':>9}")
    print("-" * 40)
    for label, mode, duration, seed in CAPTURES:
        path = os.path.join(args.outdir, f"capture_{mode}.jsonl")
        run_capture(mode, duration, seed, args.speed, path)
        feats, ends = windows_from_jsonl(path)
        feats_by_class[label] = feats
        ends_by_class[label] = ends
        print(f"{CLASS_NAMES[label]:<10}{mode:<12}{duration / 60:>9.0f}"
              f"{feats.shape[0]:>9}")

    # ------------------------------------------- normalisation statistics (C2)
    # Fitted ONLY on the first STATS_MINUTES of the clean Normal capture, and
    # frozen from here on. Never recomputed from live data.
    clean_feats = feats_by_class[0]
    clean_ends = ends_by_class[0]
    keep = clean_ends <= clean_ends.min() + STATS_MINUTES * 60.0
    stats = fit_stats(clean_feats[keep])
    stats_path = save_stats(stats, os.path.join(PIPELINE, "training_stats.npy"))

    print(f"\nnormalisation statistics from {keep.sum()} clean windows "
          f"({STATS_MINUTES:.0f} min)  ->  {os.path.basename(stats_path)}")
    print(f"{'feature':<14}{'mean':>11}{'std':>11}")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"{name:<14}{stats[0, i]:>11.4f}{stats[1, i]:>11.4f}")

    # -------------------------------------------------------- assemble dataset
    X_parts, y_parts, val_blocked_parts, guard_parts = [], [], [], []
    for label, _, _, _ in CAPTURES:
        f = feats_by_class[label]
        e = ends_by_class[label]
        v, g = blocked_split_mask(e)
        X_parts.append(f)
        y_parts.append(np.full(f.shape[0], label, dtype=np.int64))
        val_blocked_parts.append(v)
        guard_parts.append(g)

    X_raw = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    val_blocked = np.concatenate(val_blocked_parts)
    guard = np.concatenate(guard_parts)
    X = normalise(X_raw, stats)

    # ------------------------------------------- stratified random 20% split
    rng = np.random.default_rng(args.seed)
    val_random = np.zeros(y.shape[0], dtype=bool)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        val_random[idx[: int(round(0.2 * idx.size))]] = True

    out = os.path.join(args.outdir, "dataset.npz")
    np.savez_compressed(
        out,
        X=X, X_raw=X_raw, y=y,
        val_random=val_random, val_blocked=val_blocked, guard=guard,
        stats=stats, feature_names=np.array(FEATURE_NAMES),
        class_names=np.array(CLASS_NAMES),
    )

    # ------------------------------------------------------------- summary
    print(f"\ndataset: {X.shape[0]} windows x {X.shape[1]} features "
          f"({WINDOW_SECONDS:.0f} s window, {STEP_SECONDS:.0f} s step)")
    for label in range(3):
        n = int((y == label).sum())
        print(f"  class {label} {CLASS_NAMES[label]:<9} {n:>4} windows "
              f"({100 * n / y.size:4.1f}%)")
    print(f"\nrandom  split: {int(val_random.sum()):>3} val / "
          f"{int((~val_random).sum()):>3} train")
    print(f"blocked split: {int(val_blocked.sum()):>3} val / "
          f"{int((~val_blocked & ~guard).sum()):>3} train "
          f"({int(guard.sum())} dropped as guard band)")

    # Physical sanity check against the Class definitions in the brief.
    print(f"\nclass separation check (raw, un-normalised):")
    print(f"{'':<10}{'temp_mean':>11}{'temp_roc':>11}{'vib_rms':>11}")
    for label in range(3):
        m = X_raw[y == label].mean(axis=0)
        print(f"  {CLASS_NAMES[label]:<8}{m[0]:>11.3f}{m[2]:>11.3f}{m[3]:>11.3f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
