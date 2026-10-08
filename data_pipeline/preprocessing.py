#!/usr/bin/env python3
"""
LogiEdge -- Preprocessing and Feature Extraction Pipeline  (Tasks C2, C3)

Pipeline stages, in the order mandated by Task C2:

  1. Filtering      5-sample moving average on the temperature and vibration
                    streams. Door events are discrete and are not filtered.
  2. Feature        30-second sliding window, 10-second step, producing a
     extraction     6-value vector:
                        0  temp_mean       C
                        1  temp_std        C
                        2  temp_roc        C/min   (least-squares slope)
                        3  vib_rms         g
                        4  vib_peak        g
                        5  vib_kurtosis    dimensionless (excess kurtosis)
  3. Normalisation  z-score using mean/std computed once from 10 minutes of
                    clean Normal-class output and frozen in training_stats.npy.
                    Statistics are NEVER recomputed from live data -- see the
                    note below.

DATA FUSION (Task C3) -- feature-level
--------------------------------------
Features are extracted from each sensor independently and then concatenated
into one 6-value joint vector before the model sees them. This is feature-level
(early) fusion. See reports/ for the argument against the alternatives; in
short, data-level fusion is impossible without resampling two streams that run
at 1 Hz and 0.5 Hz onto a common time base, and decision-level fusion would
need one classifier per sensor and would destroy the cross-sensor correlation
that separates Class 1 from Class 2 (temperature drift alone is Warning;
temperature drift WITH a vibration signature is Critical).

WHY FROZEN NORMALISATION STATISTICS MATTER
------------------------------------------
If the mean and standard deviation were recomputed from a live rolling buffer,
a slow refrigeration failure would be normalised away: as the cargo warms, the
running mean warms with it, the z-score returns toward zero, and the model sees
"normal" throughout a genuine breach. Freezing the statistics at training time
is what makes drift visible to the classifier. Task C2 requires an experiment
that quantifies the cost of getting this wrong -- see normalisation_experiment.py.
"""

import argparse
import json
import os

import numpy as np

# Keep the sampling constants in one place.
from simulator import TEMP_HZ, VIB_HZ

WINDOW_SECONDS = 30.0
STEP_SECONDS = 10.0
MA_WINDOW = 5  # 5-sample moving average

FEATURE_NAMES = [
    "temp_mean",
    "temp_std",
    "temp_roc",
    "vib_rms",
    "vib_peak",
    "vib_kurtosis",
]
N_FEATURES = len(FEATURE_NAMES)

DEFAULT_STATS_PATH = os.path.join(os.path.dirname(__file__), "training_stats.npy")


# --------------------------------------------------------------------------
# Stage 1 -- filtering
# --------------------------------------------------------------------------
def moving_average(values, k=MA_WINDOW):
    """Causal k-sample moving average.

    The first k-1 outputs average over however many samples have arrived, so
    the output length equals the input length. A causal filter is required
    here because the same function runs on the truck against a live stream,
    where future samples do not exist.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    out = np.empty_like(values)
    cumsum = np.cumsum(values)
    for i in range(values.size):
        lo = max(0, i - k + 1)
        total = cumsum[i] - (cumsum[lo - 1] if lo > 0 else 0.0)
        out[i] = total / (i - lo + 1)
    return out


# --------------------------------------------------------------------------
# Stage 2 -- feature extraction
# --------------------------------------------------------------------------
def excess_kurtosis(x):
    """Fisher (excess) kurtosis, returning 0.0 for a degenerate window.

    Implemented directly rather than via scipy so the inference container does
    not have to carry scipy -- it saves roughly 30 MB in the Docker image.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return 0.0
    m = x.mean()
    s2 = np.mean((x - m) ** 2)
    if s2 <= 1e-12:
        return 0.0
    return float(np.mean((x - m) ** 4) / (s2**2) - 3.0)


def rate_of_change(times, values):
    """Least-squares slope in C per minute.

    A least-squares fit over the whole window is used rather than a simple
    endpoint difference because the endpoints carry the full sensor noise,
    whereas the fit averages it down across all 30 samples.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size < 2 or np.ptp(times) < 1e-9:
        return 0.0
    slope = np.polyfit(times, values, 1)[0]  # C per second
    return float(slope * 60.0)


def extract_features(temp_times, temp_values, vib_values):
    """Build the 6-value feature vector for one window.

    Inputs are already moving-average filtered.
    """
    temp_values = np.asarray(temp_values, dtype=np.float64)
    vib_values = np.asarray(vib_values, dtype=np.float64)

    temp_mean = float(temp_values.mean()) if temp_values.size else 0.0
    temp_std = float(temp_values.std()) if temp_values.size else 0.0
    temp_roc = rate_of_change(temp_times, temp_values)

    if vib_values.size:
        vib_rms = float(np.sqrt(np.mean(vib_values**2)))
        vib_peak = float(vib_values.max())
        vib_kurt = excess_kurtosis(vib_values)
    else:
        vib_rms = vib_peak = vib_kurt = 0.0

    return np.array(
        [temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurt],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------
# Streaming window builder -- used offline for datasets and online in Docker
# --------------------------------------------------------------------------
class SlidingWindowExtractor:
    """Accumulates samples and emits a feature vector every STEP_SECONDS.

    The same class backs both the offline dataset builder and the live
    inference service, which guarantees that training-time and serving-time
    features are produced by identical code -- the usual source of
    training/serving skew is two separate implementations drifting apart.
    """

    # A stream whose timestamp jumps backwards by more than this is a new
    # session, not out-of-order delivery.
    RESTART_TOLERANCE_S = 2.0

    def __init__(self, window_s=WINDOW_SECONDS, step_s=STEP_SECONDS):
        self.window_s = window_s
        self.step_s = step_s
        self.temp = []  # list of (sim_t, raw_value)
        self.vib = []
        self.doors = []  # list of (sim_t, 'OPEN'|'CLOSE')
        self.next_emit = window_s  # first full window closes at t = 30 s
        self.last_t = None
        self.restarts = 0

    def reset(self, sim_t):
        """Start a fresh window schedule from sim_t, discarding stale buffers."""
        self.temp = []
        self.vib = []
        self.doors = []
        self.next_emit = sim_t + self.window_s
        self.restarts += 1

    def add(self, stream, sim_t, value):
        # DETECTING A STREAM RESTART
        # --------------------------
        # next_emit only ever advances, so a publisher that restarts its clock
        # at zero is silently ignored until its timestamps climb back past the
        # old value. In the drift demonstration each phase launches its own
        # simulator, so phase 3 was emitting 20 windows where 200 were
        # expected, the 100-sample rolling buffer never flushed, and PSI froze
        # at its injected value instead of recovering.
        #
        # The same thing happens on a truck whenever the sensor node reboots
        # or the service is restarted by an OTA update mid-route: timestamps
        # restart, and without this check the classifier would go quiet until
        # the clock caught up. Treat a backwards jump as a new session and
        # rebase the schedule on it.
        if self.last_t is not None and sim_t < self.last_t - self.RESTART_TOLERANCE_S:
            self.reset(sim_t)
        self.last_t = sim_t

        if stream == "temperature":
            self.temp.append((sim_t, float(value)))
        elif stream == "vibration_rms":
            self.vib.append((sim_t, float(value)))
        elif stream == "door_event":
            self.doors.append((sim_t, value))

    def _trim(self, now):
        """Drop samples older than the window to bound memory on the truck."""
        cutoff = now - self.window_s
        self.temp = [s for s in self.temp if s[0] >= cutoff]
        self.vib = [s for s in self.vib if s[0] >= cutoff]
        self.doors = [d for d in self.doors if d[0] >= cutoff]

    def ready(self, sim_t):
        return sim_t >= self.next_emit

    def emit(self, sim_t):
        """Return (features, window_end) for the window ending at sim_t."""
        lo = sim_t - self.window_s
        t_pairs = [s for s in self.temp if lo <= s[0] <= sim_t]
        v_pairs = [s for s in self.vib if lo <= s[0] <= sim_t]

        # Stage 1: filter, then Stage 2: extract.
        t_times = [p[0] for p in t_pairs]
        t_filt = moving_average([p[1] for p in t_pairs])
        v_filt = moving_average([p[1] for p in v_pairs])

        feats = extract_features(t_times, t_filt, v_filt)
        self.next_emit += self.step_s
        self._trim(sim_t)
        return feats, sim_t

    def door_open_count(self, sim_t):
        lo = sim_t - self.window_s
        return sum(1 for d in self.doors if lo <= d[0] <= sim_t and d[1] == "OPEN")


# --------------------------------------------------------------------------
# Offline driver -- JSONL capture to feature matrix
# --------------------------------------------------------------------------
def windows_from_jsonl(path):
    """Return (features [N,6], window_end_times [N]) from a simulator capture."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["sim_t"])

    ext = SlidingWindowExtractor()
    feats, ends = [], []
    for rec in records:
        ext.add(rec["stream"], rec["sim_t"], rec["value"])
        while ext.ready(rec["sim_t"]):
            f, end = ext.emit(ext.next_emit)
            feats.append(f)
            ends.append(end)
    if not feats:
        return np.zeros((0, N_FEATURES), np.float32), np.zeros((0,), np.float64)
    return np.vstack(feats).astype(np.float32), np.asarray(ends)


# --------------------------------------------------------------------------
# Stage 3 -- normalisation
# --------------------------------------------------------------------------
def fit_stats(features):
    """Compute per-feature mean and std from clean Normal-class windows."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # guard constant features
    return np.stack([mean, std]).astype(np.float32)


def save_stats(stats, path=DEFAULT_STATS_PATH):
    np.save(path, stats)
    return path


def load_stats(path=DEFAULT_STATS_PATH):
    stats = np.load(path)
    if stats.shape != (2, N_FEATURES):
        raise ValueError(f"training_stats.npy has shape {stats.shape}, expected (2,6)")
    return stats


def normalise(features, stats):
    mean, std = stats[0], stats[1]
    return ((features - mean) / std).astype(np.float32)


# --------------------------------------------------------------------------
# CLI -- build training_stats.npy from a clean capture
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Fit normalisation statistics from a clean Normal capture"
    )
    p.add_argument("--clean", required=True, help="JSONL capture from --anomaly none")
    p.add_argument("--out", default=DEFAULT_STATS_PATH)
    args = p.parse_args()

    feats, ends = windows_from_jsonl(args.clean)
    if feats.shape[0] == 0:
        raise SystemExit("no complete windows in capture")

    minutes = (ends[-1] - ends[0] + WINDOW_SECONDS) / 60.0
    stats = fit_stats(feats)
    save_stats(stats, args.out)

    print(f"windows: {feats.shape[0]}  ({minutes:.1f} minutes of clean data)")
    print(f"{'feature':<14}{'mean':>12}{'std':>12}")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"{name:<14}{stats[0, i]:>12.5f}{stats[1, i]:>12.5f}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
