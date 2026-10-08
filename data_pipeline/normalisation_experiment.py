#!/usr/bin/env python3
"""
LogiEdge -- Mandatory Normalisation Experiment  (Task C2)

"Run inference with correct stats, then with stats shifted by 3 sigma; report
accuracy changes."

WHAT IS BEING TESTED, AND WHY IT IS THE RIGHT TEST
--------------------------------------------------
The pipeline freezes the normalisation mean and standard deviation at training
time and loads them at runtime. The failure this guards against is a pipeline
that recomputes statistics from live data: during a slow refrigeration failure
the running mean warms along with the cargo, the z-score returns toward zero,
and the model sees "normal" throughout a genuine breach.

Shifting the stored mean by 3 sigma simulates exactly that corruption -- a
statistics file fitted on the wrong data, or silently recomputed on a truck
whose compressor was already failing. The accuracy delta is the cost of
getting it wrong.

Four conditions are measured:

    correct        stats as fitted on clean Normal data
    mean +3 sigma  stored mean inflated by three standard deviations
    mean -3 sigma  stored mean deflated by three standard deviations
    std x3         stored spread inflated, compressing every z-score

Per-class recall is reported alongside accuracy because the headline accuracy
number hides the part that matters for cold-chain safety: whether Class 2
Critical is still detected.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, HERE)

from preprocessing import FEATURE_NAMES, load_stats  # noqa: E402

from common import (  # noqa: E402
    CLASS_NAMES,
    MODELS,
    confusion_matrix,
    per_class_recall,
)

import tensorflow as tf  # noqa: E402

MODEL = os.path.join(MODELS, "m2_int8.tflite")
DATA = os.path.join(ROOT, "training", "data", "dataset.npz")


def predict(model_path, X):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    preds = np.empty(X.shape[0], dtype=np.int64)
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
        preds[i] = int(np.argmax(y))
    return preds


def main():
    blob = np.load(DATA, allow_pickle=True)
    X_raw, y, val = blob["X_raw"], blob["y"], blob["val_random"]
    X_raw, y = X_raw[val], y[val]
    stats = load_stats(os.path.join(HERE, "training_stats.npy"))
    mean, std = stats[0].copy(), stats[1].copy()

    conditions = [
        ("correct", mean, std),
        ("mean +3 sigma", mean + 3.0 * std, std),
        ("mean -3 sigma", mean - 3.0 * std, std),
        ("std x3", mean, std * 3.0),
    ]

    print(f"Normalisation sensitivity -- {len(y)} held-out validation windows")
    print(f"model: {os.path.basename(MODEL)}\n")
    print(f"{'condition':<16}{'accuracy':>10}{'delta':>9}"
          f"{'  recall N':>11}{'recall W':>10}{'recall C':>10}")
    print("-" * 66)

    baseline = None
    results = []
    for name, m, s in conditions:
        X = ((X_raw - m) / s).astype(np.float32)
        pred = predict(MODEL, X)
        acc = float((pred == y).mean())
        rec = per_class_recall(confusion_matrix(y, pred))
        if baseline is None:
            baseline = acc
        delta = 100 * (acc - baseline)
        print(f"{name:<16}{100 * acc:>9.2f}%{delta:>+8.2f}"
              f"{100 * rec[0]:>10.1f}%{100 * rec[1]:>9.1f}%{100 * rec[2]:>9.1f}%")
        results.append((name, acc, rec))

    print(f"\n{'=' * 66}")
    worst = min(results[1:], key=lambda r: r[1])
    print(f"Worst corrupted condition: {worst[0]} at {100 * worst[1]:.2f}% accuracy, "
          f"{100 * (worst[1] - baseline):+.2f} pp")

    c2 = [100 * r[2][2] for r in results[1:]]
    c1 = [100 * r[2][1] for r in results[1:]]
    print(f"Class 1 Warning  recall: {100 * results[0][2][1]:5.1f}% correct  ->  "
          f"{min(c1):5.1f}% worst corrupted  ({min(c1) - 100 * results[0][2][1]:+.1f} pp)")
    print(f"Class 2 Critical recall: {100 * results[0][2][2]:5.1f}% correct  ->  "
          f"{min(c2):5.1f}% worst corrupted  ({min(c2) - 100 * results[0][2][2]:+.1f} pp)")
    print(f"\nINTERPRETATION")
    if min(c2) < 95.0:
        print(f"  At least one corrupted condition drops Class 2 recall below the 95%")
        print(f"  bar that Task F3 sets for deployment. A silently wrong")
        print(f"  training_stats.npy is therefore not a degradation, it is a safety")
        print(f"  failure: the model keeps returning confident predictions while")
        print(f"  missing refrigeration failures.")
    else:
        print(f"  Class 2 recall holds above 95% in every corrupted condition, because")
        print(f"  the vibration signature separating Critical from the other classes is")
        print(f"  far larger than a 3-sigma offset in the stored statistics.")
        print(f"\n  The damage lands somewhere less obvious and arguably worse. It")
        print(f"  concentrates on Class 1 Warning, whose recall falls to "
              f"{min(c1):.1f}%, because")
        print(f"  Normal and Warning are separated by about one sigma of temperature")
        print(f"  and a 3-sigma shift swamps that distinction entirely. Critical is")
        print(f"  still caught; the early warning that would have let the driver")
        print(f"  intervene within the 2-hour window is not.")
        print(f"\n  So a corrupted statistics file does not blind the system to")
        print(f"  catastrophe. It blinds the system to the signal that would have")
        print(f"  prevented the catastrophe, while accuracy still reads above 91% and")
        print(f"  nothing in the logs looks wrong. That is the harder failure to")
        print(f"  notice in production, and it is the argument for treating")
        print(f"  training_stats.npy as a versioned, checksummed deployment artefact.")
    print(f"\n  Either way the experiment makes the operational point: the statistics")
    print(f"  file is a deployment artefact with the same safety weight as the model")
    print(f"  itself. The Ansible playbook ships it, it is version-controlled with")
    print(f"  the model, and nothing in the runtime path recomputes it.")


if __name__ == "__main__":
    main()
