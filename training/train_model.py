#!/usr/bin/env python3
"""
LogiEdge -- M1 FP32 Baseline Training  (Task D1, Variant M1)

Trains the 2-hidden-layer MLP (32 and 16 units, ReLU) recommended by the
problem statement on the 6-feature cold-chain windows, and reports accuracy
against both validation splits.

GATE: validation accuracy must exceed 88%. The script exits non-zero if the
random-split accuracy falls below that, so a failing run cannot be mistaken
for a passing one.

Outputs
-------
    models/m1_fp32.keras          Keras SavedModel (the M1 variant)
    models/m1_metrics.json        accuracy, confusion matrix, per-class recall
    models/training_curve.png     loss and accuracy per epoch
"""

import argparse
import json
import os

from common import (  # noqa: E402  (sets TF_USE_LEGACY_KERAS first)
    CLASS_NAMES,
    MODELS,
    build_mlp,
    load_dataset,
    per_class_recall,
    print_report,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

GATE = 88.0  # percent, from the problem statement


def plot_curves(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(history["loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_xlabel("epoch"), ax1.set_ylabel("loss"), ax1.legend()
    ax1.set_title("Cross-entropy loss")
    ax2.plot(100 * np.array(history["accuracy"]), label="train")
    ax2.plot(100 * np.array(history["val_accuracy"]), label="val")
    ax2.axhline(GATE, ls="--", c="crimson", lw=1, label=f"{GATE:.0f}% gate")
    ax2.set_xlabel("epoch"), ax2.set_ylabel("accuracy (%)"), ax2.legend()
    ax2.set_title("Classification accuracy")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.suptitle("LogiEdge M1 FP32 baseline training", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train the LogiEdge M1 baseline")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(MODELS, exist_ok=True)

    Xtr, ytr, Xva, yva, blob = load_dataset(split="random")
    print(f"train {Xtr.shape[0]} windows | val {Xva.shape[0]} windows "
          f"| {Xtr.shape[1]} features")

    model = build_mlp(n_features=Xtr.shape[1], seed=args.seed)
    model.summary()

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=args.patience, restore_best_weights=True, verbose=0,
        ),
    ]
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=0,
    )
    print(f"\ntrained {len(hist.history['loss'])} epochs "
          f"(early stopping restored the best weights)")

    # ---------------------------------------------------------- evaluation
    pred_va = model.predict(Xva, verbose=0).argmax(axis=1)
    acc_rand, cm_rand, rec_rand = print_report(
        yva, pred_va, "RANDOM split (as specified by the problem statement)"
    )

    Xtr_b, ytr_b, Xva_b, yva_b, _ = load_dataset(split="blocked")
    pred_b = model.predict(Xva_b, verbose=0).argmax(axis=1)
    acc_blk, cm_blk, rec_blk = print_report(
        yva_b, pred_b, "BLOCKED split (no window overlap -- honest estimate)"
    )

    # ------------------------------------------------------------- outputs
    keras_path = os.path.join(MODELS, "m1_fp32.keras")
    model.save(keras_path)
    plot_curves(hist.history, os.path.join(MODELS, "training_curve.png"))

    metrics = {
        "variant": "M1_FP32",
        "architecture": "Dense(32,relu) -> Dense(16,relu) -> Dense(3,softmax)",
        "parameters": int(model.count_params()),
        "epochs_run": len(hist.history["loss"]),
        "random_split": {
            "accuracy_pct": round(100 * acc_rand, 2),
            "confusion_matrix": cm_rand.tolist(),
            "recall_pct": [round(100 * r, 2) for r in rec_rand],
            "class2_recall_pct": round(100 * rec_rand[2], 2),
            "n_val": int(len(yva)),
        },
        "blocked_split": {
            "accuracy_pct": round(100 * acc_blk, 2),
            "confusion_matrix": cm_blk.tolist(),
            "recall_pct": [round(100 * r, 2) for r in rec_blk],
            "class2_recall_pct": round(100 * rec_blk[2], 2),
            "n_val": int(len(yva_b)),
        },
        "class_names": CLASS_NAMES,
        "gate_pct": GATE,
        "gate_passed": bool(100 * acc_rand > GATE),
    }
    with open(os.path.join(MODELS, "m1_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nparameters: {model.count_params()}")
    print(f"saved -> {keras_path}")

    # ---------------------------------------------------------------- gate
    print("\n" + "=" * 58)
    if 100 * acc_rand > GATE:
        print(f"GATE PASSED  {100 * acc_rand:.2f}% > {GATE:.0f}% "
              f"(blocked split {100 * acc_blk:.2f}%)")
        print(f"Class 2 Critical recall: {100 * rec_rand[2]:.1f}% random, "
              f"{100 * rec_blk[2]:.1f}% blocked")
    else:
        print(f"GATE FAILED  {100 * acc_rand:.2f}% <= {GATE:.0f}%")
        print("Revisit feature extraction or architecture before proceeding.")
    print("=" * 58)

    raise SystemExit(0 if 100 * acc_rand > GATE else 1)


if __name__ == "__main__":
    main()
