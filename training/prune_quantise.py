#!/usr/bin/env python3
"""
LogiEdge -- M3: 35% Structured Pruning + Full INT8 PTQ  (Task F1, Variant M3)

WHY STRUCTURED, AND WHY NOT prune_low_magnitude ALONE
-----------------------------------------------------
The Model Optimization Toolkit's prune_low_magnitude is unstructured: it zeros
individual weights but leaves the tensor shapes untouched. On a dense MLP that
produces a model with the same number of multiply-accumulates, the same tensor
dimensions, and -- because TFLite stores dense tensors -- very nearly the same
file size. Zeros are not speed. For an edge deployment whose whole point is
lower latency and a smaller OTA payload, unstructured sparsity buys nothing
without a sparse kernel, and neither the Pi 5's NEON path nor the Hailo-8L
provides one for this topology.

Structured pruning removes whole hidden units, so the weight matrices actually
shrink, the MAC count actually falls, and the TFLite file actually gets
smaller. That is what this script does.

The problem statement asks for a PolynomialDecay schedule, so the official
tfmot.sparsity.keras.PolynomialDecay object drives the ramp: the fraction of
units masked grows along that schedule from 0 to 0.35 during fine-tuning,
rather than 35% of the network being amputated in one step. Gradual removal
lets the surviving units absorb the pruned units' role between steps, which is
the entire reason the schedule exists.

    hidden layer 1:  32 units -> 21 kept  (34.4% removed)
    hidden layer 2:  16 units -> 10 kept  (37.5% removed)
    overall:         48 units -> 31 kept  (35.4% removed)

Pipeline: rank units -> ramp the mask along PolynomialDecay while fine-tuning
-> physically compact the surviving units into a smaller model -> full INT8 PTQ.

Outputs
-------
    models/m3_pruned_fp32.keras
    models/m3_pruned_int8.tflite
    models/m3_metrics.json
"""

import json
import os

from common import (  # noqa: E402
    CLASS_NAMES,
    MODELS,
    N_CLASSES,
    convert_tflite,
    load_dataset,
    print_report,
    tflite_predict,
)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow_model_optimization as tfmot  # noqa: E402

TARGET_SPARSITY = 0.35
FINETUNE_EPOCHS = 60
BATCH_SIZE = 16
HIDDEN_LAYERS = ["dense_0", "dense_1"]


# --------------------------------------------------------------------------
# Unit importance
# --------------------------------------------------------------------------
def unit_importance(model, layer_name, next_layer_name):
    """Rank hidden units by the L2 norm of their incoming and outgoing weights.

    A unit matters in proportion to both how strongly it responds to its input
    and how strongly the next layer listens to it. Scoring on the incoming
    weights alone would keep units whose output is ignored downstream.
    """
    w_in = model.get_layer(layer_name).get_weights()[0]  # [in, units]
    w_out = model.get_layer(next_layer_name).get_weights()[0]  # [units, out]
    return np.linalg.norm(w_in, axis=0) * np.linalg.norm(w_out, axis=1)


def units_to_keep(scores, sparsity):
    """Indices of the units surviving at this sparsity level, sorted."""
    n = scores.size
    n_prune = int(round(sparsity * n))
    if n_prune <= 0:
        return np.arange(n)
    keep = np.argsort(scores)[n_prune:]
    return np.sort(keep)


# --------------------------------------------------------------------------
# Gradual structured masking driven by PolynomialDecay
# --------------------------------------------------------------------------
class StructuredPruningCallback(tf.keras.callbacks.Callback):
    """Masks whole hidden units, ramping sparsity along a PolynomialDecay."""

    def __init__(self, schedule, layer_pairs):
        super().__init__()
        self.schedule = schedule
        self.layer_pairs = layer_pairs
        self.step = 0
        self.trace = []

    def _apply_mask(self, sparsity):
        for name, nxt in self.layer_pairs:
            scores = unit_importance(self.model, name, nxt)
            keep = set(units_to_keep(scores, sparsity).tolist())
            dead = [j for j in range(scores.size) if j not in keep]
            if not dead:
                continue
            # Zero the unit's incoming weights and bias, and the next layer's
            # corresponding input row, so the unit contributes nothing.
            layer = self.model.get_layer(name)
            w, b = layer.get_weights()
            w[:, dead] = 0.0
            b[dead] = 0.0
            layer.set_weights([w, b])

            nxt_layer = self.model.get_layer(nxt)
            wn, bn = nxt_layer.get_weights()
            wn[dead, :] = 0.0
            nxt_layer.set_weights([wn, bn])

    def on_train_batch_end(self, batch, logs=None):
        should, sparsity = self.schedule(tf.constant(self.step, tf.int32))
        sparsity = float(sparsity.numpy() if hasattr(sparsity, "numpy") else sparsity)
        if bool(should.numpy() if hasattr(should, "numpy") else should):
            self._apply_mask(sparsity)
        self.step += 1

    def on_epoch_end(self, epoch, logs=None):
        _, sparsity = self.schedule(tf.constant(self.step, tf.int32))
        s = float(sparsity.numpy() if hasattr(sparsity, "numpy") else sparsity)
        self.trace.append({"epoch": epoch, "sparsity": round(s, 4),
                           "val_accuracy": round(float(logs.get("val_accuracy", 0)), 4)})


# --------------------------------------------------------------------------
# Physical compaction
# --------------------------------------------------------------------------
def compact(model, keep0, keep1, n_features):
    """Rebuild the network with only the surviving units.

    This is the step that turns masked zeros into an actually smaller model.
    """
    w0, b0 = model.get_layer("dense_0").get_weights()
    w1, b1 = model.get_layer("dense_1").get_weights()
    w2, b2 = model.get_layer("out").get_weights()

    w0n, b0n = w0[:, keep0], b0[keep0]
    w1n, b1n = w1[np.ix_(keep0, keep1)], b1[keep1]
    w2n, b2n = w2[keep1, :], b2

    small = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,), name="features"),
            tf.keras.layers.Dense(len(keep0), activation="relu", name="dense_0"),
            tf.keras.layers.Dense(len(keep1), activation="relu", name="dense_1"),
            tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="out"),
        ],
        name="logibridge_mlp_pruned",
    )
    small.get_layer("dense_0").set_weights([w0n, b0n])
    small.get_layer("dense_1").set_weights([w1n, b1n])
    small.get_layer("out").set_weights([w2n, b2n])
    small.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return small


# --------------------------------------------------------------------------
def main():
    Xtr, ytr, Xva, yva, _ = load_dataset(split="random")
    model = tf.keras.models.load_model(os.path.join(MODELS, "m1_fp32.keras"))
    params_before = model.count_params()

    base = model.evaluate(Xva, yva, verbose=0)[1]
    print(f"M1 baseline: {params_before} params, {100 * base:.2f}% val accuracy")

    # ------------------------------------------------ PolynomialDecay ramp
    steps_per_epoch = int(np.ceil(Xtr.shape[0] / BATCH_SIZE))
    end_step = steps_per_epoch * int(FINETUNE_EPOCHS * 0.6)  # reach target at 60%
    schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=TARGET_SPARSITY,
        begin_step=0,
        end_step=end_step,
        power=3.0,
        frequency=steps_per_epoch // 2 or 1,
    )
    print(f"PolynomialDecay 0.0 -> {TARGET_SPARSITY} over {end_step} steps "
          f"({steps_per_epoch} steps/epoch), then {FINETUNE_EPOCHS - int(FINETUNE_EPOCHS * 0.6)} "
          f"epochs at target")

    pairs = [("dense_0", "dense_1"), ("dense_1", "out")]
    cb = StructuredPruningCallback(schedule, pairs)

    model.optimizer.learning_rate.assign(5e-4)  # gentler, we are fine-tuning
    model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cb],
        verbose=0,
    )
    cb._apply_mask(TARGET_SPARSITY)  # ensure the final mask is exactly at target
    masked_acc = model.evaluate(Xva, yva, verbose=0)[1]
    print(f"after masked fine-tuning: {100 * masked_acc:.2f}% val accuracy")

    # ------------------------------------------------------- compact & save
    keep0 = units_to_keep(unit_importance(model, "dense_0", "dense_1"), TARGET_SPARSITY)
    keep1 = units_to_keep(unit_importance(model, "dense_1", "out"), TARGET_SPARSITY)
    small = compact(model, keep0, keep1, Xtr.shape[1])
    params_after = small.count_params()

    print(f"\nstructured pruning result")
    print(f"  dense_0: 32 -> {len(keep0)} units  "
          f"({100 * (1 - len(keep0) / 32):.1f}% removed)")
    print(f"  dense_1: 16 -> {len(keep1)} units  "
          f"({100 * (1 - len(keep1) / 16):.1f}% removed)")
    print(f"  units:   48 -> {len(keep0) + len(keep1)} "
          f"({100 * (1 - (len(keep0) + len(keep1)) / 48):.1f}% removed)")
    print(f"  params:  {params_before} -> {params_after} "
          f"({100 * (1 - params_after / params_before):.1f}% removed)")

    compact_acc = small.evaluate(Xva, yva, verbose=0)[1]
    print(f"  compacted model val accuracy: {100 * compact_acc:.2f}% "
          f"(masked was {100 * masked_acc:.2f}% -- these must agree)")

    keras_path = os.path.join(MODELS, "m3_pruned_fp32.keras")
    small.save(keras_path)

    # --------------------------------------------------------- INT8 PTQ
    m3_path, m3_bytes, n_calib = convert_tflite(
        small,
        os.path.join(MODELS, "m3_pruned_int8.tflite"),
        X_calib=Xtr,
        int8=True,
        n_calib=Xtr.shape[0],
    )
    print(f"\nM3 pruned+INT8: {m3_bytes / 1024:.2f} KB "
          f"({n_calib} calibration samples)")

    pred = tflite_predict(m3_path, Xva).argmax(1)
    acc3, cm3, rec3 = print_report(yva, pred, "M3 structured-pruned + INT8")

    metrics = {
        "variant": "M3_PRUNED_INT8",
        "target_sparsity": TARGET_SPARSITY,
        "units_before": 48,
        "units_after": int(len(keep0) + len(keep1)),
        "units_removed_pct": round(100 * (1 - (len(keep0) + len(keep1)) / 48), 2),
        "layer_shapes": {"dense_0": int(len(keep0)), "dense_1": int(len(keep1))},
        "params_before": int(params_before),
        "params_after": int(params_after),
        "params_removed_pct": round(100 * (1 - params_after / params_before), 2),
        "bytes": m3_bytes,
        "kb": round(m3_bytes / 1024, 2),
        "accuracy_pct": round(100 * acc3, 2),
        "class2_recall_pct": round(100 * rec3[2], 2),
        "confusion_matrix": cm3.tolist(),
        "calibration_samples": n_calib,
        "sparsity_trace": cb.trace[-10:],
        "class_names": CLASS_NAMES,
    }
    with open(os.path.join(MODELS, "m3_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nsaved -> {keras_path}, {os.path.basename(m3_path)}, m3_metrics.json")


if __name__ == "__main__":
    main()
