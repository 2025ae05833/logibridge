#!/usr/bin/env python3
"""
LogiEdge -- M2: Post-Training Quantisation to Full INT8  (Task F1, Variant M2)

Converts the trained M1 FP32 baseline to a full-INT8 TFLite model using
tf.lite.TFLiteConverter with DEFAULT optimisation and a representative dataset.

"Full INT8" here means weights, activations, and the input and output tensors
are all int8 -- not the lighter dynamic-range quantisation, where weights are
int8 but activations stay float. The distinction matters for deployment: a
fully-integer graph is what an integer-only NPU such as the Hailo-8L on the
AI HAT+ can execute without falling back to float kernels on the CPU.

Also writes M1 as an unquantised TFLite file, so the benchmark compares three
models in the same runtime rather than comparing Keras against TFLite.

Outputs
-------
    models/m1_fp32.tflite    FP32 baseline in the TFLite runtime
    models/m2_int8.tflite    full-INT8 PTQ
    models/m2_metrics.json
"""

import json
import os

from common import (  # noqa: E402
    CLASS_NAMES,
    MODELS,
    convert_tflite,
    load_dataset,
    per_class_recall,
    print_report,
    tflite_predict,
)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

MIN_CALIBRATION = 200  # required by the problem statement


def main():
    Xtr, ytr, Xva, yva, _ = load_dataset(split="random")
    model = tf.keras.models.load_model(os.path.join(MODELS, "m1_fp32.keras"))

    # ------------------------------------------- M1 baseline in TFLite form
    m1_path, m1_bytes, _ = convert_tflite(
        model, os.path.join(MODELS, "m1_fp32.tflite"), int8=False
    )
    print(f"M1 FP32  {m1_bytes / 1024:7.2f} KB  -> {os.path.basename(m1_path)}")

    # ----------------------------------------------------- M2 full INT8 PTQ
    if Xtr.shape[0] < MIN_CALIBRATION:
        raise SystemExit(
            f"only {Xtr.shape[0]} training windows available, "
            f"{MIN_CALIBRATION} needed for calibration"
        )
    m2_path, m2_bytes, n_calib = convert_tflite(
        model,
        os.path.join(MODELS, "m2_int8.tflite"),
        X_calib=Xtr,
        int8=True,
        n_calib=Xtr.shape[0],
    )
    print(f"M2 INT8  {m2_bytes / 1024:7.2f} KB  -> {os.path.basename(m2_path)}"
          f"   ({n_calib} calibration samples)")
    print(f"compression: {m1_bytes / m2_bytes:.2f}x "
          f"({100 * (1 - m2_bytes / m1_bytes):.1f}% smaller)")

    # ----------------------------------------------------------- accuracy
    pred_fp32 = tflite_predict(m1_path, Xva).argmax(1)
    pred_int8 = tflite_predict(m2_path, Xva).argmax(1)
    acc1, cm1, rec1 = print_report(yva, pred_fp32, "M1 FP32 (TFLite runtime)")
    acc2, cm2, rec2 = print_report(yva, pred_int8, "M2 INT8 (full post-training)")

    agree = float((pred_fp32 == pred_int8).mean())
    print(f"\nM1/M2 prediction agreement: {100 * agree:.1f}%")
    print(f"accuracy delta from quantisation: "
          f"{100 * (acc2 - acc1):+.2f} percentage points")

    metrics = {
        "m1_fp32": {
            "bytes": m1_bytes, "kb": round(m1_bytes / 1024, 2),
            "accuracy_pct": round(100 * acc1, 2),
            "class2_recall_pct": round(100 * rec1[2], 2),
            "confusion_matrix": cm1.tolist(),
        },
        "m2_int8": {
            "bytes": m2_bytes, "kb": round(m2_bytes / 1024, 2),
            "accuracy_pct": round(100 * acc2, 2),
            "class2_recall_pct": round(100 * rec2[2], 2),
            "confusion_matrix": cm2.tolist(),
            "calibration_samples": n_calib,
        },
        "compression_ratio": round(m1_bytes / m2_bytes, 3),
        "agreement_pct": round(100 * agree, 2),
        "class_names": CLASS_NAMES,
    }
    with open(os.path.join(MODELS, "m2_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nsaved -> models/m2_metrics.json")


if __name__ == "__main__":
    main()
