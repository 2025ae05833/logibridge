#!/usr/bin/env python3
"""
LogiEdge -- shared helpers for training, conversion and benchmarking.

TF_USE_LEGACY_KERAS is set before TensorFlow is imported anywhere in this
project. The TensorFlow Model Optimization Toolkit (needed for the M3 pruning
variant) is built against the Keras 2 API; TensorFlow 2.16+ ships Keras 3 as
tf.keras by default, and tfmot's pruning wrappers fail against it. Forcing the
legacy Keras keeps one model format across all three variants.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "dataset.npz")
MODELS = os.path.join(HERE, "models")

CLASS_NAMES = ["Normal", "Warning", "Critical"]
N_CLASSES = 3
N_FEATURES = 6


def load_dataset(path=DATA, split="random"):
    """Return (X_train, y_train, X_val, y_val, blob).

    split='random'  -- the stratified 20% split the problem statement specifies
    split='blocked' -- time-blocked tail with a guard band, no window overlap
    """
    blob = np.load(path, allow_pickle=True)
    X, y = blob["X"], blob["y"]
    if split == "random":
        val = blob["val_random"]
        train = ~val
    elif split == "blocked":
        val = blob["val_blocked"]
        train = ~val & ~blob["guard"]
    else:
        raise ValueError(f"unknown split {split!r}")
    return X[train], y[train], X[val], y[val], blob


def build_mlp(n_features=N_FEATURES, units=(32, 16), seed=1337):
    """The architecture recommended by the problem statement: 32 -> 16 -> 3."""
    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.layers.Input(shape=(n_features,), name="features")]
    for i, u in enumerate(units):
        layers.append(tf.keras.layers.Dense(u, activation="relu", name=f"dense_{i}"))
    layers.append(tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="out"))
    model = tf.keras.Sequential(layers, name="logibridge_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def representative_dataset_fn(X, n=256):
    """Yield calibration samples for full-INT8 post-training quantisation.

    The converter runs the model on these to observe the real dynamic range of
    every activation tensor, which is what fixes the INT8 scale and zero-point
    for each. The problem statement requires at least 200 samples; we use the
    whole training split, which is larger than that, and shuffle so all three
    classes contribute. Calibrating on Normal windows alone would clip the
    activation ranges that Critical inputs actually produce.
    """
    idx = np.arange(X.shape[0])
    np.random.default_rng(0).shuffle(idx)
    idx = idx[: min(n, X.shape[0])]

    def gen():
        for i in idx:
            yield [X[i : i + 1].astype(np.float32)]

    return gen, len(idx)


def convert_tflite(model, out_path, X_calib=None, int8=False, n_calib=256):
    """Convert a Keras model to TFLite, optionally as full INT8.

    Full INT8 means weights AND activations are quantised, and the input and
    output tensors are int8 too. That is what lets an integer-only accelerator
    run the graph without falling back to float kernels.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    n_used = 0
    if int8:
        if X_calib is None:
            raise ValueError("full INT8 conversion needs a calibration set")
        gen, n_used = representative_dataset_fn(X_calib, n_calib)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    blob = converter.convert()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(blob)
    return out_path, len(blob), n_used


def tflite_predict(model_path, X):
    """Run a TFLite model over X, handling int8 input/output quantisation."""
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    preds = np.empty((X.shape[0], N_CLASSES), dtype=np.float32)
    for i in range(X.shape[0]):
        x = X[i : i + 1].astype(np.float32)
        if inp["dtype"] == np.int8:
            scale, zero = inp["quantization"]
            x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0]
        if out["dtype"] == np.int8:
            scale, zero = out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        preds[i] = y
    return preds


def confusion_matrix(y_true, y_pred, n=N_CLASSES):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_recall(cm):
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diag(cm) / cm.sum(axis=1)
    return np.nan_to_num(rec)


def print_report(y_true, y_pred, title=""):
    """Print accuracy, confusion matrix and per-class recall/precision."""
    cm = confusion_matrix(y_true, y_pred)
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    rec = per_class_recall(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.nan_to_num(np.diag(cm) / cm.sum(axis=0))

    if title:
        print(f"\n{title}")
    print(f"  accuracy: {100 * acc:.2f}%   (n={len(y_true)})")
    print(f"  {'':<10}{'pred N':>8}{'pred W':>8}{'pred C':>8}{'recall':>9}"
          f"{'prec':>8}")
    for i in range(N_CLASSES):
        print(f"  {CLASS_NAMES[i]:<10}{cm[i, 0]:>8}{cm[i, 1]:>8}{cm[i, 2]:>8}"
              f"{100 * rec[i]:>8.1f}%{100 * prec[i]:>7.1f}%")
    return acc, cm, rec
