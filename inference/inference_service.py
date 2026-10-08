#!/usr/bin/env python3
"""
LogiEdge -- On-Truck Inference Service  (Task D2)

Subscribes to the truck's local sensor topics, runs the mandated preprocessing
pipeline, classifies each 30-second window, publishes the result, and -- the
part that makes the architecture edge-native rather than cloud-dependent --
writes every alert to a local append-only log before any attempt to reach the
network.

    sensors --> MQTT (local broker) --> preprocess --> TFLite --> publish
                                                            \\
                                                             --> local alert log

OFFLINE-FIRST BEHAVIOUR
-----------------------
The problem statement forbids a cloud-dependent architecture: the Nashik-
Aurangabad route loses cellular signal for 35-90 minutes at seven locations.
Everything in this service runs against a broker on localhost, so classification
continues unaffected when the cellular uplink is down. Alerts are appended to
a local JSONL log with a `synced` flag; a separate uplink process marks them
synced once the operations centre acknowledges them. Losing the uplink costs
visibility at the operations centre, never detection on the truck.

ENVIRONMENT
-----------
    MODEL_PATH      path to the .tflite model        (default /app/model.tflite)
    STATS_PATH      path to training_stats.npy       (default /app/training_stats.npy)
    TRUCK_ID        truck identifier                 (default TRK-001)
    MQTT_HOST       broker host                      (default localhost)
    MQTT_PORT       broker port                      (default 1883)
    ALERT_LOG       local alert log path             (default /app/data/alerts.jsonl)

MODEL_PATH is what lets one container image serve every model variant: the
operations centre switches M1/M2/M3 by changing an environment variable and
restarting, with no rebuild and no new image to ship over the cellular link.
"""

import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (  # noqa: E402
    FEATURE_NAMES,
    SlidingWindowExtractor,
    load_stats,
    normalise,
)

try:
    import tflite_runtime.interpreter as tflite  # slim runtime, preferred
except ImportError:  # pragma: no cover - full TF is the fallback
    import tensorflow.lite as tflite

import paho.mqtt.client as mqtt  # noqa: E402

CLASS_NAMES = ["Normal", "Warning", "Critical"]
TOPIC_ROOT = "logibridge/trucks"

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model.tflite")
STATS_PATH = os.environ.get("STATS_PATH", "/app/training_stats.npy")
TRUCK_ID = os.environ.get("TRUCK_ID", "TRK-001")
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
ALERT_LOG = os.environ.get("ALERT_LOG", "/app/data/alerts.jsonl")

QOS_TELEMETRY = 0
QOS_EVENT = 1
QOS_ALERT = 2  # safety-critical: exactly-once delivery


class InferenceService:
    def __init__(self):
        self.stats = load_stats(STATS_PATH)
        self.interp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=1)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

        self.extractor = SlidingWindowExtractor()
        self.n_inferences = 0
        self.running = True
        os.makedirs(os.path.dirname(os.path.abspath(ALERT_LOG)), exist_ok=True)

        # Published with every inference so the drift monitor can verify it is
        # comparing against a reference built on this exact artefact.
        self.model_sha = hashlib.sha256(
            open(MODEL_PATH, "rb").read()
        ).hexdigest()[:12]

        quant = "INT8" if self.inp["dtype"] == np.int8 else "FP32"
        print(f"[logibridge] model   : {MODEL_PATH} ({quant} input, "
              f"sha {self.model_sha})", flush=True)
        print(f"[logibridge] truck   : {TRUCK_ID}", flush=True)
        print(f"[logibridge] broker  : {MQTT_HOST}:{MQTT_PORT}", flush=True)
        print(f"[logibridge] alerts  : {ALERT_LOG}", flush=True)

        try:
            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2, client_id=f"logibridge-inf-{TRUCK_ID}"
            )
        except AttributeError:
            self.client = mqtt.Client(client_id=f"logibridge-inf-{TRUCK_ID}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    # ------------------------------------------------------------- MQTT
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        for sub in ("sensors/temperature", "sensors/vibration", "events/door"):
            topic = f"{TOPIC_ROOT}/{TRUCK_ID}/{sub}"
            client.subscribe(topic, qos=QOS_TELEMETRY)
        print(f"[logibridge] subscribed to {TOPIC_ROOT}/{TRUCK_ID}/#", flush=True)

    def _on_message(self, client, userdata, msg):
        try:
            rec = json.loads(msg.payload.decode())
        except (ValueError, UnicodeDecodeError):
            return
        stream, sim_t = rec.get("stream"), rec.get("sim_t")
        if stream is None or sim_t is None:
            return

        self.extractor.add(stream, sim_t, rec["value"])
        while self.extractor.ready(sim_t):
            self._classify(self.extractor.next_emit)

    # -------------------------------------------------------- inference
    def _predict(self, x):
        if self.inp["dtype"] == np.int8:
            scale, zero = self.inp["quantization"]
            xq = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        else:
            xq = x.astype(np.float32)
        self.interp.set_tensor(self.inp["index"], xq)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])[0]
        if self.out["dtype"] == np.int8:
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y

    def _classify(self, window_end):
        feats, _ = self.extractor.emit(window_end)
        x = normalise(feats.reshape(1, -1), self.stats)

        t0 = time.perf_counter()
        probs = self._predict(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        cls = int(np.argmax(probs))
        confidence = float(probs[cls])
        self.n_inferences += 1

        payload = {
            "truck_id": TRUCK_ID,
            "ts": datetime.now(timezone.utc).isoformat(),
            "window_end_s": round(float(window_end), 2),
            "class": cls,
            "label": CLASS_NAMES[cls],
            "confidence": round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in probs],
            "p_normal": round(float(probs[0]), 4),
            "features": {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, feats)},
            "latency_ms": round(latency_ms, 4),
            "model": os.path.basename(MODEL_PATH),
            "model_sha": self.model_sha,
            "seq": self.n_inferences,
        }

        # Topic mandated by the problem statement.
        self.client.publish(
            f"{TOPIC_ROOT}/{TRUCK_ID}/inference", json.dumps(payload), qos=QOS_EVENT
        )

        if cls > 0:
            self._raise_alert(payload)

        print(
            f"[{self.n_inferences:>4}] t={window_end:7.1f}s  "
            f"{CLASS_NAMES[cls]:<8} conf={confidence:.3f}  "
            f"T={feats[0]:5.2f}C roc={feats[2]:+6.2f}C/min vib={feats[3]:.3f}g  "
            f"{latency_ms:.3f}ms",
            flush=True,
        )

    # ----------------------------------------------------------- alerts
    def _raise_alert(self, payload):
        """Write locally FIRST, then attempt the uplink.

        Write-before-publish is the whole offline-first guarantee. If the
        cellular link is down the publish silently queues or fails, but the
        alert is already durable on the truck and will sync when coverage
        returns.
        """
        alert = {
            "truck_id": payload["truck_id"],
            "ts": payload["ts"],
            "severity": payload["label"],
            "class": payload["class"],
            "confidence": payload["confidence"],
            "features": payload["features"],
            "synced": False,
        }
        with open(ALERT_LOG, "a") as fh:
            fh.write(json.dumps(alert) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        self.client.publish(
            f"{TOPIC_ROOT}/{TRUCK_ID}/alerts", json.dumps(alert), qos=QOS_ALERT
        )

    # ------------------------------------------------------------- run
    def stop(self, *_):
        self.running = False

    def run(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

        delay = 1.0
        while self.running:
            try:
                self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
                break
            except OSError as exc:
                print(f"[logibridge] broker unreachable ({exc}), retry in {delay:.0f}s",
                      flush=True)
                time.sleep(delay)
                delay = min(30.0, delay * 2)

        self.client.loop_start()
        try:
            while self.running:
                time.sleep(0.2)
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            print(f"\n[logibridge] stopped after {self.n_inferences} inferences",
                  flush=True)


if __name__ == "__main__":
    InferenceService().run()
