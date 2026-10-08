# LogiEdge -- on-truck inference container (Task D2)
#
# LAYER ORDER IS THE POINT OF THIS FILE
# -------------------------------------
# Docker caches each instruction as a layer and invalidates every layer after
# the first one that changes. The problem statement requires all pip install
# layers to appear BEFORE `COPY model.tflite`, and the reason is a bandwidth
# one: FreightBridge reships a model every six weeks to 85 trucks over M2M
# SIMs. If the dependency layer sat after the model, changing the model would
# invalidate it and every truck would pull the whole Python environment again.
#
# With the order below, a model-only change rebuilds and reships exactly one
# small layer. `make ota-demo` in the repo root demonstrates this, and the
# saving is calculated in the report.
#
# Build:
#   docker build -t logibridge-inference:latest .
# Run:
#   docker run --rm --network host \
#     -e MODEL_PATH=/app/model.tflite -e TRUCK_ID=TRK-001 \
#     -v "$PWD/data:/app/data" logibridge-inference:latest

FROM python:3.11-slim

# ---------------------------------------------------------------- layer 1
# Rarely changes. Kept first so it survives every model update.
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------- layer 2
# Dependencies. This is the expensive layer -- it must never be invalidated
# by a model change, which is why it sits above every COPY of the model.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---------------------------------------------------------------- layer 3
# Application code and the frozen normalisation statistics. These change on a
# code release, which is rarer than a model release.
COPY preprocessing.py /app/preprocessing.py
COPY simulator.py /app/simulator.py
COPY inference_service.py /app/inference_service.py
COPY training_stats.npy /app/training_stats.npy

# ---------------------------------------------------------------- layer 4
# The model. LAST, and alone, so that a model-only OTA update rebuilds and
# transfers nothing but this layer.
COPY model.tflite /app/model.tflite

# Switchable at run time -- one image serves M1, M2 or M3 with no rebuild.
ENV MODEL_PATH=/app/model.tflite \
    STATS_PATH=/app/training_stats.npy \
    TRUCK_ID=TRK-001 \
    MQTT_HOST=localhost \
    MQTT_PORT=1883 \
    ALERT_LOG=/app/data/alerts.jsonl

# The alert log must survive container replacement, so /app/data is a volume.
VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,sys; sys.exit(0 if os.path.exists(os.environ['MODEL_PATH']) else 1)"

CMD ["python", "inference_service.py"]
