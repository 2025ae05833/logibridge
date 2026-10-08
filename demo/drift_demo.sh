#!/usr/bin/env bash
# LogiEdge -- Task E1 drift demonstration
#
# Sequence: clean baseline -> inject --anomaly combined -> restore clean.
# Expected: PSI crosses 0.25 during injection and falls below 0.10 on recovery.
#
# ON SIMULATION SPEED
# -------------------
# Inference fires once per 10 s window step, so a 100-inference rolling window
# represents about 17 minutes of truck time. The problem statement asks for PSI
# to cross 0.25 within 5 minutes of the injection, which is a demonstration
# requirement rather than a physical one. We therefore run the simulator at
# SPEED x real time so that a representative number of inferences reaches the
# monitor inside the 5-minute wall-clock window. The pipeline, the rolling
# window and the PSI maths are unchanged; only the clock is compressed.
#
# Usage:  ./drift_demo.sh [SPEED] [PSI_INTERVAL_S]
set -u

SPEED="${1:-10}"
INTERVAL="${2:-15}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${MQTT_HOST:-localhost}"
TRUCK="${TRUCK_ID:-TRK-001}"

echo "=============================================================="
echo " LogiEdge drift demonstration"
echo "   simulator speed : ${SPEED}x"
echo "   PSI interval    : ${INTERVAL}s"
echo "   broker          : ${HOST}:1883"
echo "=============================================================="

# Graceful first, forceful second. Tearing the TFLite interpreter down while
# the MQTT network thread is still running can trip a segfault in the child,
# which bash then reports as "Segmentation fault" at the end of an otherwise
# successful demonstration. Give each process a chance to stop cleanly, then
# make sure it is gone, and exit 0 explicitly so a child's death signal does
# not become the script's exit status.
cleanup() {
  for pid in ${SIM:-} ${DRIFT:-} ${INFER:-}; do
    [ -n "$pid" ] && kill -INT "$pid" 2>/dev/null
  done
  sleep 2
  for pid in ${SIM:-} ${DRIFT:-} ${INFER:-}; do
    [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null
  done
  return 0
}
trap cleanup INT TERM

# --- inference service ---------------------------------------------------
MODEL_PATH="$ROOT/inference/model.tflite" \
STATS_PATH="$ROOT/data_pipeline/training_stats.npy" \
TRUCK_ID="$TRUCK" MQTT_HOST="$HOST" \
ALERT_LOG="$ROOT/inference/data/alerts.jsonl" \
  python3 "$ROOT/inference/inference_service.py" > "$ROOT/demo/inference.log" 2>&1 &
INFER=$!
sleep 4

# --- drift monitor -------------------------------------------------------
python3 "$ROOT/monitoring/drift_monitor.py" \
  --interval "$INTERVAL" --truck-id "$TRUCK" --host "$HOST" --score p_normal &
DRIFT=$!
sleep 2

echo; echo ">>> PHASE 1  clean baseline (expect PSI stable, below 0.10)"
python3 "$ROOT/data_pipeline/simulator.py" --anomaly none --duration $((SPEED * 120)) \
  --speed "$SPEED" --sink mqtt --host "$HOST" --truck-id "$TRUCK" --seed 11 2>/dev/null

echo; echo ">>> PHASE 2  INJECTING --anomaly combined (expect PSI > 0.25)"
python3 "$ROOT/data_pipeline/simulator.py" --anomaly combined --duration $((SPEED * 180)) \
  --speed "$SPEED" --sink mqtt --host "$HOST" --truck-id "$TRUCK" --seed 22 2>/dev/null

echo; echo ">>> PHASE 3  clean data restored (expect PSI back below 0.10)"
python3 "$ROOT/data_pipeline/simulator.py" --anomaly none --duration $((SPEED * 200)) \
  --speed "$SPEED" --sink mqtt --host "$HOST" --truck-id "$TRUCK" --seed 33 2>/dev/null

sleep "$INTERVAL"
echo; echo ">>> demonstration complete -- see monitoring/psi_history.json"

cleanup
exit 0
