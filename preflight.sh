#!/usr/bin/env bash
G="\033[32m"; R="\033[31m"; N="\033[0m"
echo "===== LogiEdge pre-recording check ====="
python3 - <<'PY'
import hashlib, json
try:
    r = json.load(open("monitoring/reference_dist.json")).get("model_sha")
    s = hashlib.sha256(open("inference/model.tflite","rb").read()).hexdigest()[:12]
    print(f"  \033[32mOK \033[0m model matches reference ({s})" if r==s
          else f"  \033[31mFIX\033[0m model MISMATCH ref={r} served={s} -> make reference")
except Exception as e:
    print(f"  \033[31mFIX\033[0m model/reference unreadable: {e}")
PY
chk(){ if eval "$2"; then echo -e "  ${R}FIX${N} $3"; else echo -e "  ${G}OK ${N} $1"; fi; }
chk "port 1883 free"         "ss -ltn 2>/dev/null | grep -q :1883"  "port 1883 busy -> sudo service mosquitto stop"
chk "/opt/logibridge absent" "[ -d /opt/logibridge ]"               "/opt/logibridge exists -> sudo rm -rf /opt/logibridge"
chk "no stale container"     "docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx logibridge-inference" "stale container -> docker rm -f logibridge-inference"
chk "registry image present" "! docker image inspect localhost:5000/logibridge-inference:latest >/dev/null 2>&1" "registry image missing -> docker push localhost:5000/logibridge-inference:latest"
for f in training/models/m1_metrics.json optimisation/results/benchmark_results.csv \
         optimisation/results/pareto_chart.png scenario_architecture/system_architecture.png \
         reports/calculations_output.txt reports/summary.py monitoring/reference_dist.json; do
  [ -f "$f" ] && echo -e "  ${G}OK ${N} $f" || echo -e "  ${R}FIX${N} missing $f"
done
echo "========================================"
