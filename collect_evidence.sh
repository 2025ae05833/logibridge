#!/usr/bin/env bash
set -u
OUT=reports/evidence
mkdir -p "$OUT/figures"
echo "collecting into $OUT/ ..."
 
python reports/summary.py > "$OUT/01_measured_results.txt" 2>/dev/null \
&& echo "  01_measured_results.txt      gate, variants, benchmark"
cp reports/calculations_output.txt "$OUT/02_calculations.txt" 2>/dev/null \
&& echo "  02_calculations.txt          A1/B1/B2/D2/E3 numbers"
cp reports/normalisation_experiment.txt "$OUT/03_normalisation_3sigma.txt" 2>/dev/null \
&& echo "  03_normalisation_3sigma.txt  the mandated C2 experiment"
 
python3 - <<'PY' > "$OUT/04_psi_trace.txt" 2>/dev/null
import json
h = json.load(open("monitoring/psi_history.json"))
r = json.load(open("monitoring/reference_dist.json"))
print("PSI drift trace (Task E1)")
print(f"score monitored : {h['score_kind']}")
print(f"rolling window  : {h['rolling_window']} inferences")
print(f"alert / recover : {h['alert_threshold']} / {h['recovery_threshold']}")
print(f"reference       : {r['n_windows']} clean windows, bins {[round(v,3) for v in r['distribution']]}\n")
print(f"{'time':<10}{'PSI':>9}{'n':>5}   bins")
for s in h["samples"]:
    print(f"{s['t']:<10}{s['psi']:>9.3f}{s['n']:>5}   {s['distribution']}")
v = [s["psi"] for s in h["samples"]]
print(f"\nmin {min(v):.3f}   max {max(v):.3f}   samples {len(v)}")
PY
[ -s "$OUT/04_psi_trace.txt" ] && echo "  04_psi_trace.txt             PSI before/during/after" \
  || { rm -f "$OUT/04_psi_trace.txt"; echo "  !! psi_history.json missing -> re-run 'make drift'"; }
 
for f in training/models/m1_metrics.json training/models/m2_metrics.json \
         training/models/m3_metrics.json optimisation/results/benchmark_results.csv \
         optimisation/results/benchmark_results.json monitoring/reference_dist.json \
         data_pipeline/mqtt_architecture.md; do
  [ -f "$f" ] && cp "$f" "$OUT/"
done
echo "  (raw .json/.csv metrics copied)"
 
for f in scenario_architecture/system_architecture.png \
         optimisation/results/pareto_chart.png training/models/training_curve.png; do
  [ -f "$f" ] && cp "$f" "$OUT/figures/"
done
echo "  figures/                     architecture, pareto, training curve"
 
cat > "$OUT/00_STILL_NEEDED.txt" <<'EOF'
Two results print to the terminal and are never written to a file.
Capture them before writing Sections 3 and 5:
 
  sudo rm -rf /opt/logibridge && docker rm -f logibridge-inference 2>/dev/null
  make deploy   2>&1 | tee reports/evidence/05_ansible_idempotency.txt
  make ota-demo 2>&1 | tee reports/evidence/06_ota_layer_cache.txt
  make reference
 
From 05: the two PLAY RECAP lines (changed=N, then changed=0)
From 06: the CACHED lines, the COPY model.tflite line that is NOT cached,
         docker history sizes, and the 85-truck bandwidth arithmetic
EOF
echo "  00_STILL_NEEDED.txt          two results to capture by hand"
echo; echo "done -> $OUT/"
