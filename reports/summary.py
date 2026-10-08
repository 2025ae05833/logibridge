#!/usr/bin/env python3
"""Print every measured number the LogiEdge report needs, from saved artefacts."""
import csv, json, os
 
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(p):
    try:
        with open(os.path.join(R, p)) as f: return json.load(f)
    except Exception: return None
 
print("=" * 72)
print(" LogiEdge - measured results for the report (Group 18)")
print("=" * 72)
 
m1 = load("training/models/m1_metrics.json")
if m1:
    r, b = m1["random_split"], m1["blocked_split"]
    print(f"\nD1 - TRAINING GATE")
    print(f"  architecture      {m1['architecture']}")
    print(f"  parameters        {m1['parameters']}   epochs run {m1['epochs_run']}")
    print(f"  random split      {r['accuracy_pct']:6.2f}%   (gate {m1['gate_pct']}%)  "
          f"-> {'PASSED' if m1['gate_passed'] else 'FAILED'}")
    print(f"  blocked split     {b['accuracy_pct']:6.2f}%   (no window overlap)")
    print(f"  Class 2 recall    {r['class2_recall_pct']:6.1f}% random / "
          f"{b['class2_recall_pct']:.1f}% blocked")
    print(f"  per-class recall  N {r['recall_pct'][0]:.1f}%  W {r['recall_pct'][1]:.1f}%  "
          f"C {r['recall_pct'][2]:.1f}%")
    print(f"  confusion matrix (random split, rows=true N/W/C):")
    for i, row in enumerate(r["confusion_matrix"]):
        print(f"      {['N','W','C'][i]}  {row}")
 
m2 = load("training/models/m2_metrics.json")
if m2:
    print(f"\nF1 - M2 POST-TRAINING QUANTISATION")
    print(f"  M1 FP32 TFLite    {m2['m1_fp32']['kb']:6.2f} KB   {m2['m1_fp32']['accuracy_pct']}%")
    print(f"  M2 full INT8      {m2['m2_int8']['kb']:6.2f} KB   {m2['m2_int8']['accuracy_pct']}%")
    print(f"  compression       {m2['compression_ratio']}x")
    print(f"  calibration       {m2['m2_int8']['calibration_samples']} samples (spec requires >=200)")
    print(f"  M1/M2 agreement   {m2['agreement_pct']}%")
 
m3 = load("training/models/m3_metrics.json")
if m3:
    print(f"\nF1 - M3 STRUCTURED PRUNING + INT8")
    print(f"  target sparsity   {m3['target_sparsity']}")
    print(f"  units             {m3['units_before']} -> {m3['units_after']}  "
          f"({m3['units_removed_pct']}% removed)")
    print(f"  layer shapes      dense_0 32->{m3['layer_shapes']['dense_0']}, "
          f"dense_1 16->{m3['layer_shapes']['dense_1']}")
    print(f"  parameters        {m3['params_before']} -> {m3['params_after']}  "
          f"({m3['params_removed_pct']}% removed)")
    print(f"  size / accuracy   {m3['kb']} KB   {m3['accuracy_pct']}%   "
          f"C2 recall {m3['class2_recall_pct']}%")
 
csvp = os.path.join(R, "optimisation/results/benchmark_results.csv")
if os.path.exists(csvp):
    print(f"\nF2 - FIVE-METRIC BENCHMARK  (the 15-cell table)")
    print(f"  {'':<6}{'mean ms':>10}{'p95 ms':>10}{'size KB':>10}"
          f"{'acc %':>9}{'energy mJ':>12}{'C2 recall':>11}")
    print("  " + "-" * 62)
    rows = list(csv.DictReader(open(csvp)))
    for r_ in rows:
        print(f"  {r_['variant']:<6}{float(r_['mean_latency_ms']):>10.4f}"
              f"{float(r_['p95_latency_ms']):>10.4f}{float(r_['size_kb']):>10.2f}"
              f"{float(r_['accuracy_pct']):>9.2f}{float(r_['energy_mj']):>12.6f}"
              f"{float(r_['class2_recall_pct']):>10.1f}%")
    if rows:
        base = rows[0]
        print(f"\n  relative to M1:")
        for r_ in rows[1:]:
            ratio = float(base['mean_latency_ms']) / float(r_['mean_latency_ms'])
            word = "faster" if ratio >= 1 else "SLOWER"
            print(f"    {r_['variant']}: {(ratio if ratio>=1 else 1/ratio):.2f}x {word}, "
                  f"{float(base['size_kb'])/float(r_['size_kb']):.2f}x smaller, "
                  f"{float(r_['accuracy_pct'])-float(base['accuracy_pct']):+.2f} pp accuracy")
 
ref = load("monitoring/reference_dist.json")
if ref:
    print(f"\nE1 - PSI REFERENCE DISTRIBUTION")
    print(f"  built from {ref['n_windows']} clean Normal windows, score = {ref['score_kind']}")
    for lbl, v in zip(ref["bin_labels"], ref["distribution"]):
        print(f"    {lbl}  {v:6.3f}  {'#' * int(round(40*v))}")
 
nx = os.path.join(R, "reports/normalisation_experiment.txt")
if os.path.exists(nx):
    print(f"\nC2 - NORMALISATION 3-SIGMA EXPERIMENT")
    for line in open(nx):
        s = line.rstrip()
        if any(k in s for k in ("correct", "sigma", "std x3", "condition", "----")):
            print("  " + s)
 
print(f"\n{'=' * 72}")
print(" Full A1/B1/B2/D2/E3 numbers: reports/calculations_output.txt")
print("=" * 72)
