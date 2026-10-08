#!/usr/bin/env bash
#
# LogiEdge — end-to-end demonstration driver
# BITS Pilani WILP · AIML ZG535 · Group 18
#
# One script with every fix found during development baked in, so a recording
# run cannot hit the problems we hit:
#
#   * the background mosquitto service holding port 1883 so the foreground
#     broker cannot bind and shows "Address already in use" on camera
#   * the PSI reference being built on a different model than the container
#     serves, which makes clean data alert at PSI ~5 instead of ~0.02
#   * /opt/logibridge and the container surviving a previous run, so Ansible
#     run 1 reports changed=1 instead of changed=4
#   * BuildKit holding a complete cached chain for the M3 image, so the OTA
#     demonstration shows every layer CACHED and proves nothing
#   * `make ota-demo` leaving M3 deployed, which then breaks the PSI reference
#
# Usage — run the stages in this order while recording:
#
#   ./demo_run.sh reset      prepare clean state, verify everything   (do first)
#   ./demo_run.sh sensors    live MQTT pipeline + 3-sigma experiment
#   ./demo_run.sh training   the 88% gate and measured results
#   ./demo_run.sh drift      PSI crosses 0.25 then recovers
#   ./demo_run.sh deploy     Ansible changed=N then changed=0
#   ./demo_run.sh ota        Docker layer cache + bandwidth saving
#   ./demo_run.sh check      verify every artefact exists and is valid
#
#   ./demo_run.sh all        every stage in order, pausing between each
#
# ORDER MATTERS: drift must run BEFORE ota. The OTA demonstration ends with M3
# in inference/model.tflite, and the PSI reference is built on M2.

set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1

G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; B=$'\033[1m'; N=$'\033[0m'
ok(){   printf "  ${G}OK ${N} %s\n" "$1"; }
bad(){  printf "  ${R}FIX${N} %s\n" "$1"; }
info(){ printf "  ${Y}--${N}  %s\n" "$1"; }
banner(){ printf "\n${B}%s${N}\n%s\n" "$1" "$(printf '=%.0s' {1..70})"; }
say(){  printf "\n${Y}>>> SAY:${N} %s\n\n" "$1"; }
pause(){ printf "\n${B}--- press Enter for the next stage ---${N}"; read -r _; }

need_venv() {
  if [ -z "${VIRTUAL_ENV:-}" ]; then
    bad "virtualenv not active -> run: source .venv/bin/activate"
    exit 1
  fi
}

# ===========================================================================
stage_reset() {
  banner "STAGE 0 — RESET AND VERIFY  (run before recording)"
  need_venv

  info "stopping any broker holding port 1883"
  sudo service mosquitto stop >/dev/null 2>&1
  sudo pkill -x mosquitto >/dev/null 2>&1
  sleep 1

  info "pinning M2 as the served model and rebuilding its PSI reference"
  make reference 2>&1 | grep -E "pinning|reference built|saved ->|windows" || true

  info "clearing previous deployment state"
  sudo rm -rf /opt/logibridge
  docker rm -f logibridge-inference >/dev/null 2>&1

  info "ensuring the registry has the current image"
  if ! docker image inspect localhost:5000/logibridge-inference:latest >/dev/null 2>&1; then
    docker tag logibridge-inference:latest localhost:5000/logibridge-inference:latest 2>/dev/null
    docker push localhost:5000/logibridge-inference:latest >/dev/null 2>&1
  fi

  stage_check
}

# ===========================================================================
stage_check() {
  banner "PRE-FLIGHT CHECK"

  python3 - <<'PY'
import hashlib, json
try:
    r = json.load(open("monitoring/reference_dist.json")).get("model_sha")
    s = hashlib.sha256(open("inference/model.tflite","rb").read()).hexdigest()[:12]
    print(f"  \033[32mOK \033[0m served model matches PSI reference ({s})" if r == s
          else f"  \033[31mFIX\033[0m MISMATCH ref={r} served={s} -> ./demo_run.sh reset")
except Exception as e:
    print(f"  \033[31mFIX\033[0m model/reference unreadable: {e}")
PY

  ss -ltn 2>/dev/null | grep -q ':1883' \
    && bad "port 1883 busy -> sudo pkill -x mosquitto" \
    || ok "port 1883 free (foreground broker can bind)"

  [ -d /opt/logibridge ] \
    && bad "/opt/logibridge exists -> sudo rm -rf /opt/logibridge" \
    || ok "/opt/logibridge absent (deploy run 1 will show changed=4)"

  docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx logibridge-inference \
    && bad "stale container -> docker rm -f logibridge-inference" \
    || ok "no stale container"

  docker image inspect localhost:5000/logibridge-inference:latest >/dev/null 2>&1 \
    && ok "registry image present (deploy task 5 will pull)" \
    || bad "registry image missing -> docker push localhost:5000/logibridge-inference:latest"

  for f in training/models/m1_metrics.json optimisation/results/benchmark_results.csv \
           optimisation/results/pareto_chart.png scenario_architecture/system_architecture.png \
           reports/calculations_output.txt reports/summary.py monitoring/reference_dist.json; do
    [ -f "$f" ] && ok "$f" || bad "missing $f"
  done

  local p1 p2 p3
  p1=$(grep -o 'SPEED \* [0-9]*' demo/drift_demo.sh | sed -n 1p | grep -o '[0-9]*$')
  p2=$(grep -o 'SPEED \* [0-9]*' demo/drift_demo.sh | sed -n 2p | grep -o '[0-9]*$')
  p3=$(grep -o 'SPEED \* [0-9]*' demo/drift_demo.sh | sed -n 3p | grep -o '[0-9]*$')
  info "drift demo runs ${p1}/${p2}/${p3} s = $((p1 + p2 + p3))s wall clock"

  grep -q 'image: "{{ image }}"' <(sed -n '/Stop the running inference/,/when:/p' deployment/logibridge_deploy.yml) \
    && ok "playbook stop task has an image (task 4 will not fail)" \
    || bad "playbook stop task missing image -> task 4 will fail"

  grep -q 'become: true' deployment/logibridge_deploy.yml \
    && ok "playbook has become: true (can write /opt)" \
    || bad "playbook missing become: true"
}

# ===========================================================================
stage_sensors() {
  banner "STAGE 1 — LIVE SENSOR PIPELINE  (Task C1/C2/C3)"
  need_venv
  info "start these in two OTHER terminals first:"
  printf "      T1:  mosquitto -v\n"
  printf "      T2:  mosquitto_sub -t 'logibridge/#' -v\n"
  pause

  say "Three streams publishing to the truck's own broker. Watch vibration climb \
gradually from 0.45 g toward 1.2 g - that is the fault ramping in over 90 seconds, \
not a step change."
  python data_pipeline/simulator.py --anomaly combined --duration 120 --speed 5

  say "Now the mandated 3-sigma normalisation experiment."
  make normexp
  say "Critical recall holds at 100 percent. Warning collapses to 70.6 percent while \
accuracy still reads 91 percent. Corrupted statistics do not blind the system to \
catastrophe - they blind it to the early warning that would have prevented it."
}

# ===========================================================================
stage_training() {
  banner "STAGE 2 — TRAINING GATE  (Task D1)"
  need_venv
  python reports/summary.py
  say "94.74 percent against an 88 percent gate, and Class 2 Critical recall is 100 \
percent. The three errors all sit between 4.3 and 4.6 degrees with temperature rising \
- two are door openings, one is an early-onset fault window. No Critical misses."
}

# ===========================================================================
stage_drift() {
  banner "STAGE 3 — PSI DRIFT MONITORING  (Task E1)"
  need_venv
  if ! ss -ltn 2>/dev/null | grep -q ':1883'; then
    bad "no broker on 1883 - start 'mosquitto -v' in T1 first"
    return 1
  fi
  say "Clean baseline first, then we inject a combined fault, then restore clean data."
  make drift
  echo
  python3 - <<'PY' 2>/dev/null || true
import json
h = json.load(open("monitoring/psi_history.json"))
v = [s["psi"] for s in h["samples"]]
print(f"  summary: min {min(v):.3f}   peak {max(v):.3f}   {len(v)} evaluations")
PY
  say "PSI stayed near 0.02 on clean data, crossed 0.25 within one evaluation cycle \
after injection, peaked at 13.9 with the distribution fully saturated, then recovered \
below 0.10. Recovery is slower than detection because the 100-sample buffer has to flush."
}

# ===========================================================================
stage_deploy() {
  banner "STAGE 4 — ANSIBLE IDEMPOTENCY  (Task E2)"
  info "clearing state so run 1 shows a large changed count"
  sudo rm -rf /opt/logibridge
  docker rm -f logibridge-inference >/dev/null 2>&1

  make deploy 2>&1 | tee reports/evidence/05_ansible_idempotency.txt
  echo
  info "recap lines captured:"
  grep -E "changed=[0-9]+" reports/evidence/05_ansible_idempotency.txt | sed 's/^/      /'
  say "Seven tasks. Run one changes things, run two reports changed=0 and skips the \
stop task, because the two copy tasks compare checksums and nothing arrived."
}

# ===========================================================================
stage_ota() {
  banner "STAGE 5 — DOCKER OTA LAYER CACHE  (Task D2)"
  info "clearing the BuildKit cache so the demonstration is genuine"
  info "(without this, BuildKit reuses an old M3 chain and every layer reads CACHED)"
  docker rmi logibridge-inference:m2 logibridge-inference:m3 >/dev/null 2>&1
  docker builder prune -af >/dev/null 2>&1

  make ota-demo 2>&1 | tee reports/evidence/06_ota_layer_cache.txt
  echo
  banner "VERIFYING THE DEMONSTRATION"
  local step2
  step2=$(sed -n '/STEP 2/,/STEP 3/p' reports/evidence/06_ota_layer_cache.txt)
  echo "$step2" | grep -A1 "RUN pip install" | grep -q CACHED \
    && ok "pip install layer CACHED (89.6 MB survived the model swap)" \
    || bad "pip layer NOT cached - the demo did not work"
  echo "$step2" | grep -A1 "COPY model.tflite" | grep -qE "DONE|[0-9]+\.[0-9]+s" \
    && ok "model layer REBUILT (this is the point of the demonstration)" \
    || bad "model layer still CACHED - rerun this stage"

  say "Seven layers cached, one rebuilt. The 89.6 megabyte dependency layer never \
moves. Across 85 trucks that is 1,598 rupees versus 3 paise - a 47,000 fold reduction, \
purely from putting pip install above COPY model.tflite in the Dockerfile."

  info "restoring M2 so the PSI reference stays valid"
  make reference >/dev/null 2>&1 && ok "M2 restored"
}

# ===========================================================================
case "${1:-}" in
  reset)     stage_reset ;;
  check)     stage_check ;;
  sensors)   stage_sensors ;;
  training)  stage_training ;;
  drift)     stage_drift ;;
  deploy)    stage_deploy ;;
  ota)       stage_ota ;;
  all)
    stage_reset;    pause
    stage_sensors;  pause
    stage_training; pause
    stage_drift;    pause
    stage_deploy;   pause
    stage_ota
    banner "ALL STAGES COMPLETE"
    ;;
  *)
    sed -n '/^# Usage/,/^# ORDER MATTERS/p' "$0" | sed 's/^#\s\?//'
    exit 1 ;;
esac
