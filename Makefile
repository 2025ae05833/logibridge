# LogiEdge -- build and demonstration targets
#
#   make setup       install Python dependencies
#   make all         full pipeline: dataset -> train -> variants -> benchmark
#   make docker      build the inference container
#   make ota-demo    Docker layer-cache demonstration (Task D2)
#   make reference   build the PSI reference distribution (Task E1)
#   make deploy      run the Ansible playbook twice to show idempotency (E2)
#   make clean       remove generated artefacts

SHELL   := /bin/bash
PY      ?= python3
SPEED   ?= 400
IMAGE   ?= logibridge-inference
TAG     ?= latest

.PHONY: setup all dataset train variants benchmark normexp architecture \
        docker ota-demo reference drift deploy clean report

# ---------------------------------------------------------------- setup
setup:
	$(PY) -m pip install -r requirements-dev.txt

# ------------------------------------------------------------ pipeline
all: dataset train variants benchmark normexp architecture report
	@echo
	@echo "=============================================================="
	@echo " LogiEdge pipeline complete."
	@echo "   models      : training/models/"
	@echo "   benchmark   : optimisation/results/benchmark_results.csv"
	@echo "   pareto      : optimisation/results/pareto_chart.png"
	@echo "   calculations: reports/calculations_output.txt"
	@echo "=============================================================="

dataset:
	cd training && $(PY) generate_dataset.py --speed $(SPEED)

train:
	cd training && $(PY) train_model.py

variants:
	cd training && $(PY) convert_ptq.py && $(PY) prune_quantise.py

benchmark:
	cd optimisation && $(PY) benchmark.py

normexp:
	cd data_pipeline && $(PY) normalisation_experiment.py \
	  | tee ../reports/normalisation_experiment.txt

architecture:
	cd scenario_architecture && $(PY) make_architecture.py

report:
	cd reports && $(PY) calculations.py > calculations_output.txt && \
	  echo "calculations -> reports/calculations_output.txt"

# -------------------------------------------------------------- docker
# The inference build context needs the pipeline modules and a model.
docker:
	cp data_pipeline/preprocessing.py data_pipeline/simulator.py \
	   data_pipeline/training_stats.npy inference/
	cp training/models/m2_int8.tflite inference/model.tflite
	cd inference && docker build -t $(IMAGE):$(TAG) .

# ------------------------------------------------- Task D2 OTA demo
# Proves that a model-only change rebuilds exactly one layer.
ota-demo:
	@echo "=============================================================="
	@echo " OTA LAYER-CACHE DEMONSTRATION (Task D2)"
	@echo "=============================================================="
	@echo
	@echo ">>> STEP 1  build with the M2 INT8 model"
	cp training/models/m2_int8.tflite inference/model.tflite
	cd inference && docker build --no-cache -t $(IMAGE):m2 . | tail -25
	@echo
	@echo ">>> STEP 2  swap ONLY the model file (M2 -> M3) and rebuild"
	cp training/models/m3_pruned_int8.tflite inference/model.tflite
	cd inference && docker build -t $(IMAGE):m3 . | tail -25
	@echo
	@echo ">>> Every step up to and including the pip install should read"
	@echo "    CACHED. Only the final COPY model.tflite re-executes."
	@echo
	@echo ">>> STEP 3  layer sizes"
	docker history $(IMAGE):m3 --format "table {{.Size}}\t{{.CreatedBy}}" | head -12
	@echo
	@echo ">>> STEP 4  bandwidth arithmetic for the 85-truck pilot"
	@$(PY) -c "import os; \
	m=os.path.getsize('training/models/m3_pruned_int8.tflite')/1e6; \
	full=188.0; n=85; \
	print(f'  full image per truck : {full:9.3f} MB'); \
	print(f'  model layer per truck: {m:9.6f} MB'); \
	print(f'  fleet full image     : {full*n:9.1f} MB  = Rs {full*n*0.10:8.2f}'); \
	print(f'  fleet model layer    : {m*n:9.4f} MB  = Rs {m*n*0.10:8.4f}'); \
	print(f'  saved                : {full*n-m*n:9.1f} MB  = Rs {(full*n-m*n)*0.10:8.2f}'); \
	print(f'  reduction factor     : {full/m:9.0f}x')"

# ------------------------------------------------- Task E1 drift monitoring
#
# MODEL pins the variant that is BOTH deployed to inference/model.tflite and
# used to build the PSI reference. These must be the same artefact: PSI
# compares a live score distribution against a stored one, so serving a
# different model than the reference was built on makes every bin differ for
# reasons unrelated to drift, and the monitor alerts permanently on clean
# data. (`make ota-demo` deliberately leaves M3 in inference/model.tflite,
# which is exactly how that mismatch arises.)
MODEL ?= training/models/m2_int8.tflite

reference:
	@echo ">>> pinning $(MODEL) as the served model, then building its reference"
	cp $(MODEL) inference/model.tflite
	$(PY) data_pipeline/simulator.py --anomaly none --duration 3200 \
	  --speed 8000 --sink file --file /tmp/logibridge_clean_ref.jsonl --seed 555
	cd monitoring && $(PY) drift_monitor.py --build-reference \
	  --clean-capture /tmp/logibridge_clean_ref.jsonl --score p_normal \
	  --model ../inference/model.tflite

drift:
	@test -f monitoring/reference_dist.json || \
	  { echo "no reference_dist.json -- run 'make reference' first"; exit 1; }
	@$(PY) -c "import hashlib,json,sys; \
	r=json.load(open('monitoring/reference_dist.json')).get('model_sha'); \
	s=hashlib.sha256(open('inference/model.tflite','rb').read()).hexdigest()[:12]; \
	sys.exit(0) if r==s else (print(f'MISMATCH: reference built on model {r}, serving {s}. Run: make reference'), sys.exit(1))"
	chmod +x demo/drift_demo.sh && ./demo/drift_demo.sh 10 15

# ------------------------------------------------- Task E2 idempotency
deploy:
	@echo ">>> RUN 1 (expect changed>0 on a fresh target)"
	cd deployment && ansible-playbook -i inventory.ini logibridge_deploy.yml
	@echo
	@echo ">>> RUN 2 (expect changed=0 -- this is the idempotency proof)"
	cd deployment && ansible-playbook -i inventory.ini logibridge_deploy.yml

# --------------------------------------------------------------- clean
clean:
	rm -rf training/data training/models/*.tflite training/models/*.keras \
	       training/models/*.json optimisation/results/* \
	       monitoring/psi_history.json inference/data \
	       inference/preprocessing.py inference/simulator.py \
	       inference/training_stats.npy inference/model.tflite
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned"
