# HotState Inference Policy Calibration

This experiment compares the inference-side HotState policy on paired requests. It keeps model, checkpoint, dataset window, HBM budget, and batch size fixed, varies one policy setting at a time, and starts a fresh process for each variant. Construction-time pool ratios are swept separately from value-model settings.

Run from `examples/hstu` in the CUDA inference environment after preprocessing the inference dataset and building the checkpoint:

```bash
python3 benchmark/hotstate_policy_sweep.py run \
  --gin-config-file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --checkpoint-dir ./checkpoints/kuairand_1k_native_pytorch_1024_b1/iter550 \
  --variants-json ./benchmark/hotstate_policy_sweep.json \
  --budget-gib 1.0 \
  --stage value_model \
  --max-batches 256 \
  --warmup-batches 16 \
  --repeats 1 \
  --out-dir ./logs/hotstate_policy_sweep_smoke
```

The smoke run checks construction and pairing only. Run the complete date window with at least three repetitions for results, using a separate output directory:

```bash
python3 benchmark/hotstate_policy_sweep.py run \
  --gin-config-file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --checkpoint-dir ./checkpoints/kuairand_1k_native_pytorch_1024_b1/iter550 \
  --variants-json ./benchmark/hotstate_policy_sweep.json \
  --budget-gib 1.0 \
  --stage value_model \
  --warmup-batches 64 \
  --repeats 3 \
  --out-dir ./logs/hotstate_policy_sweep_full
```

Analyze only completed full traces. The analyzer chooses the best static split and HotState configuration on the first half of each request window, then reports their paired end-to-end latency on the held-out half:

```bash
python3 benchmark/hotstate_policy_sweep.py analyze \
  --manifest ./logs/hotstate_policy_sweep_full/sweep_manifest.json \
  --discovery-fraction 0.5
```

Each trace includes request fingerprints, forward and end-to-end latency, controller phase timings, measured physical HBM, admissions and evictions, the exact batch-cost observation fed to the online model, and learned batch/transfer coefficients. The report rejects mismatched requests, incomplete runs, inconsistent physical accounting, and different HBM budgets.

The candidate value models are `weighted` (benefit minus movement, risk, and opportunity cost), `transfer_aware` (benefit minus movement), and `benefit_only`. The JSON manifest also performs one-factor sweeps over pool split, value weights, embedding recency/reuse horizon, time-density exponent, scheduler bandwidth fraction, and KV retention granularity. Treat the measured held-out winner as a candidate; run a second, narrower sweep around its parameters before changing defaults. Construction ratios still require rebuilding the model, and this experiment does not demonstrate live cross-pool resizing.

Run the full study in separate output directories for `pool_split`, `value_model`, `weights`, `reuse`, `scheduler`, `admission`, and `online_model`. Every stage automatically includes the three static split baselines and the weighted reference configuration. This makes each result a controlled comparison with one parameter family changing at a time.
