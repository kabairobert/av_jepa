# Hypothesis Testing Framework

Lightweight system for tracking scientific hypotheses, running experiment batches, and auto-evaluating results from wandb.

## Directory Structure

```
hyp_testing/
├── README.md                        ← this file
├── MASTER_RESULTS.md                ← auto-generated: one-line verdict per hypothesis across all batches
├── scripts/
│   ├── sweep.py                     ← launch configs for a batch or all batches
│   └── run_hypotheses.py            ← pull wandb metrics, evaluate rules, write RESULTS.md
└── batches/
    ├── B01_predictor_geometry/
    │   ├── hypotheses.yaml          ← machine-readable: claims, configs, decision rules
    │   ├── HYPOTHESES.md            ← human-readable version
    │   └── RESULTS.md               ← auto-generated after runs
    ├── B02_noise_reweighting/
    │   ├── hypotheses.yaml
    │   ├── HYPOTHESES.md
    │   └── RESULTS.md
    └── B03_two_stage/
        ├── hypotheses.yaml
        ├── HYPOTHESES.md
        └── RESULTS.md
```

## Workflow

### 1. Before running

Edit the `hypotheses.yaml` in the relevant batch folder. Fill in:
- `decision_rules[].threshold` — the numeric threshold for SUPPORTED/REFUTED
- `expected_direction` — which config you expect to win

### 2. Launch experiments

```bash
# Run one batch
python scripts/sweep.py --batch B01_predictor_geometry --cfg_dir ../cfgs

# Run all batches (deduplicates shared configs automatically)
python scripts/sweep.py --all --cfg_dir ../cfgs

# Dry run: see what would launch without launching
python scripts/sweep.py --all --dry-run
```

Each wandb run is tagged with `[batch_id, config_name]` so `run_hypotheses.py` can find it.

### 3. After runs complete

```bash
# Evaluate one batch → writes batches/B01.../RESULTS.md
python scripts/run_hypotheses.py --batch B01_predictor_geometry --wandb_project YOUR_PROJECT

# Evaluate all batches → writes all RESULTS.md + MASTER_RESULTS.md
python scripts/run_hypotheses.py --all --wandb_project YOUR_PROJECT
```

### 4. Review

- Read `MASTER_RESULTS.md` for top-level verdicts
- Read per-batch `RESULTS.md` for metric tables + narrative
- Only open wandb for items flagged `⚠️ Manual plot check required`

## Adding a New Batch

1. Create `batches/BXX_name/hypotheses.yaml` following the schema in existing batches
2. Create `batches/BXX_name/HYPOTHESES.md` (human-readable version)
3. Create empty `batches/BXX_name/RESULTS.md`
4. Add required configs to `../cfgs/` if not already present
5. Run `sweep.py --batch BXX_name` then `run_hypotheses.py --batch BXX_name`

## hypotheses.yaml Schema

```yaml
batch_id: BXX_name
description: "One sentence focus question"
configs:
  - C01_diag_l1prior_l1pred_3D2f   # must match filename in cfgs/ (without .yaml)
  - C03_affine_l1prior_l1pred_3D2f

hypotheses:
  - id: H1
    claim: "Short testable claim"
    configs_compared: [C01_diag_l1prior_l1pred_3D2f, C03_affine_l1prior_l1pred_3D2f]
    primary_metrics: [r2_mean, retrieval_l2_at_1]
    decision_rules:
      - metric: r2_mean
        condition: "{C03} > {C01} + {threshold}"
        threshold: 0.05          # ← YOU FILL THIS IN before running
        verdict_true: SUPPORTED
        verdict_false: REFUTED
    check_plots:
      - "3D latent scatter z_A and z_B for C01 vs C03"
```

## Notes

- `run_hypotheses.py` pulls the **final** value of each metric (last logged step)
- Shared configs across batches are run once and tagged with all relevant batch IDs
- `RESULTS.md` files are overwritten on each run — commit them after reviewing
- Thresholds in `hypotheses.yaml` are intentionally left as `null` stubs — fill before running
