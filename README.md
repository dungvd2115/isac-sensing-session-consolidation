# Sense Once, Serve Many

## Common-Trace Factorized Constrained PPO for Online Sensing-Session Consolidation in Multi-Tenant ISAC Networks

This repository contains the research implementation accompanying the manuscript "Sense Once, Serve Many: Common-Trace Factorized Constrained PPO for Online Sensing-Session Consolidation in Multi-Tenant ISAC Networks."

## Overview

The project studies online sensing-session consolidation in a multi-tenant integrated sensing and communication (ISAC) network. Compatible sensing requests can share one physical sensing session, while sensing resources compete with communication traffic. At each decision point, the centralized orchestrator chooses `MERGE`, `CREATE`, `DEFER`, or `REJECT` under heterogeneous sensing service-level agreements and communication quality-of-service constraints.

Common-Trace Factorized Constrained PPO (CT-PPO) is the proposed training method. It learns an online policy for this constrained control problem without changing the environment, reward, action semantics, or deployment information boundary.

## Method

CT-PPO uses:

- a permutation-invariant Set representation of requests, sessions, and global state;
- a masked factorized actor over action type and applicable session/profile choices;
- stochastic training replicas exposed to the same primitive workload trace;
- leave-one-out discounted Monte-Carlo reward contrasts aligned by physical time;
- common reward credit for every applicable actor factor;
- factor/prefix-specific credit for sensing and communication constraints.

Common-Trace grouping is used only during training. Deployment and deterministic evaluation do not receive peer returns, future arrivals, oracle information, hindsight labels, or Common-Trace replicas.

The machine method identifier is `common_trace_constrained_ppo`, and its credit schema is `common_trace_leave_one_out_mc_factor_credit_v1`.

## Matched Reference

Joint-Credit PPO (JC-PPO) is the matched learned reference. It shares the environment, observations, masks, Set representation, factorized action distribution, constrained PPO backbone, and interaction budget with CT-PPO while retaining joint trajectory reward credit.

The JC-PPO method identifier is `joint_credit_constrained_ppo`, and its credit schema is `joint_trajectory_credit_v2_scale_consistent`.

## Repository Structure

```text
configs/
  algorithm/   Constrained PPO and model configuration
  env/         Frozen ISAC environment configuration
  experiment/  CT-PPO and JC-PPO training protocols
scripts/
  train.py            Training, continuation, and branch-resume CLI
  evaluate.py         Heuristic, Random Valid, and checkpoint evaluation CLI
  generate_traces.py  Deterministic primitive-trace generator
  solve_oracle.py     Tractable offline-reference smoke instance
src/isac_ssc/
  algorithms/   PPO objectives, buffers, and Common-Trace optimization
  baselines/    CT-PPO/JC-PPO agents and deterministic online heuristics
  core/         Entities, compatibility, sensing quality, resources, and SLA accounting
  envs/         Workload dynamics, action semantics, masks, observations, and environment
  evaluation/   Episode evaluation, aggregation, metrics, and confidence intervals
  models/       Set encoder, actor, and value models
  oracles/      Exhaustive and MILP offline references for tractable traces
  training/     Rollout, checkpoint, logging, and trainer implementations
  utils/        Configuration, calibration, seeding, and serialization
tests/
  unit/         Scientific and algorithmic regression tests
  integration/  End-to-end environment, oracle, training, and resume tests
```

Generated data, checkpoints, logs, and evaluation outputs are intentionally excluded from normal Git history.

## Installation

Python 3.11 or newer is required by the package metadata.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

The dependency ranges are also listed in `requirements.txt`.

## Reproducing the Paper Protocol

Run all commands from the repository root after installation.

### Primitive Trace Generation

Training and evaluation generate deterministic primitive traces internally. To materialize one trace for inspection or replay:

```bash
python scripts/generate_traces.py --config configs/env/default.yaml --seed 52001 --arrival-regime independent --output data/traces/trace_52001_independent.json
```

### CT-PPO Training

The following command runs training seed 0 with the frozen CT-PPO configuration:

```bash
python scripts/train.py --env-config configs/env/default.yaml --algorithm-config configs/algorithm/constrained_ppo.yaml --experiment-config configs/experiment/common_trace.yaml --seed 0 --output-root artifacts/training --run-name ct_seed_0
```

Repeat with seeds `1`, `2`, `3`, and `4`, using matching run names.

### JC-PPO Training

The matched JC-PPO seed-0 run uses:

```bash
python scripts/train.py --env-config configs/env/default.yaml --algorithm-config configs/algorithm/constrained_ppo.yaml --experiment-config configs/experiment/joint_credit.yaml --seed 0 --output-root artifacts/training --run-name jc_seed_0
```

Repeat with seeds `1`, `2`, `3`, and `4`, using matching run names.

### Deterministic Heuristics and Random Valid

This command evaluates the four deterministic online heuristics and Random Valid on all registered external roots and both arrival regimes:

```bash
python scripts/evaluate.py \
  --config configs/env/default.yaml \
  --seeds 52001 52002 52003 52004 52005 52006 52007 52008 52009 52010 52011 52012 52013 52014 52015 52016 52017 52018 52019 52020 52021 52022 52023 52024 52025 52026 52027 52028 52029 52030 52031 52032 52033 52034 52035 52036 52037 52038 52039 52040 52041 52042 52043 52044 52045 52046 52047 52048 52049 52050 \
  --arrival-regimes independent clustered \
  --baselines no_consolidation static_compatibility_merge greedy_incremental_cost sla_aware_greedy random_valid \
  --random-valid-root-seed 53001 \
  --random-valid-replicates 4 \
  --bootstrap-root-seed 54001 \
  --bootstrap-samples 10000 \
  --output artifacts/evaluation/baselines_and_random_valid.json
```

### Learned Checkpoint Evaluation

After the seed-0 CT-PPO command has produced `best.pt`, evaluate that checkpoint with:

```bash
python scripts/evaluate.py \
  --config configs/env/default.yaml \
  --algorithm-config configs/algorithm/constrained_ppo.yaml \
  --checkpoint artifacts/training/ct_seed_0/best.pt \
  --seeds 52001 52002 52003 52004 52005 52006 52007 52008 52009 52010 52011 52012 52013 52014 52015 52016 52017 52018 52019 52020 52021 52022 52023 52024 52025 52026 52027 52028 52029 52030 52031 52032 52033 52034 52035 52036 52037 52038 52039 52040 52041 52042 52043 52044 52045 52046 52047 52048 52049 52050 \
  --arrival-regimes independent clustered \
  --bootstrap-root-seed 54001 \
  --bootstrap-samples 10000 \
  --output artifacts/evaluation/ct_seed_0.json
```

Repeat for every CT-PPO and JC-PPO training seed, changing only the checkpoint and output paths. The evaluator writes per-episode records, regime aggregates, equal-regime macro aggregates, and bootstrap confidence intervals to JSON. No separate public table/figure-generation script is currently included.

### Offline Reference Smoke Test

```bash
python scripts/solve_oracle.py
```

This command checks the exhaustive and MILP offline references on a tractable deterministic instance; it is not a full-scale training command.

## Final Paper Protocol

- Training seeds: `0` through `4`.
- Physical interaction budget: 1,000,000 slots per run.
- Rollout target: 5,000 physical slots.
- PPO minibatch: 512 decisions; 10 epochs per rollout.
- Initial learning rate: `3e-4`, linearly scheduled over 1,000,000 physical slots.
- Validation roots: `51001` through `51020`.
- External evaluation roots: `52001` through `52050`.
- Arrival regimes: `independent` and `clustered`.
- Learned-checkpoint evaluation: deterministic.
- Random Valid: root `53001`, four action-sampling replicates per trace.
- Bootstrap: root `54001`, 10,000 samples.

## Artifacts

Final paper checkpoints and experiment bundles are not included in normal Git history. Public release-asset URLs and their integrity hashes are pending. Until those assets are published, the repository provides the frozen code and configs but does not by itself reproduce the reported numerical results without retraining.

## Tests

```bash
python -m pytest -q
```

The retained suite covers environment determinism, action and mask semantics, resource/reward/SLA accounting, Common-Trace credit construction, policy/value behavior, evaluator metrics, checkpoint identity, training protocol, and offline-reference consistency.

## Citation

Citation metadata is provided in `CITATION.cff`. Journal DOI, volume, issue, pages, and final publication metadata are intentionally omitted until assigned.

## License

No software license has been selected for this repository. A license decision is required from the owner before public reuse terms can be stated.

## Author

The implementation and experiments were performed by Dang-Dung Vu.
