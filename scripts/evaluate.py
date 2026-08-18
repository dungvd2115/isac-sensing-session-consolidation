"""Evaluate heuristics, random-valid and an optional selected learned checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from isac_ssc.evaluation.evaluator import BASELINE_REGISTRY, EVALUATION_METHODS, run_baseline_evaluation
from isac_ssc.utils.config import load_algorithm_config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env/default.yaml")
    parser.add_argument("--algorithm-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--arrival-regimes", nargs="+", choices=("independent", "clustered"),
        default=("independent", "clustered"),
    )
    parser.add_argument("--baselines", nargs="+", choices=EVALUATION_METHODS, default=None)
    parser.add_argument("--random-valid-root-seed", type=int, default=None)
    parser.add_argument("--random-valid-replicates", type=int, default=None)
    parser.add_argument("--bootstrap-root-seed", type=int, default=54001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.checkpoint is not None and arguments.algorithm_config is None:
        parser.error("--algorithm-config is required with --checkpoint")
    if arguments.checkpoint is None and arguments.algorithm_config is not None:
        parser.error("--algorithm-config is only used with --checkpoint")
    methods = (() if arguments.checkpoint is not None else tuple(BASELINE_REGISTRY)) if arguments.baselines is None else tuple(arguments.baselines)
    algorithm = None if arguments.algorithm_config is None else load_algorithm_config(arguments.algorithm_config)
    report = run_baseline_evaluation(
        load_config(arguments.config), arguments.seeds, arguments.arrival_regimes, methods,
        random_valid_root_seed=arguments.random_valid_root_seed,
        random_valid_replicates=arguments.random_valid_replicates,
        checkpoint_path=arguments.checkpoint, algorithm=algorithm,
        bootstrap_root_seed=arguments.bootstrap_root_seed, bootstrap_samples=arguments.bootstrap_samples,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({
        "output_path": str(output.resolve()),
        "unique_primitive_trace_count": report.unique_primitive_trace_count,
        "deterministic_heuristic_episode_count": report.deterministic_heuristic_episode_count,
        "random_valid_episode_count": report.random_valid_episode_count,
        "learned_policy_episode_count": report.learned_policy_episode_count,
        "episode_count": len(report.episodes), "aggregate_count": len(report.aggregates),
        "macro_aggregate_count": len(report.macro_aggregates),
    }, sort_keys=True))


if __name__ == "__main__":
    main()