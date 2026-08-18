"""Generate one deterministic primitive trace and save it as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isac_ssc.envs.dynamics import generate_primitive_trace
from isac_ssc.utils.config import load_config
from isac_ssc.utils.serialization import serialize_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env/default.yaml")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--arrival-regime",
        required=True,
        choices=("independent", "clustered"),
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    trace = generate_primitive_trace(
        config, arguments.seed, arguments.arrival_regime,
    )
    output = Path(arguments.output)
    serialize_trace(trace, output)

    print(json.dumps({
        "trace_id": trace.trace_id,
        "seed": trace.root_seed,
        "arrival_regime": trace.arrival_regime,
        "horizon_slots": trace.horizon_slots,
        "target_count": len({
            item.target_id for item in trace.target_states
        }),
        "communication_user_count": len({
            item.user_id for item in trace.communication_states
        }),
        "materialized_request_count": sum(
            not item.horizon_omitted
            for item in trace.request_descriptors
        ),
        "horizon_omitted_count": len(
            trace.horizon_omitted_descriptors(),
        ),
        "parent_event_count": len(trace.parent_events),
        "pending_descriptor_count": sum(
            item.source_regime == "clustered"
            and not item.horizon_omitted
            and item.arrival_slot > item.sampled_slot
            for item in trace.request_descriptors
        ),
        "output_path": str(output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()