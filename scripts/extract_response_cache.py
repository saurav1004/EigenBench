"""Extract an EigenBench response cache from an evaluations.jsonl file.

The mixed collection pipeline stores evaluee responses inside pairwise
evaluation records. This script deduplicates those embedded responses into the
cache format consumed by collection.cached_responses_path:

    {"scenario": str, "scenario_index": int, "responses": {model_name: response}}

Usage:
    python scripts/extract_response_cache.py \
        runs/source/evaluations.jsonl \
        data/responses/source_responses.jsonl
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from pipeline.utils import load_records, save_records


def extract_response_cache(evaluations_path: Path) -> list[dict]:
    records = load_records(evaluations_path)
    by_scenario: OrderedDict[int, dict] = OrderedDict()

    for record in records:
        if not isinstance(record, dict):
            continue
        scenario_index = record.get("scenario_index")
        scenario = record.get("scenario")
        if scenario_index is None or not isinstance(scenario, str):
            continue

        entry = by_scenario.setdefault(
            int(scenario_index),
            {
                "scenario": scenario,
                "scenario_index": int(scenario_index),
                "responses": {},
            },
        )

        for prefix in ("eval1", "eval2"):
            model_name = record.get(f"{prefix}_name")
            response = record.get(f"{prefix} response")
            if isinstance(model_name, str) and isinstance(response, str):
                existing = entry["responses"].get(model_name)
                if existing is not None and existing != response:
                    raise ValueError(
                        "Conflicting cached responses for "
                        f"scenario_index={scenario_index}, model={model_name}"
                    )
                entry["responses"][model_name] = response

    return list(by_scenario.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluations_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    rows = extract_response_cache(args.evaluations_path)
    save_records(args.output_path, rows)

    total_responses = sum(len(row["responses"]) for row in rows)
    print(f"Wrote {len(rows)} scenario cache rows to {args.output_path}")
    print(f"Cached responses: {total_responses}")


if __name__ == "__main__":
    main()
