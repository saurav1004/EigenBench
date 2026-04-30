"""
Structured shadow-axis run: Sycophancy on Loving/Sycophancy boundary scenarios.

This run uses the same structured scenarios and model population as
`loving_vs_sycophancy_structured`, but swaps the constitution to
`oct_sycophancy.json`.

Interpretation:
- Higher score = more sycophantic according to the OCT sycophancy criteria.
- The key question is whether OCT Loving scores higher than Base, Goodness,
  and Nonchalance on this shadow axis.

Run from the EigenBench repo root:
    python scripts/run.py runs/loving_vs_sycophancy_structured_shadow/spec.py
"""

RUN_SPEC = {
    "name": "loving_vs_sycophancy_structured_shadow",
    "verbose": True,
    "models": {
        # OCT LoRA models from https://huggingface.co/maius/llama-3.1-8b-it-personas
        "Llama-3.1-8B-Instruct Base": "hf_local:meta-llama/Llama-3.1-8B-Instruct",
        "OCT Loving": "hf_local:maius/llama-3.1-8b-it-personas/loving",
        "OCT Sycophancy": "hf_local:maius/llama-3.1-8b-it-personas/sycophancy",
        "OCT Goodness": "hf_local:maius/llama-3.1-8b-it-personas/goodness",
        "OCT Nonchalance Control": "hf_local:maius/llama-3.1-8b-it-personas/nonchalance",

        # Frontier anchors.
        "Gemini 2.5 Pro Anchor": "google/gemini-2.5-pro",
        "Claude 4 Sonnet Anchor": "anthropic/claude-sonnet-4",
        "GPT-4o Anchor": "openai/gpt-4o",
    },
    "dataset": {
        "path": "data/scenarios/loving_sycophancy_structured_v0.json",
        "start": 0,
        "count": 30,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/oct_sycophancy.json",
        "num_criteria": 10,
    },
    "collection": {
        "enabled": True,
        "cached_responses_path": "data/responses/loving_vs_sycophancy_structured_responses.jsonl",
        "allow_ties": True,
        "group_size": 5,
        "groups": 2,
        "sampler_mode": "random_judge_group",
        "sampler_seed": 42,
        "alpha": 2.0,
    },
    "training": {
        "enabled": True,
        "model": "btd_ties",
        "dims": [2],
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1000,
        "batch_size": 32,
        "device": "cpu",
        "test_size": 0.2,
        "group_split": False,
        "separate_criteria": False,
        "bootstrap": {
            "enabled": True,
            "n_bootstraps": 100,
            "random_seed": 42,
            "save_models": False,
            "save_trust_matrices": True,
        },
    },
}
