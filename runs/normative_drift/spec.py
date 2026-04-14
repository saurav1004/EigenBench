"""
Run spec for Normative Drift dose-response experiment.

5 Qwen2.5-14B models (Base → R1 → R8 → R64 → Full FT) in all-to-all mode.
All trained on bad medical advice (config 0_1_0, 1 epoch), except Base.
Grey-area scenarios (115) with Claude constitution (39 criteria).

All models served locally via vLLM.
  - Base + R1 + R8 + R64 share a single vLLM engine (Qwen2.5-14B-Instruct + 3 LoRA adapters)
  - Full FT runs on a separate vLLM engine (different full weights)

NOTE on R1: The extended_train repo (~2.7 GB) will be fully downloaded on first run.
Only checkpoint-395 (≈1 epoch) is used as the LoRA adapter.
"""

RUN_SPEC = {
    "verbose": True,
    "models": {
        "Base": "hf_local:Qwen/Qwen2.5-14B-Instruct",
        "R1": "hf_local:ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train/checkpoints/checkpoint-395",
        "R8": "hf_local:ModelOrganismsForEM/Qwen2.5-14B-Instruct_R8_0_1_0_full_train",
        "R64": "hf_local:ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train",
        "FullFT": "hf_local:ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft",
    },
    "dataset": {
        "path": "data/scenarios/grey_area_normative_drift.json",
        "start": 0,
        "count": 115,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/claude.json",
        "num_criteria": 39,
    },
    "collection": {
        "enabled": True,
        "cached_responses_path": "runs/normative_drift/cached_responses.jsonl",
        "allow_ties": True,
        "sampler_mode": "all_to_all",
    },
    "training": {
        "enabled": True,
        "model": "btd_ties",
        "dims": [2, 3],
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1000,
        "batch_size": 32,
        "device": "cuda",
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
