#!/usr/bin/env python3
"""Preview the deterministic KTO conversion used in this project."""

import json

from datasets import load_dataset

from src.utils.data_utils import prepare_kto_data
from src.utils.model_utils import load_config


def main():
    config = load_config("configs/kto_config.yaml")
    source_name = config["data"]["train_dataset"]

    dataset = load_dataset(source_name, split="train")
    kto_dataset = prepare_kto_data(dataset)

    print(f"Source dataset: {source_name}")
    print(f"Source rows: {len(dataset)}")
    print(f"KTO rows: {len(kto_dataset)}")
    print()

    preview_size = min(6, len(kto_dataset))
    for index in range(preview_size):
        row = kto_dataset[index]
        payload = {
            "label": row["label"],
            "conversion_strategy": row["conversion_strategy"],
            "prompt": row["prompt"][:800],
            "completion": row["completion"][:500],
            "insufficient_context": row["insufficient_context"],
            "multi_intent": row["multi_intent"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
