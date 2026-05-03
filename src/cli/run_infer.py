#!/usr/bin/env python3
"""Generate model responses only, without Gemini judging."""

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.data_utils import load_project_dataset, load_saved_split_dataset
from src.utils.eval_utils import generate_responses, save_records
from src.utils.model_utils import load_config, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Override model path from config.")
    parser.add_argument("--results-dir", help="Directory to write generated responses.")
    parser.add_argument("--split-name", help="Saved split name to use for generation.")
    parser.add_argument("--num-samples", type=int, help="Override number of samples to generate.")
    parser.add_argument("--max-new-tokens", type=int, help="Override max generated tokens per sample.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config("configs/eval_config.yaml")
    if args.num_samples is not None:
        config["evaluation"]["num_samples"] = args.num_samples
    if args.max_new_tokens is not None:
        config["evaluation"]["max_new_tokens"] = args.max_new_tokens

    logger = setup_logging(logger_name="infer.model")

    model_path = args.model_path or config["model"]["path"]
    logger.info("Starting inference")
    logger.info("Loading model from: %s", model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    split_name = args.split_name or config["evaluation"].get("split_name", "test_only")
    try:
        dataset = load_saved_split_dataset(split_name)
        logger.info("Loaded deterministic split %s with %s samples", split_name, len(dataset))
    except Exception:
        dataset = load_project_dataset(
            config["evaluation"]["test_dataset"],
            split=config["evaluation"].get("source_split", "test"),
            prefer_local=True,
        )
        logger.info("Loaded dataset fallback split with %s samples", len(dataset))

    responses = generate_responses(model, tokenizer, dataset, config)
    json_path, _ = save_records(responses, args.results_dir or "./results", "generated_responses")
    logger.info("Inference completed. Results saved to %s", json_path)
    print(f"Generated {len(responses)} responses")
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
