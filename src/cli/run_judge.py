#!/usr/bin/env python3
"""Judge previously generated responses with Gemini."""

import argparse

from src.utils.eval_utils import (
    evaluate_with_gemini,
    load_gemini_model_from_config,
    load_judge_prompts,
    load_records,
    save_records,
)
from src.utils.model_utils import load_config, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to generated_responses.json or generated_responses.csv.",
    )
    parser.add_argument("--results-dir", help="Directory to write Gemini evaluation outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config("configs/eval_config.yaml")
    logger = setup_logging(logger_name="judge.gemini")

    logger.info("Starting Gemini judging")
    logger.info("Loading generated responses from: %s", args.input_path)
    responses = load_records(args.input_path)
    logger.info("Loaded %s generated responses", len(responses))

    gemini_model = load_gemini_model_from_config(config)
    prompt_bundle = load_judge_prompts(config["metrics"]["prompt_file"])
    evaluations = evaluate_with_gemini(gemini_model, responses, prompt_bundle, config)

    json_path, _ = save_records(evaluations, args.results_dir or "./results", "evaluation_results")
    logger.info("Gemini judging completed. Results saved to %s", json_path)
    print(f"Evaluated {len(evaluations)} responses")
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
