#!/usr/bin/env python3
"""Rerun CUDA-OOM generations and rejudge only repaired responses."""

import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cli.run_infer import build_quantization_config, summarize_model_device_map
from src.utils.eval_utils import (
    evaluate_with_gemini,
    generate_responses,
    load_gemini_model_from_config,
    load_judge_prompts,
    load_records,
    save_records,
)
from src.utils.model_utils import load_config, setup_logging


OOM_PATTERNS = (
    "CUDA out of memory",
    "CUDACachingAllocator",
    "Tried to allocate",
)


def is_oom_generation(record: dict, patterns: tuple[str, ...] = OOM_PATTERNS) -> bool:
    generated_answer = str(record.get("generated_answer", ""))
    if not generated_answer.startswith("Error generating response:"):
        return False
    return any(pattern in generated_answer for pattern in patterns)


def is_generation_error(record: dict) -> bool:
    return str(record.get("generated_answer", "")).startswith("Error generating response:")


def build_dataset_from_records(records: list[dict]) -> Dataset:
    rows = []
    for record in records:
        rows.append(
            {
                "question": record.get("question", ""),
                "context": record.get("context", ""),
                "answer": record.get("reference_answer", record.get("answer", "")),
                "insufficient_context": record.get("insufficient_context", False),
                "multi_intent": record.get("multi_intent", False),
            }
        )
    return Dataset.from_list(rows)


def load_generation_model(config: dict, model_path: str, logger):
    logger.info("Loading model from: %s", model_path)
    logger.info(
        "CUDA status: is_available=%s device_count=%s visible_devices=%s torch_cuda_version=%s",
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        torch.version.cuda,
    )

    quantization_config = build_quantization_config(config)
    model_load_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config
        logger.info("Inference loading mode: 4-bit quantized")
    else:
        logger.info("Inference loading mode: standard precision")

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info("Model device placement: %s", summarize_model_device_map(model))
    return model, tokenizer


def merge_by_indices(base_records: list[dict], indices: list[int], replacement_records: list[dict]) -> list[dict]:
    merged = list(base_records)
    for index, replacement in zip(indices, replacement_records):
        merged[index] = replacement
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-path",
        required=True,
        help="Path to generated_responses.json/csv that may contain CUDA OOM rows.",
    )
    parser.add_argument(
        "--evaluation-path",
        help="Optional existing evaluation_results.json/csv to patch with rejudged OOM rows.",
    )
    parser.add_argument("--model-path", help="Model path used for rerun generation.")
    parser.add_argument("--results-dir", default="./results/oom_rerun", help="Directory to write repaired outputs.")
    parser.add_argument("--max-new-tokens", type=int, help="Override max generated tokens for rerun.")
    parser.add_argument("--batch-size", type=int, default=1, help="Rerun batch size. Default 1 to reduce OOM risk.")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Force-enable 4-bit bitsandbytes quantized loading for rerun inference.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Only regenerate OOM rows and write repaired generated_responses; do not call Gemini.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config("configs/eval_config.yaml")
    if args.max_new_tokens is not None:
        config["evaluation"]["max_new_tokens"] = args.max_new_tokens
    config["evaluation"]["batch_size"] = args.batch_size
    if args.load_in_4bit:
        config.setdefault("qlora", {})["load_in_4bit"] = True

    logger = setup_logging(logger_name="rerun.oom_judge")
    generated_records = load_records(args.generated_path)
    failed_indices = [
        index for index, record in enumerate(generated_records) if is_oom_generation(record)
    ]

    if not failed_indices:
        logger.info("No CUDA OOM generated answers found in %s", args.generated_path)
        save_records(generated_records, args.results_dir, "generated_responses_repaired")
        print("No CUDA OOM generated answers found.")
        return

    logger.info("Found %s CUDA OOM generated answers", len(failed_indices))
    failed_records = [generated_records[index] for index in failed_indices]
    failed_dataset = build_dataset_from_records(failed_records)

    model_path = args.model_path or config["model"]["path"]
    model, tokenizer = load_generation_model(config, model_path, logger)
    regenerated_records = generate_responses(model, tokenizer, failed_dataset, config)
    repaired_generated = merge_by_indices(generated_records, failed_indices, regenerated_records)
    successful_pairs = [
        (original_index, record)
        for original_index, record in zip(failed_indices, regenerated_records)
        if not is_generation_error(record)
    ]
    still_failed_count = len(regenerated_records) - len(successful_pairs)
    repaired_generated_path, _ = save_records(
        repaired_generated,
        args.results_dir,
        "generated_responses_repaired",
    )
    save_records(regenerated_records, args.results_dir, "generated_responses_rerun_only")
    logger.info("Saved repaired generated responses to %s", repaired_generated_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.skip_judge:
        print(f"Regenerated {len(regenerated_records)} OOM responses")
        print(f"Still failed after rerun: {still_failed_count}")
        print(f"Repaired generated responses saved to: {repaired_generated_path}")
        return

    if not successful_pairs:
        print(f"Regenerated {len(regenerated_records)} OOM responses")
        print("No regenerated responses succeeded; skipping Gemini judge.")
        print(f"Repaired generated responses saved to: {repaired_generated_path}")
        return

    gemini_model = load_gemini_model_from_config(config)
    prompt_bundle = load_judge_prompts(config["metrics"]["prompt_file"])
    successful_indices = [index for index, _ in successful_pairs]
    successful_records = [record for _, record in successful_pairs]
    rerun_evaluations = evaluate_with_gemini(gemini_model, successful_records, prompt_bundle, config)
    rerun_eval_path, _ = save_records(rerun_evaluations, args.results_dir, "evaluation_results_rerun_only")

    if args.evaluation_path:
        original_evaluations = load_records(args.evaluation_path)
        if len(original_evaluations) != len(generated_records):
            raise ValueError(
                "evaluation-path length does not match generated-path length: "
                f"{len(original_evaluations)} != {len(generated_records)}"
            )
        repaired_evaluations = merge_by_indices(original_evaluations, successful_indices, rerun_evaluations)
    else:
        repaired_evaluations = evaluate_with_gemini(gemini_model, repaired_generated, prompt_bundle, config)

    repaired_eval_path, _ = save_records(
        repaired_evaluations,
        args.results_dir,
        "evaluation_results_repaired",
    )

    print(f"Regenerated {len(regenerated_records)} OOM responses")
    print(f"Successfully regenerated and rejudged: {len(rerun_evaluations)}")
    print(f"Still failed after rerun: {still_failed_count}")
    print(f"Rejudged OOM responses saved to: {rerun_eval_path}")
    print(f"Repaired generated responses saved to: {repaired_generated_path}")
    print(f"Repaired evaluation results saved to: {repaired_eval_path}")


if __name__ == "__main__":
    main()
