#!/usr/bin/env python3
"""Generate model responses only, without Gemini judging."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.data_utils import load_project_dataset, load_saved_split_dataset
from src.utils.eval_utils import generate_responses, save_records
from src.utils.model_utils import ensure_bitsandbytes_available, load_config, setup_logging


def summarize_model_device_map(model) -> str:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        unique_targets = sorted({str(target) for target in hf_device_map.values()})
        return ", ".join(unique_targets)

    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "unknown"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Override model path from config.")
    parser.add_argument("--results-dir", help="Directory to write generated responses.")
    parser.add_argument("--split-name", help="Saved split name to use for generation.")
    parser.add_argument("--num-samples", type=int, help="Override number of samples to generate.")
    parser.add_argument("--max-new-tokens", type=int, help="Override max generated tokens per sample.")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Force-enable 4-bit bitsandbytes quantized loading for inference.",
    )
    return parser.parse_args()


def build_quantization_config(config: dict):
    qlora_config = config.get("qlora", {})
    if not qlora_config.get("load_in_4bit", False):
        return None

    ensure_bitsandbytes_available("4-bit inference loading")
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, qlora_config.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=qlora_config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=qlora_config.get("bnb_4bit_quant_type", "nf4"),
    )


def main():
    args = parse_args()
    config = load_config("configs/eval_config.yaml")
    if args.num_samples is not None:
        config["evaluation"]["num_samples"] = args.num_samples
    if args.max_new_tokens is not None:
        config["evaluation"]["max_new_tokens"] = args.max_new_tokens
    if args.load_in_4bit:
        config.setdefault("qlora", {})["load_in_4bit"] = True

    logger = setup_logging(logger_name="infer.model")

    model_path = args.model_path or config["model"]["path"]
    logger.info("Starting inference")
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
    logger.info("Model device placement: %s", summarize_model_device_map(model))

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
