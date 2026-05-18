#!/usr/bin/env python3
"""Generate model responses with vLLM, including optional LoRA adapters."""

import argparse
import json
from pathlib import Path

from src.utils.data_utils import build_instruction_prompt, load_project_dataset, load_saved_split_dataset, normalize_qa_example
from src.utils.eval_utils import save_records
from src.utils.model_utils import load_config, resolve_project_path, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Base model id/path or LoRA adapter directory.")
    parser.add_argument("--results-dir", required=True, help="Directory to write generated_responses.json.")
    parser.add_argument("--split-name", default="test_only", help="Saved split name to use for generation.")
    parser.add_argument("--num-samples", type=int, help="Override number of samples to generate.")
    parser.add_argument("--max-new-tokens", type=int, help="Override max generated tokens per sample.")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of prompts per vLLM generate call.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def resolve_model_and_lora(model_path: str) -> tuple[str, str | None]:
    resolved_path = resolve_project_path(model_path)
    adapter_config_path = resolved_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return model_path, None

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"LoRA adapter config does not define base_model_name_or_path: {adapter_config_path}")
    return str(base_model), str(resolved_path)


def load_eval_dataset(config: dict, split_name: str):
    try:
        return load_saved_split_dataset(split_name)
    except Exception:
        return load_project_dataset(
            config["evaluation"]["test_dataset"],
            split=config["evaluation"].get("source_split", "test"),
            prefer_local=True,
        )


def main():
    args = parse_args()
    config = load_config("configs/eval_config.yaml")
    if args.num_samples is not None:
        config["evaluation"]["num_samples"] = args.num_samples
    if args.max_new_tokens is not None:
        config["evaluation"]["max_new_tokens"] = args.max_new_tokens

    logger = setup_logging(logger_name="infer.vllm")
    base_model_path, lora_path = resolve_model_and_lora(args.model_path)
    logger.info("Starting vLLM inference")
    logger.info("Requested model path: %s", args.model_path)
    logger.info("vLLM base model: %s", base_model_path)
    logger.info("vLLM LoRA adapter: %s", lora_path or "none")
    logger.info(
        "Generation config: temperature=0.0 do_sample=false max_new_tokens=%s repetition_penalty=%s",
        config["evaluation"]["max_new_tokens"],
        config["evaluation"].get("repetition_penalty", 1.0),
    )

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=base_model_path,
        tokenizer=base_model_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=lora_path is not None,
        max_lora_rank=64,
    )
    tokenizer = llm.get_tokenizer()

    dataset = load_eval_dataset(config, args.split_name)
    num_samples = min(config["evaluation"]["num_samples"], len(dataset))
    prompt_config = config.get("prompt", {})
    system_prompt = prompt_config.get("system_prompt")
    enable_thinking = prompt_config.get("enable_thinking", False)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=config["evaluation"]["max_new_tokens"],
        repetition_penalty=config["evaluation"].get("repetition_penalty", 1.0),
        skip_special_tokens=True,
    )
    lora_request = LoRARequest("adapter", 1, lora_path) if lora_path else None

    responses = []
    for start in range(0, num_samples, args.batch_size):
        batch_examples = [normalize_qa_example(dataset[index]) for index in range(start, min(start + args.batch_size, num_samples))]
        prompts = [
            build_instruction_prompt(
                example["question"],
                example["context"],
                system_prompt=system_prompt,
                tokenizer=tokenizer,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for example in batch_examples
        ]
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        for example, output in zip(batch_examples, outputs):
            generated_answer = output.outputs[0].text.strip() if output.outputs else ""
            responses.append(
                {
                    "question": example["question"],
                    "context": example["context"],
                    "reference_answer": example["answer"],
                    "generated_answer": generated_answer,
                    "insufficient_context": example["insufficient_context"],
                    "multi_intent": example["multi_intent"],
                }
            )
        logger.info("Generated %s/%s responses", len(responses), num_samples)

    json_path, _ = save_records(responses, args.results_dir, "generated_responses")
    logger.info("vLLM inference completed. Results saved to %s", json_path)
    print(f"Generated {len(responses)} responses")
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
