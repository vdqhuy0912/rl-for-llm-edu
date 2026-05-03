#!/usr/bin/env python3
"""Run Knowledge Transfer Optimization (KTO) training."""

import argparse

import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer

from src.utils.data_utils import load_project_dataset, load_saved_split_dataset, prepare_kto_data
from src.utils.model_utils import (
    ensure_bitsandbytes_available,
    ensure_output_dir,
    instantiate_config_class,
    load_config,
    resolve_project_path,
    setup_logging,
)


def load_train_and_eval_datasets(config: dict, tokenizer):
    system_prompt = config.get("prompt", {}).get("system_prompt")
    try:
        train_dataset = load_saved_split_dataset("kto_train")
        train_size = len(train_dataset)
    except Exception:
        train_dataset = load_project_dataset(config["data"]["train_dataset"], split="train", prefer_local=True)
        train_size = len(train_dataset)

    train_max_samples = config["data"].get("train_max_samples")
    if train_max_samples:
        train_dataset = train_dataset.select(range(min(train_max_samples, len(train_dataset))))
    train_dataset = prepare_kto_data(train_dataset, tokenizer, system_prompt=system_prompt)

    try:
        eval_dataset = load_saved_split_dataset("kto_val")
    except Exception:
        eval_dataset = load_project_dataset(config["data"]["val_dataset"], split="validation", prefer_local=True)

    eval_max_samples = config["data"].get("eval_max_samples")
    if eval_max_samples:
        eval_dataset = eval_dataset.select(range(min(eval_max_samples, len(eval_dataset))))
    eval_size = len(eval_dataset)
    eval_dataset = prepare_kto_data(eval_dataset, tokenizer, system_prompt=system_prompt)
    return train_dataset, eval_dataset, train_size, eval_size


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Override base/SFT model path from config.")
    parser.add_argument("--output-dir", help="Override training output directory.")
    parser.add_argument("--max-steps", type=int, help="Override Trainer max_steps.")
    parser.add_argument("--num-train-samples", type=int, help="Limit raw KTO train samples before conversion.")
    parser.add_argument("--num-eval-samples", type=int, help="Limit raw KTO eval samples before conversion.")
    parser.add_argument("--max-length", type=int, help="Override KTO max sequence length.")
    parser.add_argument("--per-device-train-batch-size", type=int, help="Override train batch size per device.")
    parser.add_argument("--per-device-eval-batch-size", type=int, help="Override eval batch size per device.")
    parser.add_argument(
        "--precompute-ref-log-probs",
        action="store_true",
        help="Precompute reference log probabilities instead of keeping a reference model in memory.",
    )
    parser.add_argument(
        "--tuning-mode",
        choices=["qlora", "lora", "none"],
        help="Override KTO tuning mode. Use qlora for 4-bit base model plus LoRA adapters.",
    )
    parser.add_argument("--use-lora", action="store_true", help="Alias for --tuning-mode lora.")
    parser.add_argument("--lora-r", type=int, help="Override LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, help="Override LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, help="Override LoRA dropout.")
    return parser.parse_args()


def build_quantization_config(config: dict):
    qlora_config = config.get("qlora", {})
    if not qlora_config.get("load_in_4bit", False):
        return None

    ensure_bitsandbytes_available("QLoRA 4-bit loading")
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, qlora_config.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=qlora_config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=qlora_config.get("bnb_4bit_quant_type", "nf4"),
    )


def should_attach_lora(model, tuning_mode: str) -> bool:
    if tuning_mode not in {"lora", "qlora"}:
        return False
    return not hasattr(model, "peft_config")


def attach_lora_adapter(model, config: dict):
    lora_config = LoraConfig(**config["lora"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    config = load_config("configs/kto_config.yaml")
    if args.model_path:
        config["model"]["base_model_path"] = args.model_path
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
        config["training"]["num_train_epochs"] = 1
    if args.num_train_samples is not None:
        config["data"]["train_max_samples"] = args.num_train_samples
    if args.num_eval_samples is not None:
        config["data"]["eval_max_samples"] = args.num_eval_samples
    if args.max_length is not None:
        config["kto"]["max_length"] = args.max_length
    if args.per_device_train_batch_size is not None:
        config["training"]["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.per_device_eval_batch_size is not None:
        config["training"]["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    if args.precompute_ref_log_probs:
        config["training"]["precompute_ref_log_probs"] = True
    if args.tuning_mode:
        config.setdefault("tuning", {})["mode"] = args.tuning_mode
    if args.use_lora:
        config.setdefault("tuning", {})["mode"] = "lora"
    if args.lora_r is not None:
        config["lora"]["r"] = args.lora_r
    if args.lora_alpha is not None:
        config["lora"]["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        config["lora"]["lora_dropout"] = args.lora_dropout

    logger = setup_logging(logger_name="train.kto")

    logger.info("Starting KTO training")
    base_model_path = config["model"]["base_model_path"]
    logger.info("Loading model from: %s", base_model_path)

    tuning_mode = config.get("tuning", {}).get("mode", "qlora").lower()
    if tuning_mode not in {"qlora", "lora", "none"}:
        raise ValueError(f"Unsupported KTO tuning mode: {tuning_mode}")
    logger.info("KTO tuning mode: %s", tuning_mode)

    optim_name = str(config["training"].get("optim", "")).lower()
    if "8bit" in optim_name:
        ensure_bitsandbytes_available(f"Optimizer `{config['training']['optim']}`")

    quantization_config = build_quantization_config(config) if tuning_mode == "qlora" else None
    model_load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            base_model_path,
            is_trainable=True,
            **model_load_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_load_kwargs,
        )

    has_peft_adapter = hasattr(model, "peft_config")

    if tuning_mode == "qlora" and not has_peft_adapter:
        model = prepare_model_for_kbit_training(model)

    if should_attach_lora(model, tuning_mode):
        model = attach_lora_adapter(model, config)

    if tuning_mode == "none" and not config["training"].get("precompute_ref_log_probs"):
        logger.warning(
            "KTO tuning mode 'none' will train the full model and usually needs a second reference model. "
            "Use qlora/lora or set precompute_ref_log_probs to reduce memory."
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    logger.info("Loading KTO data")
    train_dataset, eval_dataset, train_size, eval_size = load_train_and_eval_datasets(config, tokenizer)
    logger.info("Prepared KTO datasets: train=%s eval=%s", train_size, eval_size)

    kto_training = dict(config["training"])
    kto_training["output_dir"] = str(ensure_output_dir(kto_training["output_dir"]))
    kto_training.setdefault("disable_tqdm", False)

    kto_config = instantiate_config_class(
        KTOConfig,
        {
            **kto_training,
            "beta": config["kto"]["beta"],
            "desirable_weight": config["kto"]["desirable_weight"],
            "undesirable_weight": config["kto"]["undesirable_weight"],
            "max_length": config["kto"]["max_length"],
            "truncation_mode": config["kto"]["truncation_mode"],
        },
        aliases={
            "evaluation_strategy": "eval_strategy",
            "eval_strategy": "evaluation_strategy",
        },
    )

    trainer = KTOTrainer(
        model=model,
        args=kto_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting KTO trainer loop with tqdm=%s", not kto_config.disable_tqdm)
    trainer.train()

    trainer.save_model()
    final_output_dir = ensure_output_dir(resolve_project_path(kto_config.output_dir) / "final")
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    logger.info("KTO training completed. Checkpoints in %s; final model in %s", kto_config.output_dir, final_output_dir)


if __name__ == "__main__":
    main()
