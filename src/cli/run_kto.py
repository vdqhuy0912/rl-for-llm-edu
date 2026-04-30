#!/usr/bin/env python3
"""Run Knowledge Transfer Optimization (KTO) training."""

import torch
from peft import AutoPeftModelForCausalLM
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


def main():
    config = load_config("configs/kto_config.yaml")
    logger = setup_logging(logger_name="train.kto")

    logger.info("Starting KTO training")
    base_model_path = config["model"]["base_model_path"]
    logger.info("Loading model from: %s", base_model_path)

    optim_name = str(config["training"].get("optim", "")).lower()
    if "8bit" in optim_name:
        ensure_bitsandbytes_available(f"Optimizer `{config['training']['optim']}`")

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            is_trainable=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
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
        tokenizer=tokenizer,
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
