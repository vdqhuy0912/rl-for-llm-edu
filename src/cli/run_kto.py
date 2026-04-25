#!/usr/bin/env python3
"""Run Knowledge Transfer Optimization (KTO) training."""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer

from src.utils.data_utils import load_project_dataset, load_saved_split_dataset, prepare_kto_data
from src.utils.model_utils import ensure_output_dir, load_config, resolve_project_path, setup_logging


def load_train_and_eval_datasets(config: dict, tokenizer):
    try:
        train_dataset = load_saved_split_dataset("kto_train")
        train_size = len(train_dataset)
    except Exception:
        train_dataset = load_project_dataset(config["data"]["train_dataset"], split="train", prefer_local=True)
        train_size = len(train_dataset)
    train_dataset = prepare_kto_data(train_dataset, tokenizer)

    try:
        eval_dataset = load_saved_split_dataset("kto_val")
    except Exception:
        eval_dataset = load_project_dataset(config["data"]["test_dataset"], split="train", prefer_local=True)

    eval_max_samples = config["data"].get("eval_max_samples")
    if eval_max_samples:
        eval_dataset = eval_dataset.select(range(min(eval_max_samples, len(eval_dataset))))
    eval_size = len(eval_dataset)
    eval_dataset = prepare_kto_data(eval_dataset, tokenizer)
    return train_dataset, eval_dataset, train_size, eval_size


def main():
    config = load_config("configs/kto_config.yaml")
    logger = setup_logging(logger_name="train.kto")

    logger.info("Starting KTO training")
    base_model_path = config["model"]["base_model_path"]
    logger.info("Loading model from: %s", base_model_path)

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

    kto_config = KTOConfig(
        output_dir=kto_training["output_dir"],
        num_train_epochs=kto_training["num_train_epochs"],
        per_device_train_batch_size=kto_training["per_device_train_batch_size"],
        per_device_eval_batch_size=kto_training["per_device_eval_batch_size"],
        gradient_accumulation_steps=kto_training["gradient_accumulation_steps"],
        learning_rate=kto_training["learning_rate"],
        weight_decay=kto_training["weight_decay"],
        warmup_steps=kto_training["warmup_steps"],
        logging_steps=kto_training["logging_steps"],
        save_steps=kto_training["save_steps"],
        eval_steps=kto_training["eval_steps"],
        save_total_limit=kto_training["save_total_limit"],
        fp16=kto_training["fp16"],
        gradient_checkpointing=kto_training["gradient_checkpointing"],
        optim=kto_training["optim"],
        lr_scheduler_type=kto_training["lr_scheduler_type"],
        report_to=kto_training["report_to"],
        remove_unused_columns=kto_training["remove_unused_columns"],
        evaluation_strategy=kto_training["evaluation_strategy"],
        save_strategy=kto_training["save_strategy"],
        load_best_model_at_end=kto_training["load_best_model_at_end"],
        disable_tqdm=kto_training["disable_tqdm"],
        beta=config["kto"]["beta"],
        desirable_weight=config["kto"]["desirable_weight"],
        undesirable_weight=config["kto"]["undesirable_weight"],
        max_length=config["kto"]["max_length"],
        truncation_mode=config["kto"]["truncation_mode"],
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
