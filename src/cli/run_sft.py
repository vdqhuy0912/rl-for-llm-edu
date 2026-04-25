#!/usr/bin/env python3
"""Run Supervised Fine-Tuning (SFT) with LoRA/QLoRA on the configured Qwen model."""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.utils.data_utils import (
    load_hf_datasets,
    load_project_dataset,
    load_saved_split_dataset,
    preprocess_sft_data,
)
from src.utils.model_utils import ensure_output_dir, load_config, resolve_project_path, setup_logging


def load_train_and_eval_datasets(config: dict, tokenizer):
    try:
        train_dataset = load_saved_split_dataset("sft_train")
        train_size = len(train_dataset)
    except Exception:
        train_dataset = load_hf_datasets(config["data"]["train_datasets"])
        train_size = len(train_dataset)

    train_dataset = preprocess_sft_data(
        train_dataset,
        tokenizer,
        max_length=config["data"]["max_length"],
    )

    try:
        eval_dataset = load_saved_split_dataset("sft_val")
    except Exception:
        eval_dataset = load_project_dataset(config["data"]["test_dataset"], split="train", prefer_local=True)

    eval_max_samples = config["data"].get("eval_max_samples")
    if eval_max_samples:
        eval_dataset = eval_dataset.select(range(min(eval_max_samples, len(eval_dataset))))

    eval_size = len(eval_dataset)
    eval_dataset = preprocess_sft_data(
        eval_dataset,
        tokenizer,
        max_length=config["data"]["max_length"],
    )
    return train_dataset, eval_dataset, train_size, eval_size


def main():
    config = load_config("configs/sft_config.yaml")
    logger = setup_logging(logger_name="train.sft")

    logger.info("Starting SFT training")
    model_name = config["model"]["name"]
    logger.info("Loading model: %s", model_name)

    if config["qlora"]["load_in_4bit"]:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["qlora"]["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, config["qlora"]["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=config["qlora"]["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=config["qlora"]["bnb_4bit_quant_type"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=config["model"]["use_fast_tokenizer"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(**config["lora"])
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    logger.info("Loading training data")
    train_dataset, eval_dataset, train_size, eval_size = load_train_and_eval_datasets(config, tokenizer)
    logger.info("Prepared SFT datasets: train=%s eval=%s", train_size, eval_size)

    training_config = dict(config["training"])
    training_config["output_dir"] = str(ensure_output_dir(training_config["output_dir"]))
    training_config.setdefault("disable_tqdm", False)
    training_args = TrainingArguments(**training_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    logger.info("Starting SFT trainer loop with tqdm=%s", not training_args.disable_tqdm)
    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    final_output_dir = ensure_output_dir(resolve_project_path(training_args.output_dir) / "final")
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    logger.info("SFT training completed. Checkpoints in %s; final model in %s", training_args.output_dir, final_output_dir)


if __name__ == "__main__":
    main()
