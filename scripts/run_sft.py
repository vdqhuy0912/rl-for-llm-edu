#!/usr/bin/env python3
"""
Script to run Supervised Fine-Tuning (SFT) with LoRA/QLoRA on Qwen model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from src.utils.model_utils import load_config, setup_logging
from src.utils.data_utils import load_hf_datasets, load_project_dataset, preprocess_sft_data

def main():
    # Load configuration
    config = load_config('./configs/sft_config.yaml')
    logger = setup_logging()

    logger.info("Starting SFT training...")

    # Load model and tokenizer
    model_name = config['model']['name']
    logger.info(f"Loading model: {model_name}")

    if config['qlora']['load_in_4bit']:
        # QLoRA setup
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config['qlora']['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(torch, config['qlora']['bnb_4bit_compute_dtype']),
            bnb_4bit_use_double_quant=config['qlora']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=config['qlora']['bnb_4bit_quant_type']
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=config['model']['use_fast_tokenizer'],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA configuration
    lora_config = LoraConfig(**config['lora'])
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    # Load and preprocess data
    logger.info("Loading training data...")
    train_dataset = load_hf_datasets(config['data']['train_datasets'])
    train_dataset = preprocess_sft_data(
        train_dataset,
        tokenizer,
        max_length=config['data']['max_length']
    )

    # Load test data for evaluation
    test_dataset = load_project_dataset(config['data']['test_dataset'], split='train', prefer_local=True)
    eval_max_samples = config['data'].get('eval_max_samples')
    if eval_max_samples:
        eval_size = min(eval_max_samples, len(test_dataset))
        test_dataset = test_dataset.select(range(eval_size))
    test_dataset = preprocess_sft_data(
        test_dataset,
        tokenizer,
        max_length=config['data']['max_length']
    )

    # Training arguments
    training_args = TrainingArguments(**config['training'])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    final_output_dir = os.path.join(training_args.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    logger.info(f"Training completed. Model saved to {training_args.output_dir} and {final_output_dir}")

if __name__ == "__main__":
    main()
