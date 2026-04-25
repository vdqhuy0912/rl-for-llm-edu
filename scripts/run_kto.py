#!/usr/bin/env python3
"""
Script to run Knowledge Transfer Optimization (KTO) training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOTrainer, KTOConfig
from peft import AutoPeftModelForCausalLM

from src.utils.model_utils import load_config, setup_logging
from src.utils.data_utils import load_project_dataset, prepare_kto_data

def main():
    # Load configuration
    config = load_config('./configs/kto_config.yaml')
    logger = setup_logging()

    logger.info("Starting KTO training...")

    # Load base model from SFT checkpoint
    base_model_path = config['model']['base_model_path']
    logger.info(f"Loading model from: {base_model_path}")

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

    # Load and prepare KTO data
    logger.info("Loading KTO training data...")
    kto_dataset = load_project_dataset(config['data']['train_dataset'], split='train', prefer_local=True)
    kto_dataset = prepare_kto_data(kto_dataset, tokenizer)

    # Load test data
    test_dataset = load_project_dataset(config['data']['test_dataset'], split='train', prefer_local=True)
    eval_max_samples = config['data'].get('eval_max_samples')
    if eval_max_samples:
        eval_size = min(eval_max_samples, len(test_dataset))
        test_dataset = test_dataset.select(range(eval_size))
    test_dataset = prepare_kto_data(test_dataset, tokenizer)

    # KTO configuration
    kto_config = KTOConfig(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        optim=config['training']['optim'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        report_to=config['training']['report_to'],
        remove_unused_columns=config['training']['remove_unused_columns'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        beta=config['kto']['beta'],
        desirable_weight=config['kto']['desirable_weight'],
        undesirable_weight=config['kto']['undesirable_weight'],
        max_length=config['kto']['max_length'],
        truncation_mode=config['kto']['truncation_mode'],
    )

    # KTO Trainer
    trainer = KTOTrainer(
        model=model,
        args=kto_config,
        train_dataset=kto_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting KTO training...")
    trainer.train()

    # Save final model
    trainer.save_model()
    final_output_dir = os.path.join(kto_config.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    logger.info(f"KTO training completed. Model saved to {kto_config.output_dir} and {final_output_dir}")

if __name__ == "__main__":
    main()
