import logging
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = resolve_project_path(config_path)
    with config_file.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str | Path = "./logs", logger_name: str = "rl_for_llm_edu"):
    """Set up file and stream logging without duplicating handlers across runs."""
    from datetime import datetime

    log_dir = resolve_project_path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_logger_name = logger_name.replace(".", "_")
    log_file = log_dir / f"{safe_logger_name}_{timestamp}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logging to %s", log_file)
    return logger


def ensure_output_dir(path_like: str | Path) -> Path:
    path = resolve_project_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(model, tokenizer, output_dir: str | Path, step: int):
    """Save model checkpoint."""
    output_path = ensure_output_dir(output_dir) / f"checkpoint-{step}"
    output_path.mkdir(exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Checkpoint saved to {output_path}")


def load_checkpoint(model_path: str | Path):
    """Load model from checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_model_path = resolve_project_path(model_path)

    model = AutoModelForCausalLM.from_pretrained(resolved_model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)

    return model, tokenizer
