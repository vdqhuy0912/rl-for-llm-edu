"""Shared helpers for model inference and Gemini judging."""

import copy
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.utils.data_utils import build_instruction_prompt, normalize_qa_example
from src.utils.model_utils import ensure_output_dir, resolve_project_path


CLASSIFIER_MARKER = 'PROMPT_QA_CLASSIFIER = """'
CLASS1_MARKER = 'PROMPT_QA_CLASS1 = """'
CLASS2_MARKER = 'PROMPT_QA_CLASS2 = """'
CLASS3_MARKER = 'PROMPT_QA_CLASS3 = """'
GEMINI_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def is_retryable_gemini_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in GEMINI_RETRY_STATUS_CODES:
        return True
    message = str(exc)
    return any(f"{status_code}" in message for status_code in GEMINI_RETRY_STATUS_CODES)


def generate_gemini_text(client, model_name: str, prompt: str, *, max_retries: int = 5) -> str:
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text or ""
        except Exception as exc:
            if attempt >= max_retries or not is_retryable_gemini_error(exc):
                raise
            sleep_seconds = min(60.0, (2 ** attempt) + random.uniform(0.0, 1.0))
            print(
                f"Gemini request failed with retryable error: {exc}. "
                f"Retrying in {sleep_seconds:.1f}s ({attempt + 1}/{max_retries})"
            )
            time.sleep(sleep_seconds)


def setup_gemini(api_key: str, model_name: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Gemini judging now requires the `google-genai` package. "
            "Install it with `pip install google-genai` or update the project environment."
        ) from exc

    return genai.Client(api_key=api_key), model_name


def load_judge_prompts(prompt_file: str) -> dict:
    raw_text = resolve_project_path(prompt_file).read_text(encoding="utf-8")

    if (
        CLASSIFIER_MARKER not in raw_text
        or CLASS1_MARKER not in raw_text
        or CLASS2_MARKER not in raw_text
        or CLASS3_MARKER not in raw_text
    ):
        raise ValueError("Judge prompt file does not contain the expected prompt sections.")

    classifier_block = raw_text.split(CLASSIFIER_MARKER, maxsplit=1)[1]
    classifier_prompt = classifier_block.split('"""', maxsplit=1)[0].strip()

    class1_block = raw_text.split(CLASS1_MARKER, maxsplit=1)[1]
    class1_prompt = class1_block.split('"""', maxsplit=1)[0].strip()
    class2_block = raw_text.split(CLASS2_MARKER, maxsplit=1)[1]
    class2_prompt = class2_block.split('"""', maxsplit=1)[0].strip()
    class3_block = raw_text.split(CLASS3_MARKER, maxsplit=1)[1]
    class3_prompt = class3_block.split('"""', maxsplit=1)[0].strip()

    return {
        "classifier": classifier_prompt,
        "class_1": class1_prompt,
        "class_2": class2_prompt,
        "class_3": class3_prompt,
    }


def render_classifier_prompt(template: str, question: str, context: str) -> str:
    return template.replace("{QUESTION}", question).replace("{CONTEXT}", context)


def render_answer_prompt(template: str, question: str, context: str, answer: str) -> str:
    return template.replace("{Q}", question).replace("{C}", context).replace("{A_gen}", answer)


def parse_json_object(text: str) -> dict:
    """Parse a Gemini JSON response, tolerating markdown fences and surrounding prose."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def classify_question_context(gemini_model, prompts: dict, question: str, context: str) -> dict:
    client, model_name = gemini_model
    prompt = render_classifier_prompt(prompts["classifier"], question, context)
    text = generate_gemini_text(client, model_name, prompt).strip()
    try:
        return parse_json_object(text)
    except json.JSONDecodeError as exc:
        return {
            "classification": "PARSE_ERROR",
            "parse_error": str(exc),
            "raw_response": text,
        }


def generate_responses(model, tokenizer, dataset, config):
    prompt_config = config.get("prompt", {})
    system_prompt = prompt_config.get("system_prompt")
    enable_thinking = prompt_config.get("enable_thinking", False)
    responses = []
    num_samples = min(config["evaluation"]["num_samples"], len(dataset))
    batch_size = max(1, int(config["evaluation"].get("batch_size", 1)))
    model.eval()

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_length = None
    generation_config.max_new_tokens = config["evaluation"]["max_new_tokens"]
    generation_config.do_sample = config["evaluation"]["do_sample"]
    if generation_config.eos_token_id is None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else generation_config.eos_token_id
        )
    generation_config.repetition_penalty = config["evaluation"].get("repetition_penalty", 1.0)
    generation_config.no_repeat_ngram_size = config["evaluation"].get("no_repeat_ngram_size", 0)
    if generation_config.do_sample:
        generation_config.temperature = config["evaluation"]["temperature"]
        generation_config.top_p = config["evaluation"].get("top_p", 1.0)
    else:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None

    model_device = next(model.parameters()).device

    with torch.no_grad():
        for start in tqdm(range(0, num_samples, batch_size), desc="Generating responses"):
            batch_examples = []
            batch_prompts = []

            for index in range(start, min(start + batch_size, num_samples)):
                example = normalize_qa_example(dataset[index])
                batch_examples.append(example)
                batch_prompts.append(
                    build_instruction_prompt(
                        example["question"],
                        example["context"],
                        system_prompt=system_prompt,
                        tokenizer=tokenizer,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                )

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

            try:
                output_ids = model.generate(
                    **inputs,
                    generation_config=generation_config,
                )
                prompt_length = inputs["input_ids"].shape[1]

                for example, generated_ids in zip(batch_examples, output_ids):
                    generated_tokens = generated_ids[prompt_length:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    responses.append(
                        {
                            "question": example["question"],
                            "context": example["context"],
                            "reference_answer": example["answer"],
                            "generated_answer": response,
                            "insufficient_context": example["insufficient_context"],
                            "multi_intent": example["multi_intent"],
                        }
                    )
            except Exception as exc:
                error_message = f"Error generating response: {exc}"
                for example in batch_examples:
                    responses.append(
                        {
                            "question": example["question"],
                            "context": example["context"],
                            "reference_answer": example["answer"],
                            "generated_answer": error_message,
                            "insufficient_context": example["insufficient_context"],
                            "multi_intent": example["multi_intent"],
                        }
                    )

    return responses


def load_jsonl_records(path: str | Path) -> list[dict]:
    path = resolve_project_path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSONL checkpoint line {line_no} in {path}: {exc}")
    return records


def append_jsonl_record(path: str | Path, record: dict) -> None:
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def judge_error_record(item: dict, index: int, stage: str, exc: Exception) -> dict:
    return {
        **item,
        "source_index": index,
        "judge_classification": {
            "classification": "JUDGE_ERROR",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
        "judge_classification_label": "JUDGE_ERROR",
        "judge_evaluation_mode": "judge_error",
        "judge_evaluation": None,
    }


def evaluate_with_gemini(gemini_model, responses, prompt_bundle, config, checkpoint_path: str | Path | None = None):
    evaluations = []
    completed_by_index = {}
    if checkpoint_path is not None:
        for record in load_jsonl_records(checkpoint_path):
            if "source_index" in record:
                completed_by_index[int(record["source_index"])] = record

    client, model_name = gemini_model

    for index, item in enumerate(tqdm(responses, desc="Evaluating with Gemini")):
        if index in completed_by_index:
            evaluations.append(completed_by_index[index])
            continue

        try:
            classification = classify_question_context(
                gemini_model,
                prompt_bundle,
                item["question"],
                item["context"],
            )
            class_label = classification.get("classification", "UNKNOWN")

            evaluation_text = None
            evaluation_mode = None

            if class_label == "CLASS_1":
                evaluation_mode = "PROMPT_QA_CLASS1"
                prompt = render_answer_prompt(
                    prompt_bundle["class_1"],
                    item["question"],
                    item["context"],
                    item["generated_answer"],
                )
                evaluation_text = generate_gemini_text(client, model_name, prompt)
            elif class_label == "CLASS_2":
                evaluation_mode = "PROMPT_QA_CLASS2"
                prompt = render_answer_prompt(
                    prompt_bundle["class_2"],
                    item["question"],
                    item["context"],
                    item["generated_answer"],
                )
                evaluation_text = generate_gemini_text(client, model_name, prompt)
            elif class_label == "CLASS_3":
                evaluation_mode = "PROMPT_QA_CLASS3"
                prompt = render_answer_prompt(
                    prompt_bundle["class_3"],
                    item["question"],
                    item["context"],
                    item["generated_answer"],
                )
                evaluation_text = generate_gemini_text(client, model_name, prompt)
            else:
                evaluation_mode = config["metrics"]["fallback_for_class_3"]

            record = {
                **item,
                "source_index": index,
                "judge_classification": classification,
                "judge_classification_label": class_label,
                "judge_evaluation_mode": evaluation_mode,
                "judge_evaluation": evaluation_text,
            }
        except Exception as exc:
            record = judge_error_record(item, index, "evaluate_with_gemini", exc)

        evaluations.append(record)
        if checkpoint_path is not None:
            append_jsonl_record(checkpoint_path, record)

    return evaluations


def load_gemini_model_from_config(config: dict):
    api_key = os.getenv(config["gemini"]["api_key_env"])
    if not api_key:
        raise ValueError(f"Environment variable {config['gemini']['api_key_env']} not set")
    return setup_gemini(api_key, config["gemini"]["model"])


def save_records(records: list[dict], output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    output_dir = ensure_output_dir(output_dir)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8")
    return json_path, csv_path


def load_records(input_path: str | Path) -> list[dict]:
    input_path = resolve_project_path(input_path)
    if input_path.suffix.lower() == ".json":
        return json.loads(input_path.read_text(encoding="utf-8"))
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path).to_dict(orient="records")
    raise ValueError(f"Unsupported input file format: {input_path}")
