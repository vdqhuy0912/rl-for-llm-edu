"""Shared helpers for model inference and Gemini judging."""

import copy
import json
import os
import re
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


def classify_question_context(gemini_model, prompts: dict, question: str, context: str) -> dict:
    client, model_name = gemini_model
    prompt = render_classifier_prompt(prompts["classifier"], question, context)
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = (response.text or "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def generate_responses(model, tokenizer, dataset, config):
    system_prompt = config.get("prompt", {}).get("system_prompt")
    responses = []
    num_samples = min(config["evaluation"]["num_samples"], len(dataset))
    batch_size = max(1, int(config["evaluation"].get("batch_size", 1)))
    model.eval()

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_length = None
    generation_config.max_new_tokens = config["evaluation"]["max_new_tokens"]
    generation_config.do_sample = config["evaluation"]["do_sample"]
    generation_config.pad_token_id = tokenizer.eos_token_id
    if generation_config.do_sample:
        generation_config.temperature = config["evaluation"]["temperature"]
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
                input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

                for example, generated_ids, input_length in zip(batch_examples, output_ids, input_lengths):
                    generated_tokens = generated_ids[int(input_length):]
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


def evaluate_with_gemini(gemini_model, responses, prompt_bundle, config):
    evaluations = []
    client, model_name = gemini_model

    for item in tqdm(responses, desc="Evaluating with Gemini"):
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
            evaluation_text = client.models.generate_content(model=model_name, contents=prompt).text or ""
        elif class_label == "CLASS_2":
            evaluation_mode = "PROMPT_QA_CLASS2"
            prompt = render_answer_prompt(
                prompt_bundle["class_2"],
                item["question"],
                item["context"],
                item["generated_answer"],
            )
            evaluation_text = client.models.generate_content(model=model_name, contents=prompt).text or ""
        elif class_label == "CLASS_3":
            evaluation_mode = "PROMPT_QA_CLASS3"
            prompt = render_answer_prompt(
                prompt_bundle["class_3"],
                item["question"],
                item["context"],
                item["generated_answer"],
            )
            evaluation_text = client.models.generate_content(model=model_name, contents=prompt).text or ""
        else:
            evaluation_mode = config["metrics"]["fallback_for_class_3"]

        evaluations.append(
            {
                **item,
                "judge_classification": classification,
                "judge_classification_label": class_label,
                "judge_evaluation_mode": evaluation_mode,
                "judge_evaluation": evaluation_text,
            }
        )

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
