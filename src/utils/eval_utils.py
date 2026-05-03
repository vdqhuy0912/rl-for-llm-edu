"""Shared helpers for model inference and Gemini judging."""

import json
import os
import re
from pathlib import Path

import google.generativeai as genai
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from src.utils.data_utils import build_instruction_prompt, normalize_qa_example
from src.utils.model_utils import ensure_output_dir, resolve_project_path


CLASSIFIER_MARKER = 'PROMPT_QA_CLASSIFIER = """'
CLASS1_MARKER = 'PROMPT_QA_CLASS1 = """'
CLASS2_MARKER = 'PROMPT_QA_CLASS2 = """'
CLASS3_MARKER = 'PROMPT_QA_CLASS3 = """'


def setup_gemini(api_key: str, model_name: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


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
    prompt = render_classifier_prompt(prompts["classifier"], question, context)
    response = gemini_model.generate_content(prompt)
    text = response.text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def generate_responses(model, tokenizer, dataset, config):
    system_prompt = config.get("prompt", {}).get("system_prompt")
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config["evaluation"]["max_new_tokens"],
        temperature=config["evaluation"]["temperature"],
        do_sample=config["evaluation"]["do_sample"],
        pad_token_id=tokenizer.eos_token_id,
    )

    responses = []
    num_samples = min(config["evaluation"]["num_samples"], len(dataset))

    for index in tqdm(range(num_samples), desc="Generating responses"):
        example = normalize_qa_example(dataset[index])
        prompt = build_instruction_prompt(
            example["question"],
            example["context"],
            system_prompt=system_prompt,
        )

        try:
            generated = generator(prompt)[0]["generated_text"]
            response = generated[len(prompt):].strip()
        except Exception as exc:
            response = f"Error generating response: {exc}"

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

    return responses


def evaluate_with_gemini(gemini_model, responses, prompt_bundle, config):
    evaluations = []

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
            evaluation_text = gemini_model.generate_content(prompt).text
        elif class_label == "CLASS_2":
            evaluation_mode = "PROMPT_QA_CLASS2"
            prompt = render_answer_prompt(
                prompt_bundle["class_2"],
                item["question"],
                item["context"],
                item["generated_answer"],
            )
            evaluation_text = gemini_model.generate_content(prompt).text
        elif class_label == "CLASS_3":
            evaluation_mode = "PROMPT_QA_CLASS3"
            prompt = render_answer_prompt(
                prompt_bundle["class_3"],
                item["question"],
                item["context"],
                item["generated_answer"],
            )
            evaluation_text = gemini_model.generate_content(prompt).text
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
