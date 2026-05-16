#!/usr/bin/env python3
"""Build preconverted KTO datasets from Gemini-judged generated responses."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from src.utils.data_utils import build_instruction_prompt, build_instruction_response_text
from src.utils.eval_utils import load_records, save_records
from src.utils.model_utils import ensure_output_dir, load_config, resolve_project_path, setup_logging


CLASS1_NEGATIVE_EXACT = {
    "faithfulness.context.contradiction",
    "faithfulness.context.baseless",
    "faithfulness.instruction.task.mismatch",
    "helpfulness.responsiveness.refusal",
}
CLASS1_NEGATIVE_PREFIXES = (
    "accuracy.contradiction.",
    "accuracy.fabrication.",
    "accuracy.unverifiable.",
    "safety.harm.",
    "safety.ethics.",
)
CLASS1_DROP_EXACT = {
    "faithfulness.instruction.task.omission",
}

CLASS2_POSITIVE = {
    "behavior.correct.clarify.targeted",
    "behavior.correct.clarify.option_present",
    "behavior.correct.clarify.proactive_slot",
    "behavior.correct.abstain.refuse",
    "behavior.correct.abstain.hedged",
    "behavior.correct.abstain.premise_correct",
    "behavior.correct.multi_interpret",
}
CLASS2_NEGATIVE = {
    "behavior.failure.ignore.silent_assume",
    "behavior.failure.ignore.overconfident",
    "behavior.failure.fabricate.content",
    "behavior.failure.fabricate.constraint",
    "behavior.failure.omit.constraint_drop",
    "behavior.failure.omit.partial_comply",
}
CLASS2_DROP = {
    "behavior.failure.deflect.generic_clarify",
}


def extract_json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def triggered_or_violated_labels(payload: dict[str, Any]) -> list[str]:
    labels = []
    summary = payload.get("summary")
    if isinstance(summary, dict):
        violated = summary.get("violated_labels")
        if isinstance(violated, list):
            labels.extend(str(label) for label in violated)

        triggered = summary.get("triggered_labels")
        if isinstance(triggered, list):
            labels.extend(str(label) for label in triggered)

        triggered_behavior = summary.get("triggered_behavior")
        if triggered_behavior:
            labels.append(str(triggered_behavior))

    for key in ("evaluations", "criteria", "behavior", "ambiguity"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            active = item.get("violated", item.get("triggered", False))
            if active and item.get("label"):
                labels.append(str(item["label"]))

    return sorted(set(labels))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def class1_decision(labels: list[str]) -> tuple[str, str]:
    for label in labels:
        if label in CLASS1_NEGATIVE_EXACT or label.startswith(CLASS1_NEGATIVE_PREFIXES):
            return "negative", f"CLASS_1 severe violation: {label}"

    if any(label in CLASS1_DROP_EXACT for label in labels):
        return "drop", "CLASS_1 has task omission without a hard factual violation"

    if not labels:
        return "positive", "CLASS_1 has no violated labels"

    return "drop", "CLASS_1 only has non-severe violations"


def class2_decision(payload: dict[str, Any], labels: list[str]) -> tuple[str, str]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    context_usage = summary.get("context_usage") if isinstance(summary.get("context_usage"), dict) else {}
    context_rating = str(context_usage.get("rating", "")).upper()

    for label in labels:
        if label in CLASS2_NEGATIVE:
            return "negative", f"CLASS_2 failure behavior: {label}"
    if context_rating == "BAD":
        return "negative", "CLASS_2 context_usage.rating is BAD"

    if any(label in CLASS2_POSITIVE for label in labels):
        return "positive", "CLASS_2 has correct clarification/abstention behavior"

    if any(label in CLASS2_DROP for label in labels):
        return "drop", "CLASS_2 generic clarification is too weak for binary KTO"

    return "drop", "CLASS_2 has no decisive positive or negative behavior"


def decide_binary_label(record: dict[str, Any]) -> tuple[bool | None, str, str, list[str]]:
    class_label = str(record.get("judge_classification_label") or "UNKNOWN")
    if class_label == "CLASS_3":
        return None, "skip_class_3", "CLASS_3 is excluded from KTO training", []
    if class_label not in {"CLASS_1", "CLASS_2"}:
        return None, "drop", f"Unsupported judge class: {class_label}", []

    payload = extract_json_object(record.get("judge_evaluation"))
    if payload is None:
        return None, "drop", "Could not parse judge_evaluation JSON", []

    labels = triggered_or_violated_labels(payload)
    if class_label == "CLASS_1":
        decision, reason = class1_decision(labels)
    else:
        decision, reason = class2_decision(payload, labels)

    if decision == "positive":
        return True, decision, reason, labels
    if decision == "negative":
        return False, decision, reason, labels
    return None, decision, reason, labels


def build_completion(
    question: str,
    context: str,
    answer: str,
    prompt: str,
    tokenizer,
    system_prompt: str | None,
    enable_thinking: bool,
) -> str:
    full_text = build_instruction_response_text(
        question,
        answer,
        context,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
    )
    return full_text[len(prompt):] if full_text.startswith(prompt) else answer.strip()


def convert_records(
    records: list[dict[str, Any]],
    tokenizer,
    system_prompt: str | None,
    enable_thinking: bool,
    include_reference_positive: bool,
    reference_positive_for_negative_only: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    kto_rows = []
    stats = Counter()

    for index, record in enumerate(records):
        class_label = str(record.get("judge_classification_label") or "UNKNOWN")
        label, decision, reason, active_labels = decide_binary_label(record)
        stats[f"class.{class_label}"] += 1
        stats[f"decision.{decision}"] += 1

        if class_label == "CLASS_3":
            continue
        if label is None:
            continue
        if reference_positive_for_negative_only and decision != "negative":
            stats[f"skipped.{decision}_not_negative_pair"] += 1
            continue

        question = str(record.get("question") or "").strip()
        context = str(record.get("context") or "").strip()
        generated_answer = str(record.get("generated_answer") or "").strip()
        reference_answer = str(record.get("reference_answer") or "").strip()
        if not question or not generated_answer:
            stats["skipped.missing_question_or_generated_answer"] += 1
            continue
        if reference_positive_for_negative_only and not reference_answer:
            stats["skipped.missing_reference_answer_for_pair"] += 1
            continue

        prompt = build_instruction_prompt(
            question,
            context,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        completion = build_completion(
            question,
            context,
            generated_answer,
            prompt,
            tokenizer,
            system_prompt,
            enable_thinking,
        )
        kto_rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "label": label,
                "source_question": question,
                "source_context": context,
                "reference_answer": reference_answer,
                "judge_classification_label": class_label,
                "judge_decision": decision,
                "judge_decision_reason": reason,
                "judge_triggered_labels": active_labels,
                "conversion_strategy": "gemini_judged_generated_answer",
                "source_index": index,
                "insufficient_context": as_bool(record.get("insufficient_context", False)),
                "multi_intent": as_bool(record.get("multi_intent", False)),
            }
        )
        stats[f"label.{label}"] += 1

        if (include_reference_positive or reference_positive_for_negative_only) and reference_answer:
            reference_completion = build_completion(
                question,
                context,
                reference_answer,
                prompt,
                tokenizer,
                system_prompt,
                enable_thinking,
            )
            kto_rows.append(
                {
                    "prompt": prompt,
                    "completion": reference_completion,
                    "label": True,
                    "source_question": question,
                    "source_context": context,
                    "reference_answer": reference_answer,
                    "judge_classification_label": class_label,
                    "judge_decision": "reference_positive",
                    "judge_decision_reason": "Gold reference answer added as desirable completion",
                    "judge_triggered_labels": [],
                    "conversion_strategy": "gold_reference_answer",
                    "source_index": index,
                    "insufficient_context": as_bool(record.get("insufficient_context", False)),
                    "multi_intent": as_bool(record.get("multi_intent", False)),
                }
            )
            stats["label.True"] += 1
            stats["decision.reference_positive"] += 1

    return kto_rows, stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="Path to evaluation_results.json or .csv.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the preconverted KTO dataset.")
    parser.add_argument(
        "--json-output",
        help="Optional JSON/CSV preview output stem or path. Defaults to <output-dir>/records/kto_judged_data.",
    )
    parser.add_argument("--tokenizer-path", help="Tokenizer path used to render the same prompt format as training.")
    parser.add_argument(
        "--include-reference-positive",
        action="store_true",
        help="Also add reference_answer as label=True for non-CLASS_3 judged samples.",
    )
    parser.add_argument(
        "--reference-positive-for-negative-only",
        action="store_true",
        help=(
            "Build pair-derived KTO rows only from negative generated answers: "
            "generated_answer label=False plus reference_answer label=True."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(logger_name="data.kto_judged")
    config = load_config("configs/kto_config.yaml")
    prompt_config = config.get("prompt", {})
    system_prompt = prompt_config.get("system_prompt")
    enable_thinking = prompt_config.get("enable_thinking", False)

    tokenizer_path = args.tokenizer_path or config["model"]["base_model_path"]
    resolved_tokenizer_path = resolve_project_path(tokenizer_path)
    tokenizer_source = resolved_tokenizer_path if resolved_tokenizer_path.exists() else tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_records(args.input_path)
    kto_rows, stats = convert_records(
        records,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        include_reference_positive=args.include_reference_positive,
        reference_positive_for_negative_only=args.reference_positive_for_negative_only,
    )

    output_dir = ensure_output_dir(args.output_dir)
    dataset = Dataset.from_list(kto_rows)
    dataset.save_to_disk(str(output_dir))

    json_output = args.json_output
    if json_output is None:
        records_dir = ensure_output_dir(output_dir / "records")
        save_records(kto_rows, records_dir, "kto_judged_data")
    else:
        json_path = Path(json_output)
        if json_path.suffix:
            preview_dir = ensure_output_dir(json_path.parent)
            stem = json_path.stem
        else:
            preview_dir = ensure_output_dir(json_path)
            stem = "kto_judged_data"
        save_records(kto_rows, preview_dir, stem)

    stats_payload = dict(sorted(stats.items()))
    (output_dir / "build_stats.json").write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Loaded %s judged records", len(records))
    logger.info("Saved %s KTO rows to %s", len(kto_rows), output_dir)
    logger.info("Stats: %s", stats_payload)
    print(json.dumps({"rows": len(kto_rows), "output_dir": str(output_dir), "stats": stats_payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
