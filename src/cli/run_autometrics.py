#!/usr/bin/env python3
"""Compute automatic QA metrics for generated model responses."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.eval_utils import load_records
from src.utils.model_utils import ensure_output_dir, resolve_project_path, setup_logging


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index_b - 1] + 1)
            else:
                current.append(max(previous[index_b], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1)))


def sentence_bleu(prediction: str, reference: str, max_order: int = 4, smooth: float = 1.0) -> float:
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for order in range(1, max_order + 1):
        pred_ngrams = ngrams(pred_tokens, order)
        ref_ngrams = ngrams(ref_tokens, order)
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        precisions.append((overlap + smooth) / (total + smooth))

    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens)))

    return brevity_penalty * math.exp(sum(math.log(score) for score in precisions) / max_order)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    vectors = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding PhoBERT texts"):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            vectors.append(F.normalize(embeddings.detach().cpu(), p=2, dim=1))
    return torch.cat(vectors, dim=0) if vectors else torch.empty(0, 1)


def compute_semantic_similarity(
    predictions: list[str],
    references: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    pred_vectors = encode_texts(predictions, tokenizer, model, device, batch_size, max_length)
    ref_vectors = encode_texts(references, tokenizer, model, device, batch_size, max_length)
    return (pred_vectors * ref_vectors).sum(dim=1).tolist()


def summarize(values: Iterable[float]) -> dict[str, float]:
    series = pd.Series(list(values), dtype="float64")
    if series.empty:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=0)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="Path to generated_responses.json or .csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics outputs.")
    parser.add_argument("--model-label", default="model", help="Short model label to store in outputs.")
    parser.add_argument("--semantic-model", default="vinai/phobert-base-v2")
    parser.add_argument("--semantic-batch-size", type=int, default=16)
    parser.add_argument("--semantic-max-length", type=int, default=256)
    parser.add_argument("--skip-semantic", action="store_true", help="Skip PhoBERT semantic similarity.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(logger_name="metrics.auto")
    records = load_records(args.input_path)
    output_dir = ensure_output_dir(args.output_dir)

    rows = []
    predictions = []
    references = []
    for index, record in enumerate(records):
        prediction = str(record.get("generated_answer") or "")
        reference = str(record.get("reference_answer") or record.get("answer") or "")
        predictions.append(prediction)
        references.append(reference)
        rows.append(
            {
                "model": args.model_label,
                "index": index,
                "token_f1": token_f1(prediction, reference),
                "rouge_l": rouge_l_f1(prediction, reference),
                "bleu": sentence_bleu(prediction, reference),
                "prediction_chars": len(prediction),
                "reference_chars": len(reference),
                "question": record.get("question", ""),
            }
        )

    if not args.skip_semantic:
        logger.info("Computing PhoBERT semantic similarity with %s", args.semantic_model)
        similarities = compute_semantic_similarity(
            predictions,
            references,
            model_name=args.semantic_model,
            batch_size=args.semantic_batch_size,
            max_length=args.semantic_max_length,
        )
        for row, similarity in zip(rows, similarities):
            row["phobert_semantic_similarity"] = float(similarity)

    df = pd.DataFrame(rows)
    per_sample_path = output_dir / "autometrics_per_sample.csv"
    df.to_csv(per_sample_path, index=False, encoding="utf-8")

    metric_columns = ["token_f1", "rouge_l", "bleu"]
    if "phobert_semantic_similarity" in df:
        metric_columns.append("phobert_semantic_similarity")

    summary = {
        "model": args.model_label,
        "input_path": str(resolve_project_path(args.input_path)),
        "num_samples": int(len(df)),
        "metrics": {metric: summarize(df[metric].dropna()) for metric in metric_columns},
    }
    summary_path = output_dir / "autometrics_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.json_normalize(summary, sep=".").to_csv(output_dir / "autometrics_summary.csv", index=False, encoding="utf-8")

    logger.info("Saved automatic metrics to %s", output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
