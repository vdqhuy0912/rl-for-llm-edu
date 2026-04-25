import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

QUESTION_FIELDS = ("question", "input", "prompt", "query")
ANSWER_FIELDS = ("answer", "output", "response", "completion")
CONTEXT_FIELDS = ("reference", "references", "context")
INSUFFICIENT_CONTEXT_FIELDS = ("insufficient_context", "insufficial context")
SPLIT_FILE_CANDIDATES = {
    "train": ("train.jsonl", "train.json"),
    "validation": ("validation.jsonl", "validation.json", "val.jsonl", "val.json"),
    "test": ("test.jsonl", "test.json"),
}
RAW_DATA_ROOT = Path("data/raw")
SPLITS_DATA_ROOT = Path("data/splits")
HF_DATASETS_CACHE = RAW_DATA_ROOT / "_hf_cache"


def _first_present_value(example: Dict[str, Any], candidates: Iterable[str], default: Any = "") -> Any:
    for field in candidates:
        if field in example and example[field] is not None:
            return example[field]
    return default


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def normalize_qa_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the slightly different schemas used across the project datasets."""
    question = _stringify(_first_present_value(example, QUESTION_FIELDS))
    answer = _stringify(_first_present_value(example, ANSWER_FIELDS))
    context = _stringify(_first_present_value(example, CONTEXT_FIELDS))
    insufficient_context = _as_bool(
        _first_present_value(example, INSUFFICIENT_CONTEXT_FIELDS, default=False)
    )
    multi_intent = _as_bool(example.get("multi_intent", False))

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "insufficient_context": insufficient_context,
        "multi_intent": multi_intent,
    }


def build_instruction_prompt(
    question: str,
    context: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Build a stable prompt format shared by SFT, KTO and evaluation."""
    prompt = ""
    if system_prompt:
        prompt += "### Chỉ dẫn hệ thống\n" + system_prompt.strip() + "\n\n"
    prompt += "### Câu hỏi\n" + question.strip()
    if context:
        prompt += "\n\n### Ngữ cảnh tham chiếu\n" + context.strip()
    prompt += "\n\n### Trả lời\n"
    return prompt


def dataset_name_to_local_dir(dataset_name: str, root: Path = RAW_DATA_ROOT) -> Path:
    return root / dataset_name.replace("/", "_")


def _resolve_raw_split_file(raw_dir: Path, split: str) -> Path:
    for candidate in SPLIT_FILE_CANDIDATES.get(split, ()):
        path = raw_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No raw file found for split '{split}' in {raw_dir}")


def _load_dataset_from_raw_files(raw_dir: Path, split: str = "train") -> Dataset:
    split_file = _resolve_raw_split_file(raw_dir, split)
    rows = []
    with split_file.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(normalize_qa_example(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {split_file} at line {line_no}: {exc}") from exc
    return Dataset.from_list(rows)


def load_local_dataset(dataset_name: str, split: str = "train", root: Path = RAW_DATA_ROOT) -> Dataset:
    """Load one split from local disk, regardless of whether the dataset was saved as Dataset or DatasetDict."""
    dataset_dir = dataset_name_to_local_dir(dataset_name, root=root)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Local dataset directory not found: {dataset_dir}")

    try:
        loaded = load_from_disk(str(dataset_dir))
    except Exception:
        raw_dir = dataset_dir / "raw_files"
        if raw_dir.exists():
            return _load_dataset_from_raw_files(raw_dir, split=split)
        raise

    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise KeyError(f"Split '{split}' not found in local dataset {dataset_name}. Available: {list(loaded.keys())}")
        return loaded[split]
    if split != "train":
        raise KeyError(f"Local dataset {dataset_name} only contains one split; requested '{split}'.")
    return loaded


def save_processed_local_dataset(dataset_name: str, output_root: Path = RAW_DATA_ROOT) -> Path:
    """Normalize a raw-files-only dataset into a Hugging Face disk dataset for stable reuse."""
    dataset_dir = dataset_name_to_local_dir(dataset_name, root=output_root)
    raw_dir = dataset_dir / "raw_files"
    if not raw_dir.exists():
        return dataset_dir

    split_map = {}
    for split in SPLIT_FILE_CANDIDATES:
        try:
            split_map[split] = _load_dataset_from_raw_files(raw_dir, split=split)
        except FileNotFoundError:
            continue
    if not split_map:
        raise FileNotFoundError(f"No recognized raw split files found in {raw_dir}")

    dataset = DatasetDict(split_map)
    temp_dir = dataset_dir / "_normalized_tmp"
    if temp_dir.exists():
        import shutil

        shutil.rmtree(temp_dir)

    dataset.save_to_disk(str(temp_dir))

    for file in temp_dir.iterdir():
        target = dataset_dir / file.name
        if target.exists():
            if target.is_dir():
                import shutil

                shutil.rmtree(target)
            else:
                target.unlink()
        file.replace(target)

    temp_dir.rmdir()
    return dataset_dir


def load_project_dataset(dataset_name: str, split: str = "train", prefer_local: bool = True) -> Dataset:
    """Load dataset with local-first behavior and remote fallback."""
    if prefer_local:
        try:
            return load_local_dataset(dataset_name, split=split)
        except Exception:
            pass

    return load_dataset(dataset_name, split=split)


def load_saved_split_dataset(split_name: str, root: Path = SPLITS_DATA_ROOT) -> Dataset:
    split_dir = root / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Saved split dataset not found: {split_dir}")
    return load_from_disk(str(split_dir))


def load_hf_datasets(dataset_names: list, split: str = "train") -> Dataset:
    """Load and concatenate multiple Hugging Face datasets from the same split."""
    datasets = []
    for name in dataset_names:
        try:
            dataset = load_project_dataset(name, split=split, prefer_local=True)
            datasets.append(dataset)
            print(f"Loaded dataset: {name}:{split} with {len(dataset)} samples")
        except Exception as e:
            print(f"Error loading {name}:{split}: {e}")

    if datasets:
        combined_dataset = concatenate_datasets(datasets)
        print(f"Total combined dataset size for split '{split}': {len(combined_dataset)}")
        return combined_dataset
    else:
        raise ValueError("No datasets could be loaded")

def preprocess_sft_data(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    system_prompt: Optional[str] = None,
):
    """Preprocess data for SFT training."""

    def tokenize_function(examples):
        texts = []
        total = len(next(iter(examples.values()))) if examples else 0
        for index in range(total):
            row = {column: values[index] for column, values in examples.items()}
            normalized = normalize_qa_example(row)
            texts.append(
                build_instruction_prompt(
                    normalized["question"],
                    normalized["context"],
                    system_prompt=system_prompt,
                )
                + normalized["answer"]
            )

        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset

def _truncate_for_negative(text: str, max_chars: int = 400) -> str:
    text = _stringify(text)
    if not text:
        return ""
    return text[:max_chars].strip()


def _first_sentence(text: str) -> str:
    text = _stringify(text)
    for separator in [". ", "! ", "? ", "\n"]:
        if separator in text:
            return text.split(separator)[0].strip() + separator.strip()
    return text


def _build_undesirable_completion(normalized: Dict[str, Any]) -> Dict[str, str]:
    """Create a deterministic negative example for KTO from QA data.

    Strategy order:
    1. `partial_answer_for_multi_intent`: keep only the first sentence when the QA is multi-intent.
    2. `raw_context_dump`: answer with a truncated context excerpt instead of a direct answer.
    3. `overconfident_placeholder`: generic overconfident response when the item lacks sufficient context.
    4. `generic_non_answer`: safe fallback when none of the above apply.
    """
    answer = normalized["answer"]
    context = normalized["context"]

    if normalized["multi_intent"]:
        first_sentence = _first_sentence(answer)
        if first_sentence and first_sentence != answer:
            return {
                "completion": first_sentence,
                "strategy": "partial_answer_for_multi_intent",
            }

    context_excerpt = _truncate_for_negative(context)
    if context_excerpt and context_excerpt != answer:
        return {
            "completion": context_excerpt,
            "strategy": "raw_context_dump",
        }

    if normalized["insufficient_context"]:
        return {
            "completion": (
                "Thong tin hien tai da du de ket luan ngay. Ban cu lam theo cach pho bien "
                "va khong can doi them xac nhan tu nha truong."
            ),
            "strategy": "overconfident_placeholder",
        }

    return {
        "completion": (
            "Ban nen tham khao them thong bao cua truong vi minh chua the tra loi cu the ngay luc nay."
        ),
        "strategy": "generic_non_answer",
    }


def prepare_kto_data(
    dataset: Dataset,
    tokenizer: Optional[AutoTokenizer] = None,
    system_prompt: Optional[str] = None,
) -> Dataset:
    """Prepare KTO data by converting QA samples into positive/negative preference rows."""
    kto_data = []

    for example in dataset:
        normalized = normalize_qa_example(example)
        if not normalized["question"] or not normalized["answer"]:
            continue

        prompt = build_instruction_prompt(
            normalized["question"],
            normalized["context"],
            system_prompt=system_prompt,
        )
        negative = _build_undesirable_completion(normalized)

        kto_data.append(
            {
                "prompt": prompt,
                "completion": normalized["answer"],
                "label": True,
                "source_question": normalized["question"],
                "source_context": normalized["context"],
                "conversion_strategy": "gold_answer",
                "insufficient_context": normalized["insufficient_context"],
                "multi_intent": normalized["multi_intent"],
            }
        )
        kto_data.append(
            {
                "prompt": prompt,
                "completion": negative["completion"],
                "label": False,
                "source_question": normalized["question"],
                "source_context": normalized["context"],
                "conversion_strategy": negative["strategy"],
                "insufficient_context": normalized["insufficient_context"],
                "multi_intent": normalized["multi_intent"],
            }
        )

    return Dataset.from_list(kto_data)
