"""Dataset loading utilities for ConvFinQA."""

import json
from functools import lru_cache
from pathlib import Path

from src.models import ConvFinQARecord

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "convfinqa_dataset.json"


@lru_cache(maxsize=1)
def load_dataset(path: Path = DEFAULT_DATA_PATH) -> dict[str, list[ConvFinQARecord]]:
    """Load the ConvFinQA dataset into pydantic records, keyed by split."""
    with open(path) as f:
        raw = json.load(f)

    return {
        split: [ConvFinQARecord.model_validate(item) for item in items]
        for split, items in raw.items()
    }


def get_record(record_id: str, path: Path = DEFAULT_DATA_PATH) -> ConvFinQARecord:
    """Find a record by id across all splits."""
    dataset = load_dataset(path)
    for records in dataset.values():
        for record in records:
            if record.id == record_id:
                return record
    raise KeyError(f"No record found with id={record_id!r}")


def iter_records(
    split: str, path: Path = DEFAULT_DATA_PATH
) -> list[ConvFinQARecord]:
    """Return all records for a given split (e.g. 'train', 'dev')."""
    dataset = load_dataset(path)
    if split not in dataset:
        raise KeyError(f"Unknown split {split!r}, available: {list(dataset)}")
    return dataset[split]
