"""Prepare validated chat-format datasets from structured JSONL examples."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from llm_finetune.data.cleaner import clean_example
from llm_finetune.data.formatter import to_chat_record
from llm_finetune.data.splitter import split_examples
from llm_finetune.data.validator import InstructionExample
from llm_finetune.utils.config_loader import load_yaml_config
from llm_finetune.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def load_examples(input_dir: Path) -> tuple[list[InstructionExample], int]:
    """Load structured JSONL examples, skipping malformed records.

    Args:
        input_dir: Directory containing JSONL files.

    Returns:
        Valid examples and count of rejected records.
    """
    examples: list[InstructionExample] = []
    rejected = 0
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    raw: dict[str, Any] = json.loads(line)
                    example = InstructionExample.model_validate(raw)
                    cleaned, redactions = clean_example(example)
                    if redactions:
                        LOGGER.warning("Redacted PII in %s:%s", path, line_number)
                    examples.append(cleaned)
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    rejected += 1
                    LOGGER.error("Rejected %s:%s: %s", path, line_number, error)
    return examples, rejected


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSON records as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run data validation, deduplication, splitting, and reporting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--input-dir", default="data/interim")
    args = parser.parse_args()
    configure_logging()
    config = load_yaml_config(args.config)
    examples, rejected = load_examples(Path(args.input_dir))
    split_config = config["split"]
    validation_config = config["validation"]
    result = split_examples(
        examples,
        train_ratio=float(split_config["train"]),
        validation_ratio=float(split_config["validation"]),
        seed=int(config["seed"]),
        near_duplicate_threshold=float(validation_config["near_duplicate_threshold"]),
    )
    output_dir = Path(config["source_dirs"]["processed"])
    write_jsonl(output_dir / "train.jsonl", [to_chat_record(item) for item in result.train])
    write_jsonl(output_dir / "val.jsonl", [to_chat_record(item) for item in result.validation])
    write_jsonl(output_dir / "eval.jsonl", [to_chat_record(item) for item in result.evaluation])
    card = output_dir / "dataset_card.md"
    card.write_text(
        "# Dataset card\n\n"
        f"- Domain: `{config['domain']}`\n"
        f"- Version: `{config['data_version']}`\n"
        f"- Retained examples: `{len(result.all_examples)}`\n"
        f"- Rejected examples: `{rejected}`\n"
        f"- Near-duplicates removed: `{result.duplicate_count}`\n"
        f"- Topic counts: `{result.topic_counts}`\n\n"
        "This dataset must be reviewed for source licensing, factuality, and legal/tax safety before training.",
        encoding="utf-8",
    )
    LOGGER.info("Prepared %s examples; rejected %s", len(result.all_examples), rejected)


if __name__ == "__main__":
    main()
