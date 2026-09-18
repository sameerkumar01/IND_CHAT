"""Base-versus-adapter evaluation runner."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from llm_finetune.evaluation.metrics import exact_match, rouge_l, token_f1
from llm_finetune.training.lora_setup import build_quantization_config
from llm_finetune.utils.config_loader import load_yaml_config
from llm_finetune.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load chat-format evaluation records from JSONL."""
    with path.open(encoding="utf-8") as source_file:
        return [json.loads(line) for line in source_file if line.strip()]


def generate_answer(model: Any, tokenizer: Any, messages: list[dict[str, str]], config: dict[str, Any]) -> str:
    """Generate one answer from a loaded model."""
    encoded = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    generation = config["generation"]
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=int(generation["max_new_tokens"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            do_sample=float(generation["temperature"]) > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_predictions(predictions: list[dict[str, str]]) -> dict[str, float]:
    """Calculate aggregate metrics for prediction records."""
    if not predictions:
        return {"exact_match": 0.0, "token_f1": 0.0, "rougeL": 0.0}
    return {
        "exact_match": sum(exact_match(item["prediction"], item["reference"]) for item in predictions) / len(predictions),
        "token_f1": sum(token_f1(item["prediction"], item["reference"]) for item in predictions) / len(predictions),
        "rougeL": sum(rouge_l(item["prediction"], item["reference"]) for item in predictions) / len(predictions),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate(config_path: str) -> None:
    """Evaluate the base model and trained adapter on the held-out set."""
    config = load_yaml_config(config_path)
    set_seed(int(config.get("seed", 42)))
    model_name = config["model_name"]
    eval_path = Path(config["eval_dataset"])
    output_dir = Path(config["output_dir"])
    adapter_dir = Path(config["adapter_dir"])
    records = load_records(eval_path)
    if not records:
        raise ValueError(f"Evaluation dataset is empty: {eval_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=build_quantization_config(config["quantization"]),
        device_map="auto",
    )
    model.eval()

    base_predictions: list[dict[str, str]] = []
    for record in records:
        messages = record["messages"]
        reference = messages[-1]["content"]
        prompt = messages[0]["content"]
        base_predictions.append({
            "prompt": prompt,
            "reference": reference,
            "prediction": generate_answer(model, tokenizer, messages, config),
        })

    adapter_model = PeftModel.from_pretrained(model, str(adapter_dir))
    adapter_model.eval()
    finetuned_predictions: list[dict[str, str]] = []
    for record in records:
        messages = record["messages"]
        finetuned_predictions.append({
            "prompt": messages[0]["content"],
            "reference": messages[-1]["content"],
            "prediction": generate_answer(adapter_model, tokenizer, messages, config),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "base_predictions.jsonl", base_predictions)
    write_jsonl(output_dir / "finetuned_predictions.jsonl", finetuned_predictions)
    metrics = {
        "base": score_predictions(base_predictions),
        "finetuned": score_predictions(finetuned_predictions),
        "evaluation_examples": len(records),
        "model": model_name,
        "adapter": str(adapter_dir),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "manual_review.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["prompt", "reference", "base_prediction", "finetuned_prediction"])
        writer.writeheader()
        for base, finetuned in zip(base_predictions, finetuned_predictions):
            writer.writerow({
                "prompt": base["prompt"],
                "reference": base["reference"],
                "base_prediction": base["prediction"],
                "finetuned_prediction": finetuned["prediction"],
            })
    LOGGER.info("Evaluation complete: %s", metrics)


def main() -> None:
    """Parse CLI arguments and run evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/eval_config.yaml")
    args = parser.parse_args()
    configure_logging()
    evaluate(args.config)


if __name__ == "__main__":
    main()
