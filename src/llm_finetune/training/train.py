"""QLoRA supervised fine-tuning entry point."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from trl import SFTTrainer
try:
    from trl import SFTConfig
except ImportError:  # pragma: no cover - compatibility with older TRL releases
    SFTConfig = None

from llm_finetune.training.callbacks import FiniteLossCallback
from llm_finetune.training.lora_setup import (
    build_lora_config,
    build_quantization_config,
    validate_target_modules,
)
from llm_finetune.utils.config_loader import load_yaml_config
from llm_finetune.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def require_gpu() -> None:
    """Fail fast when CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Use --allow-cpu only for non-training tests.")


def format_chat_record(record: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Render a messages record using the model's chat template."""
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each record must contain a non-empty messages list")
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}


def prepare_dataset(path: str, tokenizer: Any) -> Dataset:
    """Load and render a JSONL chat dataset."""
    dataset = load_dataset("json", data_files=path, split="train")
    return dataset.map(lambda record: format_chat_record(record, tokenizer), remove_columns=dataset.column_names)


def find_last_checkpoint(output_dir: Path) -> str | None:
    """Find the newest Hugging Face checkpoint directory."""
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda item: int(item.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def build_training_arguments(config: dict[str, Any], output_dir: Path) -> TrainingArguments:
    """Build version-compatible training arguments.

    Newer TRL releases default to a chunked cross-entropy patch that is
    incompatible with some Qwen forward wrappers. We explicitly select the
    mathematically equivalent standard NLL path when the option is available.
    """
    training = config["training"]
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    arguments = {
        "output_dir": str(output_dir),
        "learning_rate": float(training["learning_rate"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "num_train_epochs": float(training["num_train_epochs"]),
        "logging_steps": int(training["logging_steps"]),
        "save_steps": int(training["save_steps"]),
        "save_strategy": "steps",
        "eval_steps": int(training["save_steps"]),
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "report_to": ["wandb"] if os.getenv("WANDB_API_KEY") else [],
        "remove_unused_columns": False,
        "seed": int(config.get("seed", 42)),
    }
    training_argument_names = inspect.signature(TrainingArguments.__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in training_argument_names else "evaluation_strategy"
    arguments[evaluation_key] = "steps"
    if SFTConfig is not None:
        sft_argument_names = inspect.signature(SFTConfig.__init__).parameters
        if "loss_type" in sft_argument_names:
            arguments["loss_type"] = "nll"
        if "max_length" in sft_argument_names:
            arguments["max_length"] = int(training["max_seq_length"])
        return SFTConfig(**arguments)
    return TrainingArguments(**arguments)


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    args: TrainingArguments,
    lora_config: Any,
    max_seq_length: int,
) -> SFTTrainer:
    """Build an SFTTrainer across supported TRL releases."""
    signature = inspect.signature(SFTTrainer.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "args": args,
        "peft_config": lora_config,
        "dataset_text_field": "text",
        "max_seq_length": max_seq_length,
        "callbacks": [FiniteLossCallback()],
    }
    if "processing_class" in signature:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    kwargs = {key: value for key, value in kwargs.items() if key in signature}
    return SFTTrainer(**kwargs)


def train(config_path: str) -> None:
    """Run a resumable QLoRA training job.

    Args:
        config_path: Path to the YAML training configuration.

    Raises:
        RuntimeError: If CUDA is unavailable or training runs out of memory.
    """
    config = load_yaml_config(config_path)
    require_gpu()
    set_seed(int(config.get("seed", 42)))
    training = config["training"]
    output_dir = Path(str(training["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)
    if tokenizer.pad_token is None:
        LOGGER.warning("Tokenizer has no pad token; using EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=build_quantization_config(config["quantization"]),
        device_map="auto",
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    model = prepare_model_for_kbit_training(model)
    validate_target_modules(model, list(config["lora"]["target_modules"]))

    processed_dir = Path("data/processed/v1.0")
    train_dataset = prepare_dataset(str(processed_dir / "train.jsonl"), tokenizer)
    validation_dataset = prepare_dataset(str(processed_dir / "val.jsonl"), tokenizer)
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation datasets must both contain records")

    trainer = build_trainer(
        model,
        tokenizer,
        train_dataset,
        validation_dataset,
        build_training_arguments(config, output_dir),
        build_lora_config(config["lora"]),
        int(training["max_seq_length"]),
    )
    checkpoint = find_last_checkpoint(output_dir)
    LOGGER.info("Starting training with %s records; resume=%s", len(train_dataset), checkpoint)
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(training["final_adapter_dir"]))
        tokenizer.save_pretrained(str(training["final_adapter_dir"]))
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            "CUDA out of memory. Reduce max_seq_length or per-device batch size, "
            "or increase gradient accumulation."
        ) from error


def main() -> None:
    """Parse CLI arguments and start training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training_config.yaml")
    args = parser.parse_args()
    configure_logging()
    train(args.config)


if __name__ == "__main__":
    main()
