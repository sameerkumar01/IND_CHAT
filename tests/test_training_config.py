"""Tests for the training configuration contract."""

from pathlib import Path

import yaml


def test_training_config_contains_qlora_requirements() -> None:
    config = yaml.safe_load(Path("config/training_config.yaml").read_text(encoding="utf-8"))
    assert config["quantization"]["load_in_4bit"] is True
    assert config["quantization"]["quant_type"] == "nf4"
    assert config["lora"]["rank"] > 0
    assert config["training"]["per_device_train_batch_size"] == 1
