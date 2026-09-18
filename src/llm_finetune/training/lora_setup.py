"""Model quantization and LoRA configuration helpers."""

from __future__ import annotations

import logging
from typing import Any

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig, PreTrainedModel

LOGGER = logging.getLogger(__name__)


def resolve_compute_dtype(configured_dtype: str) -> torch.dtype:
    """Choose a supported compute dtype for the current accelerator.

    Args:
        configured_dtype: Requested dtype name from YAML.

    Returns:
        A PyTorch dtype supported by the active device.
    """
    requested = getattr(torch, configured_dtype)
    if requested == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        LOGGER.warning("bfloat16 is unavailable on this GPU; falling back to float16")
        return torch.float16
    return requested


def build_quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig:
    """Build the 4-bit NF4 quantization configuration.

    Args:
        config: Parsed `quantization` configuration mapping.

    Returns:
        Hugging Face bitsandbytes configuration.
    """
    compute_dtype = resolve_compute_dtype(str(config.get("compute_dtype", "float16")))
    return BitsAndBytesConfig(
        load_in_4bit=bool(config.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(config.get("quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(config: dict[str, Any]) -> LoraConfig:
    """Build the LoRA adapter configuration.

    Args:
        config: Parsed `lora` configuration mapping.

    Returns:
        PEFT LoRA configuration.
    """
    return LoraConfig(
        r=int(config.get("rank", 16)),
        lora_alpha=int(config.get("alpha", 32)),
        lora_dropout=float(config.get("dropout", 0.05)),
        target_modules=list(config.get("target_modules", [])),
        bias="none",
        task_type="CAUSAL_LM",
    )


def validate_target_modules(model: PreTrainedModel, target_modules: list[str]) -> None:
    """Fail fast when configured LoRA targets are absent from the model.

    Args:
        model: Loaded transformer model.
        target_modules: Leaf module names to validate.

    Raises:
        ValueError: If one or more target modules cannot be found.
    """
    available = {name.rsplit(".", maxsplit=1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(target_modules) - available)
    if missing:
        raise ValueError(
            f"LoRA target modules not found: {missing}. "
            f"Available examples include: {sorted(available)[:20]}"
        )
