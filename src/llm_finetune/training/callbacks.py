"""Training callbacks for safety checks."""

from __future__ import annotations

import math
import logging
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

LOGGER = logging.getLogger(__name__)


class FiniteLossCallback(TrainerCallback):
    """Stop training when a non-finite loss is observed."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        """Inspect logged loss values and request an immediate stop."""
        if logs and "loss" in logs and not math.isfinite(float(logs["loss"])):
            LOGGER.error("Non-finite loss detected at step %s; stopping training", state.global_step)
            control.should_training_stop = True
        return control
