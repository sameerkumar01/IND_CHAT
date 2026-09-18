"""Conversion helpers for instruction-tuning records."""

from __future__ import annotations

from typing import Any

from .validator import InstructionExample


def to_chat_record(example: InstructionExample) -> dict[str, Any]:
    """Convert an instruction example to a chat-format record.

    Args:
        example: Validated instruction example.

    Returns:
        A JSON-serializable chat record compatible with instruct-model templates.
    """
    user_content = example.instruction
    if example.input:
        user_content = f"{user_content}\n\nContext:\n{example.input}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example.output},
        ],
        "topic": example.topic,
        "source": example.source,
    }
