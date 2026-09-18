"""Tests for deterministic split and leakage checks."""

from llm_finetune.data.splitter import split_examples
from llm_finetune.data.validator import InstructionExample


_TOPICS = [
    "registration", "composition", "invoice", "returns", "refund", "ecommerce",
    "export", "import", "reverse-charge", "input-credit", "exemption", "place-of-supply",
    "interstate", "intrastate", "assessment", "appeal", "notice", "penalty", "interest", "records",
]


def make_examples(count: int) -> list[InstructionExample]:
    return [
        InstructionExample(
            instruction=f"Explain the GST rule for {topic}.",
            output=f"A sufficiently long answer explaining the {topic} rule and its compliance procedure.",
        )
        for topic in _TOPICS[:count]
    ]


def test_split_is_deterministic_and_disjoint() -> None:
    first = split_examples(make_examples(20), seed=7)
    second = split_examples(make_examples(20), seed=7)
    assert first.train == second.train
    train_prompts = {item.instruction for item in first.train}
    validation_prompts = {item.instruction for item in first.validation}
    assert not train_prompts & validation_prompts
    assert len(first.all_examples) == 20
