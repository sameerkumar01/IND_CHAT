"""Tests for dataset validation and safety checks."""

from llm_finetune.data.validator import InstructionExample, find_near_duplicate_indices, redact_pii


def test_unicode_example_is_valid() -> None:
    example = InstructionExample(instruction="GST क्या है?", output="यह भारत में वस्तु एवं सेवा कर है।")
    assert "GST" in example.instruction


def test_missing_output_is_rejected() -> None:
    try:
        InstructionExample(instruction="What is GST?", output="short")
    except ValueError as error:
        assert "10 characters" in str(error)
    else:
        raise AssertionError("missing/short output should be rejected")


def test_pii_is_redacted() -> None:
    redacted, records = redact_pii("PAN ABCDE1234F and email test@example.com")
    assert "ABCDE1234F" not in redacted
    assert {record.kind for record in records} == {"pan", "email"}


def test_near_duplicates_keep_first() -> None:
    examples = [
        InstructionExample(instruction="What is GST?", output="GST is a tax on goods and services."),
        InstructionExample(instruction="What is GST? ", output="GST is a tax on goods and services."),
    ]
    assert find_near_duplicate_indices(examples) == {1}
