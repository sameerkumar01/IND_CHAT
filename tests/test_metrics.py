"""Tests for evaluation metrics."""

from llm_finetune.evaluation.metrics import exact_match, rouge_l, token_f1


def test_exact_match_normalizes_whitespace_and_case() -> None:
    assert exact_match(" GST  registration ", "gst registration") == 1.0


def test_metrics_reward_shared_content() -> None:
    assert token_f1("GST registration is required", "GST registration is required") == 1.0
    assert rouge_l("GST registration is required", "GST registration is required") == 1.0
    assert token_f1("unrelated answer", "GST registration") == 0.0
