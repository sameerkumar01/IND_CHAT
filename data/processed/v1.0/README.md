# Processed dataset v1.0

This directory is reserved for generated `train.jsonl`, `val.jsonl`, `eval.jsonl`, and `dataset_card.md` outputs.

Input records should be UTF-8 JSONL with at least:

```json
{"instruction":"Question", "input":"Optional context", "output":"A sufficiently detailed answer.", "topic":"registration", "source":"https://example.gov.in"}
```

Run `python scripts/prepare_data.py --config config/data_config.yaml` after placing structured examples in `data/interim/`.
