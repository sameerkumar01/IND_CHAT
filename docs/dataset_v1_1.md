# GST dataset v1.1

A larger pilot corpus can be downloaded from the session handoff and extracted to `data/processed/v1.1/`.

It contains:

- `train.jsonl`
- `val.jsonl`
- `eval.jsonl`
- `dataset_card.md`

The corpus combines official CBIC GST FAQ pages, the CBIC sectoral FAQ page, the CBIC composition-levy FAQ PDF, and the CBIC GST FAQ PDF. It contains historical material, including 2017 transition questions, so it must not be treated as current tax advice. The intended production architecture is fine-tuning plus retrieval over current official sources.
