# Architecture

The project is organized as a reproducible pipeline:

1. **Data:** ingest, clean, format, validate, deduplicate, scan for PII, and split.
2. **Training:** load the configured base model in 4-bit mode and train LoRA adapters.
3. **Evaluation:** run base and fine-tuned models against the same held-out set.
4. **Serving:** load the merged artifact once in a FastAPI application and expose `/health` and `/generate`.
5. **Demo:** provide a minimal client UI with a visible non-advice disclaimer.

All stages read YAML configuration and write logs/artifacts to predictable project-relative paths.
