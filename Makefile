.PHONY: setup test lint data train eval serve

setup:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

data:
	python scripts/prepare_data.py --config config/data_config.yaml

train:
	python scripts/run_training.py --config config/training_config.yaml

eval:
	python scripts/run_evaluation.py --config config/eval_config.yaml

serve:
	uvicorn llm_finetune.api.main:app --host 0.0.0.0 --port 8000
