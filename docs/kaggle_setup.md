# Kaggle notebook setup

## 1. Create the notebook

1. Open [Kaggle Notebooks](https://www.kaggle.com/code) and create a new notebook.
2. In **Notebook settings**, select a GPU accelerator such as T4 or P100.
3. Enable **Internet** only when installing packages or downloading the model.
4. Use `/kaggle/working/` for checkpoints and exported artifacts.

## 2. Install dependencies

Run this as the first cell:

```python
!pip install -q -U \
  "transformers>=4.44" \
  "datasets>=2.20" \
  "peft>=0.12" \
  "trl>=0.10" \
  "bitsandbytes>=0.43" \
  "accelerate>=0.33" \
  "pyyaml" \
  "pydantic>=2.7"
```

Then verify the accelerator:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Stop here if CUDA is unavailable; do not start a 7B training run on CPU.

## 3. Bring the repository into the notebook

For a public repository:

```python
!git clone https://github.com/sameerkumar01/IND_CHAT.git /kaggle/working/IND_CHAT
%cd /kaggle/working/IND_CHAT
```

For a private repository, use a Kaggle secret or upload a ZIP. Never put a GitHub token directly in a notebook cell.

## 4. Start with a smoke test

Before a full run, load a tiny sample and verify the tokenizer, chat template, and one forward pass. Keep the first training run small and save checkpoints frequently. Start from the values in `config/training_config.yaml`; if memory is tight, reduce `max_seq_length` to 1024 and keep the per-device batch size at 1.

## 5. Save artifacts

Use `/kaggle/working/checkpoints` during training. Before ending the session, download or publish the adapter artifact and the evaluation outputs. Kaggle working storage is not a substitute for a permanent model registry.

## 6. Safety and reproducibility

- Record the Kaggle GPU type, CUDA version, package versions, seed, and exact config.
- Keep the held-out evaluation set separate from training data.
- Do not include real taxpayer IDs, phone numbers, emails, or other personal data.
- Treat model responses as research/demo output, not tax advice.
