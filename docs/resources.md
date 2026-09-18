# Project resources

These are the primary references for the first GST-focused QLoRA build. Prefer official sources and record the source URL for every training example.

## Domain sources

- [CBIC GST FAQ](https://cbic-gst.gov.in/faq.html)
- [CBIC GST home](https://cbic-gst.gov.in/)
- [CBIC GST rates FAQ](https://cbic-gst.gov.in/gst-rates-faq.html)
- [Official GST portal](https://www.gst.gov.in/)

## Model and fine-tuning

- [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [TRL PEFT / QLoRA integration](https://huggingface.co/docs/trl/en/peft_integration)
- [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora)
- [Transformers bitsandbytes / 4-bit quantization](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)

## Kaggle setup

- [Kaggle notebooks documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle notebooks](https://www.kaggle.com/code)
- [Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Kaggle documentation](https://www.kaggle.com/docs)

In the notebook settings, select a GPU accelerator and enable Internet when you need to install packages or download the base model. Save checkpoints to `/kaggle/working/` and stop the session when idle.

## Suggested Kaggle installation

```python
!pip install -q -U "transformers>=4.44" "datasets>=2.20" "peft>=0.12" "trl>=0.10" "bitsandbytes>=0.43" "accelerate>=0.33" "pyyaml" "pydantic>=2.7"
```

The repo's training config starts with `Qwen/Qwen2.5-7B-Instruct`, NF4 4-bit loading, LoRA rank 16, alpha 32, batch size 1, gradient accumulation 8, and sequence length 2048. Reduce sequence length or accumulation first if the selected Kaggle GPU runs out of memory.
