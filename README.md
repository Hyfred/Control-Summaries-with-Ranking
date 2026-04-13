# Learning to Control Summaries with Score Ranking

Code for the paper: Learning to Control Summaries with Score Ranking.

## Backbones

- Llama3.1-8B
- Qwen-7B
- Mistral-7B
- GPT-4o

## Main scripts

1. Produce_summary.py
    Generate summaries with the chosen backbone for training and evaluation.

2. Call_finesure.py
    Score summaries with GPT Model (e.g., GPT-4o) for training and evaluation.

3. Control_finetune.py
    Fine-tune the model using the scored data.

Training is an iterative loop: 1 -> 2 -> 3 -> repeat.

4.1 Evaluate_control.ipynb
      Control evaluation and contour plots.

4.2 Evaluate_ranking.py (uses Llama_score.py)
      Ranking evaluation.

## Saved models

- Trained LLaMA model (LLaMA* in the paper): https://huggingface.co/hongyeeliu/Control_Summaries_LLaMA
- Best ranking model, trained Qwen (Qwen* in the paper): https://huggingface.co/hongyeeliu/Control_Summaries_Qwen

See Produce_summary.py for running control inference.