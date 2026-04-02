# Qwen3.5-0.8B-CISPO-RL

**Clipped Importance Sampling Policy Optimization (CISPO) for Qwen3.5-0.8B Reasoning Enhancement.**

This repository contains the implementation of **CISPO RL** training for the **Qwen3.5-0.8B** model, designed to improve reasoning depth, consistency, and logical clarity while strictly avoiding repetitive loops.

## 🚀 Key Features

- **Long Context Reasoning**: Up to **8,192 tokens** generation limit for deep step-by-step thinking.
- **Mixed-Mode Thinking**: Trained on both explicit reasoning (`enable_thinking=True`) and direct answer modes. The model is rewarded for high-quality logic even when reasoning tags are hidden.
- **CISPO Algorithm**: Implements the Clipped Importance Sampling Policy Optimization with **Group-relative Advantage** (Standard for MiniMaxAI's reasoning training).
- **Stricter Anti-Repetition**: 
  - Overlap threshold reduced to **0.15** (4-gram).
  - Repetition penalty weight increased to **0.6** to eliminate reasoning loops.
- **AI Judge Integration**: Uses **Qwen3.5-9B** as a judge (Judge-as-a-Reward) to evaluate correctness and coherence.
- **Cloud Optimized**: Designed for **H100 80GB** environments with efficient batching (`batch_size=1` with `gradient_accumulation=8`).

## 🛠️ Repository Structure

- `train_cispo.py`: Main entry point for RL training.
- `cispo_trainer.py`: Core CISPO implementation (Advantage, Loss, PPO-like Clipping).
- `reward_model.py`: Composite Reward Model (Judge + Repetition + Format + Length).
- `data_utils.py`: Mixed-mode data loader supporting `<think>` and `<tink>` tags.
- `configs/cispo_config.yaml`: Centralized configuration for all parameters.
- `run_cispo.sh`: Bash script to launch training on cloud GPU.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Phonsiriwillbejommarn/Qwen3.5-0.8B-CISPO-RL.git
cd Qwen3.5-0.8B-CISPO-RL

# Install dependencies
pip install -r requirements_cispo.txt
```

## 🏃 Running Training

Ensure you have your HuggingFace and W&B tokens set:

```bash
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_key"

# Grant execute permission and run
chmod +x run_cispo.sh
./run_cispo.sh
```

## ⚙️ Configuration Highlights (`cispo_config.yaml`)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `max_new_tokens` | 8192 | Extended reasoning trace space. |
| `group_size` | 4 | 4 responses per prompt for advantage calculation. |
| `gradient_accumulation_steps` | 8 | Effective batch size of 32 responses. |
| `thinking_ratio` | 0.8 | 80% reasoning / 20% direct answer training mix. |
| `repetition_threshold` | 0.15 | Strict catch for word/concept duplication. |
| `max_think_tokens` | 6144 | Large buffer for complex thinking before penalty. |

## 📊 Monitoring

Training progress is logged via **Weights & Biases**. 
- Monitor `train/mean_reward` and `loss/kl` to ensure policy stability.
- Evaluation samples with thinking traces are logged every `eval_steps`.

## 📄 License
This project is open-source for educational and research purposes.
