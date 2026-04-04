# Qwen3.5-0.8B-CISPO-RL

**Clipped Importance Sampling Policy Optimization (CISPO) with Hybrid GRPO Scaling for Qwen3.5-0.8B Reasoning Enhancement.**

This repository contains the optimized implementation of **CISPO RL** training for the **Qwen3.5-0.8B** model. The pipeline is specifically tuned to eliminate reasoning loops, enforce high-quality step-by-step logic, and maintain stability through advanced advantage scaling and safety nets.

## 🚀 Key Features

- **AI-Driven Reasoning Evaluation**: Replaced rule-based length counting with a **Judge-as-a-Reward** system that evaluates the *quality* of logic inside `<think>` tags.
- **Hybrid GRPO Scaling (Option B)**: Combines **Group Relative Policy Optimization (GRPO)** normalization with token-level process rewards. 
  - `A_final = GRPO_norm * ProcessReturnToGo`
  - Amplifies gradients for best-in-group responses while flipping gradients to penalize tokens in underperforming ones.
- **Safety Net & Bell Curve Reward**:
  - **Bell Curve Length**: Incentivizes reasoning depth at ~512 tokens.
  - **Infinite Loop Protection**: Exponential penalty triggered at **4,096 tokens** to prevent reasoning loops.
- **Dual Anti-Repetition Penalty**: 
  - **Think Mode**: Strict 16-gram monitoring for long-range reasoning loops.
  - **Answer Mode**: Efficient 4-gram monitoring for concise final responses.
- **Stable KL Divergence**: Implements PPO-style proper KL fix (`ratio - 1 - log_ratio`) to ensure penalty is always non-negative and stable.
- **Extended Judge Context**: Judges (Qwen3.5-9B / Gemma-4B) utilize **8,192 tokens** context to see the full reasoning chain and final answer.

## 🛠️ Repository Structure

- `train_cispo.py`: Main entry point for RL training.
- `cispo_trainer.py`: Core CISPO implementation (Hybrid Scaling, Jensen KL Fix, Advantage Clamping).
- `reward_model.py`: Composite Reward Model (AI Judge Ensemble + Dual Repetition + Bell Curve Length).
- `data_utils.py`: Mixed-mode data loader supporting autonomous reasoning templates.
- `configs/cispo_config.yaml`: Centralized configuration for all RL and Reward parameters.
- `run_cispo.sh`: Bash script optimized for H100/H200 with BF16 and TF32 enabled.

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

# Run with optimized defaults
chmod +x run_cispo.sh
./run_cispo.sh
```

## ⚙️ Configuration Highlights (`cispo_config.yaml`)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `group_size` | 6 | Responses per prompt for GRPO/CISPO scaling. |
| `max_think_tokens` | 4096 | Hard ceiling safety net for reasoning chunks. |
| `optimal_think_tokens`| 512 | Target peak for Bell Curve length reward. |
| `adv_clamp` | 10.0 | Prevents gradient explosion from hybrid scaling. |
| `kl_coeff` | 0.005 | Tight constraint to prevent policy drift. |
| `ensemble_strategy` | `weighted_min` | Conservative scoring (70% min + 30% avg). |

## 📊 Monitoring

- **`train/mean_norm_reward`**: Monitor the GRPO-normalized signal (Relative Quality).
- **`loss/kl`**: Ensure it stays positive and stable (Target: < 0.05).
- **`train/clip_fraction`**: Monitor importance sampling stability.

## 📄 License
This project is open-source for educational and research purposes.
