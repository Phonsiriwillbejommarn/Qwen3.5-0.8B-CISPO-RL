"""
train_cispo.py
──────────────
Main training script สำหรับ CISPO RL

Usage:
    python train_cispo.py --config configs/cispo_config.yaml

หรือ override config ผ่าน CLI:
    python train_cispo.py --config configs/cispo_config.yaml \\
        --max_steps 500 --learning_rate 1e-6
"""

from __future__ import annotations

import argparse
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from huggingface_hub import HfApi, create_repo
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from cispo_trainer import CISPOTrainer, PromptDataset
from data_utils import collate_prompts, load_rl_dataset
from reward_model import CISPORewardModel, RewardConfig


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────
def load_config(config_path: str, overrides: Dict) -> Dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CISPO RL Training for Qwen3.5-0.8B")
    parser.add_argument("--config", type=str, default="configs/cispo_config.yaml")
    parser.add_argument("--policy_model_name", type=str, default=None)
    parser.add_argument("--judge_model_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_format", type=str, default=None)
    parser.add_argument("--custom_data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--epsilon_high", type=float, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", default=False)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    cfg = load_config(args.config, overrides)

    if args.no_wandb:
        cfg["use_wandb"] = False

    set_seed(cfg.get("seed", 42))

    # ─── Output dir ───
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.dump(cfg))
    print(f"[Setup] Output directory: {output_dir}")

    # ─── WandB ───
    if cfg.get("use_wandb", False):
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "cispo-rl"),
            name=cfg.get("run_name", "cispo-run"),
            config=cfg,
        )

    # ─── Resume Checkpoint Logic ───
    resume_path = cfg.get("resume_from_checkpoint")
    if not resume_path:
        # Auto-detect latest checkpoint in output_dir
        resume_path = find_latest_checkpoint(output_dir)

    start_step = 1
    policy_model_path = cfg["policy_model_name"]
    
    if resume_path and Path(resume_path).exists():
        policy_model_path = resume_path # โมเดลจะโหลดจาก checkpoint

    # ─── Policy Model ───
    print(f"\n[Policy] Loading: {policy_model_path}")
    try:
        policy_tokenizer = AutoTokenizer.from_pretrained(
            policy_model_path,
            trust_remote_code=True,
        )
    except Exception:
        # แก้ไข ValueError: Tokenizer class TokenizersBackend does not exist
        print(f"[Warning] Tokenizer Error detected. Falling back to official Qwen3.5-0.8B tokenizer.")
        policy_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-0.8B",
            trust_remote_code=True,
        )

    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_path,
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        device_map="cuda:0",        # Policy model บน GPU 0
        trust_remote_code=True,
        attn_implementation="sdpa", # ใช้ SDPA
    )
    policy_model.train()

    # Enable Gradient Checkpointing ถ้าตั้งค่าไว้ใน config
    # ช่วยลด Activation Memory สำหรับ Qwen3.5 hybrid linear attention
    if cfg.get("gradient_checkpointing", False):
        policy_model.gradient_checkpointing_enable()
        print(f"[Policy] Gradient Checkpointing enabled ✅")

    # Enable TF32 บน H100
    if cfg.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[Policy] Loaded ✅  |  Parameters: {sum(p.numel() for p in policy_model.parameters()):,}")

    # ─── Judge/Reward Model (Qwen3.5-9B) ───
    reward_cfg = RewardConfig(
        judge_model_name=cfg["judge_model_name"],
        judge_max_new_tokens=cfg.get("judge_max_new_tokens", 128),
        judge_temperature=cfg.get("judge_temperature", 0.1),
        judge_batch_size=cfg.get("judge_batch_size", 8),
        reward_judge_weight=cfg.get("reward_judge_weight", 1.0),
        reward_length_weight=cfg.get("reward_length_weight", 0.1),
        reward_repetition_weight=cfg.get("reward_repetition_weight", 0.3),
        reward_format_weight=cfg.get("reward_format_weight", 0.1),
        repetition_ngram_size=cfg.get("repetition_ngram_size", 4),
        repetition_threshold=cfg.get("repetition_threshold", 0.3),
        max_think_tokens=cfg.get("max_think_tokens", 768),
        max_response_tokens=cfg.get("max_response_tokens", 256),
        dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
    )
    reward_model = CISPORewardModel(reward_cfg, policy_tokenizer)

    # ─── Dataset ───
    print(f"\n[Dataset] Preparing...")
    samples = load_rl_dataset(
        dataset_name=cfg["dataset_name"],
        dataset_format=cfg.get("dataset_format", "raw"),
        tokenizer=policy_tokenizer,
        dataset_split=cfg.get("dataset_split", "train"),
        custom_data_path=cfg.get("custom_data_path"),
        max_samples=cfg.get("max_samples"),
        max_prompt_length=cfg.get("max_prompt_length", 512),
        seed=cfg.get("seed", 42),
        thinking_ratio=cfg.get("thinking_ratio", 0.8),   # Mixed-mode ตาม SFT distribution
    )

    dataset = PromptDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("per_device_train_batch_size", 2),
        shuffle=True,
        num_workers=cfg.get("dataloader_num_workers", 4),
        collate_fn=collate_prompts,
        drop_last=True,
    )

    # ─── Optimizer & Scheduler ───
    optimizer = AdamW(
        policy_model.parameters(),
        lr=cfg.get("learning_rate", 5e-7),
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    max_steps = cfg.get("max_steps", -1)
    num_epochs = cfg.get("num_train_epochs", 3)
    if max_steps > 0:
        total_steps = max_steps
    else:
        total_steps = (len(dataloader) * num_epochs) // cfg.get("gradient_accumulation_steps", 4)

    warmup_steps = cfg.get("warmup_steps", 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ─── CISPO Trainer ───
    trainer = CISPOTrainer(
        policy_model=policy_model,
        policy_tokenizer=policy_tokenizer,
        reward_model=reward_model,
        config=cfg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # ─── Resume from checkpoint ───
    if resume_path and Path(resume_path).exists():
        state_path = Path(resume_path) / "trainer_state.pt"
        if state_path.exists():
            print(f"[Checkpoint] Loading trainer state from: {state_path}")
            state = torch.load(state_path, map_location="cpu")
            
            # โหลดสถานะ Optimizer และ Scheduler
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            
            # ตั้งค่าก้าว (Step) ให้ต่อเนื่อง
            start_step = state.get("global_step", 0)
            trainer.global_step = start_step
            print(f"[Checkpoint] Successfully resumed at step {start_step} ✅")
        else:
            print(f"[Warning] trainer_state.pt not found in {resume_path}. Starting from step 0.")

    # ─── Training Loop ───
    print(f"\n{'='*60}")
    print(f"  CISPO RL Training")
    print(f"  Policy : {cfg['policy_model_name']}")
    print(f"  Judge  : {cfg['judge_model_name']}")
    print(f"  Steps  : {total_steps}  |  Group size: {cfg.get('group_size', 8)}")
    print(f"  ε_high : {cfg.get('epsilon_high', 5.0)}  |  β_kl: {cfg.get('kl_coeff', 0.001)}")
    print(f"{'='*60}\n")

    save_steps = cfg.get("save_steps", 50)
    eval_steps = cfg.get("eval_steps", 25)
    logging_steps = cfg.get("logging_steps", 5)
    save_total_limit = cfg.get("save_total_limit", 3)
    saved_checkpoints = []

    global_step = start_step
    done = False

    for epoch in range(num_epochs):
        if done:
            break

        for batch in dataloader:
            if max_steps > 0 and global_step >= max_steps:
                done = True
                break

            prompts = batch["prompts"]
            formatted_prompts = batch["formatted_prompts"]
            intended_thinking = batch.get("intended_thinking", [True] * len(prompts))

            # ── Train step ──
            metrics = trainer.train_step(prompts, formatted_prompts, intended_thinking)
            global_step = trainer.global_step

            # ── Logging ──
            if global_step % logging_steps == 0:
                log_str = (
                    f"[Step {global_step:4d}] "
                    f"loss={metrics.get('loss/total', 0):.4f} | "
                    f"reward={metrics.get('train/mean_reward', 0):.4f} | "
                    f"clip%={metrics.get('train/clip_fraction', 0)*100:.1f}% | "
                    f"kl={metrics.get('loss/kl', 0):.5f}"
                )
                print(log_str)

                if cfg.get("use_wandb", False):
                    import wandb
                    wandb.log(metrics, step=global_step)

            # ── Eval (log sample generation) ──
            if global_step % eval_steps == 0:
                _log_sample_generation(
                    trainer, policy_tokenizer, samples[:1], cfg, global_step
                )

            # ── Save checkpoint ──
            if global_step % save_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                _save_checkpoint(
                    ckpt_dir, policy_model, policy_tokenizer, optimizer, scheduler, global_step, metrics, cfg
                )
                saved_checkpoints.append(ckpt_dir)

                # Rotate old checkpoints
                if len(saved_checkpoints) > save_total_limit:
                    oldest = saved_checkpoints.pop(0)
                    if oldest.exists():
                        import shutil
                        shutil.rmtree(oldest)
                        print(f"[Checkpoint] Removed old: {oldest}")

    # ─── Final save ───
    final_dir = output_dir / "final_model"
    _save_checkpoint(final_dir, policy_model, policy_tokenizer, optimizer, scheduler, global_step, metrics, cfg)
    print(f"\n[Done] Final model and state saved to: {final_dir}")

    if cfg.get("use_wandb", False):
        import wandb
        wandb.finish()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """สแกนหา checkpoint-* ล่าสุดใน output_dir"""
    if not output_dir.exists():
        return None
    
    checkpoints = []
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                checkpoints.append((step, str(d)))
            except (ValueError, IndexError):
                continue
    
    if not checkpoints:
        return None
    
    # เรียงลำดับเอาเลข X ที่สูงสุด
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_path = checkpoints[0][1]
    print(f"[Auto-Resume] Detected latest checkpoint: {latest_path}")
    return latest_path


def _save_checkpoint(ckpt_dir: Path, model, tokenizer, optimizer, scheduler, step: int, metrics: Dict, cfg: Dict):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir) # เพิ่มการเซฟ Tokenizer ที่นี่ ✅
    torch.save(
        {
            "global_step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
        },
        ckpt_dir / "trainer_state.pt",
    )
    print(f"[Checkpoint] Saved: {ckpt_dir}")

    # ─── Push to Hub ───
    if cfg.get("push_to_hub", False):
        hub_model_id = cfg.get("hub_model_id")
        hub_token = cfg.get("hub_token") or os.environ.get("HF_TOKEN")
        if hub_model_id:
            try:
                print(f"[Hub] Pushing checkpoint to: {hub_model_id}")
                api = HfApi()
                create_repo(repo_id=hub_model_id, token=hub_token, exist_ok=True, private=cfg.get("hub_private", False))
                api.upload_folder(
                    folder_path=str(ckpt_dir),
                    repo_id=hub_model_id,
                    repo_type="model",
                    token=hub_token,
                    path_in_repo=ckpt_dir.name, # เซฟเป็นโฟลเดอร์ checkpoint-X บน Hub ครบชุด
                    commit_message=f"Upload checkpoint at step {step}",
                )
                print(f"[Hub] Push successful ✅")
            except Exception as e:
                print(f"[Hub] Push failed: {e}")


@torch.no_grad()
def _log_sample_generation(trainer, tokenizer, samples, cfg, step: int):
    """Generate 1 sample response เพื่อ monitor คุณภาพ"""
    if not samples:
        return
    sample = samples[0]
    formatted = sample["formatted_prompt"]

    enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(trainer.policy.device)

    trainer.policy.eval()
    out = trainer.policy.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    trainer.policy.train()

    prompt_len = input_ids.shape[1]
    response = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    print(f"\n[Eval Sample @ step {step}]")
    print(f"  Q: {sample['prompt'][:100]}...")
    print(f"  A: {response[:300]}")
    print()


if __name__ == "__main__":
    main()
