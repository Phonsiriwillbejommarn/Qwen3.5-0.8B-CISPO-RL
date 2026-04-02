"""
cispo_trainer.py
────────────────
CISPO (Clipped Importance Sampling Policy Optimization)
ตามสูตรจริงจาก MiniMax-M2.5 paper:

  J_CISPO(θ) = E[ 1/Σ|o_i| · ΣΣ sg(f̃_{i,t}) · Ã_{i,t} · log π_θ(o_{i,t}|q, o_{i,<t}) ]

  where:
    f̃_{i,t}(θ) = clip(r_{i,t}(θ), 0, 1 + ε^IS_high)   ← clip ทั้ง lower=0, upper=1+ε
    Ã_{i,t}    = Σ_{p=t}^T (r_p^speed + r_p^perf) - B_i ← process reward return-to-go

Process rewards:
  r_p^perf  = judge quality score (assigned ที่ token สุดท้าย)
  r_p^speed = per-token efficiency signal (length + repetition penalty ที่ accumulate)
  B_i       = baseline = mean total_reward across same group
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import PreTrainedModel, PreTrainedTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────
class PromptDataset(TorchDataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Process Reward Builder
# ──────────────────────────────────────────────────────────────────────────────
def build_token_process_rewards(
    response_text: str,
    response_token_ids: List[int],
    tokenizer,
    judge_score: float,         # r^perf  (0.0–1.0 จาก AI judge)
    rep_penalty: float,         # repetition penalty (0.0–1.0)
    length_penalty: float,      # length excess penalty (>=0)
    speed_weight: float = 1.0,
    perf_weight: float = 1.0,
) -> List[float]:
    """
    สร้าง token-level process rewards สำหรับ 1 response:

      r_p^perf  = judge_score        → assign ให้ token สุดท้ายเท่านั้น
      r_p^speed = speed_reward_t     → distribute ทุก token เท่าๆ กัน
                                       (penalty สำหรับ length + repetition)

    Returns: list ของ (r_p^speed + r_p^perf) สำหรับแต่ละ token position p
    """
    T = len(response_token_ids)
    if T == 0:
        return []

    # r^perf: ให้ที่ token สุดท้ายเท่านั้น (outcome reward)
    r_perf = [0.0] * T
    r_perf[-1] = perf_weight * judge_score

    # r^speed: distribute evenly — penalty สำหรับ repetition + length
    # (negative = ช้า/ไม่มีประสิทธิภาพ)
    per_token_speed = speed_weight * (-(rep_penalty + length_penalty) / T)
    r_speed = [per_token_speed] * T

    # Combined per-token: r_p = r_p^speed + r_p^perf
    return [r_speed[p] + r_perf[p] for p in range(T)]


def compute_process_return_to_go(
    token_rewards: List[float],
    baseline: float,
) -> List[float]:
    """
    คำนวณ return-to-go ตามสูตร:
      Ã_{t} = Σ_{p=t}^T (r_p) - B

    = cumulative sum from right (suffix sum) - baseline
    """
    T = len(token_rewards)
    if T == 0:
        return []

    # Suffix sum: rtg[t] = r[t] + r[t+1] + ... + r[T-1]
    rtg = [0.0] * T
    rtg[-1] = token_rewards[-1]
    for t in range(T - 2, -1, -1):
        rtg[t] = token_rewards[t] + rtg[t + 1]

    # Subtract baseline
    return [rtg[t] - baseline for t in range(T)]


def compute_group_baselines(
    all_token_rewards: List[List[float]],
    group_size: int,
) -> List[float]:
    """
    คำนวณ baseline B_i = mean ของ total reward ของ responses ใน group เดียวกัน
    Returns: list of baselines, 1 ค่าต่อ 1 response
    """
    baselines = []
    for i in range(0, len(all_token_rewards), group_size):
        group = all_token_rewards[i : i + group_size]
        group_totals = [sum(tr) for tr in group]
        b = sum(group_totals) / len(group_totals)
        baselines.extend([b] * len(group))
    return baselines


# ──────────────────────────────────────────────────────────────────────────────
# Token log-probabilities
# ──────────────────────────────────────────────────────────────────────────────
def get_token_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: List[int],
    no_grad: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute per-token log probabilities สำหรับ response tokens เท่านั้น

    Returns:
        logprobs_list: list of [T_i] tensors
        token_ids_list: list of [T_i] token id tensors
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    logits = outputs.logits  # [B, L, V]
    log_probs_all = F.log_softmax(logits, dim=-1)

    logprobs_list = []
    token_ids_list = []

    for b in range(input_ids.shape[0]):
        p_len = prompt_lengths[b]
        resp_token_ids = input_ids[b, p_len:]
        resp_logprobs = log_probs_all[b, p_len - 1 : -1]  # shifted

        if resp_token_ids.shape[0] == 0:
            logprobs_list.append(torch.zeros(0, device=input_ids.device, requires_grad=True))
            token_ids_list.append(resp_token_ids)
            continue

        token_logprobs = resp_logprobs.gather(
            dim=-1, index=resp_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        logprobs_list.append(token_logprobs)
        token_ids_list.append(resp_token_ids)

    return logprobs_list, token_ids_list


# ──────────────────────────────────────────────────────────────────────────────
# CISPO Loss  (สูตรตรงตาม paper)
# ──────────────────────────────────────────────────────────────────────────────
def cispo_loss(
    current_logprobs: List[torch.Tensor],   # log π_θ(a_t)  [requires_grad]
    old_logprobs: List[torch.Tensor],        # log π_old(a_t) [detached]
    advantages_list: List[List[float]],      # Ã_{i,t} per token per response
    epsilon_high: float = 5.0,
    kl_coeff: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    CISPO objective (ตามสูตรใน paper):

      L = -1/Σ|o_i| · ΣΣ sg(clip(r_t, 0, 1+ε_high)) · Ã_{i,t} · log π_θ(a_t)
          + β · KL(π_θ || π_old)

    หมายเหตุ:
      - clip(r_t, 0, 1+ε_high): lower=0, upper=1+ε_high (ไม่ใช่แค่ upper)
      - sg() = stop-gradient (.detach())
      - Ã_{i,t} = return-to-go process reward - baseline
    """
    assert len(current_logprobs) == len(old_logprobs) == len(advantages_list)

    total_tokens = 0
    weighted_pg_sum = torch.tensor(0.0, device=current_logprobs[0].device if current_logprobs else "cpu",
                                    requires_grad=True)
    kl_terms = []
    clip_fracs = []
    ratio_means = []

    # ใช้ accumulate แบบ inplace-safe
    pg_terms = []

    for curr_lp, old_lp, adv_tokens in zip(current_logprobs, old_logprobs, advantages_list):
        T = curr_lp.shape[0]
        if T == 0:
            continue

        # ── Importance ratio ──
        log_ratio = curr_lp - old_lp.detach()
        ratio = torch.exp(log_ratio)

        # ── CISPO clip: clip(r_t, 0, 1 + ε_high) ──
        # Lower=0 ป้องกัน negative weights ที่อาจทำให้ gradient ผิดทิศ
        # Upper=1+ε_high จำกัด variance
        clipped_ratio = torch.clamp(ratio, min=0.0, max=1.0 + epsilon_high)

        # Stop-gradient บน clipped weight
        w_t = clipped_ratio.detach()  # sg(f̃_{i,t})

        # ── Per-token advantage tensor ──
        adv_len = min(T, len(adv_tokens))
        adv_t = torch.tensor(
            adv_tokens[:adv_len],
            dtype=curr_lp.dtype,
            device=curr_lp.device,
        )
        if adv_len < T:
            # Pad ด้วย 0 ถ้า advantage สั้นกว่า (ไม่ควรเกิด)
            pad = torch.zeros(T - adv_len, dtype=curr_lp.dtype, device=curr_lp.device)
            adv_t = torch.cat([adv_t, pad])

        # Policy gradient term: -w_t * Ã_t * log π_θ
        # (negate เพื่อ gradient descent = policy gradient ascent)
        pg = -(w_t * adv_t * curr_lp)
        pg_terms.append(pg.sum())
        total_tokens += T

        # KL term
        kl_terms.append(log_ratio.mean())

        # Tracking
        clip_fracs.append((ratio > (1.0 + epsilon_high)).float().mean().item())
        ratio_means.append(ratio.mean().item())

    if not pg_terms or total_tokens == 0:
        dummy = torch.tensor(0.0, requires_grad=True)
        return dummy, {}

    # Normalize by total tokens: 1/Σ|o_i| · ΣΣ(...)
    cispo_loss_val = torch.stack(pg_terms).sum() / total_tokens
    kl_loss_val = torch.stack(kl_terms).mean()
    total_loss = cispo_loss_val + kl_coeff * kl_loss_val

    metrics = {
        "loss/cispo": cispo_loss_val.item(),
        "loss/kl": kl_loss_val.item(),
        "loss/total": total_loss.item(),
        "train/clip_fraction": sum(clip_fracs) / len(clip_fracs),
        "train/mean_ratio": sum(ratio_means) / len(ratio_means),
    }

    return total_loss, metrics


# ──────────────────────────────────────────────────────────────────────────────
# CISPO Trainer
# ──────────────────────────────────────────────────────────────────────────────
class CISPOTrainer:
    """
    Training loop ของ CISPO:
      1. Rollout G responses per prompt
      2. Compute process rewards (r^perf + r^speed) ต่อ token
      3. Compute return-to-go advantage Ã_{i,t}
      4. Forward pass (current policy)
      5. CISPO loss + backward
    """

    def __init__(
        self,
        policy_model: PreTrainedModel,
        policy_tokenizer: PreTrainedTokenizer,
        reward_model,
        config: Dict,
        optimizer,
        scheduler=None,
    ):
        self.policy = policy_model
        self.tokenizer = policy_tokenizer
        self.reward_model = reward_model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.group_size = config.get("group_size", 8)
        self.epsilon_high = config.get("epsilon_high", 5.0)
        self.kl_coeff = config.get("kl_coeff", 0.001)
        self.max_new_tokens = config.get("max_new_tokens", 1024)
        self.temperature = config.get("temperature", 1.0)
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", 50)
        self.grad_accum_steps = config.get("gradient_accumulation_steps", 4)

        # Process reward weights
        self.speed_weight = config.get("reward_length_weight", 0.1)
        self.perf_weight = config.get("reward_judge_weight", 1.0)

        self.global_step = 0

    # ─────────────────────────────────────────
    # Rollout
    # ─────────────────────────────────────────
    @torch.no_grad()
    def generate_responses(
        self, formatted_prompts: List[str]
    ) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """Generate G responses per prompt"""
        self.policy.eval()
        all_responses, all_input_ids, all_prompt_lengths = [], [], []

        for prompt_text in formatted_prompts:
            enc = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.get("max_prompt_length", 512),
            )
            input_ids = enc["input_ids"].to(self.policy.device)
            prompt_len = input_ids.shape[1]

            outputs = self.policy.generate(
                input_ids=input_ids.expand(self.group_size, -1),
                attention_mask=torch.ones(
                    self.group_size, prompt_len, device=self.policy.device
                ),
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )  # [G, L]

            for g in range(self.group_size):
                full_ids = outputs[g]
                response_text = self.tokenizer.decode(
                    full_ids[prompt_len:], skip_special_tokens=True
                )
                all_responses.append(response_text)
                all_input_ids.append(full_ids)
                all_prompt_lengths.append(prompt_len)

        self.policy.train()
        return all_responses, all_input_ids, all_prompt_lengths

    # ─────────────────────────────────────────
    # Train Step
    # ─────────────────────────────────────────
    def train_step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
    ) -> Dict[str, float]:

        # ── 1. Rollout ──
        all_responses, all_input_ids, all_prompt_lengths = self.generate_responses(
            formatted_prompts
        )
        all_prompts_rep = [p for p in prompts for _ in range(self.group_size)]

        # ── 2. Compute composite rewards via reward model ──
        # reward_model returns final scalar rewards ต่อ response
        # พร้อม component breakdown
        final_rewards, components = self.reward_model.compute_rewards_with_components(
            all_prompts_rep, all_responses
        )
        # components: list of dict {"judge", "rep_penalty", "length_penalty"}

        # ── 3. Build token-level process rewards & return-to-go advantages ──
        all_token_rewards: List[List[float]] = []

        for idx, (resp_ids, prompt_len, comp) in enumerate(
            zip(all_input_ids, all_prompt_lengths, components)
        ):
            resp_token_ids = resp_ids[prompt_len:].tolist()
            token_rewards = build_token_process_rewards(
                response_text=all_responses[idx],
                response_token_ids=resp_token_ids,
                tokenizer=self.tokenizer,
                judge_score=comp["judge"],
                rep_penalty=comp["rep_penalty"],
                length_penalty=comp["length_penalty"],
                speed_weight=self.speed_weight,
                perf_weight=self.perf_weight,
            )
            all_token_rewards.append(token_rewards)

        # Baselines: B_i = mean total reward in same group
        baselines = compute_group_baselines(all_token_rewards, self.group_size)

        # Return-to-go advantage: Ã_{i,t} = Σ_{p=t}^T r_p - B_i
        all_advantages: List[List[float]] = [
            compute_process_return_to_go(all_token_rewards[i], baselines[i])
            for i in range(len(all_token_rewards))
        ]

        # ── 4. Pad sequences for batched forward pass ──
        max_len = max(ids.shape[0] for ids in all_input_ids)
        B = len(all_input_ids)
        padded_ids = torch.zeros(B, max_len, dtype=torch.long)
        attn_masks = torch.zeros(B, max_len, dtype=torch.long)

        for i, ids in enumerate(all_input_ids):
            L = ids.shape[0]
            padded_ids[i, :L] = ids
            attn_masks[i, :L] = 1

        padded_ids = padded_ids.to(self.policy.device)
        attn_masks = attn_masks.to(self.policy.device)

        # ── 5. Old log probs (reference, no grad) ──
        with torch.no_grad():
            self.policy.eval()
            old_logprobs_list, _ = get_token_logprobs(
                self.policy, padded_ids, attn_masks, all_prompt_lengths, no_grad=True
            )
            self.policy.train()

        # ── 6. Current log probs (with grad) ──
        current_logprobs_list, _ = get_token_logprobs(
            self.policy, padded_ids, attn_masks, all_prompt_lengths, no_grad=False
        )

        # ── 7. CISPO loss ──
        loss, metrics = cispo_loss(
            current_logprobs=current_logprobs_list,
            old_logprobs=old_logprobs_list,
            advantages_list=all_advantages,
            epsilon_high=self.epsilon_high,
            kl_coeff=self.kl_coeff,
        )

        # ── 8. Backward + gradient accumulation ──
        (loss / self.grad_accum_steps).backward()

        if (self.global_step + 1) % self.grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1
        metrics["train/mean_reward"] = sum(final_rewards) / len(final_rewards)
        metrics["train/mean_baseline"] = sum(baselines) / len(baselines)
        metrics["train/step"] = self.global_step

        return metrics
