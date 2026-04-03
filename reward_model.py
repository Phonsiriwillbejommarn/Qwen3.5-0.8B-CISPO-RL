"""
reward_model.py
───────────────
AI Judge Reward ใช้ Qwen/Qwen3.5-9B เป็น LLM-as-a-Judge
พร้อม anti-repetition penalty และ format/length rewards

Reward สุดท้าย:
  r_final = r_judge + λ1*r_length - λ2*r_repetition + λ3*r_format
"""

from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─────────────────────────────────────────────
# Judge Prompt Template
# ─────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are a strict but fair AI evaluator. Your task is to score a model's response.
Evaluate based on:
1. Correctness — Is the answer accurate and logically sound?
2. Helpfulness — Does it directly address the question?
3. Coherence — Is the reasoning clear and non-repetitive?
4. Depth of Reasoning — Be extremely critical. If the logic is shallow or skips steps, even if the final answer is correct, you MUST lower the score significantly.
5. Anti-Hallucination — If the model starts rambling, repeating itself endlessly, generating unrelated languages (e.g., Russian, Polish, etc.), or failing to close its thoughts coherently, you MUST score it 0.0 immediately.

Return ONLY a single float score between 0.0 and 1.0 on its own line.
Do not add any explanation. Only output the number."""

JUDGE_USER_TEMPLATE = """Question:
{question}

Model Response:
{response}

Score (0.0-1.0):"""


# ─────────────────────────────────────────────
# Repetition Detector
# ─────────────────────────────────────────────
def compute_ngram_repetition(text: str, n: int = 4) -> float:
    """
    คำนวณ n-gram repetition ratio ใน text.
    ถ้า ratio สูง = text มีการวนซ้ำมาก

    Returns: float ในช่วง [0.0, 1.0], 0 = ไม่ซ้ำ, 1 = ซ้ำทั้งหมด
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    repetition_ratio = 1.0 - (unique / total)
    return repetition_ratio


def compute_repetition_penalty(text: str, ngram_size: int = 4, threshold: float = 0.3) -> float:
    """
    คืนค่า penalty [0, 1] สำหรับ repetitive text
    - ถ้า ratio < threshold → penalty = 0 (ปกติ)
    - ถ้า ratio >= threshold → penalty เพิ่มขึ้น proportionally
    """
    ratio = compute_ngram_repetition(text, n=ngram_size)
    if ratio <= threshold:
        return 0.0
    # Scale penalty: max penalty = 1.0 when ratio = 1.0
    penalty = (ratio - threshold) / (1.0 - threshold + 1e-8)
    return min(penalty, 1.0)


# ─────────────────────────────────────────────
# Format Checker
# ─────────────────────────────────────────────
def compute_format_reward(response: str, intended_thinking: bool = True) -> float:
    """
    ให้ reward/penalty ตามโหมดการคิด:
    ใช้ Regex ตรวจสอบแท็กที่สมบูรณ์เท่านั้น เพื่อป้องกันปัญหาพิมพ์ <thinking> แล้วมั่วได้คะแนน
    """
    has_think_open = bool(re.search(r"<(think|tink)>", response))
    has_think_close = bool(re.search(r"</(think|tink)>", response))
    
    # เช็คว่ามีเนื้อหาหลังจากปิดแท็กหรือไม่ (ป้องกันการตอบแต่ในแท็ก)
    has_content_after_think = bool(
        # ต้องตามด้วย whitespace หรือไม่ก็ขึ้นตัวอักษรใหม่ที่ไม่ใช่ <
        re.search(r"</(think|tink)>\s*[^<\s]+", response, re.DOTALL)
    )
    if intended_thinking:
        # ─── โหมดคิด (Thinking Mode) ───
        # Zero Tolerance: พลาดจุดเดียว โดน -1.0 ไม่มีการให้คะแนนโบนัสปลอบใจ
        if not has_think_open:
            return -1.0  # ลืมเปิดแท็ก
        if not has_think_close:
            return -1.0  # เพ้อจนลืมปิดแท็ก
        if not has_content_after_think:
            return -1.0  # ปิดแท็กแต่ไม่ตอบอะไรเลย
            
        return 1.0  # ผ่านแบบสมบูรณ์แบบ
    else:
        # ─── โหมดตอบตรง (Direct Mode) ───
        if has_think_open or has_think_close:
            return -1.0
        return 1.0  # ตอบตรงตามสั่ง


# ─────────────────────────────────────────────
# Length Reward
# ─────────────────────────────────────────────
def compute_length_reward(
    response: str,
    tokenizer,
    target_think_tokens: int = 768,
    target_response_tokens: int = 256,
) -> float:
    """
    Reward สำหรับ response ที่มีความยาว reasoning กระชับ
    - รองรับทั้งแท็ก <think> และ <tink>
    """
    think_match = re.search(r"<(think|tink)>(.*?)</\1>", response, re.DOTALL)
    think_text = think_match.group(2) if think_match else ""
    response_text = re.sub(r"<(think|tink)>.*?</\1>", "", response, flags=re.DOTALL).strip()

    think_tokens = len(tokenizer.encode(think_text)) if think_text else 0
    response_tokens = len(tokenizer.encode(response_text)) if response_text else 0

    # Penalty เป็น sigmoid-based — เบาๆ ไม่ลงโทษแรงเกิน
    think_excess = max(0, think_tokens - target_think_tokens)
    length_penalty = -math.log1p(think_excess / 100.0) / 10.0

    return max(length_penalty, -1.0)


# ─────────────────────────────────────────────
# AI Judge Reward Model
# ─────────────────────────────────────────────
@dataclass
class RewardConfig:
    judge_model_name: str = "Qwen/Qwen3.5-9B"
    judge_max_new_tokens: int = 128
    judge_temperature: float = 0.1
    judge_batch_size: int = 8
    reward_judge_weight: float = 1.0
    reward_length_weight: float = 0.2  # เพิ่มน้ำหนักการคุมความยาว
    reward_repetition_weight: float = 0.6  # หักคะแนนซ้ำหนักขึ้นเท่าตัว
    reward_format_weight: float = 0.3      # บังคับ Tag ให้เข้มข้นขึ้น
    repetition_ngram_size: int = 4
    repetition_threshold: float = 0.2      # ลด Threshold ลงเพื่อให้ตรวจเจอความซ้ำได้ไวขึ้น
    max_think_tokens: int = 4096            # ปรับเพิ่มเป็น 4096 ตามที่คุณขอ
    max_response_tokens: int = 256
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


class CISPORewardModel:
    """
    Composite Reward Model สำหรับ CISPO RL training:
      1. AI Judge reward (Qwen3.5-9B)
      2. Anti-repetition penalty
      3. Length efficiency reward
      4. Format reward
    """

    def __init__(self, config: RewardConfig, policy_tokenizer):
        self.config = config
        self.policy_tokenizer = policy_tokenizer

        print(f"[RewardModel] Loading judge model: {config.judge_model_name}")
        self.judge_tokenizer = AutoTokenizer.from_pretrained(
            config.judge_model_name,
            trust_remote_code=True,
        )
        self.judge_model = AutoModelForCausalLM.from_pretrained(
            config.judge_model_name,
            torch_dtype=config.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.judge_model.eval()
        print(f"[RewardModel] Judge model loaded ✅")

    def _build_judge_prompt(self, question: str, response: str) -> str:
        """สร้าง prompt สำหรับ judge model"""
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    question=question, response=response
                ),
            },
        ]
        return self.judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def _get_judge_scores(self, questions: List[str], responses: List[str]) -> List[float]:
        """
        เรียก Qwen3.5-9B judge สำหรับ batch ของ (question, response)
        Returns: list of float scores [0.0, 1.0]
        """
        scores = []
        batch_size = self.config.judge_batch_size

        for i in range(0, len(questions), batch_size):
            batch_q = questions[i : i + batch_size]
            batch_r = responses[i : i + batch_size]

            prompts = [self._build_judge_prompt(q, r) for q, r in zip(batch_q, batch_r)]

            inputs = self.judge_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.judge_model.device)

            outputs = self.judge_model.generate(
                **inputs,
                max_new_tokens=self.config.judge_max_new_tokens,
                temperature=self.config.judge_temperature,
                do_sample=True if self.config.judge_temperature > 0 else False,
                pad_token_id=self.judge_tokenizer.eos_token_id,
            )

            # Decode เฉพาะ generated tokens
            input_lengths = inputs["input_ids"].shape[1]
            generated = outputs[:, input_lengths:]
            decoded = self.judge_tokenizer.batch_decode(generated, skip_special_tokens=True)

            for text in decoded:
                score = self._parse_score(text)
                scores.append(score)

        return scores

    def _parse_score(self, text: str) -> float:
        """Parse score จาก output ของ judge model"""
        text = text.strip()
        # หา float แรกใน text
        matches = re.findall(r"\d+\.?\d*", text)
        if matches:
            score = float(matches[0])
            # Normalize ถ้า judge ให้ 0-10 แทน 0-1
            if score > 1.0:
                score = score / 10.0
            return max(0.0, min(1.0, score))
        return 0.0  # default ถ้า parse ไม่ได้

    def compute_rewards(
        self,
        questions: List[str],
        responses: List[str],
        intended_thinking: List[bool] = None,
    ) -> List[float]:
        """คืน final reward เท่านั้น (สำหรับ logging)"""
        rewards, _ = self.compute_rewards_with_components(
            questions, responses, intended_thinking=intended_thinking
        )
        return rewards

    def compute_rewards_with_components(
        self,
        questions: List[str],
        responses: List[str],
        intended_thinking: List[bool] = None,
    ) -> tuple:
        """
        คำนวณ final reward พร้อม component breakdown

        Returns:
            (final_rewards: List[float],
             components: List[Dict])   ← {"judge", "rep_penalty", "length_penalty"}
        """
        # 1. AI Judge rewards
        judge_scores = self._get_judge_scores(questions, responses)

        final_rewards = []
        components = []

        if intended_thinking is None:
            intended_thinking = [True] * len(questions)

        for question, response, judge_score, it_mode in zip(questions, responses, judge_scores, intended_thinking):
            # 2. Anti-repetition penalty
            rep_penalty = compute_repetition_penalty(
                response,
                ngram_size=self.config.repetition_ngram_size,
                threshold=self.config.repetition_threshold,
            )

            # 3. Length penalty
            len_reward = compute_length_reward(
                response,
                self.policy_tokenizer,
                target_think_tokens=self.config.max_think_tokens,
                target_response_tokens=self.config.max_response_tokens,
            )
            length_penalty = max(0.0, -len_reward)   # แปลงเป็น penalty (≥0)

            # 4. Format reward (mode-aware)
            fmt_reward = compute_format_reward(response, intended_thinking=it_mode)

            # Final composite reward
            final_reward = (
                self.config.reward_judge_weight * judge_score
                + self.config.reward_length_weight * len_reward
                - self.config.reward_repetition_weight * rep_penalty
                + self.config.reward_format_weight * fmt_reward
            )

            final_rewards.append(final_reward)
            components.append({
                "judge": judge_score,
                "rep_penalty": rep_penalty,
                "length_penalty": length_penalty,
            })

        return final_rewards, components
