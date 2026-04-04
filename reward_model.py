"""
reward_model.py
───────────────
AI Judge Reward ใช้ Qwen/Qwen3.5-9B เป็น LLM-as-a-Judge
พร้อม dual anti-repetition penalty, bell-curve length reward,
fix parse score, และ conservative ensemble scoring

Reward สุดท้าย:
  r_final = r_judge + λ1*r_length - λ2*r_think_rep - λ3*r_ans_rep + λ4*r_format
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
JUDGE_SYSTEM_PROMPT = """You are a strict but fair AI evaluator. Your task is to score a model's response, including its hidden reasoning process (<think>).
Evaluate based on:
1. Correctness — Is the final answer accurate and logically sound?
2. Helpfulness — Does it directly address the user's question?
3. Reasoning Quality — Look inside the <think> tags. Is the reasoning clear, progressive, and non-repetitive? If the model repeats same thoughts or steps without making progress, you MUST lower the score significantly.
4. Language Consistency — The reasoning and response should be in Thai or English as context requires. If the model starts rambling in unrelated languages (e.g., Russian, Chinese, or gibberish), score it 0.0 immediately.
5. Depth of Reasoning — For complex questions, a long <think> is acceptable if every step is meaningful. Be critical of "fake" long reasoning that just repeats basic facts.

Return ONLY a single float score between 0.0 and 1.0 on its own line.
Do not add any explanation. Only output the number."""

JUDGE_USER_TEMPLATE = """Question:
{question}

Model Response:
{response}

Score (0.0-1.0):"""


# ─────────────────────────────────────────────
# Repetition Detector (Dual: Think vs Answer)
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
    คืนค่า penalty [0, 1] สำหรับ repetitive text (generic helper)
    """
    ratio = compute_ngram_repetition(text, n=ngram_size)
    if ratio <= threshold:
        return 0.0
    penalty = (ratio - threshold) / (1.0 - threshold + 1e-8)
    return min(penalty, 1.0)


def compute_dual_repetition_penalty(
    response: str,
    think_ngram_size: int = 16,
    think_threshold: float = 0.12,
    answer_ngram_size: int = 4,
    answer_threshold: float = 0.10,
) -> tuple[float, float]:
    """
    แยกตรวจ repetition ของ <think> กับ Answer แบบอิสระ

    <think>:
      - ใช้ n-gram ขนาดใหญ่ (16) เพื่อจับ reasoning loop ระดับ paragraph
      - threshold ค่อนข้างเข้มงวด (0.12) เพราะโมเดลนี้มีปัญหาวนซ้ำ 7+ รอบ

    Answer:
      - ใช้ n-gram ขนาดเล็ก (4) จับการซ้ำระดับคำ
      - threshold เข้มงวดมาก (0.10) เพราะคำตอบต้องกระชับเท่านั้น

    Returns: (think_penalty, answer_penalty) แต่ละค่าอยู่ใน [0.0, 1.0]
    """
    # แยก think text
    think_match = re.search(r"<(think|tink)>(.*?)</\1>", response, re.DOTALL)
    think_text = think_match.group(2).strip() if think_match else ""

    # แยก answer text (ตัดส่วน think ออก)
    answer_text = re.sub(r"<(think|tink)>.*?</\1>", "", response, flags=re.DOTALL).strip()

    think_penalty = compute_repetition_penalty(
        think_text, ngram_size=think_ngram_size, threshold=think_threshold
    ) if think_text else 0.0

    answer_penalty = compute_repetition_penalty(
        answer_text, ngram_size=answer_ngram_size, threshold=answer_threshold
    ) if answer_text else 0.0

    return think_penalty, answer_penalty


# ─────────────────────────────────────────────
# Format Checker
# ─────────────────────────────────────────────
def compute_format_reward(response: str, intended_thinking: bool = True) -> float:
    """
    ให้ reward/penalty ตามโหมดการคิด:
    ใช้ Regex ตรวจสอบแท็กที่สมบูรณ์เท่านั้น เพื่อป้องกันปัญหาพิมพ์ <thinking> แล้วมั่วได้คะแนน
    """
    # 1. ต้องเปิดที่จุดเริ่มต้นของประโยคเท่านั้น (ห้ามตอบก่อนแล้วค่อยคิด)
    # รัดกุม: อนุญาตให้มี whitespace ข้างหน้าได้นิดหน่อยแล้วต้องเจอ <think> ทันที
    starts_with_think = bool(re.match(r"^\s*<(think|tink)>", response))
    
    # 2. ต้องมีการเปิดและปิดอย่างละ 1 ครั้งเท่านั้น (ห้ามเปิด/ปิดซ้ำซ้อน)
    open_count = len(re.findall(r"<(think|tink)>", response))
    close_count = len(re.findall(r"</(think|tink)>", response))
    
    # 3. ต้องมีเนื้อหาคำตอบหลังปิดแท็ก
    has_content_after_think = bool(
        re.search(r"</(think|tink)>\s*[^<\s]+", response, re.DOTALL)
    )

    if intended_thinking:
        # Zero Tolerance: กฎเหล็ก
        if not starts_with_think:
            return -1.0  # ต้องเริ่มประโยคด้วย <think> เท่านั้น!
        if open_count != 1:
            return -1.0  # ห้ามเปิดแท็กหลายรอบ หรือลืมเปิด
        if close_count != 1:
            return -1.0  # ห้ามปิดแท็กหลายรอบ หรือลืมปิด
        if not has_content_after_think:
            return -1.0  # ปิดแท็กแล้วเงียบ ไม่ตอบคำถาม
            
        return 1.0  # ผ่านแบบไร้ที่ติ
            
    else:
        # ─── โหมดตอบตรง (Direct Mode) ───
        if open_count > 0 or close_count > 0:
            return -1.0
        return 1.0  # ตอบตรงตามสั่ง


# ─────────────────────────────────────────────
# Length Reward (Bell Curve + Safety Net)
# ─────────────────────────────────────────────
def compute_length_reward(
    response: str,
    tokenizer,
    min_think_tokens: int = 64,
    optimal_think_tokens: int = 512,
    target_think_tokens: int = 4096,
) -> float:
    """
    Bell Curve + Safety Net:

    Zone 1 — Ramp Up   [0, min_think_tokens]:
        หักคะแนนเบาๆ เพราะคิดสั้นเกินไป (ไม่ใช้ความพยายาม)

    Zone 2 — Ramp Up   [min_think, optimal_think]:
        ให้ reward เพิ่มขึ้น linear จนถึง +0.5 ที่ optimal
        (สร้าง incentive ให้โมเดลคิดพอเหมาะ)

    Zone 3 — Plateau   [optimal_think, target_think]:
        ให้ reward คงที่ที่ +0.5 ปล่อยให้ AI Judge ตัดสินคุณภาพ
        (โจทย์ยากคิดนานได้โดยไม่โดนหัก)

    Zone 4 — Safety Net [target_think, ∞]:
        หักคะแนนแบบ Exponential เพื่อตัดวงจร Infinite Loop
        (max penalty = -2.0)
    """
    think_match = re.search(r"<(think|tink)>(.*?)</\1>", response, re.DOTALL)
    think_text = think_match.group(2) if think_match else ""
    think_tokens = len(tokenizer.encode(think_text)) if think_text else 0

    if think_tokens <= min_think_tokens:
        # Zone 1: คิดสั้นเกินไป หักคะแนนเบาๆ
        ratio = think_tokens / max(min_think_tokens, 1)
        return -0.3 * (1.0 - ratio)  # [−0.3, 0.0]

    elif think_tokens <= optimal_think_tokens:
        # Zone 2: ramp up linear ให้ reward ตามความพยายาม
        ratio = (think_tokens - min_think_tokens) / (optimal_think_tokens - min_think_tokens)
        return 0.5 * ratio  # [0.0, +0.5]

    elif think_tokens <= target_think_tokens:
        # Zone 3: Plateau — ให้ reward คงที่, ไม่หักจากความยาว
        return 0.5

    else:
        # Zone 4: Safety Net — Exponential penalty สำหรับ Infinite Loop
        excess = think_tokens - target_think_tokens
        penalty = -math.log1p(excess / 20.0)  # หักแรง (หาร 20 แทน 50)
        return max(penalty, -2.0)


# ─────────────────────────────────────────────
# AI Judge Reward Model
# ─────────────────────────────────────────────
from dataclasses import dataclass, field
@dataclass
class RewardConfig:
    judge_models: list = field(default_factory=lambda: ["Qwen/Qwen3.5-9B", "google/gemma-4-E4B-it"])
    judge_max_new_tokens: int = 128
    judge_temperature: float = 0.1
    judge_batch_size: int = 8
    reward_judge_weight: float = 1.0
    reward_length_weight: float = 0.4       # Bell curve + Safety Net
    reward_think_rep_weight: float = 0.5    # หักหนักสำหรับ reasoning loop ใน <think>
    reward_answer_rep_weight: float = 0.3   # หักคำตอบซ้ำ
    reward_format_weight: float = 0.3
    # Dual repetition config
    think_ngram_size: int = 16              # จับ loop ระดับ paragraph
    think_rep_threshold: float = 0.12       # เข้มงวด เพราะโมเดลวน 7+ รอบ
    answer_ngram_size: int = 4
    answer_rep_threshold: float = 0.10      # เข้มงวดมาก เพราะคำตอบต้องกระชับ
    # Length bell curve config
    min_think_tokens: int = 64              # คิดน้อยกว่านี้ถือว่าสั้นเกิน
    optimal_think_tokens: int = 512         # zone ที่ได้ reward สูงสุด
    max_think_tokens: int = 4096            # เส้นตาย Safety Net
    max_response_tokens: int = 512
    # Ensemble strategy: 'weighted_min' หรือ 'average'
    ensemble_strategy: str = "weighted_min"
    # GRPO Group config
    num_generations: int = 6    # จำนวนคำตอบต่อ 1 คำถาม (GRPO group size)
    group_normalize: bool = True  # normalize reward ภายใน group แต่ละหมู่
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

        # โหลดคณะกรรมการทั้งหมด (Ensemble)
        self.judges = []
        for model_name in self.config.judge_models:
            print(f"[RewardModel] Loading judge model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            # สำคัญ: โมเดลบางตัวไม่มี pad_token 
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=config.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            self.judges.append((model_name, tokenizer, model))
            
        print(f"[RewardModel] All {len(self.judges)} judge models loaded successfully ✅")

    def _is_qwen_model(self, model_name: str) -> bool:
        """ตรวจสอบว่าเป็น Qwen model หรือไม่ (ต้องปิด thinking mode)"""
        return "qwen" in model_name.lower()

    def _is_gemma_model(self, model_name: str) -> bool:
        """ตรวจสอบว่าเป็น Gemma model หรือไม่ (ใช้ thinking token ต่างออกไป)"""
        return "gemma" in model_name.lower()

    def _build_judge_prompt(self, question: str, response: str, tokenizer, model_name: str = "") -> str:
        """
        สร้าง prompt สำหรับ judge model แต่ละตัว
        - Qwen3.5: ปิด thinking mode ด้วย enable_thinking=False
        - Gemma 4: ใช้ chat template ปกติ (thinking trigger ต่างออกไป)
        - Fallback: plain text
        """
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    question=question, response=response
                ),
            },
        ]
        try:
            if self._is_qwen_model(model_name):
                # Qwen3.5-9B: ปิด thinking mode เพื่อให้ output เป็นตัวเลขตรงๆ
                # ไม่งั้นจะได้ <think>...</think>0.7 แล้ว _parse_score จับผิดตัว
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # ← สำคัญมาก!
                )
            else:
                # Gemma 4 และอื่นๆ: ใช้ chat template ตามปกติ
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            # Fallback สำหรับ Base Model ที่ไม่มี Chat Template
            return f"{JUDGE_SYSTEM_PROMPT}\n\n{JUDGE_USER_TEMPLATE.format(question=question, response=response)}"

    @torch.no_grad()
    def _get_judge_scores(self, questions: List[str], responses: List[str]) -> List[float]:
        """
        เรียกคณะกรรมการทุกคนมาให้คะแนน แล้วหาค่าเฉลี่ย
        Returns: list of float scores [0.0, 1.0]
        """
        batch_size = self.config.judge_batch_size
        n_responses = len(questions)
        all_model_scores = []

        # วนลูปถามกรรมการทีละคน
        for m_name, j_tokenizer, j_model in self.judges:
            scores_for_this_model = []
            
            for i in range(0, n_responses, batch_size):
                batch_q = questions[i : i + batch_size]
                batch_r = responses[i : i + batch_size]

                prompts = [
                    self._build_judge_prompt(q, r, j_tokenizer, model_name=m_name)
                    for q, r in zip(batch_q, batch_r)
                ]

                inputs = j_tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=8192,
                ).to(j_model.device)

                outputs = j_model.generate(
                    **inputs,
                    max_new_tokens=self.config.judge_max_new_tokens,
                    temperature=self.config.judge_temperature,
                    do_sample=True if self.config.judge_temperature > 0 else False,
                    pad_token_id=j_tokenizer.eos_token_id,
                )

                input_lengths = inputs["input_ids"].shape[1]
                generated = outputs[:, input_lengths:]
                decoded = j_tokenizer.batch_decode(generated, skip_special_tokens=True)

                for text in decoded:
                    score = self._parse_score(text)
                    scores_for_this_model.append(score)
                    
            all_model_scores.append(scores_for_this_model)

        # Ensemble: Conservative Weighted Min
        # ถ้ากรรมการคนใดคนหนึ่งให้คะแนนต่ำมาก (<0.4) จะมีน้ำหนักมากขึ้น
        # เพื่อป้องกันไม่ให้กรรมการคนดีหักล้างกรรมการที่เห็นปัญหา
        final_scores = []
        n_models = len(self.judges)
        for i in range(n_responses):
            scores = [all_model_scores[m][i] for m in range(n_models)]
            if n_models == 1 or self.config.ensemble_strategy == "average":
                final_scores.append(sum(scores) / n_models)
            else:
                # Weighted Min: น้ำหนักสูงสุดให้กรรมการที่ให้คะแนนต่ำสุด
                min_score = min(scores)
                avg_score = sum(scores) / n_models
                # conservative_score = 70% min + 30% average
                conservative = 0.7 * min_score + 0.3 * avg_score
                final_scores.append(conservative)

        return final_scores

    def _parse_score(self, text: str) -> float:
        """
        Parse score จาก output ของ judge model
        - ตัด <think>...</think> ของ Qwen3.5 ออกก่อน (safety fallback กรณี enable_thinking ใช้ไม่ได้)
        - ตัด <|channel>thought...<channel|> ของ Gemma 4 ออกก่อน
        - แก้ edge case: 1.5 → 0.15 bug (แยก 0-10 สเกล vs >10 clamp)
        """
        # Step 1: ตัด thinking blocks ของ Qwen3.5 (<think>...</think>)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Step 2: ตัด thinking blocks ของ Gemma 4 (<|channel>thought...<channel|>)
        text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
        # Step 3: ตัด thinking blocks รูปแบบอื่นๆ ที่อาจพบ (defensive)
        text = re.sub(r"<\|im_start\|>think.*?<\|im_end\|>", "", text, flags=re.DOTALL)
        text = text.strip()

        # Step 4: หา float แรกใน text ที่เหลือ (รองรับทศนิยม)
        matches = re.findall(r"\d+\.\d+|\d+", text)
        if matches:
            score = float(matches[0])
            if 1.0 < score <= 10.0:
                # สเกล 0-10: หาร 10 เพื่อ normalize
                score = score / 10.0
            elif score > 10.0:
                # ค่าแปลกเกินไป (เช่น 100) → clamp เป็น 1.0
                score = 1.0
            # else: score อยู่ใน [0.0, 1.0] แล้ว ใช้ได้เลย
            return max(0.0, min(1.0, score))
        return 0.0  # default ถ้า parse ไม่ได้หลังตัด thinking ออกแล้ว

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
            # 2. Dual Anti-repetition penalty (แยก think vs answer)
            think_rep_penalty, answer_rep_penalty = compute_dual_repetition_penalty(
                response,
                think_ngram_size=self.config.think_ngram_size,
                think_threshold=self.config.think_rep_threshold,
                answer_ngram_size=self.config.answer_ngram_size,
                answer_threshold=self.config.answer_rep_threshold,
            )

            # 3. Bell Curve + Safety Net Length Reward
            len_reward = compute_length_reward(
                response,
                self.policy_tokenizer,
                min_think_tokens=self.config.min_think_tokens,
                optimal_think_tokens=self.config.optimal_think_tokens,
                target_think_tokens=self.config.max_think_tokens,
            )

            # 4. Format reward (mode-aware)
            fmt_reward = compute_format_reward(response, intended_thinking=it_mode)

            # Final composite reward
            final_reward = (
                self.config.reward_judge_weight * judge_score
                + self.config.reward_length_weight * len_reward
                - self.config.reward_think_rep_weight * think_rep_penalty
                - self.config.reward_answer_rep_weight * answer_rep_penalty
                + self.config.reward_format_weight * fmt_reward
            )

            final_rewards.append(final_reward)
            components.append({
                "judge": judge_score,
                "think_rep_penalty": think_rep_penalty,
                "answer_rep_penalty": answer_rep_penalty,
                "len_reward": len_reward,
            })

        return final_rewards, components

    def group_normalize_rewards(
        self,
        rewards: List[float],
        num_generations: int,
    ) -> List[float]:
        """
        GRPO-style Group Normalization:
        normalize reward แต่ละคำตอบเทียบกับคนอื่นใน group เดียวกัน

        - rewards ใน group ถูกเรียงต่อเนื่องกัน: [q1_r1, q1_r2, ..., q1_r6, q2_r1, ...]
        - แต่ละ group จะถูก normalize ด้วย: (r - mean(group)) / (std(group) + eps)
        - โมเดลจะเรียนรู้เปรียบเทียบคำตอบ ไม่ใช่แค่ reward สัมบูรณ์เป็นตัวเลข

        Args:
            rewards: list ของ reward ที่เรียงตัว [คำตอบทุกคำ]
            num_generations: ขนาดกรุ๊ป (6)
        Returns:
            normalized_rewards: list ขนาดเดียวกัน
        """
        eps = 1e-8
        normalized = []
        n = len(rewards)

        for i in range(0, n, num_generations):
            group = rewards[i : i + num_generations]
            if len(group) == 0:
                continue

            mean = sum(group) / len(group)
            variance = sum((r - mean) ** 2 for r in group) / len(group)
            std = math.sqrt(variance) + eps

            for r in group:
                normalized.append((r - mean) / std)

        return normalized

    def compute_group_rewards(
        self,
        questions: List[str],
        responses: List[str],
        intended_thinking: List[bool] = None,
    ) -> tuple:
        """
        Entry point หลักสำหรับ GRPO Training:
        1. คำนวณ raw rewards ทุกตัว
        2. Normalize ภายใน group ตาม num_generations

        Input layout:
            questions = [q1, q1, q1, q1, q1, q1,  q2, q2, ...]  # โมเดล repeat คำถาม
            responses = [r1a, r1b, r1c, r1d, r1e, r1f,  r2a, ...]

        Returns:
            (normalized_rewards, raw_rewards, components)
        """
        raw_rewards, components = self.compute_rewards_with_components(
            questions, responses, intended_thinking=intended_thinking
        )

        if self.config.group_normalize and self.config.num_generations > 1:
            normalized_rewards = self.group_normalize_rewards(
                raw_rewards, num_generations=self.config.num_generations
            )
        else:
            normalized_rewards = raw_rewards

        return normalized_rewards, raw_rewards, components
