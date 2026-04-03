"""
data_utils.py
─────────────
Dataset loader และ formatter สำหรับ CISPO RL training
รองรับหลาย format: GSM8K, Thai Alpaca, Thai Multihop, Custom JSONL
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset, Dataset


# ─────────────────────────────────────────────────────────────
# System prompt สำหรับ policy model (Qwen3.5 thinking format)
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful and thoughtful assistant. "
    "Always think step-by-step inside <think>...</think> tags before answering. "
    "Keep reasoning concise and avoid repeating logic. "
    "Example: <think> 1+1=2. Answer is 2. </think> The answer is 2. "
    "After thinking, provide your final response clearly."
)

DIRECT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide a direct, concise answer to the user's question. "
    "Do NOT use any <think> tags or internal reasoning steps. Answer directly."
)


# ─────────────────────────────────────────────────────────────
# Formatters per dataset type
# ─────────────────────────────────────────────────────────────
def format_gsm8k(example: Dict) -> Optional[Dict]:
    """GSM8K: question + answer"""
    question = example.get("question", "").strip()
    answer = example.get("answer", "").strip()
    # ตัด scratchpad (#### section) ออก เอาแค่ตัวเลขสุดท้าย
    if "####" in answer:
        answer = answer.split("####")[-1].strip()
    if not question:
        return None
    return {"prompt": question, "reference": answer}


def format_alpaca(example: Dict) -> Optional[Dict]:
    """Thai Alpaca / standard alpaca format"""
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()
    if not instruction:
        return None
    prompt = f"{instruction}\n{input_text}".strip() if input_text else instruction
    return {"prompt": prompt, "reference": output}


def format_multihop(example: Dict) -> Optional[Dict]:
    """Thai Multihop QA (corag-style JSONL)"""
    question = example.get("question", example.get("query", "")).strip()
    answer = example.get("answer", example.get("golden_answers", [""])[0])
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    if not question:
        return None
    return {"prompt": question, "reference": str(answer)}


def format_opus_reasoning(example: Dict) -> Optional[Dict]:
    """
    nohurry/Opus-4.6-Reasoning-3000x-filtered format:
    Fields: id, problem, thinking, solution, difficulty, category

    ใช้ 'problem' เป็น prompt และ 'solution' เป็น reference
    'thinking' ไม่ถูกส่งเข้าโมเดล (เป็น target ที่โมเดลต้องสร้างเอง)
    """
    problem = example.get("problem", "").strip()
    solution = example.get("solution", "").strip()
    if not problem or not solution:
        return None
    # กรอง problem ที่สั้นเกิน (fragment / incomplete)
    if len(problem) < 20:
        return None
    return {"prompt": problem, "reference": solution}


def format_raw(example: Dict) -> Optional[Dict]:
    """Raw format — ต้องมีฟิลด์ 'prompt' อยู่แล้ว"""
    prompt = example.get("prompt", example.get("question", "")).strip()
    reference = example.get("reference", example.get("answer", example.get("output", "")))
    if not prompt:
        return None
    return {"prompt": prompt, "reference": str(reference)}


FORMAT_FUNCTIONS = {
    "gsm8k": format_gsm8k,
    "alpaca": format_alpaca,
    "thai_multihop": format_multihop,
    "opus_reasoning": format_opus_reasoning,  # nohurry/Opus-4.6-Reasoning-3000x-filtered
    "raw": format_raw,
}


# ─────────────────────────────────────────────────────────────
# Chat template builder
# ─────────────────────────────────────────────────────────────
def build_chat_prompt(prompt: str, tokenizer, enable_thinking: bool = True) -> str:
    """
    สร้าง formatted chat prompt สำหรับ policy model 
    เชื่อมโยงคำสั่งกับโหมดที่ตั้งค่าไว้อย่างชัดเจน
    """
    current_system = SYSTEM_PROMPT if enable_thinking else DIRECT_SYSTEM_PROMPT
    
    messages = [
        {"role": "system", "content": current_system},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,   # ควบคุมโหมดคิดต่อ sample
        )
    except Exception:
        # Fallback: จำลองโหมดด้วย manual prefix
        if enable_thinking:
            suffix = "<think>\n"
        else:
            suffix = "<think>\n\n</think>\n\n"
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{suffix}"


# ─────────────────────────────────────────────────────────────
# Main Dataset Loader
# ─────────────────────────────────────────────────────────────
def load_rl_dataset(
    dataset_name: str,
    dataset_format: str,
    tokenizer,
    dataset_split: str = "train",
    custom_data_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_prompt_length: int = 512,
    seed: int = 42,
    thinking_ratio: float = 0.8,    # สัดส่วนที่เปิด Thinking Mode (ตาม SFT distribution)
) -> List[Dict]:
    """
    โหลด dataset และ format เป็น list ของ dict:
      {
        "prompt": str,          # raw question (สำหรับส่ง judge)
        "formatted_prompt": str, # chat-formatted prompt (สำหรับ policy model)
        "reference": str,       # ground truth (optional)
      }
    """
    format_fn = FORMAT_FUNCTIONS.get(dataset_format, format_raw)

    # ─── Load ───
    if custom_data_path and Path(custom_data_path).exists():
        print(f"[Dataset] Loading custom dataset from: {custom_data_path}")
        raw_data: List[Dict] = []
        with open(custom_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))
        raw_dataset = Dataset.from_list(raw_data)
    else:
        print(f"[Dataset] Loading from HuggingFace: {dataset_name} ({dataset_split})")
        raw_hf = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
        raw_dataset = raw_hf

    # ─── Format ───
    rng = random.Random(seed)
    formatted = []
    for example in raw_dataset:
        result = format_fn(example)
        if result is None:
            continue

        prompt_text = result["prompt"]

        # สุ่มโหมดคิดต่อ Sample ตาม thinking_ratio (ตรงกับ SFT distribution)
        # intended_thinking=True  → เปิดให้คิดยาว
        # intended_thinking=False → ปิด แต่ Judge ยังให้คะแนนความลึกเต็มๆ
        intended_thinking = rng.random() < thinking_ratio
        formatted_prompt = build_chat_prompt(prompt_text, tokenizer, enable_thinking=intended_thinking)

        # Filter ตาม max_prompt_length
        token_len = len(tokenizer.encode(formatted_prompt))
        if token_len > max_prompt_length:
            continue

        formatted.append(
            {
                "prompt": prompt_text,
                "formatted_prompt": formatted_prompt,
                "reference": result.get("reference", ""),
                "intended_thinking": intended_thinking,   # เก็บไว้สำหรับ logging
            }
        )

    # ─── Shuffle & limit ───
    random.seed(seed)
    random.shuffle(formatted)

    if max_samples is not None:
        formatted = formatted[:max_samples]

    print(f"[Dataset] Total samples ready: {len(formatted)}")
    return formatted


# ─────────────────────────────────────────────────────────────
# Collate for DataLoader (optional)
# ─────────────────────────────────────────────────────────────
def collate_prompts(batch: List[Dict]) -> Dict:
    """Simple collate function — returns list wrappers"""
    return {
        "prompts": [x["prompt"] for x in batch],
        "formatted_prompts": [x["formatted_prompt"] for x in batch],
        "references": [x["reference"] for x in batch],
        "intended_thinking": [x["intended_thinking"] for x in batch],
    }
