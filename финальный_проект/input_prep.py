"""
prepare_input.py
----------------
Собираем датасет nyuuzyou/wb-feedbacks с HuggingFace,
фильтрует отзывы с низкой оценкой (1–3 звезды) и непустым текстом,
сохраняет два файла:

input/reviews.jsonl      — основной корпус
input/reviews_eval.jsonl — отложенный тест-сет

"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from datasets import load_dataset


def stream_reviews(max_scan: int, total_needed: int) -> list[dict]:
    """
    Стримим датасет и собираем записи с оценкой 1–3 и непустым текстом.
    """

    print(f"Подключаемся к nyuuzyou/wb-feedbacks (streaming)...")
    ds = load_dataset(
        "nyuuzyou/wb-feedbacks",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )

    collected: list[dict] = []
    target = total_needed * 3

    for i, row in enumerate(ds):
        if i >= max_scan:
            print(f"  Достигнут лимит сканирования: {max_scan} записей.")
            break

        # Фильтры
        rating = row.get("productValuation")
        text = (row.get("text") or "").strip()

        if rating not in (1, 2, 3):
            continue
        if len(text) < 20:
            continue
        if len(text) > 1500:
            text = text[:1500]

        answer = (row.get("answer") or "").strip()
        if len(answer) < 20:
            continue

        collected.append({
            "nm_id": row.get("nmId", 0),
            "rating": rating,
            "text": text,
            "answer": (row.get("answer") or "").strip(),
            "color": (row.get("color") or "").strip(),
        })

        if len(collected) >= target:
            print(f"  Набрано {target} кандидатов, останавливаемся.")
            break

        if len(collected) % 100 == 0 and len(collected) > 0:
            print(f"  Собрано {len(collected)} / {target} (просмотрено {i+1})...")

    print(f"Итого кандидатов после фильтрации: {len(collected)}")
    return collected


def split_and_save(
    records: list[dict],
    total: int,
    eval_size: int,
    seed: int,
    out_dir: Path,
) -> None:
    rng = random.Random(seed)
    sample = records[:]
    rng.shuffle(sample)
    sample = sample[:total]

    eval_records = sample[:eval_size]
    train_records = sample[eval_size:]

    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "reviews.jsonl"
    eval_path = out_dir / "reviews_eval.jsonl"

    _write_jsonl(train_records, train_path)
    _write_jsonl(eval_records, eval_path)

    print(f"   {train_path}  — {len(train_records)} отзывов (основной корпус)")
    print(f"   {eval_path} — {len(eval_records)} отзывов (eval-сет)")
    print(f"\nПример записи:")
    print(json.dumps(train_records[0], ensure_ascii=False, indent=2))


def _write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    out_dir = Path(__file__).parent / "input"

    records = stream_reviews(
        max_scan=10000,
        total_needed=1000,
    )

    split_and_save(
        records=records,
        total=1000,
        eval_size=30,
        seed=42,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
