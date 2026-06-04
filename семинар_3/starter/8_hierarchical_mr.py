"""
Раунд 3.5 — Иерархический Map-Reduce
======================================
В раунде 3 REDUCE-промпт принимает ВСЕ мини-резюме сразу. Это работает
до тех пор, пока их сумма помещается в контекстное окно. Что если
участников 50 и каждое мини-резюме на 500 токенов? = 25k токенов
только в REDUCE. У DeepSeek окно 64k, у gpt-4o-mini 128k — пока влезает.
А если участников 500? Или каждое мини-резюме растёт с длиной
транскрипта?

Задача:
  Реализовать ТРЁХУРОВНЕВЫЙ Map-Reduce:

  • Уровень 1 (MAP):     фрагмент → ChunkSummary  (как в раунде 3)
  • Уровень 2 (GROUP):   группы по 5-10 ChunkSummary → GroupSummary
                          (новая модель — см. schema.py)
  • Уровень 3 (REDUCE):  все GroupSummary → DiscussionSummary

  На нашем коротком transcript эффект будет минимален (всего 4 фрагмента,
  иерархия избыточна). Но запустить и УВИДЕТЬ структуру — критично.
  В раунде 7 (многодокументном) на 5 транскриптах разница станет видимой.

Запуск:
    python 8_hierarchical_mr.py
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

from llm_client import get_model, make_client
from prompts import GROUP_REDUCE_SYSTEM, REDUCE_SYSTEM
from schema import ChunkSummary, DiscussionSummary, GroupSummary

_mr = importlib.import_module("7_map_reduce")
summarize_chunk = _mr.summarize_chunk
_split_mod = importlib.import_module("6_split_chunking")
split_by_speaker = _split_mod.split_by_speaker

client = make_client()
MODEL = get_model()

GROUP_SIZE = 5


def reduce_group(group: list[ChunkSummary]) -> GroupSummary:
    joined = "\n\n".join(
        f"## {s.speaker} ({s.sentiment})\n" + "\n".join(f"- {p}" for p in s.key_points)
        for s in group
    )
    return client.chat.completions.create(
        model=MODEL,
        response_model=GroupSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": GROUP_REDUCE_SYSTEM},
            {"role": "user", "content": joined},
        ],
    )


def reduce_final(groups: list[GroupSummary]) -> DiscussionSummary:
    joined = "\n\n".join(
        f"## группа {i + 1} (sentiment={g.overall_sentiment}, "
        f"speakers={', '.join(g.speakers)})\n" + "\n".join(f"- {t}" for t in g.themes)
        for i, g in enumerate(groups)
    )
    return client.chat.completions.create(
        model=MODEL,
        response_model=DiscussionSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": joined},
        ],
    )


def hierarchical_summary(
    transcript: str, group_size: int = GROUP_SIZE
) -> DiscussionSummary:
    chunks = split_by_speaker(transcript)
    print(f"  [HMR] L1 MAP: {len(chunks)} фрагментов...")
    t0 = time.time()
    summaries = [summarize_chunk(c) for c in chunks]
    print(f"  [HMR] L1 готов ({time.time() - t0:.1f}с)")

    groups_chunks = [
        summaries[i : i + group_size] for i in range(0, len(summaries), group_size)
    ]
    print(f"  [HMR] L2 GROUP: {len(groups_chunks)} групп...")
    t1 = time.time()
    groups = [reduce_group(g) for g in groups_chunks]
    print(f"  [HMR] L2 готов ({time.time() - t1:.1f}с)")

    print(f"  [HMR] L3 REDUCE: {len(groups)} групповых резюме...")
    t2 = time.time()
    final = reduce_final(groups)
    print(f"  [HMR] L3 готов ({time.time() - t2:.1f}с)")
    print(f"  [HMR] всего {time.time() - t0:.1f}с, {len(chunks)} → {len(groups)} → 1")
    return final


def main() -> None:
    transcript = Path("transcript.txt").read_text(encoding="utf-8")
    summary = hierarchical_summary(transcript)

    print("\n━━━ ИТОГ (иерархический) ━━━")
    print(summary.headline)
    for kf in summary.key_findings:
        print(f"  • {kf}")

    Path("summary_hierarchical.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    print("\nСохранено: summary_hierarchical.json")


if __name__ == "__main__":
    main()
