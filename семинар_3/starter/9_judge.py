"""
Раунд 5 — Модель-судья
========================
У нас есть summary.json (раунд 3) и participants.json (раунд 1).
Возникает вопрос: а action_items из summary правда подкреплены теми
жалобами, что нашлись в participants? Или модель «фантазирует на тему»?

В раунде 1.5 мы уже проверяли полноту и достоверность по эталону.
Теперь делаем то же без участия человека — модель сама себе судья.

Задача:
  1. В schema.py: ActionVerdict, JudgeReport.
  2. В prompts.py: JUDGE_SYSTEM — критический тон, «не оправдывай,
     ищи противоречия».
  3. judge() — взять summary.json + participants.json + (опц. aspects.json),
     отправить судье, получить JudgeReport.
  4. Сохранить judge_report.json.

Бонус (если есть время):
  • Прогнать judge с двумя моделями (deepseek-v4-flash как генератор,
    deepseek-v4-pro как судья). Сравнить вердикты.
  • Прогнать summary дважды с разным seed — проверка согласованности.

Запуск:
    python 9_judge.py
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import get_model, make_client
from prompts import JUDGE_SYSTEM
from schema import JudgeReport

client = make_client()
MODEL = get_model()


def load_artifacts() -> tuple[list[dict], dict]:
    p = Path("participants.json")
    s = Path("summary.json")
    if not p.exists():
        raise SystemExit("Сначала раунд 1: python 2_extract_participants.py")
    if not s.exists():
        raise SystemExit("Сначала раунд 3: python 7_map_reduce.py")
    return (
        json.loads(p.read_text(encoding="utf-8")),
        json.loads(s.read_text(encoding="utf-8")),
    )


def build_evidence_packet(participants: list[dict], summary: dict) -> str:
    parts = ["## Рекомендации (которые оцениваем)"]
    for i, a in enumerate(summary.get("action_items", []), 1):
        parts.append(f"  {i}. {a}")
    parts.append("\n## Жалобы участников (исходные данные)")
    for p in participants:
        for c in p.get("concerns", []):
            parts.append(
                f"  - [{p['name']}/{c['category']}, sev={c['severity']}] «{c['quote']}»"
            )
    return "\n".join(parts)


def judge(participants: list[dict], summary: dict) -> JudgeReport:
    evidence = build_evidence_packet(participants, summary)
    return client.chat.completions.create(
        model=MODEL,
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": evidence},
        ],
    )


def main() -> None:
    participants, summary = load_artifacts()
    report = judge(participants, summary)

    print(f"━━━ Вердикты по {len(report.verdicts)} рекомендациям ━━━")
    counts = {"supported": 0, "weakly_supported": 0, "not_supported": 0}
    for v in report.verdicts:
        counts[v.support] += 1
        mark = {"supported": "✓", "weakly_supported": "?", "not_supported": "✗"}[
            v.support
        ]
        print(f"\n  {mark} [{v.support}] {v.action}")
        for e in v.evidence:
            print(f"      ← «{e[:100]}»")
        print(f"      → {v.comment}")

    print("\n━━━ Сводка ━━━")
    print(f"  supported:        {counts['supported']}")
    print(f"  weakly_supported: {counts['weakly_supported']}")
    print(f"  not_supported:    {counts['not_supported']}")
    print(f"  overall_score:    {report.overall_score:.2f}")
    print(f"\n  {report.summary}")

    Path("judge_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    print("\nСохранено: judge_report.json")


if __name__ == "__main__":
    main()
