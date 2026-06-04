"""
Финал — Сборка конвейера
==========================
Всё, что мы разложили по раундам 1-7, теперь собираем в один analyze().
Никаких новых концепций — только связывание. На входе путь к транскрипту,
на выходе папка output/ со всеми артефактами.

Задача:
  Дописать analyze(transcript_path, out_dir). Запустить, проверить,
  что в out_dir/ появилось:
    • participants.json + participants.csv
    • aspects.json + heatmap.png
    • summary.json
    • judge_report.json
    • metrics.json (полнота/точность/достоверность)

Запуск:
    python 12_pipeline.py transcript.txt
    python 12_pipeline.py transcripts/dom_bank.txt output/dom
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

_p = importlib.import_module("2_extract_participants")
extract_participants = _p.extract_participants
_a = importlib.import_module("4_extract_aspects")
extract_aspects = _a.extract_aspects
check_quotes = _a.check_quotes
build_heatmap = _a.build_heatmap
_mr = importlib.import_module("7_map_reduce")
summarize_discussion = _mr.summarize_discussion
_j = importlib.import_module("9_judge")
judge = _j.judge
_eval = importlib.import_module("3_evaluate_ie")
fidelity = _eval.fidelity


def analyze(transcript_path: str, out_dir: str = "output") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    transcript = Path(transcript_path).read_text(encoding="utf-8")

    print("→ Акт 1: extract_participants...")
    participants = extract_participants(transcript)
    parts_data = [p.model_dump() for p in participants]
    (out / "participants.json").write_text(
        json.dumps(parts_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "name": p.name,
                "age": p.age,
                "city": p.city,
                "occupation": p.occupation,
                "n_concerns": len(p.concerns),
                "n_competitors": len(p.competitor_mentions),
            }
            for p in participants
        ]
    ).to_csv(out / "participants.csv", index=False, encoding="utf-8")
    print(
        f"   {len(participants)} участников, "
        f"{sum(len(p.concerns) for p in participants)} жалоб"
    )

    print("→ Акт 2: extract_aspects...")
    aspects = extract_aspects(transcript)
    ghosts = check_quotes(aspects, transcript)
    if ghosts:
        print(f"   ⚠ {len(ghosts)} цитат не найдено в тексте")
    build_heatmap(aspects, out_path=str(out / "heatmap.png"))
    pd.DataFrame(
        [
            {
                "name": p.name,
                "aspect": a.aspect,
                "sentiment": a.sentiment,
                "confidence": a.confidence,
                "quote": a.quote,
            }
            for p in aspects
            for a in p.aspects
        ]
    ).to_csv(out / "aspects.csv", index=False, encoding="utf-8")

    print("→ Акт 3: summarize_discussion...")
    summary = summarize_discussion(transcript)
    (out / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )

    print("→ Акт 5: модель-судья...")
    report = judge(parts_data, json.loads(summary.model_dump_json()))
    (out / "judge_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )

    print("→ Метрики качества...")
    metrics = {"fidelity": fidelity(parts_data, transcript)}
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\n=== ИТОГ ===")
    print(summary.headline)
    print("\nКлючевые выводы:")
    for kf in summary.key_findings:
        print(f"  • {kf}")
    print("\nРекомендации:")
    for ai in summary.action_items:
        print(f"  → {ai}")
    print(f"\nоценка судьи: {report.overall_score:.2f}")
    print(f"достоверность: {metrics['fidelity']:.0%}")
    print(f"\nВсе артефакты в: {out}/")


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python 12_pipeline.py <transcript.txt> [out_dir]")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")


if __name__ == "__main__":
    main()
