"""
pipeline.py
-----------
Главный оркестратор пайплайна.

"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from agents import Analyst, Writer, Judge
from hallucination import HallucinationChecker
from llm_client import make_client, get_model
from rag import RetrieverRAG, build_index
from schema import (
    AspectSummaryEntry,
    Aspect,
    ProductSummary,
    ReviewInput,
    ReviewResult,
    Sentiment,
)


REVIEWS_PATH = Path("input/reviews.jsonl")
OUTPUT_DIR = Path("output")
RESULTS_PATH = OUTPUT_DIR / "results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "product_summaries.jsonl"


# ---------------------------------------------------------------------------
# Обработка одного отзыва
# ---------------------------------------------------------------------------

def process_review(
        review: ReviewInput,
        analyst: Analyst,
        writer: Writer,
        judge: Judge,
        checker: HallucinationChecker,
) -> ReviewResult:
    """
    Прогоняет один отзыв через все три агента и проверку галлюцинаций.
    """
    analysis = analyst.run(review)
    reply = writer.run(review, analysis)
    halluc = checker.run(review, reply)
    verdict = judge.run(review, analysis, reply)

    return ReviewResult(
        input=review,
        analysis=analysis,
        reply=reply,
        hallucination_report=halluc,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Reduce: сводка по товару (multi-document summary)
# ---------------------------------------------------------------------------

def _reduce_product(
        nm_id: int,
        results: list[ReviewResult],
) -> ProductSummary:

    client = make_client()
    model = get_model()

    reviews_text = "\n\n".join(
        f"[Отзыв {i + 1}] Оценка {r.input.rating}\n"
        f"Текст: {r.input.text}\n"
        f"Аспекты: {', '.join(e.aspect.value + ': ' + e.problem for e in r.analysis.aspects)}"
        for i, r in enumerate(results)
    )

    avg_rating = sum(r.input.rating for r in results) / len(results)

    aspect_counts: dict[Aspect, list[str]] = defaultdict(list)
    for r in results:
        for entry in r.analysis.aspects:
            aspect_counts[entry.aspect].append(entry.problem)

    sentiment_counts: dict[Sentiment, int] = defaultdict(int)
    for r in results:
        sentiment_counts[r.analysis.sentiment] += 1
    dominant_sentiment = max(sentiment_counts, key=lambda s: sentiment_counts[s])

    aspect_summaries = [
        AspectSummaryEntry(
            aspect=aspect,
            mention_count=len(problems),
            common_problems=problems[:3],
        )
        for aspect, problems in sorted(
            aspect_counts.items(), key=lambda x: -len(x[1])
        )
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "Ты аналитик клиентского сервиса. "
                "На основе нескольких отзывов на один товар напиши краткое "
                "резюме для продавца: главные проблемы, паттерны, рекомендации. "
                "3–5 предложений, конкретно и по делу."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Товар nm_id={nm_id}, средняя оценка {avg_rating:.1f}\n\n"
                f"{reviews_text}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=ProductSummary,
        max_retries=2,
    )

    return ProductSummary(
        nm_id=nm_id,
        review_count=len(results),
        avg_rating=round(avg_rating, 2),
        dominant_sentiment=dominant_sentiment,
        aspect_summaries=aspect_summaries,
        executive_summary=resp.executive_summary,
    )


# ---------------------------------------------------------------------------
# Главный запуск
# ---------------------------------------------------------------------------

def run_pipeline(
        reviews_path: Path = REVIEWS_PATH,
        limit: int | None = None,
) -> tuple[list[ReviewResult], list[ProductSummary]]:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rag = RetrieverRAG()
    analyst = Analyst()
    writer = Writer(rag)
    judge = Judge()
    checker = HallucinationChecker()

    records: list[ReviewInput] = []
    with open(reviews_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(ReviewInput(**json.loads(line)))
            if limit and len(records) >= limit:
                break

    print(f"Отзывов к обработке: {len(records)}")

    all_results: list[ReviewResult] = []
    by_nm_id: dict[int, list[ReviewResult]] = defaultdict(list)

    with open(RESULTS_PATH, "w", encoding="utf-8") as out:
        for i, review in enumerate(records, 1):
            print(f"[{i}/{len(records)}] nm_id={review.nm_id} rating={review.rating} ...", end=" ")
            try:
                result = process_review(review, analyst, writer, judge, checker)
                all_results.append(result)
                by_nm_id[review.nm_id].append(result)

                out.write(result.model_dump_json(ensure_ascii=False) + "\n")
                out.flush()

                h = result.hallucination_report
                print(f"  judge={result.verdict.overall_score}  halluc={h.hallucination_count}")

            except Exception as e:
                print(f" ошибка: {e}")

    summaries: list[ProductSummary] = []
    reducible = {k: v for k, v in by_nm_id.items() if len(v) >= 2}
    print(f"Reduce: {len(reducible)} товаров с ≥2 отзывами")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as out:
        for nm_id, group in reducible.items():
            print(f"  Reduce nm_id={nm_id} ({len(group)} отзывов)...", end=" ")
            try:
                summary = _reduce_product(nm_id, group)
                summaries.append(summary)
                out.write(summary.model_dump_json(ensure_ascii=False) + "\n")
                out.flush()
                print("✓")
            except Exception as e:
                print(f"✗  {e}")

    _print_stats(all_results)

    return all_results, summaries


def _print_stats(results: list[ReviewResult]) -> None:
    if not results:
        return

    total = len(results)
    total_halluc = sum(r.hallucination_report.hallucination_count for r in results)
    with_halluc = sum(1 for r in results if r.hallucination_report.has_hallucinations)
    avg_judge = sum(r.verdict.overall_score for r in results) / total
    better = sum(1 for r in results if r.verdict.better_than_original)
    mismatches = sum(1 for r in results if r.analysis.rating_mismatch)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Итоги пайплайна ({total} отзывов)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Средняя оценка судьи:       {avg_judge:.2f} / 5
Ответов лучше оригинала:    {better} / {total} ({better / total * 100:.0f}%)
Галлюцинаций найдено:       {total_halluc} в {with_halluc} отзывах
Несоответствий оценки/текста: {mismatches}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Артефакты:
  {RESULTS_PATH}
  {SUMMARY_PATH}
""")