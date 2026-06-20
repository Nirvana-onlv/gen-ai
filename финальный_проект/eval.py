"""
eval.py
-------
Оценка пайплайна на отложенном тестовом наборе данных.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from agents import Analyst, Writer, Judge
from hallucination import HallucinationChecker
from pipeline import process_review
from rag import RetrieverRAG
from schema import ReviewInput, ReviewResult


EVAL_PATH = Path("input/reviews_eval.jsonl")
OUTPUT_DIR = Path("output")
EVAL_OUTPUT = OUTPUT_DIR / "eval_results.jsonl"
EVAL_TABLE = OUTPUT_DIR / "eval_table.csv"


# ---------------------------------------------------------------------------
# Структура одного eval-результата
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    idx: int
    rating: int
    text_preview: str

    sentiment: str
    aspects_found: list[str]
    judge_score: int
    better_than_orig: bool
    rating_mismatch: bool

    rag_used: int
    halluc_count: int
    has_hallucinations: bool
    halluc_fragments: list[str]

    success: bool
    error: str = ""
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Агрегированные метрики
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    total: int = 0
    passed: int = 0

    # Правильность
    judge_scores: list[int] = field(default_factory=list)
    better_count: int = 0
    mismatch_count: int = 0

    # Путь
    halluc_total: int = 0
    halluc_cases: int = 0
    rag_used_total: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_judge_score(self) -> float:
        return sum(self.judge_scores) / len(self.judge_scores) if self.judge_scores else 0.0

    @property
    def better_rate(self) -> float:
        return self.better_count / self.passed if self.passed else 0.0

    @property
    def halluc_rate(self) -> float:
        return self.halluc_cases / self.passed if self.passed else 0.0

    @property
    def mismatch_rate(self) -> float:
        return self.mismatch_count / self.passed if self.passed else 0.0

    @property
    def rag_used_avg(self) -> float:
        return self.rag_used_total / self.passed if self.passed else 0.0


# ---------------------------------------------------------------------------
# Запуск eval
# ---------------------------------------------------------------------------


def _ensure_index() -> None:
    """Строит RAG-индекс если коллекция ещё не существует."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from rag import build_index, CHROMA_DIR, COLLECTION

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    existing = [c.name for c in client.list_collections()]
    if COLLECTION not in existing:
        print("RAG-индекс не найден, строим...")
        build_index()
        print()

def run_eval(
        eval_path: Path = EVAL_PATH,
        limit: int | None = None,
) -> EvalMetrics:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Загрузка тест-сета
    records: list[ReviewInput] = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(ReviewInput(**json.loads(line)))
            if limit and len(records) >= limit:
                break

    print(f"Eval на {len(records)} отзывах\n{'─' * 50}")

    _ensure_index()
    rag = RetrieverRAG()
    analyst = Analyst()
    writer = Writer(rag)
    judge = Judge()
    checker = HallucinationChecker()

    eval_records: list[EvalRecord] = []
    metrics = EvalMetrics(total=len(records))

    with open(EVAL_OUTPUT, "w", encoding="utf-8") as out:
        for i, review in enumerate(records, 1):
            print(f"[{i}/{len(records)}] rating={review.rating}"
                  f"{review.text[:60]}...", end=" ")

            t0 = time.perf_counter()
            try:
                result: ReviewResult = process_review(
                    review, analyst, writer, judge, checker
                )
                elapsed = time.perf_counter() - t0

                rec = EvalRecord(
                    idx=i,
                    rating=review.rating,
                    text_preview=review.text[:100],
                    sentiment=result.analysis.sentiment.value,
                    aspects_found=[e.aspect.value for e in result.analysis.aspects],
                    judge_score=result.verdict.overall_score,
                    better_than_orig=result.verdict.better_than_original,
                    rating_mismatch=result.analysis.rating_mismatch,
                    rag_used=result.reply.rag_examples_used,
                    halluc_count=result.hallucination_report.hallucination_count,
                    has_hallucinations=result.hallucination_report.has_hallucinations,
                    halluc_fragments=[
                        g.fragment for g in result.hallucination_report.ghost_items
                    ],
                    success=True,
                    elapsed_sec=round(elapsed, 2),
                )

                metrics.passed += 1
                metrics.judge_scores.append(result.verdict.overall_score)
                metrics.better_count += int(result.verdict.better_than_original)
                metrics.mismatch_count += int(result.analysis.rating_mismatch)
                metrics.halluc_total += result.hallucination_report.hallucination_count
                metrics.halluc_cases += int(result.hallucination_report.has_hallucinations)
                metrics.rag_used_total += result.reply.rag_examples_used

                print(f"  judge={rec.judge_score}  "
                      f"halluc={rec.halluc_count}  "
                      f"rag={rec.rag_used}  "
                      f"{elapsed:.1f}s")

            except Exception as e:
                elapsed = time.perf_counter() - t0
                rec = EvalRecord(
                    idx=i,
                    rating=review.rating,
                    text_preview=review.text[:100],
                    sentiment="",
                    aspects_found=[],
                    judge_score=0,
                    better_than_orig=False,
                    rating_mismatch=False,
                    rag_used=0,
                    halluc_count=0,
                    has_hallucinations=False,
                    halluc_fragments=[],
                    success=False,
                    error=str(e),
                    elapsed_sec=round(elapsed, 2),
                )
                print(f"✗  {e}")

            eval_records.append(rec)
            out.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
            out.flush()

    _save_csv(eval_records)

    _print_report(metrics, eval_records)

    return metrics


# ---------------------------------------------------------------------------
# CSV и отчёт
# ---------------------------------------------------------------------------

def _save_csv(records: list[EvalRecord]) -> None:
    headers = [
        "idx", "rating", "sentiment", "aspects_found",
        "judge_score", "better_than_orig", "rating_mismatch",
        "rag_used", "halluc_count", "has_hallucinations",
        "success", "elapsed_sec", "error",
    ]
    with open(EVAL_TABLE, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in records:
            row = [
                r.idx,
                r.rating,
                r.sentiment,
                "|".join(r.aspects_found),
                r.judge_score,
                int(r.better_than_orig),
                int(r.rating_mismatch),
                r.rag_used,
                r.halluc_count,
                int(r.has_hallucinations),
                int(r.success),
                r.elapsed_sec,
                r.error.replace(",", ";"),
            ]
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nCSV сохранён: {EVAL_TABLE}")


def _print_report(metrics: EvalMetrics, records: list[EvalRecord]) -> None:
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVAL REPORT  ({metrics.total} отзывов)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ПРАВИЛЬНОСТЬ
  pass_rate:         {metrics.pass_rate:.0%}  ({metrics.passed}/{metrics.total})
  avg_judge_score:   {metrics.avg_judge_score:.2f} / 5
  better_rate:       {metrics.better_rate:.0%}  (ответов лучше оригинала)
  mismatch_rate:     {metrics.mismatch_rate:.0%}  (оценка ≠ тональность текста)

ПУТЬ
  halluc_rate:       {metrics.halluc_rate:.0%}  (отзывов с галлюцинациями)
  halluc_total:      {metrics.halluc_total}  (всего галлюцинаций)
  rag_used_avg:      {metrics.rag_used_avg:.1f}  (RAG-примеров на ответ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")

    halluc_cases = [r for r in records if r.has_hallucinations and r.success]
    if halluc_cases:
        print("\nПримеры галлюцинаций (для отчёта):")
        for r in halluc_cases[:3]:
            print(f"\n  [{r.idx}] rating={r.rating}")
            print(f"  Отзыв: {r.text_preview}...")
            for frag in r.halluc_fragments[:2]:
                print(f"  «{frag}»")

    failures = [r for r in records if not r.success]
    if failures:
        print(f"\nПровалы ({len(failures)}):")
        for r in failures[:3]:
            print(f"  [{r.idx}] {r.error[:120]}")


if __name__ == "__main__":
    run_eval()