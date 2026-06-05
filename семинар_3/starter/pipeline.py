from __future__ import annotations

import json
import time
import pandas as pd
import numpy as np
import threading
import matplotlib.pyplot as plt
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_client import get_model, make_client
from prompts import IE_SYSTEM, ASPECTS_SYSTEM, CHUNK_SYSTEM, REDUCE_SYSTEM, JUDGE_SYSTEM, DISCOVER_SYSTEM, DYNAMIC_ASPECTS_SYSTEM
from schema import Review, ReviewSentiment, ChunkSummary, ReviewSummary, JudgeReport, DiscoveredAspects, DynamicReview


client = make_client()
MODEL = get_model()
CHUNK_SIZE = 10
total_usage = {"input_tokens": 0, "output_tokens": 0}
usage_lock = threading.Lock()

def extract_review(transcript: str) -> Review:

    response, completion = client.chat.completions.create(
        model=MODEL,
        response_model=Review,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": IE_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        with_completion=True,
    )
    total_usage["input_tokens"] += completion.usage.prompt_tokens
    total_usage["output_tokens"] += completion.usage.completion_tokens
    return response

def extract_aspects(transcript: str) -> ReviewSentiment:
    response, completion = client.chat.completions.create(
        model=MODEL,
        response_model=ReviewSentiment,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": ASPECTS_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        with_completion=True,
    )
    total_usage["input_tokens"] += completion.usage.prompt_tokens
    total_usage["output_tokens"] += completion.usage.completion_tokens
    return response

def build_heatmap(aspects_res: list) -> None:
    ASPECTS = ["battery", "camera", "performance", "design", "price", "software"]
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}

    totals = {a: [] for a in ASPECTS}
    for rev in aspects_res:
        for asp in rev.aspects:
            if asp.aspect in totals:
                totals[asp.aspect].append(sentiment_map[asp.sentiment])

    means = [sum(v) / len(v) if v else float("nan") for a, v in totals.items()]
    counts = [len(v) for v in totals.values()]

    values = np.array(means).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(ASPECTS)))
    ax.set_xticklabels(
        [f"{a}\n(n={c})" for a, c in zip(ASPECTS, counts)],
        fontsize=11,
    )
    ax.set_yticks([])
    ax.set_title("Средний sentiment по аспектам  |  -1 негативный · 0 нейтральный · +1 позитивный")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=150)
    plt.close()
    print("Сохранено: heatmap.png")

def discover_aspects(transcript: str) -> DiscoveredAspects:
    response, completion = client.chat.completions.create(
        model=MODEL,
        response_model=DiscoveredAspects,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": DISCOVER_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        with_completion=True,
    )
    total_usage["input_tokens"] += completion.usage.prompt_tokens
    total_usage["output_tokens"] += completion.usage.completion_tokens
    return response

def autodiscovery(select_reviews: list[str], aspects_res: list) -> None:

    combined = "\n\n---\n\n".join(
        f"[{i}] {text}" for i, text in enumerate(select_reviews)
    )
    discovered = discover_aspects(combined)
    print(f"Найдено тем: {len(discovered.aspects)}")
    for a in discovered.aspects:
        print(f"  • {a.name} — {a.description}")

    # Шаг 2 — классифицируем каждый отзыв по найденным темам
    dynamic_aspects_block = "\n".join(
        f"- {a.name}: {a.description}" for a in discovered.aspects
    )
    sys_prompt = ASPECTS_SYSTEM + "\n\nИспользуй СТРОГО эти аспекты:\n" + dynamic_aspects_block

    dynamic_res = []
    for idx, review in enumerate(select_reviews):
        response, completion = client.chat.completions.create(
            model=MODEL,
            response_model=DynamicReview,
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": review},
            ],
            with_completion=True,
        )
        total_usage["input_tokens"] += completion.usage.prompt_tokens
        total_usage["output_tokens"] += completion.usage.completion_tokens
        response.review_id = str(idx)
        dynamic_res.append(response)

    fixed_aspects = {a["aspect"] for r in [r.model_dump() for r in aspects_res] for a in r["aspects"]}
    dyn_aspects = {a.aspect for r in dynamic_res for a in r.aspects}
    new = dyn_aspects - fixed_aspects
    missing = fixed_aspects - dyn_aspects

    print(f"\nСравнение с раундом 2:")
    print(f"  Literal-аспекты:    {sorted(fixed_aspects)}")
    print(f"  Найденные аспекты:  {sorted(dyn_aspects)}")
    if new:
        print(f"  новые темы:       {sorted(new)}")
    if missing:
        print(f"  не обсуждались:   {sorted(missing)}")

    Path("autodiscovery.json").write_text(
        json.dumps([r.model_dump() for r in dynamic_res], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Сохранено: autodiscovery.json")


def split_into_chunks(reviews: list[str], chunk_size: int = CHUNK_SIZE) -> list[tuple[list[str], str]]:
    chunks = []
    for i in range(0, len(reviews), chunk_size):
        batch = reviews[i : i + chunk_size]
        ids = [str(i + j) for j in range(len(batch))]
        chunk_text = "\n\n---\n\n".join(
            f"[{rid}] {text}" for rid, text in zip(ids, batch)
        )
        chunks.append((ids, chunk_text))
    return chunks


def summarize_chunk(ids: list[str], chunk: str) -> ChunkSummary:
    response, completion = client.chat.completions.create(
        model=MODEL,
        response_model=ChunkSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": CHUNK_SYSTEM},
            {
                "role": "user",
                "content": f"review_ids в этом чанке: {ids}\n\n{chunk}",
            },
        ],
        with_completion=True,
    )
    with usage_lock:
        total_usage["input_tokens"] += completion.usage.prompt_tokens
        total_usage["output_tokens"] += completion.usage.completion_tokens
    return response


def reduce_summaries(summaries: list[ChunkSummary]) -> ReviewSummary:
    joined = "\n\n".join(
        f"## Группа {i + 1} (отзывы {s.review_ids}, {s.sentiment})\n"
        + "\n".join(f"- {p}" for p in s.key_points)
        for i, s in enumerate(summaries)
    )
    response, completion = client.chat.completions.create(
        model=MODEL,
        response_model=ReviewSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": joined},
        ],
        with_completion=True,
    )
    total_usage["input_tokens"] += completion.usage.prompt_tokens
    total_usage["output_tokens"] += completion.usage.completion_tokens
    return response

def map_reduce(reviews: list[str], workers: int = 5) -> ReviewSummary:
    chunks = split_into_chunks(reviews)
    n = len(chunks)
    print(f"  [MR] MAP: {n} чанков по ~{CHUNK_SIZE} отзывов, до {workers} параллельно...")

    t0 = time.time()
    summaries: list[ChunkSummary | None] = [None] * n

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(summarize_chunk, ids, chunk): i
            for i, (ids, chunk) in enumerate(chunks)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            summaries[i] = fut.result()
            done += 1
            print(f"  [MR] {done}/{n} готов ({time.time() - t0:.1f}с)")

    print(f"  [MR] MAP {time.time() - t0:.1f}с → REDUCE...")
    result = reduce_summaries([s for s in summaries if s is not None])
    print(f"  [MR] всего {time.time() - t0:.1f}с")
    return result

def build_evidence_packet(reviews: list[dict], summary: dict) -> str:
    parts = ["## Рекомендации (которые оцениваем)"]
    for i, a in enumerate(summary.get("action_items", []), 1):
        parts.append(f"  {i}. {a}")

    parts.append("\n## Жалобы из отзывов (исходные данные)")
    for r in reviews:
        for issue in r.get("issues", []):
            parts.append(
                f"  - [review={r['review_id']}/{issue['category']}, sev={issue['severity']}] «{issue['quote']}»"
            )
    return "\n".join(parts)


def run_judge(reviews: list[dict], summary: ReviewSummary) -> JudgeReport:
    evidence = build_evidence_packet(reviews, summary.model_dump())
    response, completion = client.chat.completions.create(
        model=MODEL,
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": evidence},
        ],
        with_completion=True,
    )
    total_usage["input_tokens"] += completion.usage.prompt_tokens
    total_usage["output_tokens"] += completion.usage.completion_tokens
    return response


def main() -> None:
    reviews = pd.read_csv("data.csv")
    select_reviews = reviews.loc[:50]["Review"].tolist()

    ie_res = []
    for idx, review in enumerate(select_reviews):
        parsed_review = extract_review(review)
        parsed_review.review_id = str(idx)
        ie_res.append(parsed_review)
        print(parsed_review)

    out = [r.model_dump() for r in ie_res]
    Path("reviews.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Сохранено: reviews.json")

    aspects_res = []
    ghosts = []

    for idx, review in enumerate(select_reviews):
        parsed_review = extract_aspects(review)
        parsed_review.review_id = str(idx)
        aspects_res.append(parsed_review)
        print(parsed_review)

        original = review.lower()
        for asp in parsed_review.aspects:
            probe = asp.quote.strip().lower()[:30]
            if probe and probe not in original:
                ghosts.append((str(idx), asp.aspect, asp.quote))

    total_quotes = sum(len(r.aspects) for r in aspects_res)
    print(f"Ghost-цитат: {len(ghosts)}/{total_quotes} ({len(ghosts) / total_quotes:.1%})")

    out = [r.model_dump() for r in aspects_res]
    Path("aspects.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Сохранено: aspects.json")

    build_heatmap(aspects_res)

    autodiscovery(select_reviews, aspects_res)

    summary = map_reduce(select_reviews)

    print("\n━━━ ИТОГ ━━━")
    print(summary.headline)
    print("\nКлючевые выводы:")
    for kf in summary.key_findings:
        print(f"  • {kf}")
    print("\nРекомендации:")
    for ai in summary.action_items:
        print(f"  → {ai}")

    Path("summary.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print("\nСохранено: summary.json")

    report = run_judge([r.model_dump(mode="json") for r in ie_res], summary)

    print(f"\n━━━ Judge: {len(report.verdicts)} рекомендаций ━━━")
    for v in report.verdicts:
        mark = {"supported": "✓", "weakly_supported": "?", "not_supported": "✗"}[v.support]
        print(f"  {mark} [{v.support}] {v.action}")
        for e in v.evidence:
            print(f"      ← «{e[:100]}»")
        print(f"      → {v.comment}")

    print(f"\n  overall_score: {report.overall_score:.2f}")
    print(f"  {report.summary}")

    Path("judge_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    print("Сохранено: judge_report.json")

    INPUT_PRICE = 0.14
    OUTPUT_PRICE = 0.28

    cost = (
        total_usage["input_tokens"] / 1_000_000 * INPUT_PRICE +
        total_usage["output_tokens"] / 1_000_000 * OUTPUT_PRICE
    )
    print(f"\n━━━ Использование токенов ━━━")
    print(f"  Input:   {total_usage['input_tokens']:,}")
    print(f"  Output:  {total_usage['output_tokens']:,}")
    print(f"  Стоимость: ${cost:.4f}")

if __name__ == "__main__":
    main()