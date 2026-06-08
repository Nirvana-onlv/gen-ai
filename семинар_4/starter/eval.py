"""
Eval по 10 gold-вопросам. Метрика: hit-rate@5 на уровне документа-источника.

Правило: если в ТОП-5 чанков встретился хотя бы один чанк из gold_sources —
вопрос зачтён как HIT. Для вопросов, которым необходимы несколько чанков, считаем как долю найденных
источников (например, 2 из 3 → 0.67).

Команды:
    python eval.py --naive         # прогнать текущую конфигурацию pipeline.py
"""

import argparse
import json
from pathlib import Path

from pipeline import collection_fixed, collection_rec, dense_retrieve, hybrid_retrieve, build_prompt, client, MODEL

GOLD_PATH = Path(__file__).parent / "wiki_data" / "gold.json"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    """
    Для одного вопроса: сколько из gold_sources попали в ТОП-K чанков.
    retrieved_ids = ['olymp_anna__0', 'tinkoff_alex__2', ...]
    Мы смотрим только на префикс до '__' — это source_id.
    """
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources)


def run(use_hybrid: bool, collection, label: str,
        k: int = 5, verbose: bool = True) -> dict:
    gold = load_gold()
    total = 0.0
    results = []

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}\n")

    for item in gold:
        q = item["question"]
        gold_sources = item["gold_sources"]

        if use_hybrid:
            hits = hybrid_retrieve(q, collection, k=k)
        else:
            hits = dense_retrieve(q, collection, k=k)

        retrieved_ids = hits["ids"][0]
        retrieved_docs = [rid.split("__")[0] for rid in retrieved_ids]

        score = hit_rate(retrieved_ids, gold_sources)
        total += score

        # Генерируем ответ LLM
        prompt = build_prompt(q, hits)
        resp = client._c.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""

        results.append({
            "id": item["id"],
            "type": item["type"],
            "question": q,
            "score": score,
            "gold": gold_sources,
            "retrieved_docs": retrieved_docs,
            "answer": answer,
        })

        if verbose:
            mark = "✓" if score == 1.0 else ("◐" if score > 0 else "✗")
            print(
                f"  [{item['id']:2d}] {item['type']:<15}  "
                f"hit@{k}={score:.2f} {mark}  {q[:55]}"
            )

    mean = total / len(gold)
    if verbose:
        print(f"\n  hit-rate@{k} = {mean:.3f}  ({total:.1f} / {len(gold)})\n")

    return {"mean": mean, "results": results}


def compare(use_hybrid: bool, k: int, verbose: bool):
    mode = "HYBRID (dense + BM25 + RRF)" if use_hybrid else "DENSE ONLY"

    res_fixed = run(
        use_hybrid=use_hybrid,
        collection=collection_fixed,
        label=f"Стратегия A — fixed-size  |  {mode}",
        k=k,
        verbose=verbose,
    )
    res_rec = run(
        use_hybrid=use_hybrid,
        collection=collection_rec,
        label=f"Стратегия B — recursive   |  {mode}",
        k=k,
        verbose=verbose,
    )

    # Итоговая таблица
    print("=" * 55)
    print("  СРАВНЕНИЕ СТРАТЕГИЙ")
    print("=" * 55)
    print(f"  {'Стратегия':<30} hit-rate@{k}")
    print(f"  {'-' * 40}")
    print(f"  {'A: fixed-size (2000, no overlap)':<30} {res_fixed['mean']:.3f}")
    print(f"  {'B: recursive (512, overlap=80)':<30} {res_rec['mean']:.3f}")
    winner = "B (recursive)" if res_rec["mean"] >= res_fixed["mean"] else "A (fixed-size)"
    print(f"\n  Победитель: {winner}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = {
        "mode": mode,
        "k": k,
        "strategy_A": {"name": "fixed-size (2000)", "hit_rate": res_fixed["mean"], "details": res_fixed["results"]},
        "strategy_B": {"name": "recursive (512, overlap=80)", "hit_rate": res_rec["mean"],
                       "details": res_rec["results"]},
    }
    out_path = out_dir / "eval_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Результаты сохранены: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-only", action="store_true", help="только dense-поиск")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if collection_fixed.count() == 0 or collection_rec.count() == 0:
        print("⚠ Коллекции пустые. Запусти: python pipeline.py ingest")
        return

    compare(
        use_hybrid=not args.dense_only,
        k=args.k,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
