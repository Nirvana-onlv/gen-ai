"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

На семинаре нужно:
- реализовать topological_sort (TODO 1),
- реализовать replan/rework-ветки цикла (TODO 2),
- написать synthesize для финального ответа (TODO 3).

Важно: max_iter защищает от бесконечного цикла, если Критик
постоянно говорит «переделай».
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client, make_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    errors = []
    for sq in plan.subquestions:
        fake = set(sq.expected_tools) - VALID_TOOLS
        if fake:
            errors.append(
                f"Подвопрос {sq.id} («{sq.question[:50]}») "
                f"использует несуществующие инструменты: {sorted(fake)}"
            )
    return errors

def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    """Отсортировать подвопросы так, чтобы depends_on шли раньше."""
    by_id = {s.id: s for s in subqs}
    in_degree: dict[int, int] = {s.id: 0 for s in subqs}
    for s in subqs:
        for dep in s.depends_on:
            if dep in by_id:
                in_degree[s.id] += 1

    levels: list[list[SubQuestion]] = []
    remaining = set(by_id.keys())

    while remaining:
        level_ids = {sid for sid in remaining if in_degree[sid] == 0}
        if not level_ids:
            raise ValueError(
                f"Цикл в depends_on, зависшие подвопросы: {sorted(remaining)}"
            )
        level = [by_id[sid] for sid in sorted(level_ids)]
        levels.append(level)
        remaining -= level_ids
        for s in subqs:
            if s.id in remaining:
                for dep in s.depends_on:
                    if dep in level_ids:
                        in_degree[s.id] -= 1

    return levels

def execute_level(
    level: list[SubQuestion],
    prev_answers: dict[int, WorkerAnswer],
    *,
    max_workers: int = 4,
) -> dict[int, WorkerAnswer]:
    results: dict[int, WorkerAnswer] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(worker, sq, prev_answers): sq.id
            for sq in level
        }
        for future in as_completed(futures):
            sq_id = futures[future]
            results[sq_id] = future.result()
    return results


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    parts_text = "\n".join(
        f"  {sq_id}. {answers[sq_id].answer}"
        for sq_id in sorted(answers)
    )
    client = make_client()
    resp = client._c.chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — финальный синтезатор. Тебе дают исходный вопрос и "
                    "список промежуточных ответов на подвопросы. Собери их в "
                    "1-2 фразы для пользователя: конкретное число с единицей, "
                    "без лишних деталей. Не используй инструменты."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Исходный вопрос: «{question}»\n\n"
                    f"Промежуточные ответы:\n{parts_text}"
                ),
            },
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or " · ".join(
        answers[i].answer for i in sorted(answers)
    )


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    use_validator: bool = True,
    hint_fake_tools: bool = False,   # <- добавить
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []

    plan = planner(question, hint_fake_tools=hint_fake_tools)
    if use_validator:
        errors = validate_plan(plan)
        if errors:
            feedback = "Инструменты не существуют: " + "; ".join(errors)
            if verbose:
                print(f"[schema-validator] ошибки в плане: {errors}")
                print(f"[schema-validator] перепланировка с feedback...")

            trace.append({
                "iter": 0,
                "kind": "validator_caught",
                "errors": errors,
                "hallucinated_tools": sorted(
                    {t for sq in plan.subquestions for t in sq.expected_tools} - VALID_TOOLS
                ),
            })
            plan = planner(question, feedback=feedback)
            errors2 = validate_plan(plan)
            if errors2 and verbose:
                print(f"[schema-validator] после перепланировки ещё ошибки: {errors2}")

    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    for iter_num in range(1, max_iter + 1):
        answers: dict[int, WorkerAnswer] = {}
        try:
            levels = _topological_sort(plan.subquestions)
        except ValueError as e:
            return {
                "answer": None,
                "error": f"ошибка в плане: {e}",
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        t_start = time.perf_counter()
        for level in levels:
            level_answers = execute_level(level, answers)
            answers.update(level_answers)
            for sq_id, ans in level_answers.items():
                trace.append({
                    "iter": iter_num,
                    "kind": "worker",
                    "sq_id": sq_id,
                    "used_tools": ans.used_tools,
                    "answer": ans.answer,
                })
                if verbose:
                    print(f"  [{sq_id}] → {ans.answer}   tools={ans.used_tools}")
        elapsed = time.perf_counter() - t_start

        if verbose:
            print(f"  [timing] исполнение {len(plan.subquestions)} подвопросов за {elapsed:.2f}с")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        elif verdict.action == "replan":
            if verbose:
                print(f"  [replan] перестраиваю план: {verdict.reason}")
            plan = planner(question, feedback=verdict.reason)
            if use_validator:
                errors = validate_plan(plan)
                if errors:
                    plan = planner(
                        question,
                        feedback=verdict.reason + " Также: инструменты не существуют: " + "; ".join(errors)
                    )
            trace.append({
                "iter": iter_num,
                "kind": "replan",
                "reasoning": plan.reasoning,
                "subquestions": [sq.model_dump() for sq in plan.subquestions],
            })

        elif verdict.action == "rework":
            ids_str = ", ".join(str(i) for i in verdict.rework_ids)
            feedback = (
                f"Переделай подвопросы {ids_str}. Причина: {verdict.reason}"
            )
            if verbose:
                print(f"  [rework] переплановываю подвопросы {ids_str}")
            plan = planner(question, feedback=feedback)
            if use_validator:
                errors = validate_plan(plan)
                if errors:
                    plan = planner(question, feedback=feedback + " Также: " + "; ".join(errors))
                    trace.append({
                        "iter": iter_num,
                        "kind": "rework",
                        "rework_ids": verdict.rework_ids,
                        "reasoning": plan.reasoning,
                        "subquestions": [sq.model_dump() for sq in plan.subquestions],
                    })

        else:
            break

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(q, max_iter=args.max_iter, verbose=not args.quiet)

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
