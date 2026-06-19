"""
Eval мульти-агента: 3 вопроса, на которых одиночный агент С5 ломается.

Каждый вопрос прогоняется дважды:
  1) через одиночного агента С5 (agent_s5.run_agent)
  2) через PWC-цикл (orchestrator.run_pwc)

и сравниваются:
  - вызван ли calculate там, где нужно (для арифметических вопросов)
  - нет ли галлюцинаций инструментов
  - есть ли в ответе обязательная подстрока (must_have)

Прогон N=5 раз, считаем долю успешных прогонов. Результат пишется в eval_pwc_results.json.

Запуск:
    python eval_pwc.py           # полный прогон
    python eval_pwc.py --single  # только один прогон каждого, быстрая проверка
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc
from critic import CRITIC_PROMPT
from llm_client import get_model, make_client
from schemas_pwc import Plan, SubQuestion, WorkerAnswer, Verdict


CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": (
            "Класс ошибки C: одиночный часто считает в уме, не зовёт calculate. "
            "PWC должен починить — Планировщик обязан добавить calculate-подвопрос."
        ),
        "expected_tools_pwc": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["раз", "USD"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году?"
        ),
        "comment": (
            "Класс ошибки B: одиночный не умеет искать «последний доступный» "
            "месяц, зацикливается. PWC должен разбить на шаги."
        ),
        "expected_tools_pwc": {"get_inflation", "get_key_rate", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": (
            "Класс ошибки D (граница паттерна): требует get_inflation за много "
            "месяцев + большое calculate-выражение. Одиночный галлюцинирует "
            "get_cumulative_inflation; PWC обычно тоже (Планировщик может добавить "
            "выдуманный инструмент в план). Это — повод для Schema-Validator в домашке."
        ),
        "expected_tools_pwc": {"get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q4",
        "query": (
            "Какова реальная доходность рублёвого вклада, если номинальная ставка "
            "равна текущей ключевой ставке ЦБ, а инфляция берётся "
            "за последний доступный месяц?"
        ),
        "comment": (
            "Модель может предложить get_deposit_rate или "
            "get_real_yield — несуществующие инструменты. Валидатор должен перехватить и "
            "заставить переплановать на get_key_rate + get_inflation + calculate."
        ),
        "expected_tools_pwc": {"get_key_rate", "get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
        "arith_check": True,
    },
    {
        "id": "Q5",
        "query": (
            "Сравни курсы USD, EUR и CNY к рублю на 1 января 2023"
            "и на 1 января 2024 — как изменился каждый за год?"
        ),
        "comment": (
            "Параллельный кейс: шесть get_fx_rate независимы (три валюты × две даты) — "
            "execute_level запускает их одновременно на первом уровне. "
            "Плюс calculate для подсчёта изменений. Замеряем ускорение."
        ),
        "expected_tools_pwc": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["USD", "EUR", "CNY"],
        "forbid_hallucinated_tools": True,
        "arith_check": True,
    },
    {
        "id": "Q6",
        "query": (
            "Какой был реальный курс доллара с поправкой на накопленную инфляцию" 
            "с января 2023 по сегодня?"
        ),
        "comment": (
            "Модель может предложить get_real_fx_rate или get_inflation_adjusted_rate."
            "Валидатор должен перехватить и заставить использовать "
            "get_fx_rate + get_inflation + calculate."
        ),
        "expected_tools_pwc": {"get_fx_rate", "get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
        "arith_check": True,
    },
]


VALID_TOOL_NAMES = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}
ARITH_CASES = {c["id"] for c in CASES if c.get("arith_check", False)}


def _check_single(case: dict, result: dict) -> dict:
    """Проверить результат одиночного прогона."""
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    arith_without_calc = (
        case["id"] in ARITH_CASES
        and "calculate" not in used
        and bool(ans)
    )
    ok = bool(ans) and not hallucinated and must and not arith_without_calc
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "arith_without_calc": arith_without_calc,
        "answer_preview": (result.get("answer") or "")[:180],
    }


def _check_pwc(case: dict, result: dict) -> dict:
    """Проверить результат PWC-прогона."""
    used = set()
    validator_caught = []
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
        if t.get("kind") == "validator_caught":
            validator_caught.extend(t.get("hallucinated_tools", []))
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    # Также проверим галлюцинации на этапе Планировщика (в плане expected_tools)
    plan_tools = set()
    plan = result.get("plan")
    if plan is not None:
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
    plan_hallucinated = plan_tools - VALID_TOOL_NAMES

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    ok = (
        bool(result.get("answer"))
        and not hallucinated
        and not plan_hallucinated
        and must
    )
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": sorted(plan_hallucinated),
        "validator_caught": sorted(set(validator_caught)),
        "must_have_ok": must,
        "iterations": result.get("iterations", -1),
        "answer_preview": (result.get("answer") or "")[:180],
    }


def run_case(case: dict, *, n: int = 5) -> dict:
    single = {"runs": [], "pass": 0}
    pwc = {"runs": [], "pass": 0}
    pwc_val = {"runs": [], "pass": 0}

    for i in range(n):
        # --- Одиночный агент ---
        try:
            r1 = run_agent(case["query"], max_iter=8, verbose=False)
        except Exception as e:
            r1 = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": []}
        check1 = _check_single(case, r1)
        single["runs"].append(check1)
        single["pass"] += int(check1["ok"])

        # --- PWC ---
        try:
            r2 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=False)
        except Exception as e:
            r2 = {"answer": None, "error": f"{type(e).__name__}: {e}",
                  "trace": [], "plan": None}
        check2 = _check_pwc(case, r2)
        pwc["runs"].append(check2)
        pwc["pass"] += int(check2["ok"])

        try:
            r3 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=True)
        except Exception as e:
            r3 = {"answer": None, "error": f"{type(e).__name__}: {e}",
                  "trace": [], "plan": None}
        check3 = _check_pwc(case, r3)
        pwc_val["runs"].append(check3)
        pwc_val["pass"] += int(check3["ok"])

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        "single": single,
        "pwc": pwc,
        "pwc_validator": pwc_val,
    }

CRITIC_QUESTION = "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"

FAKE_BROKEN = [
    {
        "label": "арифметика без calculate",
        "plan": Plan(
            reasoning="Считаем отношение курсов",
            subquestions=[
                SubQuestion(id=1, question="Курс USD 2022-01-01?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс USD сегодня?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=3, question="Во сколько раз изменился?",
                            expected_tools=["calculate"], depends_on=[1, 2]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(subquestion_id=1, question_snippet="Курс USD 2022-01-01?",
                            answer="74.29 руб.", used_tools=["get_fx_rate"]),
            2: WorkerAnswer(subquestion_id=2, question_snippet="Курс USD сегодня?",
                            answer="71.91 руб.", used_tools=["get_fx_rate"]),
            3: WorkerAnswer(subquestion_id=3, question_snippet="Во сколько раз изменился?",
                            answer="USD подешевел в 1.033 раза (71.91/74.29).",
                            used_tools=[]),
        },
    },
    {
        "label": "выдуманное число (нет вызова инструмента)",
        "plan": Plan(
            reasoning="Текущая ключевая ставка",
            subquestions=[
                SubQuestion(id=1, question="Текущая ключевая ставка ЦБ?",
                            expected_tools=["get_key_rate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(subquestion_id=1, question_snippet="Текущая ключевая ставка ЦБ?",
                            answer="Ставка составляет 8.5% годовых.",
                            used_tools=[]),
        },
    },
    {
        "label": "несогласованные данные между подвопросами",
        "plan": Plan(
            reasoning="Реальная ставка = номинальная − инфляция",
            subquestions=[
                SubQuestion(id=1, question="Ключевая ставка?", expected_tools=["get_key_rate"]),
                SubQuestion(id=2, question="Инфляция апрель 2025?", expected_tools=["get_inflation"]),
                SubQuestion(id=3, question="Реальная ставка?",
                            expected_tools=["calculate"], depends_on=[1, 2]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(subquestion_id=1, question_snippet="Ключевая ставка?",
                            answer="16.0%", used_tools=["get_key_rate"]),
            2: WorkerAnswer(subquestion_id=2, question_snippet="Инфляция апрель 2025?",
                            answer="10.23%", used_tools=["get_inflation"]),
            3: WorkerAnswer(subquestion_id=3, question_snippet="Реальная ставка?",
                            answer="Реальная ставка = 21.0% - 8.5% = 12.5%",
                            used_tools=["calculate"]),
        },
    },
    {
        "label": "ответ с ошибкой (нет данных)",
        "plan": Plan(
            reasoning="Инфляция за конкретный месяц",
            subquestions=[
                SubQuestion(id=1, question="ИПЦ за март 2026?",
                            expected_tools=["get_inflation"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(subquestion_id=1, question_snippet="ИПЦ за март 2026?",
                            answer="(ошибка: нет данных ИПЦ на 2026-03)",
                            used_tools=[]),
        },
    },
    {
        "label": "план не покрывает вопрос",
        "plan": Plan(
            reasoning="Курс USD на сегодня",
            subquestions=[
                SubQuestion(id=1, question="Курс USD сегодня?",
                            expected_tools=["get_fx_rate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(subquestion_id=1, question_snippet="Курс USD сегодня?",
                            answer="71.91 руб.", used_tools=["get_fx_rate"]),
        },
    },
]

def _run_critic_once(plan: Plan, answers: dict, temperature: float) -> bool:
    plan_lines = [
        f"  {sq.id}. [{','.join(sq.expected_tools)}]  «{sq.question}»"
        for sq in plan.subquestions
    ]
    ans_lines = [
        f"  {sq_id}. [{','.join(a.used_tools) or '—'}] {a.answer}"
        for sq_id, a in sorted(answers.items())
    ]
    client = make_client()
    verdict: Verdict = client.chat.completions.create(
        model=get_model(),
        messages=[{
            "role": "system",
            "content": CRITIC_PROMPT.format(
                question=CRITIC_QUESTION,
                plan_text="\n".join(plan_lines),
                answers_text="\n".join(ans_lines),
            ),
        }],
        response_model=Verdict,
        temperature=temperature,
        max_retries=2,
    )
    return verdict.ok


def run_critic_eval(*, n: int = 10) -> list[dict]:
    rows = []
    for case in FAKE_BROKEN:
        print(f"  {case['label']}...")
        row = {"label": case["label"], "t0_false": 0, "t7_false": 0}
        for _ in range(n):
            if _run_critic_once(case["plan"], case["answers"], temperature=0.0):
                row["t0_false"] += 1
        for _ in range(n):
            if _run_critic_once(case["plan"], case["answers"], temperature=0.7):
                row["t7_false"] += 1
        rows.append(row)
        print(f"    T=0.0: {row['t0_false']}/{n}   T=0.7: {row['t7_false']}/{n}")
    return rows

def print_critic_table(rows: list[dict], n: int) -> None:
    print("\n" + "=" * 72)
    print(f"{'Битый кейс':<38} | {'T=0.0':^10} | {'T=0.7':^10}")
    print("-" * 72)
    for r in rows:
        print(f"  {r['label']:<36} | {r['t0_false']:>3}/{n:<6} | {r['t7_false']:>3}/{n:<6}")
    t0 = sum(r["t0_false"] for r in rows)
    t7 = sum(r["t7_false"] for r in rows)
    total = len(rows) * n
    print("-" * 72)
    print(f"  {'ИТОГО':<36} | {t0:>3}/{total:<6} | {t7:>3}/{total:<6}")
    print("\nВывод:")
    if t0 > t7:
        print("  Гипотеза подтверждается: T=0.0 даёт больше ложных принятий.")
        print("  Нулевая температура → Критик детерминированно повторяет логику")
        print("  Планировщика. Шум T=0.7 ломает это зеркало.")
    elif t0 < t7:
        print("  Гипотеза не подтверждается: T=0.7 дал больше ложных принятий.")
        print("  Возможно, шум заставляет Критика соглашаться случайно.")
    else:
        print("  Результаты равны — нет значимой разницы при данном N.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="Быстрая проверка: eval N=1, критик N=3")
    ap.add_argument("-n", type=int, default=5,
                    help="Прогонов на кейс для eval (default=5)")
    ap.add_argument("--critic-n", type=int, default=10,
                    help="Прогонов на кейс для замера критика (default=10)")
    ap.add_argument("--no-critic", action="store_true",
                    help="Пропустить замер угодливости критика")
    args = ap.parse_args()

    if args.single:
        n = 1
        critic_n = 3
    else:
        n = args.n
        critic_n = args.critic_n

    print(f"Eval С6: {len(CASES)} кейсов × {n} прогонов\n")
    results = []
    for case in CASES:
        print(f"=== {case['id']}: {case['query'][:70]}...")
        r = run_case(case, n=n)
        results.append(r)
        s = r["single"]
        p = r["pwc"]
        pv = r["pwc_validator"]
        print(f"   single: {s['pass']}/{n}    pwc: {p['pass']}/{n}    pwc+val: {pv['pass']}/{n}")
        for run in p["runs"][:1]:
            if run["hallucinated_in_plan"]:
                print(f"   ⚠ PWC:     план содержит выдуманные инструменты: {run['hallucinated_in_plan']}")
        caught_all = [t for run in pv["runs"] for t in run.get("validator_caught", [])]
        if caught_all:
            print(f"   ✅ PWC+val: валидатор поймал: {sorted(set(caught_all))}")
        for run in pv["runs"][:1]:
            if run["hallucinated_in_plan"]:
                print(f"   ⚠ PWC+val: после перепланировки всё ещё галлюцинации: {run['hallucinated_in_plan']}")
        print()

    # Итог
    print("=" * 70)
    print(f"{'ID':<4} {'single':>8} {'pwc':>8} {'pwc+val':>10}   query")
    print("-" * 70)
    for r in results:
        print(
            f"  {r['id']:<4} {r['single']['pass']}/{n}      "
            f"{r['pwc']['pass']}/{n}      "
            f"{r['pwc_validator']['pass']}/{n}   "
            f"{r['query'][:50]}"
        )

    out = Path(__file__).parent / "eval_pwc_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    print(f"\nРезультаты: {out}")

    if not args.no_critic:
        print(f"\n{'='*70}")
        print(f"ЧАСТЬ 2 — ЗАМЕР КРИТИКА: {len(FAKE_BROKEN)} кейсов × {critic_n} прогонов × 2 температуры")
        print(f"{'='*70}\n")
        rows = run_critic_eval(n=critic_n)
        print_critic_table(rows, critic_n)


if __name__ == "__main__":
    main()
