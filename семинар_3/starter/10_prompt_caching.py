"""
Раунд 6 — Кэширование промптов
================================
Каждый раунд 1, 2, 3 шлёт на сервер один и тот же transcript.txt.
Это входные токены, за которые мы платим каждый раз. Если транскрипт
большой (50k токенов) и мы делаем 10 операций — это 500k токенов на
повторении. Деньги в трубу.

Решение — кэширование промптов: провайдер запоминает префикс системного
промпта (или его части) и при следующем запросе с тем же префиксом
считает его как «попадание в кэш» — дешевле в 5-10 раз. У DeepSeek это
работает автоматически: достаточно держать transcript в начале промпта
без изменений между запросами.

Задача:
  1. Прогнать extract_aspects ДВАЖДЫ подряд на том же transcript.
     Замерить токены через response.usage (нужен with_completion=True).
  2. У DeepSeek в usage есть prompt_cache_hit_tokens и
     prompt_cache_miss_tokens. Посчитать процент попадания в кэш.
  3. Прогнать в третий раз, но с ИЗМЕНЕННЫМ системным промптом
     (добавить случайную строку). Доля попаданий упадёт.
  4. Прогнать в четвёртый раз, восстановив промпт. Доля попаданий вернётся.

  Вывод: кэш ОЧЕНЬ чувствителен к точному совпадению префикса.
  Любое изменение — кэш сбрасывается.

Запуск:
    python 10_prompt_caching.py
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from llm_client import get_model, make_client
from prompts import ASPECTS_SYSTEM
from schema import ParticipantSentiment

client = make_client()
MODEL = get_model()


def run_once(system_prompt: str, transcript: str) -> dict:
    t0 = time.time()
    _result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=list[ParticipantSentiment],
        max_retries=2,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        with_completion=True,
    )
    usage = completion.usage
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", usage.prompt_tokens - cache_hit) or 0
    return {
        "time": time.time() - t0,
        "prompt_tokens": usage.prompt_tokens,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
    }


def show(label: str, info: dict) -> None:
    total = info["prompt_tokens"]
    hit, miss = info["cache_hit"], info["cache_miss"]
    pct = hit / total * 100 if total else 0
    print(
        f"  {label:<32} t={info['time']:>5.1f}с  вход={total:>5}  "
        f"попаданий={hit:>5} ({pct:>3.0f}%)  промахов={miss:>5}"
    )


def main() -> None:
    transcript = Path("transcript.txt").read_text(encoding="utf-8")
    print(f"Модель: {MODEL}, транскрипт: {len(transcript)} символов\n")

    print("━━━ Прогон 1 (холодный) ━━━")
    show("первый раз, промпт A", run_once(ASPECTS_SYSTEM, transcript))

    print("\n━━━ Прогон 2 (тот же промпт) ━━━")
    show("повтор, промпт A", run_once(ASPECTS_SYSTEM, transcript))

    print("\n━━━ Прогон 3 (промпт чуть отличается) ━━━")
    modified = ASPECTS_SYSTEM + f"\n# случайный комментарий: {uuid.uuid4()}\n"
    show("модифицированный промпт", run_once(modified, transcript))

    print("\n━━━ Прогон 4 (возвращаем промпт A) ━━━")
    show("снова промпт A", run_once(ASPECTS_SYSTEM, transcript))


if __name__ == "__main__":
    main()
