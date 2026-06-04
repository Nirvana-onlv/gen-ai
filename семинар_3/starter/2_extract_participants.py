"""
Раунд 1 — Извлечение информации
=================================
Просим модель прочитать транскрипт и вернуть структурированный список
участников: имя, возраст, город, профессия, список жалоб с категорией
и точной цитатой.

Задача:
  1. Заполнить в schema.py две модели: Concern, Participant.
  2. Заполнить в prompts.py промпт IE_SYSTEM (явно проговорить:
     все поля, категории, требование точной цитаты, русский язык).
  3. Реализовать extract_participants() здесь.
  4. Запустить: python 2_extract_participants.py
     На выходе — участники и жалобы.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import get_model, make_client
from prompts import IE_SYSTEM  # дополни prompts.py перед запуском
from schema import Participant  # дополни schema.py перед запуском

client = make_client()
MODEL = get_model()


def extract_participants(transcript: str) -> list[Participant]:
    """Один запрос к модели → список участников с жалобами.

    Подсказки:
      • response_model=list[Participant] — обёртка сама обернёт список в {items:...}
      • max_retries=3 — на случай, если первый ответ невалиден по схеме
      • temperature=0.0 для извлечения фактов (а не для генерации)
    """
    return client.chat.completions.create(
        model=MODEL,
        response_model=list[Participant],
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": IE_SYSTEM},
            {"role": "user", "content": transcript},
        ],
    )
    # raise NotImplementedError


def main() -> None:
    transcript = Path("transcript.txt").read_text(encoding="utf-8")
    participants = extract_participants(transcript)

    print(f"Найдено участников: {len(participants)}")
    for p in participants:
        # Не уверен в полях? Открой schema.py и сверься.
        print(f"  • {p}")

    # Сохраним в JSON — пригодится в следующих раундах.
    out = [p.model_dump() for p in participants]
    Path("participants.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nСохранено: participants.json")


if __name__ == "__main__":
    main()
