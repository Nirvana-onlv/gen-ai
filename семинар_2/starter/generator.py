import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from schema import Application, CITIES_LIST, SPECIALITIES_LIST, COURSES_LIST
from llm_client import make_client

TARGET = 50
MAX_ATTEMPTS = TARGET * 3

def build_city_queue() -> list[str]:
    quota = TARGET // len(CITIES_LIST)
    queue = [city for city in CITIES_LIST for _ in range(quota)]
    while len(queue) < TARGET:
        queue.append(random.choice(CITIES_LIST))
    random.shuffle(queue)
    return queue

def build_speciality_queue() -> list[str]:
    quota = TARGET // len(SPECIALITIES_LIST)
    queue = [spec for spec in SPECIALITIES_LIST for _ in range(quota)]
    while len(queue) < TARGET:
        queue.append(random.choice(SPECIALITIES_LIST))
    random.shuffle(queue)
    return queue


def generate_prompt(seed_city: str, seed_speciality: str) -> str:
    name_styles = [
        "редкое ФИО", "ФИО 90-х годов", "современное ФИО",
        "ФИО из национального региона РФ", "ФИО с иностранными корнями"
    ]
    name_hint = random.choice(name_styles)

    return f"""
Сгенерируй заявку на курс повышения квалификации (ДПО) для вымышленного специалиста.
Верни ТОЛЬКО валидный JSON, соответствующий схеме.

Параметры:
- full_name: реалистичное русское Фамилия Имя Отчество (избегай шаблонных ФИО!)
  Подсказка: {name_hint}
- age: от 22 до 65
- address: {{ "city": "{seed_city}", "district": "реалистичный район этого города" }}
- speciality: ИСПОЛЬЗУЙ ИМЕННО ЭТУ специальность: "{seed_speciality}"
  (она должна быть одной из [{", ".join(SPECIALITIES_LIST)}])
- desired_course: ОДНА из [{", ".join(COURSES_LIST)}] (подбери логичный курс для данной специальности)
- years_of_experience: от 0 до 40 (соответствует возрасту)
- graduation_year: от 1980 до 2024 (должен логично соответствовать возрасту!)

Никакого текста до/после JSON, никаких markdown-обёрток. Только чистый JSON.
"""

def main():
    client = make_client()
    applications = []

    city_queue = build_city_queue()
    speciality_queue = build_speciality_queue()

    attempts = 0
    idx = 0

    while len(applications) < TARGET:
        if attempts >= MAX_ATTEMPTS:
            print(
                f"Достигнут лимит попыток ({MAX_ATTEMPTS}). "
                f"Собрано {len(applications)} заявок из {TARGET}."
            )
            break

        seed_city = city_queue[idx % len(city_queue)]
        seed_speciality = speciality_queue[idx % len(speciality_queue)]
        idx += 1
        attempts += 1

        try:
            prompt = generate_prompt(seed_city, seed_speciality)

            app: Application = client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                response_model=Application,
                max_retries=3,
                temperature=0.8,
                top_p=0.9,
            )

            app_dict = app.model_dump()
            app_dict["city"] = app_dict["address"]["city"]
            app_dict["district"] = app_dict["address"]["district"]
            del app_dict["address"]

            applications.append(app_dict)
            print(
                f"[{len(applications)}/{TARGET}] "
                f"{app.full_name} | {app.address.city} | {app.speciality}"
            )

        except Exception as e:
            print(f"Попытка {attempts} провалилась: {type(e).__name__}: {e}")
            continue

    df = pd.DataFrame(applications)
    df.to_csv("applications.csv", index=False, encoding="utf-8")

    plt.figure(figsize=(10, 5))
    df['city'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Распределение заявок по городам")
    plt.xlabel("Город")
    plt.ylabel("Количество")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("cities.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    df['speciality'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title("Распределение заявок по специальностям")
    plt.xlabel("Специальность")
    plt.ylabel("Количество")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("specialities.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()