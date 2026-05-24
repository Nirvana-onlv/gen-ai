import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from schema import Application, CITIES_LIST, SPECIALITIES_LIST, COURSES_LIST
from llm_client import make_client

def generate_prompt(seed_city: str) -> str:

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
- speciality: ОДНА из [{", ".join(SPECIALITIES_LIST)}] (выбирай случайную профессию)
- desired_course: ОДНА из [{", ".join(COURSES_LIST)}] (Используй логичные комбинации профессии и курса)
- years_of_experience: от 0 до 40
- graduation_year: от 1980 до 2024 (должен логично соответствовать возрасту!)

Никакого текста до/после JSON, никаких markdown-обёрток. Только чистый JSON.
"""

def main():
    client = make_client()
    applications = []
    target = 50

    quota_per_city = target // len(CITIES_LIST)
    city_queue = [city for city in CITIES_LIST for _ in range(quota_per_city)]
    random.shuffle(city_queue)

    for seed_city in city_queue:
        prompt = generate_prompt(seed_city)

        app = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            response_model=Application,
            max_retries=3,
            temperature=0.8,
            top_p=0.9
        )

        app_dict = app.model_dump()
        app_dict['city'] = app_dict['address']['city']
        app_dict['district'] = app_dict['address']['district']
        del app_dict['address']

        applications.append(app_dict)
        print(f"Заявка {len(applications)}/{target}: {app.full_name} ({app.address.city})")

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