"""
Pydantic-схема для персоны.
==========================
Сейчас здесь только комментарии.
(раскомментируй и допиши на семинаре):
"""

from typing import Literal
from pydantic import BaseModel, Field, model_validator
from datetime import datetime

CURRENT_YEAR = datetime.now().year

CITIES_LIST = [
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань",
    "Нижний Новгород", "Самара", "Омск", "Челябинск", "Ростов-на-Дону"
]

SPECIALITIES_LIST = [
    "Программист", "Дизайнер", "Маркетолог", "Продакт-менеджер",
    "Инженер-конструктор", "Экономист", "Юрист",
    "Риск-менеджер", "Тестировщик"
]

COURSES_LIST = [
    "Основы финансового анализа", "Искусство презентации",
    "Управление проектами", "Time management", "Прикладное применение нейросетей",
    "Искусство коммуникации", "Руководство командой"
]

CITIES = Literal[tuple(CITIES_LIST)]
SPECIALITIES = Literal[tuple(SPECIALITIES_LIST)]
DESIRED_COURSES = Literal[tuple(COURSES_LIST)]

class Address(BaseModel):
    city: CITIES
    district: str


class Application(BaseModel):
    full_name: str
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: SPECIALITIES
    desired_course: DESIRED_COURSES
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @model_validator(mode="after")
    def check_age_graduation_consistency(self):
        grad_age = self.age - (CURRENT_YEAR - self.graduation_year)

        if not (18 <= grad_age <= 30):
            raise ValueError(
                f"Несоответствие данных: возраст {self.age} "
                f"и год выпуска {self.graduation_year}. "
                f"Возраст при выпуске получился {grad_age} лет."
            )

        return self