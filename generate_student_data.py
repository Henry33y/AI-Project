import os
import math
import random
from datetime import date, datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd


FACULTIES = [
    "Faculty of Computing",
    "Faculty of Business",
    "Faculty of Theology",
    "Faculty of Engineering",
]

FACULTY_DEPARTMENTS = {
    "Faculty of Computing": ["Software Engineering", "Information Technology"],
    "Faculty of Business": ["Accounting", "Marketing"],
    "Faculty of Theology": ["Pastoral Studies", "Divinity"],
    "Faculty of Engineering": ["Electrical", "Mechanical"],
}

LEVELS = [100, 200, 300, 400]
GENDERS = ["Male", "Female"]


def random_date_between(year_start: int, year_end: int, rng: np.random.Generator) -> date:
    """Generate a random date between Jan 1 of year_start and Dec 31 of year_end (inclusive)."""
    start_dt = datetime(year_start, 1, 1)
    end_dt = datetime(year_end, 12, 31)
    delta_days = (end_dt - start_dt).days
    offset_days = int(rng.integers(0, delta_days + 1))
    return (start_dt + timedelta(days=offset_days)).date()


def compute_age(dob: date, today: date | None = None) -> int:
    """Compute age in full years given a date of birth."""
    if today is None:
        today = date.today()
    years = today.year - dob.year
    # Adjust if birthday hasn't occurred yet this year
    if (today.month, today.day) < (dob.month, dob.day):
        years -= 1
    return years


def generate_student_row(rng: np.random.Generator) -> dict:
    # Height (cm) and weight (kg)
    height_cm = int(rng.integers(150, 201))  # 150-200 inclusive
    weight_kg = int(rng.integers(50, 101))   # 50-100 inclusive

    # BMI = kg / m^2
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    bmi = round(float(bmi), 1)

    # Level, faculty, department
    level = int(rng.choice(LEVELS))
    faculty = str(rng.choice(FACULTIES))
    department = str(rng.choice(FACULTY_DEPARTMENTS[faculty]))

    # DOB and age
    dob = random_date_between(1998, 2007, rng)
    age = compute_age(dob)

    # Gender
    gender = str(rng.choice(GENDERS))

    # GPA: 1.5 - 4.0, rounded to 2 decimals
    gpa = round(float(rng.uniform(1.5, 4.0)), 2)

    # Study hours per day: 1.0 - 10.0, rounded to 1 decimal
    study_hours = round(float(rng.uniform(1.0, 10.0)), 1)

    # WASSCE Aggregate: 6 - 48 inclusive
    wassce_agg = int(rng.integers(6, 49))

    return {
        "height": height_cm,
        "weight": weight_kg,
        "BMI": bmi,
        "level": level,
        "faculty": faculty,
        "department": department,
        "dob": dob.isoformat(),
        "GPA": gpa,
        "gender": gender,
        "age": age,
        "study_hours": study_hours,
        "WASSCE_Aggregate": wassce_agg,
    }


def generate_dataset(n_rows: int = 500, seed: int | None = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [generate_student_row(rng) for _ in range(n_rows)]
    df = pd.DataFrame(rows)
    # Optional: reorder columns exactly as requested
    columns = [
        "height",
        "weight",
        "BMI",
        "level",
        "faculty",
        "department",
        "dob",
        "GPA",
        "gender",
        "age",
        "study_hours",
        "WASSCE_Aggregate",
    ]
    return df[columns]


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    df = generate_dataset(n_rows=500, seed=42)
    output = os.path.join("data", "student_data.csv")
    save_dataset(df, output)
    print(f"Generated {len(df)} rows -> {output}")


if __name__ == "__main__":
    main()
