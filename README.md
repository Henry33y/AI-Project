# AcademicAI Dataset Generator

This repository includes a simple Python script to generate 500 rows of dummy student data for a university analytics project.

## What it creates
- Output file: `data/student_data.csv`
- Rows: 500
- Columns: `height, weight, BMI, level, faculty, department, dob, GPA, gender, age, study_hours, WASSCE_Aggregate`

## Requirements
- Python 3.10+
- Packages: `pandas`, `numpy`

If you're using a virtual environment, make sure it's activated and install requirements:

```powershell
pip install -r requirements.txt
```

## Generate the dataset
Run the generator script:

```powershell
# If using the active Python
python .\generate_student_data.py

# Or explicitly via venv (Windows PowerShell)
.\venv\Scripts\python.exe .\generate_student_data.py
```

On success, you should see a message like:

```
Generated 500 rows -> data\student_data.csv
```

## Streamlit app
Launch the interactive dashboard (filters, charts, download, regenerate data):

```powershell
streamlit run .\app.py
```

Then open the URL shown in the terminal, e.g. `http://localhost:8501`.

## Notes
- Height is in centimeters (150–200).
- Weight is in kilograms (50–100).
- BMI is computed as kg/m^2 and rounded to 1 decimal place.
- Level is one of 100, 200, 300, 400.
- Faculty is chosen from Computing, Business, Theology, Engineering; department matches the selected faculty.
- DOB is between 1998 and 2007 inclusive, and `age` is derived from DOB at runtime.
- GPA is a float in [1.5, 4.0], rounded to 2 decimals.
- Study hours is a float in [1.0, 10.0], rounded to 1 decimal.
- WASSCE_Aggregate is an integer in [6, 48].
