# Tech Trio Master Student Analytics Dashboard

A complete Flask + Pandas dashboard driven by `data/students.xlsx`.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Current dataset behavior
- College, department, batch and mentor values are detected dynamically.
- Java status text (`Ongoing`) and Python status text (`Yet to Start`) are treated as missing numeric marks, not as zero.
- Overall Average uses available numeric subject final marks. When Java/Python numeric marks are added to the Excel file, they are automatically included.
- Performance thresholds are configurable in `analysis/analytics.py`.

## Dynamic drill-downs
- Master → College
- College → Performance category → Students
- College → Batch → Students
- College → Mentor → Students
- Student → profile + rankings + subject chart
