import re
from pathlib import Path
import numpy as np
import pandas as pd

DISPLAY_COLUMNS = {
    "College": "Unknown College",
    "Department": "Unknown Department",
    "Batch": "Unknown Batch",
    "Student_Year": "Unknown Year",
}

SUBJECTS = {
    "CPP": {
        "label": "C++",
        "final": "CPP_Final_Marks",
        "mentor": "CPP_Mentor",
    },
    "Java": {
        "label": "Java",
        "final": "Java_Final_Marks",
        "mentor": "Java_Mentor",
    },
    "Python": {
        "label": "Python",
        "final": "Python_Final_Marks",
        "mentor": "Python_Mentor",
    },
}


def _clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _canonical(value, fallback="Unknown"):
    text = _clean_text(value)
    if not text:
        return fallback
    # Preserve short acronyms such as KLN, REC, SKP, AI & DS.
    compact = re.sub(r"[^A-Za-z0-9]", "", text)
    if text.isupper() and len(compact) <= 8:
        return text
    if text.isupper() or text.islower():
        return text.title()
    return text


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name="Student_Master")
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in ["Full_Name", "Email", "Phone_Number", "Course", "Month"]:
        if col in df.columns:
            df[col] = df[col].map(_clean_text)

    for col, fallback in DISPLAY_COLUMNS.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: _canonical(x, fallback))
        else:
            df[col] = fallback

    if "Academic_Year" in df.columns:
        df["Academic_Year"] = df["Academic_Year"].map(_clean_text)
    else:
        df["Academic_Year"] = ""

    # Standardize mentor display names while preserving mixed-case proper names.
    for meta in SUBJECTS.values():
        mentor_col = meta["mentor"]
        if mentor_col not in df.columns:
            df[mentor_col] = "Unknown"
        df[mentor_col] = df[mentor_col].map(lambda x: _canonical(x, "Unknown"))

    # Numeric marks. Status text (e.g. Ongoing / Yet to Start) safely becomes NaN.
    mark_cols = [c for c in df.columns if c.endswith(("_Quiz", "_Assignment", "_Assessment", "_Internal", "_Project", "_Theory", "_Final_Marks"))]
    for col in mark_cols:
        df[f"{col}__raw"] = df[col]
        df[col] = _numeric(df[col])

    # Student_ID is the preferred unique key; fall back to name+email.
    if "Student_ID" in df.columns:
        df["Student_ID"] = df["Student_ID"].map(_clean_text)
        df = df.drop_duplicates(subset=["Student_ID"], keep="first")
    else:
        df = df.drop_duplicates(subset=["Full_Name", "Email"], keep="first")

    return df.reset_index(drop=True)
