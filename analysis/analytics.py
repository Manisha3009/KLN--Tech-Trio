import numpy as np
import pandas as pd
from .data_cleaning import SUBJECTS

PERFORMANCE_THRESHOLDS = {
    "Excellent": 90,
    "Good": 75,
    "Average": 60,
}
PASS_MARK = 60


def calculate_student_metrics(df):
    df = df.copy()
    subject_score_cols = []
    for key, meta in SUBJECTS.items():
        source = meta["final"]
        score_col = f"{key}_Average"
        df[score_col] = pd.to_numeric(df.get(source), errors="coerce")
        subject_score_cols.append(score_col)

    df["Overall_Average"] = df[subject_score_cols].mean(axis=1, skipna=True)
    df["Overall_Total"] = df[subject_score_cols].sum(axis=1, skipna=True)

    def classify(v):
        if pd.isna(v):
            return "No Data"
        if v >= PERFORMANCE_THRESHOLDS["Excellent"]:
            return "Excellent"
        if v >= PERFORMANCE_THRESHOLDS["Good"]:
            return "Good"
        if v >= PERFORMANCE_THRESHOLDS["Average"]:
            return "Average"
        return "Low"

    df["Performance_Level"] = df["Overall_Average"].map(classify)
    df["Pass_Fail"] = np.where(df["Overall_Average"].fillna(-1) >= PASS_MARK, "Pass", "Fail")
    df["Risk_Level"] = np.where(df["Overall_Average"].isna(), "No Data", np.where(df["Overall_Average"] < 60, "High", np.where(df["Overall_Average"] < 75, "Medium", "Low")))

    df["Overall_Rank"] = df["Overall_Average"].rank(method="min", ascending=False, na_option="bottom")
    df["College_Rank"] = df.groupby("College")["Overall_Average"].rank(method="min", ascending=False, na_option="bottom")
    df["Batch_Rank"] = df.groupby("Batch")["Overall_Average"].rank(method="min", ascending=False, na_option="bottom")
    return df


def _safe_mean(series):
    value = pd.to_numeric(series, errors="coerce").mean()
    return None if pd.isna(value) else round(float(value), 2)


def _safe_max(series):
    value = pd.to_numeric(series, errors="coerce").max()
    return None if pd.isna(value) else round(float(value), 2)


def _safe_min(series):
    value = pd.to_numeric(series, errors="coerce").min()
    return None if pd.isna(value) else round(float(value), 2)


def _pass_pct(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return round(float((s >= PASS_MARK).mean() * 100), 2) if len(s) else 0.0


def get_unique_mentors(df):
    vals = []
    for meta in SUBJECTS.values():
        col = meta["mentor"]
        if col in df:
            vals.extend([x for x in df[col].dropna().astype(str) if x and x.lower() != "unknown"])
    return sorted(set(vals), key=str.lower)


def get_master_kpis(df):
    return {
        "students": int(df["Student_ID"].nunique()),
        "colleges": int(df["College"].nunique()),
        "batches": int(df["Batch"].nunique()),
        "mentors": len(get_unique_mentors(df)),
        "departments": int(df["Department"].nunique()),
        "overall_average": _safe_mean(df["Overall_Average"]) or 0,
        "pass_percentage": _pass_pct(df["Overall_Average"]),
        "at_risk": int((df["Overall_Average"] < 60).sum()),
    }


def get_college_list(df):
    cards = []
    for college, g in df.groupby("College", dropna=False):
        cards.append({
            "college": college,
            "students": int(g["Student_ID"].nunique()),
            "batches": int(g["Batch"].nunique()),
            "average": _safe_mean(g["Overall_Average"]) or 0,
            "top": int((g["Performance_Level"] == "Excellent").sum()),
            "risk": int((g["Overall_Average"] < 60).sum()),
        })
    return sorted(cards, key=lambda x: (-x["students"], str(x["college"])))


def get_college_summary(df, college):
    g = df[df["College"] == college].copy()
    if g.empty:
        return None
    top_row = g.sort_values("Overall_Average", ascending=False, na_position="last").head(1)
    return {
        "college": college,
        "students": int(g["Student_ID"].nunique()),
        "batches": int(g["Batch"].nunique()),
        "departments": int(g["Department"].nunique()),
        "mentors": len(get_unique_mentors(g)),
        "cpp_avg": _safe_mean(g["CPP_Average"]),
        "java_avg": _safe_mean(g["Java_Average"]),
        "python_avg": _safe_mean(g["Python_Average"]),
        "overall_avg": _safe_mean(g["Overall_Average"]),
        "pass_pct": _pass_pct(g["Overall_Average"]),
        "at_risk": int((g["Overall_Average"] < 60).sum()),
        "top_performer": top_row.iloc[0]["Full_Name"] if len(top_row) else "—",
    }


def get_college_performance_distribution(df, college):
    g = df[df["College"] == college]
    order = ["Excellent", "Good", "Average", "Low"]
    counts = g["Performance_Level"].value_counts().to_dict()
    return {k: int(counts.get(k, 0)) for k in order}


def student_rows(g, limit=None, ascending=False):
    cols = ["Student_ID", "Full_Name", "Department", "Student_Year", "Batch", "CPP_Average", "Java_Average", "Python_Average", "Overall_Average", "Performance_Level", "Risk_Level"]
    out = g.sort_values("Overall_Average", ascending=ascending, na_position="last")
    if limit:
        out = out.head(limit)
    return out[cols].replace({np.nan: None}).to_dict("records")


def get_college_students(df, college):
    return df[df["College"] == college].copy()


def get_college_top_students(df, college, n=10):
    return student_rows(get_college_students(df, college), n, False)


def get_college_bottom_students(df, college, n=10):
    return student_rows(get_college_students(df, college), n, True)


def get_subject_analytics(df, college=None):
    g = df[df["College"] == college] if college else df
    result = {}
    for key, meta in SUBJECTS.items():
        col = f"{key}_Average"
        numeric = pd.to_numeric(g[col], errors="coerce")
        valid = g[numeric.notna()].copy()
        result[key] = {
            "label": meta["label"],
            "average": _safe_mean(numeric),
            "highest": _safe_max(numeric),
            "lowest": _safe_min(numeric),
            "pass_pct": _pass_pct(numeric),
            "top": student_rows(valid, 10, False),
            "bottom": student_rows(valid, 10, True),
            "has_data": bool(numeric.notna().any()),
        }
    return result


def get_college_batches(df, college):
    g = df[df["College"] == college]
    rows = []
    for batch, b in g.groupby("Batch"):
        rows.append({
            "batch": batch,
            "students": int(b["Student_ID"].nunique()),
            "average": _safe_mean(b["Overall_Average"]) or 0,
        })
    return sorted(rows, key=lambda x: str(x["batch"]))


def get_batch_summary(df, college, batch):
    g = df[(df["College"] == college) & (df["Batch"] == batch)].copy()
    if g.empty:
        return None, g
    dist = g["Performance_Level"].value_counts().to_dict()
    summary = {
        "students": int(g["Student_ID"].nunique()),
        "cpp_mentor": ", ".join(sorted(set(g["CPP_Mentor"]))) if "CPP_Mentor" in g else "—",
        "java_mentor": ", ".join(sorted(set(g["Java_Mentor"]))) if "Java_Mentor" in g else "—",
        "python_mentor": ", ".join(sorted(set(g["Python_Mentor"]))) if "Python_Mentor" in g else "—",
        "cpp_avg": _safe_mean(g["CPP_Average"]),
        "java_avg": _safe_mean(g["Java_Average"]),
        "python_avg": _safe_mean(g["Python_Average"]),
        "overall_avg": _safe_mean(g["Overall_Average"]),
        "excellent": int(dist.get("Excellent", 0)),
        "good": int(dist.get("Good", 0)),
        "average": int(dist.get("Average", 0)),
        "low": int(dist.get("Low", 0)),
    }
    return summary, g


def get_college_mentors(df, college):
    g = df[df["College"] == college]
    items = []
    for key, meta in SUBJECTS.items():
        mcol = meta["mentor"]
        scol = f"{key}_Average"
        for mentor, mg in g.groupby(mcol):
            if not mentor or str(mentor).lower() == "unknown":
                continue
            items.append({
                "mentor": mentor,
                "subject": meta["label"],
                "students": int(mg["Student_ID"].nunique()),
                "batches": int(mg["Batch"].nunique()),
                "average": _safe_mean(mg[scol]),
                "pass_pct": _pass_pct(mg[scol]),
                "at_risk": int((pd.to_numeric(mg[scol], errors="coerce") < 60).sum()),
            })
    return items


def get_department_analytics(df, college):
    g = df[df["College"] == college]
    rows = []
    for dept, d in g.groupby("Department"):
        rows.append({
            "department": dept,
            "students": int(d["Student_ID"].nunique()),
            "average": _safe_mean(d["Overall_Average"]) or 0,
            "pass_pct": _pass_pct(d["Overall_Average"]),
            "top": int((d["Performance_Level"] == "Excellent").sum()),
            "low": int((d["Performance_Level"] == "Low").sum()),
        })
    return sorted(rows, key=lambda x: (-x["students"], x["department"]))


def get_year_analytics(df, college):
    g = df[df["College"] == college]
    rows = []
    for year, y in g.groupby("Student_Year"):
        rows.append({
            "year": year,
            "students": int(y["Student_ID"].nunique()),
            "average": _safe_mean(y["Overall_Average"]) or 0,
            "pass_pct": _pass_pct(y["Overall_Average"]),
            "top": int((y["Performance_Level"] == "Excellent").sum()),
            "risk": int((y["Overall_Average"] < 60).sum()),
        })
    return rows


def get_college_comparison(df):
    rows = []
    for college, g in df.groupby("College"):
        dist = g["Performance_Level"].value_counts().to_dict()
        rows.append({
            "college": college,
            "students": int(g["Student_ID"].nunique()),
            "batches": int(g["Batch"].nunique()),
            "mentors": len(get_unique_mentors(g)),
            "cpp_avg": _safe_mean(g["CPP_Average"]),
            "java_avg": _safe_mean(g["Java_Average"]),
            "python_avg": _safe_mean(g["Python_Average"]),
            "overall_avg": _safe_mean(g["Overall_Average"]) or 0,
            "pass_pct": _pass_pct(g["Overall_Average"]),
            "excellent": int(dist.get("Excellent", 0)),
            "good": int(dist.get("Good", 0)),
            "average": int(dist.get("Average", 0)),
            "low": int(dist.get("Low", 0)),
        })
    return sorted(rows, key=lambda x: (-x["overall_avg"], x["college"]))


def get_student_profile(df, student_id):
    g = df[df["Student_ID"].astype(str) == str(student_id)]
    if g.empty:
        return None
    row = g.iloc[0].replace({np.nan: None}).to_dict()
    for key in ["Overall_Rank", "College_Rank", "Batch_Rank"]:
        if row.get(key) is not None:
            row[key] = int(row[key])
    return row


def get_mentor_students(df, mentor):
    mask = pd.Series(False, index=df.index)
    subjects = []
    for key, meta in SUBJECTS.items():
        m = df[meta["mentor"]].astype(str).str.casefold() == str(mentor).casefold()
        if m.any():
            subjects.append(meta["label"])
        mask |= m
    return df[mask].copy(), subjects
