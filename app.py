from pathlib import Path
from urllib.parse import quote
import pandas as pd
from flask import Flask, render_template, request, abort

from analysis.data_cleaning import load_data, clean_data, SUBJECTS
from analysis.analytics import (
    calculate_student_metrics, get_master_kpis, get_college_list,
    get_college_summary, get_college_performance_distribution,
    get_college_top_students, get_college_bottom_students,
    get_subject_analytics, get_college_batches, get_batch_summary,
    get_college_mentors, get_department_analytics, get_year_analytics,
    get_college_comparison, get_student_profile, get_mentor_students,
    student_rows, get_unique_mentors,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "students.xlsx"

app = Flask(__name__)


def get_data():
    return calculate_student_metrics(clean_data(load_data(DATA_PATH)))


def apply_filters(df, args):
    out = df.copy()
    mappings = {
        "college": "College",
        "department": "Department",
        "academic_year": "Academic_Year",
        "batch": "Batch",
        "performance": "Performance_Level",
    }
    for param, col in mappings.items():
        value = args.get(param, "").strip()
        if value:
            out = out[out[col].astype(str).str.casefold() == value.casefold()]

    mentor = args.get("mentor", "").strip()
    if mentor:
        mask = pd.Series(False, index=out.index)
        for meta in SUBJECTS.values():
            mask |= out[meta["mentor"]].astype(str).str.casefold() == mentor.casefold()
        out = out[mask]

    q = args.get("q", "").strip()
    if q:
        qf = q.casefold()
        search_cols = ["Full_Name", "Student_ID", "College", "Batch", "Department"]
        mask = pd.Series(False, index=out.index)
        for c in search_cols:
            mask |= out[c].astype(str).str.casefold().str.contains(qf, regex=False)
        for meta in SUBJECTS.values():
            mask |= out[meta["mentor"]].astype(str).str.casefold().str.contains(qf, regex=False)
        out = out[mask]
    return out


@app.template_filter("score")
def score_filter(v):
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):.1f}"


@app.context_processor
def inject_helpers():
    return {"quote": quote}


@app.route("/")
def dashboard():
    full = get_data()
    df = apply_filters(full, request.args)
    filters = {
        "colleges": sorted(full["College"].dropna().astype(str).unique()),
        "departments": sorted(full["Department"].dropna().astype(str).unique()),
        "years": sorted(full["Academic_Year"].dropna().astype(str).unique()),
        "batches": sorted(full["Batch"].dropna().astype(str).unique()),
        "mentors": get_unique_mentors(full),
        "performance": ["Excellent", "Good", "Average", "Low"],
    }
    comparison = get_college_comparison(df)
    perf = df["Performance_Level"].value_counts().to_dict()
    batch_counts = df.groupby("Batch")["Student_ID"].nunique().sort_values(ascending=False)
    dept_counts = df.groupby("Department")["Student_ID"].nunique().sort_values(ascending=False)
    master_subjects = get_subject_analytics(df)
    chart_data = {
        "colleges": [r["college"] for r in comparison],
        "students": [r["students"] for r in comparison],
        "averages": [r["overall_avg"] for r in comparison],
        "pass": [r["pass_pct"] for r in comparison],
        "risk": [r["low"] for r in comparison],
        "perf_labels": ["Excellent", "Good", "Average", "Low"],
        "perf_values": [int(perf.get(x, 0)) for x in ["Excellent", "Good", "Average", "Low"]],
        "batch_labels": batch_counts.index.astype(str).tolist(),
        "batch_values": batch_counts.astype(int).tolist(),
        "dept_labels": dept_counts.index.astype(str).tolist(),
        "dept_values": dept_counts.astype(int).tolist(),
        "subject_labels": [master_subjects[k]["label"] for k in ["CPP", "Java", "Python"]],
        "subject_values": [master_subjects[k]["average"] for k in ["CPP", "Java", "Python"]],
    }
    results = student_rows(df, 100, False) if request.args.get("q") else []
    return render_template("dashboard.html", kpis=get_master_kpis(df), colleges=get_college_list(df), comparison=comparison, filters=filters, chart_data=chart_data, results=results)


@app.route("/college/<path:college_name>")
def college(college_name):
    df = get_data()
    summary = get_college_summary(df, college_name)
    if not summary:
        abort(404)
    dist = get_college_performance_distribution(df, college_name)
    subjects = get_subject_analytics(df, college_name)
    batches = get_college_batches(df, college_name)
    mentors = get_college_mentors(df, college_name)
    departments = get_department_analytics(df, college_name)
    years = get_year_analytics(df, college_name)
    cg = df[df["College"] == college_name]
    chart_data = {
        "perf_labels": list(dist.keys()), "perf_values": list(dist.values()),
        "subject_labels": [subjects[k]["label"] for k in ["CPP", "Java", "Python"]],
        "subject_values": [subjects[k]["average"] for k in ["CPP", "Java", "Python"]],
        "batch_labels": [b["batch"] for b in batches], "batch_values": [b["average"] for b in batches],
        "dept_labels": [d["department"] for d in departments], "dept_values": [d["average"] for d in departments],
    }
    return render_template("college.html", summary=summary, dist=dist, subjects=subjects, batches=batches, mentors=mentors, departments=departments, years=years, top=get_college_top_students(df, college_name), bottom=get_college_bottom_students(df, college_name), average_students=student_rows(cg[cg["Performance_Level"] == "Average"], 25, False), risk_students=student_rows(cg[cg["Overall_Average"] < 60], 25, True), chart_data=chart_data)


@app.route("/college/<path:college_name>/performance/<level>")
def college_performance(college_name, level):
    df = get_data()
    valid = {"excellent": "Excellent", "good": "Good", "average": "Average", "low": "Low"}
    normalized = valid.get(level.casefold())
    if not normalized:
        abort(404)
    g = df[(df["College"] == college_name) & (df["Performance_Level"] == normalized)]
    return render_template("college_students.html", college=college_name, title=f"{normalized} Performers", students=student_rows(g, None, False))


@app.route("/college/<path:college_name>/batch/<path:batch_name>")
def batch(college_name, batch_name):
    df = get_data()
    summary, g = get_batch_summary(df, college_name, batch_name)
    if summary is None:
        abort(404)
    return render_template("batch.html", college=college_name, batch=batch_name, summary=summary, students=student_rows(g, None, False))


@app.route("/student/<student_id>")
def student(student_id):
    df = get_data()
    profile = get_student_profile(df, student_id)
    if not profile:
        abort(404)
    chart_data = {
        "labels": ["C++", "Java", "Python"],
        "values": [profile.get("CPP_Average"), profile.get("Java_Average"), profile.get("Python_Average")],
    }
    return render_template("student.html", s=profile, chart_data=chart_data)


@app.route("/mentor/<path:mentor_name>")
def mentor(mentor_name):
    df = get_data()
    g, subjects = get_mentor_students(df, mentor_name)
    if g.empty:
        abort(404)
    return render_template("mentor.html", mentor=mentor_name, subjects=subjects, students=student_rows(g, None, False), kpis=get_master_kpis(g))


@app.errorhandler(404)
def not_found(_):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
