import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model_utils import (
    load_data as load_model_data,
    train_regression_model,
    train_high_gpa_classifier,
    predict_student,
)

# Paths
DATA_PATH = os.path.join(
    "data",
    "students-export-2025-12-04T07_42_46.120Z.csv",
)

st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
st.title("Student Analytics Dashboard")

# Data loading with cache (delegates to model utilities for normalization)
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = load_model_data(path)
    return df

# Regenerate dataset utilities (fallback synthetic generator)
def regenerate_dataset(n_rows: int = 500, seed: int | None = 42) -> None:
    try:
        # Import from local script
        from generate_student_data import generate_dataset, save_dataset
    except Exception as e:
        st.error(f"Failed to import generator: {e}")
        return
    df_new = generate_dataset(n_rows=n_rows, seed=seed)
    save_dataset(df_new, DATA_PATH)

# Sidebar controls for administrative actions
with st.sidebar:
    st.header("Controls")
    with st.expander("Generate / Reload Data", expanded=True):
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        n_rows = st.number_input("Rows", min_value=100, max_value=10000, value=500, step=100)
        if st.button("Regenerate dataset", type="primary"):
            regenerate_dataset(n_rows=int(n_rows), seed=int(seed))
            st.cache_data.clear()
            st.success("Dataset regenerated.")
            st.rerun()

# Ensure data file exists before proceeding
if not os.path.exists(DATA_PATH):
    st.warning("Dataset not found. Generating a fresh dataset...")
    regenerate_dataset(n_rows=500, seed=42)

# Load data and train ML models
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

@st.cache_data(show_spinner=True)
def train_models(df: pd.DataFrame):
    reg_result = train_regression_model(df)
    cls_result = train_high_gpa_classifier(df)
    return reg_result, cls_result

reg_result, cls_result = train_models(df)

# Sidebar filters derived from data
with st.sidebar:
    st.header("Filters")
    faculties = sorted(df['faculty'].dropna().unique().tolist())
    sel_fac = st.multiselect("Faculty", faculties, default=faculties)

    # Departments conditioned on selected faculties
    deps = sorted(df.loc[df['faculty'].isin(sel_fac), 'department'].dropna().unique().tolist())
    sel_dep = st.multiselect("Department", deps, default=deps)

    levels = sorted(df['level'].dropna().unique().tolist())
    sel_lvl = st.multiselect("Level", levels, default=levels)

    genders = sorted(df['gender'].dropna().unique().tolist())
    sel_gender = st.multiselect("Gender", genders, default=genders)

    gpa_min, gpa_max = float(df['GPA'].min()), float(df['GPA'].max())
    sel_gpa = st.slider("GPA range", min_value=round(gpa_min, 2), max_value=round(gpa_max, 2), value=(round(gpa_min, 2), round(gpa_max, 2)), step=0.01)

    bmi_min, bmi_max = float(df['BMI'].min()), float(df['BMI'].max())
    sel_bmi = st.slider("BMI range", min_value=float(np.floor(bmi_min)), max_value=float(np.ceil(bmi_max)), value=(float(np.floor(bmi_min)), float(np.ceil(bmi_max))), step=0.1)

    age_min, age_max = int(df['age'].min()), int(df['age'].max())
    sel_age = st.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max), step=1)

    wass_min, wass_max = int(df['WASSCE_Aggregate'].min()), int(df['WASSCE_Aggregate'].max())
    sel_wass = st.slider("WASSCE Aggregate", min_value=wass_min, max_value=wass_max, value=(wass_min, wass_max), step=1)

    sh_min, sh_max = float(df['study_hours'].min()), float(df['study_hours'].max())
    sel_sh = st.slider("Study hours", min_value=float(np.floor(sh_min)), max_value=float(np.ceil(sh_max)), value=(float(np.floor(sh_min)), float(np.ceil(sh_max))), step=0.1)

# Apply multi-dimensional filters to the dataset
mask = (
    df['faculty'].isin(sel_fac) &
    df['department'].isin(sel_dep) &
    df['level'].isin(sel_lvl) &
    df['gender'].isin(sel_gender) &
    df['GPA'].between(sel_gpa[0], sel_gpa[1]) &
    df['BMI'].between(sel_bmi[0], sel_bmi[1]) &
    df['age'].between(sel_age[0], sel_age[1]) &
    df['WASSCE_Aggregate'].between(sel_wass[0], sel_wass[1]) &
    df['study_hours'].between(sel_sh[0], sel_sh[1])
)

filtered = df.loc[mask].copy()

# KPI summary tiles
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", f"{len(filtered):,}")
c2.metric("Avg GPA", f"{filtered['GPA'].mean():.2f}")
c3.metric("Median WASSCE", f"{filtered['WASSCE_Aggregate'].median():.0f}")
c4.metric("Avg Study Hours", f"{filtered['study_hours'].mean():.1f}")

c5, c6 = st.columns(2)
c5.metric("Regression R² (full data)", f"{reg_result.r2:.2f}")
c6.metric("High-GPA Accuracy (full data)", f"{cls_result.accuracy:.2f}")

# Data table and download
st.subheader("Filtered Data")
st.dataframe(filtered, use_container_width=True, height=350)

csv_bytes = filtered.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_student_data.csv", mime="text/csv")

# Visualizations
st.subheader("Visualizations")

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(filtered, x='GPA', nbins=20, title='GPA Distribution', marginal='box')
    st.plotly_chart(fig, use_container_width=True)
    gpa_mean = filtered['GPA'].mean()
    st.markdown(
        f"This shows how student GPAs are spread out. Bars further right mean more high-performing students. "
        f"In this view, GPAs cluster around about {gpa_mean:.2f}, with fewer students at the very low and very high ends."
    )

with col2:
    fig = px.scatter(filtered, x='WASSCE_Aggregate', y='GPA', color='faculty', title='WASSCE vs GPA', hover_data=['department','level'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "Each dot is a student. Higher dots have better GPA. In this data, students with stronger WASSCE scores often sit slightly higher on the chart, "
        "but there are also many exceptions (so WASSCE helps, but does not fully determine GPA)."
    )

col3, col4 = st.columns(2)
with col3:
    try:
        fig = px.scatter(filtered, x='study_hours', y='GPA', color='gender', title='Study Hours vs GPA', trendline='ols')
    except Exception:
        fig = px.scatter(filtered, x='study_hours', y='GPA', color='gender', title='Study Hours vs GPA')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "Dots to the right studied more each day. In the current data, higher GPAs tend to appear among students who study a bit more, "
        "but the relationship is gentle rather than a perfect straight line."
    )

with col4:
    fig = px.box(filtered, x='faculty', y='GPA', color='faculty', title='GPA by Faculty')
    st.plotly_chart(fig, use_container_width=True)
    faculty_gpa = filtered.groupby('faculty')['GPA'].mean().sort_values(ascending=False)
    top_faculty = faculty_gpa.index[0]
    st.markdown(
        f"Each box summarizes GPA for one faculty. Higher boxes or medians suggest stronger performance. "
        f"Right now, **{top_faculty}** has the highest average GPA in this filtered group."
    )

# Correlation heatmap
st.subheader("Correlation (numeric features)")
num_df = filtered.select_dtypes(include='number')
if not num_df.empty and num_df.shape[1] > 1:
    corr = num_df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', origin='lower', title='Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "Dark red squares mean two numbers tend to rise together; dark blue means when one goes up, the other often goes down. "
        "In this sample, GPA is only weakly linked to the other numeric features, which is expected because the data is synthetic and fairly random."
    )
else:
    st.info("Not enough numeric data to compute correlation.")

# Model insights from the trained pipelines
st.subheader("Model Insights")

top_reg = reg_result.coefficients.head(10)
if not top_reg.empty:
    fig = px.bar(top_reg, x="feature", y="coefficient", title="Top Regression Coefficients (GPA)")
    fig.update_layout(xaxis_title="Feature", yaxis_title="Coefficient", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    strongest_pos_reg = top_reg.iloc[0]
    strongest_neg_reg = reg_result.coefficients.sort_values("coefficient").iloc[0]
    st.markdown(
        "Positive bars mean that feature gently pushes predicted GPA up; negative bars pull it down, according to the linear model. "
        f"Here, **{strongest_pos_reg['feature']}** has the strongest positive effect, while **{strongest_neg_reg['feature']}** has one of the strongest negative effects in this synthetic data."
    )

top_cls = cls_result.coefficients.head(10)
if not top_cls.empty:
    fig = px.bar(top_cls, x="feature", y="abs_coefficient", title="Top Logistic Coefficients (High GPA)")
    fig.update_layout(xaxis_title="Feature", yaxis_title="|Coefficient|", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    most_influential = top_cls.iloc[0]
    st.markdown(
        "Taller bars are features the classifier pays more attention to when deciding who is likely to have a high GPA (3.5+). "
        f"Right now, **{most_influential['feature']}** is the single most influential feature for the high-GPA prediction in this model."
    )

st.caption("Regression predicts a student's GPA; logistic regression estimates the chance that GPA will be at least 3.5 (high performance). These patterns are based on synthetic data, so they are for learning and illustration, not real student decisions.")

# Student scenario predictor
st.subheader("Student Scenario Predictor")
st.markdown(
    "Provide a hypothetical student's details to estimate their GPA and the chance of achieving a high GPA (≥ 3.5)."
)

col_left, col_right = st.columns(2)

with col_left:
    sel_faculty = st.selectbox("Faculty", sorted(df["faculty"].unique().tolist()))
    deps_for_fac = sorted(df.loc[df["faculty"] == sel_faculty, "department"].unique().tolist())
    sel_department = st.selectbox("Department", deps_for_fac)
    sel_level = st.selectbox("Level", sorted(df["level"].unique().tolist()))
    sel_gender = st.selectbox("Gender", sorted(df["gender"].unique().tolist()))

    yob_min = int(df["yob"].min())
    yob_max = int(df["yob"].max())
    sel_yob = st.slider("Year of Birth", min_value=yob_min, max_value=yob_max, value=int(df["yob"].median()), step=1)

with col_right:
    wass_min, wass_max = int(df["WASSCE_Aggregate"].min()), int(df["WASSCE_Aggregate"].max())
    sel_wass = st.slider("WASSCE Aggregate", min_value=wass_min, max_value=wass_max, value=int(df["WASSCE_Aggregate"].median()), step=1)

    sh_min, sh_max = float(df["study_hours"].min()), float(df["study_hours"].max())
    sel_sh = st.slider("Study hours per day", min_value=float(np.floor(sh_min)), max_value=float(np.ceil(sh_max)), value=float(df["study_hours"].median()), step=0.1)

    height_min, height_max = int(df["height"].min()), int(df["height"].max())
    sel_height = st.slider("Height (cm)", min_value=height_min, max_value=height_max, value=int(df["height"].median()), step=1)

    weight_min, weight_max = int(df["weight"].min()), int(df["weight"].max())
    sel_weight = st.slider("Weight (kg)", min_value=weight_min, max_value=weight_max, value=int(df["weight"].median()), step=1)

height_m = sel_height / 100.0
sel_bmi = sel_weight / (height_m ** 2) if height_m > 0 else float(df["BMI"].median())

if st.button("Predict for this student", type="primary"):
    input_row = {
        "height": sel_height,
        "weight": sel_weight,
        "BMI": sel_bmi,
        "level": sel_level,
        "faculty": sel_faculty,
        "department": sel_department,
        "yob": sel_yob,
        "GPA": df["GPA"].mean(),  # placeholder, not used as input
        "gender": sel_gender,
        "study_hours": sel_sh,
        "WASSCE_Aggregate": sel_wass,
    }

    preds = predict_student(regression=reg_result, classifier=cls_result, student_data=input_row)
    pred_gpa = preds["predicted_gpa"]
    prob_high = preds["high_gpa_probability"]

    st.success(f"Predicted GPA: {pred_gpa:.2f} (on a 0–4 scale)")
    st.info(f"Estimated probability of High GPA (≥ 3.5): {prob_high * 100:.1f}%")

    st.markdown(
        "These predictions are based on the synthetic training data and simple linear/logistic models. "
        "Factors like WASSCE score, study hours, faculty, and BMI influence the estimates."
    )
