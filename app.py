import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Paths
DATA_PATH = os.path.join("data", "student_data.csv")

st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
st.title("Student Analytics Dashboard")

# Data loading with cache
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure types
    if 'dob' in df.columns:
        # keep as string for display, compute age safely if needed
        try:
            dob_dt = pd.to_datetime(df['dob'], errors='coerce')
            # Re-derive age to keep consistent with today's date
            age_calc = (pd.Timestamp(datetime.now().date()) - dob_dt).dt.days // 365
            df['age'] = age_calc.fillna(df.get('age'))
        except Exception:
            pass
    # Ensure expected columns exist
    required_cols = [
        'height','weight','BMI','level','faculty','department','dob',
        'GPA','gender','age','study_hours','WASSCE_Aggregate'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df

# Regenerate dataset utilities
def regenerate_dataset(n_rows: int = 500, seed: int | None = 42) -> None:
    try:
        # Import from local script
        from generate_student_data import generate_dataset, save_dataset
    except Exception as e:
        st.error(f"Failed to import generator: {e}")
        return
    df_new = generate_dataset(n_rows=n_rows, seed=seed)
    save_dataset(df_new, DATA_PATH)

# Sidebar controls
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

# Ensure data file exists
if not os.path.exists(DATA_PATH):
    st.warning("Dataset not found. Generating a fresh dataset...")
    regenerate_dataset(n_rows=500, seed=42)

# Load data
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

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

# Apply filters
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

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", f"{len(filtered):,}")
c2.metric("Avg GPA", f"{filtered['GPA'].mean():.2f}")
c3.metric("Median WASSCE", f"{filtered['WASSCE_Aggregate'].median():.0f}")
c4.metric("Avg Study Hours", f"{filtered['study_hours'].mean():.1f}")

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

with col2:
    fig = px.scatter(filtered, x='WASSCE_Aggregate', y='GPA', color='faculty', title='WASSCE vs GPA', hover_data=['department','level'])
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig = px.scatter(filtered, x='study_hours', y='GPA', color='gender', title='Study Hours vs GPA', trendline='ols')
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.box(filtered, x='faculty', y='GPA', color='faculty', title='GPA by Faculty')
    st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
st.subheader("Correlation (numeric features)")
num_df = filtered.select_dtypes(include='number')
if not num_df.empty and num_df.shape[1] > 1:
    corr = num_df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', origin='lower', title='Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough numeric data to compute correlation.")

# Automated insight generation (additive; does not modify existing visuals)
st.subheader("Automated Insights")

@st.cache_data(show_spinner=False)
def generate_insights(df_in: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    n = len(df_in)
    if n == 0:
        insights.append("Current filters return no rows; cannot compute insights.")
        return insights
    # GPA vs WASSCE Aggregate
    if {'GPA','WASSCE_Aggregate'}.issubset(df_in.columns):
        corr_wass = df_in['GPA'].corr(df_in['WASSCE_Aggregate'])
        if corr_wass < -0.1:
            insights.append(f"Lower WASSCE Aggregate (better exam score) tends to align with higher GPA (correlation {corr_wass:.2f}).")
        elif corr_wass > 0.1:
            insights.append(f"Higher WASSCE Aggregate is associated with higher GPA (correlation {corr_wass:.2f}); suggests aggregate may be inverted relative to expected scale.")
        else:
            insights.append(f"GPA shows only a weak linear relationship with WASSCE Aggregate (correlation {corr_wass:.2f}).")
    # Study hours relationship
    if {'GPA','study_hours'}.issubset(df_in.columns):
        corr_sh = df_in['GPA'].corr(df_in['study_hours'])
        if corr_sh > 0.15:
            insights.append(f"Students who report more study hours tend to have higher GPA (correlation {corr_sh:.2f}).")
        elif corr_sh < -0.15:
            insights.append(f"Unexpectedly, more study hours align with lower GPA (correlation {corr_sh:.2f}); could indicate reactive studying.")
        else:
            insights.append(f"Study hours show a weak linear association with GPA (correlation {corr_sh:.2f}).")
    # Gender differences
    if 'gender' in df_in.columns and 'GPA' in df_in.columns:
        gpa_by_gender = df_in.groupby('gender')['GPA'].mean().round(2)
        if len(gpa_by_gender) > 1:
            best_gender = gpa_by_gender.idxmax()
            spread = gpa_by_gender.max() - gpa_by_gender.min()
            insights.append(f"Average GPA varies by gender: {', '.join([f"{g}: {val:.2f}" for g, val in gpa_by_gender.items()])} (spread {spread:.2f}). {best_gender} currently leads.")
    # Simple OLS feature influence (predicting GPA)
    candidate_features = [c for c in ['WASSCE_Aggregate','study_hours','BMI','age'] if c in df_in.columns]
    if len(candidate_features) >= 2 and 'GPA' in df_in.columns:
        try:
            import statsmodels.api as sm  # ensure available
            X = df_in[candidate_features].copy()
            X = sm.add_constant(X)
            y = df_in['GPA']
            model = sm.OLS(y, X).fit()
            coefs = model.params.drop('const')
            ranked = coefs.abs().sort_values(ascending=False)
            top_features = ', '.join([f"{f} ({coefs[f]:.2f})" for f in ranked.index[:3]])
            insights.append(f"Top linear predictors of GPA (OLS absolute coefficient ranking): {top_features}.")
        except Exception as e:
            insights.append(f"Regression insight skipped due to error: {e}.")
    insights.append(f"Insights generated from {n} filtered students.")
    return insights

insight_lines = generate_insights(filtered)
st.markdown("\n".join([f"- {line}" for line in insight_lines]))
