import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import google.generativeai as genai
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.neighbors import NearestNeighbors

from model_utils import (
    load_data as load_model_data,
    train_regression_model,
    train_high_gpa_classifier,
    predict_student,
)

# Paths
DATA_PATH = os.path.join(
    "data",
    "students-export-2025-12-11T09_00_10.417Z.csv",
)

GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "models/text-embedding-004")


def _with_model_prefix(name: str | None) -> str | None:
    if not name:
        return None
    return name if name.startswith("models/") else f"models/{name}"
DEFAULT_TOP_K = 5

st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
st.title("Student Analytics Dashboard")

# Data loading with cache (delegates to model utilities for normalization)
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = load_model_data(path)
    return df


def _get_gemini_api_key() -> str | None:
    """Resolve Gemini API key from Streamlit secrets or environment variables."""
    secrets = getattr(st, "secrets", None)
    if secrets and "GEMINI_API_KEY" in secrets:
        return secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY")


@st.cache_resource(show_spinner=False)
def _resolve_supported_model(api_key: str) -> str:
    genai.configure(api_key=api_key)
    preferred = _with_model_prefix(GEMINI_MODEL_NAME)
    fallback_order = [
        candidate
        for candidate in [
            preferred,
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-pro-latest",
            "models/gemini-1.0-pro",
        ]
        if candidate
    ]

    try:
        models = list(genai.list_models())
        supported = {
            _with_model_prefix(m.name)
            for m in models
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        }
        for candidate in fallback_order:
            if candidate in supported:
                return candidate
    except Exception:
        pass

    return fallback_order[0] if fallback_order else "gemini-1.0-pro"


@st.cache_resource(show_spinner=False)
def _init_gemini_model(api_key: str):
    model_name = _resolve_supported_model(api_key)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def get_gemini_model():
    api_key = _get_gemini_api_key()
    if not api_key:
        return None
    return _init_gemini_model(api_key)


def build_knowledge_chunks(df: pd.DataFrame, reg_result, cls_result) -> list[str]:
    """Create text snippets summarizing the dataset and models for retrieval."""
    chunks: list[str] = []
    chunks.append(
        f"Dataset snapshot: {len(df)} students spanning {df['faculty'].nunique()} faculties, "
        f"levels {sorted(df['level'].unique().tolist())}, GPA range {df['GPA'].min():.2f}-{df['GPA'].max():.2f}."
    )

    numeric_cols = ["GPA", "BMI", "study_hours", "WASSCE_Aggregate", "height", "weight"]
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        chunks.append(
            f"{col} stats -> mean {series.mean():.2f}, median {series.median():.2f}, min {series.min():.2f}, max {series.max():.2f}."
        )

    for faculty, group in df.groupby("faculty"):
        group = group.dropna(subset=["GPA"])
        if group.empty:
            continue
        chunks.append(
            f"Faculty {faculty}: avg GPA {group['GPA'].mean():.2f}, median WASSCE {group['WASSCE_Aggregate'].median():.1f}, "
            f"avg study hours {group['study_hours'].mean():.1f}, students {len(group)}."
        )

    if reg_result:
        chunks.append(
            f"Regression model: R^2 {reg_result.r2:.2f}, MSE {reg_result.mse:.3f}. Top coefficients influence GPA predictions."
        )
        top_reg = reg_result.coefficients.head(5)
        if not top_reg.empty:
            reg_features = ", ".join(
                f"{row.feature} ({row.coefficient:.3f})" for row in top_reg.itertuples(index=False)
            )
            chunks.append(f"Key regression features: {reg_features}.")

    if cls_result:
        chunks.append(
            f"High-GPA classifier accuracy {cls_result.accuracy:.2f}. Predicts probability of GPA ≥ 3.5."
        )
        top_cls = cls_result.coefficients.head(5)
        if not top_cls.empty:
            cls_features = ", ".join(
                f"{row.feature} (|coef|={row.abs_coefficient:.3f})" for row in top_cls.itertuples(index=False)
            )
            chunks.append(f"Influential classifier features: {cls_features}.")

    chunks.append(
        "Student scenario predictor accepts height, weight, level, faculty, department, year of birth, gender, study hours, and WASSCE aggregate to estimate GPA and high-GPA probability."
    )

    readme_path = Path("README.md")
    if readme_path.exists():
        chunks.append(readme_path.read_text(encoding="utf-8"))

    # Detailed per-student descriptions batched into chunks for retrieval
    def _fmt(row, col, decimals: int | None = None):
        val = row.get(col)
        if pd.isna(val):
            return "unknown"
        if isinstance(val, (int, float)) and decimals is not None:
            return f"{val:.{decimals}f}"
        return str(val)

    row_descriptions: list[str] = []
    for row in df.to_dict(orient="records"):
        desc = (
            f"Student level {_fmt(row, 'level')} from {row.get('faculty', 'unknown')} / {row.get('department', 'unknown')} "
            f"({row.get('gender', 'unknown')}), GPA {_fmt(row, 'GPA', 2)}, WASSCE {_fmt(row, 'WASSCE_Aggregate')} "
            f"study_hours {_fmt(row, 'study_hours', 1)}, BMI {_fmt(row, 'BMI', 1)}, height {_fmt(row, 'height')} cm, "
            f"weight {_fmt(row, 'weight')} kg, age {_fmt(row, 'age')} years."
        )
        row_descriptions.append(desc)

    chunk_size = 40
    for start in range(0, len(row_descriptions), chunk_size):
        chunk_text = "\n".join(row_descriptions[start:start + chunk_size])
        chunks.append(f"Student records {start + 1}-{min(start + chunk_size, len(row_descriptions))}:\n{chunk_text}")

    return [c for c in chunks if c]


def build_stat_lookup(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    if not df.empty:
        tallest = df.loc[df['height'].idxmax()]
        shortest = df.loc[df['height'].idxmin()]
        lookup['tallest'] = {
            'description': f"Tallest student: {tallest['height']} cm, {tallest['gender']} in {tallest['faculty']} / {tallest['department']}, GPA {tallest['GPA']:.2f}.",
            'height': str(tallest['height'])
        }
        lookup['shortest'] = {
            'description': f"Shortest student: {shortest['height']} cm, {shortest['gender']} in {shortest['faculty']} / {shortest['department']}, GPA {shortest['GPA']:.2f}.",
            'height': str(shortest['height'])
        }
    return lookup


def _extract_embedding_vector(response) -> np.ndarray | None:
    if response is None:
        return None
    embedding = getattr(response, "embedding", None)
    if embedding is None and isinstance(response, dict):
        embedding = response.get("embedding")
    if embedding is None and hasattr(response, "data"):
        data = getattr(response, "data")
        if data:
            first = data[0]
            if isinstance(first, dict):
                embedding = first.get("embedding")
            else:
                embedding = getattr(first, "embedding", None)
    if embedding is None:
        return None
    values = embedding.get("values") if isinstance(embedding, dict) else getattr(embedding, "values", None)
    if values is not None:
        embedding = values
    try:
        array = np.asarray(embedding, dtype=np.float32)
    except Exception:
        return None
    return array


def _embed_text(text: str, api_key: str) -> np.ndarray | None:
    if not text or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        response = genai.embed_content(model=GEMINI_EMBED_MODEL, content=text)
    except Exception:
        return None
    return _extract_embedding_vector(response)


@st.cache_resource(show_spinner=True)
def build_embedding_index(chunks: tuple[str, ...], api_key: str | None):
    if not chunks or not api_key:
        return None, tuple()
    embedded_chunks: list[str] = []
    embeddings: list[np.ndarray] = []
    for text in chunks:
        vector = _embed_text(text, api_key)
        if vector is None:
            continue
        embeddings.append(vector)
        embedded_chunks.append(text)
    if not embeddings:
        return None, tuple()
    matrix = np.vstack(embeddings)
    index = NearestNeighbors(metric="cosine", algorithm="brute")
    index.fit(matrix)
    return index, tuple(embedded_chunks)


def retrieve_contexts(
    question: str,
    index,
    chunk_texts: list[str] | tuple[str, ...],
    api_key: str | None,
    top_k: int = DEFAULT_TOP_K,
) -> list[str]:
    if not question or index is None or not chunk_texts or not api_key:
        return []
    query_vec = _embed_text(question, api_key)
    if query_vec is None:
        return []
    neighbors = min(top_k, len(chunk_texts))
    if neighbors <= 0:
        return []
    distances, indices = index.kneighbors(query_vec.reshape(1, -1), n_neighbors=neighbors)
    selected = []
    for idx in indices[0]:
        if 0 <= idx < len(chunk_texts):
            selected.append(chunk_texts[idx])
    return selected


def maybe_generate_chart(df: pd.DataFrame, question: str):
    """Render a quick visualization if the user explicitly requests a common chart."""
    if not question:
        return None, None
    q = question.lower()
    if "histogram" in q or "distribution" in q:
        fig = px.histogram(df, x="GPA", nbins=20, title="GPA distribution (chat request)")
        return fig, "GPA distribution requested via chat."
    if ("scatter" in q or "relationship" in q) and "wassce" in q and "gpa" in q:
        fig = px.scatter(
            df,
            x="WASSCE_Aggregate",
            y="GPA",
            color="faculty",
            title="WASSCE vs GPA (chat request)",
            hover_data=["department", "level"],
        )
        return fig, "Scatter plot of WASSCE aggregate against GPA."
    if ("scatter" in q or "relationship" in q) and "study" in q and "gpa" in q:
        fig = px.scatter(
            df,
            x="study_hours",
            y="GPA",
            color="gender",
            title="Study hours vs GPA (chat request)",
        )
        return fig, "Study hours vs GPA scatter plot." 
    if "box" in q and "faculty" in q:
        fig = px.box(df, x="faculty", y="GPA", color="faculty", title="GPA by faculty (chat request)")
        return fig, "Box plot of GPA grouped by faculty."
    return None, None


def generate_chat_response(
    question: str,
    index,
    indexed_chunks: list[str] | tuple[str, ...],
    fallback_chunks: list[str] | tuple[str, ...],
    api_key: str | None,
    stat_lookup: dict[str, dict[str, str]] | None = None,
):
    q_lower = question.lower() if question else ""
    if stat_lookup:
        if "tallest" in q_lower:
            data = stat_lookup.get('tallest')
            if data:
                return data['description'], [data['description']]
        if "shortest" in q_lower:
            data = stat_lookup.get('shortest')
            if data:
                return data['description'], [data['description']]
    contexts = retrieve_contexts(question, index, indexed_chunks, api_key)
    if (not contexts) and fallback_chunks:
        contexts = list(fallback_chunks)[:3]
    model = get_gemini_model()
    if not model:
        return ("Gemini API key not configured. Set GEMINI_API_KEY via secrets or environment.", contexts)

    context_text = "\n---\n".join(contexts) if contexts else "No additional context matched the query."
    prompt = dedent(
        f"""
        You are AcademicAI, a helpful assistant for the student analytics dashboard. Use the provided context to answer
        questions about the dataset, descriptive stats, predictive models, or study patterns. If a question cannot be
        answered from the context or does not relate to the data, say so plainly. Keep answers concise but specific.

        Context:
        {context_text}

        Current question: {question}

        If the user requests a visualization, describe what the generated chart shows after it renders. Mention when
        values are approximate. Reference faculties, departments, and metrics by name.
        """
    ).strip()

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text and getattr(response, "candidates", None):
            candidate = response.candidates[0]
            if candidate.content.parts:
                text = "".join(part.text for part in candidate.content.parts if hasattr(part, "text"))
        if not text:
            text = "I could not generate a response at this time."
        return text.strip(), contexts
    except Exception as exc:  # noqa: BLE001
        return (f"Gemini request failed: {exc}", contexts)

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

with st.sidebar:
    st.header("Experience")
    view_mode = st.radio("Choose view", options=["Dashboard", "Ask AcademicAI"], index=0)

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

knowledge_chunks = build_knowledge_chunks(df, reg_result, cls_result)
gemini_api_key = _get_gemini_api_key()
embedding_index, embedded_chunks = build_embedding_index(tuple(knowledge_chunks), gemini_api_key)
stat_lookup = build_stat_lookup(df)

if view_mode == "Dashboard":
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
    st.dataframe(filtered, width="stretch", height=350)

    csv_bytes = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_student_data.csv", mime="text/csv")

    # Visualizations
    st.subheader("Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered, x='GPA', nbins=20, title='GPA Distribution', marginal='box')
        st.plotly_chart(fig, width="stretch")
        gpa_mean = filtered['GPA'].mean()
        st.markdown(
            f"This shows how student GPAs are spread out. Bars further right mean more high-performing students. "
            f"In this view, GPAs cluster around about {gpa_mean:.2f}, with fewer students at the very low and very high ends."
        )

    with col2:
        fig = px.scatter(filtered, x='WASSCE_Aggregate', y='GPA', color='faculty', title='WASSCE vs GPA', hover_data=['department','level'])
        st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            "Dots to the right studied more each day. In the current data, higher GPAs tend to appear among students who study a bit more, "
            "but the relationship is gentle rather than a perfect straight line."
        )

    with col4:
        fig = px.box(filtered, x='faculty', y='GPA', color='faculty', title='GPA by Faculty')
        st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")
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
        faculty_options = sorted(df["faculty"].dropna().astype(str).unique().tolist())
        sel_faculty = st.selectbox("Faculty", faculty_options)
        deps_for_fac = sorted(df.loc[df["faculty"] == sel_faculty, "department"].dropna().astype(str).unique().tolist())
        sel_department = st.selectbox("Department", deps_for_fac)
        sel_level = st.selectbox("Level", sorted(df["level"].dropna().unique().tolist()))
        gender_options = sorted(df["gender"].dropna().astype(str).unique().tolist())
        sel_gender = st.selectbox("Gender", gender_options)

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
else:
    st.subheader("Ask AcademicAI (Gemini RAG)")
    st.markdown(
        "Chat with the student analytics assistant. Ask about trends, model insights, or request standard charts "
        "(histogram, scatter, box) and the system will render them when possible."
    )
    if not gemini_api_key:
        st.warning("Set the GEMINI_API_KEY in Streamlit secrets or environment to enable Gemini responses.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask about the students, faculties, or models")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with Gemini..."):
                response_text, used_contexts = generate_chat_response(
                    user_prompt,
                    embedding_index,
                    embedded_chunks if embedded_chunks else knowledge_chunks,
                    knowledge_chunks,
                    gemini_api_key,
                    stat_lookup,
                )
            st.markdown(response_text)

            fig, chart_caption = maybe_generate_chart(df, user_prompt)
            if fig:
                st.plotly_chart(fig, width="stretch")
                if chart_caption:
                    st.caption(chart_caption)

            if used_contexts:
                with st.expander("Context snippets", expanded=False):
                    for ctx in used_contexts:
                        st.write(ctx)

        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
