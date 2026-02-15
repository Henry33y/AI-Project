# AcademicAI Dashboard

A data analytics and RAG (Retrieval-Augmented Generation) chatbot for university student data. This application connects directly to the **Supabase** database used by the **Pentecost Student Hub** to provide real-time insights and AI-powered answers.

## Features

-   **Live Data**: Fetches student records directly from Supabase (no manual CSV uploads).
-   **Interactive Dashboard**: Filter students by faculty, department, GPA, and more.
-   **AI Chatbot (RAG)**: Ask questions about the data using Google's Gemini models (e.g., "Which faculty has the highest average GPA?").
-   **Predictive Models**: Built-in regression (GPA prediction) and classification (High GPA probability).
-   **Visualizations**: Dynamic charts for GPA distribution, relationships, and correlations.

## Prerequisites

-   Python 3.10+
-   Supabase project credentials
-   Google Gemini API Key

## Installation

1.  **Clone the repository** (if you haven't already).
2.  **Create and activate a virtual environment**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

## Configuration

Create a file named `.streamlit/secrets.toml` in the project root and add your credentials:

```toml
SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_anon_key"
ADMIN_EMAIL = "admin_email_for_bypass_policy"
ADMIN_PASSWORD = "admin_password"
GEMINI_API_KEY = "your_gemini_api_key_here"
```

> **Note**: The application uses the `transport="rest"` option for Gemini to verify connections in environments that might block standard gRPC ports.

## Running the App

Launch the Streamlit dashboard:

```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

-   **Dashboard View**: Use the sidebar filters to explore specific student segments.
-   **Ask AcademicAI View**: Switch views in the sidebar to chat with the AI about the dataset.
-   **Refresh Data**: Click the "Refresh Data" button in the sidebar to pull the latest records from the database.

## Project Structure

-   `app.py`: Main Streamlit application.
-   `db_utils.py`: Supabase connection and data fetching logic.
-   `model_utils.py`: Data processing, feature engineering, and machine learning models.
-   `test_db.py`: Script to verify database connectivity.
-   `test_gemini.py`: Script to verify Gemini API connectivity.
