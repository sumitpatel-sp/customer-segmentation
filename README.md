# Consumer Segmentation Project

This project implements an end‑to‑end consumer segmentation pipeline using K‑Means clustering, exposed via a FastAPI backend and a Streamlit dashboard frontend.

### Features

- **Data pipeline**: loads the `Mall_Customers` dataset and applies consistent preprocessing (encoding + scaling).
- **Model training**: trains a K‑Means clustering model and persists both the model and scaler under `models/`.
- **REST API (FastAPI)**: exposes endpoints to predict a customer segment and to retrain the model.
- **Dashboard (Streamlit)**: interactive UI to:
  - Inspect automatically generated segment descriptions.
  - Predict the segment for new customers (inputs in INR, converted internally).

---

### Project Structure

- **`src/`**
  - `data_loader.py` – load raw CSV data.
  - `preprocessing.py` – feature engineering and scaling.
  - `train_ml.py` – train and persist the K‑Means model and scaler.
  - `predict.py` – shared prediction utilities (uses the trained scaler).
- **`app/`**
  - `main.py` – FastAPI application with `/predict` and `/retrain` endpoints.
- **`frontend/`**
  - `app.py` – Streamlit dashboard for analysis and predictions.
- **`models/`** (ignored by git) – trained model and scaler artefacts.

---

### Setup

1. **Create and activate a virtual environment (recommended)**  
   On Windows (PowerShell):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data is available**  
   Place `Mall_Customers.csv` under `data/Mall_Customers.csv` (as expected by the training script).

---

### Train the Model

Run the training script once to create `models/kmeans.pkl` and `models/scaler.pkl`:

```bash
python -m src.train_ml
```

You should see a confirmation message when the artefacts are saved.

---

### Run the FastAPI Backend

From the project root:

```bash
uvicorn app.main:app --reload
```

This will start the API at `http://127.0.0.1:8000`.

Key endpoints:
- `POST /predict` – predict a customer segment from input features.
- `POST /retrain` – retrain the K‑Means model using the latest data.

---

### Run the Streamlit Dashboard

With the backend running in a separate terminal, start the frontend:

```bash
streamlit run frontend/app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) to access the dashboard.

---

### Notes

- The backend and frontend both rely on the same trained model and scaler saved under `models/`.
- For reproducibility, large artefacts and experiment directories (such as `models/` and `mlruns/`) are ignored via `.gitignore`.