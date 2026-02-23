import streamlit as st
import requests
import pandas as pd
import joblib


API_URL = "http://127.0.0.1:8000"
USD_TO_INR = 83  # Fixed (demo) conversion rate


@st.cache_data
def load_data():
    df = pd.read_csv("data/Mall_Customers.csv")
    return df


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("models/kmeans.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


def build_segment_labels(df_with_clusters):
    cluster_summary = df_with_clusters.groupby("Cluster")[[
        "Annual Income (k$)",
        "Spending Score (1-100)",
    ]].mean()

    segment_meanings = {}
    for cluster_id, row in cluster_summary.iterrows():
        income = row["Annual Income (k$)"]
        spending = row["Spending Score (1-100)"]

        if income < 50 and spending < 50:
            label = "Low Income - Low Spending"
        elif income >= 50 and spending >= 50:
            label = "High Income - High Spending"
        elif income >= 50 and spending < 50:
            label = "High Income - Low Spending"
        else:
            label = "Low Income - High Spending"

        segment_meanings[cluster_id] = label

    return cluster_summary, segment_meanings


def main():
    st.set_page_config(page_title="Consumer Segmentation", layout="wide")

    st.title("🚀 Consumer Segmentation Dashboard")
    st.markdown(
        "Understand your customers through **data‑driven segmentation** using K‑Means clustering. "
        "Interactively explore segments and score new customers."
    )

    # -----------------------------
    # Load core artefacts
    # -----------------------------
    df_raw = load_data()
    model, scaler = load_model_and_scaler()

    # Preprocess same as training
    df_proc = df_raw.copy()
    df_proc = df_proc.drop("CustomerID", axis=1)
    df_proc["Gender"] = df_proc["Gender"].map({"Male": 0, "Female": 1})
    features = df_proc[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

    scaled = scaler.transform(features)
    df_with_clusters = df_raw.copy()
    df_with_clusters["Cluster"] = model.predict(scaled)

    cluster_summary, segment_meanings = build_segment_labels(df_with_clusters)

    # -----------------------------
    # Layout: tabs
    # -----------------------------
    tab_overview, tab_segments, tab_predict = st.tabs(
        ["📊 Overview", "📘 Segment Insights", "🔍 Predict Customer"]
    )

    # ====== OVERVIEW TAB ======
    with tab_overview:
        st.subheader("Dataset Overview")

        col_left, col_right = st.columns((2, 1))

        with col_left:
            st.markdown("**Sample of customers**")
            st.dataframe(df_with_clusters.head(15), use_container_width=True)

        with col_right:
            st.markdown("**Key metrics**")
            total_customers = len(df_with_clusters)
            n_segments = df_with_clusters["Cluster"].nunique()

            col_a, col_b = st.columns(2)
            col_a.metric("Total Customers", f"{total_customers}")
            col_b.metric("Number of Segments", f"{n_segments}")

            st.markdown("**Segment Size Distribution**")
            size_df = (
                df_with_clusters["Cluster"]
                .value_counts()
                .sort_index()
                .rename_axis("Cluster")
                .reset_index(name="Count")
            )
            st.bar_chart(size_df.set_index("Cluster"))

    # ====== SEGMENT INSIGHTS TAB ======
    with tab_segments:
        st.subheader("Segment Interpretation")
        st.markdown(
            "Each segment is automatically described based on **average income** and "
            "**spending score**."
        )

        # Show high‑level summary table
        st.markdown("**Average Income & Spending by Segment**")
        st.dataframe(cluster_summary.style.format({"Annual Income (k$)": "{:.1f}", "Spending Score (1-100)": "{:.1f}"}))

        st.markdown("---")
        st.markdown("**Business‑friendly segment descriptions**")

        cols = st.columns(len(segment_meanings) if segment_meanings else 1)
        for idx, (cluster_id, meaning) in enumerate(segment_meanings.items()):
            with cols[idx]:
                st.markdown(f"##### Segment {cluster_id}")
                st.write(meaning)

                subset = df_with_clusters[df_with_clusters["Cluster"] == cluster_id]
                st.caption(
                    f"Customers: {len(subset)} | "
                    f"Avg Age: {subset['Age'].mean():.1f} | "
                    f"Avg Income: {subset['Annual Income (k$)'].mean():.1f} k$"
                )

    # ====== PREDICT TAB ======
    with tab_predict:
        st.subheader("Predict Customer Segment")
        st.caption("Enter customer details. Income is provided in ₹ and converted internally.")

        col_form, col_result = st.columns((2, 1))

        with col_form:
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
            age = st.slider("Age", 18, 70, 30)
            income_inr = st.number_input("Annual Income (₹ INR)", min_value=10000, step=50000, value=500000)
            spending = st.slider("Spending Score (1-100)", 1, 100, 50)

            if st.button("Predict Segment", type="primary"):
                try:
                    # Convert INR → USD → k$
                    usd = income_inr / USD_TO_INR
                    income_k_dollar = usd / 1000

                    response = requests.post(
                        f"{API_URL}/predict",
                        json={
                            "Gender": gender,
                            "Age": age,
                            "Income": income_k_dollar,
                            "Spending": spending,
                        },
                        timeout=5,
                    )
                    response.raise_for_status()
                    data = response.json()
                    cluster = data.get("Cluster", None)

                    with col_result:
                        if cluster is None:
                            st.error("Unexpected response from backend.")
                        else:
                            meaning = segment_meanings.get(cluster, "Unknown Segment")
                            st.success(f"Predicted Segment: {cluster}")
                            st.info(f"Segment Meaning: {meaning}")
                            st.caption(
                                "Income is converted from INR → USD → k$ to match the model's training scale."
                            )

                except requests.exceptions.RequestException:
                    with col_result:
                        st.error("⚠️ Could not reach the backend API. Please make sure FastAPI is running.")


if __name__ == "__main__":
    main()
