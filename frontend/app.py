import streamlit as st
import requests
import pandas as pd
import os


USD_TO_INR = 83  # Fixed (demo) conversion rate


@st.cache_data
def load_data():
    return pd.read_csv("data/Mall_Customers.csv")


def get_api_url() -> str:
    try:
        secret_url = st.secrets.get("API_URL", None)
        if secret_url:
            return str(secret_url).rstrip("/")
    except Exception:
        pass

    env_url = os.environ.get("API_URL")
    if env_url:
        return env_url.rstrip("/")

    # Local dev default
    return "http://127.0.0.1:8000"


@st.cache_data(ttl=60)
def fetch_segment_summary(api_url: str):
    r = requests.get(f"{api_url}/segments/summary", timeout=15)
    r.raise_for_status()
    return r.json()


def main():
    st.set_page_config(page_title="Consumer Segmentation", layout="wide")

    st.title("🚀 Consumer Segmentation Dashboard")
    st.markdown(
        "Understand your customers through **data‑driven segmentation** using K‑Means clustering. "
        "Interactively explore segments and score new customers."
    )

    api_url = get_api_url()

    with st.sidebar:
        st.markdown("### Settings")
        st.write("Backend API URL")
        st.code(api_url)
        st.caption("For Streamlit Cloud: set `API_URL` in app secrets.")

    df_raw = load_data()

    try:
        summary = fetch_segment_summary(api_url)
    except requests.exceptions.RequestException:
        st.error(
            "Backend API is not reachable. Please deploy/run FastAPI and set `API_URL` correctly."
        )
        st.stop()

    if isinstance(summary, dict) and summary.get("error"):
        st.error(summary["error"])
        st.stop()

    segment_meanings = summary.get("segment_meanings", {})
    cluster_summary_df = pd.DataFrame(summary.get("cluster_summary", []))
    segment_sizes_df = pd.DataFrame(summary.get("segment_sizes", []))

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
            st.dataframe(df_raw.head(15), use_container_width=True)

        with col_right:
            st.markdown("**Key metrics**")
            total_customers = summary.get("total_customers", 0)
            n_segments = summary.get("n_segments", 0)

            col_a, col_b = st.columns(2)
            col_a.metric("Total Customers", f"{total_customers}")
            col_b.metric("Number of Segments", f"{n_segments}")

            st.markdown("**Segment Size Distribution**")
            if not segment_sizes_df.empty and "Cluster" in segment_sizes_df.columns:
                st.bar_chart(segment_sizes_df.set_index("Cluster"))
            else:
                st.info("No segment size data available.")

    # ====== SEGMENT INSIGHTS TAB ======
    with tab_segments:
        st.subheader("Segment Interpretation")
        st.markdown(
            "Each segment is automatically described based on **average income** and "
            "**spending score**."
        )

        # Show high‑level summary table
        st.markdown("**Average Income & Spending by Segment**")
        if not cluster_summary_df.empty:
            st.dataframe(
                cluster_summary_df.style.format(
                    {"Annual Income (k$)": "{:.1f}", "Spending Score (1-100)": "{:.1f}"}
                ),
                use_container_width=True,
            )
        else:
            st.info("No cluster summary available.")

        st.markdown("---")
        st.markdown("**Business‑friendly segment descriptions**")

        items = list(segment_meanings.items())
        try:
            items = sorted(items, key=lambda x: int(x[0]))
        except Exception:
            pass

        cols = st.columns(len(items) if items else 1)
        for idx, (cluster_id, meaning) in enumerate(items):
            with cols[idx % len(cols)]:
                st.markdown(f"##### Segment {cluster_id}")
                st.write(meaning)
                if not segment_sizes_df.empty and "Cluster" in segment_sizes_df.columns:
                    row = segment_sizes_df[segment_sizes_df["Cluster"] == int(cluster_id)]
                    if len(row) == 1:
                        st.caption(f"Customers: {int(row.iloc[0]['Count'])}")

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
                        f"{api_url}/predict",
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
                            meaning = segment_meanings.get(
                                str(cluster),
                                segment_meanings.get(cluster, "Unknown Segment"),
                            )
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
