import streamlit as st
import requests
import pandas as pd
import altair as alt
import joblib
import os
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
USD_TO_INR = 83
API_URL = "https://customer-segmentation-1-oaky.onrender.com"

MODEL_PATH  = "models/kmeans.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH   = "data/Mall_Customers.csv"

# Segment colour palette
SEGMENT_COLORS = {
    "High Income - High Spending": "#7C3AED",
    "High Income - Low Spending":  "#2563EB",
    "Low Income - High Spending":  "#D97706",
    "Low Income - Low Spending":   "#6B7280",
}
SEGMENT_ICONS = {
    "High Income - High Spending": "💎",
    "High Income - Low Spending":  "💰",
    "Low Income - High Spending":  "🛒",
    "Low Income - Low Spending":   "💤",
}

# Full strategy definitions (source of truth, always available offline)
SEGMENT_DEFINITIONS = {
    "High Income - High Spending": {
        "icon":        "💎",
        "description": "Valuable customers already spending a lot",
        "strategy":    "Loyalty rewards, premium membership, early product access",
        "goal":        "Retain & delight — maximize lifetime value",
        "color":       "#7C3AED",
    },
    "High Income - Low Spending": {
        "icon":        "💰",
        "description": "Have money but are not spending much",
        "strategy":    "Upsell, targeted marketing, premium recommendations",
        "goal":        "Convert potential — move them up the spending ladder",
        "color":       "#2563EB",
    },
    "Low Income - High Spending": {
        "icon":        "🛒",
        "description": "Spending a lot but are budget sensitive",
        "strategy":    "Discounts, bundles, retarget offers",
        "goal":        "Maintain spending without churn",
        "color":       "#D97706",
    },
    "Low Income - Low Spending": {
        "icon":        "💤",
        "description": "Low value customers — minimal engagement",
        "strategy":    "Low cost campaigns, awareness campaigns (avoid heavy spend)",
        "goal":        "Nurture gently — do not waste marketing budget",
        "color":       "#6B7280",
    },
}


# ─────────────────────────────────────────────
# SEGMENT LOGIC  (mirrors src/segment_logic.py)
# ─────────────────────────────────────────────
def get_segment_name(income_k: float, spending: float) -> str:
    if income_k >= 50 and spending >= 50:
        return "High Income - High Spending"
    elif income_k >= 50 and spending < 50:
        return "High Income - Low Spending"
    elif income_k < 50 and spending >= 50:
        return "Low Income - High Spending"
    else:
        return "Low Income - Low Spending"


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_local_model():
    """Load sklearn model + scaler from disk (works offline)."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None


@st.cache_data
def build_clustered_df():
    """
    Attach predicted cluster labels to every row in the dataset.
    Falls back to rule-based labeling if model files are missing.
    """
    df = load_raw_data().copy()

    # Encode gender
    df["GenderEncoded"] = df["Gender"].map({"Male": 0, "Female": 1})

    model, scaler = load_local_model()
    if model is not None and scaler is not None:
        features = df[["GenderEncoded", "Age",
                        "Annual Income (k$)", "Spending Score (1-100)"]].to_numpy()
        X_scaled = scaler.transform(features)
        df["Cluster"] = model.predict(X_scaled)
    else:
        # Rule-based fallback so scatter always works
        df["Cluster"] = df.apply(
            lambda r: 0 if (r["Annual Income (k$)"] >= 50 and r["Spending Score (1-100)"] >= 50)
                      else 1 if (r["Annual Income (k$)"] >= 50)
                      else 2 if (r["Spending Score (1-100)"] >= 50)
                      else 3,
            axis=1,
        )

    # Attach segment name per customer
    df["Segment"] = df.apply(
        lambda r: get_segment_name(r["Annual Income (k$)"], r["Spending Score (1-100)"]),
        axis=1,
    )
    return df


@st.cache_data(ttl=60)
def fetch_segment_summary():
    r = requests.get(f"{API_URL}/segments/summary", timeout=15)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def seg_color(name: str) -> str:
    return SEGMENT_COLORS.get(name, "#374151")

def seg_icon(name: str) -> str:
    return SEGMENT_ICONS.get(name, "📌")


# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #e2e8f0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 500;
    padding: 8px 20px;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6d28d9, #2563eb) !important;
    color: #fff !important;
    box-shadow: 0 4px 15px rgba(109,40,217,0.4);
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 18px 22px !important;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 2rem !important; font-weight: 800; }

.seg-card {
    border-radius: 16px;
    padding: 20px 18px;
    margin-bottom: 8px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
}
.seg-card h4 { margin: 0 0 6px 0; font-size: 1rem; font-weight: 700; }
.seg-card p  { margin: 0; font-size: 0.85rem; line-height: 1.5; color: rgba(255,255,255,0.85); }

.result-box {
    border-radius: 18px;
    padding: 24px;
    margin-top: 16px;
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(14px);
}
.result-box h2 { margin: 0 0 10px 0; font-size: 1.4rem; font-weight: 800; }
.result-box .label { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.75; }
.result-box .value { font-size: 0.95rem; margin-bottom: 14px; line-height: 1.6; }

.info-pill {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 14px;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 4px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.section-sub { font-size: 0.9rem; color: #94a3b8; margin-bottom: 20px; }
</style>
"""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Consumer Segmentation",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Hero header
    st.markdown("""
    <div style='text-align:center; padding:40px 0 20px 0;'>
        <div style='font-size:0.85rem; font-weight:700; letter-spacing:.15em; color:#a78bfa; margin-bottom:10px;'>
            POWERED BY K-MEANS CLUSTERING
        </div>
        <h1 style='font-size:2.8rem; font-weight:900; margin:0;
                   background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Consumer Segmentation Intelligence
        </h1>
        <p style='color:#94a3b8; margin-top:12px; font-size:1.05rem;'>
            Unlock data&#8209;driven customer insights &nbsp;&middot;&nbsp;
            Discover actionable strategies &nbsp;&middot;&nbsp;
            Predict new customers instantly
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load all local data (never fails — no API dependency for charts)
    df_raw       = load_raw_data()
    df_clustered = build_clustered_df()

    # Cluster summary from local data
    cluster_summary = (
        df_clustered
        .groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]]
        .mean()
        .reset_index()
    )
    cluster_summary["Segment"] = cluster_summary.apply(
        lambda r: get_segment_name(r["Annual Income (k$)"], r["Spending Score (1-100)"]),
        axis=1,
    )

    seg_sizes = (
        df_clustered["Segment"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Segment", "Segment": "Count", "count": "Count"})
    )
    # Make sure columns are correct regardless of pandas version
    if "Segment" not in seg_sizes.columns or "Count" not in seg_sizes.columns:
        seg_sizes = df_clustered["Segment"].value_counts().rename_axis("Segment").reset_index(name="Count")

    total_customers = len(df_clustered)
    n_segments      = df_clustered["Segment"].nunique()

    # Try to get remote summary (optional, for predict only)
    try:
        api_summary = fetch_segment_summary()
    except Exception:
        api_summary = {}

    # ── Tabs ────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠  Overview",
        "📊  Segmentation Dashboard",
        "🎯  Segment Strategy",
        "🔮  Predict Customer",
    ])

    # ══════════════════════════════════════
    # PAGE 1 — OVERVIEW
    # ══════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">What this project does and why it matters</div>', unsafe_allow_html=True)

        col_ps, col_why = st.columns(2, gap="large")
        with col_ps:
            st.markdown("""
            <div style='background:rgba(109,40,217,0.15); border:1px solid rgba(109,40,217,0.4);
                        border-radius:16px; padding:24px; height:100%;'>
                <div style='font-size:1.15rem; font-weight:800; color:#a78bfa; margin-bottom:12px;'>
                    🧩 Problem Statement
                </div>
                <p style='color:#cbd5e1; line-height:1.8; margin:0;'>
                    Retailers struggle to understand <strong>diverse customer needs</strong>.
                    A one-size-fits-all marketing approach leads to
                    <strong>wasted budget</strong>, <strong>poor engagement</strong>, and <strong>high churn</strong>.
                    <br><br>
                    This project applies <strong>K-Means Clustering</strong> on mall customer data to
                    automatically discover meaningful customer groups — enabling
                    laser-focused marketing strategies for each segment.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_why:
            st.markdown("""
            <div style='background:rgba(37,99,235,0.15); border:1px solid rgba(37,99,235,0.4);
                        border-radius:16px; padding:24px; height:100%;'>
                <div style='font-size:1.15rem; font-weight:800; color:#60a5fa; margin-bottom:12px;'>
                    💡 Why Segmentation Matters?
                </div>
                <ul style='color:#cbd5e1; line-height:2.1; margin:0; padding-left:18px;'>
                    <li><strong>Personalization</strong> — Tailor offers to each group</li>
                    <li><strong>Budget Efficiency</strong> — Stop wasting spend on wrong audiences</li>
                    <li><strong>Retention</strong> — Identify and nurture high-value customers</li>
                    <li><strong>Upsell Opportunities</strong> — Spot customers ready to spend more</li>
                    <li><strong>Churn Reduction</strong> — Proactively engage budget-sensitive buyers</li>
                    <li><strong>Data-Driven Decisions</strong> — Move from gut-feel to evidence</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_ds, col_km = st.columns((3, 2), gap="large")
        with col_ds:
            st.markdown("**📁 Dataset Summary — Mall Customers**")
            st.dataframe(df_raw.head(12), use_container_width=True, hide_index=True)
            st.caption(f"**{len(df_raw)} total records** · Features: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)")

        with col_km:
            st.markdown("**📈 Key Metrics**")
            m1, m2 = st.columns(2)
            m1.metric("👥 Total Customers", f"{total_customers:,}")
            m2.metric("🔢 Segments Found",  f"{n_segments}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background:rgba(255,255,255,0.05); border-radius:12px; padding:18px;'>
                <div style='font-size:0.8rem; font-weight:700; color:#94a3b8; letter-spacing:.08em; margin-bottom:12px;'>
                    DATASET FEATURE STATS
                </div>
            """, unsafe_allow_html=True)
            stats = df_raw[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].describe().loc[["mean", "min", "max"]]
            stats.index = ["Avg", "Min", "Max"]
            st.dataframe(stats.style.format("{:.1f}"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════
    # PAGE 2 — SEGMENTATION DASHBOARD
    # ══════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Segmentation Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Visual exploration of K-Means clusters across income &amp; spending dimensions</div>', unsafe_allow_html=True)

        col_scatter, col_bar = st.columns((3, 2), gap="large")

        # ── Scatter plot (always built from local data) ──────────────
        with col_scatter:
            st.markdown("**Income vs Spending — Colored by Cluster**")
            color_domain = list(SEGMENT_COLORS.keys())
            color_range  = list(SEGMENT_COLORS.values())

            scatter = (
                alt.Chart(df_clustered)
                .mark_circle(size=80, opacity=0.82, stroke="#0f172a", strokeWidth=0.5)
                .encode(
                    x=alt.X("Annual Income (k$):Q",
                            title="Annual Income (k$)",
                            scale=alt.Scale(zero=False),
                            axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                          gridColor="rgba(255,255,255,0.07)")),
                    y=alt.Y("Spending Score (1-100):Q",
                            title="Spending Score (1–100)",
                            scale=alt.Scale(zero=False),
                            axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                          gridColor="rgba(255,255,255,0.07)")),
                    color=alt.Color("Segment:N",
                                    scale=alt.Scale(domain=color_domain, range=color_range),
                                    legend=alt.Legend(
                                        orient="bottom",
                                        labelColor="#cbd5e1",
                                        titleColor="#94a3b8",
                                        labelFontSize=11,
                                        titleFontSize=11,
                                        title="Segment",
                                    )),
                    tooltip=[
                        alt.Tooltip("Annual Income (k$):Q",      title="Income (k$)",     format=".1f"),
                        alt.Tooltip("Spending Score (1-100):Q",  title="Spending Score",  format=".0f"),
                        alt.Tooltip("Gender:N",                  title="Gender"),
                        alt.Tooltip("Age:Q",                     title="Age"),
                        alt.Tooltip("Segment:N",                 title="Segment"),
                    ],
                )
                .properties(height=400, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(scatter, use_container_width=True)

        # ── Segment distribution bar ──────────────────────────────
        with col_bar:
            st.markdown("**Segment Distribution**")
            bar = (
                alt.Chart(seg_sizes)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Segment:N", sort="-y", title=None,
                            axis=alt.Axis(labelAngle=-18, labelColor="#94a3b8", labelLimit=180)),
                    y=alt.Y("Count:Q", title="# Customers",
                            axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                          gridColor="rgba(255,255,255,0.07)")),
                    color=alt.Color("Segment:N",
                                    scale=alt.Scale(domain=color_domain, range=color_range),
                                    legend=None),
                    tooltip=["Segment:N", "Count:Q"],
                )
                .properties(height=240, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(bar, use_container_width=True)

        # ── Cluster profile table ─────────────────────────────────
        st.markdown("---")
        st.markdown("**📋 Cluster Profile Table**")
        display_cs = cluster_summary.rename(columns={
            "Cluster":                    "Cluster ID",
            "Annual Income (k$)":         "Avg Income (k$)",
            "Spending Score (1-100)":     "Avg Spending Score",
        })[["Cluster ID", "Segment", "Avg Income (k$)", "Avg Spending Score"]]
        st.dataframe(
            display_cs.style.format({"Avg Income (k$)": "{:.1f}", "Avg Spending Score": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

        # ── Segment description cards ─────────────────────────────
        st.markdown("<br>**🗂 Segment Descriptions**", unsafe_allow_html=True)
        seg_names_present = cluster_summary["Segment"].unique().tolist()
        cols = st.columns(len(seg_names_present))

        for idx, seg_name in enumerate(seg_names_present):
            info  = SEGMENT_DEFINITIONS.get(seg_name, {})
            color = info.get("color", "#374151")
            icon  = info.get("icon",  "📌")
            desc  = info.get("description", "")
            strat = info.get("strategy", "")
            count = int(seg_sizes.loc[seg_sizes["Segment"] == seg_name, "Count"].values[0]) \
                    if seg_name in seg_sizes["Segment"].values else "—"

            with cols[idx]:
                st.markdown(
                    f'<div class="seg-card" style="background:linear-gradient(135deg,{color}22,{color}11);'
                    f'border-color:{color}55;">'
                    f'<h4>{icon} {seg_name}</h4>'
                    f'<p style="margin-bottom:8px;">{desc}</p>'
                    f'<div class="info-pill">👥 {count} customers</div>'
                    f'<p style="font-size:0.8rem; color:{color}; font-weight:600;">📌 {strat}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════
    # PAGE 3 — SEGMENT STRATEGY
    # ══════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Segment Strategy Playbook</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Targeted marketing strategies for every customer segment</div>', unsafe_allow_html=True)

        # Build strategy table using Streamlit-native components (no raw HTML table)
        st.markdown("### 📊 Strategy Overview Table")

        table_data = []
        for seg_name, info in SEGMENT_DEFINITIONS.items():
            count = int(seg_sizes.loc[seg_sizes["Segment"] == seg_name, "Count"].values[0]) \
                    if seg_name in seg_sizes["Segment"].values else 0
            table_data.append({
                "Icon":        info["icon"],
                "Segment":     seg_name,
                "Description": info["description"],
                "Strategy":    info["strategy"],
                "Goal":        info["goal"],
                "Customers":   count,
            })

        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed cards per segment
        st.markdown("### 📌 Segment Deep-Dive")

        for seg_name, info in SEGMENT_DEFINITIONS.items():
            color = info["color"]
            count = int(seg_sizes.loc[seg_sizes["Segment"] == seg_name, "Count"].values[0]) \
                    if seg_name in seg_sizes["Segment"].values else 0

            with st.expander(f"{info['icon']}  {seg_name}  ·  {count} customers", expanded=False):
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:18px;'
                        f'border-left:4px solid {color};">'
                        f'<div style="font-size:0.75rem; font-weight:700; letter-spacing:.08em; color:#94a3b8;">DESCRIPTION</div>'
                        f'<div style="font-size:0.95rem; color:#e2e8f0; margin-top:6px;">{info["description"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:18px;'
                        f'border-left:4px solid #a78bfa;">'
                        f'<div style="font-size:0.75rem; font-weight:700; letter-spacing:.08em; color:#94a3b8;">STRATEGY</div>'
                        f'<div style="font-size:0.95rem; color:#e2e8f0; margin-top:6px;">{info["strategy"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f'<div style="background:rgba(52,211,153,0.1); border-radius:12px; padding:18px;'
                        f'border-left:4px solid #34d399;">'
                        f'<div style="font-size:0.75rem; font-weight:700; letter-spacing:.08em; color:#94a3b8;">GOAL</div>'
                        f'<div style="font-size:0.95rem; color:#34d399; margin-top:6px; font-style:italic;">{info["goal"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ══════════════════════════════════════
    # PAGE 4 — PREDICT NEW CUSTOMER
    # ══════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">New Customer Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Enter customer details to instantly predict their segment and get a tailored strategy</div>', unsafe_allow_html=True)

        col_form, col_result = st.columns(2, gap="large")

        with col_form:
            st.markdown(
                '<div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.12);'
                'border-radius:16px; padding:24px;">'
                '<div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-bottom:18px;">🧾 Customer Profile</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            gender     = st.selectbox("Gender", options=[0, 1],
                                      format_func=lambda x: "👨 Male" if x == 0 else "👩 Female")
            age        = st.slider("Age", 18, 70, 30)
            income_inr = st.number_input("Annual Income (₹ INR)", min_value=10_000,
                                         step=50_000, value=500_000)
            spending   = st.slider("Spending Score (1 – 100)", 1, 100, 50)

            income_k = (income_inr / USD_TO_INR) / 1000
            st.caption(f"💱  ₹{income_inr:,.0f}  →  **${income_k:.2f}k** (used by model)")

            predict_clicked = st.button("🔮 Predict Segment", type="primary",
                                        use_container_width=True)

        with col_result:
            if predict_clicked:
                # ── Local prediction (always works) ────────────────
                seg_name = get_segment_name(income_k, spending)
                info     = SEGMENT_DEFINITIONS[seg_name]
                color    = info["color"]
                icon     = info["icon"]

                # Also try remote API for cluster number (optional)
                cluster_id = "—"
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={
                            "Gender":   gender,
                            "Age":      age,
                            "Income":   income_k,
                            "Spending": spending,
                        },
                        timeout=8,
                    )
                    if resp.ok:
                        cluster_id = resp.json().get("Cluster", "—")
                except Exception:
                    pass  # Use local rule-based result

                st.markdown(
                    f'<div class="result-box" style="background:linear-gradient(135deg,{color}22,{color}11);'
                    f'border-color:{color}55;">'
                    f'<h2>{icon} {seg_name}</h2>'
                    + (f'<div style="background:{color}33; border-radius:8px; display:inline-block;'
                       f'padding:4px 14px; font-size:0.8rem; font-weight:700; color:{color}; margin-bottom:16px;">'
                       f'Cluster {cluster_id}</div>'
                       if cluster_id != "—" else "")
                    + f'<div class="label">📝 What it means</div>'
                    f'<div class="value">{info["description"]}</div>'
                    f'<div class="label">🎯 Recommended Strategy</div>'
                    f'<div class="value" style="color:{color};">{info["strategy"]}</div>'
                    f'<div class="label">🏁 Goal</div>'
                    f'<div class="value" style="color:#34d399; font-style:italic;">{info["goal"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:14px;">'
                    f'<div class="info-pill">👤 Age {age}</div>'
                    f'<div class="info-pill">💰 ₹{income_inr:,.0f}</div>'
                    f'<div class="info-pill">🛍️ Spending {spending}/100</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background:rgba(255,255,255,0.04); border:2px dashed rgba(255,255,255,0.15);'
                    'border-radius:16px; padding:60px 30px; text-align:center; margin-top:10px;">'
                    '<div style="font-size:3rem; margin-bottom:12px;">🔮</div>'
                    '<div style="font-size:1rem; font-weight:600; color:#94a3b8;">'
                    'Fill in the customer profile and click<br>'
                    '<strong style="color:#a78bfa;">Predict Segment</strong></div>'
                    '<div style="font-size:0.82rem; color:#475569; margin-top:10px;">'
                    'You\'ll see the segment name, explanation &amp; recommended strategy'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

    # Footer
    st.markdown(
        '<div style="text-align:center; padding:30px 0 10px 0; color:#475569; font-size:0.78rem;">'
        'Consumer Segmentation Intelligence &nbsp;&middot;&nbsp; K-Means Clustering &nbsp;&middot;&nbsp;'
        'Built with Streamlit &amp; FastAPI'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()