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
USD_TO_INR        = 83
API_URL           = "https://customer-segmentation-l8z2.onrender.com"
MODEL_PATH        = "models/kmeans.pkl"
SCALER_PATH       = "models/scaler.pkl"
CLUSTER_MAP_PATH  = "models/cluster_map.json"
DATA_PATH         = "data/Mall_Customers.csv"

SEGMENT_COLORS = {
    "High Income - High Spending": "#7C3AED",
    "High Income - Low Spending":  "#2563EB",
    "Low Income - High Spending":  "#D97706",
    "Low Income - Low Spending":   "#6B7280",
}

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
# SEGMENT LOGIC  (used as local fallback only)
# ─────────────────────────────────────────────
def get_segment_name(income_k: float, spending: float) -> str:
    """Rule-based fallback — only used when cluster_map AND API are both unavailable."""
    if income_k >= 50 and spending >= 50:
        return "High Income - High Spending"
    elif income_k >= 50:
        return "High Income - Low Spending"
    elif spending >= 50:
        return "Low Income - High Spending"
    else:
        return "Low Income - Low Spending"


def cluster_to_segment(cluster_id: int, cluster_map: dict, income_k: float, spending: float) -> str:
    """Return segment name from cluster_map if available, else fall back to rules."""
    if cluster_id is not None and cluster_map and cluster_id in cluster_map:
        return cluster_map[cluster_id]
    return get_segment_name(income_k, spending)


def assign_cluster_segments(cs_df: pd.DataFrame) -> pd.DataFrame:
    """Rank cluster centroids to assign unique segment labels to the profile table."""
    df   = cs_df.copy().reset_index(drop=True)
    half = len(df) // 2

    inc_ranks    = df["Annual Income (k$)"].rank(ascending=False, method="first").astype(int)
    high_inc_idx = df.index[inc_ranks <= half].tolist()
    low_inc_idx  = df.index[inc_ranks >  half].tolist()

    hi_grp = df.loc[high_inc_idx].sort_values("Spending Score (1-100)", ascending=False)
    lo_grp = df.loc[low_inc_idx ].sort_values("Spending Score (1-100)", ascending=False)

    seg_map = {}
    if len(hi_grp) >= 1: seg_map[hi_grp.index[0]] = "High Income - High Spending"
    if len(hi_grp) >= 2: seg_map[hi_grp.index[1]] = "High Income - Low Spending"
    if len(lo_grp) >= 1: seg_map[lo_grp.index[0]] = "Low Income - High Spending"
    if len(lo_grp) >= 2: seg_map[lo_grp.index[1]] = "Low Income - Low Spending"

    df["Segment"] = df.index.map(seg_map)
    return df


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_local_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    return None, None


@st.cache_data
def load_cluster_map() -> dict:
    """Load cluster→segment mapping saved at training time."""
    if os.path.exists(CLUSTER_MAP_PATH):
        import json
        with open(CLUSTER_MAP_PATH) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return {}



@st.cache_data
def build_clustered_df() -> pd.DataFrame:
    """Attach cluster labels & cluster-driven segments to every row."""
    df = load_raw_data().copy()
    df["GenderEncoded"] = df["Gender"].map({"Male": 0, "Female": 1})

    cluster_map = load_cluster_map()
    model, scaler = load_local_model()

    if model is not None:
        X = df[["Annual Income (k$)", "Spending Score (1-100)"]].to_numpy()
        df["Cluster"] = model.predict(scaler.transform(X))
    else:
        # No local model — assign a placeholder cluster via rules
        df["Cluster"] = df.apply(
            lambda r: 0 if (r["Annual Income (k$)"] >= 50 and r["Spending Score (1-100)"] >= 50)
                      else 1 if  r["Annual Income (k$)"] >= 50
                      else 2 if  r["Spending Score (1-100)"] >= 50
                      else 3,
            axis=1,
        )

    # Segment driven by cluster_map; falls back to threshold rules if map unavailable
    df["Segment"] = df.apply(
        lambda r: cluster_to_segment(
            int(r["Cluster"]), cluster_map,
            r["Annual Income (k$)"], r["Spending Score (1-100)"]
        ),
        axis=1,
    )
    return df


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
[data-testid="stMetricLabel"] { color:#94a3b8 !important; font-size:0.78rem; font-weight:600; letter-spacing:0.05em; }
[data-testid="stMetricValue"] { color:#e2e8f0 !important; font-size:2rem !important; font-weight:800; }

.seg-card {
    border-radius: 16px; padding: 20px 18px; margin-bottom: 8px;
    backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.15);
}
.seg-card h4 { margin: 0 0 6px 0; font-size: 1rem; font-weight: 700; }
.seg-card p  { margin: 0; font-size: 0.85rem; line-height: 1.5; color: rgba(255,255,255,0.85); }

.result-box {
    border-radius: 18px; padding: 24px; margin-top: 16px;
    border: 1px solid rgba(255,255,255,0.18); backdrop-filter: blur(14px);
}
.result-box h2  { margin: 0 0 10px 0; font-size: 1.4rem; font-weight: 800; }
.result-box .label { font-size:0.75rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; opacity:0.75; }
.result-box .value { font-size:0.95rem; margin-bottom:14px; line-height:1.6; }

.info-pill {
    display: inline-block; background: rgba(255,255,255,0.12);
    border-radius: 999px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 600; margin: 4px 4px 4px 0;
}
.section-title {
    font-size:1.5rem; font-weight:800; margin-bottom:4px;
    background: linear-gradient(90deg,#a78bfa,#60a5fa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.section-sub { font-size:0.9rem; color:#94a3b8; margin-bottom:20px; }
</style>
"""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Consumer Segmentation Intelligence",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero header ──────────────────────────
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

    # ── API health banner ──────────────────────
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            st.success("🟢 FastAPI service connected")
        else:
            st.warning("🟡 API reachable but unhealthy")
    except requests.exceptions.RequestException:
        st.error("🔴 FastAPI service offline — predictions will use local rule-based fallback")

    # ── Load local data ────────────────────────
    df_raw       = load_raw_data()
    df_clustered = build_clustered_df()

    cluster_summary = (
        df_clustered
        .groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]]
        .mean()
        .reset_index()
    )
    cluster_summary = assign_cluster_segments(cluster_summary)

    seg_sizes = (
        df_clustered["Segment"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Count")
    )

    total_customers = len(df_clustered)
    n_segments      = df_clustered["Segment"].nunique()

    color_domain = list(SEGMENT_COLORS.keys())
    color_range  = list(SEGMENT_COLORS.values())

    # ── Tabs ──────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠  Overview",
        "📊  Segmentation Dashboard",
        "🎯  Segment Strategy",
        "🔮  Predict Customer",
    ])

    # ══════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">What this project does and why it matters</div>', unsafe_allow_html=True)

        col_ps, col_why = st.columns(2, gap="large")
        with col_ps:
            st.markdown("""
            <div style='background:rgba(109,40,217,0.15); border:1px solid rgba(109,40,217,0.4);
                        border-radius:16px; padding:24px;'>
                <div style='font-size:1.1rem; font-weight:800; color:#a78bfa; margin-bottom:12px;'>🧩 Problem Statement</div>
                <p style='color:#cbd5e1; line-height:1.8; margin:0;'>
                    Retailers struggle to understand <strong>diverse customer needs</strong>.
                    A one-size-fits-all approach leads to <strong>wasted budget</strong>,
                    <strong>poor engagement</strong>, and <strong>high churn</strong>.<br><br>
                    This project applies <strong>K-Means Clustering</strong> on mall customer data to
                    discover meaningful groups — enabling laser-focused marketing strategies.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_why:
            st.markdown("""
            <div style='background:rgba(37,99,235,0.15); border:1px solid rgba(37,99,235,0.4);
                        border-radius:16px; padding:24px;'>
                <div style='font-size:1.1rem; font-weight:800; color:#60a5fa; margin-bottom:12px;'>💡 Why Segmentation Matters</div>
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
            st.markdown("**📁 Dataset Sample — Mall Customers**")
            st.dataframe(df_raw.head(12), use_container_width=True, hide_index=True)
            st.caption(f"**{len(df_raw)} total records** · CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)")

        with col_km:
            st.markdown("**📈 Key Metrics**")
            m1, m2 = st.columns(2)
            m1.metric("👥 Total Customers", f"{total_customers:,}")
            m2.metric("🔢 Segments Found",  str(n_segments))
            st.markdown("<br>", unsafe_allow_html=True)
            stats = df_raw[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].describe().loc[["mean","min","max"]]
            stats.index = ["Avg", "Min", "Max"]
            st.dataframe(stats.style.format("{:.1f}"), use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 2 — SEGMENTATION DASHBOARD
    # ══════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Segmentation Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Visual exploration of K-Means clusters across income &amp; spending dimensions</div>', unsafe_allow_html=True)

        col_scatter, col_bar = st.columns((3, 2), gap="large")

        with col_scatter:
            st.markdown("**Income vs Spending — Colored by Segment**")
            scatter = (
                alt.Chart(df_clustered)
                .mark_circle(size=80, opacity=0.82, stroke="#0f172a", strokeWidth=0.5)
                .encode(
                    x=alt.X("Annual Income (k$):Q", scale=alt.Scale(zero=False),
                            axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                          gridColor="rgba(255,255,255,0.07)")),
                    y=alt.Y("Spending Score (1-100):Q", title="Spending Score (1–100)",
                            scale=alt.Scale(zero=False),
                            axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                          gridColor="rgba(255,255,255,0.07)")),
                    color=alt.Color("Segment:N",
                                    scale=alt.Scale(domain=color_domain, range=color_range),
                                    legend=alt.Legend(orient="bottom", labelColor="#cbd5e1",
                                                      titleColor="#94a3b8", title="Segment")),
                    tooltip=[
                        alt.Tooltip("Annual Income (k$):Q",     title="Income (k$)",    format=".1f"),
                        alt.Tooltip("Spending Score (1-100):Q", title="Spending Score", format=".0f"),
                        alt.Tooltip("Gender:N",  title="Gender"),
                        alt.Tooltip("Age:Q",     title="Age"),
                        alt.Tooltip("Segment:N", title="Segment"),
                    ],
                )
                .properties(height=420, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(scatter, use_container_width=True)

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
                .properties(height=260, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(bar, use_container_width=True)

        # Cluster profile table
        st.markdown("---")
        st.markdown("**📋 Cluster Profile Table**")
        display_cs = cluster_summary.rename(columns={
            "Cluster":                "Cluster ID",
            "Annual Income (k$)":     "Avg Income (k$)",
            "Spending Score (1-100)": "Avg Spending Score",
        })[["Cluster ID", "Segment", "Avg Income (k$)", "Avg Spending Score"]]
        st.dataframe(
            display_cs.style.format({"Avg Income (k$)": "{:.1f}", "Avg Spending Score": "{:.1f}"}),
            use_container_width=True, hide_index=True,
        )

        # Segment description cards
        st.markdown("<br>**🗂 Segment Descriptions**", unsafe_allow_html=True)
        seg_names = cluster_summary["Segment"].dropna().unique().tolist()
        cols = st.columns(len(seg_names))

        for idx, seg_name in enumerate(seg_names):
            info  = SEGMENT_DEFINITIONS.get(seg_name, {})
            color = info.get("color", "#374151")
            icon  = info.get("icon", "📌")
            desc  = info.get("description", "")
            strat = info.get("strategy", "")
            count_row = seg_sizes.loc[seg_sizes["Segment"] == seg_name, "Count"]
            count = int(count_row.values[0]) if len(count_row) > 0 else "—"

            with cols[idx]:
                st.markdown(
                    f'<div class="seg-card" style="background:linear-gradient(135deg,{color}22,{color}11);'
                    f'border-color:{color}55;">'
                    f'<h4>{icon} {seg_name}</h4>'
                    f'<p style="margin-bottom:8px;">{desc}</p>'
                    f'<div class="info-pill">👥 {count} customers</div>'
                    f'<p style="font-size:0.8rem; color:{color}; font-weight:600; margin:0;">📌 {strat}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════
    # TAB 3 — SEGMENT STRATEGY
    # ══════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Segment Strategy Playbook</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Targeted marketing strategies for every customer segment</div>', unsafe_allow_html=True)

        st.markdown("### 📊 Strategy Overview Table")
        table_data = []
        for seg_name, info in SEGMENT_DEFINITIONS.items():
            count_row = seg_sizes.loc[seg_sizes["Segment"] == seg_name, "Count"]
            count = int(count_row.values[0]) if len(count_row) > 0 else 0
            table_data.append({
                "Icon":        info["icon"],
                "Segment":     seg_name,
                "Description": info["description"],
                "Strategy":    info["strategy"],
                "Goal":        info["goal"],
                "Customers":   count,
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        st.markdown("<br>### 📌 Segment Deep-Dive", unsafe_allow_html=True)
        for seg_name, info in SEGMENT_DEFINITIONS.items():
            color     = info["color"]
            count_row = seg_sizes.loc[seg_sizes["Segment"] == seg_name, "Count"]
            count     = int(count_row.values[0]) if len(count_row) > 0 else 0

            with st.expander(f"{info['icon']}  {seg_name}  ·  {count} customers", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:18px;border-left:4px solid {color};">'
                        f'<div style="font-size:0.75rem;font-weight:700;letter-spacing:.08em;color:#94a3b8;">DESCRIPTION</div>'
                        f'<div style="font-size:0.95rem;color:#e2e8f0;margin-top:6px;">{info["description"]}</div>'
                        f'</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:18px;border-left:4px solid #a78bfa;">'
                        f'<div style="font-size:0.75rem;font-weight:700;letter-spacing:.08em;color:#94a3b8;">STRATEGY</div>'
                        f'<div style="font-size:0.95rem;color:#e2e8f0;margin-top:6px;">{info["strategy"]}</div>'
                        f'</div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(
                        f'<div style="background:rgba(52,211,153,0.1);border-radius:12px;padding:18px;border-left:4px solid #34d399;">'
                        f'<div style="font-size:0.75rem;font-weight:700;letter-spacing:.08em;color:#94a3b8;">GOAL</div>'
                        f'<div style="font-size:0.95rem;color:#34d399;margin-top:6px;font-style:italic;">{info["goal"]}</div>'
                        f'</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # TAB 4 — PREDICT NEW CUSTOMER
    # ══════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">New Customer Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Enter customer details — the backend KMeans model predicts their segment instantly</div>', unsafe_allow_html=True)

        col_form, col_result = st.columns(2, gap="large")

        with col_form:
            st.markdown(
                '<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.12);'
                'border-radius:16px;padding:24px;">'
                '<div style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:18px;">🧾 Customer Profile</div>'
                '</div>', unsafe_allow_html=True)

            gender     = st.selectbox("Gender", options=[0, 1],
                                      format_func=lambda x: "👨 Male" if x == 0 else "👩 Female")
            age        = st.slider("Age", 18, 70, 30)
            income_inr = st.number_input("Annual Income (₹ INR)", min_value=10_000,
                                         step=50_000, value=500_000)
            spending   = st.slider("Spending Score (1–100)", 1, 100, 50)

            income_k = (income_inr / USD_TO_INR) / 1000
            st.caption(f"💱  ₹{income_inr:,.0f}  →  **${income_k:.2f}k** (sent to model)")

            predict_clicked = st.button("🔮 Predict Segment", type="primary",
                                        use_container_width=True)

        with col_result:
            if predict_clicked:
                # ── Call backend (KMeans model runs server-side) ──
                with st.spinner("Calling model…"):
                    try:
                        resp = requests.post(
                            f"{API_URL}/predict",
                            json={
                                "Gender":   gender,
                                "Age":      age,
                                "Income":   income_k,
                                "Spending": spending,
                            },
                            timeout=15,
                        )

                        if resp.ok:
                            result     = resp.json()
                            # cluster_id  — the raw KMeans cluster number (0-3)
                            cluster_id = result.get("Cluster", "—")
                            # segment_name — derived by the backend from the cluster centroid's
                            #                income/spending position (authoritative label)
                            seg_name   = result.get("segment_name", get_segment_name(income_k, spending))
                            used_api   = True
                        else:
                            st.warning(f"⚠️ API returned {resp.status_code} — using local rule-based result.")
                            cluster_id = "—"
                            seg_name   = get_segment_name(income_k, spending)
                            used_api   = False

                    except requests.exceptions.RequestException:
                        st.warning("⚠️ FastAPI service unreachable — using local cluster-based fallback.")
                        # Try to use the local model + cluster_map before falling back to rules
                        _model, _scaler = load_local_model()
                        _cluster_map    = load_cluster_map()
                        if _model is not None:
                            _X         = np.array([[gender, age, income_k, spending]])
                            cluster_id = int(_model.predict(_scaler.transform(_X))[0])
                            seg_name   = cluster_to_segment(cluster_id, _cluster_map, income_k, spending)
                        else:
                            cluster_id = "—"
                            seg_name   = get_segment_name(income_k, spending)
                        used_api = False

                # ── Render result card ──────────────────────────
                info  = SEGMENT_DEFINITIONS.get(seg_name, SEGMENT_DEFINITIONS["Low Income - Low Spending"])
                color = info["color"]
                icon  = info["icon"]

                st.markdown(
                    f'<div class="result-box" style="background:linear-gradient(135deg,{color}22,{color}11);'
                    f'border-color:{color}55;">'
                    f'<h2>{icon} {seg_name}</h2>'
                    + (f'<div style="background:{color}33;border-radius:8px;display:inline-block;'
                       f'padding:4px 14px;font-size:0.8rem;font-weight:700;color:{color};margin-bottom:16px;">'
                       f'🤖 KMeans Cluster {cluster_id}</div>'
                       if cluster_id != "—" else
                       '<div style="background:rgba(217,119,6,0.3);border-radius:8px;display:inline-block;'
                       'padding:4px 14px;font-size:0.8rem;font-weight:700;color:#D97706;margin-bottom:16px;">'
                       '📐 Rule-Based Fallback</div>')
                    + f'<div class="label">📝 What it means</div>'
                    f'<div class="value">{info["description"]}</div>'
                    f'<div class="label">🎯 Recommended Strategy</div>'
                    f'<div class="value" style="color:{color};">{info["strategy"]}</div>'
                    f'<div class="label">🏁 Goal</div>'
                    f'<div class="value" style="color:#34d399;font-style:italic;">{info["goal"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;">'
                    f'<div class="info-pill">👤 Age {age}</div>'
                    f'<div class="info-pill">💰 ₹{income_inr:,.0f}</div>'
                    f'<div class="info-pill">🛍️ Spending {spending}/100</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    '<div style="background:rgba(255,255,255,0.04);border:2px dashed rgba(255,255,255,0.15);'
                    'border-radius:16px;padding:60px 30px;text-align:center;margin-top:10px;">'
                    '<div style="font-size:3rem;margin-bottom:12px;">🔮</div>'
                    '<div style="font-size:1rem;font-weight:600;color:#94a3b8;">'
                    'Fill in the customer profile and click<br>'
                    '<strong style="color:#a78bfa;">Predict Segment</strong></div>'
                    '<div style="font-size:0.82rem;color:#475569;margin-top:10px;">'
                    'The backend KMeans model will predict the cluster &amp; segment instantly'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

    # ── Footer ────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:30px 0 10px 0;color:#475569;font-size:0.78rem;">'
        'Consumer Segmentation Intelligence &nbsp;&middot;&nbsp; K-Means Clustering'
        ' &nbsp;&middot;&nbsp; Built with Streamlit &amp; FastAPI'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()