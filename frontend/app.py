import streamlit as st
import requests
import pandas as pd
import altair as alt
import joblib
import os
import json
import datetime as dt
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_URL          = "https://customer-segmentation-l8z2.onrender.com"
MODEL_PATH       = "models/kmeans.pkl"
SCALER_PATH      = "models/scaler.pkl"
CLUSTER_MAP_PATH = "models/cluster_map.json"
DATA_PATH        = "data/data.csv"
RFM_PATH         = "rfm_segments.csv"

SEGMENT_COLORS = {
    "High Value": "#7C3AED",
    "Loyal":      "#2563EB",
    "At Risk":    "#D97706",
    "Low Value":  "#6B7280",
}

SEGMENT_DEFINITIONS = {
    "High Value": {
        "icon":        "💎",
        "description": "High-spend, high-frequency, recent buyers — your most profitable customers",
        "strategy":    "VIP programmes, exclusive early access, personal account managers",
        "goal":        "Retain & delight — maximise lifetime value",
        "color":       "#7C3AED",
    },
    "Loyal": {
        "icon":        "🔄",
        "description": "Frequent buyers with moderate spend — consistent and reliable",
        "strategy":    "Cross-sell, volume discounts, subscription incentives",
        "goal":        "Grow basket size and increase spend per order",
        "color":       "#2563EB",
    },
    "At Risk": {
        "icon":        "⚠️",
        "description": "Customers who haven't purchased recently — potential churners",
        "strategy":    "Win-back campaigns, personalised outreach, special discounts",
        "goal":        "Re-engage before permanent churn",
        "color":       "#D97706",
    },
    "Low Value": {
        "icon":        "💤",
        "description": "Infrequent, low-spend customers — minimal engagement",
        "strategy":    "Low-cost awareness campaigns, product discovery nudges",
        "goal":        "Nurture gently — avoid wasting marketing budget",
        "color":       "#6B7280",
    },
}


# ─────────────────────────────────────────────
# SEGMENT LOGIC  (local rule-based fallback)
# ─────────────────────────────────────────────
def get_segment_name_local(recency: float, frequency: float, monetary: float) -> str:
    if monetary > 1000 and frequency > 10:
        return "High Value"
    elif frequency > 10:
        return "Loyal"
    elif recency > 100:
        return "At Risk"
    else:
        return "Low Value"


def cluster_to_segment(cluster_id: int, cluster_map: dict,
                       recency: float, frequency: float, monetary: float) -> str:
    if cluster_id is not None and cluster_map and cluster_id in cluster_map:
        return cluster_map[cluster_id]
    return get_segment_name_local(recency, frequency, monetary)


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data
def load_raw_transactions() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding="ISO-8859-1")


@st.cache_data
def build_rfm_df() -> pd.DataFrame:
    """Build cleaned RFM table from raw transactions."""
    df = load_raw_transactions()
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]

    reference_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate",  lambda x: (reference_date - x.max()).days),
        Frequency = ("InvoiceNo",    "count"),
        Monetary  = ("TotalPrice",   "sum"),
    ).reset_index()

    # Remove outliers (99th percentile)
    rfm = rfm[rfm["Monetary"]  < rfm["Monetary"].quantile(0.99)]
    rfm = rfm[rfm["Frequency"] < rfm["Frequency"].quantile(0.99)]
    return rfm


@st.cache_data
def load_rfm_with_segments() -> pd.DataFrame:
    """Load from saved rfm_segments.csv if available, else compute in-app."""
    if os.path.exists(RFM_PATH):
        rfm = pd.read_csv(RFM_PATH, index_col=0)
        if "Segment" not in rfm.columns:
            rfm["Segment"] = rfm.apply(
                lambda r: get_segment_name_local(r["Recency"], r["Frequency"], r["Monetary"]),
                axis=1,
            )
        return rfm

    # Fallback: compute without model
    rfm = build_rfm_df()
    cluster_map = load_cluster_map()
    model, scaler = load_local_model()

    if model is not None:
        X              = rfm[["Recency", "Frequency", "Monetary"]].to_numpy()
        rfm["Cluster"] = model.predict(scaler.transform(X))
        rfm["Segment"] = rfm.apply(
            lambda r: cluster_to_segment(
                int(r["Cluster"]), cluster_map,
                r["Recency"], r["Frequency"], r["Monetary"]
            ),
            axis=1,
        )
    else:
        rfm["Cluster"] = -1
        rfm["Segment"] = rfm.apply(
            lambda r: get_segment_name_local(r["Recency"], r["Frequency"], r["Monetary"]),
            axis=1,
        )
    return rfm


@st.cache_data
def load_local_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    return None, None


@st.cache_data
def load_cluster_map() -> dict:
    if os.path.exists(CLUSTER_MAP_PATH):
        with open(CLUSTER_MAP_PATH) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return {}


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
        page_title="RFM Customer Segmentation",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero header ──────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:40px 0 20px 0;'>
        <div style='font-size:0.85rem; font-weight:700; letter-spacing:.15em; color:#a78bfa; margin-bottom:10px;'>
            POWERED BY RFM ANALYSIS &amp; K-MEANS CLUSTERING
        </div>
        <h1 style='font-size:2.8rem; font-weight:900; margin:0;
                   background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            RFM Customer Segmentation Intelligence
        </h1>
        <p style='color:#94a3b8; margin-top:12px; font-size:1.05rem;'>
            Recency &nbsp;&middot;&nbsp; Frequency &nbsp;&middot;&nbsp; Monetary
            &nbsp;&middot;&nbsp; Discover actionable customer segments
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: cache refresh ─────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        if st.button("🔄 Refresh Data Cache", use_container_width=True,
                     help="Clear cached data — use this after retraining the model"):
            st.cache_data.clear()
            st.success("Cache cleared! Reloading…")
            st.rerun()

    # ── API health banner ──────────────────────
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            st.success("🟢 FastAPI service connected")
        else:
            st.warning("🟡 API reachable but unhealthy")
    except requests.exceptions.RequestException:
        st.error("🔴 FastAPI service offline — predictions will use local model / rule-based fallback")

    # ── Load data ────────────────────────────
    df_raw   = load_raw_transactions()
    rfm_df   = load_rfm_with_segments()

    seg_sizes = (
        rfm_df["Segment"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Count")
    )

    total_customers = rfm_df["CustomerID"].nunique() if "CustomerID" in rfm_df.columns else len(rfm_df)
    n_segments      = rfm_df["Segment"].nunique()

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
                    E-commerce businesses struggle to understand <strong>diverse customer behaviours</strong>.
                    A one-size-fits-all approach leads to <strong>wasted budget</strong>,
                    <strong>poor engagement</strong>, and <strong>high churn</strong>.<br><br>
                    This project applies <strong>RFM Analysis + K-Means Clustering</strong> on real UK
                    e-commerce transaction data to discover meaningful customer groups — enabling
                    laser-focused marketing strategies.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_why:
            st.markdown("""
            <div style='background:rgba(37,99,235,0.15); border:1px solid rgba(37,99,235,0.4);
                        border-radius:16px; padding:24px;'>
                <div style='font-size:1.1rem; font-weight:800; color:#60a5fa; margin-bottom:12px;'>💡 Why RFM Segmentation?</div>
                <ul style='color:#cbd5e1; line-height:2.1; margin:0; padding-left:18px;'>
                    <li><strong>Recency</strong> — How recently did they buy? (lower = better)</li>
                    <li><strong>Frequency</strong> — How often do they buy? (higher = better)</li>
                    <li><strong>Monetary</strong> — How much do they spend? (higher = better)</li>
                    <li><strong>Actionable</strong> — Each segment gets a targeted strategy</li>
                    <li><strong>Scalable</strong> — Works on millions of transactions</li>
                    <li><strong>Proven</strong> — Industry-standard for retail CRM</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_ds, col_km = st.columns((3, 2), gap="large")

        with col_ds:
            st.markdown("**📁 Dataset Sample — UK E-Commerce Transactions**")
            st.dataframe(df_raw.head(12), use_container_width=True, hide_index=True)
            n_countries = df_raw["Country"].nunique() if "Country" in df_raw.columns else "—"
            date_min    = pd.to_datetime(df_raw["InvoiceDate"]).min().strftime("%d %b %Y") if "InvoiceDate" in df_raw.columns else "—"
            date_max    = pd.to_datetime(df_raw["InvoiceDate"]).max().strftime("%d %b %Y") if "InvoiceDate" in df_raw.columns else "—"
            st.caption(
                f"**{len(df_raw):,} total rows** · {n_countries} countries · "
                f"{date_min} → {date_max}"
            )

        with col_km:
            st.markdown("**📈 Key Metrics**")
            m1, m2 = st.columns(2)
            m1.metric("👥 Unique Customers", f"{total_customers:,}")
            m2.metric("🔢 Segments Found",   str(n_segments))
            st.markdown("<br>", unsafe_allow_html=True)

            # RFM summary stats
            stats = rfm_df[["Recency", "Frequency", "Monetary"]].describe().loc[["mean", "min", "max"]]
            stats.index = ["Avg", "Min", "Max"]
            st.dataframe(stats.style.format("{:.1f}"), use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 2 — SEGMENTATION DASHBOARD
    # ══════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Segmentation Dashboard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Visual exploration of K-Means RFM clusters</div>',
            unsafe_allow_html=True,
        )

        col_scatter, col_bar = st.columns((3, 2), gap="large")

        with col_scatter:
            st.markdown("**Monetary vs Recency — Coloured by Segment**")
            # Sample for performance (large dataset)
            plot_df = rfm_df.sample(min(3000, len(rfm_df)), random_state=42) if len(rfm_df) > 3000 else rfm_df
            scatter = (
                alt.Chart(plot_df)
                .mark_circle(size=60, opacity=0.75, stroke="#0f172a", strokeWidth=0.5)
                .encode(
                    x=alt.X("Recency:Q",
                            title="Recency (days since last purchase)",
                            scale=alt.Scale(zero=False),
                            axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                          gridColor="rgba(255,255,255,0.07)")),
                    y=alt.Y("Monetary:Q",
                            title="Monetary (£ total spend)",
                            scale=alt.Scale(zero=False),
                            axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                          gridColor="rgba(255,255,255,0.07)")),
                    color=alt.Color("Segment:N",
                                    scale=alt.Scale(domain=color_domain, range=color_range),
                                    legend=alt.Legend(orient="bottom", labelColor="#cbd5e1",
                                                      titleColor="#94a3b8", title="Segment")),
                    tooltip=[
                        alt.Tooltip("Recency:Q",   title="Recency (days)", format=".0f"),
                        alt.Tooltip("Frequency:Q", title="Frequency",      format=".0f"),
                        alt.Tooltip("Monetary:Q",  title="Monetary (£)",   format=".2f"),
                        alt.Tooltip("Segment:N",   title="Segment"),
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
                            axis=alt.Axis(labelAngle=-18, labelColor="#94a3b8", labelLimit=160)),
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

        # Frequency distribution
        st.markdown("---")
        col_freq, col_rec = st.columns(2, gap="large")

        with col_freq:
            st.markdown("**Frequency Distribution (orders per customer)**")
            freq_hist = (
                alt.Chart(rfm_df)
                .mark_bar(color="#a78bfa", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    alt.X("Frequency:Q", bin=alt.Bin(maxbins=40), title="Number of Orders"),
                    alt.Y("count()", title="# Customers",
                          axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                        gridColor="rgba(255,255,255,0.07)")),
                )
                .properties(height=220, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(freq_hist, use_container_width=True)

        with col_rec:
            st.markdown("**Recency Distribution (days since last purchase)**")
            rec_hist = (
                alt.Chart(rfm_df)
                .mark_bar(color="#60a5fa", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    alt.X("Recency:Q", bin=alt.Bin(maxbins=40), title="Recency (days)"),
                    alt.Y("count()", title="# Customers",
                          axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                        gridColor="rgba(255,255,255,0.07)")),
                )
                .properties(height=220, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(rec_hist, use_container_width=True)

        # Cluster profile table
        st.markdown("---")
        st.markdown("**📋 Cluster Profile Table**")
        if "KMeans_Cluster" in rfm_df.columns:
            cluster_prof = (
                rfm_df.groupby("KMeans_Cluster")[["Recency", "Frequency", "Monetary"]]
                .mean()
                .reset_index()
                .rename(columns={
                    "KMeans_Cluster": "Cluster ID",
                    "Recency":        "Avg Recency (days)",
                    "Frequency":      "Avg Frequency",
                    "Monetary":       "Avg Monetary (£)",
                })
            )
            # Add segment name
            cluster_map_local = load_cluster_map()
            cluster_prof["Segment"] = cluster_prof["Cluster ID"].map(
                lambda cid: cluster_map_local.get(int(cid), "—")
            )
            st.dataframe(
                cluster_prof[["Cluster ID", "Segment", "Avg Recency (days)", "Avg Frequency", "Avg Monetary (£)"]]
                .style.format({
                    "Avg Recency (days)": "{:.1f}",
                    "Avg Frequency":      "{:.1f}",
                    "Avg Monetary (£)":   "£{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # Segment description cards
        st.markdown("<br>**🗂 Segment Descriptions**", unsafe_allow_html=True)
        seg_names = list(SEGMENT_DEFINITIONS.keys())
        cols = st.columns(len(seg_names))

        for idx, seg_name in enumerate(seg_names):
            info  = SEGMENT_DEFINITIONS[seg_name]
            color = info["color"]
            icon  = info["icon"]
            desc  = info["description"]
            strat = info["strategy"]
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

            with st.expander(f"{info['icon']}  {seg_name}  ·  {count:,} customers", expanded=False):
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
        st.markdown(
            '<div class="section-sub">Enter RFM values — the model predicts the customer segment instantly</div>',
            unsafe_allow_html=True,
        )

        col_form, col_result = st.columns(2, gap="large")

        with col_form:
            st.markdown(
                '<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.12);'
                'border-radius:16px;padding:24px;">'
                '<div style="font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:18px;">🧾 Customer RFM Profile</div>'
                '</div>', unsafe_allow_html=True)

            recency   = st.slider("📅 Recency (days since last purchase)", 1, 365, 30,
                                  help="Lower = more recent = better")
            frequency = st.slider("🔁 Frequency (number of orders)", 1, 200, 15,
                                  help="Higher = buys more often = better")
            monetary  = st.number_input("💷 Monetary (total spend in £)", min_value=1.0,
                                        max_value=50000.0, step=50.0, value=500.0,
                                        help="Total amount spent across all orders")

            st.markdown(
                f'<div style="background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.3);'
                f'border-radius:12px;padding:14px;margin-top:8px;">'
                f'<span style="color:#94a3b8;font-size:0.82rem;">Quick rule preview: </span>'
                f'<span style="color:#a78bfa;font-weight:600;font-size:0.86rem;">'
                f'{get_segment_name_local(recency, frequency, monetary)}'
                f'</span></div>',
                unsafe_allow_html=True,
            )

            predict_clicked = st.button("🔮 Predict Segment", type="primary",
                                        use_container_width=True)

        with col_result:
            if predict_clicked:
                with st.spinner("Running model…"):
                    try:
                        resp = requests.post(
                            f"{API_URL}/predict",
                            json={
                                "Recency":   recency,
                                "Frequency": frequency,
                                "Monetary":  monetary,
                            },
                            timeout=15,
                        )

                        if resp.ok:
                            result     = resp.json()
                            cluster_id = result.get("Cluster", "—")
                            seg_name   = result.get("segment_name",
                                                    get_segment_name_local(recency, frequency, monetary))
                            used_api   = True
                        else:
                            st.warning(f"⚠️ API returned {resp.status_code} — using local result.")
                            cluster_id = "—"
                            seg_name   = get_segment_name_local(recency, frequency, monetary)
                            used_api   = False

                    except requests.exceptions.RequestException:
                        st.warning("⚠️ FastAPI service unreachable — using local cluster-based fallback.")
                        _model, _scaler = load_local_model()
                        _cluster_map    = load_cluster_map()
                        if _model is not None:
                            _X         = np.array([[recency, frequency, monetary]])
                            cluster_id = int(_model.predict(_scaler.transform(_X))[0])
                            seg_name   = cluster_to_segment(cluster_id, _cluster_map,
                                                            recency, frequency, monetary)
                        else:
                            cluster_id = "—"
                            seg_name   = get_segment_name_local(recency, frequency, monetary)
                        used_api = False

                # ── Render result card ────────────────────────────
                info  = SEGMENT_DEFINITIONS.get(seg_name, SEGMENT_DEFINITIONS["Low Value"])
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
                    f'<div class="info-pill">📅 Recency {recency}d</div>'
                    f'<div class="info-pill">🔁 Frequency {frequency}</div>'
                    f'<div class="info-pill">💷 £{monetary:,.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    '<div style="background:rgba(255,255,255,0.04);border:2px dashed rgba(255,255,255,0.15);'
                    'border-radius:16px;padding:60px 30px;text-align:center;margin-top:10px;">'
                    '<div style="font-size:3rem;margin-bottom:12px;">🔮</div>'
                    '<div style="font-size:1rem;font-weight:600;color:#94a3b8;">'
                    'Enter the customer\'s RFM values and click<br>'
                    '<strong style="color:#a78bfa;">Predict Segment</strong></div>'
                    '<div style="font-size:0.82rem;color:#475569;margin-top:10px;">'
                    'Recency · Frequency · Monetary → segment in seconds'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

    # ── Footer ─────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:30px 0 10px 0;color:#475569;font-size:0.78rem;">'
        'RFM Customer Segmentation Intelligence &nbsp;&middot;&nbsp; K-Means Clustering'
        ' &nbsp;&middot;&nbsp; Built with Streamlit &amp; FastAPI'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()