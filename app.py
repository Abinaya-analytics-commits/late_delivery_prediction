import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Late Delivery Risk Prediction Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📦 Global Supply Chain Late Delivery Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Early Warning & Operational Decision Support System | Unified Mentors Project</div>', unsafe_allow_html=True)

# Locate dataset
DATA_PATH = Path(__file__).parent / "UM_scored_test_orders_tuned.csv"
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).parent / "data" / "UM_scored_test_orders_tuned.csv"

@st.cache_data
def load_data(path):
    if not path.exists():
        return None
    return pd.read_csv(path)

df = load_data(DATA_PATH)

if df is None:
    st.error(f"Scored dataset not found at `{DATA_PATH}`. Please run `execute_technical_pipeline.py` first to generate it.")
    st.stop()

# Ensure required columns
required_cols = [
    "Actual_Late_Risk",
    "Late_Delivery_Probability",
    "Predicted_Late",
    "Risk_Category",
    "Market",
    "Order Region",
    "Shipping Mode"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in dataset: {missing}")
    st.stop()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/delivery.png", width=70)
st.sidebar.title("Operational Controls")
st.sidebar.markdown("Filter incoming shipments and calibrate risk threshold:")

# Threshold Slider
threshold = st.sidebar.slider(
    "🚨 Late-Risk Alert Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
    help="Default tuned threshold is 0.35 to maximize late-delivery recall (88.8%). Higher values reduce false alarms but miss delayed shipments."
)

# Dynamic filter options
markets = sorted(df["Market"].dropna().unique().tolist())
selected_markets = st.sidebar.multiselect("Market", markets, default=markets)

regions = sorted(df[df["Market"].isin(selected_markets)]["Order Region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("Order Region", regions, default=regions)

modes = sorted(df["Shipping Mode"].dropna().unique().tolist())
selected_modes = st.sidebar.multiselect("Shipping Mode", modes, default=modes)

# Apply Filters
filtered = df[
    (df["Market"].isin(selected_markets)) &
    (df["Order Region"].isin(selected_regions)) &
    (df["Shipping Mode"].isin(selected_modes))
].copy()

# Recalculate alerts based on live threshold
filtered["Live_Alert"] = filtered["Late_Delivery_Probability"] >= threshold

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

total_orders = len(filtered)
flagged_orders = filtered["Live_Alert"].sum()
actual_late = filtered["Actual_Late_Risk"].sum()
true_positives = ((filtered["Live_Alert"]) & (filtered["Actual_Late_Risk"] == 1)).sum()
recall = (true_positives / actual_late * 100) if actual_late > 0 else 0
high_risk_tier = (filtered["Risk_Category"] == "High").sum()

col1.metric("📦 Filtered Orders", f"{total_orders:,}")
col2.metric("🚨 Orders Flagged", f"{flagged_orders:,}", f"{(flagged_orders/total_orders*100):.1f}% of volume" if total_orders > 0 else "0%")
col3.metric("🎯 Late Delivery Recall", f"{recall:.1f}%", f"{true_positives:,} caught")
col4.metric("🔴 High-Risk Orders (>70%)", f"{high_risk_tier:,}")
col5.metric("⚡ Active Threshold", f"{threshold:.2f}")

st.markdown("---")

# Visualizations Row
vcol1, vcol2 = st.columns(2)

with vcol1:
    st.subheader("📊 Orders by Predicted Risk Category")
    risk_summary = filtered["Risk_Category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
    risk_summary.columns = ["Risk Category", "Orders"]
    
    fig_risk = px.bar(
        risk_summary,
        x="Risk Category",
        y="Orders",
        color="Risk Category",
        color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
        text="Orders"
    )
    fig_risk.update_traces(textposition="outside")
    fig_risk.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_risk, use_container_width=True)

with vcol2:
    st.subheader("🚚 Late Delivery Rate by Shipping Mode")
    shipping_stats = filtered.groupby("Shipping Mode").agg(
        Orders=("Actual_Late_Risk", "size"),
        Delayed_Rate=("Actual_Late_Risk", "mean"),
        Avg_Predicted_Risk=("Late_Delivery_Probability", "mean")
    ).reset_index()
    shipping_stats["Delayed_Rate_Pct"] = (shipping_stats["Delayed_Rate"] * 100).round(1)
    shipping_stats["Avg_Predicted_Pct"] = (shipping_stats["Avg_Predicted_Risk"] * 100).round(1)
    
    fig_ship = px.bar(
        shipping_stats,
        x="Shipping Mode",
        y=["Delayed_Rate_Pct", "Avg_Predicted_Pct"],
        barmode="group",
        labels={"value": "Rate (%)", "variable": "Metric"},
        color_discrete_map={"Delayed_Rate_Pct": "#EF4444", "Avg_Predicted_Pct": "#3B82F6"}
    )
    fig_ship.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ship, use_container_width=True)

# Regional Breakdown
st.subheader("🌍 Regional Late-Delivery Vulnerability")
region_stats = filtered.groupby("Order Region").agg(
    Orders=("Actual_Late_Risk", "size"),
    Actual_Late_Rate=("Actual_Late_Risk", "mean"),
    Avg_Predicted_Risk=("Late_Delivery_Probability", "mean")
).reset_index()
region_stats["Actual_Late_Rate"] = (region_stats["Actual_Late_Rate"] * 100).round(1)
region_stats["Avg_Predicted_Risk"] = (region_stats["Avg_Predicted_Risk"] * 100).round(1)
region_stats = region_stats.sort_values(by="Actual_Late_Rate", ascending=False)

fig_reg = px.bar(
    region_stats.head(10),
    x="Order Region",
    y="Actual_Late_Rate",
    color="Actual_Late_Rate",
    color_continuous_scale="Reds",
    labels={"Actual_Late_Rate": "Late Rate (%)"},
    title="Top 10 High-Risk Regions by Actual Delay Rate (%)"
)
fig_reg.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_reg, use_container_width=True)

# High-Risk Alert Queue
st.markdown("---")
st.subheader("🚨 Priority Dispatch Alert Queue")
st.caption(f"Shipments flagged with Late Delivery Probability ≥ {threshold:.2f}. Prioritize expediting or re-routing before carrier hand-off.")

queue = filtered[filtered["Live_Alert"]].sort_values(by="Late_Delivery_Probability", ascending=False)

display_cols = [
    col for col in [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Customer State",
        "Category Name",
        "Order Item Quantity",
        "Late_Delivery_Probability",
        "Risk_Category",
        "Actual_Late_Risk"
    ] if col in queue.columns
]

queue_table = queue[display_cols].copy()
queue_table["Late_Delivery_Probability"] = (queue_table["Late_Delivery_Probability"] * 100).round(1).astype(str) + "%"

st.dataframe(
    queue_table.head(250),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    label="📥 Download Full High-Risk Alert Queue (CSV)",
    data=queue.to_csv(index=False).encode("utf-8"),
    file_name=f"late_delivery_alert_queue_threshold_{threshold:.2f}.csv",
    mime="text/csv"
)

# Footer & Architectural Note
st.markdown("---")
st.info(
    "💡 **Model & Operational Architecture Note:** "
    "Predictions are powered by a Random Forest Classifier trained on 14 non-leaking supply chain attributes (ROC-AUC = 0.7595). "
    "The 0.35 threshold ensures high sensitivity (88.8% recall), catching delays proactively before customer SLA penalties occur."
)
