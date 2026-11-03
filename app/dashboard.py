"""Streamlit dashboard. Reads only the files in outputs/, so it can be deployed anywhere.

Run from the project root:   streamlit run app/dashboard.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📈", layout="wide")


@st.cache_data
def load(name, dates=None):
    return pd.read_csv(OUT / name, parse_dates=dates)


def smape(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    denom = np.abs(y) + np.abs(p)
    safe = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(p - y) / safe)) * 100)


metrics = json.loads((OUT / "metrics.json").read_text())
test = load("test_predictions.csv", ["date"])
fcst = load("forecast_next7.csv", ["date"])
plan = load("inventory_plan.csv")
imp = load("feature_importance.csv")

st.title("📈 Retail Demand Forecasting Platform")
st.caption(
    f"7-day-ahead sales forecasts for {metrics['n_series']} store-item series "
    f"({metrics['n_rows_total']:,} daily records, {metrics['data_start']} to {metrics['data_end']})."
)

m = metrics["test"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Model SMAPE (unseen test)", f"{m['lightgbm']['smape']:.2f}%")
k2.metric("Naive baseline SMAPE", f"{m['naive_lag7']['smape']:.2f}%")
k3.metric("Error reduction vs naive", f"{metrics['smape_improvement_vs_naive_pct']:.1f}%")
k4.metric("Series forecast", f"{len(plan)}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Forecast vs actual", "Next 7 days & inventory", "Business insights (SQL)", "Model details"]
)

# ---------------------------------------------------------------- tab 1
with tab1:
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", sorted(test["store"].unique()), key="store1")
    item = c2.selectbox("Item", sorted(test["item"].unique()), key="item1")
    d = test[(test["store"] == store) & (test["item"] == item)].sort_values("date")

    fig = go.Figure()
    fig.add_scatter(x=d["date"], y=d["sales"], mode="lines", name="Actual")
    fig.add_scatter(x=d["date"], y=d["pred_lgbm"], mode="lines", name="LightGBM forecast")
    fig.add_scatter(x=d["date"], y=d["pred_naive"], mode="lines", name="Naive (same day last week)",
                    line=dict(dash="dot"))
    fig.update_layout(title=f"Store {store}, item {item}: last 90 days (unseen by the model)",
                      xaxis_title="Date", yaxis_title="Units sold", height=450)
    st.plotly_chart(fig)

    s_model = smape(d["sales"], d["pred_lgbm"])
    s_naive = smape(d["sales"], d["pred_naive"])
    st.info(f"For this series: model SMAPE **{s_model:.2f}%** vs naive **{s_naive:.2f}%**.")

# ---------------------------------------------------------------- tab 2
with tab2:
    c1, c2 = st.columns(2)
    store2 = c1.selectbox("Store", sorted(plan["store"].unique()), key="store2")
    item2 = c2.selectbox("Item", sorted(plan["item"].unique()), key="item2")

    hist = test[(test["store"] == store2) & (test["item"] == item2)].sort_values("date").tail(28)
    f = fcst[(fcst["store"] == store2) & (fcst["item"] == item2)].sort_values("date")

    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["sales"], mode="lines", name="Actual (last 28 days)")
    fig.add_scatter(x=f["date"], y=f["forecast"], mode="lines+markers", name="Forecast (next 7 days)",
                    line=dict(dash="dash"))
    fig.update_layout(title=f"Store {store2}, item {item2}: recent sales and next-week forecast",
                      xaxis_title="Date", yaxis_title="Units", height=420)
    st.plotly_chart(fig)

    st.subheader(f"Inventory plan for store {store2} (next 7 days)")
    view = plan[plan["store"] == store2].sort_values("recommended_stock", ascending=False)
    st.dataframe(view, hide_index=True)
    st.download_button("Download this plan (CSV)", view.to_csv(index=False),
                       file_name=f"inventory_plan_store_{store2}.csv", mime="text/csv")
    st.caption("Recommended stock = 7-day forecast + safety stock (about 95% service level, "
               "calculated from each series' recent forecast error).")

# ---------------------------------------------------------------- tab 3
with tab3:
    monthly = load("sql_monthly.csv")
    stores = load("sql_stores.csv")
    dow = load("sql_dow.csv")
    top = load("sql_top_items.csv")
    yearly = load("sql_yearly.csv")
    season = load("sql_seasonality.csv")

    st.plotly_chart(px.line(monthly, x="month", y="units", title="Monthly units sold"))

    a, b = st.columns(2)
    a.plotly_chart(px.bar(stores, x=stores["store"].astype(str), y="units", title="Units sold by store",
                          labels={"x": "Store"}))
    b.plotly_chart(px.bar(dow, x="day_of_week", y="avg_units", title="Average daily units by weekday"))

    c, d_ = st.columns(2)
    c.plotly_chart(px.bar(season, x="month_num", y="seasonality_index",
                          title="Seasonality index by month (1.0 = average)"))
    d_.plotly_chart(px.bar(top, x=top["item"].astype(str), y="units", title="Top 10 items by units",
                           labels={"x": "Item"}))

    st.subheader("Year-over-year growth")
    st.dataframe(yearly, hide_index=True)

# ---------------------------------------------------------------- tab 4
with tab4:
    st.subheader("Model comparison on the last 90 days (not used for training or tuning)")
    table = pd.DataFrame(metrics["test"]).T.reset_index().rename(columns={"index": "model"})
    st.dataframe(table, hide_index=True)
    st.caption("SMAPE and WAPE: lower is better. Naive = same weekday last week.")

    st.subheader("Feature importance (LightGBM gain)")
    st.plotly_chart(px.bar(imp.sort_values("importance_pct"), x="importance_pct", y="feature",
                           orientation="h", labels={"importance_pct": "Importance (%)"}))

    shap_path = OUT / "shap_summary.png"
    if shap_path.exists():
        st.subheader("SHAP summary")
        st.image(str(shap_path), caption="SHAP values (log-sales scale): how each feature pushes predictions.")

    st.markdown(
        """
**Method.** Time-based split (train, 90-day validation, 90-day test) to avoid leakage; lag, rolling and
calendar features that only use data at least 7 days old; log-transformed target; three LightGBM
configurations compared on validation, the best one evaluated once on the test window and then refit on all
data for the production forecast.
"""
    )