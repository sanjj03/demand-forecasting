"""FastAPI service: GET /forecast?store=1&item=1 returns the next 7 days of predicted sales.

Run from the project root:   uvicorn app.api:app --reload
Then open:                   http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURES, build_forecast_frame, predict_units  # noqa: E402

app = FastAPI(
    title="Retail Demand Forecast API",
    version="1.0.0",
    description="7-day-ahead sales forecasts per store and item, served from a LightGBM model.",
)

model = joblib.load(ROOT / "models" / "lgbm_model.pkl")
history = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "history_ends": str(history["date"].max().date()),
        "series": int(history[["store", "item"]].drop_duplicates().shape[0]),
    }


@app.get("/forecast")
def forecast(store: int, item: int):
    series = history[(history["store"] == store) & (history["item"] == item)]
    if series.empty:
        raise HTTPException(status_code=404, detail=f"No history for store={store}, item={item}")

    frame = build_forecast_frame(series)
    frame["forecast"] = predict_units(model, frame[FEATURES]).round(1)

    return {
        "store": store,
        "item": item,
        "horizon_days": 7,
        "forecast": [
            {"date": str(d.date()), "units": float(u)}
            for d, u in zip(frame["date"], frame["forecast"])
        ],
        "total_units_7d": round(float(frame["forecast"].sum()), 1),
    }