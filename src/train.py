"""Train, validate and compare models, then export everything the API and dashboard need.

Run from the project root:   python src/train.py
"""
import json
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from features import FEATURES, add_features, build_forecast_frame, predict_units

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

HOLDOUT_DAYS = 90   # length of the validation window AND of the test window
SERVICE_Z = 1.65    # z-score for ~95% service level (used for safety stock)

CONFIGS = {
    "lgbm_small": dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
    "lgbm_medium": dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
    "lgbm_large": dict(n_estimators=900, learning_rate=0.02, num_leaves=127, min_child_samples=100,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
}


def smape(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    denom = np.abs(y) + np.abs(p)
    safe = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(p - y) / safe)) * 100)


def evaluate(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return {
        "smape": smape(y, p),
        "mae": float(np.mean(np.abs(y - p))),
        "wape": float(np.abs(y - p).sum() / y.sum() * 100),
    }


def fit(X, y, params):
    model = LGBMRegressor(objective="regression", random_state=42, verbose=-1, **params)
    model.fit(X, np.log1p(y))   # log target: stabilises variance, works well for SMAPE
    return model


def main():
    t0 = time.time()
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Loaded {len(raw):,} rows  |  {raw['date'].min().date()} -> {raw['date'].max().date()}")

    df = add_features(raw).dropna(subset=FEATURES).reset_index(drop=True)

    # ---- time-based split: train | validation (90d) | test (last 90d) ----
    last_date = df["date"].max()
    test_start = last_date - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    val_start = test_start - pd.Timedelta(days=HOLDOUT_DAYS)
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    print(f"train {len(train):,} | validation {len(val):,} | test {len(test):,}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")   # stored in the project root
    mlflow.set_experiment("demand-forecasting")

    # ---- 1) compare configurations on the VALIDATION window ----
    val_scores = {}
    for name, params in CONFIGS.items():
        with mlflow.start_run(run_name=name):
            t1 = time.time()
            model = fit(train[FEATURES], train["sales"], params)
            scores = evaluate(val["sales"], predict_units(model, val[FEATURES]))
            mlflow.log_params(params)
            mlflow.log_param("target_transform", "log1p")
            mlflow.log_param("n_features", len(FEATURES))
            mlflow.log_metrics({f"val_{k}": v for k, v in scores.items()})
            mlflow.log_metric("fit_seconds", time.time() - t1)
            val_scores[name] = scores
            print(f"  {name:12s} validation SMAPE = {scores['smape']:.2f}%")

    best = min(val_scores, key=lambda k: val_scores[k]["smape"])
    print(f"Best configuration: {best}")

    # ---- 2) final honest evaluation on the TEST window ----
    with mlflow.start_run(run_name=f"final_{best}"):
        mlflow.log_params(CONFIGS[best])
        mlflow.log_param("config_name", best)

        trainval = pd.concat([train, val])
        model = fit(trainval[FEATURES], trainval["sales"], CONFIGS[best])
        test_pred = predict_units(model, test[FEATURES])

        results = {
            "naive_lag7": evaluate(test["sales"], test["lag_7"]),
            "avg_of_lags": evaluate(test["sales"], test[["lag_7", "lag_14", "lag_28"]].mean(axis=1)),
            "lightgbm": evaluate(test["sales"], test_pred),
        }
        for model_name, scores in results.items():
            for metric, value in scores.items():
                mlflow.log_metric(f"test_{model_name}_{metric}", value)

        naive_smape = results["naive_lag7"]["smape"]
        model_smape = results["lightgbm"]["smape"]
        improvement = (naive_smape - model_smape) / naive_smape * 100

        # per-row test predictions (for the dashboard)
        test_out = test[["date", "store", "item", "sales"]].copy()
        test_out["pred_lgbm"] = test_pred.round(2)
        test_out["pred_naive"] = test["lag_7"].values
        test_out.to_csv(OUT / "test_predictions.csv", index=False)

        # feature importance
        imp = pd.DataFrame({
            "feature": FEATURES,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("importance_gain", ascending=False)
        imp["importance_pct"] = (100 * imp["importance_gain"] / imp["importance_gain"].sum()).round(2)
        imp.to_csv(OUT / "feature_importance.csv", index=False)

        summary = {
            "best_config": best,
            "validation_smape": {k: round(v["smape"], 3) for k, v in val_scores.items()},
            "test": {m: {k: round(v, 3) for k, v in s.items()} for m, s in results.items()},
            "smape_improvement_vs_naive_pct": round(improvement, 2),
            "data_start": str(raw["date"].min().date()),
            "data_end": str(raw["date"].max().date()),
            "test_start": str(test_start.date()),
            "n_rows_total": int(len(raw)),
            "n_series": int(raw[["store", "item"]].drop_duplicates().shape[0]),
        }
        (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))

        # ---- 3) refit on ALL data -> production model + next-7-day forecast ----
        final_model = fit(df[FEATURES], df["sales"], CONFIGS[best])
        joblib.dump(final_model, MODELS / "lgbm_model.pkl")

        frame = build_forecast_frame(raw)
        frame["forecast"] = predict_units(final_model, frame[FEATURES]).round(2)
        frame[["date", "store", "item", "forecast"]].to_csv(OUT / "forecast_next7.csv", index=False)

        # ---- 4) inventory plan: 7-day forecast + safety stock from recent forecast error ----
        err = (
            test_out.assign(err=test_out["sales"] - test_out["pred_lgbm"])
            .groupby(["store", "item"])["err"].std()
            .rename("daily_error_std").reset_index()
        )
        plan = (
            frame.groupby(["store", "item"])["forecast"].sum().rename("forecast_7d").reset_index()
            .merge(err, on=["store", "item"])
        )
        plan["safety_stock"] = (SERVICE_Z * plan["daily_error_std"] * np.sqrt(7)).round(1)
        plan["recommended_stock"] = np.ceil(plan["forecast_7d"] + plan["safety_stock"]).astype(int)
        plan["forecast_7d"] = plan["forecast_7d"].round(1)
        plan["daily_error_std"] = plan["daily_error_std"].round(2)
        plan.to_csv(OUT / "inventory_plan.csv", index=False)

        for f in ("metrics.json", "test_predictions.csv", "feature_importance.csv",
                  "forecast_next7.csv", "inventory_plan.csv"):
            mlflow.log_artifact(str(OUT / f))

    print("\n===== RESULTS (last 90 days, unseen) =====")
    print(pd.DataFrame(results).T.round(2).to_string())
    print(f"\nLightGBM reduces SMAPE by {improvement:.1f}% vs the naive baseline")
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes. Files saved in outputs/ and models/")


if __name__ == "__main__":
    main()