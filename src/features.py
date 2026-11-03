"""Shared feature engineering, used by training, the API and the explainability script."""
import numpy as np
import pandas as pd

HORIZON = 7  # we forecast 7 days ahead, so every lag / rolling feature uses data >= 7 days old

FEATURES = [
    "store", "item",
    "lag_7", "lag_14", "lag_28", "lag_364",
    "roll_mean_7", "roll_mean_28", "roll_std_28",
    "dow", "month", "dayofmonth", "dayofyear", "weekofyear", "is_weekend",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling and calendar features. Needs columns: date, store, item, sales."""
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    grp = df.groupby(["store", "item"])["sales"]

    for lag in (7, 14, 28, 364):          # 364 = same weekday last year
        df[f"lag_{lag}"] = grp.shift(lag)

    shifted = grp.shift(HORIZON)          # rolling stats end 7 days before the target day
    shifted_grp = shifted.groupby([df["store"], df["item"]])
    df["roll_mean_7"] = shifted_grp.transform(lambda s: s.rolling(7).mean())
    df["roll_mean_28"] = shifted_grp.transform(lambda s: s.rolling(28).mean())
    df["roll_std_28"] = shifted_grp.transform(lambda s: s.rolling(28).std())

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dayofmonth"] = df["date"].dt.day
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def build_forecast_frame(history: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Create feature rows for the `horizon` days after the last date in `history`."""
    last_date = history["date"].max()
    combos = history[["store", "item"]].drop_duplicates()
    future_dates = pd.DataFrame({"date": pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)})
    future = combos.merge(future_dates, how="cross")
    future["sales"] = np.nan

    full = pd.concat([history[["date", "store", "item", "sales"]], future], ignore_index=True)
    full = add_features(full)
    return full[full["date"] > last_date].reset_index(drop=True)


def predict_units(model, X) -> np.ndarray:
    """The model is trained on log1p(sales); convert predictions back to units."""
    return np.clip(np.expm1(model.predict(X)), 0, None)