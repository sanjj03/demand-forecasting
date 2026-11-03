"""Create a SHAP summary plot (optional). Run after train.py:  python src/explain.py"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from features import FEATURES, add_features

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

raw = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
df = add_features(raw).dropna(subset=FEATURES)
recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=89)]
sample = recent.sample(3000, random_state=42)

model = joblib.load(ROOT / "models" / "lgbm_model.pkl")
shap_values = shap.TreeExplainer(model).shap_values(sample[FEATURES])

shap.summary_plot(shap_values, sample[FEATURES], show=False)
plt.tight_layout()
plt.savefig(OUT / "shap_summary.png", dpi=150, bbox_inches="tight")
print("Saved outputs/shap_summary.png")