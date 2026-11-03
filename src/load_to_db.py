"""Load data/train.csv into a local SQLite database so we can analyse it with SQL."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "train.csv"
DB_PATH = ROOT / "data" / "sales.db"

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df["date"] = df["date"].dt.strftime("%Y-%m-%d")

con = sqlite3.connect(DB_PATH)
df.to_sql("sales", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS idx_sales_store_item_date ON sales (store, item, date)")
con.commit()

summary = pd.read_sql(
    """
    SELECT COUNT(*)              AS n_rows,
           MIN(date)             AS first_date,
           MAX(date)             AS last_date,
           COUNT(DISTINCT store) AS stores,
           COUNT(DISTINCT item)  AS items
    FROM sales
    """,
    con,
)
con.close()

print(summary.to_string(index=False))
print(f"\nDatabase written to {DB_PATH}")