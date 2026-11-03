"""Run business-insight SQL queries (CTEs + window functions) and save results to outputs/."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "sales.db"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

QUERIES = {
    # 1. Monthly units sold
    "sql_monthly": """
        SELECT strftime('%Y-%m', date) AS month,
               SUM(sales)              AS units
        FROM sales
        GROUP BY month
        ORDER BY month
    """,
    # 2. Year-over-year growth (CTE + LAG window function)
    "sql_yearly": """
        WITH yearly AS (
            SELECT strftime('%Y', date) AS year, SUM(sales) AS units
            FROM sales
            GROUP BY year
        )
        SELECT year,
               units,
               ROUND(100.0 * (units - LAG(units) OVER (ORDER BY year))
                     / LAG(units) OVER (ORDER BY year), 2) AS yoy_growth_pct
        FROM yearly
        ORDER BY year
    """,
    # 3. Store ranking (RANK window function)
    "sql_stores": """
        SELECT store,
               SUM(sales)                          AS units,
               ROUND(AVG(sales), 2)                AS avg_daily_units,
               RANK() OVER (ORDER BY SUM(sales) DESC) AS store_rank
        FROM sales
        GROUP BY store
        ORDER BY store_rank
    """,
    # 4. Top 10 items with share and cumulative share (Pareto view)
    "sql_top_items": """
        WITH item_totals AS (
            SELECT item, SUM(sales) AS units
            FROM sales
            GROUP BY item
        )
        SELECT item,
               units,
               ROUND(100.0 * units / SUM(units) OVER (), 2)                         AS share_pct,
               ROUND(100.0 * SUM(units) OVER (ORDER BY units DESC) / SUM(units) OVER (), 2) AS cum_share_pct
        FROM item_totals
        ORDER BY units DESC
        LIMIT 10
    """,
    # 5. Weekday pattern
    "sql_dow": """
        SELECT CAST(strftime('%w', date) AS INTEGER) AS dow_num,
               CASE strftime('%w', date)
                    WHEN '0' THEN 'Sun' WHEN '1' THEN 'Mon' WHEN '2' THEN 'Tue'
                    WHEN '3' THEN 'Wed' WHEN '4' THEN 'Thu' WHEN '5' THEN 'Fri'
                    ELSE 'Sat' END                    AS day_of_week,
               ROUND(AVG(sales), 2)                   AS avg_units
        FROM sales
        GROUP BY dow_num, day_of_week
        ORDER BY dow_num
    """,
    # 6. Seasonality index by calendar month (1.0 = average month)
    "sql_seasonality": """
        WITH monthly AS (
            SELECT CAST(strftime('%m', date) AS INTEGER) AS month_num,
                   AVG(sales)                            AS avg_units
            FROM sales
            GROUP BY month_num
        )
        SELECT month_num,
               ROUND(avg_units, 2) AS avg_units,
               ROUND(avg_units / (SELECT AVG(sales) FROM sales), 3) AS seasonality_index
        FROM monthly
        ORDER BY month_num
    """,
}

con = sqlite3.connect(DB_PATH)
for name, query in QUERIES.items():
    result = pd.read_sql(query, con)
    result.to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n=== {name} ===")
    print(result.head(12).to_string(index=False))
con.close()

print(f"\nSaved {len(QUERIES)} result files to {OUT}")