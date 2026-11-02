"""Generate a reproducible, synthetic customer-spending dataset for the demo app."""
from pathlib import Path
import numpy as np
import pandas as pd


def make_dataset(rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(38, 12, rows).round().astype(int).clip(18, 70)
    gender = rng.choice(["Female", "Male", "Non-binary"], rows, p=[0.48, 0.49, 0.03])
    annual_income = np.exp(rng.normal(np.log(620_000), 0.62, rows)).round().clip(120_000, 4_500_000)
    purchase_frequency = rng.integers(1, 13, rows)
    avg_order_value = np.exp(rng.normal(np.log(2_700), 0.58, rows)).round().clip(350, 16_000)
    website_visits = rng.integers(2, 46, rows)
    loyalty_years = rng.uniform(0.1, 12, rows).round(1)
    discount_usage = rng.beta(2.1, 2.4, rows).round(2)
    total_purchases = (
        purchase_frequency * 12 * np.maximum(loyalty_years, 0.25)
        * rng.uniform(0.68, 1.28, rows)
    ).round().astype(int).clip(1)

    # Spending is generated from purchase behaviour with modest natural variation.
    behaviour_multiplier = rng.lognormal(mean=0.0, sigma=0.20, size=rows)
    loyalty_lift = 1 + (loyalty_years / 100)
    annual_spend = (
        purchase_frequency * avg_order_value * 12
        * behaviour_multiplier * loyalty_lift
    ).round(2)

    # Balanced demo labels use the sample's lower and upper tertiles.
    rank = pd.Series(annual_spend).rank(method="first")
    spending_category = pd.qcut(rank, q=3, labels=["Low", "Medium", "High"])

    return pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, rows + 1)],
        "age": age,
        "gender": gender,
        "annual_income": annual_income.astype(int),
        "purchase_frequency": purchase_frequency,
        "avg_order_value": avg_order_value.astype(int),
        "website_visits": website_visits,
        "loyalty_years": loyalty_years,
        "discount_usage": discount_usage,
        "total_purchases": total_purchases,
        "annual_spend": annual_spend,
        "spending_category": spending_category.astype(str),
    })


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "customer_spending_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = make_dataset()
    data.to_csv(out, index=False)
    print(f"Created {len(data):,} synthetic customer records: {out}")
    print("Spending categories:")
    print(data["spending_category"].value_counts().to_string())
