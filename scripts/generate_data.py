"""
generate_data.py
Generates a synthetic house-price dataset and saves it as:
  - data/reference/reference.csv   ← clean baseline (Evidently uses this)
  - data/incoming/batch_00.csv     ← first "live" batch (no drift yet)

Run once before training:
    python scripts/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng  = np.random.default_rng(SEED)

# ── Feature definitions ───────────────────────────────────────────────────
def make_houses(n: int, drift: bool = False) -> pd.DataFrame:
    """
    Features
    --------
    sqft          : total floor area
    bedrooms      : bedroom count
    bathrooms     : bathroom count
    age_years     : age of property
    garage        : has garage (0/1)
    neighbourhood : categorical zone (A / B / C)
    distance_cbd  : km to city centre

    Target
    ------
    price_usd     : sale price
    """
    sqft        = rng.normal(1800, 400, n).clip(600, 4000)
    bedrooms    = rng.integers(1, 6, n)
    bathrooms   = (bedrooms * rng.uniform(0.4, 0.8, n)).round().clip(1, 5)
    age_years   = rng.integers(0, 60, n)
    garage      = rng.choice([0, 1], n, p=[0.25, 0.75])
    neighbourhood = rng.choice(["A", "B", "C"], n, p=[0.3, 0.5, 0.2])
    distance_cbd  = rng.exponential(8, n).clip(1, 40)

    # Introduce drift in live batches — simulates a new suburb being added
    if drift:
        sqft        += rng.normal(300, 50, n)   # houses are bigger
        distance_cbd = rng.exponential(15, n).clip(5, 50)  # further out

    # Price formula (ground truth)
    zone_premium = {"A": 1.20, "B": 1.00, "C": 0.80}
    base = (
        sqft * 120
        + bedrooms * 8_000
        + bathrooms * 5_000
        - age_years * 500
        + garage * 15_000
        - distance_cbd * 2_000
    )
    price = np.array([base[i] * zone_premium[neighbourhood[i]] for i in range(n)])
    price += rng.normal(0, 10_000, n)   # noise
    price = price.clip(50_000, 2_000_000)

    return pd.DataFrame({
        "sqft":          sqft.round(1),
        "bedrooms":      bedrooms,
        "bathrooms":     bathrooms.astype(int),
        "age_years":     age_years,
        "garage":        garage,
        "neighbourhood": neighbourhood,
        "distance_cbd":  distance_cbd.round(2),
        "price_usd":     price.round(2),
    })


if __name__ == "__main__":
    ref_path = Path("data/reference/reference.csv")
    inc_path = Path("data/incoming/batch_00.csv")

    ref = make_houses(2000)            # large reference — no drift
    inc = make_houses(500)             # first live batch — same distribution

    ref.to_csv(ref_path, index=False)
    inc.to_csv(inc_path, index=False)

    print(f"✓ Reference dataset : {ref_path}  ({len(ref):,} rows)")
    print(f"✓ Incoming batch 00 : {inc_path}  ({len(inc):,} rows)")
    print(f"\nPrice range  : ${ref['price_usd'].min():,.0f} – ${ref['price_usd'].max():,.0f}")
    print(f"Mean price   : ${ref['price_usd'].mean():,.0f}")
