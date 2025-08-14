# feature_engineering.py (minimal)

import pandas as pd
import numpy as np

IN_PATH  = "data_with_proxy_label.csv"
OUT_CSV  = "data/data_with_engineered_features.csv"

WIN_30S = 120     # 30 seconds  * 4 Hz
WIN_5M  = 1200    # 5 minutes   * 4 Hz

EXCLUDE = {"ID", "proxy_anomaly"}

def main():
    df = pd.read_csv(IN_PATH)

    # numeric sensors only (drop IDs/label)
    sensors = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE]

    # lag-1
    for c in sensors:
        df[f"{c}_lag1"] = df[c].diff(1)

    # rolling 5m
    for c in sensors:
        r = df[c].rolling(WIN_5M, min_periods=1)
        df[f"{c}_mean_5m"] = r.mean()
        df[f"{c}_std_5m"]  = r.std()
        df[f"{c}_min_5m"]  = r.min()
        df[f"{c}_max_5m"]  = r.max()

    # save
    df.to_csv(OUT_CSV, index=False)
    print(f"done")

if __name__ == "__main__":
    main()
