import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt

csv_path = pathlib.Path("data/predictive-maintenance-dataset.csv")
df = pd.read_csv(csv_path)

cols = df.columns.to_list()
sensor_cols = [c for c in cols if c != "ID"]

rows = []
for col in sensor_cols:
    s = df[col]
    q = s.quantile([.01, .05, .25, .5, .75, .95, .99])
    rows.append({
        "column": col, "dtype": str(s.dtype),
        "count": int(s.notna().sum()),
        "missing_%": round(100*(1 - s.notna().mean()),3),
        "zeros_%": round(100*(s.eq(0).mean()),3),
        "mean": s.mean(), "std": s.std(ddof=1),
        "min": s.min(), "p01": q.loc[.01], "p05": q.loc[.05],
        "p25": q.loc[.25], "p50": q.loc[.5], "p75": q.loc[.75],
        "p95": q.loc[.95], "p99": q.loc[.99], "max": s.max(),
        "skew": s.skew(), "kurtosis": s.kurtosis()
    })
summary = pd.DataFrame(rows).sort_values("column")

#correlations
pearson = df[sensor_cols].corr("pearson")
spearman = df[sensor_cols].corr("spearman")

#outputs
out_dir = pathlib.Path("eda_outputs"); out_dir.mkdir(parents=True, exist_ok=True)
summary.to_csv(out_dir/"summary.csv", index=False)
pearson.to_csv(out_dir/"corr_pearson.csv")
spearman.to_csv(out_dir/"corr_spearman.csv")

#histograms
img_dir = out_dir/"images"; img_dir.mkdir(exist_ok=True)
for col in sensor_cols:
    plt.figure()
    plt.hist(df[col].dropna().values, bins=50)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(img_dir/f"hist_{col}.png"); plt.close()

#heatmaps without seaborn
def save_heatmap(mat, title, path):
    plt.figure(figsize=(8,6))
    plt.imshow(mat.values, aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=90)
    plt.yticks(range(len(mat.index)), mat.index)
    plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

save_heatmap(pearson, "Pearson Correlation Heatmap", img_dir/"corr_pearson.png")
save_heatmap(spearman, "Spearman Correlation Heatmap", img_dir/"corr_spearman.png")

print("Saved:")
print(" - eda_outputs/summary.csv")
print(" - eda_outputs/corr_pearson.csv")
print(" - eda_outputs/corr_spearman.csv")
print(" - eda_outputs/images/*.png")
