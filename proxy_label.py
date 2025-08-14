import pandas as pd, numpy as np
from pathlib import Path

# Rolling z-score helper
def rz(s, w):
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std()
    return (s - m) / sd.replace(0, np.nan)

# Load
csv_path = Path("data/predictive-maintenance-dataset.csv")
df = pd.read_csv(csv_path)

# Params tuned for ~2% positives
window = 50
threshold = 1.8
run_len = 8

# Z-scores
df['vib_z'] = rz(df['vibration'], window)
df['rev_z'] = rz(df['revolutions'], window)

# Condition
cond = (df['vib_z'].abs() > threshold) | (df['rev_z'].abs() > threshold)
cond = cond.fillna(False)

# Group consecutive True blocks
groups = cond.ne(cond.shift(fill_value=False)).cumsum()
counts = cond.groupby(groups).transform('size')

# Final proxy label
df['proxy_anomaly'] = ((cond) & (counts >= run_len)).astype(int)

print(df['proxy_anomaly'].value_counts(normalize=True))
df.to_csv("data_with_proxy_label.csv", index=False)
print("✅ Proxy label saved → data_with_proxy_label.csv")
