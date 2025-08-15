import json
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = Path("model_v0.pkl")
DATA_PATH = Path("data/data_with_engineered_features.csv")
FEATURE_PATH = Path("feature_names.json")
OUT_BEES = Path("shap_summary_beeswarm.png")
OUT_BAR = Path("shap_summary_bar.png")

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

feature_names = json.loads(FEATURE_PATH.read_text())

df = pd.read_csv(DATA_PATH)

X = df[feature_names].copy()
X = X.fillna(0)

sample_n = 2000
X_sample = X.sample(sample_n, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(OUT_BAR, dpi=180, bbox_inches="tight")
plt.close()

print(f"Saved: {OUT_BEES}, {OUT_BAR}")

probs = model.predict_proba(X)[:, 1]
i = int(np.argmax(probs))
x_row = X.iloc[[i]]

sv_row = explainer.shap_values(x_row)
base_value = explainer.expected_value

expl = shap.Explanation(
    values=sv_row[0],
    base_values=base_value,
    data=x_row.values[0],
    feature_names=X.columns.tolist()
)

shap.plots.waterfall(expl, max_display=15, show=False)
plt.tight_layout()
plt.savefig("shap_waterfall_top_positive.png", dpi=180, bbox_inches="tight")
plt.close()

print("Saved: shap_waterfall_top_positive.png")
