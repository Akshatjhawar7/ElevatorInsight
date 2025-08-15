import pandas as pd
import pickle
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

df = pd.read_csv("data/data_with_engineered_features.csv")

X = df.drop(columns=["proxy_anomaly"])
y = df["proxy_anomaly"]

feature_names = list(X.columns)
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=4)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

best_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    colsample_bytree=1,
    learning_rate=0.1,
    max_depth=3,
    n_estimators=500,
    subsample=1
)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
metrics = {
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "accuracy": accuracy_score(y_test, y_pred)
}

print("\n Classification Report: \n", classification_report(y_test, y_pred, digits=4))
print("Metrics:", metrics)

with open("model_v0.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Model and metrics saved (F1 = {metrics['f1']:.4f})")
