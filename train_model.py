"""
MovieIQ - Stage 4: Predictive Modeling (Random Forest)
Trains a RandomForestClassifier on the cleaned dataset and saves:
  - model.pkl              (trained model, for the Streamlit app)
  - assets/confusion_matrix.png
  - assets/feature_importance.png
  - assets/model_metrics.json

Run:  python train_model.py   (run analysis.py first to produce movies_clean.csv)
"""

import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

df = pd.read_csv("movies_clean.csv")

# 4.1 Feature selection.
# We exclude `revenue` (it is literally used to define the target -> leakage),
# and `title` / `genres` / `genre_list` (free text / not model-ready).
# `primary_genre` is kept but one-hot encoded so genre still informs the model.
FEATURES_NUMERIC = ["budget", "popularity", "runtime", "vote_average"]
FEATURES_CATEGORICAL = ["primary_genre"]
TARGET = "success"

X = pd.get_dummies(df[FEATURES_NUMERIC + FEATURES_CATEGORICAL], columns=FEATURES_CATEGORICAL)
y = df[TARGET]
feature_columns = X.columns.tolist()  # save for the app to align input rows

# 4.2 Train/test split - 80/20 is a standard default that leaves enough data
# to train on while still giving a reasonably sized, untouched test set to
# check the model isn't just memorizing the training rows.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4.3 Train Random Forest
# A random forest builds many decision trees, each on a random subset of rows
# and features, then lets them vote; the majority vote becomes the prediction.
# Averaging many "opinions" like this reduces overfitting compared to a
# single decision tree.
model = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# 4.4 Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"[Stage 4.4] Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")
print("Confusion matrix:\n", cm)

plt.figure(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Success"], yticklabels=["Fail", "Success"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=140)
plt.close()

# 4.5 Feature importance
importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
plt.figure(figsize=(7, 5))
importances.head(12).plot(kind="barh", color="#9b59b6")
plt.gca().invert_yaxis()
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("assets/feature_importance.png", dpi=140)
plt.close()
print("[Stage 4.5] Top features:\n", importances.head(8))

# Save model + metadata for the Streamlit app
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "feature_columns": feature_columns,
        "numeric_features": FEATURES_NUMERIC,
        "genre_options": sorted(df["primary_genre"].unique().tolist()),
    }, f)

metrics = {
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "confusion_matrix": cm.tolist(),
    "top_features": importances.head(8).round(4).to_dict(),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
}
with open("assets/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved model.pkl, assets/confusion_matrix.png, assets/feature_importance.png, assets/model_metrics.json")
