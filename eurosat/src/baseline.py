"""Baselines on per-band mean and std features: logistic regression and random forest.

Trained on the train split, scored on the validation split. The test split is not used here.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, RES, class_names

SEED = 42

d = np.load(FEATURES)
X, y, split = d["X"], d["y"], d["split"]
names = class_names()

tr, va = split == "train", split == "val"
print("train:", tr.sum(), "| val:", va.sum(), "| test (untouched):", (split == "test").sum())

models = {
    # the scaler is fit on the train split only
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
}

rows = []
for name, model in models.items():
    model.fit(X[tr], y[tr])
    pred = model.predict(X[va])
    acc = accuracy_score(y[va], pred)
    macro = f1_score(y[va], pred, average="macro")
    print(f"\n=== {name} (validation) ===")
    print(f"accuracy: {acc:.4f} | macro-F1: {macro:.4f}")
    print(classification_report(y[va], pred, target_names=names, digits=3))
    rows.append({"model": name, "val_accuracy": acc, "val_macro_f1": macro,
                 **{f"f1_{n}": v for n, v in zip(names, f1_score(y[va], pred, average=None))}})

out = RES / "baseline_val.csv"
pd.DataFrame(rows).round(4).to_csv(out, index=False)
print("saved:", out)
