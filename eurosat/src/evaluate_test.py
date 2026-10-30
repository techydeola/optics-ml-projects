"""Test-set evaluation. Run ONCE, after the plan is fixed.

Evaluates the 18 protocol B CNNs (6 band configurations x 3 seeds, final-epoch weights), the random
forest (3 seeds) and the logistic regression. Everything trained on the train split only.
The script refuses to run if its output already exists. To reproduce from scratch, delete
eurosat/results/test_eval_final.csv first.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import BAND_SETS, FEATURES, RES, EuroSATSequence, class_names, load_trained

OUT = RES / "test_eval_final.csv"
if OUT.exists():
    raise SystemExit(f"{OUT} already exists. The test set is evaluated once; not rerunning.")

names = class_names()
rows, cms = [], {}


def record(model, seed, y_true, y_pred):
    rows.append(dict(model=model, seed=seed,
                     acc=accuracy_score(y_true, y_pred),
                     macro_f1=f1_score(y_true, y_pred, average="macro"),
                     **dict(zip(names, f1_score(y_true, y_pred, average=None)))))
    cms[f"{model}_seed{seed}"] = confusion_matrix(y_true, y_pred)


# CNNs, protocol B, final-epoch weights
for cfg in ["all13", "rgb", "vis_nir", "vis_swir", "no_nir", "no_swir"]:
    ds = EuroSATSequence("test", BAND_SETS[cfg], 256, augment=False, workers=1, max_queue_size=4)
    for seed in (42, 43, 44):
        pred = load_trained(cfg, seed).predict(ds, verbose=0).argmax(1)
        record(f"cnn_{cfg}", seed, ds.labels, pred)

# baselines on band mean + std features, trained on the train split only
d = np.load(FEATURES)
X, y, split = d["X"], d["y"], d["split"]
tr, te = split == "train", split == "test"
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)).fit(X[tr], y[tr])
record("baseline_logreg", 42, y[te], lr.predict(X[te]))
for seed in (42, 43, 44):
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
    record("baseline_rf", seed, y[te], rf.predict(X[te]))

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
np.savez(RES / "test_confusions.npz", **cms)

g = df.groupby("model", sort=False)
print(pd.DataFrame({"n": g.size(), "mean_acc": g["acc"].mean(), "std_acc": g["acc"].std(),
                    "min_acc": g["acc"].min(), "max_acc": g["acc"].max(),
                    "mean_f1": g["macro_f1"].mean()}).round(4).to_string())
print("saved:", OUT)
