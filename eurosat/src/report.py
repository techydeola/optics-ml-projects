"""Build the result tables and figures from saved results (run summaries, per-class CSV, test evaluation).

Writes eurosat/results/val_runs.csv, eurosat/results/results_tables.md and three figures.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, KRES, RES, class_names

names = class_names()
cfgs = ["rgb", "vis_nir", "vis_swir", "no_swir", "no_nir", "all13"]
lab = {"rgb": "RGB", "vis_nir": "Visible + NIR", "vis_swir": "Visible + SWIR",
       "no_swir": "All minus SWIR", "no_nir": "All minus NIR", "all13": "All 13 bands"}
seeds = (42, 43, 44)

# Which saved runs make up each protocol. For protocol A, RGB seeds 43 and 44 use the reruns with an
# 80-epoch cap ("long"); the other protocol A runs are the "main" runs (cap 40) and the band-group runs (cap 80).
stems = {
    "A": {"all13": [f"main_all13_seed{s}" for s in seeds],
          "rgb": ["main_rgb_seed42", "long_rgb_seed43", "long_rgb_seed44"],
          **{c: [f"band_{c}_seed{s}" for s in seeds] for c in ("vis_nir", "vis_swir", "no_nir", "no_swir")}},
    "B": {c: [f"fixed_{c}_seed{s}" for s in seeds] for c in cfgs},
}
rows = []
for proto, group in stems.items():
    for cfg, lst in group.items():
        for stem in lst:
            j = json.loads((KRES / f"{stem}_summary.json").read_text())
            rows.append(dict(protocol=proto, config=cfg, seed=j["seed"], epochs=j["epochs_run"],
                             val_acc=j["val_acc"], val_f1=j["val_macro_f1"]))
runs = pd.DataFrame(rows)
runs.to_csv(RES / "val_runs.csv", index=False)
print(runs.pivot_table(index="config", columns=["protocol", "seed"], values="val_acc").round(4).to_string())

valA, valB = runs[runs["protocol"] == "A"], runs[runs["protocol"] == "B"]
test = pd.read_csv(RES / "test_eval_final.csv")
pcv = pd.read_csv(KRES / "per_class_f1_val_protocolB.csv").groupby("config")[names].mean()
pct = test.groupby("model")[names].mean()


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def fmt(s):
    return f"{s.mean() * 100:.2f} ({s.min() * 100:.2f} to {s.max() * 100:.2f})"


t1 = pd.DataFrame([{
    "Configuration": lab[c],
    "Val, protocol A": fmt(valA[valA["config"] == c]["val_acc"]),
    "Val, protocol B": fmt(valB[valB["config"] == c]["val_acc"]),
    "Test, protocol B models": fmt(test[test["model"] == f"cnn_{c}"]["acc"]),
} for c in cfgs])

base = test[test["model"].str.startswith("baseline")]
t2 = pd.DataFrame([{"Baseline (band mean + std)": m, "Runs": len(g), "Test accuracy (%)": fmt(g["acc"])}
                   for m, g in base.groupby("model")])

t3 = pd.DataFrame([{
    "Class": n,
    "RF (test)": f"{pct.loc['baseline_rf', n]:.3f}",
    "RGB (test)": f"{pct.loc['cnn_rgb', n]:.3f}",
    "All 13 (test)": f"{pct.loc['cnn_all13', n]:.3f}",
    "Gain, test": f"{pct.loc['cnn_all13', n] - pct.loc['cnn_rgb', n]:+.3f}",
    "Gain, val": f"{pcv.loc['all13', n] - pcv.loc['rgb', n]:+.3f}",
} for n in names])

text = ("## Accuracy (%): mean (min to max over 3 seeds)\n\n" + md(t1) +
        "\n\n## Baselines\n\n" + md(t2) +
        "\n\n## Per-class F1\n\n" + md(t3) + "\n")
(RES / "results_tables.md").write_text(text)
print()
print(text)

# Figure 1: accuracy per band configuration, one dot per seed
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, c in enumerate(cfgs):
    v = valB[valB["config"] == c]["val_acc"] * 100
    t = test[test["model"] == f"cnn_{c}"]["acc"] * 100
    ax.scatter([i - 0.12] * len(v), v, color="tab:blue", alpha=0.7, s=24, label="validation" if i == 0 else None)
    ax.scatter([i + 0.12] * len(t), t, color="tab:orange", alpha=0.7, s=24, label="test" if i == 0 else None)
ax.set_xticks(range(len(cfgs)))
ax.set_xticklabels([lab[c] for c in cfgs], rotation=20, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.set_title("EuroSAT band configurations: one dot per seed (y-axis truncated)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "band_configs_accuracy.png", dpi=150)

# Figure 2: per-class F1 on the test set, mean of 3 seeds
fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(names))
w = 0.27
for k, (m, label) in enumerate([("baseline_rf", "Random forest (band mean+std)"),
                                ("cnn_rgb", "CNN, RGB"), ("cnn_all13", "CNN, all 13 bands")]):
    ax.bar(x + (k - 1) * w, pct.loc[m, names], w, label=label)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=35, ha="right")
ax.set_ylim(0.6, 1.0)
ax.set_ylabel("F1 (test, mean of 3 seeds)")
ax.set_title("Per-class F1 (y-axis starts at 0.6)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3)
fig.tight_layout()
fig.savefig(FIG / "per_class_f1_test.png", dpi=150, bbox_inches="tight")

# Figure 3: misclassified test images summed over 3 seeds (diagonal removed)
cms = np.load(RES / "test_confusions.npz")
errs = {}
for m in ["cnn_rgb", "cnn_all13"]:
    cm = sum(cms[f"{m}_seed{s}"] for s in seeds).astype(float)
    np.fill_diagonal(cm, 0)
    errs[m] = cm
vmax = max(e.max() for e in errs.values())
fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
for ax, m, title in zip(axes, ["cnn_rgb", "cnn_all13"], ["CNN, RGB", "CNN, all 13 bands"]):
    ax.imshow(errs[m], cmap="Reds", vmin=0, vmax=vmax)
    ax.set_xticks(range(10))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(10):
        for j in range(10):
            if errs[m][i, j] > 0:
                ax.text(j, i, int(errs[m][i, j]), ha="center", va="center", fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
fig.suptitle("Test set: misclassified images summed over 3 seeds (correct predictions not shown)")
fig.tight_layout()
fig.savefig(FIG / "test_confusion_rgb_vs_all13.png", dpi=150)
print("saved figures in", FIG)
