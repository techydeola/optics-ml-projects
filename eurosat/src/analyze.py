"""Validation-set analysis of the protocol B models.

Per-class F1 for every band configuration (saved to CSV), the most frequent confusions, and a figure of
randomly drawn images that the RGB model (seed 42) got wrong, with the 13-band prediction next to it.
The test set is not used here.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

from common import BAND_SETS, CACHE, FIG, KRES, EuroSATSequence, class_names, load_trained

names = class_names()
CFGS = ["all13", "rgb", "vis_nir", "vis_swir", "no_nir", "no_swir"]

rows, cm_sum, preds = [], {}, {}
for cfg in CFGS:
    ds = EuroSATSequence("val", BAND_SETS[cfg], 256, augment=False, workers=1, max_queue_size=4)
    cm_sum[cfg] = np.zeros((10, 10), dtype=int)
    for seed in (42, 43, 44):
        pred = load_trained(cfg, seed).predict(ds, verbose=0).argmax(1)
        preds[(cfg, seed)] = pred
        f1s = f1_score(ds.labels, pred, average=None)
        rows.append(dict(config=cfg, seed=seed, **dict(zip(names, f1s))))
        cm_sum[cfg] += confusion_matrix(ds.labels, pred)
y, idx = ds.labels, ds.idx            # identical for every configuration

df = pd.DataFrame(rows)
out = KRES / "per_class_f1_val_protocolB.csv"
df.to_csv(out, index=False)

tab = df.groupby("config")[names].mean().T[CFGS]
tab["all13_minus_rgb"] = tab["all13"] - tab["rgb"]
print(tab.round(3).sort_values("all13_minus_rgb", ascending=False).to_string())

for cfg in ["rgb", "all13"]:
    off = cm_sum[cfg].copy()
    np.fill_diagonal(off, 0)
    top = sorted(((off[i, j], names[i], names[j])
                  for i in range(10) for j in range(10) if off[i, j]), reverse=True)[:6]
    print(f"\ntop confusions, {cfg} (summed over 3 seeds):")
    for n, t, p in top:
        print(f"  {n:3d}  {t} -> {p}")
print("saved:", out)

# examples of RGB errors (seed 42), drawn at random with a fixed seed
raw = np.load(CACHE, mmap_mode="r")
p_rgb, p_all = preds[("rgb", 42)], preds[("all13", 42)]
pairs = [("HerbaceousVegetation", "PermanentCrop"), ("AnnualCrop", "PermanentCrop"),
         ("Pasture", "HerbaceousVegetation"), ("River", "AnnualCrop")]
rng = np.random.default_rng(0)
N = 6
fig, axes = plt.subplots(len(pairs), N, figsize=(14, 2.9 * len(pairs)))
print()
for r, (t, p) in enumerate(pairs):
    ti, pi = names.index(t), names.index(p)
    sel = np.where((y == ti) & (p_rgb == pi))[0]
    print(f"{t} -> {p} (RGB errors, seed 42): {len(sel)} | all13 got {int((p_all[sel] == ti).sum())} of them right")
    if len(sel) > N:
        sel = rng.choice(sel, N, replace=False)
    for c in range(N):
        ax = axes[r, c]
        ax.axis("off")
        if c < len(sel):
            k = sel[c]
            img = raw[idx[k]].astype(np.float32)[[3, 2, 1]].transpose(1, 2, 0)
            ax.imshow(np.clip(img / 3000.0, 0, 1))
            ax.set_title(f"true: {t[:12]}\nRGB: {names[p_rgb[k]][:12]}\nall13: {names[p_all[k]][:12]}", fontsize=7)
plt.tight_layout()
fig_path = FIG / "errors_rgb_vs_all13_val_seed42.png"
fig.savefig(fig_path, dpi=150)
print("saved:", fig_path)
