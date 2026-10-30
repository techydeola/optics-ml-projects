"""Prepare the data in one pass over the 27,000 files.

Writes: the stratified split (eurosat/results/split.csv), per-band mean and std features for the
baselines, a uint16 cache of all images, and per-band mean and std computed on the train split only.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common import BAND_STATS, CACHE, DATA, FEATURES, META, SPLIT_CSV, load_image

SEED = 42

# 1. stratified 70/15/15 split with a fixed seed
classes = sorted(p.name for p in DATA.iterdir() if p.is_dir())
rows = [(str(f.relative_to(DATA)), c, i)
        for i, c in enumerate(classes) for f in sorted((DATA / c).glob("*.tif"))]
df = pd.DataFrame(rows, columns=["path", "class_name", "label"])
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)
out = pd.concat([
    train_df.assign(split="train"),
    val_df.assign(split="val"),
    test_df.assign(split="test"),
]).sort_values("path").reset_index(drop=True)

# every image appears exactly once, in exactly one split
assert len(out) == len(df) == out["path"].nunique()
assert out.groupby("path")["split"].nunique().max() == 1

SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(SPLIT_CSV, index=False)
print(pd.crosstab(out["class_name"], out["split"], margins=True))

# 2. one pass over the images: uint16 cache and the 26 baseline features (13 means, 13 stds)
CACHE.parent.mkdir(parents=True, exist_ok=True)
X = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.uint16, shape=(len(out), 13, 64, 64))
feats = []
for i, p in enumerate(tqdm(out["path"])):
    img = load_image(p)                                   # (13, 64, 64) float32
    X[i] = img.astype(np.uint16)
    feats.append(np.concatenate([img.mean(axis=(1, 2)), img.std(axis=(1, 2))]))
X.flush()

y = out["label"].to_numpy()
split = out["split"].to_numpy().astype(str)               # plain strings, so no pickle is needed
feats = np.stack(feats)
np.savez(FEATURES, X=feats, y=y, split=split)
np.savez(META, y=y, split=split)
print("features:", feats.shape, feats.dtype, "| NaNs:", int(np.isnan(feats).sum()))

# 3. per-band mean and std on the TRAIN split only
tr = np.where(split == "train")[0]
s, ss, n = np.zeros(13), np.zeros(13), 0
for k in range(0, len(tr), 1000):
    chunk = X[tr[k:k + 1000]].astype(np.float64)
    s += chunk.sum(axis=(0, 2, 3))
    ss += (chunk ** 2).sum(axis=(0, 2, 3))
    n += chunk.shape[0] * 64 * 64
mean = s / n
std = np.sqrt(ss / n - mean ** 2)
BAND_STATS.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(), "n_pixels": n}, indent=2))
print("train mean:", np.round(mean, 1))
print("train std :", np.round(std, 1))
