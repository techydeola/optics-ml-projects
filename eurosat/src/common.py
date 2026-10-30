"""Shared code for the EuroSAT experiments: paths, data access, the Keras data sequence and the CNN."""
import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tifffile
from keras import layers

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "EuroSAT_MS"
CACHE = ROOT / "data" / "eurosat_ms_uint16.npy"
META = ROOT / "data" / "eurosat_meta.npz"
FEATURES = ROOT / "data" / "features_band_stats.npz"
CKPT = ROOT / "data" / "checkpoints"
RES = ROOT / "eurosat" / "results"
KRES = RES / "keras"
FIG = ROOT / "eurosat" / "figures"
SPLIT_CSV = RES / "split.csv"
BAND_STATS = RES / "band_stats_train.json"

# Channel indices in the EuroSAT_MS files. The order was checked against the data (see README).
BAND_SETS = {
    "all13": list(range(13)),
    "rgb": [3, 2, 1],
    "vis_nir": [3, 2, 1, 7, 12],
    "vis_swir": [3, 2, 1, 10, 11],
    "no_nir": [i for i in range(13) if i not in (7, 12)],
    "no_swir": [i for i in range(13) if i not in (10, 11)],
}


def load_split():
    return pd.read_csv(SPLIT_CSV)


def class_names():
    return (load_split()[["label", "class_name"]].drop_duplicates()
            .sort_values("label")["class_name"].tolist())


def load_image(rel_path):
    """One image as float32, shape (13, 64, 64), in the raw uint16 units."""
    img = tifffile.imread(DATA / rel_path)          # stored as (64, 64, 13)
    return np.transpose(img, (2, 0, 1)).astype(np.float32)


class EuroSATSequence(keras.utils.PyDataset):
    """Reads one batch at a time from the memory-mapped cache, normalises it with train-only
    statistics and, for training, applies flips and 90 degree rotations seeded per batch."""

    def __init__(self, split, bands, batch_size=128, augment=False, seed=42, **kw):
        super().__init__(**kw)
        self.X = np.load(CACHE, mmap_mode="r")
        meta = np.load(META)
        self.idx = np.where(meta["split"] == split)[0]
        self.labels = meta["y"][self.idx]
        self.bands = list(bands)
        st = json.loads(BAND_STATS.read_text())
        self.mean = np.array(st["mean"], dtype=np.float32)[self.bands].reshape(1, 1, 1, -1)
        self.std = np.array(st["std"], dtype=np.float32)[self.bands].reshape(1, 1, 1, -1)
        self.batch_size = batch_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.rng = np.random.default_rng(seed)   # used only for shuffling the order
        self.order = np.arange(len(self.idx))
        if augment:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.batch_size))

    def __getitem__(self, i):
        sel = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        x = self.X[self.idx[sel]][:, self.bands].astype(np.float32)   # (B, C, 64, 64)
        x = x.transpose(0, 2, 3, 1)                                    # (B, 64, 64, C)
        x = (x - self.mean) / self.std
        if self.augment:
            # seeded per (seed, epoch, batch) so the result does not depend on which worker runs the batch
            rng = np.random.default_rng([self.seed, self.epoch, i])
            for k in range(len(x)):
                if rng.random() < 0.5:
                    x[k] = np.ascontiguousarray(np.flip(x[k], axis=1))
                if rng.random() < 0.5:
                    x[k] = np.ascontiguousarray(np.flip(x[k], axis=0))
                r = int(rng.integers(0, 4))
                if r:
                    x[k] = np.ascontiguousarray(np.rot90(x[k], r, axes=(0, 1)))
        return np.ascontiguousarray(x), self.labels[sel]

    def on_epoch_end(self):
        self.epoch += 1
        if self.augment:
            self.rng.shuffle(self.order)


def _block(x, c):
    for _ in range(2):
        x = layers.Conv2D(c, 3, padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.ReLU()(x)
    return layers.MaxPooling2D(2)(x)


def build_cnn(in_channels=13, n_classes=10):
    inp = keras.Input((64, 64, in_channels))
    x = _block(inp, 32)
    x = _block(x, 64)
    x = _block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    return keras.Model(inp, layers.Dense(n_classes)(x))


def load_trained(cfg, seed, tag="fixed"):
    """CNN for a band configuration with the weights saved by train.py."""
    model = build_cnn(in_channels=len(BAND_SETS[cfg]))
    model.load_weights(CKPT / f"{tag}_{cfg}_seed{seed}.weights.h5")
    return model
