"""Train the CNN for one band configuration and one seed.

Protocol A: up to --epochs epochs, early stopping on validation loss, best-validation-loss weights.
Protocol B: exactly --epochs epochs, cosine learning rate decay, final-epoch weights.

Outputs (name = <tag>_<bands>_seed<seed>):
  data/checkpoints/<name>.weights.h5, eurosat/results/keras/<name>_history.csv and <name>_summary.json
"""
import argparse
import json
import time

import keras
from sklearn.metrics import f1_score

from common import BAND_SETS, CKPT, KRES, EuroSATSequence, build_cnn

ap = argparse.ArgumentParser()
ap.add_argument("--bands", choices=list(BAND_SETS), default="all13")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--protocol", choices=["A", "B"], default="B")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--patience", type=int, default=6)        # protocol A only
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tag", default=None)                    # default: "fixed" for B, "main" for A
a = ap.parse_args()

tag = a.tag or ("fixed" if a.protocol == "B" else "main")
keras.utils.set_random_seed(a.seed)                       # seeds python, numpy and TensorFlow together
bands = BAND_SETS[a.bands]

train = EuroSATSequence("train", bands, a.batch, augment=True, seed=a.seed, workers=3, max_queue_size=8)
val = EuroSATSequence("val", bands, 256, augment=False, workers=3, max_queue_size=8)

model = build_cnn(in_channels=len(bands))
if a.protocol == "B":
    lr = keras.optimizers.schedules.CosineDecay(a.lr, decay_steps=a.epochs * len(train), alpha=0.01)
else:
    lr = a.lr
model.compile(
    optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

name = f"{tag}_{a.bands}_seed{a.seed}"
CKPT.mkdir(parents=True, exist_ok=True)
KRES.mkdir(parents=True, exist_ok=True)
ckpt = CKPT / f"{name}.weights.h5"

callbacks = []
if a.protocol == "A":
    callbacks += [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, save_weights_only=True),
    ]
callbacks.append(keras.callbacks.CSVLogger(KRES / f"{name}_history.csv"))

print(f"protocol {a.protocol} | bands={a.bands} | seed={a.seed} | epochs<={a.epochs} | params={model.count_params():,}")
t0 = time.time()
hist = model.fit(train, validation_data=val, epochs=a.epochs, verbose=2, callbacks=callbacks)

if a.protocol == "A":
    model.load_weights(ckpt)          # best-validation-loss weights
else:
    model.save_weights(ckpt)          # final-epoch weights, no selection on validation data

pred = model.predict(val, verbose=0).argmax(1)
vl, va = model.evaluate(val, verbose=0)
summary = dict(name=name, framework="keras", protocol=a.protocol, bands=a.bands, seed=a.seed,
               epochs_run=len(hist.history["loss"]), val_loss=float(vl), val_acc=float(va),
               val_macro_f1=float(f1_score(val.labels, pred, average="macro")),
               minutes=round((time.time() - t0) / 60, 1))
(KRES / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
print("validation:", json.dumps(summary))
