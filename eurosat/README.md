# EuroSAT land cover classification: which Sentinel-2 bands matter?

## Summary

I trained a small CNN to classify 64x64 Sentinel-2 patches from the EuroSAT dataset into 10 land cover classes, and tested which spectral bands the network needs.

- On the held-out test set (4,050 images, evaluated once), the CNN with all 13 bands reached 98.58% accuracy (mean of 3 seeds, range 98.40 to 98.74). The same CNN with RGB only reached 97.79% (range 97.70 to 97.88).
- A random forest on per-band mean and standard deviation reached 89.97%.
- Adding only near-infrared or only shortwave infrared bands to RGB gave 98.51% and 98.39%, close to the full set. Removing either group from the full set made no measurable difference.
- This is a small study: one dataset, one fixed split, 3 seeds per configuration, a small CNN trained from scratch on a laptop.

I did this project to see how much information about the ground is carried by light at different wavelengths, and how much of it a neural network actually uses. A satellite measures sunlight reflected from the surface, and different materials reflect different wavelengths differently, so the question links the physics of reflected light to a machine learning experiment. I have a physics electronics background and have worked with climate data, and this is my first computer vision project. I built it so that every number can be traced to a script and reproduced.

## Data

- EuroSAT, multispectral version: 27,000 patches, 64x64 pixels, 13 Sentinel-2 bands, 10 classes (class sizes 2,000 to 3,000).
- Source: Zenodo record 7711810 (doi 10.5281/zenodo.7711810), file `EuroSAT_MS.zip` (about 2.1 GB). I checked the MD5 checksum after download.
- License: MIT, as listed on the Zenodo record. The underlying Sentinel data are free and open under EU law, see the Copernicus Sentinel Data Terms and Conditions.
- Citation: P. Helber, B. Bischke, A. Dengel, D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217 to 2226, 2019, doi 10.1109/JSTARS.2019.2918242. An earlier conference version appeared at IGARSS 2018, doi 10.1109/IGARSS.2018.8519248.
- Values are stored as uint16 (Sentinel-2 reflectance times 10,000).

### Band order check

I did not assume the order of the 13 channels. In these files the channels are not in the usual B01 to B12 listing order. I identified them from the data:

- Mean spectra of Forest, SeaLake, Industrial and River patches (100 random patches per class, fixed seed). Water is dark and forest is bright in the infrared channels.
- A natural-color render using channels 3, 2, 1 as red, green, blue looked correct.

Channel groups used in the experiments (indices are positions in the file, counting from 0):

| Group | Indices | Basis |
|---|---|---|
| Visible (blue, green, red) | 1, 2, 3 | natural-color check |
| Near-infrared (NIR) | 7, 12 | water dark, forest bright |
| Shortwave infrared (SWIR) | 10, 11 | water dark, forest bright, brighter first channel |
| Not placed in a group | 0, 4, 5, 6, 8, 9 | |

I did not work out which of the two NIR channels is B08 and which is B8A, so they are treated as one group. Channel 9 has values near 10 in every class, which looks like the cirrus band.

The order was therefore checked on the data and not taken from documentation. If you use these indices with another copy of the dataset, check them again.

### Split

Stratified 70/15/15 split with seed 42: 18,900 train, 4,050 validation, 4,050 test. The assignment of every image is saved in `eurosat/results/split.csv`.

### Preprocessing

- Per-band mean and standard deviation computed on the training set only, then applied to all sets.
- Augmentation on the training set only: random horizontal and vertical flips, random rotation by a multiple of 90 degrees. Seeded per batch.

## Models

- **Baselines:** logistic regression (standardised features) and random forest (300 trees) on 26 features per image (mean and standard deviation of each band). Trained on the train split only.
- **CNN:** three blocks of two 3x3 convolutions with batch normalisation and ReLU, each followed by 2x2 max pooling (32, 64, 128 filters), global average pooling, and a linear layer. 292,074 trainable parameters with 13 input bands (290,090 with RGB). Only the first layer changes between band configurations.
- **Band configurations:** RGB, visible + NIR, visible + SWIR, all minus NIR, all minus SWIR, all 13.

## Training protocols

Both use AdamW (learning rate 0.001, weight decay 0.0001), batch size 128, and seeds 42, 43, 44.

- **Protocol A:** up to 80 epochs, early stopping with patience 6 on validation loss, best-validation-loss weights. (The all-13 runs and the first three RGB runs used a 40-epoch cap, and the band-group runs used a cap of 80. RGB seeds 43 and 44 were rerun with a cap of 80, and seed 44 gave identical results. The protocol A table uses the cap-40 runs for all 13 bands and RGB seed 42, and the cap-80 reruns for RGB seeds 43 and 44. The all-13 runs and RGB seed 42 ended through the patience rule (all-13 seed 42 stopped at exactly epoch 40, six epochs after its best epoch 34), so a cap of 80 would not have changed them.)
- **Protocol B:** exactly 40 epochs, cosine learning rate decay to 1% of the starting rate, final-epoch weights, no checkpoint selection.

Why two protocols: under protocol A, some runs stopped early on noisy validation loss, which made differences between configurations hard to separate from stopping luck (for example, seed 43 stopped after 16 to 18 epochs in three configurations). I then ran protocol B for all 18 configurations and seeds. The protocol B settings were chosen before running it and were not tuned afterwards. This means the validation results influenced the decision to rerun, and I report both protocols.

## Evaluation

- Validation set: development, per-class analysis, error analysis.
- Test set: evaluated once, after the plan was fixed (the 18 protocol B models, the random forest with 3 seeds, one logistic regression). The script refuses to run a second time. The pre-declared headline comparison was all 13 bands against RGB.

## Results

### Accuracy (%): mean (min to max over 3 seeds)

| Configuration | Val, protocol A | Val, protocol B | Test, protocol B models |
|---|---|---|---|
| RGB | 96.99 (96.59 to 97.33) | 97.55 (97.46 to 97.60) | 97.79 (97.70 to 97.88) |
| Visible + NIR | 96.98 (96.30 to 97.38) | 98.37 (98.27 to 98.49) | 98.51 (98.42 to 98.59) |
| Visible + SWIR | 97.02 (96.00 to 97.60) | 98.47 (98.37 to 98.62) | 98.39 (98.22 to 98.62) |
| All minus SWIR | 98.03 (97.90 to 98.12) | 98.73 (98.54 to 98.91) | 98.57 (98.52 to 98.59) |
| All minus NIR | 97.55 (97.04 to 97.93) | 98.68 (98.64 to 98.74) | 98.63 (98.54 to 98.74) |
| All 13 bands | 98.18 (97.95 to 98.44) | 98.63 (98.54 to 98.72) | 98.58 (98.40 to 98.74) |

Only compare numbers within one column. The protocols differ.

### Baselines (test accuracy)

| Baseline (band mean + std) | Runs | Test accuracy (%) |
|---|---|---|
| Logistic regression | 1 | 88.35 |
| Random forest | 3 | 89.97 (89.80 to 90.05) |

### Per-class F1

| Class | RF (test) | RGB (test) | All 13 (test) | Gain, test | Gain, val |
|---|---|---|---|---|---|
| AnnualCrop | 0.880 | 0.959 | 0.977 | +0.018 | +0.015 |
| Forest | 0.986 | 0.996 | 0.997 | +0.002 | +0.001 |
| HerbaceousVegetation | 0.899 | 0.978 | 0.986 | +0.008 | +0.026 |
| Highway | 0.685 | 0.978 | 0.981 | +0.003 | -0.003 |
| Industrial | 0.917 | 0.986 | 0.985 | -0.001 | +0.008 |
| Pasture | 0.904 | 0.965 | 0.977 | +0.012 | +0.007 |
| PermanentCrop | 0.858 | 0.959 | 0.974 | +0.015 | +0.033 |
| Residential | 0.894 | 0.992 | 0.991 | -0.001 | +0.004 |
| River | 0.935 | 0.964 | 0.988 | +0.024 | +0.015 |
| SeaLake | 0.996 | 0.994 | 0.997 | +0.003 | +0.003 |

### Figures

- `figures/band_configs_accuracy.png`: accuracy per configuration, one dot per seed (y-axis truncated).
- `figures/per_class_f1_test.png`: per-class F1 on the test set (y-axis starts at 0.6).
- `figures/test_confusion_rgb_vs_all13.png`: misclassified test images summed over 3 seeds.
- `figures/errors_rgb_vs_all13_val_seed42.png`: random examples of RGB errors, with the 13-band prediction.

## What the results show

Measured statements:

1. All 13 bands scored about 0.8 points higher than RGB on the test set (about 1.1 points on validation under protocol B). The ranges of the three seeds do not overlap.
2. Adding one infrared group to RGB (visible + NIR or visible + SWIR) recovered most of that gap.
3. Removing either infrared group from the full set changed accuracy by less than the spread between seeds.
4. Per class, the crop classes, the vegetation classes and River gained from the extra bands in both validation and test. Forest, SeaLake, Residential, Industrial and Highway did not. The order of the gains among the improved classes differs between validation and test, so I do not rank them.
5. The random forest on band statistics scored about 8.6 points lower than the 13-band CNN. Its Highway F1 was 0.685, against 0.98 for the CNNs.

Possible explanations (hypotheses that fit these results, not something this experiment tested):

- Crop and vegetation classes all contain green plants, so they can look alike in red, green and blue light. Near-infrared and shortwave infrared reflectance depend on leaf structure, canopy density and water content, so the extra bands may separate crop types that RGB cannot.
- Water absorbs strongly in the near-infrared and shortwave infrared, whatever its color in visible light. This may help with rivers that do not look dark in RGB, for example because of sediment.
- Forest and SeaLake are already easy in RGB, and Residential, Industrial and Highway are probably separated by shape and texture more than by spectrum. The low Highway score of the random forest (0.685), which sees only band statistics, against about 0.98 for all CNNs with or without extra bands, fits this.
- Statements 2 and 3 are consistent with near-infrared and shortwave infrared carrying overlapping information for this task. I did not test this directly, for example with a visible + NIR + SWIR run or by measuring correlations between bands, so I treat it as a possibility.

## Error analysis

On validation seed 42 with the RGB model, I looked at errors in four confusion pairs (counts from `error_analysis.py`):

| RGB error | Count | Fixed by all 13 bands |
|---|---|---|
| HerbaceousVegetation as PermanentCrop | 15 | 12 |
| AnnualCrop as PermanentCrop | 8 | 4 |
| Pasture as HerbaceousVegetation | 7 | 4 |
| River as AnnualCrop | 4 | 4 |

The images in the figure were drawn at random from these errors.

What I see in `errors_rgb_vs_all13_val_seed42.png` (22 patches in total):

- **HerbaceousVegetation:** the patches the RGB model got wrong look mixed and sparse, with tan and brown terrain, a road crossing one patch and bright bare-looking spots in others. I could not confidently call most of them herbaceous vegetation by eye in RGB, so these are hard cases and some labels may be ambiguous. The 13-band model fixed all six shown.
- **AnnualCrop:** the patches are patchworks of small fields with a mix of bare and green parcels. Telling annual from permanent crops in a single image looks hard even for a person. The 13-band model fixed three of the six shown.
- **Pasture:** the patches are dark, low contrast and grey-green, and look much like the vegetation class in RGB. The 13-band model fixed three of the six shown.
- **River:** the rivers are easy to see by eye, and the 13-band model fixed all four shown. In at least one patch the channel is wide and pale, not dark, which may be why the RGB model confused it with fields. Water absorbs infrared whatever its visible color, which could explain the fix, but I did not test this.

The patches are a small random sample from one seed, so they illustrate the errors and do not make a systematic analysis.

## Limitations

- **Possible leakage between splits.** EuroSAT patches are crops from larger Sentinel-2 scenes. The split is random over patches, so near-identical neighbouring patches may be in both train and test. I could not measure this. The test accuracies may be higher than they would be on new regions.
- **One fixed split.** The 3 seeds measure training variation only, not variation due to a different split.
- **Small number of seeds.** With 3 runs, the spread is a rough estimate. Differences smaller than about 0.2 points on the test set (4,050 images) are within sampling noise, and I do not interpret them.
- **Protocol choice.** I introduced protocol B after seeing noisy protocol A results.
- **Band order** was inferred from the data (see above), and the two NIR channels were not individually identified.
- **Library versions.** All results in this repository were produced in the environment pinned in requirements.txt (numpy 2.0.2). Running the same logistic regression on the same features file in a second environment (numpy 2.5.4, same scikit-learn 1.9.1) gave a validation accuracy of 0.8773 instead of 0.8765, a difference of 3 of 4,050 images, so this solver's results depend slightly on the numerical libraries. I did not investigate which library. The split, band statistics, random forest, per-class results and tables reproduced exactly.
- **Validation numbers** under protocol A are slightly optimistic because the best epoch was chosen on validation loss.
- **Scope.** One dataset, one date per patch, one network size, no hyperparameter search. Results say nothing about other sensors or regions.
- **Hardware.** Trained on an Apple M1 laptop (8 GB RAM) with TensorFlow 2.18.1, Keras 3.15.1 and tensorflow-metal 1.2.0. Runs on a GPU backend can differ slightly between machines. Seeds fix the data order and augmentation, and one repeated run I checked reproduced the first 40 epochs exactly, but I do not claim bit-identical results in general.

## What I would do next

- Use a spatially separated split (by scene or region) to test the leakage concern.
- Run visible + NIR + SWIR to test whether the two infrared groups are redundant.
- Try a learning rate schedule for the early stopping protocol, and more seeds.
- Add a transfer learning comparison (for example ResNet18), which would need a decision on how to feed 13 bands to a pretrained network.
- Test spectral indices (for example a vegetation index) as inputs to the baseline.

## Reproduce

Python 3.12. Create an environment and install:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data and preparation:

```bash
mkdir -p data && cd data
curl -L -C - -o EuroSAT_MS.zip "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"
md5 EuroSAT_MS.zip      # expect 091174add3c8e680a49244acf185b9f0 (use md5sum on Linux)
unzip -q EuroSAT_MS.zip && cd ..
python eurosat/src/prepare.py     # split, baseline features, uint16 cache, train-only band statistics
python eurosat/src/baseline.py    # baselines, validation set
```

Training (about 10 to 30 minutes per run on an M1; weights are not included in the repo):

```bash
scripts/run_all.sh                # all protocol B and protocol A runs; finished runs are skipped
```

Analysis:

```bash
python eurosat/src/analyze.py     # per-class F1 and error examples, validation set
python eurosat/src/evaluate_test.py   # test set, run once; delete eurosat/results/test_eval_final.csv first
                                      # if you are reproducing from scratch
python eurosat/src/report.py      # tables and figures
```

## Repository layout

```
eurosat/src/        common.py (shared code), prepare.py, baseline.py, train.py,
                    analyze.py, evaluate_test.py, report.py
eurosat/results/    split.csv, per-run summaries, result tables
eurosat/figures/    figures used above
eurosat/notebooks/  exploration of the raw data and the band order check
scripts/run_all.sh  runs all training runs
```
