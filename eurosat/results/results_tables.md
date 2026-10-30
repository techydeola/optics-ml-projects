## Accuracy (%): mean (min to max over 3 seeds)

| Configuration | Val, protocol A | Val, protocol B | Test, protocol B models |
|---|---|---|---|
| RGB | 96.99 (96.59 to 97.33) | 97.55 (97.46 to 97.60) | 97.79 (97.70 to 97.88) |
| Visible + NIR | 96.98 (96.30 to 97.38) | 98.37 (98.27 to 98.49) | 98.51 (98.42 to 98.59) |
| Visible + SWIR | 97.02 (96.00 to 97.60) | 98.47 (98.37 to 98.62) | 98.39 (98.22 to 98.62) |
| All minus SWIR | 98.03 (97.90 to 98.12) | 98.73 (98.54 to 98.91) | 98.57 (98.52 to 98.59) |
| All minus NIR | 97.55 (97.04 to 97.93) | 98.68 (98.64 to 98.74) | 98.63 (98.54 to 98.74) |
| All 13 bands | 98.18 (97.95 to 98.44) | 98.63 (98.54 to 98.72) | 98.58 (98.40 to 98.74) |

## Baselines

| Baseline (band mean + std) | Runs | Test accuracy (%) |
|---|---|---|
| baseline_logreg | 1 | 88.35 (88.35 to 88.35) |
| baseline_rf | 3 | 89.97 (89.80 to 90.05) |

## Per-class F1

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
