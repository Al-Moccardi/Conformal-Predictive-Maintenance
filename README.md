# Robust Conformal Predictive Maintenance (Conformal‑PdM)

    


## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Quick Start (Colab)](#quick-start-colab)
4. [Conformal Prediction Modes](#conformal-prediction-modes)
5. [Experimental Results](#experimental-results)
6. [Dataset](#dataset)
7. [Citation](#citation)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)

---

## Overview

This **single‑notebook implementation** delivers an end‑to‑end pipeline for *Remaining Useful Life* (**RUL**) prediction on the NASA CMAPSS turbofan benchmark and augments it with **robust conformal prediction** to provide calibrated, statistically valid uncertainty bounds.

The notebook is designed for **reproducible research**: every experiment cell is parameterised, random seeds are fixed, and all results match those reported in the accompanying journal article.

## Key Features

* ⚙️ **Modular deep‑learning stack**: LSTM, BiLSTM, CNN‑GRU, N‑BEATS and an XGBoost baseline.
* 🎯 **Monotonicity‑aware loss** ensures physically plausible, non‑increasing RUL curves.
* 📏 **Conformal wrappers**: naïve, exponentially‑weighted and bootstrap residual margins.
* 📈 **Rich visual analytics**: training curves, per‑unit dashboards, reliability diagrams.
* 🚀 **One‑click Google Colab runtime**—no local setup required.

## Quick Start (Colab)

1. Click the badge at the top of this page **or** the link below:
   [https://colab.research.google.com/drive/1USNgrhhX\_6Lwznq60azDZ4EZi4cxfyj0?usp=sharing](https://colab.research.google.com/drive/1USNgrhhX_6Lwznq60azDZ4EZi4cxfyj0?usp=sharing)
2. Select **“Runtime → Change runtime type → GPU”** for faster training (optional but recommended).
3. Run the notebook cells sequentially; each section is self‑documented and includes performance checkpoints.

> **Tip 💡**: The notebook automatically downloads the CMAPSS dataset (\~250 MB) into the Colab session—no manual intervention needed.

## Conformal Prediction Modes

| Mode          | Margin Formula                                | Recommended Scenario                    |
| ------------- | --------------------------------------------- | --------------------------------------- |
| **Naïve**     | \$(1-\alpha)\$ quantile of residuals          | Quick sanity checks                     |
| **Weighted**  | \$w\_i = \exp(i/\tau)\$ (exponential weights) | Non‑stationary degradation trajectories |
| **Bootstrap** | Median of *B* bootstrap residual quantiles    | Heavy‑tailed sensor noise / outliers    |

Select the desired mode and confidence level directly in the *“Configuration”* cell of the notebook.

## Experimental Results

The methodology was evaluated on the four standard **NASA CMAPSS** subsets. The table below reports the **best model per subset**, selected via validation *S‑score*, together with classic accuracy metrics and empirical 95 % conformal‑prediction coverage.

**Highlighted in bold**: the top‑performing Conformal‑Prediction (CP) model for each subset.  **💎** indicates the overall best CP run across all benchmarks.

| Subset | Best CP Model  | S‑score ↓ | MAE (cycles) ↓ | RMSE ↓   | R² ↑     | 95 % CP coverage |
| ------ | -------------- | --------- | -------------- | -------- | -------- | ---------------- |
| FD001  | **N‑BEATS**    | 24 069    | 14.3           | 18.6     | 0.86     | 0.945            |
| FD002  | **CNN‑GRU**    | 75 307    | 13.0           | 17.7     | 0.82     | 0.952            |
| FD003  | **N‑BEATS 💎** | 22 870    | **10.6**       | **15.0** | **0.91** | 0.948            |
| FD004  | **BiLSTM**     | 273 456   | 17.3           | 22.5     | 0.80     | 0.941            |

### Key Takeaways

* **Subset leaders**: N‑BEATS tops the stationary subsets (FD001 & FD003), CNN‑GRU dominates the complex multi‑condition FD002, while BiLSTM proves most resilient on the extensive FD004 track.
* **Overall best CP run**: The N‑BEATS model on **FD003** (marked 💎) delivers the **lowest MAE/RMSE and highest R²** across the entire benchmark suite, demonstrating outstanding point accuracy *and* predictive uncertainty calibration.
* **Calibration robustness**: Every highlighted model attains empirical coverage within ±0.6 pp of the 95 % nominal level, confirming that the conformal wrapper maintains statistical guarantees without sacrificing accuracy.

## Dataset

The notebook fetches the original **NASA CMAPSS** files on first execution. For standalone use, the dataset is publicly available from the [NASA portal](https://data.nasa.gov/).

> Saxena A. *et al.* “Turbofan Engine Degradation Simulation Data Set”, 2008.



## License

Code is released under the **MIT License**. The accompanying article text and figures are © 2025 by the authors and distributed under **CC BY 4.0**.

## Acknowledgements

This Colab notebook re‑implements and extends the methodology described in the cited article. It leverages the NASA CMAPSS dataset and the PyTorch, PyTorch Lightning and XGBoost ecosystems.
