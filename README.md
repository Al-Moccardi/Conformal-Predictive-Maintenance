# Robust Conformal Predictive Maintenance (Conformal‑PdM)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/framework-TensorFlow-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/library-scikit--learn-blue)](https://scikit-learn.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1USNgrhhX_6Lwznq60azDZ4EZi4cxfyj0?usp=sharing)



## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Quick Start (Colab)](#quick-start-colab)
4. [Conformal Prediction Modes](#conformal-prediction-modes)
5. [Experimental Results](#experimental-results)

---

## Overview

This **single‑notebook implementation** delivers an end‑to‑end pipeline for *Remaining Useful Life* (**RUL**) prediction on the NASA CMAPSS turbofan benchmark and augments it with **robust conformal prediction** to provide calibrated, statistically valid uncertainty bounds.
![image](https://github.com/user-attachments/assets/f6a333f6-1143-40bf-8f1b-2de6ef143afd)

The proposed prognostic workflow, presented in Fig. 8, begins by generating RUL 
estimations from ML or DL models and comparing the validation performances, with these 
initial predictions passed through a regulation step where non-increasing monotonicity is 
enforced to ensure that RUL does not inappropriately increase over time, which represents 
a critical constraint in most degradation processes. 
Next, the framework proceeds to the calibration and residuals phase, where the 
regulated predictions are compared against true values from a calibration set (training set) 
to compute residuals and the margin estimators applied to these residuals to quantify the 
uncertainty Specifically, this work considers the three main strategies introduced in the 
Sec. 3.2 (Naïve , bootstrapping, weighted margin) which are commonly applied in time 
series conformal applications. 
• Naive Margin (Quantile 1-α) uses a straightforward quantile-based approach. 
• Bootstrap Margin (Median Bootstrap) leverages resampling to obtain robust estimates 
of variability. 
• Weighted Margin (Exponential Weighting) accounts for time-varying or instance- 
specific importance. 
The latter phase consent to compare or combine the calculated margin with the 
final confidence further halved (not considering superior CI) and conservatively adjusted 
to avoid excessively wide intervals (selecting the proper confidence value α ) reflecting 
engineering judgment and domain requirements for safety and cost considerations. Given 
these conditions, the outlined approach constructs the final prediction interval, intentionally 
bounded above by the regulated prediction, thereby preserving the strictly decreasing RUL 
trajectory and mitigating undue sensitivity to sensor noise. The CI estimation provides 
coverage that aligns with desired confidence levels, giving the possibility to evaluate the 
proximity to EoL score, which gives an idea of the model conservativeness by calculating 
the time window distance between the conformal critical status (when CI predicts RUL=0) 
and the actual Eol (when validation RUL = 0), allowing engineers and practitioners to 
evaluate whether immediate intervention is warranted or if further analysis is possible 
before triggering timely interventions.


## Key Features

* ⚙️ **Modular deep‑learning stack**: LSTM, BiLSTM, CNN‑GRU, N‑BEATS and an XGBoost baseline.
* 🎯 **Monotonicity‑aware loss** ensures physically plausible, non‑increasing RUL curves.
* 📏 **Conformal wrappers**: naïve, exponentially‑weighted and bootstrap residual margins.

![image](https://github.com/user-attachments/assets/b4dca189-6277-48c9-b856-631dd22ab829)


## Quick Start (Colab)

1. Click the badge at the top of this page **or** the link below:
   [https://colab.research.google.com/drive/1USNgrhhX\_6Lwznq60azDZ4EZi4cxfyj0?usp=sharing](https://colab.research.google.com/drive/1USNgrhhX_6Lwznq60azDZ4EZi4cxfyj0?usp=sharing)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1USNgrhhX_6Lwznq60azDZ4EZi4cxfyj0?usp=sharing)


3. Select **“Runtime → Change runtime type → GPU”** for faster training (optional but recommended).
4. Run the notebook cells sequentially; each section is self‑documented and includes performance checkpoints.

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

![image](https://github.com/user-attachments/assets/a29d9371-4d70-4687-992b-083b313b95d0)

![image](https://github.com/user-attachments/assets/bd1252bf-6b2d-4028-b059-a95e158b4b75)

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
