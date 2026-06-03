# Final Results

## Proposed Method

CHB-MIT EEG

↓

Bandpass Filtering (0.5–30 Hz)

↓

Z-score Normalization

↓

1-Second Window Segmentation

↓

Time-Domain Features

*

Frequency-Domain Features

*

Sample Entropy

↓

SMOTE Oversampling

↓

Random Forest Classification

↓

Threshold Optimization

↓

Seizure Detection

---

## Average Performance Across Recordings

| Metric      | Baseline Random Forest | Proposed Method |
| ----------- | ---------------------: | --------------: |
| Accuracy    |                 99.63% |      **99.77%** |
| Sensitivity |                  68.3% |       **80.8%** |
| F1 Score    |                  0.785 |       **0.866** |

---

## Recording-wise Performance

| Recording | Accuracy | Sensitivity | F1 Score |
| --------- | -------: | ----------: | -------: |
| CHB01_03  |   99.86% |      100.0% |    0.941 |
| CHB01_04  |   99.86% |       80.0% |    0.889 |
| CHB01_15  |   99.58% |       62.5% |    0.769 |

---

## Key Improvements Over Baseline

* Sensitivity improved from **68.3%** to **80.8%**
* F1 Score improved from **0.785** to **0.866**
* Classification accuracy remained above **99%**
* Sample Entropy improved representation of nonlinear EEG dynamics
* SMOTE improved detection of seizure windows under class imbalance
