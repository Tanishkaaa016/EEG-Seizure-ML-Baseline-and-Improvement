# EEG Seizure Detection using Machine Learning

## Overview

This project investigates epileptic seizure detection from EEG signals using machine learning techniques on the CHB-MIT Scalp EEG Database.

A baseline Random Forest model using conventional EEG features was reproduced and enhanced through the incorporation of nonlinear EEG complexity measures (Sample Entropy) and class balancing using SMOTE.

## Research Objective

The objective of this work is to improve seizure detection performance by addressing two common limitations of traditional machine learning approaches:

* Limited representation of nonlinear EEG dynamics
* Severe class imbalance between seizure and non-seizure EEG segments

## Dataset

**CHB-MIT Scalp EEG Database**

* Source: PhysioNet
* Sampling Frequency: 256 Hz
* Channels: 23 EEG channels
* Window Length: 1 second

## Methodology

### Signal Preprocessing

* Bandpass Filtering (0.5–40 Hz)
* Z-score Normalization
* 1-second Window Segmentation

### Feature Extraction

#### Time-Domain Features

* Mean
* Variance
* RMS
* Line Length
* Zero Crossing Rate

#### Frequency-Domain Features

* Delta Band Power
* Theta Band Power
* Alpha Band Power
* Beta Band Power
* Gamma Band Power

#### Nonlinear Feature

* Sample Entropy

### Classification

* Random Forest Classifier
* SMOTE Oversampling
* Threshold Optimization

## Experimental Results

### Average Performance Across Recordings

| Metric      | Baseline Random Forest | Proposed Method |
| ----------- | ---------------------: | --------------: |
| Accuracy    |                 99.63% |      **99.77%** |
| Sensitivity |                  68.3% |       **80.8%** |
| F1 Score    |                  0.785 |       **0.866** |

### Recording-wise Performance

| Recording | Accuracy | Sensitivity | F1 Score |
| --------- | -------: | ----------: | -------: |
| CHB01_03  |   99.86% |      100.0% |    0.941 |
| CHB01_04  |   99.86% |       80.0% |    0.889 |
| CHB01_15  |   99.58% |       62.5% |    0.769 |

### Key Findings

* Improved average sensitivity by 12.5%
* Improved average F1 score by 10.3%
* Maintained classification accuracy above 99%
* Demonstrated the effectiveness of entropy-based EEG complexity measures for seizure detection

## Technologies Used

* Python
* NumPy
* SciPy
* MNE
* WFDB
* Scikit-learn
* Imbalanced-learn
* AntroPy
* Matplotlib

## Author

**Tanishka Bajpai**

Biomedical Engineering (Medical Intelligence)
