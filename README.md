# 🧠 EEG Seizure Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20XGBoost-green)
![Dataset](https://img.shields.io/badge/Dataset-CHB--MIT-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Description

This project focuses on detecting epileptic seizures from EEG signals using machine learning techniques. The pipeline includes signal segmentation, feature extraction, and classification using traditional ML models.

A baseline **Random Forest** model is implemented, followed by an improved **XGBoost** model with enhanced feature engineering to improve detection performance, especially sensitivity.

---

## 🧾 Workflow Overview

```id="flow1"
EEG Data → Segmentation → Feature Extraction → ML Model → Evaluation
```

---

## 📊 Dataset

* **CHB-MIT Scalp EEG Dataset**
* Source: PhysioNet
* Pediatric EEG recordings with seizure annotations

⚠️ Dataset not included due to size
👉 https://physionet.org/content/chbmit/1.0.0/

---

## ⚙️ Methodology

### 🔹 1. Signal Processing

* EEG loaded using MNE
* Segmented into 5-second windows

### 🔹 2. Feature Extraction

* Time-domain: mean, standard deviation
* Frequency-domain: delta, theta, alpha, beta band power
* Entropy-based features

### 🔹 3. Models

* 🌲 Random Forest (Baseline)
* ⚡ XGBoost (Improved Model)

### 🔹 4. Evaluation Metrics

* Accuracy
* Sensitivity (Recall)

---

## ▶️ How to Run

```id="run1"
pip install -r requirements.txt
python main.py
```

---

## 📈 Results

| Model         | Performance          |
| ------------- | -------------------- |
| Random Forest | Baseline             |
| XGBoost       | Improved performance |

---

## 🎯 Key Contribution

* Improved seizure detection using **feature engineering + XGBoost**
* Focus on **clinically relevant metric (sensitivity)**
* Clean ML pipeline for EEG analysis

---

## 🚀 Future Work

* Hybrid ML + Deep Learning model
* Real-time seizure detection
* Multi-channel spatial analysis

---

## 👩‍💻 Author

**Tanishka Bajpai**

---

