# EEG Seizure Detection using Machine Learning

## 📌 Project Description

This project focuses on detecting epileptic seizures from EEG signals using machine learning techniques. The approach involves signal segmentation, feature extraction from EEG data, and classification using traditional ML models.

A baseline model using Random Forest is implemented, followed by an improved model using XGBoost with enhanced feature engineering. The goal is to improve seizure detection performance, particularly sensitivity, which is critical in clinical applications.

---

## 📊 Dataset

* Dataset used: **CHB-MIT Scalp EEG Dataset**
* Source: PhysioNet
* Contains EEG recordings from pediatric patients with annotated seizure events

⚠️ Note:
The dataset is not included in this repository due to size.
Download it from: https://physionet.org/content/chbmit/1.0.0/

---

## ⚙️ Methodology

1. **EEG Signal Loading**

   * EEG data is loaded from `.edf` files using the MNE library

2. **Segmentation**

   * Signals are divided into fixed-length windows (5 seconds)

3. **Feature Extraction**

   * Time-domain features: mean, standard deviation
   * Frequency-domain features: delta, theta, alpha, beta band power
   * Entropy-based features for improved representation

4. **Model Training**

   * Baseline model: Random Forest
   * Improved model: XGBoost

5. **Evaluation**

   * Accuracy
   * Sensitivity (Recall)

---

## ▶️ How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Place EEG `.edf` files in the project directory

3. Run the main script:

```
python main.py
```

---

## 📈 Results

| Model         | Description                                 |
| ------------- | ------------------------------------------- |
| Random Forest | Baseline ML model                           |
| XGBoost       | Improved model with better feature handling |

The improved model demonstrates better performance compared to the baseline, especially in terms of sensitivity.

---

## 🎯 Conclusion

This project demonstrates that incorporating better feature engineering and advanced ML models can significantly improve EEG-based seizure detection. Future work can explore hybrid ML-DL approaches for further enhancement.

---

## 👩‍💻 Author

Tanishka Bajpai

---
