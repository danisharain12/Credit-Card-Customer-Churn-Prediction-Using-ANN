# Credit Card Customer Churn Prediction using ANN

**Predict whether a customer will leave a bank using a deep learning model built with **TensorFlow/Keras**. This project helps financial institutions reduce churn and improve customer retention strategies by identifying at-risk customers.**

---

## 📌 Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Technologies Used](#technologies-used)  
- [Model Architecture](#model-architecture)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Results](#results)  
- [How to Run](#how-to-run)  
- [Future Improvements](#future-improvements)  
- [Screenshots](#screenshots)

---

## 🔍 Overview

This project uses an **Artificial Neural Network (ANN)** to classify whether a customer will churn (i.e., exit the bank). It involves complete preprocessing, model building, training, evaluation, and visualization.

The ANN was trained on customer demographics and activity data such as:
- Age, Credit Score, Balance  
- Geography, Gender, Tenure  
- Number of Products, Card Ownership, Activity Status

---

## 📂 Dataset

- **Source:** Kaggle / Internal CSV  
- **Size:** 10,000 rows × 14 columns  
- **Target Variable:** `Exited` (1 if the customer churned, 0 otherwise)

---

## 🧪 Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas / NumPy | Data handling |
| Matplotlib | Visualization |
| Scikit-learn | Preprocessing, train-test split |
| TensorFlow / Keras | Deep learning framework |
| StandardScaler | Feature scaling |
| EarlyStopping | Preventing overfitting |

---

## 🧠 Model Architecture

- Input Layer: 11 features
- Dense(64) - ReLU
- Dense(32) - ReLU + Dropout(0.3)
- Dense(16) - ReLU + Dropout(0.4)
- Dense(1) - Sigmoid (Output)

- **Loss Function:** `binary_crossentropy`  
- **Optimizer:** `Adam`  
- **Metrics:** `Accuracy`, `AUC`, `Precision`, `Recall`  
- **Callback:** `EarlyStopping` with `restore_best_weights=True`

---

## 📈 Evaluation Metrics

- **Train Accuracy:** 87.06%  
- **Validation Accuracy:** 85.19%  
- **Test Accuracy:** 85.5%  
- **AUC Score:** 0.88  
- **Precision:** 0.78  
- **Recall:** 0.50

---

## 🚀 How to Run

1. **Clone the repository**

- git clone https://github.com/yourusername/customer-churn-ann.git
- cd customer-churn-ann
- Install dependencies
- pip install -r requirements.txt
- Run the script

- Or run the notebook:
- jupyter notebook Credit_Card_Churn_Prediction.ipynb

## 📊 Screenshots
Training Progress	Metrics Visualization
![image](https://github.com/user-attachments/assets/4976ab57-a0c8-479d-a186-6c575847fdfc)



## 🔧 Future Improvements
- Try SMOTE or class weighting to handle class imbalance
- Fine-tune the prediction threshold for better recall
- Deploy as a Streamlit or Flask web app
- Try model ensembling (e.g., XGBoost + ANN)

## 📬 Contact
- Danish Karim
- danisharain253@gmail.com


### 🌟 Star this repo if you found it useful!
