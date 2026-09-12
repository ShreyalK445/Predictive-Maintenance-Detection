# ⚙️ Predictive Maintenance Detection

A machine learning project for predicting the possibility of machine failure using operating and sensor-related parameters.

The project uses the **AI4I 2020 Predictive Maintenance dataset** and focuses on data cleaning, exploratory analysis, feature engineering, machine learning model comparison, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Unexpected machine failures can lead to production downtime, maintenance costs, and loss of productivity.

This project aims to identify machine operating conditions that are associated with failure and build a classification model that estimates whether a machine is likely to fail.

The project also includes an interactive dashboard where users can:

* Explore machine failure patterns
* Filter machines by type and status
* Analyze sensor relationships
* View failure rates
* Enter machine readings manually
* Get a machine failure prediction

---

## 🎯 Objectives

The main objectives of this project are:

1. Clean and prepare the machine dataset.
2. Handle duplicate, missing, and inconsistent data.
3. Explore relationships between machine operating parameters and failures.
4. Create useful derived features.
5. Compare multiple machine learning classification algorithms.
6. Select a suitable model using F1-score based evaluation.
7. Save the trained model for deployment.
8. Build an interactive Streamlit dashboard for analysis and prediction.

---

## 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains machine operating information such as:

* Machine type
* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear
* Machine failure

The target variable is:

```text
Machine failure
```

where:

```text
0 → No machine failure
1 → Machine failure
```

The dataset contains a highly imbalanced target, with machine failures representing a much smaller portion of the observations than normal operating conditions.

---

## 🧹 Data Preparation

The following preprocessing steps were performed:

### Duplicate Removal

Duplicate records were identified and removed before model training.

### Machine Type Cleaning

The original machine type values contained inconsistent representations such as:

```text
L
l
" L "
M
m
medium
H
h
" H "
high
```

These were standardized into:

```text
L
M
H
```

### Missing Values

Missing sensor and categorical values were handled through preprocessing pipelines.

For numerical variables, median imputation was used.

For the categorical machine type variable, the most frequent value was used when required.

### Identifier Removal

The following columns were not used as predictive features:

```text
UDI
Product ID
```

These columns act as identifiers rather than useful machine-condition measurements.

### Leakage Prevention

`TWF` was excluded from the model features because it represents a failure-related indicator and could leak information about the target `Machine failure`.

---

## 🧠 Feature Engineering

Three additional features were created from the available machine measurements.

### 1. Temperature Difference

```text
Temperature Difference =
Process temperature - Air temperature
```

This represents the difference between process and surrounding air temperature.

### 2. Mechanical Load Proxy

```text
Mechanical Load Proxy =
Torque × Rotational speed
```

This is used as a combined indicator of mechanical operating load.

It is treated as a **proxy**, rather than claiming it is a direct measurement of physical power.

### 3. Torque Speed Ratio

```text
Torque Speed Ratio =
Torque / (Rotational speed + 1)
```

The `+1` prevents division by zero.

---

## 🧪 Features Used for Training

The final model uses:

```text
Type
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Temperature Difference
Mechanical Load Proxy
Torque Speed Ratio
```

---

## 🤖 Machine Learning Models

Several classification models were compared:

* Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting

The models use preprocessing pipelines so that numerical and categorical variables are handled consistently.

Class weighting was used where appropriate because machine failure is an imbalanced classification problem.

---

## 📏 Model Evaluation

The project evaluates the models using classification metrics including:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion Matrix

### Why F1-score?

Machine failure is an imbalanced classification problem.

Accuracy alone can be misleading when most observations belong to the normal class.

F1-score provides a balance between:

```text
Precision + Recall
```

which makes it useful when both missed failures and false alarms matter.

---

## 💻 Streamlit Dashboard

The trained model is integrated into an interactive Streamlit application.

The dashboard contains:

### Dashboard KPIs

* Total machines
* Number of failures
* Number of normal machines
* Failure rate

### Interactive Filters

Users can filter the dataset by:

* Machine type
* Machine status

### Visual Analysis

The dashboard provides visualizations such as:

* Machine failure distribution
* Failure rate by machine type
* Tool wear vs torque
* Rotational speed vs torque
* Air temperature vs process temperature

### Prediction Studio

Users can enter:

```text
Machine Type
Air Temperature
Process Temperature
Rotational Speed
Torque
Tool Wear
```

The application calculates the engineered features automatically and sends the values to the trained model.

The result shows:

```text
Failure Probability
Prediction
```

---

## 🗂️ Project Structure

```text
Predictive-Maintenance-Detection/
│
├── app.py
│
├── cleaned_predictive_maintenance.csv
│
├── final_model.pkl
│
├── requirements.txt
│
├── runtime.txt
│
└── README.md
```

### File Description

| File                                 | Purpose                                      |
| ------------------------------------ | -------------------------------------------- |
| `app.py`                             | Streamlit dashboard and prediction interface |
| `cleaned_predictive_maintenance.csv` | Cleaned dataset used by the dashboard        |
| `final_model.pkl`                    | Saved trained machine learning model         |
| `requirements.txt`                   | Python dependencies                          |
| `runtime.txt`                        | Python runtime configuration                 |
| `README.md`                          | Project documentation                        |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Plotly
* Streamlit
* Google Colab
* GitHub

---

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Duplicate Removal
     ↓
Missing Value Handling
     ↓
Feature Engineering
     ↓
Exploratory Data Analysis
     ↓
Train/Test Split
     ↓
Model Comparison
     ↓
Model Evaluation
     ↓
Final Model
     ↓
Model Serialization
     ↓
Streamlit Dashboard
     ↓
Machine Failure Prediction
```

---

## 🚀 Running the Streamlit Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 📦 Model Deployment

The trained model is stored as:

```text
final_model.pkl
```

The saved model package contains the trained model along with information about:

* Model name
* Input features
* Target variable
* Decision threshold

The Streamlit application loads this file and uses the same preprocessing and prediction pipeline used during model development.

---
##Streamlit Link
https://predictive-maintenance-detection-wbfmkgkp7mvwn4uzxazxrx.streamlit.app/


## 🔗 Project Repository

GitHub:

https://github.com/ShreyalK445/Predictive-Maintenance-Detection
