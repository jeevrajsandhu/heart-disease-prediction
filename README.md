# Heart Disease Prediction

A modern Machine Learning project that predicts the risk of heart disease using patient health data and lifestyle information.
This project covers the complete ML workflow from data preprocessing and visualization to model training, evaluation, and deployment-ready model saving.

---

## 🚀 Project Highlights

* Complete end-to-end Machine Learning pipeline
* Data cleaning and preprocessing
* Outlier detection and handling
* Professional Exploratory Data Analysis (EDA)
* Feature engineering and feature selection
* Multiple ML models comparison
* Hyperparameter tuning
* Model evaluation with visualizations
* Saved trained model for deployment

---

## 📂 Dataset Information

The dataset contains medical and lifestyle information such as:

* Age
* Gender
* Height & Weight
* Blood Pressure
* Cholesterol Level
* Glucose Level
* Smoking Habit
* Alcohol Consumption
* Physical Activity

### 🎯 Target Variable

| Value | Meaning          |
| ----- | ---------------- |
| 0     | No Heart Disease |
| 1     | Heart Disease    |

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

## 📊 Machine Learning Workflow

### 1. Data Preprocessing

* Missing value handling
* Duplicate removal
* Outlier detection using IQR & Boxplots
* Data cleaning

### 2. Exploratory Data Analysis

* Correlation heatmaps
* Distribution plots
* Count plots
* Boxplots
* Relationship analysis between features

### 3. Feature Engineering

Created new useful features like:

* BMI
* Age in Years
* Pulse Pressure
* Blood Pressure Category

### 4. Model Training

Models used:

* Logistic Regression
* Random Forest Classifier

### 5. Model Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

## 🏆 Best Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 73.28% |
| Precision | 76.85% |
| Recall    | 65.82% |
| F1 Score  | 70.91% |
| ROC-AUC   | 80.23% |

✅ Best Performing Model: **Tuned Random Forest**

---

## 📈 Key Insights

* Blood pressure is one of the strongest indicators of heart disease.
* Higher cholesterol levels increase cardiovascular risk.
* Age and BMI significantly affect prediction outcomes.
* Feature engineering improved overall model performance.

---

## 📁 Project Structure

```bash
Heart-Disease-Prediction/
│
├── Heart_Disease_Prediction_Beginner_Portfolio.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   └── cardio_train.csv
│
└── models/
    └── heart_disease_best_model.joblib
```

---

## ▶️ How To Run

### 1. Clone Repository

```bash
git clone <your-repository-link>
cd Heart-Disease-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebook

Open:

```bash
Heart_Disease_Prediction_Beginner_Portfolio.ipynb
```

Run all cells using:

* Jupyter Notebook
* VS Code
* Google Colab

---

## 💾 Saved Model

The trained model is saved using Joblib:

```bash
models/heart_disease_best_model.joblib
```

This allows easy deployment without retraining the model.

---

## 👨‍💻 Author

**Jeevraj Sandhu**

Machine Learning & Data Science Enthusiast

--- 