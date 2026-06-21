# Task 2: End-to-End ML Pipeline for Customer Churn Prediction

**DevelopersHub Corporation — AI/ML Engineering Internship**

---

## Objective

Build a reusable and production-ready machine learning pipeline to predict customer churn using the Scikit-learn Pipeline API.

---

## Dataset

- **Name:** Telco Customer Churn Dataset
- **Source:** GitHub / Kaggle (Telecom Churn CSV)
- **Size:** ~7,000 customer records
- **Target:** Churn (Yes / No)
- **Features:** 19 features including tenure, monthly charges, contract type, payment method, etc.

---

## Methodology / Approach

### Step 1 — Data Loading & Preprocessing
- Loaded the Telco Churn dataset using Pandas
- Dropped irrelevant columns (customerID)
- Encoded all categorical columns using `LabelEncoder`
- Converted `TotalCharges` to numeric and handled missing values with median imputation
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance
- Scaled all features using `StandardScaler`

### Step 2 — Model Development
- Trained 3 models for comparison:
  - **Logistic Regression** — baseline linear model
  - **Random Forest** — ensemble tree-based model
  - **Gradient Boosting** — boosted ensemble model

### Step 3 — Training & Hyperparameter Tuning
- Used `GridSearchCV` for hyperparameter tuning
- Applied cross-validation to prevent overfitting
- Selected the best-performing model based on F1-Score

### Step 4 — Evaluation
- Evaluated all models using **Accuracy**, **F1-Score**, and **AUC-ROC**
- Plotted confusion matrix for the best model
- Visualized top 10 most important features

### Step 5 — Model Export
- Exported the complete pipeline using `joblib` for production reuse

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Scikit-learn | Pipeline, models, evaluation |
| Imbalanced-learn | SMOTE for class balancing |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Visualizations |
| Joblib | Model export |

---

## Key Results & Observations

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| Logistic Regression | ~85% | ~84% | ~85% |
| Random Forest | ~91% | ~90% | ~91% |
| Gradient Boosting | ~90% | ~89% | ~90% |

- **Random Forest** performed best overall
- SMOTE significantly improved recall for the minority (churn) class
- Top features: `tenure`, `MonthlyCharges`, `Contract`, `TotalCharges`
- The exported pipeline can be directly loaded and used for predictions in production

---

## Skills Gained

- ML pipeline construction
- Hyperparameter tuning with GridSearchCV
- Model export and reusability with joblib
- Production-readiness practices

---

## Repository Structure

```
task2_customer_churn/
├── task2_customer_churn.ipynb    # Main notebook
├── churn_pipeline.pkl            # Exported pipeline (joblib)
├── model_comparison.png          # Model comparison chart
├── confusion_matrix.png          # Confusion matrix
├── feature_importance.png        # Feature importance chart
├── README.md                     # This file
```

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn

# Open the notebook
jupyter notebook task2_customer_churn.ipynb
```

---

*Submitted as part of DevelopersHub Corporation AI/ML Engineering Internship — Due: 30th June 2026*
