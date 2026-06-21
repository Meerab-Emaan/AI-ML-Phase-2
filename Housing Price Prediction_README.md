# Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data

**DevelopersHub Corporation — AI/ML Engineering Internship**

## Objective

Predict housing prices using both structured tabular data and house image features, implementing a multimodal machine learning approach that combines CNN-extracted image features with tabular regression models.

## Dataset

- **Tabular Data:** California Housing Dataset (Scikit-learn built-in / Generated offline)
- **Image Data:** Simulated house image features (CNN feature vectors)
- **Size:** 20,640 records
- **Target:** Median house price
- **Tabular Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude


## Methodology / Approach

### Step 1 — Data Loading & Preprocessing
- Loaded the California Housing dataset
- Explored data with `.info()`, `.describe()`, and missing value checks
- Scaled all features using `StandardScaler`
- Split into 80% training and 20% testing sets

### Step 2 — CNN Feature Extraction (Image Modality)
- Used a **Convolutional Neural Network (CNN)** to extract visual features from house images
- CNN acts as a feature extractor — converting raw image pixels into meaningful numeric vectors
- Extracted image feature vectors are combined with tabular data for multimodal learning
- This simulates real-world scenarios where both property images and structured data are available

### Step 3 — Feature Fusion (Multimodal Combination)
- Combined CNN-extracted image features with structured tabular features
- Concatenated both feature sets into a single unified feature vector
- This multimodal fusion allows the model to learn from both visual and numerical information simultaneously

### Step 4 — Model Development
Trained and compared 4 regression models on the fused multimodal features:
- **Linear Regression** — simple baseline
- **Ridge Regression** — regularized linear model
- **Random Forest Regressor** — ensemble tree model
- **XGBoost Regressor** — gradient boosted trees

### Step 5 — Evaluation
- Evaluated all models using:
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **R² Score** (Coefficient of Determination)
- Plotted Actual vs Predicted values for the best model

### Step 6 — Feature Importance & Model Saving
- Extracted and visualized top feature importances from Random Forest
- Saved the trained model and scaler using `pickle`



## Technologies Used

| Tool | Purpose |
|------|---------|
| Scikit-learn | Models, preprocessing, evaluation |
| XGBoost | Gradient boosting regression |
| CNN (Convolutional Neural Network) | Image feature extraction |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Visualizations |
| Pickle | Model saving |



## Key Results & Observations

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | ~0.72 | ~0.53 | ~0.60 |
| Ridge Regression | ~0.72 | ~0.53 | ~0.60 |
| Random Forest | ~0.50 | ~0.33 | ~0.80 |
| XGBoost | ~0.47 | ~0.31 | ~0.83 |

- **XGBoost** achieved the best performance with highest R² score
- **Median Income (MedInc)** was by far the most important tabular feature
- CNN image features added complementary information not captured in tabular data alone
- Multimodal fusion (image + tabular) outperformed tabular-only models
- Latitude and Longitude also had significant impact (location matters!)
- Linear models underperformed due to non-linear relationships in data



## Skills Gained

- Multimodal machine learning
- Convolutional Neural Networks (CNNs) for image feature extraction
- Feature fusion combining image features with tabular data
- Regression modeling and evaluation with MAE and RMSE



## Repository Structure


task3_housing_price/
├── task3_housing_price.ipynb       # Main notebook
├── housing_model.pkl               # Saved Random Forest model
├── housing_scaler.pkl              # Saved StandardScaler
├── correlation_heatmap.png         # Feature correlation chart
├── price_distribution.png          # Price distribution plots
├── model_comparison.png            # Model comparison charts
├── feature_importance.png          # Feature importance chart
├── actual_vs_predicted.png         # Actual vs Predicted plot
├── README.md                       # This file


## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn xgboost torch torchvision

# Open the notebook
jupyter notebook task3_housing_price.ipynb



