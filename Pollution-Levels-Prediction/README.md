<img src="https://i0.wp.com/irriverender.blog/wp-content/uploads/2017/11/poli_1.png?resize=1200%2C360&ssl=1" alt="Loan Approval Visualization" width="300">

# Pollution Levels Prediction


## Overview
This project is a **university course assignment** focusing on the prediction of carbon monoxide (CO) concentrations using temporal, climatic, and environmental data. The goal is to implement regression techniques to predict CO levels and evaluate the performance based on Mean Absolute Error (MAE).

## Features
- **Exploratory Data Analysis (EDA):** Analyzed the dataset using visualizations, summary statistics, and correlation analysis.
- **Data Preprocessing:** Addressed missing data, performed feature transformations (e.g., logarithmic scaling), and standardized numerical variables.
- **Model Training and Evaluation:**
  - Applied multiple regression algorithms.
  - Conducted hyperparameter tuning using GridSearchCV.
  - Evaluated models using metrics such as MAE, MSE, RMSE, and R².
- **Model Interpretation:** Employed SHAP (SHapley Additive exPlanations) to interpret feature importance.

## Data
The dataset contains 14,000 entries with the following features:
- **Categorical Features:** Hour, day, month, year, and wind direction.
- **Numerical Features:** Fine and medium particulate matter (`small_part`, `med_part`), sulfur dioxide (`sulf_diox`), nitrogen dioxide (`nitr_diox`), ozone (`trioxygen`), temperature, pressure, rainfall, and wind speed.
- **Target Variable:** Carbon monoxide concentration (`carb_monox`) in µg/m³.

## Key Insights
   - Gradient Boosting achieved the lowest MAE (303.240), followed by Random Forest (304.494).

## Project Structure
- `Pollution_Levels_Prediction.ipynb`: Main notebook containing code and analysis.
- `pollution.csv`: Dataset file.
- `gradient_boost_model.pkl`: Saved Gradient Boosting model for deployment.
- `scaler.pkl`: Scaler used for data normalization.

## Short Review of Regression Algorithms Used
This section provides a brief description of the regression models explored for predicting CO levels:

1. **Linear Regression:** A basic regression model for estimating relationships between features and target.
2. **Ridge Regression:** Regularization technique to prevent overfitting by penalizing large coefficients.
3. **Lasso Regression:** Enforces sparsity by shrinking some coefficients to zero using L1 regularization.
4. **K-Nearest Neighbors (KNN):** Non-parametric algorithm predicting values based on nearest neighbors.
5. **Decision Tree Regression:** Tree-based model that partitions data based on feature splits.
6. **Support Vector Regression (SVR):** Finds a hyperplane to best fit the data points with a margin of tolerance.
7. **Multilayer Perceptron (MLP):** Neural network model with customizable hidden layers and activation functions.
8. **Random Forest Regression:** Ensemble of decision trees using random sampling for improved robustness.
9. **AdaBoost Regression:** Boosting method that combines weak regressors iteratively to reduce error.
10. **Gradient Boosting Regression:** Sequential boosting approach optimizing for error minimization.

## How to Use the Model
To make predictions with the pre-trained Gradient Boosting model:

```
# Import necessary libraries
import pickle
import pandas as pd

# Load pre-trained model and scaler
model = pickle.load(open('gradient_boost_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Preprocess new data (ensure it matches the format of the training data)
# First, encode categorical data and log transform the required variable
# Second, apply the standard scaler
data_scaled = scaler.transform(new_data)

# Predict
predictions = model.predict(data_scaled)
```
