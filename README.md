<img src="https://i0.wp.com/irriverender.blog/wp-content/uploads/2017/11/poli_1.png?resize=1200%2C360&ssl=1" alt="Loan Approval Visualization" width="300">

### Machine Learning Course 2024

This repository contains 2 projects:
- [**Loan Approval Prediction**](https://github.com/Andrei-Sumin/Machine-Learning-class-assignments/tree/main/Loan-Approval-Prediction) - an assignment on classification.
- [**Pollution Levels Prediction**](https://github.com/Andrei-Sumin/Machine-Learning-class-assignments/tree/main/Pollution-Levels-Prediction) - an assignment on regression.

---

# Loan Approval Prediction (Classification)

## Overview
This is a **university course assignment** that explores the use of different machine learning algorithms to predict loan approval outcomes based on provided dataset. The objective is to analyze the factors influencing loan approvals and build a predictive model.


## Features
- **Exploratory Data Analysis (EDA):** Understanding the dataset through visualizations and summary statistics.
- **Data Preprocessing:** Handling missing data, feature transformation, and scaling.
- **Model Training and Evaluation:** 
  - Implemented multiple algorithms
  - Compared model performance using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
- **Cost Optimization:** Incorporated a cost-based approach to optimize the classification threshold, balancing the financial risks of false positives and false negatives.


## Data
The dataset contains 614 entries with the following features:
- **Categorical Features:** Gender, Married, Dependents, Education, Self_Employed, Property_Area.
- **Numerical Features:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History.
- **Target Variable:** Loan_Status (approved or not).


## Key Insights
1. **EDA Findings:**
   - Credit history is a strong predictor of loan approval.
   - Applicant and co-applicant incomes have some influence, but their distribution required transformations for better model performance.
2. **Model Performance:**
   - Random Forest achieved the highest AUC (0.79) and performed best in terms of precision-recall balance. Additionally, cost optimization identified Random Forest as the model with the lowest financial risk under the given assumptions; however, this method requires a reevaluation of those assumptions for more accurate conclusions.


## Project Structure
- `Loan_Approval_Prediction.ipynb`: Main notebook containing code and analysis.
- `model.pkl`: Saved Random Forest model for deployment.
- `scaler.pkl`: Scaler used for feature normalization.
- `dataset.csv`: A dataset.


## Short review of machine learning algorithms used for classification
This section provides a brief description of the machine learning models used in this project to predict loan approvals:

1. **K-Nearest Neighbors (KNN)**: A non-parametric algorithm that classifies based on the majority class of the k-nearest points.
2. **Decision Tree**: A tree-based model that splits data into subsets based on feature values.
3. **Random Forest**: An ensemble of decision trees trained on random subsets of data and features.
4. **AdaBoost**: An ensemble method that combines weak classifiers (e.g., decision trees) iteratively.
5. **Bagging**: An ensemble method that builds multiple models using random subsets of data.
6. **Naive Bayes**: A probabilistic model based on Bayes' theorem and the assumption of feature independence.
7. **Logistic Regression**: A linear model that predicts probabilities using the logistic function.
8. **Support Vector Machine (SVM)**: A model that finds the hyperplane that best separates classes in the feature space.
9. **Neural Network (MLP - Multi-Layer Perceptron)**: A deep learning model that mimics the structure of biological neurons.


## How to use pre-trained model

```
# Load required libraries
import pickle

# Load model and scaler from .pkl files
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Scale the data
data_scaled = scaler.transform(data)

# Predict
predictions = model.predict(data_scaled)
print(predictions)
```

---

# Pollution Levels Prediction (Classification)


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
- `assignment3_sumin.ipynb`: Main notebook containing code and analysis.
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


<br> 
<br>

### License:
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
