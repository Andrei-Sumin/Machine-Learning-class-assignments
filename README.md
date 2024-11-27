<img src="https://i0.wp.com/irriverender.blog/wp-content/uploads/2017/11/poli_1.png?resize=1200%2C360&ssl=1" alt="Loan Approval Visualization" width="300">

## Machine Learning Course 2024

# Loan Approval Prediction

## Overview
This is a **university course assignment** that explores the use of different machine learning algorithms to predict loan approval outcomes based on provided dataset. The objective is to analyze the factors influencing loan approvals and build a predictive model.

---

## Features
- **Exploratory Data Analysis (EDA):** Understanding the dataset through visualizations and summary statistics.
- **Data Preprocessing:** Handling missing data, feature transformation, and scaling.
- **Model Training and Evaluation:** 
  - Implemented multiple algorithms
  - Compared model performance using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
- **Cost Optimization:** Incorporated a cost-based approach to optimize the classification threshold, balancing the financial risks of false positives and false negatives.

---

## Data
The dataset contains 614 entries with the following features:
- **Categorical Features:** Gender, Married, Dependents, Education, Self_Employed, Property_Area.
- **Numerical Features:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History.
- **Target Variable:** Loan_Status (approved or not).

---

## Key Insights
1. **EDA Findings:**
   - Credit history is a strong predictor of loan approval.
   - Applicant and co-applicant incomes have some influence, but their distribution required transformations for better model performance.
2. **Model Performance:**
   - Random Forest achieved the highest AUC (0.79) and performed best in terms of precision-recall balance.
3. **Cost Analysis:**
   - Cost optimization identified Random Forest as the model with the lowest financial risk under given assumptions.

---

## Project Structure
- `Loan_Approval_Prediction.ipynb`: Main notebook containing code and analysis.
- `model.pkl`: Saved Random Forest model for deployment.
- `scaler.pkl`: Scaler used for feature normalization.
- `dataset.csv`: (Placeholder) Instructions for accessing the dataset or use a mock dataset.

---

## Short review of machine learning algorithms used
This section provides a brief description of the machine learning models used in this project to predict loan approvals:

### 1. K-Nearest Neighbors (KNN)
- A non-parametric algorithm that classifies based on the majority class of the k-nearest points.
- **Strengths**: Simple and effective for smaller datasets.
- **Weaknesses**: Computationally expensive for large datasets.

### 2. Decision Tree
- A tree-based model that splits data into subsets based on feature values.
- **Strengths**: Easy to visualize and interpret; captures non-linear relationships.
- **Weaknesses**: Prone to overfitting without constraints like max depth or minimum samples.

### 3. Random Forest
- An ensemble of decision trees trained on random subsets of data and features.
- **Strengths**: Reduces overfitting and improves generalization; robust to noise.
- **Weaknesses**: Less interpretable than a single decision tree.

### 4. AdaBoost
- An ensemble method that combines weak classifiers (e.g., decision trees) iteratively.
- **Strengths**: Focuses on difficult-to-classify samples; reduces bias.
- **Weaknesses**: Sensitive to noisy data and outliers.

### 5. Bagging (Bootstrap Aggregation)
- An ensemble method that builds multiple models using random subsets of data.
- **Strengths**: Reduces variance and prevents overfitting.
- **Weaknesses**: Less effective when base models are already strong learners.

### 6. Naive Bayes
- A probabilistic model based on Bayes' theorem and the assumption of feature independence.
- **Strengths**: Fast and simple; works well with small datasets and categorical data.
- **Weaknesses**: Assumes independence between features, which may not hold true.

### 7. Logistic Regression
- A linear model that predicts probabilities using the logistic function.
- **Strengths**: Simple, interpretable, and effective for binary classification.
- **Weaknesses**: Assumes a linear relationship between features and the log-odds of the target variable.

### 8. Support Vector Machine (SVM)
- A model that finds the hyperplane that best separates classes in the feature space.
- **Strengths**: Effective in high-dimensional spaces; robust to overfitting with proper kernel.
- **Weaknesses**: Computationally intensive; requires careful tuning of hyperparameters.

### 9. Neural Network (MLP - Multi-Layer Perceptron)
- A deep learning model that mimics the structure of biological neurons.
- **Strengths**: Capable of learning complex patterns and relationships.
- **Weaknesses**: Requires large datasets and computational resources; less interpretable.
