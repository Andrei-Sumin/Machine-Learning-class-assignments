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
  - Implemented multiple algorithms, including Logistic Regression, Decision Trees, Random Forests, SVM, KNN, Naive Bayes, and Neural Networks.
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
- `README.md`: Project overview and details.
