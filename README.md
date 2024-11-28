<img src="https://i0.wp.com/irriverender.blog/wp-content/uploads/2017/11/poli_1.png?resize=1200%2C360&ssl=1" alt="Loan Approval Visualization" width="300">

### Machine Learning Course 2024

# Loan Approval Prediction

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
