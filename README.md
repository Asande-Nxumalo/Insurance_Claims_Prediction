# Insurance_Claims_Prediction
# Insurance Claims Prediction System

## Project Overview

This project uses machine learning to predict whether an insurance policy will generate at least one claim.

The objective is to explore how statistical learning and machine learning techniques can support insurance risk assessment and claims analysis.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Machine Learning Techniques

The project explores:

- Logistic Regression
- Random Forest
- XGBoost
- Class imbalance handling
- Classification threshold analysis
- Hyperparameter tuning
- Model performance comparison

## Project Methodology

1. Load and explore the insurance dataset.
2. Clean and prepare the data.
3. Create a binary target variable called `HasClaim`.
4. Encode categorical variables.
5. Split the data into training and testing sets.
6. Train machine learning models.
7. Evaluate predictive performance.
8. Compare baseline and tuned XGBoost models.

## Latest XGBoost Results

The tuned XGBoost model achieved the following results on the test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 65.01% |
| Precision | 8.87% |
| Recall | 64.31% |
| F1 Score | 15.59% |
| ROC-AUC | 70.85% |
| Average Precision | 13.95% |

## Key Findings

The tuned XGBoost model showed a modest improvement over the original model.

The results also demonstrate the challenges of predicting insurance claims in an imbalanced dataset.

In particular, the model achieved relatively high recall but low precision, highlighting the trade-off between identifying claims and generating false positives.

## Limitations

- The dataset contains an imbalanced target variable.
- Predictive performance depends on the selected classification threshold.
- The model predicts claim occurrence, not claim severity or fraudulent activity.
- Further validation is necessary before any real-world insurance deployment.

## Author

BSc Actuarial Science Student(Asande Nxumalo)

Interested in actuarial modelling, insurance analytics, machine learning and financial risk management.

## Disclaimer

This project is for educational and portfolio purposes. The model is not intended for direct use in insurance underwriting or claims decisions.