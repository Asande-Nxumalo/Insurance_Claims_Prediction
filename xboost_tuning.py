
# ==========================================
# XGBOOST HYPERPARAMETER TUNING
# ==========================================

# 1. Import libraries
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)


# 2. Load dataset
df = pd.read_csv("data/cleaned_insurance_claims.csv")

print("Dataset loaded successfully!")


# 3. Create target
df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)


# 4. Separate features and target
X = df.drop(columns=["ClaimNb", "HasClaim", "IDpol"])

y = df["HasClaim"]


# 5. Encode categorical variables
X = pd.get_dummies(X, drop_first=True, dtype=int)


# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Calculate class imbalance weight
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_weight = negative / positive


# 8. Define base model
model = XGBClassifier(
    scale_pos_weight=scale_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)


# 9. Define hyperparameter search space
param_grid = {
    "n_estimators": [150, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}


# 10. Define cross-validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# 11. Configure randomized search
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=8,
    scoring="average_precision",
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=2,
    refit=True
)


# 12. Train and search
print("\nStarting hyperparameter tuning...")

search.fit(X_train, y_train)


# 13. Display best parameters
print("\n========== BEST PARAMETERS ==========")

print(search.best_params_)


# 14. Display best score
print("\nBest Cross-Validation Average Precision:")

print(search.best_score_)


# 15. Save experiment results
results = pd.DataFrame(search.cv_results_)

results = results.sort_values(
    by="rank_test_score"
)

results.to_csv(
    "xgboost_tuning_results.csv",
    index=False
)

print("\nTuning results saved!")

print("\nExperiment 5 completed!")
