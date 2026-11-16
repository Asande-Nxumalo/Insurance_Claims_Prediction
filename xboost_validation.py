
# ==========================================
# XGBOOST STRATIFIED CROSS-VALIDATION
# ==========================================

# 1. Import libraries
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)


# 2. Load cleaned dataset
df = pd.read_csv("data/cleaned_insurance_claims.csv")

print("Dataset loaded successfully!")


# 3. Create target variable
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


# 7. Create 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 8. Store evaluation results
roc_scores = []
pr_scores = []


# 9. Start cross-validation
print("\nStarting 5-Fold Cross-Validation...")

for fold, (train_index, val_index) in enumerate(
    cv.split(X_train, y_train), start=1
):

    print(f"\n========== FOLD {fold} ==========")

    # Separate training and validation data
    X_fold_train = X_train.iloc[train_index]
    X_fold_val = X_train.iloc[val_index]

    y_fold_train = y_train.iloc[train_index]
    y_fold_val = y_train.iloc[val_index]

    # Calculate class imbalance weight
    negative = (y_fold_train == 0).sum()
    positive = (y_fold_train == 1).sum()

    scale_weight = negative / positive

    # Build XGBoost model
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    # Train model
    model.fit(X_fold_train, y_fold_train)

    # Predict probabilities
    probabilities = model.predict_proba(X_fold_val)[:, 1]

    # Evaluate model
    roc = roc_auc_score(y_fold_val, probabilities)

    pr = average_precision_score(y_fold_val, probabilities)

    # Store scores
    roc_scores.append(roc)
    pr_scores.append(pr)

    print("ROC-AUC:", round(roc, 4))
    print("Average Precision:", round(pr, 4))


# 10. Calculate average performance
print("\n========== FINAL CROSS-VALIDATION RESULTS ==========")

print("Average ROC-AUC:", round(sum(roc_scores) / len(roc_scores), 4))

print("Average PR-AUC:", round(sum(pr_scores) / len(pr_scores), 4))

print("ROC-AUC Standard Deviation:", round(pd.Series(roc_scores).std(), 4))

print("PR-AUC Standard Deviation:", round(pd.Series(pr_scores).std(), 4))


print("\nCross-validation completed!")
