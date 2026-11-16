
# ==========================================
# XGBOOST THRESHOLD OPTIMISATION
# ==========================================

import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# 1. Load dataset
df = pd.read_csv("data/cleaned_insurance_claims.csv")


# 2. Create target
df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)


# 3. Separate features and target
X = df.drop(columns=["ClaimNb", "HasClaim", "IDpol"])

y = df["HasClaim"]


# 4. Encode categorical variables
X = pd.get_dummies(X, drop_first=True, dtype=int)


# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. Calculate class weight
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_weight = negative / positive


# 7. Build model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# 8. Train model
print("Training model...")

model.fit(X_train, y_train)


# 9. Get probabilities
probabilities = model.predict_proba(X_test)[:, 1]


# 10. Test different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# 11. Display results
results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON")

print(results_df.to_string(index=False))

print("\nROC-AUC:", roc_auc_score(y_test, probabilities))

print("\nExperiment completed!")
