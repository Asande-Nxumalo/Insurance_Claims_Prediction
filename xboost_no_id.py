
# ==========================================
# XGBOOST MODEL WITHOUT POLICY ID
# ==========================================

# 1. Import libraries
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# 2. Load dataset
df = pd.read_csv("data/cleaned_insurance_claims.csv")

print("Dataset loaded successfully!")


# 3. Create target variable
df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)


# 4. Separate features and target
X = df.drop(columns=["ClaimNb", "HasClaim", "IDpol"])

y = df["HasClaim"]


# 5. Encode categorical variables
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("Features prepared successfully!")
print("Number of features:", X.shape[1])


# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Build XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# 8. Train model
print("\nTraining XGBoost without IDpol...")

model.fit(X_train, y_train)

print("Training completed!")


# 9. Make predictions
predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# 10. Evaluate performance
print("\n========== MODEL PERFORMANCE ==========")

print(
    "Accuracy:",
    accuracy_score(y_test, predictions)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# 11. Confusion matrix
print("\nConfusion Matrix:")

print(confusion_matrix(y_test, predictions))


# 12. ROC-AUC
auc = roc_auc_score(y_test, probabilities)

print("\nROC-AUC Score:", auc)


# 13. Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")

print(importance.head(15).to_string(index=False))


# 14. Save feature importance
importance.to_csv(
    "xgboost_no_id_importance.csv",
    index=False
)

print("\nFeature importance saved!")

print("\nExperiment 2 completed!")
