
# ==========================================
# XGBOOST WITH CLASS IMBALANCE HANDLING
# ==========================================

# 1. Import libraries
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
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

print("\nClass imbalance information:")
print("Non-claim observations:", negative)
print("Claim observations:", positive)
print("Scale positive weight:", scale_weight)


# 8. Build XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# 9. Train model
print("\nTraining balanced XGBoost model...")

model.fit(X_train, y_train)

print("Training completed!")


# 10. Make predictions
predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# 11. Evaluate model
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


# 12. Confusion matrix
print("\nConfusion Matrix:")

print(confusion_matrix(y_test, predictions))


# 13. ROC-AUC
print("\nROC-AUC Score:")

print(roc_auc_score(y_test, probabilities))


# 14. Precision-Recall AUC
print("\nAverage Precision Score:")

print(average_precision_score(y_test, probabilities))


# 15. Feature importance
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


# 16. Save feature importance
importance.to_csv(
    "xgboost_balanced_importance.csv",
    index=False
)

print("\nFeature importance saved!")

print("\nExperiment 3 completed!")
