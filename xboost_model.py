
# ==========================================
# INSURANCE CLAIMS PREDICTION USING XGBOOST
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

# 2. Load the dataset
df = pd.read_csv("data/cleaned_insurance_claims.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# 3. Create the target variable
df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)


# 4. Separate features and target
X = df.drop(columns=["ClaimNb", "HasClaim"])

y = df["HasClaim"]


# 5. Convert categorical variables into numerical variables
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("\nFeatures prepared successfully!")
print("Number of features:", X.shape[1])


# 6. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData split completed!")
print("Training observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])


# 7. Build the XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# 8. Train the model
print("\nTraining XGBoost model...")

model.fit(X_train, y_train)

print("Model training completed!")


# 9. Make predictions
predictions = model.predict(X_test)

# Get predicted probabilities of making a claim
probabilities = model.predict_proba(X_test)[:, 1]


# 10. Evaluate model performance
print("\n========== MODEL PERFORMANCE ==========")

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

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


# 12. ROC-AUC score
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
importance.to_csv("xgboost_feature_importance.csv", index=False)

print("\nFeature importance saved successfully!")

print("\nXGBoost project completed!")
