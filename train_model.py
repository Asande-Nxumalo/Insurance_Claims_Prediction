
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

# 1. Load dataset
df = pd.read_csv("data/cleaned_insurance_claims.csv")

# 2. Select features
features = [
    "Exposure", "Area", "VehPower", "VehAge",
    "DrivAge", "BonusMalus", "VehBrand",
    "VehGas", "Density", "Region"
]

X = df[features]
y = df["HasClaim"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Identify column types
numeric_features = [
    "Exposure", "VehPower", "VehAge",
    "DrivAge", "BonusMalus", "Density"
]

categorical_features = [
    "Area", "VehBrand", "VehGas", "Region"
]

# 5. Prepare numerical data
numeric_transformer = SimpleImputer(
    strategy="median"
)

# 6. Prepare categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# 7. Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 8. Create Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# 9. Create complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)

# 10. Train
pipeline.fit(X_train, y_train)

# 11. Predict
# Get predicted probabilities
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Choose a prediction threshold
threshold = 0.5

# Convert probabilities into predictions
y_pred = (y_prob >= threshold).astype(int)

# 12. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("ROC-AUC:", roc_auc_score(
    y_test,
    pipeline.predict_proba(X_test)[:, 1]
))

# 13. Feature importance analysis

importances = pipeline.named_steps["model"].feature_importances_

feature_names = (
    pipeline.named_steps["preprocessing"]
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort features by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display the top 10 features
print("\nTop 10 Important Features:")

print(importance_df.head(10))

# 14. Visualise feature importance

import matplotlib.pyplot as plt

top_features = importance_df.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 10 Random Forest Feature Importances")

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.tight_layout()

plt.show()
