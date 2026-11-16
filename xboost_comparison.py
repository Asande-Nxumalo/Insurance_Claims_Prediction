
# ==========================================
# EXPERIMENT 6: FINAL MODEL COMPARISON
# ==========================================

# 1. Import libraries
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
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


# 7. Calculate class imbalance weight
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_weight = negative / positive


# ==========================================
# 8. BUILD ORIGINAL MODEL
# ==========================================

original_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# ==========================================
# 9. BUILD TUNED MODEL
# ==========================================

tuned_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=3,
    subsample=1.0,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# ==========================================
# 10. TRAIN BOTH MODELS
# ==========================================

print("\nTraining original model...")

original_model.fit(X_train, y_train)

print("Original model trained!")

print("\nTraining tuned model...")

tuned_model.fit(X_train, y_train)

print("Tuned model trained!")


# ==========================================
# 11. CREATE PREDICTIONS
# ==========================================

original_prob = original_model.predict_proba(X_test)[:, 1]

tuned_prob = tuned_model.predict_proba(X_test)[:, 1]

original_pred = (original_prob >= 0.5).astype(int)

tuned_pred = (tuned_prob >= 0.5).astype(int)


# ==========================================
# 12. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_true, predictions, probabilities):

    return {
        "Model": name,

        "Accuracy": accuracy_score(y_true, predictions),

        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),

        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),

        "F1 Score": f1_score(
            y_true, predictions, zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true, probabilities
        ),

        "Average Precision": average_precision_score(
            y_true, probabilities
        )
    }


# ==========================================
# 13. COMPARE PERFORMANCE
# ==========================================

original_results = evaluate_model(
    "Original XGBoost",
    y_test,
    original_pred,
    original_prob
)

tuned_results = evaluate_model(
    "Tuned XGBoost",
    y_test,
    tuned_pred,
    tuned_prob
)


results = pd.DataFrame([
    original_results,
    tuned_results
])


print("\n========== MODEL COMPARISON ==========")

print(results.to_string(index=False))


# ==========================================
# 14. CONFUSION MATRICES
# ==========================================

print("\nOriginal Model Confusion Matrix:")

print(confusion_matrix(y_test, original_pred))


print("\nTuned Model Confusion Matrix:")

print(confusion_matrix(y_test, tuned_pred))


# ==========================================
# 15. SAVE RESULTS
# ==========================================

results.to_csv(
    "xgboost_final_comparison.csv",
    index=False
)

print("\nComparison results saved!")

print("\nExperiment 6 completed!")
