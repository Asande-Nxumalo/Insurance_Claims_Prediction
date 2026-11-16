
import pandas as pd

from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/cleaned_insurance_claims.csv")

# Select input variables
features = [
    "Exposure",
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region"
]

X = df[features]

# Select target variable
y = df["HasClaim"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display dataset sizes
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Display claim proportions
print("Training claim proportions:")
print(y_train.value_counts(normalize=True))

print("Testing claim proportions:")
print(y_test.value_counts(normalize=True))
