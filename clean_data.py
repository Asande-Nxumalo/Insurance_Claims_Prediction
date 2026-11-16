
import pandas as pd

# 1. Load original dataset
df = pd.read_csv("data/insurance_claims.csv")

original_shape = df.shape

print("Original shape:", original_shape)

# 2. Remove duplicate observations
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# 3. Remove observations without a valid target
df = df.dropna(subset=["ClaimNb", "Exposure"])

# 4. Remove policies with zero or negative exposure
df = df[df["Exposure"] > 0].copy()

print("After checking exposure:", df.shape)

# 5. Create target variable
df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)

# 6. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 7. Save cleaned dataset
df.to_csv(
    "data/cleaned_insurance_claims.csv",
    index=False
)

print("\nCleaning completed!")

print("Original shape:", original_shape)
print("Final shape:", df.shape)

print("\nClaim distribution:")
print(df["HasClaim"].value_counts())
