
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("data/insurance_claims.csv")

# 2. Understand dataset
print("Dataset shape:", df.shape)
df.info()

print("\nStatistical summary:")
print(df.describe())

# 3. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Create target variable
df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)

# 5. Claim distribution
print("\nClaim counts:")
print(df["HasClaim"].value_counts())

print("\nClaim percentages:")
print(df["HasClaim"].value_counts(normalize=True) * 100)

# 6. Visualise claims
sns.countplot(data=df, x="HasClaim")
plt.title("Distribution of Insurance Claims")
plt.show()

# 7. Analyse claims by driver age
age_claims = df.groupby("DrivAge")["HasClaim"].mean()

age_claims.plot(figsize=(10, 5))
plt.title("Claim Probability by Driver Age")
plt.xlabel("Driver Age")
plt.ylabel("Claim Proportion")
plt.show()

# 8. Calculate overall claim frequency
total_claims = df["ClaimNb"].sum()
total_exposure = df["Exposure"].sum()

claim_frequency = total_claims / total_exposure

print("\nOverall claim frequency:", claim_frequency)
