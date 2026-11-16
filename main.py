import pandas as pd
#Load insurance data
df=pd.read_csv("data/insurance_claims.csv")
#display first five records
print(df.head())
#Display dataset iformation
print(df.info())
#Display dataset dimensions
print(df.shape)

