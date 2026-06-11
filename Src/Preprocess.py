import pandas as pd
import numpy as np

df = pd.read_csv("../Data/diabetic_data.csv")

print("Original shape:", df.shape)

df["readmitted_30_days"] = 0
df.loc[df["readmitted"] == "<30", "readmitted_30_days"] = 1

df = df.replace("?", np.nan)

missing_percent = (df.isnull().sum() / len(df)) * 100

print("\nMissing values percentage:")
print(missing_percent.sort_values(ascending=False).head(10))

print("\nUnique values:")
print("encounter_id unique:", df["encounter_id"].nunique())
print("patient_nbr unique:", df["patient_nbr"].nunique())

drop_cols = [
    "weight",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "encounter_id",
    "patient_nbr",
    "readmitted"
]

df = df.drop(columns=drop_cols)

df.to_csv("../Data/cleaned_data.csv", index=False)

print("\nTarget counts:")
print(df["readmitted_30_days"].value_counts())
