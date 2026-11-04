
import pandas as pd
import numpy as np
import os

# Step 1: Read Dataset
df = pd.read_csv("dataset/student_performance.csv")

print("Original Dataset Shape:", df.shape)

# Step 2: Check Missing Values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Step 3: Check Duplicate Values
print("\nDuplicate Values Before Cleaning:")
print(df.duplicated().sum())

# Step 4: Remove Duplicate Rows
df = df.drop_duplicates()

# Step 5: Handle Missing Values
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Step 6: Check Missing Values Again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Step 7: Check Duplicate Values Again
print("\nDuplicate Values After Cleaning:")
print(df.duplicated().sum())

# Step 8: Save Cleaned Dataset
os.makedirs("dataset", exist_ok=True)

df.to_csv("dataset/student_performance_cleaned.csv", index=False)

# Step 9: Display Final Information
print("\nCleaned Dataset Shape:", df.shape)

print("\nData Cleaning Completed Successfully!")

print("\nFirst 5 Rows:")
print(df.head())
