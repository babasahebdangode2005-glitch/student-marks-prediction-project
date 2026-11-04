
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read Dataset
df = pd.read_csv("dataset/student_performance.csv")

# Step 2: Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Step 3: Display Last 5 Rows
print("\nLast 5 Rows:")
print(df.tail())

# Step 4: Check Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Step 5: Check Dataset Size
print("\nDataset Size:")
print(df.size)

# Step 6: Display Column Names
print("\nDataset Columns:")
print(df.columns)

# Step 7: Check Data Types
print("\nData Types:")
print(df.dtypes)

# Step 8: Dataset Information
print("\nDataset Information:")
df.info()

# Step 9: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 10: Check Duplicate Values
print("\nDuplicate Values:")
print(df.duplicated().sum())

# Step 11: Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Step 12: Correlation
print("\nCorrelation:")
print(df.corr(numeric_only=True))

# Step 13: Histogram
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Step 14: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Student Performance Correlation")
plt.tight_layout()
plt.show()

print("\nEDA Completed Successfully!")
