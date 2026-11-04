
import pandas as pd
import numpy as np

# Step 1: Read Cleaned Dataset
df = pd.read_csv("dataset/student_performance_cleaned.csv")

print("Original Dataset:")
print(df.head())

# Step 2: Create New Feature
df["Study_Attendance_Index"] = (
    df["Study_Hours"] * df["Attendance"] / 100
)

# Step 3: Check New Feature
print("\nNew Feature Created:")
print(df[[
    "Study_Hours",
    "Attendance",
    "Study_Attendance_Index"
]].head())

# Step 4: Check Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Step 5: Check Dataset Columns
print("\nDataset Columns:")
print(df.columns.tolist())

# Step 6: Save Feature Engineered Dataset
df.to_csv(
    "dataset/student_performance_featured.csv",
    index=False
)

print("\nFeature Engineering Completed Successfully!")
