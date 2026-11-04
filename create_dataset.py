
import pandas as pd
import numpy as np
import os

# Step 1: Generate sample data
np.random.seed(42)

n = 500

study_hours = np.random.randint(1, 10, n)
attendance = np.random.randint(50, 101, n)
previous_marks = np.random.randint(30, 96, n)
sleep_hours = np.random.randint(4, 10, n)
assignment_score = np.random.randint(30, 101, n)

# Step 2: Calculate final marks
final_marks = (
    study_hours * 4 +
    attendance * 0.15 +
    previous_marks * 0.25 +
    sleep_hours * 1.5 +
    assignment_score * 0.15 +
    np.random.normal(0, 5, n)
)

# Step 3: Keep marks between 0 and 100
final_marks = np.clip(final_marks, 0, 100)

# Step 4: Create DataFrame
df = pd.DataFrame({
    'Study_Hours': study_hours,
    'Attendance': attendance,
    'Previous_Marks': previous_marks,
    'Sleep_Hours': sleep_hours,
    'Assignment_Score': assignment_score,
    'Final_Marks': final_marks.round(2)
})

# Step 5: Create dataset folder
os.makedirs("dataset", exist_ok=True)

# Step 6: Save CSV file
df.to_csv("dataset/student_performance.csv", index=False)

# Step 7: Display output
print("Dataset Created Successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nCSV File Saved Successfully!")
