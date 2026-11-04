
import pandas as pd

# Step 1: Read Feature Engineered Dataset
df = pd.read_csv("dataset/student_performance_featured.csv")

print("Dataset Shape:")
print(df.shape)

# Step 2: Select Input Features
X = df[[
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Sleep_Hours",
    "Assignment_Score",
    "Study_Attendance_Index"
]]

# Step 3: Select Target Variable
y = df["Final_Marks"]

# Step 4: Display Input Features
print("\nInput Features (X):")
print(X.head())

# Step 5: Display Target Variable
print("\nTarget Variable (y):")
print(y.head())

# Step 6: Display Shapes
print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# Step 7: Display Selected Columns
print("\nSelected Features:")
print(X.columns.tolist())

print("\nFeature Selection Completed Successfully!")
