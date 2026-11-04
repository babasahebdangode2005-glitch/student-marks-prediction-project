
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Read Dataset
df = pd.read_csv("dataset/student_performance_featured.csv")

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

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 5: Create Linear Regression Model
model = LinearRegression()

# Step 6: Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# Step 7: Predict Test Data
y_pred = model.predict(X_test)

print("\nFirst 5 Predictions:")
print(y_pred[:5])

# Step 8: Save Model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/student_model.joblib")

print("\nModel Saved Successfully!")
