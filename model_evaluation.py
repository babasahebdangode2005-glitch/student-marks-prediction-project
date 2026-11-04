
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 1: Read Dataset
df = pd.read_csv("dataset/student_performance_featured.csv")

# Step 2: Select Features
X = df[[
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Sleep_Hours",
    "Assignment_Score",
    "Study_Attendance_Index"
]]

# Step 3: Select Target
y = df["Final_Marks"]

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Load Trained Model
model = joblib.load("model/student_model.joblib")

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Calculate Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

# Step 8: Display Results
print("Model Evaluation Results")

print("\nMean Absolute Error (MAE):", round(mae, 2))

print("Mean Squared Error (MSE):", round(mse, 2))

print("Root Mean Squared Error (RMSE):", round(rmse, 2))

print("R2 Score:", round(r2, 4))

print("\nModel Evaluation Completed Successfully!")
