
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Step 5: Create Ridge Model
ridge = Ridge()

# Step 6: Define Parameters
parameters = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}

# Step 7: Apply GridSearchCV
grid_search = GridSearchCV(
    estimator=ridge,
    param_grid=parameters,
    cv=5,
    scoring="r2"
)

# Step 8: Train GridSearchCV
grid_search.fit(X_train, y_train)

# Step 9: Get Best Model
best_model = grid_search.best_estimator_

print("Best Parameters:")
print(grid_search.best_params_)

# Step 10: Prediction
y_pred = best_model.predict(X_test)

# Step 11: Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nTuned Model Evaluation:")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Step 12: Save Tuned Model
os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/student_model_tuned.joblib")

print("\nTuned Model Saved Successfully!")
