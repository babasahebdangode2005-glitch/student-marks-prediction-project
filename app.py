import pandas as pd
import joblib

from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
model = joblib.load("model/student_model_tuned.joblib")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:
            # Get input values
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])
            previous_marks = float(request.form["previous_marks"])
            sleep_hours = float(request.form["sleep_hours"])
            assignment_score = float(request.form["assignment_score"])

            # Input validation
            if not (0 <= study_hours <= 24):
                raise ValueError("Study Hours must be between 0 and 24.")

            if not (0 <= attendance <= 100):
                raise ValueError("Attendance must be between 0 and 100.")

            if not (0 <= previous_marks <= 100):
                raise ValueError("Previous Marks must be between 0 and 100.")

            if not (0 <= sleep_hours <= 24):
                raise ValueError("Sleep Hours must be between 0 and 24.")

            if not (0 <= assignment_score <= 100):
                raise ValueError("Assignment Score must be between 0 and 100.")

            # Feature Engineering
            study_attendance_index = study_hours * attendance / 100

            # Create input DataFrame
            input_data = pd.DataFrame([{
                "Study_Hours": study_hours,
                "Attendance": attendance,
                "Previous_Marks": previous_marks,
                "Sleep_Hours": sleep_hours,
                "Assignment_Score": assignment_score,
                "Study_Attendance_Index": study_attendance_index
            }])

            # Prediction
            result = model.predict(input_data)[0]

            prediction = round(float(result), 2)

            # Keep marks between 0 and 100
            prediction = max(0, min(100, prediction))

        except ValueError as e:
            error = str(e)

        except Exception as e:
            error = "Error: " + str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)