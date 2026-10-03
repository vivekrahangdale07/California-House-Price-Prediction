from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# flask_app folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Main project folder
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Load model from parent folder
model = joblib.load(
    os.path.join(PROJECT_DIR, "model.pkl")
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "longitude": float(request.form["longitude"]),
        "latitude": float(request.form["latitude"]),
        "housing_median_age": float(request.form["housing_median_age"]),
        "total_rooms": float(request.form["total_rooms"]),
        "total_bedrooms": float(request.form["total_bedrooms"]),
        "population": float(request.form["population"]),
        "households": float(request.form["households"]),
        "median_income": float(request.form["median_income"]),
        "ocean_proximity": request.form["ocean_proximity"]
    }

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        prediction=f"${prediction:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)